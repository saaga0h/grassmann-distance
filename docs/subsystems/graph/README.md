# Graph Subsystem

<!-- @tier: 2 -->
<!-- @parent: ARCHITECTURE.md -->
<!-- @modules: docs/subsystems/graph/modules/ -->
<!-- @source: src/graph_types.jl, src/graph.jl, src/paths.jl, src/topology.jl, src/entity_distance.jl -->

## Overview

The graph subsystem converts a flat set of chunk embeddings into a navigable topology and provides path finding and structure analysis over that topology. It sits between the core geometry layer (tangent space estimation, Grassmann distance) and the FORGE contract layer (job dispatch, serialization).

The central artifact is `GrassmannGraph`: a fully precomputed structure containing entities, a complete `(n_entities, n_entities)` Grassmann distance matrix, and a k-NN adjacency list. Build is expensive; all queries against the built graph are cheap lookups with no matrix arithmetic.

For the mathematical rationale — why entity-to-entity distance is measured as a subspace angle, why path finding over this metric produces domain-coherent trajectories, and what gravitational basins reveal — see [CONCEPTS.md](../../../CONCEPTS.md).

---

## Key Files & Entry Points

| File | Role |
|---|---|
| `src/graph_types.jl` | All graph-domain types: `GraphConfig`, `Entity`, `GrassmannGraph`, `ConceptualPath`, `Community`, `Basin` |
| `src/entity_distance.jl` | `entity_distance()` — chunk-averaged Grassmann distance between two entities; `_representative_chunks()` for sampling |
| `src/graph.jl` | `build_graph()` — constructs the full `GrassmannGraph`; `_build_entities()` validates contiguous chunk ranges; `_build_adjacency()` builds the k-NN adjacency list |
| `src/paths.jl` | `find_greedy_path()`, `find_shortest_path()`, `reachable()`, and `_entity_idx()` |
| `src/topology.jl` | `bidirectional_edges()`, `communities()`, `basins()`, `bridges()`, `hub_centrality()`, `hub_concentration()`, `_gini_concentration()` |

Entry point for production: `_dispatch_query()` in `src/app.jl` routes `QuerySpec` structs to the appropriate function in `paths.jl` or `topology.jl`.

---

## Architecture

### Types

**`GraphConfig`** (`src/graph_types.jl:3`)

```julia
struct GraphConfig
    k_graph::Int    # neighbors per node in k-NN graph (default: 3)
    max_chunks::Int # max representative chunks per entity for distance averaging (default: 5)
end
```

Controls graph density and the chunk sampling budget for entity distance computation. `DEFAULT_GRAPH_CONFIG = GraphConfig()` uses both defaults.

**`Entity`** (`src/graph_types.jl:14`)

```julia
struct Entity
    id::String
    chunk_indices::UnitRange{Int}
end
```

A named group of chunks. `chunk_indices` is a contiguous range into the shared `embeddings` column space. Discontiguous chunks are a hard error — see Invariants.

**`GrassmannGraph`** (`src/graph_types.jl:21`)

```julia
struct GrassmannGraph
    entities::Vector{Entity}
    entity_index::Dict{String, Int}                  # id → position in entities
    embeddings::Matrix{Float64}                      # (ambient_dim, n_chunks)
    tangent_spaces::Vector{TangentSpace}             # one per chunk
    distance_matrix::Matrix{Float64}                 # (n_entities, n_entities)
    neighbors::Vector{Vector{Tuple{Int, Float64}}}   # k-NN adjacency list per entity
    grassmann_config::GrassmannConfig
    graph_config::GraphConfig
end
```

All fields are precomputed at build time. Queries read from `distance_matrix` and `neighbors`; they do not recompute geometry.

**`ConceptualPath`** (`src/graph_types.jl:34`)

```julia
struct ConceptualPath
    nodes::Vector{String}
    distances::Vector{Float64}  # per-hop; length == length(nodes) - 1
    total_distance::Float64
end
```

**`Community`** and **`Basin`** (`src/graph_types.jl:41`)

```julia
struct Community
    members::Vector{String}  # sorted
    root::String             # union-find root; identity is an implementation detail
end

struct Basin
    attractor::String
    members::Vector{String}  # sorted
end
```

### Construction Pipeline

```
caller supplies:
  embeddings       (ambient_dim, n_chunks)
  entity_ids       ordered unique names
  chunk_entity_map length-n_chunks, each chunk's entity ID

build_graph()
  │
  ├── _build_entities()
  │     Validates that each entity's chunks are contiguous.
  │     Produces Vector{Entity} with UnitRange{Int} per entity.
  │
  ├── estimate_tangent_spaces()    [src/tangent.jl]
  │     One TangentSpace per chunk column.
  │
  ├── entity distance loop  O(n_entities²)
  │     entity_distance(ts, e_i, e_j, config; max_chunks)
  │       └── _representative_chunks()  [evenly-spaced, deterministic]
  │           └── grassmann_distance()  [src/distance.jl]
  │     Fills symmetric distance_matrix; diagonal is 0.0.
  │
  └── _build_adjacency()
        For each entity, sort all other entities by distance,
        take the top k (clamped to n_entities - 1).
        Produces Vector{Vector{Tuple{Int, Float64}}}.
```

`build_graph_gpu()` in `src/gpu.jl` follows the same logical pipeline with GPU-accelerated kNN, neighborhood gather, and entity distance cross-products. SVD falls back to CPU in both paths (see ARCHITECTURE.md §5).

### How to Add a New Query Type

1. Add a new `elseif q.type == "your_type"` branch in `_dispatch_query()` in `src/app.jl`. The function takes a `GrassmannGraph` and `QuerySpec`; it must return a `QueryOutput`.
2. If the query produces a new output shape, define a new output struct in `src/job_types.jl` and add a field to `QueryOutput`. Register `StructTypes` annotations so JSON3 can serialize it.
3. Implement the query function in `src/paths.jl` (if path-shaped) or `src/topology.jl` (if structure-shaped), operating on `GrassmannGraph` fields.
4. The `QuerySpec` wire type already carries `from`, `to`, and `depth` — use these if they cover your parameters. Add new fields to `QuerySpec` in `src/job_types.jl` only if none of the existing fields apply.

---

## Data Flow

### Build job (input → `GrassmannGraph`)

```
FORGE MQTT payload (JSON)
  → parse_job()                    [src/serialization.jl]
  → _prepare_build_inputs()        flattens EntityInput[] into (embeddings, entity_ids, chunk_map)
  → _to_grassmann_config()         maps GraphConfigInput → GrassmannConfig + GraphConfig
  → build_graph() / build_graph_gpu()
  → GrassmannGraph
  → serialize_graph()              Julia Serialization + base64
  → BuildOutput.graph              opaque string returned to FORGE
```

### Query job (`GrassmannGraph` + `QuerySpec` → `QueryOutput`)

```
FORGE MQTT payload (JSON)
  → parse_job()
  → deserialize_graph(qp.graph)    base64 → Julia Serialization → GrassmannGraph
  → _dispatch_query(graph, q)
      q.type == "greedy_path"   → find_greedy_path()
      q.type == "shortest_path" → find_shortest_path()
      q.type == "reachable"     → reachable()
      q.type == "communities"   → communities()
      q.type == "basins"        → basins()
      q.type == "topology"      → communities() + basins() + bridges()
                                   + hub_centrality() + hub_concentration()
                                   + bidirectional_edges()
  → QueryOutput (path | reachable | topology; others are nothing)
  → serialize_result()
  → JobResult JSON → FORGE MQTT
```

---

## Path Algorithms

### `find_greedy_path` — open-ended (`src/paths.jl:9`)

Starts at `from_id`. At each step, selects the unvisited entity with the smallest `distance_matrix[current, j]` across all entities (not just k-NN neighbors). Stops when `depth` hops are taken or no unvisited entity exists. Never revisits a node. Returns a `ConceptualPath`.

`QuerySpec.depth` maps to the `depth` parameter (default: 4).

### `find_greedy_path` — targeted (`src/paths.jl:45`)

Starts at `from_id`, navigates toward `to_id`. At each step selects the unvisited entity `j` with the smallest `distance_matrix[j, to]` (closest to the target), not the closest to the current position. Hop distance recorded is `distance_matrix[current, j]`. Returns `nothing` if the target is not reached within `max_depth` hops.

`QuerySpec.depth` maps to `max_depth` (default: 10).

**Non-obvious:** the greedy targeted path steers by proximity to the target, not by proximity to the current node. The recorded hop cost is still the Grassmann distance of the hop actually taken, not the distance to the target.

### `find_shortest_path` (`src/paths.jl:89`)

Dijkstra's algorithm over the k-NN adjacency list (`graph.neighbors`). Does not consult the full distance matrix for edge weights — only edges present in the k-NN adjacency are traversable. Returns `nothing` if no path exists. The k-NN graph is directed (A's neighbor list may not include B even if B's includes A), so `find_shortest_path(graph, "A", "B")` and `find_shortest_path(graph, "B", "A")` may return paths of different lengths or one may return `nothing`.

Hop distances in the returned `ConceptualPath` are taken from `distance_matrix`, not from the stored edge weight in `neighbors`, so they reflect the true Grassmann distance for each hop.

### `reachable` (`src/paths.jl:146`)

BFS over k-NN adjacency. Tracks best cumulative distance to each entity across all BFS paths. Returns `Vector{Tuple{String, Int, Float64}}` — (entity_id, hop_count, cumulative_distance) — sorted by cumulative distance, excluding the source entity. `QuerySpec.depth` maps to `max_hops` (default: 3).

---

## Topology Functions

All topology functions operate on the precomputed `GrassmannGraph` fields. None trigger matrix computation.

### `bidirectional_edges` (`src/topology.jl:9`)

Returns pairs `(a, b)` where entity `a` appears in entity `b`'s k-NN neighbor list and vice versa. Each pair is stored once with `a < b` (lexicographic). Result is sorted.

### `communities` (`src/topology.jl:28`)

Connected components of the bidirectional-edge subgraph. Uses path-compressed union-find. Each entity appears in exactly one `Community`. Result is sorted by `length(members)` descending. `Community.root` is the union-find representative — it is one of the member IDs but carries no semantic meaning beyond that.

### `basins` (`src/topology.jl:67`)

For each entity, follows the greedy nearest-neighbor chain (over `distance_matrix`, not k-NN edges) until a previously-visited entity is encountered. That entity becomes the `attractor`. Every entity belongs to exactly one `Basin`; the attractor is always a member of its own basin. Result is sorted by `length(members)` descending.

**Non-obvious:** basins use the full distance matrix, not the k-NN adjacency. Two entities may land in the same basin even if they have no k-NN edge between them.

### `bridges` (`src/topology.jl:110`)

Entities whose k-NN neighbors span more than one community. For each such entity, returns `(entity_id, home_community_index, reached_community_indices)`. Community indices are 1-based positions in the result of `communities()`. Result is sorted by `length(reached_communities)` descending. An entity with all k-NN neighbors in its own community is not included.

### `hub_centrality` (`src/topology.jl:142`)

In-degree of each entity in the k-NN graph — how many other entities' k-NN lists include this entity. Returns `Vector{Tuple{String, Int}}` sorted by in-degree descending. The sum of all in-degrees equals the total number of directed edges (`n_entities * k_graph`).

### `hub_concentration` (`src/topology.jl:162`)

Gini coefficient of the in-degree distribution. Returns a `Float64` in `[0.0, 1.0]`. 0.0 means every entity has equal in-degree; 1.0 means a single entity receives all edges. Values above 0.5 indicate hub dominance. Uses `_gini_concentration()` which computes the standard Gini formula over sorted in-degrees.

---

## Interfaces & Contracts

### Inputs to `build_graph`

| Parameter | Type | Constraint |
|---|---|---|
| `embeddings` | `AbstractMatrix{<:Real}` (ambient_dim, n_chunks) | Columns must be pre-normalized if using geodesic distance |
| `entity_ids` | `AbstractVector{<:AbstractString}` | Ordered; each ID must appear in `chunk_entity_map` |
| `chunk_entity_map` | `AbstractVector{<:AbstractString}` | Length must equal `size(embeddings, 2)`; each entity's chunk indices must be contiguous |
| `grassmann_config` | `GrassmannConfig` | From `src/types.jl` |
| `graph_config` | `GraphConfig` | `k_graph` is clamped to `n_entities - 1` internally |

### FORGE wire format for queries

Queries arrive as `QuerySpec` in `src/job_types.jl`:

| Field | Required for |
|---|---|
| `type` | all queries |
| `from` | `greedy_path`, `shortest_path`, `reachable` |
| `to` | `shortest_path`; optional for `greedy_path` (open-ended if absent) |
| `depth` | `greedy_path` (hops, default 4), `reachable` (max_hops, default 3), `shortest_path` (max_depth when targeted, default 10) |

Valid `type` values: `"greedy_path"`, `"shortest_path"`, `"reachable"`, `"communities"`, `"basins"`, `"topology"`. Any other value causes `_dispatch_query` to throw `ArgumentError`.

### Output types (src/job_types.jl)

| Query type | Populated field in `QueryOutput` |
|---|---|
| `greedy_path`, `shortest_path` | `path::PathOutput` (or `nothing` if unreachable) |
| `reachable` | `reachable::Vector{ReachableEntry}` |
| `communities`, `basins`, `topology` | `topology::TopologyOutput` (other fields null for single-dimension queries) |

---

## Invariants

**Chunk contiguity.** Each entity's chunks must occupy a contiguous block of columns in the embedding matrix. `_build_entities()` checks this by finding all column indices where `chunk_entity_map[i] == eid` and asserting they equal `first:last`. Violation throws `ArgumentError`. This is the most common source of build failures when chunk ordering is wrong upstream.

**Symmetric distance matrix.** `distance_matrix[i, j] == distance_matrix[j, i]` and `distance_matrix[i, i] == 0.0` for all `i`. `build_graph` writes both symmetric entries in the same iteration step. If you observe asymmetry, the cause is a floating-point discrepancy in the input embeddings or a bug in `grassmann_distance()`, not a graph construction issue.

**k clamped to n_entities - 1.** `build_graph` computes `k = min(graph_config.k_graph, n_entities - 1)` before calling `_build_adjacency`. When entity count is small (e.g. 2 entities with `k_graph=3`), each entity has exactly 1 neighbor. This is not an error but means paths may have limited reach.

**Deterministic chunk sampling.** `_representative_chunks` uses `round.(Int, range(1, n; length=max_chunks))`. Given the same entity and the same `max_chunks`, it always selects the same chunk indices. Entity distance is reproducible across calls.

**One entity per basin, one entity per community.** Both `basins()` and `communities()` partition the full entity set. Every entity appears exactly once in each result.

---

## Diagnosing Graph Problems

| Symptom | Check | Fix |
|---|---|---|
| `ArgumentError: entity 'X' chunks are not contiguous` | Caller's `chunk_entity_map` has entity X's chunks interleaved with another entity's chunks | Sort the embedding matrix so each entity's chunks are a contiguous block before calling `build_graph` |
| `DimensionMismatch: chunk_entity_map length ... ≠ embedding columns` | `length(chunk_entity_map) != size(embeddings, 2)` | Ensure the map covers every column in the embedding matrix |
| `ArgumentError: unknown entity: 'X'` | A path or topology query used an entity ID not present in `graph.entity_index` | Verify the entity ID against `[e.id for e in graph.entities]`; IDs are case-sensitive |
| Path returns `nothing` when a connection is expected | Entities may be in disconnected k-NN components; `find_shortest_path` only traverses k-NN edges | Increase `k_graph` and rebuild, or use `find_greedy_path` (which uses the full distance matrix) |
| All paths converge to the same entity within 2-3 hops | Hub concentration is high; one entity has dominant in-degree | Check `hub_concentration(graph)` and `hub_centrality(graph)`; consider increasing k or inspecting whether one entity's embeddings have anomalously low Grassmann distance to all others |
| `distance_matrix` appears asymmetric | Floating-point discrepancy; should not occur under normal operation | Check whether `grassmann_distance` with swapped arguments returns a different value; verify `estimate_tangent_spaces` produces the same basis regardless of ordering |
| `k_graph` neighbors fewer than configured | Entity count is small; k is clamped to `n_entities - 1` | Expected behavior; not a bug |
| `topology` query returns all entities in one community | All entities are mutually connected via bidirectional edges; `k_graph` may be too high | Reduce `k_graph` and rebuild to obtain finer community structure |

---

## Failure Behavior

| Situation | Behavior |
|---|---|
| Unknown entity ID passed to `find_greedy_path`, `find_shortest_path`, or `reachable` | `ArgumentError("unknown entity: 'X'")` thrown via `_entity_idx()` |
| Target unreachable within `max_depth` in targeted `find_greedy_path` | Returns `nothing` |
| No k-NN path exists in `find_shortest_path` | Returns `nothing` |
| `reachable` finds no entities within `max_hops` | Returns empty `Vector{Tuple{String, Int, Float64}}` |
| `bridges` finds no entities spanning multiple communities | Returns empty `Vector{Tuple{String, Int, Vector{Int}}}` |
| Unknown `q.type` in `_dispatch_query` | `ArgumentError("unknown query type: 'X'")` thrown; caught by `_process_query` and returned as a failed `JobResult` |
| Build with 0 entities or mismatched dimensions | `DimensionMismatch` or `ArgumentError` from `_build_entities` |

---

## Dependencies

**Within this module:**

- `src/types.jl` — `GrassmannConfig`, `TangentSpace`
- `src/tangent.jl` — `estimate_tangent_spaces()`
- `src/distance.jl` — `grassmann_distance()`

**Callers:**

- `src/app.jl` — `build_graph()`, `_dispatch_query()` calling all path and topology functions
- `src/gpu.jl` — `build_graph_gpu()` shares `_build_entities()` and `_build_adjacency()` from `src/graph.jl`
- `src/serialization.jl` — `serialize_graph()`, `deserialize_graph()` operating on `GrassmannGraph`

---

## Module Index

<!-- TODO: populate when docs/subsystems/graph/modules/ files are written -->

---

## Related Documents

- [CONCEPTS.md](../../../CONCEPTS.md) — conceptual explanation of Grassmann distance, tangent spaces, graph topology, paths, communities, and basins
- [ARCHITECTURE.md](../../../ARCHITECTURE.md) — system-level type inventory, GPU acceleration details, and the full build/query data flow sequence diagrams
- [docs/subsystems/geometry/README.md](../geometry/README.md) — tangent space estimation and Grassmann distance computation
- [docs/subsystems/forge/README.md](../forge/README.md) — FORGE contract types, `_dispatch_query` routing, and serialization
- [docs/messaging.md](../../messaging.md) — MQTT topic structure and `QuerySpec` wire format

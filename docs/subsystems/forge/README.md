# FORGE Job Integration

<!-- @tier: 2 -->
<!-- @parent: ARCHITECTURE.md -->
<!-- @source: src/job_types.jl, src/serialization.jl, src/app.jl, src/config.jl -->

## Overview

This subsystem is the boundary between the FORGE orchestrator and the GrassmannDistance computation core. It owns the JSON contract, type definitions, serialization, and top-level job routing. All worker entry points flow through this layer; nothing else in the codebase knows about MQTT payloads or FORGE conventions.

The subsystem covers three concerns:

- **Job types** (`src/job_types.jl`): Julia structs that map directly to the JSON wire format. Entity IDs flow through these types untouched from client request to response.
- **Serialization** (`src/serialization.jl`): JSON parsing and emission via JSON3/StructTypes, plus graph binary encoding for the opaque blob passed between build and query jobs.
- **Job processing** (`src/app.jl`): the single entry point `process_job`, which routes to mode handlers and guarantees that no exception escapes to the caller.

Worker configuration (env vars, MQTT credentials) is read in `src/config.jl` and is documented in `docs/development.md`. The MQTT transport layer is a separate subsystem (`docs/subsystems/mqtt/`).

## Key Files & Entry Points

| File | Role |
|---|---|
| `src/job_types.jl` | All input and output structs. No logic — pure type declarations. |
| `src/serialization.jl` | `parse_job`, `serialize_result`, `serialize_graph`, `deserialize_graph`, `_to_grassmann_config`, `_prepare_build_inputs`. |
| `src/app.jl` | `process_job` (public entry point), `_process_build`, `_process_query`, `_dispatch_query`. |
| `src/config.jl` | `load_config`, `WorkerConfig`. Environment variable names and defaults. |
| `test/test_serialization.jl` | Round-trip tests for graph binary encoding and JSON parse/emit. |
| `test/test_app.jl` | End-to-end `process_job` tests for all modes and error cases. |

## Architecture

### Job lifecycle

```
MQTT payload (Vector{UInt8})
        │
        ▼
  process_job(payload, worker_id, log_fn)  ← src/app.jl
        │
        ├─ parse_job(payload)              ← src/serialization.jl
        │     └─ JSON3.read(..., JobParams)
        │
        ├─ mode == "build"
        │     └─ _process_build(...)
        │           ├─ _prepare_build_inputs(entities)   ← assemble embedding matrix
        │           ├─ _to_grassmann_config(config)      ← convert string distance to Symbol
        │           ├─ build_graph(...) / build_graph_gpu(...)
        │           └─ serialize_graph(graph)            ← base64-encoded Julia Serialization blob
        │
        ├─ mode == "query"
        │     └─ _process_query(...)
        │           ├─ deserialize_graph(qp.graph)
        │           └─ _dispatch_query(graph, q)         ← routes on q.type string
        │
        └─ JobResult (always returned, never throws)
              │
              ▼
        serialize_result(result)           ← JSON3.write → Vector{UInt8}
```

### Mode routing

`process_job` dispatches on `params.mode`. Valid values are `"build"` and `"query"`. Any other value returns `JobResult` with `success=false` and `error="Unknown mode: ..."`. The outer `try/catch` around the dispatch block catches all exceptions from the computation layer and converts them to error results.

`_dispatch_query` dispatches on `q.type`. Valid values are `"greedy_path"`, `"shortest_path"`, `"reachable"`, `"communities"`, `"basins"`, and `"topology"`. Unknown types throw `ArgumentError("unknown query type: ...")`, which is caught by the outer handler in `process_job` and returned as `success=false`.

### How to add a new job mode

1. Define parameter and output structs in `src/job_types.jl` following the existing pattern (`struct FooParams`, `struct FooOutput`).
2. Add `StructTypes.StructType(::Type{FooParams}) = StructTypes.Struct()` and the output type mapping in `src/serialization.jl`. Add `omitempties` entries for any `Union{..., Nothing}` fields.
3. Add a field `foo::Union{FooParams, Nothing}` to `JobParams` in `src/job_types.jl` and add `omitempties` for it.
4. Add a `Union{FooOutput, Nothing}` field to `JobResult.result` if the result shape differs from `BuildOutput`/`QueryOutput`, or extend `QueryOutput` if the new mode is a query variant.
5. Add an `elseif params.mode == "foo"` branch in `process_job` in `src/app.jl` calling a new `_process_foo(...)` function.
6. Add a new query type string to `_dispatch_query` if the mode is a query variant rather than a top-level mode.

### How to add a new query type

1. Add a branch in `_dispatch_query` in `src/app.jl` matching on `q.type`. Return a `QueryOutput` with the appropriate field populated and the others set to `nothing`.
2. If the new type requires new output structs, add them to `src/job_types.jl`, register them in `src/serialization.jl`, and add the field to `QueryOutput`.
3. Update `QuerySpec` documentation if the new type uses `from`, `to`, or `depth` differently from existing types.

## Data Flow

### Build mode

```
Vector{EntityInput}  →  _prepare_build_inputs
    │                        │
    │                        ├─ entity_ids::Vector{String}
    │                        ├─ chunk_entity_map::Vector{String}   (chunk-to-entity association)
    │                        └─ embeddings::Matrix{Float64}        (dim × n_chunks)
    │
GraphConfigInput     →  _to_grassmann_config
                             ├─ GrassmannConfig(k, p, distance_symbol)
                             └─ GraphConfig(k_graph, max_chunks)
                                          │
                                          ▼
                              build_graph(embeddings, entity_ids, chunk_map, gc, graph_config)
                                          │
                                          ▼
                              GrassmannGraph  →  serialize_graph  →  base64 String
                                                                           │
                                                                      BuildOutput(graph, entities, chunks)
```

### Query mode

```
QueryParams.graph (base64 String)  →  deserialize_graph  →  GrassmannGraph
QueryParams.query (QuerySpec)      →  _dispatch_query
                                            │
                                            ├─ "greedy_path"   → find_greedy_path(...)  → PathOutput
                                            ├─ "shortest_path" → find_shortest_path(...) → PathOutput
                                            ├─ "reachable"     → reachable(...)          → []ReachableEntry
                                            ├─ "communities"   → communities(...)        → TopologyOutput (partial)
                                            ├─ "basins"        → basins(...)             → TopologyOutput (partial)
                                            └─ "topology"      → communities + basins + bridges + hub_centrality + hub_concentration + bidirectional_edges → TopologyOutput (full)
                                                                           │
                                                                      QueryOutput
```

## Interfaces & Contracts

### Type mappings (StructTypes)

All types use `StructTypes.Struct()`, which requires all JSON fields to match struct field names exactly. There is no lenient parsing — missing or extra fields cause a JSON3 parse error.

Optional fields use `Union{..., Nothing}` and are registered with `StructTypes.omitempties`:

| Type | Fields omitted when `nothing` |
|---|---|
| `JobParams` | `build`, `query` |
| `QuerySpec` | `from`, `to`, `depth` |
| `QueryOutput` | `path`, `reachable`, `topology` |
| `JobResult` | `error`, `result` |

When a field is omitted from an incoming JSON object, JSON3 sets it to `nothing`. When emitting, `omitempties` causes `nothing` fields to be absent from the JSON rather than written as `null`.

### Graph binary encoding

`serialize_graph` uses Julia's built-in `Serialization` module. The encoded blob is opaque to non-Julia consumers — it cannot be inspected or deserialized outside of Julia. FORGE and clients pass it through as an opaque string.

The blob is not stored server-side. The client receives it in the build response and must supply it verbatim in every subsequent query. There is no graph identity or lookup mechanism.

`deserialize_graph` type-asserts the result as `GrassmannGraph`. If the blob is corrupted or was produced by an incompatible Julia version, `deserialize_graph` will throw, which is caught by `process_job` and returned as a computation failure.

### `_to_grassmann_config` conversion

The `distance` field in `GraphConfigInput` is a JSON string. `_to_grassmann_config` maps `"chordal"` to `:chordal` and any other value to `:geodesic`. There is no validation error for unrecognized strings — they silently fall back to geodesic.

### `_prepare_build_inputs` flattening

Each `EntityInput` carries multiple embedding vectors (chunks). `_prepare_build_inputs` flattens them into a single `Matrix{Float64}` of shape `(dim, n_chunks)` and a parallel `chunk_entity_map::Vector{String}` that records which entity each column belongs to. All embedding vectors must have the same dimensionality; no runtime check enforces this — mismatched dimensions will cause an error downstream during graph construction.

### `log_fn` interface

`process_job` takes a `log_fn` argument with signature `(level::String, message::String) -> nothing`. The MQTT transport passes a closure that publishes to `compute/jobs/{job_id}/logs`. Tests use either `noop_log` (discards all messages) or `collect_log(logs)` (appends to a vector). `log_fn` is called at these points:

- `"info"`: job received (mode)
- `"info"`: building graph (entity and chunk counts)
- `"info"`: graph built (encoded byte count)
- `"info"`: query type, `from`, `to`
- `"error"`: parse failure or computation exception

## Error Handling

`process_job` never throws. It catches all exceptions at two levels:

1. **Parse failure** (before `params.job_id` is available): returns `JobResult("unknown", false, "Parse error: ...", nothing, worker_id, timestamp)`. The `job_id` field is `"unknown"` because parsing failed before it could be read.
2. **Computation failure** (after successful parse): returns `JobResult(params.job_id, false, "Computation failed: ...", nothing, worker_id, timestamp)`.

Mode validation and query type validation are handled inline before the computation try/catch: unknown modes return immediately with an error result, and unknown query types throw `ArgumentError` from `_dispatch_query`, which is caught by the outer handler.

### Diagnosing job failures

| Symptom | Check | Fix |
|---|---|---|
| `result.success == false`, `error` starts with `"Parse error:"` | JSON structure does not match `JobParams` field names or types. Use `JSON3.read(payload, JobParams)` interactively to see the parse exception. | Verify field names, types, and that `Union{..., Nothing}` fields are absent (not `null`) when not applicable. |
| `result.success == false`, `error` starts with `"Computation failed:"`, message mentions deserialization | The graph blob is corrupted, truncated, or was produced by a different Julia version. | Re-run the build job and pass the new blob to subsequent queries. Never modify the blob in transit. |
| `result.success == false`, `error` is `"build mode requires build params"` | `build` field is missing from the `JobParams` JSON for a build-mode job. | Ensure `"build": {...}` is present when `"mode": "build"`. |
| `result.success == false`, `error` starts with `"unknown query type:"` | `query.type` value is not one of the six valid strings. | Check `_dispatch_query` in `src/app.jl` for the current list of valid types. |
| `log_fn` receives `"error"` level messages | Same failure path — the message content mirrors `JobResult.error`. | Correlate log timestamps with job result to identify which job failed. |
| No log messages produced | `log_fn` is a no-op or topic publish is failing | In tests, use `collect_log(logs)` to capture messages. In production, check the MQTT transport layer (`docs/subsystems/mqtt/`). |

## Dependencies

- `JSON3` / `StructTypes` — JSON parsing and emission
- `Serialization` — graph binary encoding (Julia stdlib)
- `Base64` — base64 encoding/decoding of the graph blob (Julia stdlib)
- `Dates` — ISO 8601 UTC timestamp generation for `JobResult.timestamp`
- Graph computation core (`src/graph.jl`, `src/paths.jl`, `src/topology.jl`) — called by `_process_build` and `_dispatch_query`
- GPU backend (`src/gpu.jl`) — called by `_process_build` when `backend isa AMDGPU.ROCBackend`

## Related Documents

- `docs/messaging.md` — full MQTT protocol, JSON examples for all request/response shapes, graph blob sizing, topic map
- `docs/development.md` — environment variables, `WorkerConfig`, container build
- `docs/subsystems/mqtt/` — MQTT transport layer; owns `log_fn` construction and the subscribe/publish loop
- `docs/subsystems/graph/` — graph construction and query algorithms called by `_process_build` and `_dispatch_query`
- `docs/subsystems/gpu/` — GPU backend dispatch
- `src/job_types.jl` — authoritative type definitions
- `src/serialization.jl` — authoritative serialization functions
- `src/app.jl` — authoritative job processing and mode routing
- `test/test_app.jl` — all error cases exercised with assertions

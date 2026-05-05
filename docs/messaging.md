# Messaging Protocol

<!-- @tier: 1 -->
<!-- @source: src/mqtt.jl, src/job_types.jl, src/serialization.jl, src/app.jl, src/config.jl -->
<!-- @see-also: CONCEPTS.md -->

## Overview

The GrassmannDistance worker communicates exclusively via MQTT. It does not expose an HTTP API or CLI entry point for production use. All production interaction flows through FORGE, which acts as an orchestrator and message router between clients and workers.

The end-to-end flow:

1. Client publishes a request envelope to `compute/request/{client_id}/{correlation_id}`
2. FORGE reads the `operation` field, generates a `job_id`, and writes the job params as a retained message to `compute/jobs/{job_id}/params`
3. FORGE dispatches a one-shot worker (Nomad job: `grassmann-distance`)
4. The worker connects to the broker, reads the retained params, processes the job, and publishes its result to `compute/jobs/{job_id}/result`
5. FORGE reads the result and routes it back to `compute/response/{client_id}/{correlation_id}`

Clients only see the `compute/request` and `compute/response` topics. The `compute/jobs/*` topics are internal to FORGE and the worker.

## Topic Map

| Topic | Direction | QoS | Retained | Publisher | Subscriber |
|---|---|---|---|---|---|
| `compute/request/{client_id}/{correlation_id}` | client → FORGE | 1 | no | client | FORGE |
| `compute/response/{client_id}/{correlation_id}` | FORGE → client | 1 | no | FORGE | client |
| `compute/jobs/{job_id}/params` | FORGE → worker | 1 | **yes** | FORGE | worker |
| `compute/jobs/{job_id}/result` | worker → FORGE | 1 | no | worker | FORGE |
| `compute/jobs/{job_id}/status` | worker → FORGE | 0 | no | worker | FORGE |
| `compute/jobs/{job_id}/logs` | worker → FORGE | 0 | no | worker | FORGE |

`{client_id}` is chosen by the client (e.g. `grassmann-test-4712`). `{correlation_id}` is chosen by the client per request (e.g. `build-001`). `{job_id}` is assigned by FORGE and passed to the worker via the `JOB_ID` environment variable.

The worker subscribes only to `compute/jobs/{job_id}/params`. It does not subscribe to client topics or any wildcard.

## Request Envelope

Clients publish a JSON object to `compute/request/{client_id}/{correlation_id}`. The envelope wraps a mode-specific payload:

```json
{
  "operation": "grassmann-distance",
  "payload": {
    "mode": "build",
    ...
  }
}
```

`operation` is required and must be `"grassmann-distance"`. FORGE uses this field to route the request to the correct worker type. The `payload` object is forwarded to the worker, augmented with `job_id` by FORGE, and published as `JobParams` to the params topic.

**Important:** clients must subscribe to the response topic _before_ publishing the request. FORGE may route the response before the client has time to subscribe after publishing.

## Worker Params Message (`JobParams`)

FORGE publishes the following structure to `compute/jobs/{job_id}/params` as a retained message. This is what the worker reads directly — it is the `JobParams` type parsed by `parse_job()` in `src/serialization.jl`.

```json
{
  "job_id": "forge-assigned-job-id",
  "mode": "build",
  "build": { ... }
}
```

Or for query mode:

```json
{
  "job_id": "forge-assigned-job-id",
  "mode": "query",
  "query": { ... }
}
```

`mode` must be either `"build"` or `"query"`. The field not matching the mode (`build` or `query`) is omitted. Both are declared `Union{..., Nothing}` and use `StructTypes.omitempties`.

## Build Job

A build job constructs a `GrassmannGraph` from a set of entities and their embeddings.

**Client request envelope:**

```json
{
  "operation": "grassmann-distance",
  "payload": {
    "mode": "build",
    "build": {
      "entities": [
        {
          "id": "doc1",
          "embeddings": [
            [1.0, 2.0, 3.0, 0.4],
            [0.9, 2.1, 3.1, 0.3]
          ]
        },
        {
          "id": "doc2",
          "embeddings": [
            [7.0, 8.0, 9.0, 0.1]
          ]
        }
      ],
      "config": {
        "k": 20,
        "p": 2,
        "k_graph": 3,
        "max_chunks": 5,
        "distance": "geodesic"
      }
    }
  }
}
```

### `EntityInput`

| Field | Type | Description |
|---|---|---|
| `id` | string | Entity identifier. Flows through untouched — client names in, client names out. |
| `embeddings` | array of float arrays | One or more embedding vectors for this entity. All vectors must have the same dimensionality. |

### `GraphConfigInput`

| Field | Type | Default | Description |
|---|---|---|---|
| `k` | int | 20 | Neighborhood size for local PCA (tangent space estimation). |
| `p` | int | 2 | Subspace dimension (number of principal components per tangent space). |
| `k_graph` | int | 3 | Number of nearest neighbors per entity in the output graph. |
| `max_chunks` | int | 5 | Maximum representative chunks sampled per entity pair when computing entity distances. |
| `distance` | string | `"geodesic"` | Distance metric. `"geodesic"` (arc length on the Grassmannian) or `"chordal"` (straight-line through ambient space). |

The defaults shown above come from `DEFAULT_CONFIG_INPUT` in `src/job_types.jl`. They are not applied automatically — clients must always supply a complete `config` object.

## Query Job

A query job runs a traversal or topology analysis over a previously built graph. The graph is passed back in full as an opaque blob (see [Graph Blob](#graph-blob)).

All query jobs share the same outer structure:

```json
{
  "operation": "grassmann-distance",
  "payload": {
    "mode": "query",
    "query": {
      "graph": "<base64-encoded-graph-blob>",
      "query": {
        "type": "...",
        ...
      }
    }
  }
}
```

### `greedy_path`

At each hop, move to the nearest unvisited neighbor. Returns a path starting from `from`. If `to` is provided, the path terminates when it reaches `to` or exhausts `depth` hops.

```json
{
  "type": "greedy_path",
  "from": "alpha",
  "depth": 3
}
```

With a target:

```json
{
  "type": "greedy_path",
  "from": "alpha",
  "to": "delta",
  "depth": 4
}
```

`depth` defaults to `4` if omitted (applied in `_dispatch_query` in `src/app.jl`).

### `shortest_path`

Dijkstra over the precomputed distance matrix. Both `from` and `to` are required.

```json
{
  "type": "shortest_path",
  "from": "alpha",
  "to": "delta"
}
```

### `reachable`

BFS over the k-NN adjacency from `from`. Returns all entities reachable within `depth` hops. The source entity is excluded from results. `depth` maps to `max_hops` and defaults to `3` if omitted.

```json
{
  "type": "reachable",
  "from": "alpha",
  "depth": 2
}
```

### `topology`

Full structural analysis. No additional fields required. Computes communities, basins, bridges, hub centrality, and bidirectional edges in a single pass.

```json
{
  "type": "topology"
}
```

### `communities`

Connected components of the bidirectional-edge subgraph. Returns only the `communities` array of the `TopologyOutput`. Other `TopologyOutput` fields (`basins`, `bridges`, `hub_centrality`, `hub_concentration`, `bidirectional_edges`) are present but empty.

```json
{
  "type": "communities"
}
```

### `basins`

Gravitational basins via greedy attractor chains. Returns only the `basins` array. Other `TopologyOutput` fields are present but empty.

```json
{
  "type": "basins"
}
```

### `QuerySpec` field summary

| Field | Type | Required by | Description |
|---|---|---|---|
| `type` | string | all | One of `greedy_path`, `shortest_path`, `reachable`, `communities`, `basins`, `topology`. |
| `from` | string | `greedy_path`, `shortest_path`, `reachable` | Source entity ID. |
| `to` | string | `shortest_path` (required), `greedy_path` (optional) | Target entity ID. |
| `depth` | int | `greedy_path`, `reachable` (optional) | Max hops. Defaults: `greedy_path` → 4, `reachable` → 3. |

`from`, `to`, and `depth` are `Union{..., Nothing}` and omitted when not set (`StructTypes.omitempties`).

## Response Format

FORGE routes the worker's `JobResult` back to the client. The client receives a JSON object with the following shape.

### Success — build

```json
{
  "job_id": "forge-assigned-job-id",
  "success": true,
  "result": {
    "graph": "<base64-encoded-graph-blob>",
    "entities": 4,
    "chunks": 16
  },
  "worker_id": "grassmann-forge-assigned-job-id",
  "timestamp": "2026-05-05T12:00:00.000Z"
}
```

`result.graph` is the serialized `GrassmannGraph` (see [Graph Blob](#graph-blob)). `entities` is the count of distinct entity IDs. `chunks` is the total number of embedding vectors across all entities.

### Success — path query (`greedy_path` or `shortest_path`)

```json
{
  "job_id": "forge-assigned-job-id",
  "success": true,
  "result": {
    "path": {
      "nodes": ["alpha", "gamma", "delta"],
      "distances": [0.312, 0.489],
      "total_distance": 0.801
    }
  },
  "worker_id": "grassmann-forge-assigned-job-id",
  "timestamp": "2026-05-05T12:00:00.000Z"
}
```

`nodes` has length N. `distances` has length N-1 — the distance from `nodes[i]` to `nodes[i+1]`. If no path exists (disconnected graph), `result.path` is absent from the response (omitted by `StructTypes.omitempties`).

### Success — reachable query

```json
{
  "job_id": "forge-assigned-job-id",
  "success": true,
  "result": {
    "reachable": [
      {"id": "beta",  "hops": 1, "distance": 0.312},
      {"id": "gamma", "hops": 1, "distance": 0.489},
      {"id": "delta", "hops": 2, "distance": 0.801}
    ]
  },
  "worker_id": "grassmann-forge-assigned-job-id",
  "timestamp": "2026-05-05T12:00:00.000Z"
}
```

The source entity is not included in the `reachable` array. `distance` is cumulative Grassmann distance from source along the BFS path.

### Success — topology query

```json
{
  "job_id": "forge-assigned-job-id",
  "success": true,
  "result": {
    "topology": {
      "communities": [
        {"members": ["alpha", "beta"], "root": "alpha"},
        {"members": ["gamma", "delta"], "root": "gamma"}
      ],
      "basins": [
        {"attractor": "alpha", "members": ["alpha", "beta"]},
        {"attractor": "gamma", "members": ["gamma", "delta"]}
      ],
      "bridges": [
        {"id": "beta", "home_community": 0, "reached_communities": [1]}
      ],
      "hub_centrality": [
        {"id": "alpha", "in_degree": 3},
        {"id": "gamma", "in_degree": 2}
      ],
      "hub_concentration": 0.256,
      "bidirectional_edges": [
        ["alpha", "beta"],
        ["gamma", "delta"]
      ]
    }
  },
  "worker_id": "grassmann-forge-assigned-job-id",
  "timestamp": "2026-05-05T12:00:00.000Z"
}
```

`hub_concentration` is a Gini coefficient over basin sizes (0.0 = uniform distribution, 1.0 = single dominant hub). `bidirectional_edges` contains pairs `[A, B]` where both A's and B's k-NN neighbor lists include the other.

For `communities` and `basins` query types, the `topology` object is returned with only the relevant array populated; all other arrays are empty and `hub_concentration` is `0.0`.

### Error

```json
{
  "job_id": "forge-assigned-job-id",
  "success": false,
  "error": "unknown query type: nonsense",
  "worker_id": "grassmann-forge-assigned-job-id",
  "timestamp": "2026-05-05T12:00:00.000Z"
}
```

On error, `result` is absent. `error` contains the exception message. Common error cases:

- JSON parse failure: `"Parse error: ..."` — `job_id` will be `"unknown"` because parsing failed before the job ID could be read.
- Unknown mode: `"Unknown mode: explode"`
- Missing params for mode: `"build mode requires build params"` / `"query mode requires query params"`
- Unknown query type: `"unknown query type: nonsense"` — raised as `ArgumentError` in `_dispatch_query`
- Computation failure: `"Computation failed: ..."` — any exception from the graph algorithms

### `JobResult` field summary

| Field | Type | Omitted when |
|---|---|---|
| `job_id` | string | never (`"unknown"` on parse failure) |
| `success` | bool | never |
| `error` | string | success is true |
| `result` | `BuildOutput` or `QueryOutput` | success is false |
| `worker_id` | string | never |
| `timestamp` | string | never — ISO 8601 UTC, e.g. `"2026-05-05T12:00:00.000Z"` |

`error` and `result` use `StructTypes.omitempties` and are absent (not `null`) in the JSON when not applicable.

## Status and Log Messages

The worker publishes lightweight status and log messages during processing. These are consumed by FORGE and not forwarded to clients.

### Status message (`compute/jobs/{job_id}/status`, QoS 0)

Published twice per job: once at `"processing"` (before computation begins) and once at `"completed"` or `"error"` (after the result is published).

```json
{
  "job_id": "forge-assigned-job-id",
  "worker_id": "grassmann-forge-assigned-job-id",
  "status": "processing",
  "timestamp": "2026-05-05T12:00:00.000Z"
}
```

`status` is one of `"processing"`, `"completed"`, `"error"`.

### Log message (`compute/jobs/{job_id}/logs`, QoS 0)

Published throughout processing. The worker calls `log_fn` from `src/app.jl` at key points: job received, building graph (with entity/chunk counts), graph built (with encoded byte count), and any errors.

```json
{
  "job_id": "forge-assigned-job-id",
  "level": "info",
  "message": "Building graph: 4 entities, 16 chunks",
  "timestamp": "2026-05-05T12:00:00.000Z"
}
```

`level` is `"info"` or `"error"`.

## Graph Blob

`BuildOutput.graph` and `QueryParams.graph` carry the serialized `GrassmannGraph` as a base64-encoded string. The encoding is produced by `serialize_graph()` and consumed by `deserialize_graph()` in `src/serialization.jl`:

```julia
# Write
io = IOBuffer()
Serialization.serialize(io, graph)
encoded = Base64.base64encode(take!(io))

# Read
bytes = Base64.base64decode(encoded)
io = IOBuffer(bytes)
graph = Serialization.deserialize(io)::GrassmannGraph
```

The blob uses Julia's built-in `Serialization` module. It is **opaque to non-Julia consumers** — it cannot be inspected or deserialized outside of Julia. Clients and FORGE pass it through untouched.

The blob is returned in the build result and must be passed back verbatim in every subsequent query against that graph. There is no server-side graph storage — the client is responsible for retaining the blob between build and query calls.

Blob size scales with corpus size. For 4 entities with 4 chunks each at 64 dimensions, the blob is on the order of a few kilobytes. For production corpora (hundreds of entities, 4096D embeddings), expect tens to hundreds of kilobytes.

## Worker Lifecycle and Timing

The worker is one-shot: it connects, processes exactly one job, and exits. This is enforced by the `subscribe_and_process!` function in `src/mqtt.jl`, which blocks until one message arrives on the params topic, processes it, publishes the result and final status, then disconnects.

FORGE publishes params as a **retained** message before dispatching the worker. This means the worker receives the params immediately on subscription regardless of startup latency. The retained message is the synchronization mechanism — no polling, no ready signal.

The sequence with timing:

```
FORGE                              Worker
  │                                  │
  ├─ publish params (retained) ─────►│ (params sit on broker)
  ├─ dispatch Nomad job ────────────►│
  │                                  ├─ connect to broker
  │                                  ├─ subscribe params topic
  │                                  ├─ receive retained params immediately
  │                                  ├─ publish status: "processing"
  │                                  ├─ compute...
  │                                  ├─ publish result (QoS 1)
  │◄─ receive result ────────────────┤
  │                                  ├─ publish status: "completed"
  │◄─ receive status ────────────────┤
  │                                  └─ disconnect + exit
  │
  ├─ route result to client response topic
```

The worker uses `MQTT_BROKER`, `JOB_ID`, `WORKER_ID`, `MQTT_USER`, and `MQTT_PASSWORD` environment variables (see `src/config.jl`). `JOB_ID` is required; the worker exits with an error if it is absent. `WORKER_ID` defaults to `"grassmann-{job_id}"` if not set.

## Related Documents

- `CONCEPTS.md` — mathematical background: Grassmann manifold, tangent spaces, entity distance, graph topology
- `src/job_types.jl` — all input and output structs
- `src/serialization.jl` — JSON parsing, graph binary serialization
- `src/mqtt.jl` — topic construction, worker subscribe/publish loop, status and log publishing
- `src/app.jl` — mode routing, query dispatch, error handling
- `src/config.jl` — environment variable names and defaults
- `test/test_serialization.jl` — JSON examples for build and query modes
- `test/test_app.jl` — full job processing examples including error cases
- `test_integration.jl` — client-side MQTT usage with FORGE

# Grassmann Distance — Conceptual Path Finding in Embedding Space

Replaces cosine similarity with Grassmann distance over local tangent spaces for high-dimensional embeddings. Builds navigable graphs where paths follow domain-coherent conceptual lanes instead of converging to vocabulary hubs. GPU-accelerated FORGE compute worker via Singularity on Nomad.

**Tech stack**: Julia 1.10+, KernelAbstractions.jl, AMDGPU.jl, Mosquitto.jl (MQTT), JSON3.jl, Singularity, Nomad (batch parameterized job)

---

## Prerequisites

- Julia 1.10 or later
- `libmosquitto-dev` (for the Mosquitto.jl dependency)
- For GPU: ROCm with a functional AMD GPU (`AMDGPU.functional()` must return `true`)
- For container builds: `singularity` with `--fakeroot` capability on the build host
- For deployment: `nomad` CLI and `NOMAD_ADDR` set to the cluster address

---

## Setup (local dev)

```bash
# Instantiate dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run unit tests
julia --project=. test/runtests.jl

# Load the module in the REPL
julia --project=.
julia> using GrassmannDistance
```

For integration testing against a real MQTT broker, copy `.env.example` to `.env` and fill in credentials. Source the file before running integration scripts:

```bash
cp .env.example .env
# edit .env
source .env
julia --project=. test_real.jl
```

---

## Configuration

All configuration is read from environment variables at startup via `load_config()` in `src/config.jl`.

| Variable | Required | Default | Description |
|---|---|---|---|
| `MQTT_BROKER` | Yes | — | MQTT broker URL, e.g. `tcp://<broker-host>:1883` |
| `MQTT_USER` | No | — | MQTT username |
| `MQTT_PASSWORD` | No | — | MQTT password |
| `JOB_ID` | Yes | — | FORGE job ID; determines which MQTT topics to subscribe to |
| `WORKER_ID` | No | `grassmann-{JOB_ID}` | Identifies the worker in status/log messages |
| `USE_GPU` | No | `false` | Enable AMD GPU compute (`true` or `1`) |
| `SYSIMAGE_PATH` | No | `/app/sysimage.so` | Path to a precompiled Julia sysimage; falls back to JIT if absent |
| `JULIA_NUM_THREADS` | No | — | Julia thread count; Nomad job sets this to `1` |
| `JULIA_PROJECT` | — | `/app` | Set inside container; do not override |
| `JULIA_DEPOT_PATH` | — | `/opt/julia-depot` | Set inside container; do not override |
| `ROCM_PATH` | — | `/opt/rocm` | Set inside container and Nomad job for GPU access |

Secrets (`MQTT_BROKER`, `MQTT_USER`, `MQTT_PASSWORD`) are injected from Vault via the Nomad job template using the `forge` policy. See `deploy/nomad/grassmann-distance.hcl`.

---

## Usage

### Run unit tests

```bash
julia --project=. test/runtests.jl
```

Test suites: `Neighbors`, `TangentSpace`, `Distance`, `Ranking`, `Graph`, `Paths`, `Topology`, `Serialization`, `App`.

### Run integration / corpus tests

```bash
# Requires MQTT_BROKER, MQTT_USER, MQTT_PASSWORD in environment
julia --project=. test_real.jl
julia --project=. test_corpus_graph.jl
julia --project=. test_corpus_paths.jl
julia --project=. test_corpus_real.jl
```

### Build the container

```bash
# Build scratch must be on local disk — NFS xattrs break fakeroot unpacking
BUILD_TMPDIR="/var/tmp/singularity-build-$$"
mkdir -p "$BUILD_TMPDIR"
SINGULARITY_TMPDIR="$BUILD_TMPDIR" singularity build --fakeroot worker.sif singularity.def
rm -rf "$BUILD_TMPDIR"
```

The build runs `test/runtests.jl` as a validation step during `%post`.

### Deploy

Push to `main`. The Gitea `build` workflow (`.gitea/workflows/build.yml`) runs on the `packer` self-hosted runner, builds `worker.sif`, writes it to the NFS image store, and registers the Nomad job:

```bash
nomad job run deploy/nomad/grassmann-distance.hcl
```

Manual dispatch for testing:

```bash
nomad job dispatch -meta job_id=<uuid> grassmann-distance
```

### Sysimage (optional, reduces cold-start time)

The container ships without a sysimage. See `TODO.md` for the procedure to build one on the GPU node using ROCm's LLVM and store it on NFS for the container to pick up via `SYSIMAGE_PATH`.

---

## Project Structure

| File | Role |
|---|---|
| `src/GrassmannDistance.jl` | Module entry point; `include` order and all exports |
| `src/types.jl` | Core types: `GrassmannConfig`, `TangentSpace`, `RankingEntry` |
| `src/graph_types.jl` | Graph types: `GraphConfig`, `Entity`, `GrassmannGraph`, `ConceptualPath`, `Community`, `Basin` |
| `src/job_types.jl` | FORGE contract types: all JSON-serializable request/response structs |
| `src/config.jl` | `WorkerConfig`, `load_config()` — reads env vars at startup |
| `src/neighbors.jl` | `knn()` — brute-force k-nearest neighbors (SIMD inner loop) |
| `src/tangent.jl` | `estimate_tangent_space()`, `estimate_tangent_spaces()` — local PCA via SVD |
| `src/distance.jl` | `principal_angles()`, `grassmann_distance()` — geodesic and chordal variants |
| `src/ranking.jl` | `rank_candidates()` — full pipeline: kNN + tangent spaces + ranked distances |
| `src/entity_distance.jl` | `entity_distance()` — chunk-averaged Grassmann distance between entities |
| `src/graph.jl` | `build_graph()` — assembles `GrassmannGraph` from raw embeddings |
| `src/paths.jl` | `find_greedy_path()`, `find_shortest_path()`, `reachable()` |
| `src/topology.jl` | `communities()`, `basins()`, `bridges()`, `hub_centrality()`, `hub_concentration()`, `bidirectional_edges()` |
| `src/serialization.jl` | JSON3 struct mappings, `parse_job()`, `serialize_result()`, `serialize_graph()`, `deserialize_graph()` |
| `src/app.jl` | `process_job()` — dispatches build/query modes; entry logic for `julia_main()` |
| `src/mqtt.jl` | `subscribe_and_process!()`, `julia_main()` — MQTT client loop and worker entry point |
| `src/gpu.jl` | `build_graph_gpu()`, `select_backend()` — GPU-accelerated kNN, tangent space estimation, entity distance matrix |

---

## FORGE Job Protocol

The worker is a **parameterized batch job** — one Nomad dispatch per compute request. It does not poll for multiple jobs; it exits after completing one.

On startup, the worker:

1. Reads `JOB_ID` from the environment
2. Subscribes to `compute/jobs/{job_id}/params` (QoS 1)
3. Waits for a single JSON payload
4. Publishes status to `compute/jobs/{job_id}/status`
5. Processes the job (`build` or `query` mode)
6. Publishes the result to `compute/jobs/{job_id}/result`
7. Publishes final status and disconnects

**Modes:**

- `build` — takes `entities` (id + embeddings) and config parameters; returns a base64-encoded serialized `GrassmannGraph` plus entity/chunk counts.
- `query` — takes an encoded graph and a `QuerySpec`; returns path, reachability, or topology results depending on `query.type`.

`query.type` values: `greedy_path`, `shortest_path`, `reachable`, `communities`, `basins`, `topology`.

The graph blob is opaque to FORGE — it is written and read only by this Julia worker. Clients pass it through unchanged between build and query dispatches.

See [docs/messaging.md](docs/messaging.md) for the full MQTT protocol specification, JSON examples, and topic map.

---

## Related Documents

- [CONCEPTS.md](CONCEPTS.md) — mathematical background: why cosine fails, how tangent spaces and Grassmann distance work, design decisions
- [ARCHITECTURE.md](ARCHITECTURE.md) — system structure, component inventory, data flows, invariants
- [docs/development.md](docs/development.md) — dev setup, all commands, troubleshooting
- [docs/messaging.md](docs/messaging.md) — MQTT protocol, JSON contract, topic map
- `deploy/nomad/grassmann-distance.hcl` — Nomad job definition (constraints, resources, Vault policy, Singularity invocation)
- `TODO.md` — sysimage build procedure and open research questions

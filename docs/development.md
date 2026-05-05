# Development Guide

<!-- @tier: 1 -->
<!-- @see-also: docs/subsystems/ -->

## Overview

GrassmannDistance is a Julia module (v0.1.0) that computes Grassmann distance over local tangent spaces in high-dimensional embedding space. It exposes graph construction, path finding, and topology analysis, and operates as a parameterized batch worker dispatched by a FORGE MQTT broker.

The worker runs inside a Singularity container, consumes a job payload over MQTT, and exits. One dispatch equals one job.

## Prerequisites

- Julia 1.10 or later (1.12 used in the container image)
- `libmosquitto-dev` — required by Mosquitto.jl when building locally outside the container
- `singularity` — to build or run the container image
- `nomad` CLI — to register and dispatch jobs
- ROCm (`/opt/rocm`) on the execution host when `USE_GPU=true`

For corpus and real-embedding tests only:

- An Ollama instance reachable via `OLLAMA_HOST` (model: `qwen3-embedding:8b`)

## Setup

```bash
# Clone and instantiate dependencies
cd grassmann-distance
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Pkg reads `Manifest.toml` and resolves the exact locked versions. Do not run `Pkg.update()` unless you intend to drift the lockfile.

## Configuration

All runtime configuration is read from environment variables in `src/config.jl` and validated in the container runscript (`singularity.def`).

| Variable | Default | Required | Description |
|---|---|---|---|
| `JOB_ID` | — | yes | Nomad dispatch job ID; used as the MQTT topic key |
| `MQTT_BROKER` | `tcp://localhost:1883` | yes in production | MQTT broker URL |
| `MQTT_USER` | — | yes in production | MQTT username |
| `MQTT_PASSWORD` | — | yes in production | MQTT password |
| `WORKER_ID` | `grassmann-{JOB_ID}` | no | Worker identifier logged on startup |
| `USE_GPU` | `false` | no | Set to `true` or `1` to enable AMDGPU dispatch |
| `SYSIMAGE_PATH` | `/app/sysimage.so` | no | Path to an external PackageCompiler sysimage; falls back to JIT if absent |
| `JULIA_NUM_THREADS` | — | no | Thread count passed to Julia runtime (Nomad sets `1`) |
| `ROCM_PATH` | `/opt/rocm` | no | ROCm installation root; must be bind-mounted at runtime |
| `OLLAMA_HOST` | — | scripts only | `<host>:<port>` for embedding generation scripts |

For local testing, copy `.env.example` to `.env` and fill in broker credentials:

```
MQTT_BROKER=tcp://localhost:1883
MQTT_USER=
MQTT_PASSWORD=
```

The `.env` file is not read automatically — source it yourself or pass variables explicitly. It exists solely as a template.

## Running Tests

### Unit tests

Run the full test suite (neighbors, tangent space, distance, ranking, graph, paths, topology, serialization, app):

```bash
julia --project=. test/runtests.jl
```

Test files are in `test/` and are driven by `test/runtests.jl`. Each subsystem has its own file (`test_neighbors.jl`, `test_tangent.jl`, etc.).

### Integration test (local, no network)

Generates synthetic 4096D embeddings clustered around known subspaces and compares Grassmann distance ranking against cosine similarity:

```bash
julia --project=. test_local.jl
```

No external services required.

### Real-embedding test

Requires `test_data/real_embeddings.json` (160+ food/phrase entries at 4096D). Generate it first if absent:

```bash
OLLAMA_HOST=<host>:<port> julia --project=. scripts/generate_embeddings.jl
```

Then run:

```bash
julia --project=. test_real.jl
```

### Corpus tests

Requires `test_data/corpus_embeddings.json`. Generate it first:

```bash
OLLAMA_HOST=<host>:<port> julia --project=. scripts/generate_corpus_embeddings.jl
```

Three corpus test scripts are available at the project root:

```bash
julia --project=. test_corpus_real.jl    # Grassmann vs cosine retrieval precision per document
julia --project=. test_corpus_paths.jl   # Greedy multi-hop paths through document space
julia --project=. test_corpus_graph.jl   # Hub centrality, communities, bridges, basins
```

All three load `test_data/corpus_embeddings.json` and require no network once that file exists.

## Scripts

| Script | Purpose | Required env |
|---|---|---|
| `scripts/generate_embeddings.jl` | Generates `test_data/real_embeddings.json` by embedding 160+ terms via Ollama (`qwen3-embedding:8b`) | `OLLAMA_HOST=<host>:<port>` |
| `scripts/generate_corpus_embeddings.jl` | Chunks markdown files in `test_corpus/` by `##` headings and generates `test_data/corpus_embeddings.json` | `OLLAMA_HOST=<host>:<port>` |

Both scripts write their output under `test_data/` (created automatically) and are idempotent — re-running overwrites the output file.

## Container Build

### Local build

`deploy/build.sh` builds the Singularity image and publishes it to NFS. Run it from the project root on a host with Singularity installed and NFS mounted:

```bash
bash deploy/build.sh
```

The script:

1. Validates that `/nfs/images/grassmann-distance/` and `/nfs/cache/` are mounted and writable.
2. Runs `singularity build` using a per-process temp directory under `/nfs/cache/` as scratch (NFS xattr limitations require scratch on local or NFS stripe-friendly storage; xattrs break fakeroot unpacking).
3. Writes `worker.sif` to `/nfs/images/grassmann-distance/`.
4. Writes `manifest.json` alongside the image with `git_commit` and `built_at`.

The optional `GIT_COMMIT` environment variable overrides the auto-detected commit hash.

The `singularity.def` `%post` section installs system packages (`libmosquitto-dev`, `libdrm-amdgpu1`, `libnuma1`, and ROCm support libraries), instantiates Julia packages, runs the full unit test suite as a build-time validation, and verifies the package loads. The container does **not** build a sysimage (see [Secrets & Deployment](#secrets--deployment) and [When Things Look Wrong](#when-things-look-wrong)).

### CI build (Gitea)

`.gitea/workflows/build.yml` runs on every push to `main` on the self-hosted `packer` runner:

1. Builds the image with `singularity build --fakeroot` using a local tmpdir (`/var/tmp/singularity-build-$$`).
2. Publishes `worker.sif` and `manifest.json` to `/nfs/images/grassmann-distance/`.
3. Registers the Nomad job by running `nomad job run deploy/nomad/grassmann-distance.hcl`.

The `NOMAD_ADDR` Gitea repository variable must be set; it is passed as an environment variable to the Nomad CLI step.

## Deployment

### Nomad job registration

The job definition is at `deploy/nomad/grassmann-distance.hcl`. It defines a parameterized batch job. Registration happens automatically via CI; to register manually:

```bash
NOMAD_ADDR=<nomad-addr> nomad job run deploy/nomad/grassmann-distance.hcl
```

### Manual dispatch (testing)

```bash
nomad job dispatch -meta job_id=<uuid> grassmann-distance
```

### Job behavior

- Type: `batch`, parameterized on `job_id` (required dispatch metadata).
- Constraint: runs only on nodes with `meta.gpu = true` and `meta.rocm = true`. These are node metadata values, not device plugin fingerprints — see [When Things Look Wrong](#when-things-look-wrong).
- Driver: `raw_exec`, invokes `singularity run` directly.
- Singularity flags: `--writable-tmpfs` (required — container filesystem is read-only by default), `--bind /opt/rocm:/opt/rocm`.
- No restart or reschedule attempts. Worker failure is terminal; FORGE handles retry at a higher level.
- Kill timeout: 30 seconds (`SIGTERM`).
- Resources: 2000 MHz CPU, 4096 MB RAM.
- `.sif` is read directly from `/nfs/images/grassmann-distance/worker.sif` — no HTTP artifact download.

### FORGE integration

FORGE dispatches the Nomad job with a `job_id` metadata value. The worker reads `NOMAD_META_job_id` (exposed as `JOB_ID`) and subscribes to the corresponding MQTT topic to receive the job payload and publish results.

## Secrets & Deployment

The Nomad job uses the Vault `forge` policy to retrieve secrets from `secret/data/nomad/forge`. The following fields are expected in that path:

| Field | Injected as env var | Purpose |
|---|---|---|
| `MQTT_BROKER` | `MQTT_BROKER` | Broker URL |
| `MQTT_USER` | `MQTT_USER` | MQTT username |
| `MQTT_PASSWORD` | `MQTT_PASSWORD` | MQTT password |

These are rendered into `secrets/env` by the Nomad `template` block and sourced automatically as environment variables. The worker binary never reads them from any other location.

No secrets are baked into the container image.

## When Things Look Wrong

| Symptom | Check | Fix |
|---|---|---|
| `read-only file system` at runtime | Singularity run invoked without `--writable-tmpfs` | Add `--writable-tmpfs` to the `singularity run` invocation. The Nomad HCL already includes it; the flag is missing only in ad-hoc manual runs. |
| `LLVM ERROR: Broken module found` during sysimage build | Build is happening on a host without ROCm's LLVM; Julia's bundled LLVM cannot handle AMDGPU GCN intrinsics injected by AMDGPU.jl | Build the sysimage on the GPU node where `/opt/rocm` is present (see `TODO.md` for the exact steps). Store the result at a path set via `SYSIMAGE_PATH`. |
| Nomad allocation never placed (`missing devices` or no placement) | Device plugin fingerprinting is not used; the job constrains on `meta.gpu` and `meta.rocm` node metadata | Verify the target nodes have `meta.gpu = true` and `meta.rocm = true` in their Nomad client config. Device plugin constraints would require a different `device` stanza. |
| Precompilation fails at startup inside container | Container image was built against an older `Manifest.toml` | Rebuild the container. The `%post` step runs `Pkg.precompile()` at build time; a stale image will not re-precompile automatically. |
| SVD segfault on GPU node | `rocSOLVER` dispatch issue when SVD is attempted on an AMDGPU array | Expected — SVD runs on CPU by design. If a code path is dispatching SVD to GPU, it is a bug in the caller. |
| Worker exits with no output or error message | Log output goes to stderr, not stdout | Retrieve with `nomad alloc logs -stderr <alloc-id>`. |
| ~30s cold start on each dispatch | No sysimage present; worker JIT-compiles on startup | Acceptable for batch workloads. To eliminate it, follow the sysimage build steps in `TODO.md` and set `SYSIMAGE_PATH` in the Nomad job env block. |

## Related Documents

- `TODO.md` — sysimage build procedure and open research questions
- `CONCEPTS.md` — conceptual background on Grassmann distance and tangent spaces
- `intent-grassmann-geometry-module.md` — original design intent
- `deploy/nomad/grassmann-distance.hcl` — authoritative Nomad job definition
- `singularity.def` — authoritative container build definition

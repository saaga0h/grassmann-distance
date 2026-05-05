# GPU Acceleration

<!-- @tier: 2 -->
<!-- @parent: docs/development.md -->
<!-- @source: src/gpu.jl, src/GrassmannDistance.jl, src/app.jl, src/config.jl -->
<!-- @see-also: singularity.def, deploy/nomad/grassmann-distance.hcl -->

## Overview

GPU acceleration is opt-in and targets AMDGPU via KernelAbstractions.jl. When enabled it replaces the three computationally expensive stages of `build_graph` with GPU-backed equivalents. All other operations — query, serialization, topology analysis — run on CPU regardless of the backend setting.

The GPU path is activated by setting `USE_GPU=true` in the worker environment. `select_backend` in `src/gpu.jl` checks `AMDGPU.functional()` at startup and falls back silently to `CPU()` if no functional AMD GPU is found. This means the worker never fails solely because GPU is unavailable.

GPU compute is only used in `build` mode. The `_process_build` function in `src/app.jl` dispatches to `build_graph_gpu` when the backend is not `CPU()`.

## Key Files and Entry Points

| File | Role |
|---|---|
| `src/gpu.jl` | All GPU-accelerated functions: backend selection, three pipeline stages, device helpers |
| `src/GrassmannDistance.jl` | Imports `KernelAbstractions` and `AMDGPU`; includes `gpu.jl` last after all CPU types are defined |
| `src/app.jl` | `_process_build` — dispatches to `build_graph` or `build_graph_gpu` based on backend |
| `src/config.jl` | `load_config` — reads `USE_GPU` env var, `select_backend` is exported from the module |
| `singularity.def` | ROCm library setup, `ROCM_PATH` and `LD_LIBRARY_PATH`, `--writable-tmpfs` requirement |
| `deploy/nomad/grassmann-distance.hcl` | `USE_GPU=true`, `--bind /opt/rocm:/opt/rocm`, node constraints `meta.gpu` and `meta.rocm` |

## Architecture

### Backend selection

```julia
backend = select_backend(config.use_gpu)
```

`select_backend` takes the boolean from `WorkerConfig.use_gpu` (read from `USE_GPU`). If `use_gpu` is true and `AMDGPU.functional()` returns true, it returns `AMDGPU.ROCBackend()`. Otherwise it returns `CPU()` with a warning log. The function is exported from `GrassmannDistance` so callers outside the module can use it.

### Three GPU stages

`build_graph_gpu` runs the same logical pipeline as `build_graph` but replaces each expensive stage:

| Stage | CPU equivalent | GPU function |
|---|---|---|
| All-pairs kNN | `estimate_tangent_spaces` (calls `knn` per chunk in `neighbors.jl`) | `_gpu_all_knn` |
| Tangent space estimation | `estimate_tangent_spaces` / `estimate_tangent_space` in `tangent.jl` | `_gpu_estimate_tangent_spaces` |
| Entity distance matrix | `entity_distance` nested loop in `entity_distance.jl` | `_gpu_entity_distance_matrix` |

The adjacency build (`_build_adjacency`) always runs on CPU — it operates on the scalar distance matrix returned by stage 3.

### Adding GPU acceleration to a new computation

1. Accept a `backend` parameter alongside the data inputs.
2. Move arrays to the device with `_to_device(x, backend)`. For `CPU()` this is a no-op; for `AMDGPU.ROCBackend()` it allocates a `ROCArray`.
3. Perform operations on the returned device array. Standard broadcast, `sum`, and matrix multiply (`*`) dispatch to GPU kernels automatically via KernelAbstractions.jl.
4. Pull results back to CPU with `Array(d_result)` before any CPU-only operation (sorting, SVD, indexing with non-device indices).

```julia
function my_gpu_fn(data::Matrix{Float64}, backend)
    d_data = _to_device(data, backend)      # host → device
    d_result = sum(d_data .^ 2; dims=1)    # runs on GPU
    return Array(d_result)                  # device → host
end
```

Do not call `_to_device` with a `CPU()` backend and expect zero overhead — for CPU it returns the input unchanged, so there is no copy.

## Data Flow

### Stage 1: `_gpu_all_knn`

Input: `embeddings::Matrix{Float64}` (dim, n), `k::Int`, `backend`

1. `_to_device(embeddings, backend)` → `d_emb` (ROCArray or passthrough)
2. Squared norms: `vec(sum(d_emb .^ 2; dims=1))` → `d_norms` shape (n,)
3. Gram matrix: `d_emb' * d_emb` → `d_gram` shape (n, n), then scaled by -2
4. Distance matrix: broadcast add `d_norms` and `d_norms'` to `d_gram`
5. `Array(d_gram)` → `dist_sq` on CPU
6. `partialsortperm` per column on CPU → `knn_matrix::Matrix{Int}` shape (k, n)

The sort step deliberately runs on CPU. Sorting full rows of an (n, n) matrix on GPU is not worth the kernel launch overhead at typical corpus sizes.

### Stage 2: `_gpu_estimate_tangent_spaces`

Input: `embeddings`, `knn_matrix::Matrix{Int}` (k, n), `p::Int`, `backend`

For each chunk j:
1. Index neighbors from `d_emb` using `knn_matrix[:, j]` — column slice on GPU
2. Center the neighborhood on GPU: subtract mean column
3. `Array(d_centered)` → pull to CPU
4. `svd(h_centered)` on CPU → take first `p` left singular vectors as `TangentSpace.basis`

The SVD always runs on CPU. See [Known Issues](#known-issues).

### Stage 3: `_gpu_entity_distance_matrix`

Input: `tangent_spaces::Vector{TangentSpace}`, `entities`, `config`, `graph_config`, `backend`

1. Enumerate all representative chunk pairs across entity pairs using `_representative_chunks` (same helper as CPU path)
2. Stack all `U` basis matrices into `all_U` (dim, p * n_pairs) and `all_V` (dim, p * n_pairs)
3. `_to_device(all_U, backend)`, `_to_device(all_V, backend)` → `d_U`, `d_V`
4. For each (a, b) index in the (p, p) cross-product block: extract every p-th column from `d_U` and `d_V`, compute element-wise product and column sum on GPU → `(n_pairs,)` dot products
5. `Array(dots)` per index pair, assemble `cross_products::Matrix{Float64}` (p*p, n_pairs) on CPU
6. For each pair: `reshape` the p*p column into a (p, p) matrix M, call `_grassmann_from_cross_product(M, config.distance)` on CPU
7. Average distances across representative chunk pairs per entity pair → symmetric `dist_matrix`

`_grassmann_from_cross_product` runs `svd` on the (p, p) matrix M to get principal angles. For p ≤ 2 this is fast on CPU.

## Interfaces and Contracts

### `select_backend(use_gpu::Bool) -> Union{AMDGPU.ROCBackend, CPU}`

Exported. Called once at worker startup by `load_config` / `process_job` callers.

### `build_graph_gpu(embeddings, entity_ids, chunk_entity_map, grassmann_config, graph_config, backend) -> GrassmannGraph`

Same return type and semantics as `build_graph`. Takes an additional `backend` argument. Validates that `length(chunk_entity_map) == size(embeddings, 2)`. Calls `_build_entities` and `_build_adjacency` from `graph.jl` — these are shared with the CPU path.

### `_to_device(x::AbstractArray, backend) -> AbstractArray`

Internal. For `CPU()` returns `x` unchanged. For `AMDGPU.ROCBackend()` returns `AMDGPU.ROCArray(x)`.

## Dependencies

- `KernelAbstractions` — vendor-neutral GPU compute abstraction; imported unconditionally in `GrassmannDistance.jl`
- `AMDGPU` — AMD ROCm backend; imported unconditionally; `AMDGPU.functional()` returns false gracefully if ROCm is absent
- ROCm on the host at `/opt/rocm` — required at runtime when `USE_GPU=true`; bind-mounted into the Singularity container via `--bind /opt/rocm:/opt/rocm`

`KernelAbstractions` and `AMDGPU` are loaded unconditionally at module import. If AMDGPU cannot initialize (no ROCm libraries), it logs a warning but does not raise. Only `AMDGPU.functional()` controls whether the GPU path is actually used.

## Diagnosing GPU Problems

### Check GPU availability from Julia

```julia
using AMDGPU
AMDGPU.functional()         # false if ROCm is absent or broken
AMDGPU.device()             # shows active device if functional
```

### Verify ROCm bind mount inside container

If the container is run manually (not via Nomad), the `--bind /opt/rocm:/opt/rocm` flag must be present. Without it, `/opt/rocm` is empty inside the container and `AMDGPU.functional()` returns false.

```bash
singularity run --writable-tmpfs --bind /opt/rocm:/opt/rocm worker.sif
```

Omitting `--writable-tmpfs` causes a `read-only file system` error from GPUCompiler when it tries to write compiled kernels to the container filesystem.

### Check stderr for segfaults and LLVM errors

Worker output goes to stderr, not stdout:

```bash
nomad alloc logs -stderr <alloc-id>
```

Look for:
- `Segmentation fault` — indicates a GPU dispatch path hit an unexpected code (most likely SVD dispatching to rocSOLVER; see [Known Issues](#known-issues))
- `LLVM ERROR: Broken module found` — sysimage was built without ROCm LLVM support (see [Known Issues](#known-issues))
- `USE_GPU=true but no functional AMD GPU found` — `AMDGPU.functional()` returned false; worker continues on CPU

### Confirm the backend selected at startup

The `select_backend` function emits an `@info` log with the device name on success and a `@warn` on CPU fallback. These appear in stderr via Julia's default logging.

## Known Issues

### rocSOLVER SVD dispatch segfault

Calling `svd` on a `ROCArray` dispatches to rocSOLVER, which has an unreliable dispatch path for small matrices in the AMDGPU.jl version pinned in `Manifest.toml`. This causes a segfault with no Julia stack trace.

**Fix already applied:** both `_gpu_estimate_tangent_spaces` and `_grassmann_from_cross_product` pull matrices to CPU with `Array(...)` before calling `svd`. SVD is never dispatched to GPU. If a future code path calls `svd` on a `ROCArray`, it will segfault — route it through `Array()` first.

### GPUCompiler requires writable tmpfs

GPUCompiler writes compiled kernel objects to the container filesystem at first use. A read-only container (Singularity default without `--writable-tmpfs`) causes a `read-only file system` error at the point of first GPU kernel dispatch, not at import time.

**Fix:** always pass `--writable-tmpfs` to `singularity run`. The Nomad HCL includes this flag. Manual runs without it will fail silently until the first GPU kernel fires.

### Sysimage LLVM crash with AMDGPU

Building a PackageCompiler sysimage on a host without a full ROCm LLVM installation (e.g., the Gitea CI runner or a non-GPU node) results in `LLVM ERROR: Broken module found`. Julia's bundled LLVM cannot compile AMDGPU GCN intrinsics injected by AMDGPU.jl during precompilation.

**Workaround:** build the sysimage on the GPU execution node where `/opt/rocm` is present. See `TODO.md` for the exact steps. The container itself does not include a sysimage — the sysimage is an optional external artifact loaded via `SYSIMAGE_PATH`.

## Related Documents

- `docs/development.md` — `USE_GPU`, `SYSIMAGE_PATH`, `ROCM_PATH` environment variables; container build; deployment; troubleshooting table
- `src/graph.jl` — CPU `build_graph`, `_build_entities`, `_build_adjacency` (shared with GPU path)
- `src/neighbors.jl` — CPU `knn` (replaced by `_gpu_all_knn`)
- `src/tangent.jl` — CPU `estimate_tangent_spaces` (replaced by `_gpu_estimate_tangent_spaces`)
- `src/entity_distance.jl` — CPU `entity_distance` (replaced by `_gpu_entity_distance_matrix`)
- `TODO.md` — sysimage build procedure on GPU node

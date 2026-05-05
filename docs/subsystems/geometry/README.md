# Geometry Subsystem

<!-- @tier: 2 -->
<!-- @parent: ARCHITECTURE.md -->
<!-- @modules: docs/subsystems/geometry/modules/ -->
<!-- @source: src/types.jl, src/neighbors.jl, src/tangent.jl, src/distance.jl, src/ranking.jl, src/entity_distance.jl, src/gpu.jl -->

## Overview

The geometry subsystem is the computational core of GrassmannDistance. It takes raw embedding vectors and produces Grassmann distances — scalar measures of how similar two points' local geometries are in the ambient embedding space.

The pipeline runs in four stages:

1. **kNN** — find the k nearest neighbors of each embedding vector (brute-force, CPU SIMD)
2. **Tangent space estimation** — fit a p-dimensional subspace to each local neighborhood via SVD
3. **Grassmann distance** — compare pairs of subspaces via principal angles
4. **Ranking / entity distance** — compose the above into usable outputs

For the mathematical rationale behind these choices — why Grassmann distance instead of cosine similarity, what principal angles mean, why p=2 is a practical default — see [CONCEPTS.md](../../../CONCEPTS.md). This document covers code structure, not theory.

---

## Key Files & Entry Points

| File | Role |
|---|---|
| `src/types.jl` | Core struct definitions: `GrassmannConfig`, `TangentSpace`, `RankingEntry` |
| `src/neighbors.jl` | `knn()` — brute-force nearest neighbor search |
| `src/tangent.jl` | `estimate_tangent_space()`, `estimate_tangent_spaces()` — local PCA via SVD |
| `src/distance.jl` | `principal_angles()`, `grassmann_distance()` — the distance computation |
| `src/ranking.jl` | `rank_candidates()` — end-to-end pipeline from embeddings to sorted results |
| `src/entity_distance.jl` | `entity_distance()` — chunk-averaged distance between named entities |
| `src/gpu.jl` | GPU-accelerated variants of kNN, tangent estimation, and entity distance matrix |

---

## Architecture

### Types

Three structs defined in `src/types.jl` carry geometry state:

**`GrassmannConfig`** — parameters that govern the entire pipeline:
- `k::Int` — neighborhood size for local PCA (default: 20)
- `p::Int` — tangent space dimension / number of principal components (default: 2)
- `distance::Symbol` — `:geodesic` (default) or `:chordal`

The `DEFAULT_CONFIG` constant is `GrassmannConfig(20, 2, :geodesic)`.

**`TangentSpace`** — the product of tangent space estimation for one chunk:
- `basis::Matrix{Float64}` — shape `(ambient_dim, p)`, orthonormal columns (verified by `basis' * basis ≈ I(p)`)
- `center::Vector{Float64}` — centroid of the k-neighborhood (not the chunk itself)

The `center` field is stored but not used by `grassmann_distance` or `principal_angles`. It is available for debugging — comparing a chunk's position to its neighborhood centroid reveals how peripheral the chunk is within its local cluster.

**`RankingEntry`** — a single ranked result from `rank_candidates()`:
- `id::String` — entity identifier, passed through from caller input
- `distance::Float64` — Grassmann distance to the query

### kNN (`src/neighbors.jl`)

`knn(query, candidates, k)` computes squared Euclidean distances from `query` to all columns of `candidates` (shape `ambient_dim × n`) using `@simd`-annotated inner loops, sorts by distance, and returns up to `k` column indices.

Self-exclusion threshold is squared distance `< 1e-24` (not `< 1e-12` — the threshold is applied to the squared value). If the query appears verbatim in `candidates`, that column is skipped. If fewer than `k` non-self columns exist, the function returns what is available without error.

`knn` throws `ArgumentError` if `k ≤ 0`.

### Tangent space estimation (`src/tangent.jl`)

`estimate_tangent_space(point, neighbors, p)` accepts:
- `point::AbstractVector` — the embedding vector being described (used only implicitly; the basis is computed from `neighbors`)
- `neighbors::AbstractMatrix` — shape `(ambient_dim, k)` — the pre-selected neighborhood columns
- `p::Int` — number of principal components to retain

It centers `neighbors` by their column mean, computes `svd(centered)`, and returns `TangentSpace(F.U[:, 1:p], center)`.

The `point` argument is not used in the computation — it is accepted for API clarity. The centroid stored in `TangentSpace.center` is the mean of `neighbors`, not `point` itself.

Throws `ArgumentError` if `k < p` (underdetermined system).

`estimate_tangent_spaces(embeddings, config)` is the batch variant: for each column of `embeddings`, it calls `knn` to find neighbors then `estimate_tangent_space`. Returns `Vector{TangentSpace}` of length `n`.

### Grassmann distance (`src/distance.jl`)

`principal_angles(U, V)` computes the canonical angles between the subspaces spanned by the columns of `U` and `V` (both `ambient_dim × p`). The cross-product `M = U'V` is a `p × p` matrix; its singular values are the cosines of the principal angles. Singular values are clamped to `[0, 1]` before `acos` to guard against floating-point values slightly outside the valid domain.

Throws `DimensionMismatch` if `size(U, 2) ≠ size(V, 2)`.

`grassmann_distance(ts1, ts2; distance=:geodesic)` dispatches on the `distance` keyword:
- `:geodesic` — `sqrt(sum(θᵢ²))`, arc length on the Grassmannian
- `:chordal` — `sqrt(sum(sin²(θᵢ)))`, chord length through ambient space

Any other symbol throws `ArgumentError("unknown distance variant: $distance (use :geodesic or :chordal)")`.

The function is symmetric: `grassmann_distance(ts1, ts2) == grassmann_distance(ts2, ts1)` to floating-point precision, because singular values are invariant to transposition of `M`.

### Ranking (`src/ranking.jl`)

Two overloads of `rank_candidates` serve different caller patterns:

**Full pipeline overload:**
```julia
rank_candidates(
    query_embedding::AbstractVector,
    candidate_embeddings::AbstractMatrix,
    candidate_ids::AbstractVector{<:AbstractString},
    all_embeddings::AbstractMatrix,
    config::GrassmannConfig
) -> Vector{RankingEntry}
```
Estimates tangent spaces from scratch — one for the query (using `all_embeddings` for kNN), one for each candidate. Then delegates to the precomputed overload.

**Precomputed overload:**
```julia
rank_candidates(
    query_ts::TangentSpace,
    candidate_ts::AbstractVector{TangentSpace},
    candidate_ids::AbstractVector{<:AbstractString},
    config::GrassmannConfig
) -> Vector{RankingEntry}
```
Assumes tangent spaces are already available. Computes pairwise `grassmann_distance`, builds `RankingEntry` values, and returns them sorted ascending by distance.

Throws `DimensionMismatch` if `length(candidate_ts) ≠ length(candidate_ids)`.

Note: the full pipeline overload uses `all_embeddings` (not `candidate_embeddings`) for kNN neighborhood lookup of both the query and the candidates. This means candidates are evaluated in the context of the full corpus geometry, not just relative to each other.

### Entity distance (`src/entity_distance.jl`)

`entity_distance(tangent_spaces, entity_a, entity_b, config; max_chunks=5)` averages Grassmann distances across all pairs of representative chunks from two entities. When an entity has more than `max_chunks` chunks, `_representative_chunks` selects evenly-spaced positions using `round.(Int, range(1, n; length=max_chunks))`.

The averaging is arithmetic mean over all `|idx_a| × |idx_b|` chunk pairs. The result is always positive (distances are non-negative) and returns `NaN` only if both entities have zero chunks — which `_build_entities` prevents upstream.

`_representative_chunks(indices::UnitRange{Int}, max_chunks::Int)` is deterministic: given the same `indices` and `max_chunks`, it always returns the same positions. This makes entity distances reproducible across runs.

### GPU acceleration (`src/gpu.jl`)

The GPU path covers the same three stages as the CPU path but with different implementations:

| Stage | CPU function | GPU function |
|---|---|---|
| All-pairs kNN | `knn()` called per chunk | `_gpu_all_knn()` — single GEMM |
| Tangent space estimation | `estimate_tangent_spaces()` | `_gpu_estimate_tangent_spaces()` — gather/center on GPU, SVD on CPU |
| Entity distance matrix | `entity_distance()` called per pair | `_gpu_entity_distance_matrix()` — batched dot products on GPU |

`_grassmann_from_cross_product(M, distance)` is the GPU path's distance kernel. It accepts an already-computed `(p, p)` cross-product matrix `M = U'V` (pre-computed on GPU), runs `svd(M)` on CPU, clamps singular values, and dispatches on `distance` with the same `:geodesic` / `:chordal` logic as `grassmann_distance`.

---

## Data Flow

```
embeddings::Matrix{Float64}   (ambient_dim × n_chunks)
        │
        ├─ knn(point, embeddings, k)
        │       → Vector{Int}  (neighbor column indices)
        │
        ├─ estimate_tangent_space(point, neighbors, p)
        │       → TangentSpace  (basis: ambient_dim×p, center: ambient_dim)
        │
        ├─ [batch] estimate_tangent_spaces(embeddings, config)
        │       → Vector{TangentSpace}  (one per chunk)
        │
        ├─ principal_angles(ts1.basis, ts2.basis)
        │       → Vector{Float64}  (p principal angles in [0, π/2])
        │
        ├─ grassmann_distance(ts1, ts2; distance=:geodesic)
        │       → Float64
        │
        ├─ rank_candidates(query_embedding, candidate_embeddings, candidate_ids,
        │                  all_embeddings, config)
        │       → Vector{RankingEntry}  (sorted ascending by distance)
        │
        └─ entity_distance(tangent_spaces, entity_a, entity_b, config)
                → Float64  (chunk-averaged Grassmann distance)
```

File-to-file dependency order: `types.jl` → `neighbors.jl` → `tangent.jl` → `distance.jl` → `ranking.jl` and `entity_distance.jl` (both depend on the three above).

---

## Interfaces & Contracts

**`knn` contract:**
- Input: `candidates` is column-major, shape `(ambient_dim, n)`
- Output: column indices into `candidates`, length ≤ `k`
- Columns with squared distance `< 1e-24` to `query` are excluded (self-match suppression)
- Return length may be less than `k` if fewer non-self candidates exist

**`TangentSpace` invariant:**
- `basis' * basis ≈ I(p)` — columns are orthonormal
- This invariant is established by `estimate_tangent_space` via SVD and must be preserved by any caller that constructs `TangentSpace` directly (e.g., in tests)
- Violation silently produces wrong distances: `principal_angles` assumes orthonormal input; non-orthonormal columns cause singular values to fall outside `[0, 1]` before clamping, which the `clamp.(..., 0.0, 1.0)` call in `principal_angles` patches — but the resulting angles will be meaningless

**`grassmann_distance` contract:**
- Both `TangentSpace` arguments must have the same `p` (number of basis columns)
- `distance` must be `:geodesic` or `:chordal`; any other value throws `ArgumentError`
- Result is `0.0` for identical subspaces, `sqrt(p) * π/2` for fully orthogonal p-dimensional subspaces

**`rank_candidates` (precomputed) contract:**
- `length(candidate_ts) == length(candidate_ids)` — throws `DimensionMismatch` if violated
- All `TangentSpace` values in `candidate_ts` must have the same `p` as `query_ts`

---

## Extension Points

### Adding a new distance variant

Distance dispatch lives in two places. Both must be updated together:

**1. `grassmann_distance` in `src/distance.jl`** — add a new `elseif` branch:

```julia
elseif distance === :your_variant
    return sqrt(sum(f.(θ) .^ 2))  # replace f with your function of principal angles
```

**2. `_grassmann_from_cross_product` in `src/gpu.jl`** — add the same branch. This function is the GPU path's equivalent; it receives the already-computed cross-product matrix but applies the same angle-to-distance formula:

```julia
elseif distance === :your_variant
    return sqrt(sum(f.(θ) .^ 2))
```

If only `grassmann_distance` is updated, CPU and GPU paths will diverge silently: GPU builds will throw `ArgumentError` on the new variant while CPU builds succeed. The `ArgumentError` fallback in `_grassmann_from_cross_product` currently reads `return sqrt(sum(sin.(θ) .^ 2))` (chordal) with no guard — it falls through to chordal rather than throwing. Add an explicit `else throw(ArgumentError(...))` when adding new variants to make mismatches detectable.

---

## Diagnostics

### Distance values look wrong

**Check principal angle output directly:**

```julia
θ = principal_angles(ts1.basis, ts2.basis)
```

Expected ranges:
- All zeros: subspaces are identical
- All `π/2`: subspaces are orthogonal (maximum distance for this p)
- Values outside `[0, π/2]`: basis columns are not orthonormal — `principal_angles` clamps singular values to `[0, 1]` before `acos`, so the angles will be clipped rather than erroring, but the result is geometrically meaningless

**Verify basis orthonormality:**

```julia
ts.basis' * ts.basis ≈ I(p)  # should be true to ~1e-10
```

If this fails, the `TangentSpace` was constructed with a non-orthonormal basis. This should not happen via `estimate_tangent_space` (SVD guarantees orthonormality), but can happen if a `TangentSpace` is constructed manually in tests or by external callers.

**Check neighborhood size relative to p:**

If `k < p` at the time `estimate_tangent_space` is called, it throws `ArgumentError`. If `k` is only slightly larger than `p` (e.g., `k=3, p=2`), the tangent space is technically valid but numerically unstable — the SVD has little data to work with. Increase `k` or decrease `p`.

**Check for degenerate neighborhoods:**

If all k neighbors are near-identical (a degenerate cluster), the centered neighborhood matrix is near-zero. SVD will produce an orthonormal basis, but it will be arbitrary — the singular values will all be near zero and the resulting tangent space does not capture real local geometry. This produces distances that are numerically valid but semantically meaningless. Detect by checking `F.S[1]` (largest singular value of the centered neighborhood): values below `1e-6` indicate a degenerate neighborhood.

### Failure modes

| Symptom | Check | Fix |
|---|---|---|
| `DimensionMismatch` from `principal_angles` | `size(ts1.basis, 2) ≠ size(ts2.basis, 2)` — `p` values differ | Ensure all `TangentSpace` values were estimated with the same `GrassmannConfig.p` |
| `DimensionMismatch` from `rank_candidates` | `length(candidate_ts) ≠ length(candidate_ids)` | Align the two vectors before calling |
| `ArgumentError` from `knn` | `k ≤ 0` | Pass a positive `k` |
| `ArgumentError` from `estimate_tangent_space` | `k < p` — neighborhood too small | Increase `GrassmannConfig.k` or decrease `GrassmannConfig.p` |
| `ArgumentError` from `grassmann_distance` | Unknown `distance` symbol | Use `:geodesic` or `:chordal` |
| `NaN` distance | Basis columns not orthonormal; `clamp` in `principal_angles` masks the root cause | Verify `basis' * basis ≈ I(p)`; re-estimate tangent spaces if needed |
| Distances all identical (no contrast) | Neighborhood too large — tangent spaces average out to similar orientations | Reduce `GrassmannConfig.k`; alternatively reduce `p` |
| `knn` returns fewer than `k` results | Fewer non-self candidates than `k` | Expected behavior; not an error. Check if embedding matrix has enough columns |

---

## Dependencies

Internal (within this package):
- `types.jl` has no dependencies
- `neighbors.jl` has no dependencies
- `tangent.jl` depends on `types.jl`, `neighbors.jl`
- `distance.jl` depends on `types.jl`
- `ranking.jl` depends on `types.jl`, `neighbors.jl`, `tangent.jl`, `distance.jl`
- `entity_distance.jl` depends on `types.jl`, `distance.jl`, `graph_types.jl`
- `gpu.jl` depends on all of the above plus `graph_types.jl`, `graph.jl`

Standard library:
- `LinearAlgebra` — `svd`, `I`, `norm`, matrix multiply
- `Statistics` — `mean` (used in `estimate_tangent_space` via `vec(mean(neighbors, dims=2))`)

External:
- `AMDGPU` — GPU array type and ROCm backend (GPU path only; CPU path has no external geometry dependencies)
- `KernelAbstractions` — backend abstraction used by `select_backend()` / `CPU()` dispatch

---

## Module Index

<!-- TODO: populate when docs/subsystems/geometry/modules/ contains per-function module docs -->

---

## Related Documents

- [CONCEPTS.md](../../../CONCEPTS.md) — mathematical rationale: principal angles, Grassmann manifold, why geodesic over chordal, why p=2
- [ARCHITECTURE.md](../../../ARCHITECTURE.md) — system overview, full component inventory, GPU acceleration strategy, known constraints (rocSOLVER SVD reliability)
- [docs/subsystems/graph/README.md](../graph/README.md) — how geometry outputs feed into graph construction and topology analysis
- [docs/subsystems/gpu/README.md](../gpu/README.md) — GPU-accelerated geometry variants in detail

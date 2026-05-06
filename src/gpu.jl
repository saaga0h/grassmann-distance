# ── GPU-accelerated graph construction ───────────────────────────────────────
#
# Three stages:
# 1. All-pairs kNN      — single GPU matmul, O(n²·d)
# 2. Tangent spaces     — batch gather on GPU, parallel SVD on CPU with threads
# 3. Entity distances   — pre-stacked bases, batched GPU dot products
#
# Memory budgets (tune to available VRAM and system RAM):
#   TANGENT_BATCH_SIZE — chunks per tangent gather batch
#                        GPU + CPU: dim × k × TANGENT_BATCH_SIZE × 8 bytes each
#                        at 4096D k=20: ~1.3 GB per batch
#   ENTITY_PAIR_BATCH  — chunk pairs per entity distance batch
#                        GPU: 2 × dim × p × ENTITY_PAIR_BATCH × 8 bytes
#                        at 4096D p=2: ~6.4 GB per batch

const TANGENT_BATCH_SIZE = 2_000
const ENTITY_PAIR_BATCH  = 20_000

# ── Backend selection ─────────────────────────────────────────────────────────

function select_backend(use_gpu::Bool)
    if use_gpu
        if AMDGPU.functional()
            @info "AMD GPU backend selected" device=AMDGPU.device()
            return AMDGPU.ROCBackend()
        else
            @warn "USE_GPU=true but no functional AMD GPU found, falling back to CPU"
        end
    end
    return CPU()
end

# ── Stage 1: GPU all-pairs kNN ────────────────────────────────────────────────
#
# ||xi - xj||² = ||xi||² + ||xj||² - 2·xi'xj
# The cross-term is one (n,n) matmul on GPU.

function _gpu_all_knn(embeddings::Matrix{Float64}, k::Int, backend)
    dim, n = size(embeddings)
    d_emb   = _to_device(embeddings, backend)
    d_norms = vec(sum(d_emb .^ 2; dims=1))
    d_gram  = d_emb' * d_emb
    d_gram .*= -2.0
    d_gram  .+= d_norms
    d_gram  .+= d_norms'
    dist_sq = Array(d_gram)

    knn_matrix = Matrix{Int}(undef, k, n)
    for j in 1:n
        dist_sq[j, j] = Inf
        knn_matrix[:, j] = partialsortperm(view(dist_sq, :, j), 1:k)
    end
    return knn_matrix
end

# ── Stage 2: Batch gather + parallel SVD ──────────────────────────────────────
#
# Previous approach: per-chunk GPU→CPU round-trip then sequential SVD.
# Each Array() call syncs the GPU and does a DMA transfer — for n chunks that
# is n stalls.
#
# New approach:
#   - Gather TANGENT_BATCH_SIZE neighborhoods on GPU in one fancy-index op
#   - One GPU→CPU transfer per batch (dim × k × batch_n array)
#   - Threads.@threads parallel SVDs over the batch on CPU

function _gpu_estimate_tangent_spaces(
    embeddings::Matrix{Float64}, knn_matrix::Matrix{Int}, p::Int, backend
)
    dim, n = size(embeddings)
    k      = size(knn_matrix, 1)
    d_emb  = _to_device(embeddings, backend)
    spaces = Vector{TangentSpace}(undef, n)

    for batch_start in 1:TANGENT_BATCH_SIZE:n
        batch_end = min(batch_start + TANGENT_BATCH_SIZE - 1, n)
        batch_n   = batch_end - batch_start + 1

        # Flatten neighbor indices for the whole batch: (k*batch_n,)
        batch_idx = vec(knn_matrix[:, batch_start:batch_end])
        d_idx     = _to_device(batch_idx, backend)

        # Single GPU gather + single transfer: (dim, k, batch_n)
        h_all = reshape(Array(d_emb[:, d_idx]), dim, k, batch_n)

        Threads.@threads for local_j in 1:batch_n
            j           = batch_start + local_j - 1
            h_neighbors = h_all[:, :, local_j]        # (dim, k) — independent copy per thread
            center      = vec(mean(h_neighbors; dims=2))
            h_neighbors .-= center                    # center in-place on the copy
            F           = svd(h_neighbors)
            spaces[j]   = TangentSpace(F.U[:, 1:p], center)
        end
        AMDGPU.synchronize()
        GC.gc(false)
    end

    return spaces
end

# ── Stage 3: Batched entity distance matrix ────────────────────────────────────
#
# Pre-stacks all tangent space bases into one GPU matrix so they can be
# gathered by index without repeated host→device transfers.
# Processes (entity_i, entity_j, chunk_a, chunk_b) tuples in batches of
# ENTITY_PAIR_BATCH to keep GPU memory bounded regardless of corpus size.
# Distance computation (Threads.@threads) runs in parallel; accumulation is
# sequential (trivial vs GPU work).

function _gpu_entity_distance_matrix(
    tangent_spaces::Vector{TangentSpace},
    entities::Vector{Entity},
    config::GrassmannConfig,
    graph_config::GraphConfig,
    backend
)
    n_entities = length(entities)
    p          = config.p
    n_chunks   = length(tangent_spaces)
    dim        = size(tangent_spaces[1].basis, 1)

    # Stack all bases: columns (j-1)*p+1 : j*p hold the p-dim basis of chunk j
    h_bases = Matrix{Float64}(undef, dim, p * n_chunks)
    for (j, ts) in enumerate(tangent_spaces)
        h_bases[:, (j-1)*p+1 : j*p] = ts.basis
    end
    d_bases = _to_device(h_bases, backend)

    rep_chunks = [_representative_chunks(e.chunk_indices, graph_config.max_chunks)
                  for e in entities]

    # Build full quad-tuple list: (entity_i, entity_j, chunk_a, chunk_b)
    all_pairs = NTuple{4,Int}[]
    sizehint!(all_pairs, n_entities * (n_entities - 1) ÷ 2 * graph_config.max_chunks^2)
    for i in 1:n_entities, j in (i+1):n_entities
        for ca in rep_chunks[i], cb in rep_chunks[j]
            push!(all_pairs, (i, j, ca, cb))
        end
    end

    n_total = length(all_pairs)
    n_total == 0 && return zeros(Float64, n_entities, n_entities)

    pair_sum   = zeros(Float64, n_entities, n_entities)
    pair_count = zeros(Int,     n_entities, n_entities)

    for batch_start in 1:ENTITY_PAIR_BATCH:n_total
        batch_end = min(batch_start + ENTITY_PAIR_BATCH - 1, n_total)
        batch     = @view all_pairs[batch_start:batch_end]
        n_bp      = length(batch)

        # Build column indices into d_bases for A and B sides of each chunk pair
        A_cols = Vector{Int}(undef, n_bp * p)
        B_cols = Vector{Int}(undef, n_bp * p)
        @inbounds for (idx, (_, _, ca, cb)) in enumerate(batch)
            for q in 1:p
                A_cols[(idx-1)*p + q] = (ca-1)*p + q
                B_cols[(idx-1)*p + q] = (cb-1)*p + q
            end
        end

        d_A = d_bases[:, _to_device(A_cols, backend)]   # (dim, n_bp*p)
        d_B = d_bases[:, _to_device(B_cols, backend)]   # (dim, n_bp*p)

        # p² dot products for every chunk pair: cross_products[a*p+b, k] = u_a · v_b for pair k
        cross_products = Matrix{Float64}(undef, p * p, n_bp)
        for a in 1:p, b in 1:p
            u_cols = d_A[:, a:p:(p*n_bp)]                                    # (dim, n_bp)
            v_cols = d_B[:, b:p:(p*n_bp)]                                    # (dim, n_bp)
            cross_products[(a-1)*p + b, :] = Array(vec(sum(u_cols .* v_cols; dims=1)))
        end

        # Compute distances in parallel, then accumulate sequentially
        dists = Vector{Float64}(undef, n_bp)
        Threads.@threads for idx in 1:n_bp
            M          = reshape(view(cross_products, :, idx), p, p)
            dists[idx] = _grassmann_from_cross_product(M, config.distance)
        end

        for (idx, (ei, ej, _, _)) in enumerate(batch)
            pair_sum[ei, ej]   += dists[idx]
            pair_count[ei, ej] += 1
        end
        AMDGPU.synchronize()
        GC.gc(false)
    end

    # Average and symmetrize
    dist_matrix = zeros(Float64, n_entities, n_entities)
    for i in 1:n_entities, j in (i+1):n_entities
        c = pair_count[i, j]
        if c > 0
            avg = pair_sum[i, j] / c
            dist_matrix[i, j] = avg
            dist_matrix[j, i] = avg
        end
    end

    return dist_matrix
end

# ── Grassmann distance from cross-product matrix ──────────────────────────────
#
# For p=2 (the default), uses an analytical eigenvalue formula for the 2×2
# M'M matrix — avoids LAPACK dispatch overhead on potentially millions of calls.
# Falls back to LAPACK SVD for p > 2.

function _grassmann_from_cross_product(M::AbstractMatrix, distance::Symbol)
    if size(M, 1) == 2
        return _grassmann_2x2(M[1,1], M[2,1], M[1,2], M[2,2], distance)
    end
    F = svd(M)
    σ = clamp.(F.S, 0.0, 1.0)
    θ = acos.(σ)
    return distance === :geodesic ? sqrt(sum(θ .^ 2)) : sqrt(sum(sin.(θ) .^ 2))
end

# Analytical singular values for a 2×2 matrix.
# M stored column-major: M = [a c; b d], so M'M = [a²+b², ac+bd; ac+bd, c²+d²].
@inline function _grassmann_2x2(a::Float64, b::Float64, c::Float64, d::Float64,
                                 distance::Symbol)
    m11       = a*a + b*b
    m12       = a*c + b*d
    m22       = c*c + d*d
    half_tr   = (m11 + m22) * 0.5
    half_disc = sqrt(max(0.0, ((m11 - m22) * 0.5)^2 + m12*m12))
    σ1 = clamp(sqrt(max(0.0, half_tr + half_disc)), 0.0, 1.0)
    σ2 = clamp(sqrt(max(0.0, half_tr - half_disc)), 0.0, 1.0)
    θ1 = acos(σ1)
    θ2 = acos(σ2)
    return distance === :geodesic ? sqrt(θ1*θ1 + θ2*θ2) : sqrt(sin(θ1)^2 + sin(θ2)^2)
end

# ── GPU build_graph entry point ───────────────────────────────────────────────

function build_graph_gpu(
    embeddings::AbstractMatrix{<:Real},
    entity_ids::AbstractVector{<:AbstractString},
    chunk_entity_map::AbstractVector{<:AbstractString},
    grassmann_config::GrassmannConfig,
    graph_config::GraphConfig,
    backend
)
    n_chunks = size(embeddings, 2)
    length(chunk_entity_map) == n_chunks || throw(DimensionMismatch(
        "chunk_entity_map length ($(length(chunk_entity_map))) ≠ embedding columns ($n_chunks)"))

    emb          = Matrix{Float64}(embeddings)
    entities     = _build_entities(entity_ids, chunk_entity_map)
    entity_index = Dict(e.id => i for (i, e) in enumerate(entities))
    n_entities   = length(entities)

    @info "GPU: computing all-pairs kNN" chunks=n_chunks k=grassmann_config.k
    knn_matrix = _gpu_all_knn(emb, grassmann_config.k, backend)
    AMDGPU.synchronize(); GC.gc()

    @info "GPU: estimating tangent spaces" chunks=n_chunks p=grassmann_config.p threads=Threads.nthreads()
    ts = _gpu_estimate_tangent_spaces(emb, knn_matrix, grassmann_config.p, backend)
    AMDGPU.synchronize(); GC.gc()

    @info "GPU: computing entity distance matrix" entities=n_entities pair_batches=cld(n_entities*(n_entities-1)÷2*graph_config.max_chunks^2, ENTITY_PAIR_BATCH)
    dist_matrix = _gpu_entity_distance_matrix(ts, entities, grassmann_config, graph_config, backend)

    k   = min(graph_config.k_graph, n_entities - 1)
    adj = _build_adjacency(dist_matrix, k)

    return GrassmannGraph(
        entities, entity_index,
        emb, ts, dist_matrix, adj,
        grassmann_config, graph_config
    )
end

# ── Device memory helpers ─────────────────────────────────────────────────────

function _to_device(x::AbstractArray, backend)
    backend isa CPU ? x : AMDGPU.ROCArray(x)
end

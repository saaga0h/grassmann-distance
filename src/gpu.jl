# ── GPU-accelerated graph construction ───────────────────────────────────────
#
# Accelerates the three expensive stages of build_graph:
# 1. All-pairs distance matrix for kNN (O(n² × dim) — single GPU matmul)
# 2. Tangent space estimation (batched SVD on GPU)
# 3. Entity distance matrix (batched basis cross-products on GPU)
#
# CPU fallback: existing code in neighbors.jl, tangent.jl, entity_distance.jl

# ── Backend selection ────────────────────────────────────────────────────────

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

# ── Stage 1: GPU-accelerated all-pairs kNN ──────────────────────────────────
#
# For n chunks of dim d:
#   ||xi - xj||² = ||xi||² + ||xj||² - 2·xi'xj
# The xi'xj term is a single (n,n) matmul on GPU.

"""
    _gpu_all_knn(embeddings, k, backend) -> Matrix{Int}

Compute k nearest neighbors for all columns simultaneously on GPU.
Returns (k, n) matrix where column j contains the k neighbor indices for chunk j.
"""
function _gpu_all_knn(embeddings::Matrix{Float64}, k::Int, backend)
    dim, n = size(embeddings)

    # Move to device
    d_emb = _to_device(embeddings, backend)

    # Squared norms: (n,)
    d_norms = vec(sum(d_emb .^ 2; dims=1))

    # Gram matrix: (n, n) = -2 * X' * X (negative so smaller = closer)
    d_gram = d_emb' * d_emb
    d_gram .*= -2.0

    # Distance matrix: ||xi||² + ||xj||² - 2·xi'xj
    # Broadcast: d_gram[i,j] += d_norms[i] + d_norms[j]
    d_gram .+= d_norms
    d_gram .+= d_norms'

    # Pull back to CPU for sorting (GPU sort of full rows isn't worth it)
    dist_sq = Array(d_gram)

    # Build kNN index matrix
    knn_matrix = Matrix{Int}(undef, k, n)
    for j in 1:n
        # Zero out self-distance to push it to end after we handle it
        dist_sq[j, j] = Inf
        perm = partialsortperm(view(dist_sq, :, j), 1:k)
        knn_matrix[:, j] = perm
    end

    return knn_matrix
end

# ── Stage 2: GPU-accelerated tangent space estimation ────────────────────────

"""
    _gpu_estimate_tangent_spaces(embeddings, knn_matrix, p, backend) -> Vector{TangentSpace}

Estimate tangent spaces for all chunks using precomputed kNN indices.
Neighborhood gathering and centering on GPU, SVD via GPU BLAS.
"""
function _gpu_estimate_tangent_spaces(
    embeddings::Matrix{Float64}, knn_matrix::Matrix{Int}, p::Int, backend
)
    dim, n = size(embeddings)
    k = size(knn_matrix, 1)
    d_emb = _to_device(embeddings, backend)

    spaces = Vector{TangentSpace}(undef, n)

    for j in 1:n
        # Gather neighbors on GPU
        idx = @view knn_matrix[:, j]
        d_neighbors = d_emb[:, idx]

        # Center on GPU
        d_center = vec(sum(d_neighbors; dims=2)) ./ k
        d_centered = d_neighbors .- d_center

        # Pull to CPU for SVD — rocSOLVER dispatch is unreliable,
        # and these matrices are small (dim x k) so CPU SVD is fast
        h_centered = Array(d_centered)
        h_center = Array(d_center)

        F = svd(h_centered)
        spaces[j] = TangentSpace(F.U[:, 1:p], vec(h_center))
    end

    return spaces
end

# ── Stage 3: GPU-accelerated entity distance matrix ──────────────────────────
#
# For each entity pair, compute Grassmann distances between representative
# chunk tangent spaces. The expensive part is the basis cross-product U'V.
#
# For p=2, principal angles from singular values of a 2×2 matrix:
#   σ₁, σ₂ = singular values of U'V
#   θ₁, θ₂ = acos(clamp(σ, 0, 1))
#   geodesic = √(θ₁² + θ₂²)

"""
    _gpu_entity_distance_matrix(tangent_spaces, entities, config, graph_config, backend) -> Matrix{Float64}

Compute the full entity-to-entity distance matrix with GPU-accelerated
basis cross-products.
"""
function _gpu_entity_distance_matrix(
    tangent_spaces::Vector{TangentSpace},
    entities::Vector{Entity},
    config::GrassmannConfig,
    graph_config::GraphConfig,
    backend
)
    n_entities = length(entities)
    p = config.p

    # Collect all representative chunk indices and their entity pair assignments
    pairs = Tuple{Int, Int, Int, Int}[]  # (entity_i, entity_j, chunk_a, chunk_b)
    for i in 1:n_entities
        for j in (i+1):n_entities
            idx_a = _representative_chunks(entities[i].chunk_indices, graph_config.max_chunks)
            idx_b = _representative_chunks(entities[j].chunk_indices, graph_config.max_chunks)
            for ca in idx_a, cb in idx_b
                push!(pairs, (i, j, ca, cb))
            end
        end
    end

    n_pairs = length(pairs)

    if n_pairs == 0
        dist_matrix = zeros(Float64, n_entities, n_entities)
        return dist_matrix
    end

    # Stack all basis matrices for batch cross-product on GPU
    # Each basis is (dim, p), cross-product is (p, p) = U'V
    # Batch: build (p, n_pairs) × 2 by stacking U' rows and V columns
    bases_a = Matrix{Float64}(undef, p, size(tangent_spaces[1].basis, 1) * n_pairs)
    bases_b = Matrix{Float64}(undef, size(tangent_spaces[1].basis, 1), p * n_pairs)

    dim = size(tangent_spaces[1].basis, 1)

    # Actually, the most efficient GPU approach: batch all U'V as a single
    # operation by interleaving. But for clarity and correctness, we'll
    # compute cross-products in batches and use analytical SVD for small p.

    # For moderate pair counts, batch the cross-products on GPU
    # Stack all U matrices: (dim, p * n_pairs) and all V: (dim, p * n_pairs)
    all_U = Matrix{Float64}(undef, dim, p * n_pairs)
    all_V = Matrix{Float64}(undef, dim, p * n_pairs)

    for (idx, (_, _, ca, cb)) in enumerate(pairs)
        col_range = ((idx-1)*p+1):(idx*p)
        all_U[:, col_range] = tangent_spaces[ca].basis
        all_V[:, col_range] = tangent_spaces[cb].basis
    end

    # GPU batch cross-product: M = U' * V, but we need per-pair (p,p) blocks
    # Compute full (p*n_pairs, p*n_pairs) = all_U' * all_V on GPU, extract diagonal blocks
    # That's wasteful. Instead, compute column-wise dot products.
    #
    # For p=1: M is scalar = dot(u, v), one matmul gives all pairs
    # For p=2: M is 2×2, need 4 dot products per pair
    #
    # Reshape to (dim, p, n_pairs) and use batched matmul pattern:
    # For each pair: M[a,b] = sum(U[:,a] .* V[:,b])

    d_U = _to_device(all_U, backend)
    d_V = _to_device(all_V, backend)

    # Compute element-wise products and sum for each (p,p) block
    # M_ab for pair k = dot(U[:, k*p+a], V[:, k*p+b])
    # = sum along dim of d_U[:, k*p+a] .* d_V[:, k*p+b]
    cross_products = Matrix{Float64}(undef, p * p, n_pairs)

    for a in 1:p, b in 1:p
        # Extract every p-th column offset by a-1 and b-1
        u_cols = d_U[:, a:p:(p*n_pairs)]   # (dim, n_pairs)
        v_cols = d_V[:, b:p:(p*n_pairs)]   # (dim, n_pairs)
        dots = vec(sum(u_cols .* v_cols; dims=1))  # (n_pairs,)
        cross_products[(a-1)*p + b, :] = Array(dots)
    end

    # Now compute Grassmann distances from cross-products
    dist_matrix = zeros(Float64, n_entities, n_entities)
    pair_dists = Dict{Tuple{Int,Int}, Vector{Float64}}()

    for (idx, (ei, ej, _, _)) in enumerate(pairs)
        M = reshape(view(cross_products, :, idx), p, p)
        d = _grassmann_from_cross_product(M, config.distance)
        key = (ei, ej)
        if !haskey(pair_dists, key)
            pair_dists[key] = Float64[]
        end
        push!(pair_dists[key], d)
    end

    for ((i, j), dists) in pair_dists
        avg = sum(dists) / length(dists)
        dist_matrix[i, j] = avg
        dist_matrix[j, i] = avg
    end

    return dist_matrix
end

"""
    _grassmann_from_cross_product(M, distance_type) -> Float64

Compute Grassmann distance from the (p,p) cross-product matrix M = U'V.
Uses SVD to get principal angles, then computes geodesic or chordal distance.
For p ≤ 2, this is fast even on CPU.
"""
function _grassmann_from_cross_product(M::AbstractMatrix, distance::Symbol)
    F = svd(M)
    σ = clamp.(F.S, 0.0, 1.0)
    θ = acos.(σ)

    if distance === :geodesic
        return sqrt(sum(θ .^ 2))
    else
        return sqrt(sum(sin.(θ) .^ 2))
    end
end

# ── GPU build_graph entry point ──────────────────────────────────────────────

"""
    build_graph_gpu(embeddings, entity_ids, chunk_entity_map, grassmann_config, graph_config, backend) -> GrassmannGraph

GPU-accelerated graph construction. Same interface as `build_graph` with
an additional `backend` parameter from `select_backend()`.
"""
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

    emb = Matrix{Float64}(embeddings)

    entities = _build_entities(entity_ids, chunk_entity_map)
    entity_index = Dict(e.id => i for (i, e) in enumerate(entities))
    n_entities = length(entities)

    # Stage 1: all-pairs kNN on GPU
    @info "GPU: computing all-pairs kNN" chunks=n_chunks k=grassmann_config.k
    knn_matrix = _gpu_all_knn(emb, grassmann_config.k, backend)

    # Stage 2: tangent spaces on GPU
    @info "GPU: estimating tangent spaces" chunks=n_chunks p=grassmann_config.p
    ts = _gpu_estimate_tangent_spaces(emb, knn_matrix, grassmann_config.p, backend)

    # Stage 3: entity distance matrix on GPU
    @info "GPU: computing entity distance matrix" entities=n_entities
    dist_matrix = _gpu_entity_distance_matrix(ts, entities, grassmann_config, graph_config, backend)

    # Build k-NN adjacency (cheap, CPU only)
    k = min(graph_config.k_graph, n_entities - 1)
    adj = _build_adjacency(dist_matrix, k)

    return GrassmannGraph(
        entities, entity_index,
        emb, ts, dist_matrix, adj,
        grassmann_config, graph_config
    )
end

# ── Device memory helpers ────────────────────────────────────────────────────

function _to_device(x::AbstractArray, backend)
    if backend isa CPU
        return x
    else
        # AMDGPU.ROCBackend → ROCArray
        return AMDGPU.ROCArray(x)
    end
end

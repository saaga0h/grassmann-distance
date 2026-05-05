# ── Local tangent space estimation via PCA ───────────────────────────────────

"""
    estimate_tangent_space(point, neighbors, p) -> TangentSpace

Estimate the local tangent space at `point` from its `neighbors` matrix
(ambient_dim, k). Returns p principal components as orthonormal basis columns.
"""
function estimate_tangent_space(
    point::AbstractVector,
    neighbors::AbstractMatrix,
    p::Int
)
    k = size(neighbors, 2)
    k >= p || throw(ArgumentError("need k ≥ p neighbors for $p components, got k=$k"))

    center = vec(mean(neighbors, dims=2))
    centered = neighbors .- center
    F = svd(centered)
    basis = F.U[:, 1:p]

    return TangentSpace(basis, center)
end

"""
    estimate_tangent_spaces(embeddings, config) -> Vector{TangentSpace}

Estimate tangent spaces for all entities in `embeddings` (ambient_dim, n).
Uses kNN to find each entity's neighborhood.
"""
function estimate_tangent_spaces(
    embeddings::AbstractMatrix,
    config::GrassmannConfig
)
    n = size(embeddings, 2)
    spaces = Vector{TangentSpace}(undef, n)

    for j in 1:n
        point = @view embeddings[:, j]
        neighbor_idx = knn(point, embeddings, config.k)
        neighbor_vecs = embeddings[:, neighbor_idx]
        spaces[j] = estimate_tangent_space(point, neighbor_vecs, config.p)
    end

    return spaces
end

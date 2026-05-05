# ── Ranking pipeline ──────────────────────────────────────────────────────────

"""
    rank_candidates(query_embedding, candidate_embeddings, candidate_ids, all_embeddings, config) -> Vector{RankingEntry}

Full pipeline: estimate tangent spaces from embeddings, compute Grassmann distances, return ranked results.
`all_embeddings` (ambient_dim, N) is the full library used for kNN neighborhood lookup.
`candidate_embeddings` (ambient_dim, n) is the subset to rank.
"""
function rank_candidates(
    query_embedding::AbstractVector,
    candidate_embeddings::AbstractMatrix,
    candidate_ids::AbstractVector{<:AbstractString},
    all_embeddings::AbstractMatrix,
    config::GrassmannConfig
)
    # Estimate query tangent space
    query_neighbors_idx = knn(query_embedding, all_embeddings, config.k)
    query_neighbors = all_embeddings[:, query_neighbors_idx]
    query_ts = estimate_tangent_space(query_embedding, query_neighbors, config.p)

    # Estimate candidate tangent spaces
    n = size(candidate_embeddings, 2)
    candidate_ts = Vector{TangentSpace}(undef, n)
    for j in 1:n
        point = @view candidate_embeddings[:, j]
        neighbor_idx = knn(point, all_embeddings, config.k)
        neighbor_vecs = all_embeddings[:, neighbor_idx]
        candidate_ts[j] = estimate_tangent_space(point, neighbor_vecs, config.p)
    end

    return rank_candidates(query_ts, candidate_ts, candidate_ids, config)
end

"""
    rank_candidates(query_ts, candidate_ts, candidate_ids, config) -> Vector{RankingEntry}

Precomputed variant: tangent spaces already available, just compute distances and rank.
"""
function rank_candidates(
    query_ts::TangentSpace,
    candidate_ts::AbstractVector{TangentSpace},
    candidate_ids::AbstractVector{<:AbstractString},
    config::GrassmannConfig
)
    n = length(candidate_ts)
    length(candidate_ids) == n || throw(DimensionMismatch(
        "candidate_ts length ($n) ≠ candidate_ids length ($(length(candidate_ids)))"))

    entries = Vector{RankingEntry}(undef, n)
    for j in 1:n
        d = grassmann_distance(query_ts, candidate_ts[j]; distance=config.distance)
        entries[j] = RankingEntry(candidate_ids[j], d)
    end

    sort!(entries, by=e -> e.distance)
    return entries
end

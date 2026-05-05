# ── Brute-force k-nearest neighbors ──────────────────────────────────────────

"""
    knn(query, candidates, k) -> Vector{Int}

Find k nearest neighbors of `query` in `candidates` by Euclidean distance.
`candidates` is (ambient_dim, n) column-major. Returns column indices.
Excludes the query itself if present (distance < 1e-12).
"""
function knn(query::AbstractVector, candidates::AbstractMatrix, k::Int)
    n = size(candidates, 2)
    k > 0 || throw(ArgumentError("k must be positive, got $k"))

    dists = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        d = 0.0
        @simd for i in eachindex(query)
            diff = query[i] - candidates[i, j]
            d += diff * diff
        end
        dists[j] = d
    end

    # Sort indices by distance, skip self-matches
    order = sortperm(dists)
    result = Vector{Int}()
    sizehint!(result, k)
    for idx in order
        dists[idx] < 1e-24 && continue  # skip self (squared distance threshold)
        push!(result, idx)
        length(result) == k && break
    end

    return result
end

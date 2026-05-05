# ── Configuration ─────────────────────────────────────────────────────────────

struct GrassmannConfig
    k::Int              # neighbors for local PCA
    p::Int              # principal components / tangent space dimension
    distance::Symbol    # :chordal or :geodesic
end

const DEFAULT_CONFIG = GrassmannConfig(20, 2, :geodesic)

# ── Tangent space representation ─────────────────────────────────────────────

struct TangentSpace
    basis::Matrix{Float64}   # (ambient_dim, p) — orthonormal columns
    center::Vector{Float64}  # neighborhood centroid
end

# ── Ranking result ───────────────────────────────────────────────────────────

struct RankingEntry
    id::String
    distance::Float64
end

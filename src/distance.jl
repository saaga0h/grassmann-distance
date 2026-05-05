# ── Grassmann distance via principal angles ──────────────────────────────────

"""
    principal_angles(U, V) -> Vector{Float64}

Compute principal angles between subspaces spanned by columns of U and V.
Both matrices must have the same number of columns (subspace dimension p).
"""
function principal_angles(U::AbstractMatrix, V::AbstractMatrix)
    size(U, 2) == size(V, 2) || throw(DimensionMismatch(
        "subspace dimensions must match: $(size(U,2)) ≠ $(size(V,2))"))

    M = U' * V
    F = svd(M)
    σ = clamp.(F.S, 0.0, 1.0)
    return acos.(σ)
end

"""
    grassmann_distance(ts1, ts2; distance=:geodesic) -> Float64

Grassmann distance between two tangent spaces.
- `:geodesic` — √(Σθᵢ²), the geodesic distance on the Grassmannian
- `:chordal`  — √(Σsin²θᵢ), the chordal distance
"""
function grassmann_distance(
    ts1::TangentSpace, ts2::TangentSpace;
    distance::Symbol=:geodesic
)
    θ = principal_angles(ts1.basis, ts2.basis)

    if distance === :geodesic
        return sqrt(sum(θ .^ 2))
    elseif distance === :chordal
        return sqrt(sum(sin.(θ) .^ 2))
    else
        throw(ArgumentError("unknown distance variant: $distance (use :geodesic or :chordal)"))
    end
end

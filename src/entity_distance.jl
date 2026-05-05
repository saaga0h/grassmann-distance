# ── Chunk-averaged Grassmann distance between entities ───────────────────────

"""
    entity_distance(tangent_spaces, entity_a, entity_b, config; max_chunks=5) -> Float64

Grassmann distance between two entities, averaged over representative chunk pairs.
When an entity has more than `max_chunks` chunks, evenly-spaced samples are used.
"""
function entity_distance(
    tangent_spaces::AbstractVector{TangentSpace},
    entity_a::Entity,
    entity_b::Entity,
    config::GrassmannConfig;
    max_chunks::Int=5
)
    idx_a = _representative_chunks(entity_a.chunk_indices, max_chunks)
    idx_b = _representative_chunks(entity_b.chunk_indices, max_chunks)

    total = 0.0
    count = 0
    for ci in idx_a, cj in idx_b
        total += grassmann_distance(tangent_spaces[ci], tangent_spaces[cj]; distance=config.distance)
        count += 1
    end

    return total / count
end

"""
    _representative_chunks(indices, max_chunks) -> Vector{Int}

Select up to `max_chunks` evenly-spaced indices from a range.
"""
function _representative_chunks(indices::UnitRange{Int}, max_chunks::Int)
    n = length(indices)
    n <= max_chunks && return collect(indices)
    positions = round.(Int, range(1, n; length=max_chunks))
    return [indices[p] for p in positions]
end

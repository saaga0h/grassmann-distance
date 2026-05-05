# ── Graph construction ───────────────────────────────────────────────────────

"""
    build_graph(embeddings, entity_ids, chunk_entity_map, grassmann_config, graph_config) -> GrassmannGraph

Build a GrassmannGraph from raw embeddings.

Arguments:
- `embeddings`: (ambient_dim, n_chunks) matrix
- `entity_ids`: ordered unique entity names
- `chunk_entity_map`: length-n_chunks vector mapping each chunk column to its entity ID
- `grassmann_config`: GrassmannConfig for tangent space estimation and distance
- `graph_config`: GraphConfig for k-NN graph and chunk sampling
"""
function build_graph(
    embeddings::AbstractMatrix{<:Real},
    entity_ids::AbstractVector{<:AbstractString},
    chunk_entity_map::AbstractVector{<:AbstractString},
    grassmann_config::GrassmannConfig=DEFAULT_CONFIG,
    graph_config::GraphConfig=DEFAULT_GRAPH_CONFIG
)
    n_chunks = size(embeddings, 2)
    length(chunk_entity_map) == n_chunks || throw(DimensionMismatch(
        "chunk_entity_map length ($(length(chunk_entity_map))) ≠ embedding columns ($n_chunks)"))

    # Build entities from contiguous chunk ranges
    entities = _build_entities(entity_ids, chunk_entity_map)
    entity_index = Dict(e.id => i for (i, e) in enumerate(entities))
    n_entities = length(entities)

    # Estimate tangent spaces for all chunks
    ts = estimate_tangent_spaces(embeddings, grassmann_config)

    # Compute entity-to-entity distance matrix
    dist_matrix = Matrix{Float64}(undef, n_entities, n_entities)
    for i in 1:n_entities
        dist_matrix[i, i] = 0.0
        for j in (i+1):n_entities
            d = entity_distance(ts, entities[i], entities[j], grassmann_config;
                                max_chunks=graph_config.max_chunks)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
        end
    end

    # Build k-NN adjacency
    k = min(graph_config.k_graph, n_entities - 1)
    adj = _build_adjacency(dist_matrix, k)

    return GrassmannGraph(
        entities, entity_index,
        Matrix{Float64}(embeddings), ts, dist_matrix, adj,
        grassmann_config, graph_config
    )
end

"""
    _build_entities(entity_ids, chunk_entity_map) -> Vector{Entity}

Map ordered entity IDs to contiguous chunk ranges. Validates that each entity's
chunks form a contiguous block in the embedding matrix.
"""
function _build_entities(
    entity_ids::AbstractVector{<:AbstractString},
    chunk_entity_map::AbstractVector{<:AbstractString}
)
    entities = Entity[]
    for eid in entity_ids
        indices = findall(==(eid), chunk_entity_map)
        isempty(indices) && throw(ArgumentError("entity '$eid' has no chunks in chunk_entity_map"))
        first_idx = first(indices)
        last_idx = last(indices)
        expected = first_idx:last_idx
        indices == collect(expected) || throw(ArgumentError(
            "entity '$eid' chunks are not contiguous: found $indices, expected $expected"))
        push!(entities, Entity(eid, expected))
    end
    return entities
end

"""
    _build_adjacency(dist_matrix, k) -> Vector{Vector{Tuple{Int, Float64}}}

Build k-NN adjacency list from a distance matrix. Each entry is sorted by distance.
"""
function _build_adjacency(dist_matrix::Matrix{Float64}, k::Int)
    n = size(dist_matrix, 1)
    adj = Vector{Vector{Tuple{Int, Float64}}}(undef, n)
    for i in 1:n
        scored = [(dist_matrix[i, j], j) for j in 1:n if j != i]
        sort!(scored, by=first)
        adj[i] = [(idx, dist) for (dist, idx) in scored[1:min(k, length(scored))]]
    end
    return adj
end

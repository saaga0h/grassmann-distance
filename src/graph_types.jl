# ── Graph configuration ──────────────────────────────────────────────────────

struct GraphConfig
    k_graph::Int    # neighbors per node in k-NN graph (default: 3)
    max_chunks::Int # max representative chunks per entity for distance averaging (default: 5)
end

GraphConfig(; k_graph::Int=3, max_chunks::Int=5) = GraphConfig(k_graph, max_chunks)

const DEFAULT_GRAPH_CONFIG = GraphConfig()

# ── Entity: a named group of contiguous chunks ─────────────────────────────

struct Entity
    id::String
    chunk_indices::UnitRange{Int}
end

# ── Graph structure ─────────────────────────────────────────────────────────

struct GrassmannGraph
    entities::Vector{Entity}
    entity_index::Dict{String, Int}                  # id → position in entities
    embeddings::Matrix{Float64}                      # (ambient_dim, n_chunks)
    tangent_spaces::Vector{TangentSpace}             # one per chunk
    distance_matrix::Matrix{Float64}                 # (n_entities, n_entities)
    neighbors::Vector{Vector{Tuple{Int, Float64}}}   # k-NN adjacency list per entity
    grassmann_config::GrassmannConfig
    graph_config::GraphConfig
end

# ── Path result ─────────────────────────────────────────────────────────────

struct ConceptualPath
    nodes::Vector{String}
    distances::Vector{Float64}  # hop distances; length = length(nodes) - 1
    total_distance::Float64
end

# ── Topology result types ───────────────────────────────────────────────────

struct Community
    members::Vector{String}
    root::String
end

struct Basin
    attractor::String
    members::Vector{String}
end

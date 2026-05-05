# ── FORGE job contract types ─────────────────────────────────────────────────
# JSON-serializable types for the MQTT request/response protocol.
# Entity IDs flow through untouched — client names in, client names out.

# ── Input types ──────────────────────────────────────────────────────────────

struct EntityInput
    id::String
    embeddings::Vector{Vector{Float64}}
end

struct GraphConfigInput
    k::Int
    p::Int
    k_graph::Int
    max_chunks::Int
    distance::String   # "geodesic" or "chordal"
end

const DEFAULT_CONFIG_INPUT = GraphConfigInput(20, 2, 3, 5, "geodesic")

struct BuildParams
    entities::Vector{EntityInput}
    config::GraphConfigInput
end

struct QuerySpec
    type::String                          # "greedy_path", "shortest_path", "reachable", "communities", "basins", "topology"
    from::Union{String, Nothing}          # required for path/reachable queries
    to::Union{String, Nothing}            # optional target for directed paths
    depth::Union{Int, Nothing}            # greedy path depth / max_hops for reachable
end

struct QueryParams
    graph::String                          # base64-encoded serialized GrassmannGraph
    query::QuerySpec
end

struct JobParams
    job_id::String
    mode::String                           # "build" or "query"
    build::Union{BuildParams, Nothing}
    query::Union{QueryParams, Nothing}
end

# ── Output types ─────────────────────────────────────────────────────────────

struct PathOutput
    nodes::Vector{String}
    distances::Vector{Float64}
    total_distance::Float64
end

struct ReachableEntry
    id::String
    hops::Int
    distance::Float64
end

struct CommunityOutput
    members::Vector{String}
    root::String
end

struct BasinOutput
    attractor::String
    members::Vector{String}
end

struct BridgeOutput
    id::String
    home_community::Int
    reached_communities::Vector{Int}
end

struct HubEntry
    id::String
    in_degree::Int
end

struct TopologyOutput
    communities::Vector{CommunityOutput}
    basins::Vector{BasinOutput}
    bridges::Vector{BridgeOutput}
    hub_centrality::Vector{HubEntry}
    hub_concentration::Float64
    bidirectional_edges::Vector{Vector{String}}
end

struct BuildOutput
    graph::String                          # base64-encoded serialized GrassmannGraph
    entities::Int
    chunks::Int
end

struct QueryOutput
    path::Union{PathOutput, Nothing}
    reachable::Union{Vector{ReachableEntry}, Nothing}
    topology::Union{TopologyOutput, Nothing}
end

struct JobResult
    job_id::String
    success::Bool
    error::Union{String, Nothing}
    result::Union{BuildOutput, QueryOutput, Nothing}
    worker_id::String
    timestamp::String
end

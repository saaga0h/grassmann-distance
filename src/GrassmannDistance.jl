module GrassmannDistance

using LinearAlgebra
using Statistics
using Serialization
using Base64
using Dates
using JSON3
using StructTypes

# Types first
include("types.jl")
include("graph_types.jl")
include("job_types.jl")
include("config.jl")

# Core logic
include("neighbors.jl")
include("tangent.jl")
include("distance.jl")

# Pipeline
include("ranking.jl")

# Graph layer
include("entity_distance.jl")
include("graph.jl")
include("paths.jl")
include("topology.jl")

# FORGE job layer
include("serialization.jl")
include("app.jl")

# MQTT — only loaded when Mosquitto is available (Singularity runtime)
const HAS_MOSQUITTO = try
    @eval using Mosquitto
    true
catch
    false
end

if HAS_MOSQUITTO
    include("mqtt.jl")
end

export GrassmannConfig, DEFAULT_CONFIG, TangentSpace, RankingEntry,
       knn, estimate_tangent_space, estimate_tangent_spaces,
       principal_angles, grassmann_distance,
       rank_candidates,
       # Graph types
       GraphConfig, DEFAULT_GRAPH_CONFIG, Entity, GrassmannGraph,
       ConceptualPath, Community, Basin,
       # Graph construction
       entity_distance, build_graph,
       # Path finding
       find_greedy_path, find_shortest_path, reachable,
       # Topology analysis
       communities, basins, bridges, hub_centrality, hub_concentration,
       bidirectional_edges,
       # Job types
       EntityInput, GraphConfigInput, BuildParams, QuerySpec, QueryParams,
       JobParams, PathOutput, ReachableEntry, CommunityOutput, BasinOutput,
       BridgeOutput, HubEntry, TopologyOutput, BuildOutput, QueryOutput,
       JobResult, WorkerConfig,
       # Serialization
       parse_job, serialize_result, serialize_graph, deserialize_graph,
       # Job processing
       process_job, load_config

end

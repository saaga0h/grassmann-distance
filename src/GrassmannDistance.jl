module GrassmannDistance

using LinearAlgebra
using Statistics
using Serialization
using Base64
using Dates
using JSON3
using StructTypes
using KernelAbstractions

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

# CPU-only fallback — overridden by gpu.jl when AMDGPU loads
function select_backend(use_gpu::Bool)
    if use_gpu
        @warn "USE_GPU=true but AMDGPU not available, falling back to CPU"
    end
    return CPU()
end

# GPU — only loaded when AMDGPU is available (Singularity runtime on GPU server)
const HAS_AMDGPU = try
    @eval using AMDGPU
    true
catch e
    @warn "AMDGPU not available" reason=sprint(showerror, e)
    false
end

if HAS_AMDGPU
    include("gpu.jl")
end

# MQTT — only loaded when Mosquitto is available (Singularity runtime)
const HAS_MOSQUITTO = try
    @eval using Mosquitto
    true
catch e
    @warn "Mosquitto not available" reason=sprint(showerror, e)
    false
end

if HAS_MOSQUITTO
    include("mqtt.jl")
else
    function julia_main()::Cint
        @error "Mosquitto.jl not available — cannot run MQTT worker"
        return 1
    end
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
       process_job, load_config, select_backend

end

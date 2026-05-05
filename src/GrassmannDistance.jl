module GrassmannDistance

using LinearAlgebra
using Statistics

# Types first
include("types.jl")
include("graph_types.jl")

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
       bidirectional_edges

end

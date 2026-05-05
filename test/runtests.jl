using Test
using GrassmannDistance

@testset "GrassmannDistance" begin
    @testset "Neighbors" begin
        include("test_neighbors.jl")
    end
    @testset "TangentSpace" begin
        include("test_tangent.jl")
    end
    @testset "Distance" begin
        include("test_distance.jl")
    end
    @testset "Ranking" begin
        include("test_ranking.jl")
    end
    @testset "Graph" begin
        include("test_graph.jl")
    end
    @testset "Paths" begin
        include("test_paths.jl")
    end
    @testset "Topology" begin
        include("test_topology.jl")
    end
    @testset "Serialization" begin
        include("test_serialization.jl")
    end
    @testset "App" begin
        include("test_app.jl")
    end
end

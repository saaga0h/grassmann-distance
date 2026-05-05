using LinearAlgebra

@testset "kNN returns correct indices" begin
    # 3D, 5 candidates — query at [0.8, 0, 0]
    candidates = Float64[
        0 1 5 2 10;
        0 0 5 0 10;
        0 0 5 0 10
    ]
    query = [0.8, 0.0, 0.0]
    # distances: col1=0.8, col2=0.2, col3≈7.4, col4=1.2, col5≈16.0

    idx = knn(query, candidates, 2)
    @test idx == [2, 1]

    idx3 = knn(query, candidates, 3)
    @test idx3 == [2, 1, 4]
end

@testset "kNN excludes self-match" begin
    candidates = Float64[
        0 1 2;
        0 0 0;
        0 0 0
    ]
    query = [0.0, 0.0, 0.0]  # exact match with col 1

    idx = knn(query, candidates, 2)
    @test 1 ∉ idx  # self excluded
    @test idx == [2, 3]
end

@testset "kNN errors on k ≤ 0" begin
    candidates = Float64[1 2; 3 4]
    query = [1.0, 3.0]
    @test_throws ArgumentError knn(query, candidates, 0)
    @test_throws ArgumentError knn(query, candidates, -1)
end

@testset "kNN with k > available neighbors" begin
    candidates = Float64[
        0 1 2;
        0 0 0
    ]
    query = [0.0, 0.0]  # exact match with col 1

    # After excluding self, only 2 neighbors available but we asked for 3
    idx = knn(query, candidates, 3)
    @test length(idx) == 2  # returns what's available
end

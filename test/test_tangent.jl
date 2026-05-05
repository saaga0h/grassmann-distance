using LinearAlgebra

@testset "tangent space of points on xy-plane in 3D" begin
    # 10 points scattered on the xy-plane (z=0)
    neighbors = zeros(3, 10)
    for j in 1:10
        neighbors[1, j] = randn()
        neighbors[2, j] = randn()
        # z stays 0
    end

    ts = estimate_tangent_space(zeros(3), neighbors, 2)

    # Basis should span the xy-plane — z component should be ~0
    # Check that basis is orthonormal
    @test ts.basis' * ts.basis ≈ I(2) atol=1e-10

    # The 2D tangent space should have negligible z-component
    # Project e3 onto the tangent space — should be near zero
    e3 = [0.0, 0.0, 1.0]
    proj = ts.basis * (ts.basis' * e3)
    @test norm(proj) < 1e-10
end

@testset "tangent space of points along a line" begin
    # 20 points along the x-axis with small noise
    neighbors = zeros(3, 20)
    for j in 1:20
        neighbors[1, j] = j * 1.0
        neighbors[2, j] = randn() * 1e-10
        neighbors[3, j] = randn() * 1e-10
    end

    ts = estimate_tangent_space(zeros(3), neighbors, 1)

    # First PC should align with x-axis
    @test abs(dot(ts.basis[:, 1], [1.0, 0.0, 0.0])) > 0.999
end

@testset "basis columns are orthonormal" begin
    # Random points in 10D, extract 3 components
    neighbors = randn(10, 20)
    ts = estimate_tangent_space(zeros(10), neighbors, 3)
    @test ts.basis' * ts.basis ≈ I(3) atol=1e-10
end

@testset "errors when k < p" begin
    neighbors = randn(5, 2)  # only 2 neighbors
    @test_throws ArgumentError estimate_tangent_space(zeros(5), neighbors, 3)
end

@testset "batch estimation" begin
    # 30 points in 5D — enough for k=10, p=2
    embeddings = randn(5, 30)
    config = GrassmannConfig(10, 2, :geodesic)

    spaces = estimate_tangent_spaces(embeddings, config)
    @test length(spaces) == 30
    @test all(size(ts.basis) == (5, 2) for ts in spaces)
    @test all(ts.basis' * ts.basis ≈ I(2) for ts in spaces)
end

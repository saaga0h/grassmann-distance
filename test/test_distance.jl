using LinearAlgebra

@testset "identical subspaces → distance = 0" begin
    basis = Matrix(qr(randn(10, 3)).Q[:, 1:3])
    ts1 = TangentSpace(basis, zeros(10))
    ts2 = TangentSpace(copy(basis), zeros(10))

    @test grassmann_distance(ts1, ts2) ≈ 0.0 atol=1e-6
    @test grassmann_distance(ts1, ts2; distance=:chordal) ≈ 0.0 atol=1e-6
end

@testset "orthogonal 1D subspaces in 2D → π/2" begin
    ts1 = TangentSpace(reshape([1.0, 0.0], 2, 1), zeros(2))
    ts2 = TangentSpace(reshape([0.0, 1.0], 2, 1), zeros(2))

    @test grassmann_distance(ts1, ts2) ≈ π/2 atol=1e-12
    @test grassmann_distance(ts1, ts2; distance=:chordal) ≈ 1.0 atol=1e-12
end

@testset "known rotation angle between 1D subspaces" begin
    for θ_true in [0.1, 0.5, 1.0, π/4, π/3]
        u = [cos(0.0), sin(0.0)]
        v = [cos(θ_true), sin(θ_true)]
        ts1 = TangentSpace(reshape(u, 2, 1), zeros(2))
        ts2 = TangentSpace(reshape(v, 2, 1), zeros(2))

        @test grassmann_distance(ts1, ts2) ≈ θ_true atol=1e-10
    end
end

@testset "symmetry" begin
    basis1 = Matrix(qr(randn(10, 2)).Q[:, 1:2])
    basis2 = Matrix(qr(randn(10, 2)).Q[:, 1:2])
    ts1 = TangentSpace(basis1, zeros(10))
    ts2 = TangentSpace(basis2, zeros(10))

    @test grassmann_distance(ts1, ts2) ≈ grassmann_distance(ts2, ts1) atol=1e-12
    @test grassmann_distance(ts1, ts2; distance=:chordal) ≈
          grassmann_distance(ts2, ts1; distance=:chordal) atol=1e-12
end

@testset "triangle inequality" begin
    basis1 = Matrix(qr(randn(10, 2)).Q[:, 1:2])
    basis2 = Matrix(qr(randn(10, 2)).Q[:, 1:2])
    basis3 = Matrix(qr(randn(10, 2)).Q[:, 1:2])
    ts1 = TangentSpace(basis1, zeros(10))
    ts2 = TangentSpace(basis2, zeros(10))
    ts3 = TangentSpace(basis3, zeros(10))

    d12 = grassmann_distance(ts1, ts2)
    d23 = grassmann_distance(ts2, ts3)
    d13 = grassmann_distance(ts1, ts3)

    @test d13 ≤ d12 + d23 + 1e-10
end

@testset "chordal and geodesic give consistent ordering (p=1)" begin
    # For p=1, sin is monotone on [0, π/2] so ordering is preserved
    basis_ref = Matrix(qr(randn(10, 1)).Q[:, 1:1])
    ts_ref = TangentSpace(basis_ref, zeros(10))

    others = [TangentSpace(Matrix(qr(randn(10, 1)).Q[:, 1:1]), zeros(10)) for _ in 1:20]

    geo = [grassmann_distance(ts_ref, o) for o in others]
    cho = [grassmann_distance(ts_ref, o; distance=:chordal) for o in others]

    @test sortperm(geo) == sortperm(cho)
end

@testset "principal angles dimensions must match" begin
    U = randn(10, 2)
    V = randn(10, 3)
    @test_throws DimensionMismatch principal_angles(U, V)
end

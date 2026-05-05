using LinearAlgebra

@testset "candidates ranked by distance" begin
    # Construct tangent spaces at known distances from query
    query_basis = reshape([1.0, 0.0, 0.0], 3, 1)
    query_ts = TangentSpace(query_basis, zeros(3))

    # Candidate A: angle π/6 from query
    a_basis = reshape([cos(π/6), sin(π/6), 0.0], 3, 1)
    # Candidate B: angle π/3 from query
    b_basis = reshape([cos(π/3), sin(π/3), 0.0], 3, 1)
    # Candidate C: angle π/2 from query (orthogonal)
    c_basis = reshape([0.0, 1.0, 0.0], 3, 1)

    candidates = [
        TangentSpace(c_basis, zeros(3)),
        TangentSpace(a_basis, zeros(3)),
        TangentSpace(b_basis, zeros(3)),
    ]
    ids = ["c", "a", "b"]

    result = rank_candidates(query_ts, candidates, ids, DEFAULT_CONFIG)

    @test result[1].id == "a"
    @test result[2].id == "b"
    @test result[3].id == "c"
    @test result[1].distance < result[2].distance < result[3].distance
end

@testset "precomputed matches full pipeline" begin
    # Small dataset in 5D
    n = 40
    embeddings = randn(5, n)
    ids = ["entity_$i" for i in 1:n]
    config = GrassmannConfig(10, 2, :geodesic)

    query = @view embeddings[:, 1]
    candidate_idx = 2:n
    candidate_emb = embeddings[:, candidate_idx]
    candidate_ids = ids[candidate_idx]

    # Full pipeline
    result_full = rank_candidates(query, candidate_emb, candidate_ids, embeddings, config)

    # Precomputed
    all_ts = estimate_tangent_spaces(embeddings, config)
    query_ts = all_ts[1]
    candidate_ts_vec = all_ts[candidate_idx]

    result_pre = rank_candidates(query_ts, candidate_ts_vec, candidate_ids, config)

    # Same ranking order
    @test [r.id for r in result_full] == [r.id for r in result_pre]
    # Same distances
    @test [r.distance for r in result_full] ≈ [r.distance for r in result_pre] atol=1e-10
end

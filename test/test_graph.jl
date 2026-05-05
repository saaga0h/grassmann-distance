using LinearAlgebra
using Statistics
using Random

# ── Test helpers ─────────────────────────────────────────────────────────────

"""Build synthetic embeddings: n_entities entities, each with n_chunks_per chunks.
Entities are centered at different directions in ambient space, with small noise per chunk."""
function make_test_embeddings(;
    n_entities=5, n_chunks_per=4, dim=20, spread=0.1, rng_seed=42
)
    rng = MersenneTwister(rng_seed)
    total_chunks = n_entities * n_chunks_per
    embeddings = Matrix{Float64}(undef, dim, total_chunks)
    entity_ids = ["E$i" for i in 1:n_entities]
    chunk_entity_map = String[]

    for (ei, eid) in enumerate(entity_ids)
        center = randn(rng, dim)
        center ./= norm(center)
        for ci in 1:n_chunks_per
            col = (ei - 1) * n_chunks_per + ci
            v = center .+ spread .* randn(rng, dim)
            embeddings[:, col] = v ./ norm(v)
            push!(chunk_entity_map, eid)
        end
    end

    return embeddings, entity_ids, chunk_entity_map
end

# ── Entity distance ─────────────────────────────────────────────────────────

@testset "entity_distance" begin
    @testset "self-distance is near zero" begin
        embeddings, entity_ids, chunk_map = make_test_embeddings(n_entities=3, n_chunks_per=4, dim=20)
        config = GrassmannConfig(8, 2, :geodesic)
        ts = estimate_tangent_spaces(embeddings, config)

        e1 = Entity("E1", 1:4)
        d = entity_distance(ts, e1, e1, config)
        # Self-distance won't be exactly 0 because different chunks have different tangent spaces,
        # but it should be small relative to between-entity distances
        @test d < 0.5
    end

    @testset "symmetry" begin
        embeddings, entity_ids, chunk_map = make_test_embeddings(n_entities=3, n_chunks_per=4, dim=20)
        config = GrassmannConfig(8, 2, :geodesic)
        ts = estimate_tangent_spaces(embeddings, config)

        e1 = Entity("E1", 1:4)
        e2 = Entity("E2", 5:8)
        @test entity_distance(ts, e1, e2, config) ≈ entity_distance(ts, e2, e1, config) atol=1e-10
    end

    @testset "max_chunks sampling" begin
        embeddings, entity_ids, chunk_map = make_test_embeddings(n_entities=2, n_chunks_per=10, dim=20)
        config = GrassmannConfig(8, 2, :geodesic)
        ts = estimate_tangent_spaces(embeddings, config)

        e1 = Entity("E1", 1:10)
        e2 = Entity("E2", 11:20)
        # With max_chunks=2, uses fewer pairs than max_chunks=10
        d_few = entity_distance(ts, e1, e2, config; max_chunks=2)
        d_all = entity_distance(ts, e1, e2, config; max_chunks=10)
        # Both should be positive finite distances
        @test d_few > 0
        @test d_all > 0
        @test isfinite(d_few)
        @test isfinite(d_all)
    end
end

# ── Graph construction ───────────────────────────────────────────────────────

@testset "build_graph" begin
    @testset "basic construction" begin
        embeddings, entity_ids, chunk_map = make_test_embeddings(n_entities=5, n_chunks_per=4, dim=20)
        config = GrassmannConfig(8, 2, :geodesic)
        gc = GraphConfig(k_graph=2, max_chunks=4)

        graph = build_graph(embeddings, entity_ids, chunk_map, config, gc)

        @test length(graph.entities) == 5
        @test size(graph.distance_matrix) == (5, 5)
        @test length(graph.neighbors) == 5
        @test length(graph.tangent_spaces) == 20
        @test size(graph.embeddings) == (20, 20)
    end

    @testset "distance matrix is symmetric with zero diagonal" begin
        embeddings, entity_ids, chunk_map = make_test_embeddings(n_entities=4, n_chunks_per=3, dim=15)
        config = GrassmannConfig(6, 2, :geodesic)
        gc = GraphConfig(k_graph=2, max_chunks=3)

        graph = build_graph(embeddings, entity_ids, chunk_map, config, gc)

        for i in 1:4
            @test graph.distance_matrix[i, i] ≈ 0.0
            for j in (i+1):4
                @test graph.distance_matrix[i, j] ≈ graph.distance_matrix[j, i] atol=1e-10
            end
        end
    end

    @testset "each entity has k_graph neighbors" begin
        embeddings, entity_ids, chunk_map = make_test_embeddings(n_entities=6, n_chunks_per=3, dim=15)
        config = GrassmannConfig(6, 2, :geodesic)
        gc = GraphConfig(k_graph=3, max_chunks=3)

        graph = build_graph(embeddings, entity_ids, chunk_map, config, gc)

        for adj in graph.neighbors
            @test length(adj) == 3
        end
    end

    @testset "entity_index lookup works" begin
        embeddings, entity_ids, chunk_map = make_test_embeddings(n_entities=4, n_chunks_per=3, dim=15)
        graph = build_graph(embeddings, entity_ids, chunk_map)

        for (i, eid) in enumerate(entity_ids)
            @test graph.entity_index[eid] == i
        end
    end

    @testset "non-contiguous chunks error" begin
        embeddings = randn(10, 6)
        entity_ids = ["A", "B"]
        chunk_map = ["A", "B", "A", "B", "A", "B"]  # interleaved, not contiguous

        @test_throws ArgumentError build_graph(embeddings, entity_ids, chunk_map)
    end

    @testset "chunk count mismatch error" begin
        embeddings = randn(10, 5)
        chunk_map = ["A", "A", "B"]  # wrong length

        @test_throws DimensionMismatch build_graph(embeddings, ["A", "B"], chunk_map)
    end
end

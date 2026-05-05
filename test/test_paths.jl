using LinearAlgebra
using Random

# ── Shared test graph builder ────────────────────────────────────────────────

function build_test_graph(; n_entities=6, n_chunks_per=3, dim=15, k_graph=2, seed=42)
    rng = MersenneTwister(seed)
    total = n_entities * n_chunks_per
    embeddings = Matrix{Float64}(undef, dim, total)
    entity_ids = ["N$i" for i in 1:n_entities]
    chunk_map = String[]

    for (ei, eid) in enumerate(entity_ids)
        center = randn(rng, dim)
        center ./= norm(center)
        for ci in 1:n_chunks_per
            col = (ei - 1) * n_chunks_per + ci
            v = center .+ 0.1 .* randn(rng, dim)
            embeddings[:, col] = v ./ norm(v)
            push!(chunk_map, eid)
        end
    end

    config = GrassmannConfig(min(total - 1, 8), 2, :geodesic)
    gc = GraphConfig(k_graph=k_graph, max_chunks=n_chunks_per)
    return build_graph(embeddings, entity_ids, chunk_map, config, gc)
end

# ── Greedy path (open-ended) ─────────────────────────────────────────────────

@testset "find_greedy_path open-ended" begin
    graph = build_test_graph()

    @testset "starts at source" begin
        path = find_greedy_path(graph, "N1")
        @test path.nodes[1] == "N1"
    end

    @testset "respects depth" begin
        for d in [1, 2, 4]
            path = find_greedy_path(graph, "N1"; depth=d)
            @test length(path.nodes) <= d + 1
        end
    end

    @testset "no repeated nodes" begin
        path = find_greedy_path(graph, "N1"; depth=5)
        @test length(unique(path.nodes)) == length(path.nodes)
    end

    @testset "distances match nodes" begin
        path = find_greedy_path(graph, "N1"; depth=4)
        @test length(path.distances) == length(path.nodes) - 1
        @test path.total_distance ≈ sum(path.distances) atol=1e-10
    end

    @testset "all distances are positive" begin
        path = find_greedy_path(graph, "N2"; depth=5)
        @test all(d -> d > 0, path.distances)
    end
end

# ── Greedy path (targeted) ──────────────────────────────────────────────────

@testset "find_greedy_path targeted" begin
    graph = build_test_graph()

    @testset "self-path" begin
        path = find_greedy_path(graph, "N1", "N1")
        @test path.nodes == ["N1"]
        @test isempty(path.distances)
        @test path.total_distance == 0.0
    end

    @testset "reaches target or returns nothing" begin
        path = find_greedy_path(graph, "N1", "N3"; max_depth=10)
        if path !== nothing
            @test path.nodes[1] == "N1"
            @test path.nodes[end] == "N3"
            @test length(path.distances) == length(path.nodes) - 1
        end
    end
end

# ── Dijkstra shortest path ──────────────────────────────────────────────────

@testset "find_shortest_path" begin
    graph = build_test_graph(k_graph=3)  # denser graph for connectivity

    @testset "self-path" begin
        path = find_shortest_path(graph, "N1", "N1")
        @test path.nodes == ["N1"]
        @test path.total_distance == 0.0
    end

    @testset "path properties" begin
        path = find_shortest_path(graph, "N1", "N6")
        if path !== nothing
            @test path.nodes[1] == "N1"
            @test path.nodes[end] == "N6"
            @test length(path.distances) == length(path.nodes) - 1
            @test path.total_distance > 0
            @test all(d -> d > 0, path.distances)
        end
    end

    @testset "symmetry" begin
        p_ab = find_shortest_path(graph, "N1", "N4")
        p_ba = find_shortest_path(graph, "N4", "N1")
        # Directed graph — paths may differ, but if both exist they should be valid
        if p_ab !== nothing
            @test p_ab.nodes[1] == "N1"
            @test p_ab.nodes[end] == "N4"
        end
        if p_ba !== nothing
            @test p_ba.nodes[1] == "N4"
            @test p_ba.nodes[end] == "N1"
        end
    end
end

# ── Reachable ────────────────────────────────────────────────────────────────

@testset "reachable" begin
    graph = build_test_graph(k_graph=3)

    @testset "does not include source" begin
        r = reachable(graph, "N1"; max_hops=3)
        @test !any(x -> x[1] == "N1", r)
    end

    @testset "more hops → more or equal reachable" begin
        r1 = reachable(graph, "N1"; max_hops=1)
        r2 = reachable(graph, "N1"; max_hops=2)
        r3 = reachable(graph, "N1"; max_hops=3)
        @test length(r1) <= length(r2) <= length(r3)
    end

    @testset "hop count is correct" begin
        r = reachable(graph, "N1"; max_hops=1)
        for (_, hops, _) in r
            @test hops == 1
        end
    end

    @testset "sorted by distance" begin
        r = reachable(graph, "N1"; max_hops=3)
        dists = [x[3] for x in r]
        @test issorted(dists)
    end
end

# ── Error handling ───────────────────────────────────────────────────────────

@testset "path errors" begin
    graph = build_test_graph()

    @test_throws ArgumentError find_greedy_path(graph, "NONEXISTENT")
    @test_throws ArgumentError find_shortest_path(graph, "N1", "NONEXISTENT")
    @test_throws ArgumentError reachable(graph, "NONEXISTENT")
end

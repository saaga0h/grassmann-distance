using LinearAlgebra
using Random

# ── Build a graph with known cluster structure ──────────────────────────────

function build_clustered_graph(; seed=42)
    rng = MersenneTwister(seed)
    dim = 20
    n_chunks_per = 3

    # Two tight clusters + one outlier
    # Cluster A: N1, N2, N3 — nearby in embedding space
    # Cluster B: N4, N5, N6 — nearby but far from A
    # Outlier: N7 — somewhere in between
    cluster_a_center = randn(rng, dim); cluster_a_center ./= norm(cluster_a_center)
    cluster_b_center = -cluster_a_center  # opposite direction
    outlier_center = randn(rng, dim); outlier_center ./= norm(outlier_center)

    entity_ids = ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]
    centers = [cluster_a_center, cluster_a_center, cluster_a_center,
               cluster_b_center, cluster_b_center, cluster_b_center,
               outlier_center]

    total = length(entity_ids) * n_chunks_per
    embeddings = Matrix{Float64}(undef, dim, total)
    chunk_map = String[]

    for (ei, eid) in enumerate(entity_ids)
        c = centers[ei]
        for ci in 1:n_chunks_per
            col = (ei - 1) * n_chunks_per + ci
            v = c .+ 0.05 .* randn(rng, dim)
            embeddings[:, col] = v ./ norm(v)
            push!(chunk_map, eid)
        end
    end

    config = GrassmannConfig(min(total - 1, 10), 2, :geodesic)
    gc = GraphConfig(k_graph=2, max_chunks=n_chunks_per)
    return build_graph(embeddings, entity_ids, chunk_map, config, gc)
end

# ── Bidirectional edges ─────────────────────────────────────────────────────

@testset "bidirectional_edges" begin
    graph = build_clustered_graph()
    bidir = bidirectional_edges(graph)

    @testset "returns sorted pairs" begin
        for (a, b) in bidir
            @test a < b  # lexicographic order
        end
        @test issorted(bidir)
    end

    @testset "no self-edges" begin
        for (a, b) in bidir
            @test a != b
        end
    end
end

# ── Communities ──────────────────────────────────────────────────────────────

@testset "communities" begin
    graph = build_clustered_graph()
    comms = communities(graph)

    @testset "every entity appears exactly once" begin
        all_members = reduce(vcat, [c.members for c in comms])
        @test sort(all_members) == sort([e.id for e in graph.entities])
    end

    @testset "sorted by size descending" begin
        sizes = [length(c.members) for c in comms]
        @test issorted(sizes, rev=true)
    end

    @testset "members are sorted" begin
        for c in comms
            @test issorted(c.members)
        end
    end
end

# ── Basins ──────────────────────────────────────────────────────────────────

@testset "basins" begin
    graph = build_clustered_graph()
    bs = basins(graph)

    @testset "every entity belongs to exactly one basin" begin
        all_members = reduce(vcat, [b.members for b in bs])
        @test sort(all_members) == sort([e.id for e in graph.entities])
    end

    @testset "attractor is a member of its basin" begin
        for b in bs
            @test b.attractor in b.members
        end
    end

    @testset "sorted by size descending" begin
        sizes = [length(b.members) for b in bs]
        @test issorted(sizes, rev=true)
    end
end

# ── Bridges ─────────────────────────────────────────────────────────────────

@testset "bridges" begin
    graph = build_clustered_graph()
    br = bridges(graph)

    @testset "bridge reaches multiple communities" begin
        for (name, home, reached) in br
            @test length(reached) >= 2
            @test home in reached
        end
    end

    @testset "all bridge entities exist in graph" begin
        valid_ids = Set(e.id for e in graph.entities)
        for (name, _, _) in br
            @test name in valid_ids
        end
    end
end

# ── Hub centrality ──────────────────────────────────────────────────────────

@testset "hub_centrality" begin
    graph = build_clustered_graph()
    hc = hub_centrality(graph)

    @testset "covers all entities" begin
        @test length(hc) == length(graph.entities)
    end

    @testset "sorted descending" begin
        degrees = [d for (_, d) in hc]
        @test issorted(degrees, rev=true)
    end

    @testset "total in-degree equals total edges" begin
        total_edges = sum(length(adj) for adj in graph.neighbors)
        total_indeg = sum(d for (_, d) in hc)
        @test total_indeg == total_edges
    end
end

# ── Hub concentration ───────────────────────────────────────────────────────

@testset "hub_concentration" begin
    graph = build_clustered_graph()
    hc = hub_concentration(graph)

    @test 0.0 <= hc <= 1.0
end

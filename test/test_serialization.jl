using Random
using LinearAlgebra
using JSON3

# ── Helper: build a small graph for serialization tests ─────────────────────

function make_small_graph(; seed=42)
    rng = MersenneTwister(seed)
    dim = 15
    n_chunks_per = 3
    entity_ids = ["A", "B", "C"]
    chunk_map = String[]
    total = length(entity_ids) * n_chunks_per
    embeddings = Matrix{Float64}(undef, dim, total)

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

    config = GrassmannConfig(6, 2, :geodesic)
    gc = GraphConfig(k_graph=2, max_chunks=3)
    return build_graph(embeddings, entity_ids, chunk_map, config, gc)
end

# ── Graph round-trip serialization ──────────────────────────────────────────

@testset "serialize_graph / deserialize_graph" begin
    graph = make_small_graph()

    encoded = serialize_graph(graph)
    @test !isempty(encoded)
    @test typeof(encoded) == String

    restored = deserialize_graph(encoded)

    @test length(restored.entities) == length(graph.entities)
    @test [e.id for e in restored.entities] == [e.id for e in graph.entities]
    @test restored.distance_matrix ≈ graph.distance_matrix
    @test length(restored.neighbors) == length(graph.neighbors)
    @test restored.grassmann_config.k == graph.grassmann_config.k
    @test restored.graph_config.k_graph == graph.graph_config.k_graph
end

# ── JSON parse/serialize round-trip ─────────────────────────────────────────

@testset "parse_job build mode" begin
    json = """
    {
        "job_id": "test-123",
        "mode": "build",
        "build": {
            "entities": [
                {"id": "doc1", "embeddings": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]},
                {"id": "doc2", "embeddings": [[7.0, 8.0, 9.0]]}
            ],
            "config": {"k": 2, "p": 1, "k_graph": 1, "max_chunks": 2, "distance": "geodesic"}
        }
    }
    """

    params = parse_job(json)
    @test params.job_id == "test-123"
    @test params.mode == "build"
    @test params.build !== nothing
    @test length(params.build.entities) == 2
    @test params.build.entities[1].id == "doc1"
    @test length(params.build.entities[1].embeddings) == 2
    @test params.build.entities[2].id == "doc2"
    @test length(params.build.entities[2].embeddings) == 1
    @test params.build.config.k == 2
    @test params.build.config.distance == "geodesic"
end

@testset "parse_job query mode" begin
    graph = make_small_graph()
    encoded = serialize_graph(graph)

    json = """
    {
        "job_id": "q-456",
        "mode": "query",
        "query": {
            "graph": "$(encoded)",
            "query": {"type": "greedy_path", "from": "A", "depth": 2}
        }
    }
    """

    params = parse_job(json)
    @test params.job_id == "q-456"
    @test params.mode == "query"
    @test params.query !== nothing
    @test params.query.query.type == "greedy_path"
    @test params.query.query.from == "A"
    @test params.query.query.depth == 2
end

@testset "serialize_result" begin
    result = JobResult("test-1", true, nothing,
        BuildOutput("abc123base64", 3, 9),
        "worker-1", "2026-05-05T12:00:00.000Z")

    bytes = serialize_result(result)
    @test !isempty(bytes)

    # Round-trip through JSON3
    parsed = JSON3.read(String(bytes))
    @test parsed.job_id == "test-1"
    @test parsed.success == true
    @test parsed.result.entities == 3
    @test parsed.result.chunks == 9
end

# ── Config conversion ───────────────────────────────────────────────────────

@testset "_to_grassmann_config" begin
    input = GraphConfigInput(15, 3, 4, 6, "chordal")
    gc, graph_c = GrassmannDistance._to_grassmann_config(input)

    @test gc.k == 15
    @test gc.p == 3
    @test gc.distance == :chordal

    @test graph_c.k_graph == 4
    @test graph_c.max_chunks == 6
end

@testset "_to_grassmann_config geodesic default" begin
    input = GraphConfigInput(10, 2, 3, 5, "geodesic")
    gc, _ = GrassmannDistance._to_grassmann_config(input)
    @test gc.distance == :geodesic
end

using Random
using LinearAlgebra
using JSON3

# ── Helper ──────────────────────────────────────────────────────────────────

function make_build_payload(; n_entities=3, n_chunks_per=3, dim=15, seed=42)
    rng = MersenneTwister(seed)
    entities = []
    for i in 1:n_entities
        center = randn(rng, dim)
        center ./= norm(center)
        chunks = [Vector{Float64}((center .+ 0.1 .* randn(rng, dim)) |> v -> v ./ norm(v))
                  for _ in 1:n_chunks_per]
        push!(entities, Dict("id" => "E$i", "embeddings" => chunks))
    end

    payload = Dict(
        "job_id" => "build-test",
        "mode" => "build",
        "build" => Dict(
            "entities" => entities,
            "config" => Dict("k" => 6, "p" => 2, "k_graph" => 2, "max_chunks" => 3, "distance" => "geodesic")
        )
    )
    return Vector{UInt8}(JSON3.write(payload))
end

noop_log(level, msg) = nothing
collect_log(logs) = (level, msg) -> push!(logs, (level, msg))

# ── Build mode ───────────────────────────────────────────────────────────────

@testset "process_job build" begin
    payload = make_build_payload()
    result = process_job(payload, "test-worker", noop_log)

    @test result.success == true
    @test result.job_id == "build-test"
    @test result.worker_id == "test-worker"
    @test result.error === nothing

    build_out = result.result::BuildOutput
    @test build_out.entities == 3
    @test build_out.chunks == 9
    @test !isempty(build_out.graph)

    # Verify the graph blob is valid
    graph = deserialize_graph(build_out.graph)
    @test length(graph.entities) == 3
end

# ── Query mode: greedy path ─────────────────────────────────────────────────

@testset "process_job query greedy_path" begin
    # First build a graph
    build_payload = make_build_payload()
    build_result = process_job(build_payload, "w", noop_log)
    graph_blob = build_result.result.graph

    query_payload = Vector{UInt8}(JSON3.write(Dict(
        "job_id" => "query-1",
        "mode" => "query",
        "query" => Dict(
            "graph" => graph_blob,
            "query" => Dict("type" => "greedy_path", "from" => "E1", "depth" => 2)
        )
    )))

    result = process_job(query_payload, "w", noop_log)
    @test result.success == true
    @test result.job_id == "query-1"

    qout = result.result::QueryOutput
    @test qout.path !== nothing
    @test qout.path.nodes[1] == "E1"
    @test length(qout.path.nodes) <= 3
    @test length(qout.path.distances) == length(qout.path.nodes) - 1
end

# ── Query mode: shortest path ───────────────────────────────────────────────

@testset "process_job query shortest_path" begin
    build_result = process_job(make_build_payload(), "w", noop_log)
    graph_blob = build_result.result.graph

    query_payload = Vector{UInt8}(JSON3.write(Dict(
        "job_id" => "query-2",
        "mode" => "query",
        "query" => Dict(
            "graph" => graph_blob,
            "query" => Dict("type" => "shortest_path", "from" => "E1", "to" => "E3")
        )
    )))

    result = process_job(query_payload, "w", noop_log)
    @test result.success == true

    qout = result.result::QueryOutput
    if qout.path !== nothing
        @test qout.path.nodes[1] == "E1"
        @test qout.path.nodes[end] == "E3"
    end
end

# ── Query mode: reachable ───────────────────────────────────────────────────

@testset "process_job query reachable" begin
    build_result = process_job(make_build_payload(), "w", noop_log)
    graph_blob = build_result.result.graph

    query_payload = Vector{UInt8}(JSON3.write(Dict(
        "job_id" => "query-3",
        "mode" => "query",
        "query" => Dict(
            "graph" => graph_blob,
            "query" => Dict("type" => "reachable", "from" => "E2", "depth" => 2)
        )
    )))

    result = process_job(query_payload, "w", noop_log)
    @test result.success == true

    qout = result.result::QueryOutput
    @test qout.reachable !== nothing
    @test !any(e -> e.id == "E2", qout.reachable)
end

# ── Query mode: full topology ───────────────────────────────────────────────

@testset "process_job query topology" begin
    build_result = process_job(make_build_payload(), "w", noop_log)
    graph_blob = build_result.result.graph

    query_payload = Vector{UInt8}(JSON3.write(Dict(
        "job_id" => "query-4",
        "mode" => "query",
        "query" => Dict(
            "graph" => graph_blob,
            "query" => Dict("type" => "topology")
        )
    )))

    result = process_job(query_payload, "w", noop_log)
    @test result.success == true

    qout = result.result::QueryOutput
    @test qout.topology !== nothing
    @test !isempty(qout.topology.communities)
    @test !isempty(qout.topology.basins)
    @test !isempty(qout.topology.hub_centrality)
    @test 0.0 <= qout.topology.hub_concentration <= 1.0
end

# ── Error handling ──────────────────────────────────────────────────────────

@testset "process_job invalid json" begin
    result = process_job(Vector{UInt8}("not json"), "w", noop_log)
    @test result.success == false
    @test result.error !== nothing
end

@testset "process_job unknown mode" begin
    payload = Vector{UInt8}(JSON3.write(Dict(
        "job_id" => "bad-1",
        "mode" => "explode"
    )))
    result = process_job(payload, "w", noop_log)
    @test result.success == false
    @test occursin("Unknown mode", result.error)
end

@testset "process_job unknown query type" begin
    build_result = process_job(make_build_payload(), "w", noop_log)
    graph_blob = build_result.result.graph

    query_payload = Vector{UInt8}(JSON3.write(Dict(
        "job_id" => "bad-2",
        "mode" => "query",
        "query" => Dict(
            "graph" => graph_blob,
            "query" => Dict("type" => "nonsense", "from" => "E1")
        )
    )))

    result = process_job(query_payload, "w", noop_log)
    @test result.success == false
    @test occursin("unknown query type", result.error)
end

@testset "process_job logging" begin
    logs = Tuple{String, String}[]
    payload = make_build_payload()
    result = process_job(payload, "w", collect_log(logs))

    @test result.success == true
    @test any(l -> l[1] == "info", logs)
    @test any(l -> occursin("Building graph", l[2]), logs)
end

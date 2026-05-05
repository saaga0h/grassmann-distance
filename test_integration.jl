#!/usr/bin/env julia
# Integration test: submit jobs through FORGE via MQTT and verify results.
#
# Flow:
#   1. Client publishes request to compute/request/{client_id}/{correlation_id}
#   2. FORGE reads operation, generates job_id, publishes params to worker topic
#   3. FORGE dispatches Nomad job (grassmann-distance)
#   4. Worker processes, publishes result to compute/jobs/{job_id}/result
#   5. FORGE routes result back to compute/response/{client_id}/{correlation_id}
#
# Requires:
#   - MQTT broker reachable
#   - FORGE running
#   - grassmann-distance job registered in Nomad
#   - .env with MQTT credentials
#
# Usage:
#   julia --project=. test_integration.jl

using GrassmannDistance
using JSON3
using Mosquitto
using Random
using LinearAlgebra

# ── Load .env ────────────────────────────────────────────────────────────────

function load_dotenv(path=joinpath(@__DIR__, ".env"))
    isfile(path) || error("Missing .env file at $path")
    for line in readlines(path)
        line = strip(line)
        isempty(line) && continue
        startswith(line, '#') && continue
        key, val = split(line, '='; limit=2)
        ENV[strip(String(key))] = strip(String(val))
    end
end

load_dotenv()

broker_url = get(ENV, "MQTT_BROKER") do
    error("MQTT_BROKER not set in .env")
end
mqtt_user = get(ENV, "MQTT_USER", "")
mqtt_pass = get(ENV, "MQTT_PASSWORD", "")

# ── Generate test embeddings ─────────────────────────────────────────────────

rng = MersenneTwister(42)
dim = 64
entities = []

for name in ["alpha", "beta", "gamma", "delta"]
    center = randn(rng, dim)
    center ./= norm(center)
    chunks = [let v = center .+ 0.1 .* randn(rng, dim); v ./= norm(v); v end for _ in 1:4]
    push!(entities, Dict("id" => name, "embeddings" => chunks))
end

# ── MQTT helpers ─────────────────────────────────────────────────────────────

const CLIENT_ID = "grassmann-test-$(rand(1000:9999))"

function parse_broker(url::String)
    stripped = replace(url, r"^tcp://" => "")
    parts = split(stripped, ":")
    host = String(parts[1])
    port = length(parts) > 1 ? parse(Int, parts[2]) : 1883
    return (host, port)
end

function wait_for_message(client, topic; timeout_sec=180)
    ch = Mosquitto.get_messages_channel(client)
    deadline = time() + timeout_sec
    while time() < deadline
        Mosquitto.loop(client; timeout=500, ntimes=10)
        while !isempty(ch)
            msg = take!(ch)
            if msg.topic == topic
                return String(msg.payload)
            end
        end
    end
    error("Timeout waiting for message on $topic (waited $(timeout_sec)s)")
end

function submit_and_wait(client, correlation_id, payload; timeout_sec=180)
    # Subscribe to response BEFORE publishing request
    response_topic = "compute/response/$CLIENT_ID/$correlation_id"
    Mosquitto.subscribe(client, response_topic; qos=1)
    Mosquitto.loop(client; timeout=200, ntimes=5)

    # Publish request to FORGE
    request_topic = "compute/request/$CLIENT_ID/$correlation_id"
    Mosquitto.publish(client, request_topic, payload; qos=1)
    Mosquitto.loop(client; timeout=200, ntimes=5)

    println("   Request  → $request_topic")
    println("   Waiting  ← $response_topic (timeout $(timeout_sec)s)...")

    # Wait for FORGE to route the result back
    result_json = wait_for_message(client, response_topic; timeout_sec)
    return JSON3.read(result_json)
end

# ── Connect ──────────────────────────────────────────────────────────────────

host, port = parse_broker(broker_url)
client = Mosquitto.Client(; id=CLIENT_ID)
Mosquitto.connect(client, host, port; username=mqtt_user, password=mqtt_pass, keepalive=30)

println("=== Grassmann Distance Integration Test ===")
println("  Broker:    $broker_url")
println("  Client ID: $CLIENT_ID")
println()

# ── Test 1: Build ────────────────────────────────────────────────────────────

build_payload = JSON3.write(Dict(
    "operation" => "grassmann-distance",
    "payload" => Dict(
        "mode" => "build",
        "build" => Dict(
            "entities" => entities,
            "config" => Dict("k" => 8, "p" => 2, "k_graph" => 2, "max_chunks" => 4, "distance" => "geodesic")
        )
    )
))

println("1. Build job")
build_result = submit_and_wait(client, "build-001", build_payload)

if build_result.success
    println("   ✓ Build succeeded")
    println("     Entities: $(build_result.result.entities)")
    println("     Chunks: $(build_result.result.chunks)")
    println("     Graph blob: $(length(build_result.result.graph)) chars")
else
    println("   ✗ Build failed: $(build_result.error)")
    Mosquitto.disconnect(client)
    exit(1)
end

graph_blob = build_result.result.graph
println()

# ── Test 2: Greedy path query ───────────────────────────────────────────────

query_payload = JSON3.write(Dict(
    "operation" => "grassmann-distance",
    "payload" => Dict(
        "mode" => "query",
        "query" => Dict(
            "graph" => graph_blob,
            "query" => Dict("type" => "greedy_path", "from" => "alpha", "depth" => 3)
        )
    )
))

println("2. Greedy path query")
query_result = submit_and_wait(client, "query-001", query_payload)

if query_result.success
    path = query_result.result.path
    println("   ✓ Path found:")
    print("     alpha")
    for i in 1:length(path.distances)
        d = round(path.distances[i]; digits=3)
        print(" →[$d] $(path.nodes[i+1])")
    end
    println()
    println("     Total distance: $(round(path.total_distance; digits=3))")
else
    println("   ✗ Query failed: $(query_result.error)")
end
println()

# ── Test 3: Topology query ──────────────────────────────────────────────────

topo_payload = JSON3.write(Dict(
    "operation" => "grassmann-distance",
    "payload" => Dict(
        "mode" => "query",
        "query" => Dict(
            "graph" => graph_blob,
            "query" => Dict("type" => "topology")
        )
    )
))

println("3. Topology query")
topo_result = submit_and_wait(client, "topo-001", topo_payload)

if topo_result.success
    topo = topo_result.result.topology
    println("   ✓ Topology:")
    println("     Communities: $(length(topo.communities))")
    for c in topo.communities
        println("       $(join(c.members, ", "))")
    end
    println("     Basins: $(length(topo.basins))")
    for b in topo.basins
        println("       Attractor: $(b.attractor) → $(join(b.members, ", "))")
    end
    println("     Hub concentration: $(round(topo.hub_concentration; digits=3))")
    println("     Bidirectional edges: $(length(topo.bidirectional_edges))")
    for e in topo.bidirectional_edges
        println("       $(e[1]) ↔ $(e[2])")
    end
else
    println("   ✗ Topology failed: $(topo_result.error)")
end

println()
println("=== Integration test complete ===")
Mosquitto.disconnect(client)

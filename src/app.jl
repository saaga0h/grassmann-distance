# ── Job processing ───────────────────────────────────────────────────────────

function process_job(payload::AbstractVector{UInt8}, worker_id::String, log_fn)::JobResult
    timestamp = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")

    params = try
        parse_job(payload)
    catch e
        @error "Failed to parse job parameters" exception=(e, catch_backtrace())
        log_fn("error", "Parse error: $(e)")
        return JobResult("unknown", false, "Parse error: $(e)", nothing, worker_id, timestamp)
    end

    log_fn("info", "Job received: mode=$(params.mode)")

    try
        if params.mode == "build"
            return _process_build(params, worker_id, log_fn, timestamp)
        elseif params.mode == "query"
            return _process_query(params, worker_id, log_fn, timestamp)
        else
            return JobResult(params.job_id, false, "Unknown mode: $(params.mode)",
                             nothing, worker_id, timestamp)
        end
    catch e
        msg = "Computation failed: $(e)"
        @error "Computation failed" exception=(e, catch_backtrace())
        log_fn("error", msg)
        return JobResult(params.job_id, false, msg, nothing, worker_id, timestamp)
    end
end

# ── Build mode ───────────────────────────────────────────────────────────────

function _process_build(params::JobParams, worker_id::String, log_fn, timestamp::String)::JobResult
    build = params.build
    build !== nothing || return JobResult(params.job_id, false, "build mode requires build params",
                                          nothing, worker_id, timestamp)

    n_entities = length(build.entities)
    n_chunks = sum(length(e.embeddings) for e in build.entities)
    log_fn("info", "Building graph: $(n_entities) entities, $(n_chunks) chunks")

    embeddings, entity_ids, chunk_map = _prepare_build_inputs(build.entities)
    gc, graph_config = _to_grassmann_config(build.config)

    graph = build_graph(embeddings, entity_ids, chunk_map, gc, graph_config)

    encoded = serialize_graph(graph)
    log_fn("info", "Graph built: $(length(encoded)) bytes encoded")

    result = BuildOutput(encoded, n_entities, n_chunks)
    return JobResult(params.job_id, true, nothing, result, worker_id, timestamp)
end

# ── Query mode ───────────────────────────────────────────────────────────────

function _process_query(params::JobParams, worker_id::String, log_fn, timestamp::String)::JobResult
    qp = params.query
    qp !== nothing || return JobResult(params.job_id, false, "query mode requires query params",
                                       nothing, worker_id, timestamp)

    graph = deserialize_graph(qp.graph)
    q = qp.query
    log_fn("info", "Query: type=$(q.type) from=$(q.from) to=$(q.to)")

    result = _dispatch_query(graph, q)
    return JobResult(params.job_id, true, nothing, result, worker_id, timestamp)
end

function _dispatch_query(graph::GrassmannGraph, q::QuerySpec)::QueryOutput
    if q.type == "greedy_path"
        from = q.from::String
        depth = something(q.depth, 4)
        if q.to !== nothing
            p = find_greedy_path(graph, from, q.to; max_depth=depth)
        else
            p = find_greedy_path(graph, from; depth=depth)
        end
        path_out = p !== nothing ? PathOutput(p.nodes, p.distances, p.total_distance) : nothing
        return QueryOutput(path_out, nothing, nothing)

    elseif q.type == "shortest_path"
        from = q.from::String
        to = q.to::String
        p = find_shortest_path(graph, from, to)
        path_out = p !== nothing ? PathOutput(p.nodes, p.distances, p.total_distance) : nothing
        return QueryOutput(path_out, nothing, nothing)

    elseif q.type == "reachable"
        from = q.from::String
        max_hops = something(q.depth, 3)
        r = reachable(graph, from; max_hops=max_hops)
        entries = [ReachableEntry(id, hops, dist) for (id, hops, dist) in r]
        return QueryOutput(nothing, entries, nothing)

    elseif q.type == "communities"
        comms = communities(graph)
        comm_out = [CommunityOutput(c.members, c.root) for c in comms]
        return QueryOutput(nothing, nothing,
            TopologyOutput(comm_out, BasinOutput[], BridgeOutput[], HubEntry[], 0.0, Vector{String}[]))

    elseif q.type == "basins"
        bs = basins(graph)
        basin_out = [BasinOutput(b.attractor, b.members) for b in bs]
        return QueryOutput(nothing, nothing,
            TopologyOutput(CommunityOutput[], basin_out, BridgeOutput[], HubEntry[], 0.0, Vector{String}[]))

    elseif q.type == "topology"
        comms = communities(graph)
        bs = basins(graph)
        br = bridges(graph)
        hc = hub_centrality(graph)
        conc = hub_concentration(graph)
        bidir = bidirectional_edges(graph)

        return QueryOutput(nothing, nothing, TopologyOutput(
            [CommunityOutput(c.members, c.root) for c in comms],
            [BasinOutput(b.attractor, b.members) for b in bs],
            [BridgeOutput(name, home, reached) for (name, home, reached) in br],
            [HubEntry(id, deg) for (id, deg) in hc],
            conc,
            [[a, b] for (a, b) in bidir]
        ))

    else
        throw(ArgumentError("unknown query type: $(q.type)"))
    end
end

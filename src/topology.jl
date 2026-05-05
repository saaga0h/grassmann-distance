# ── Topology analysis on GrassmannGraph ─────────────────────────────────────

"""
    bidirectional_edges(graph) -> Vector{Tuple{String, String}}

Pairs where A→B AND B→A in the k-NN adjacency (mutual nearest neighbors).
Returned sorted with lexicographically smaller ID first.
"""
function bidirectional_edges(graph::GrassmannGraph)
    pairs = Set{Tuple{String, String}}()
    for (i, adj) in enumerate(graph.neighbors)
        for (j, _) in adj
            if any(nb == i for (nb, _) in graph.neighbors[j])
                a, b = minmax(graph.entities[i].id, graph.entities[j].id)
                push!(pairs, (a, b))
            end
        end
    end
    return sort(collect(pairs))
end

"""
    communities(graph) -> Vector{Community}

Connected components of the bidirectional-edge subgraph, via union-find.
Sorted by size (largest first).
"""
function communities(graph::GrassmannGraph)
    names = [e.id for e in graph.entities]
    bidir = bidirectional_edges(graph)

    parent = Dict(n => n for n in names)
    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function unite(x, y)
        rx, ry = find(x), find(y)
        rx != ry && (parent[rx] = ry)
    end

    for (a, b) in bidir
        unite(a, b)
    end

    groups = Dict{String, Vector{String}}()
    for name in names
        root = find(name)
        push!(get!(groups, root, String[]), name)
    end

    result = [Community(sort(members), root) for (root, members) in groups]
    sort!(result, by=c -> length(c.members), rev=true)
    return result
end

"""
    basins(graph) -> Vector{Basin}

Gravitational basins: follow the greedy nearest neighbor from each entity
(over the full distance matrix) until a cycle is reached. The cycle entry
point is the attractor.
"""
function basins(graph::GrassmannGraph)
    n = length(graph.entities)
    basin_map = Dict{String, Vector{String}}()

    for i in 1:n
        visited = [i]
        current = i
        attractor_idx = i
        for _ in 1:n
            best_idx = 0
            best_dist = Inf
            for j in 1:n
                j == current && continue
                d = graph.distance_matrix[current, j]
                if d < best_dist
                    best_dist = d
                    best_idx = j
                end
            end
            best_idx == 0 && break
            if best_idx in visited
                attractor_idx = best_idx
                break
            end
            push!(visited, best_idx)
            current = best_idx
        end

        attr_name = graph.entities[attractor_idx].id
        push!(get!(basin_map, attr_name, String[]), graph.entities[i].id)
    end

    result = [Basin(attr, sort(members)) for (attr, members) in basin_map]
    sort!(result, by=b -> length(b.members), rev=true)
    return result
end

"""
    bridges(graph) -> Vector{Tuple{String, Int, Vector{Int}}}

Documents whose k-NN neighbors span multiple communities.
Returns (entity_id, home_community_index, reached_community_indices).
"""
function bridges(graph::GrassmannGraph)
    comms = communities(graph)
    doc_comm = Dict{String, Int}()
    for (ci, comm) in enumerate(comms)
        for m in comm.members
            doc_comm[m] = ci
        end
    end

    result = Tuple{String, Int, Vector{Int}}[]
    for (i, adj) in enumerate(graph.neighbors)
        name = graph.entities[i].id
        home = doc_comm[name]
        neighbor_comms = Set{Int}()
        for (j, _) in adj
            push!(neighbor_comms, doc_comm[graph.entities[j].id])
        end
        other = setdiff(neighbor_comms, Set([home]))
        if !isempty(other)
            push!(result, (name, home, sort(collect(neighbor_comms))))
        end
    end

    sort!(result, by=x -> length(x[3]), rev=true)
    return result
end

"""
    hub_centrality(graph) -> Vector{Tuple{String, Int}}

In-degree of each entity in the k-NN graph, sorted descending.
"""
function hub_centrality(graph::GrassmannGraph)
    n = length(graph.entities)
    counts = zeros(Int, n)
    for adj in graph.neighbors
        for (j, _) in adj
            counts[j] += 1
        end
    end

    result = [(graph.entities[i].id, counts[i]) for i in 1:n]
    sort!(result, by=x -> x[2], rev=true)
    return result
end

"""
    hub_concentration(graph) -> Float64

Gini-like measure of in-degree concentration.
0 = perfectly uniform, 1 = single hub gets all edges.
"""
function hub_concentration(graph::GrassmannGraph)
    n = length(graph.entities)
    counts = zeros(Int, n)
    for adj in graph.neighbors
        for (j, _) in adj
            counts[j] += 1
        end
    end
    return _gini_concentration(counts)
end

function _gini_concentration(degrees::Vector{Int})
    sorted = sort(degrees)
    n = length(sorted)
    total = sum(sorted)
    total == 0 && return 0.0
    cumulative = cumsum(sorted) ./ total
    area = sum(cumulative) / n
    return 1.0 - 2.0 * area
end

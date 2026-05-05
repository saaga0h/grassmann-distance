# ── Path finding on GrassmannGraph ───────────────────────────────────────────

"""
    find_greedy_path(graph, from_id; depth=4) -> ConceptualPath

Greedy path: at each hop, move to the nearest unvisited neighbor in the full
distance matrix. Produces a conceptual narrative — each step is locally natural.
"""
function find_greedy_path(graph::GrassmannGraph, from_id::String; depth::Int=4)
    from = _entity_idx(graph, from_id)
    n = length(graph.entities)
    visited = Set([from])
    nodes = [from_id]
    dists = Float64[]

    current = from
    for _ in 1:depth
        best_idx = 0
        best_dist = Inf
        for j in 1:n
            j in visited && continue
            d = graph.distance_matrix[current, j]
            if d < best_dist
                best_dist = d
                best_idx = j
            end
        end
        best_idx == 0 && break
        push!(visited, best_idx)
        push!(nodes, graph.entities[best_idx].id)
        push!(dists, best_dist)
        current = best_idx
    end

    return ConceptualPath(nodes, dists, sum(dists))
end

"""
    find_greedy_path(graph, from_id, to_id; max_depth=10) -> Union{ConceptualPath, Nothing}

Greedy path toward a specific target. At each hop, move to the unvisited neighbor
(from full distance matrix) closest to the target. Returns `nothing` if target
is unreachable within `max_depth` hops.
"""
function find_greedy_path(
    graph::GrassmannGraph, from_id::String, to_id::String;
    max_depth::Int=10
)
    from = _entity_idx(graph, from_id)
    to = _entity_idx(graph, to_id)
    from == to && return ConceptualPath([from_id], Float64[], 0.0)

    n = length(graph.entities)
    visited = Set([from])
    nodes = [from_id]
    dists = Float64[]
    current = from

    for _ in 1:max_depth
        best_idx = 0
        best_target_dist = Inf
        best_hop_dist = 0.0
        for j in 1:n
            j in visited && continue
            d_to_target = graph.distance_matrix[j, to]
            if d_to_target < best_target_dist
                best_target_dist = d_to_target
                best_idx = j
                best_hop_dist = graph.distance_matrix[current, j]
            end
        end
        best_idx == 0 && return nothing
        push!(visited, best_idx)
        push!(nodes, graph.entities[best_idx].id)
        push!(dists, best_hop_dist)
        current = best_idx
        current == to && return ConceptualPath(nodes, dists, sum(dists))
    end

    return nothing
end

"""
    find_shortest_path(graph, from_id, to_id) -> Union{ConceptualPath, Nothing}

Dijkstra's shortest path over the k-NN adjacency. Returns `nothing` if no
path exists in the graph.
"""
function find_shortest_path(graph::GrassmannGraph, from_id::String, to_id::String)
    from = _entity_idx(graph, from_id)
    to = _entity_idx(graph, to_id)
    from == to && return ConceptualPath([from_id], Float64[], 0.0)

    n = length(graph.entities)
    dist = fill(Inf, n)
    prev = fill(0, n)
    dist[from] = 0.0
    visited = falses(n)

    for _ in 1:n
        # Pick unvisited node with smallest distance
        u = 0
        u_dist = Inf
        for i in 1:n
            if !visited[i] && dist[i] < u_dist
                u = i
                u_dist = dist[i]
            end
        end
        u == 0 && break
        u == to && break
        visited[u] = true

        for (v, w) in graph.neighbors[u]
            alt = dist[u] + w
            if alt < dist[v]
                dist[v] = alt
                prev[v] = u
            end
        end
    end

    dist[to] == Inf && return nothing

    # Reconstruct path
    path_indices = Int[]
    cur = to
    while cur != 0
        pushfirst!(path_indices, cur)
        cur = prev[cur]
    end

    nodes = [graph.entities[i].id for i in path_indices]
    hop_dists = [graph.distance_matrix[path_indices[i], path_indices[i+1]]
                 for i in 1:(length(path_indices)-1)]

    return ConceptualPath(nodes, hop_dists, dist[to])
end

"""
    reachable(graph, from_id; max_hops=3) -> Vector{Tuple{String, Int, Float64}}

All entities reachable from `from_id` within `max_hops` over k-NN edges.
Returns (entity_id, hop_count, shortest_distance) sorted by distance.
"""
function reachable(graph::GrassmannGraph, from_id::String; max_hops::Int=3)
    from = _entity_idx(graph, from_id)
    n = length(graph.entities)

    # BFS with distance tracking
    best_dist = fill(Inf, n)
    best_hops = fill(0, n)
    best_dist[from] = 0.0
    best_hops[from] = 0

    queue = [(from, 0, 0.0)]  # (node, hops, cumulative_dist)
    while !isempty(queue)
        u, hops, d = popfirst!(queue)
        hops >= max_hops && continue
        for (v, w) in graph.neighbors[u]
            new_d = d + w
            new_hops = hops + 1
            if new_d < best_dist[v]
                best_dist[v] = new_d
                best_hops[v] = new_hops
                push!(queue, (v, new_hops, new_d))
            end
        end
    end

    results = Tuple{String, Int, Float64}[]
    for i in 1:n
        i == from && continue
        best_dist[i] < Inf && push!(results, (graph.entities[i].id, best_hops[i], best_dist[i]))
    end

    sort!(results, by=x -> x[3])
    return results
end

# ── Helpers ──────────────────────────────────────────────────────────────────

function _entity_idx(graph::GrassmannGraph, id::String)
    haskey(graph.entity_index, id) || throw(ArgumentError("unknown entity: '$id'"))
    return graph.entity_index[id]
end

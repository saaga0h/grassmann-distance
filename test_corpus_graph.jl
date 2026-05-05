#!/usr/bin/env julia
# Corpus graph analysis — topology of document similarity space.
#
# Builds k-NN graphs for both Grassmann and cosine metrics, then analyses:
# - Hub centrality (in-degree): which documents attract everything?
# - Community structure: which documents cluster together?
# - Bridge documents: which connect different communities?
# - Bidirectional edges: where do both A→B and B→A hold?
# - Gravitational basins: groups that converge to the same attractor
#
# Usage:
#   julia --project=. test_corpus_graph.jl

using GrassmannDistance
using LinearAlgebra
using Statistics
using JSON3

# ── Load embeddings ──────────────────────────────────────────────────────────

const DATA_PATH = joinpath(@__DIR__, "test_data", "corpus_embeddings.json")

if !isfile(DATA_PATH)
    error("Missing $DATA_PATH — run the corpus embedding generator first")
end

data = JSON3.read(read(DATA_PATH, String))
n = length(data.entries)
dim = data.dimension

ids     = [String(e.id) for e in data.entries]
docs    = [String(e.doc) for e in data.entries]
embeddings = Matrix{Float64}(undef, dim, n)
for (j, e) in enumerate(data.entries)
    embeddings[:, j] = collect(Float64, e.embedding)
end
for j in 1:n
    embeddings[:, j] ./= norm(@view embeddings[:, j])
end

unique_docs = sort(unique(docs))
n_docs = length(unique_docs)
doc_indices = Dict(d => findall(==(d), docs) for d in unique_docs)
doc_idx = Dict(d => i for (i, d) in enumerate(unique_docs))

println("=== Corpus Graph Analysis ===")
println("Documents: $n_docs  Chunks: $n  Dimension: $dim")
println()

# ── Build document distance matrices ────────────────────────────────────────

const CONFIG = GrassmannConfig(min(n - 1, 20), 2, :geodesic)

# Cosine
println("Computing cosine document matrix...")
cosine_chunk = embeddings' * embeddings
cosine_doc = Matrix{Float64}(undef, n_docs, n_docs)
for (i, d1) in enumerate(unique_docs)
    for (j, d2) in enumerate(unique_docs)
        cosine_doc[i, j] = i == j ? 0.0 :
            mean(cosine_chunk[ci, cj] for ci in doc_indices[d1], cj in doc_indices[d2])
    end
end

# Grassmann (precompute tangent spaces)
println("Computing Grassmann document matrix...")
println("  Precomputing tangent spaces...")
all_ts = Vector{TangentSpace}(undef, n)
for j in 1:n
    point = @view embeddings[:, j]
    nn_idx = knn(point, embeddings, CONFIG.k)
    all_ts[j] = estimate_tangent_space(point, embeddings[:, nn_idx], CONFIG.p)
    j % 100 == 0 && print("  [$j/$n]\r")
end
println("  Computing pairwise doc distances...")

grassmann_doc = Matrix{Float64}(undef, n_docs, n_docs)
for (i, d1) in enumerate(unique_docs)
    for (j, d2) in enumerate(unique_docs)
        if i == j
            grassmann_doc[i, j] = Inf
        else
            idx1 = doc_indices[d1]
            idx2 = doc_indices[d2]
            s1 = length(idx1) <= 5 ? idx1 : idx1[round.(Int, range(1, length(idx1), length=5))]
            s2 = length(idx2) <= 5 ? idx2 : idx2[round.(Int, range(1, length(idx2), length=5))]
            grassmann_doc[i, j] = mean(
                grassmann_distance(all_ts[ci], all_ts[cj]; distance=CONFIG.distance)
                for ci in s1, cj in s2
            )
        end
    end
end
println("  Done.")
println()

# ── Build k-NN graphs ───────────────────────────────────────────────────────

const GRAPH_K = 3  # each doc connects to its 3 nearest neighbors

function build_knn_graph(dist_matrix, names; k=GRAPH_K, lower_is_closer=true)
    n = size(dist_matrix, 1)
    # adjacency: edges[i] = list of (neighbor_idx, score)
    edges = [Tuple{Int, Float64}[] for _ in 1:n]
    for i in 1:n
        scores = [(dist_matrix[i, j], j) for j in 1:n if j != i]
        sort!(scores, by=x -> lower_is_closer ? x[1] : -x[1])
        for rank in 1:min(k, length(scores))
            push!(edges[i], (scores[rank][2], scores[rank][1]))
        end
    end
    return edges
end

g_graph = build_knn_graph(grassmann_doc, unique_docs; lower_is_closer=true)
c_graph = build_knn_graph(cosine_doc, unique_docs; lower_is_closer=false)

# ── 1. Hub centrality (in-degree) ───────────────────────────────────────────

println("=== Hub Centrality (in-degree in $GRAPH_K-NN graph) ===")
println("  How many documents point TO this document as a nearest neighbor?")
println()

function in_degrees(graph, names)
    n = length(graph)
    counts = zeros(Int, n)
    for i in 1:n
        for (j, _) in graph[i]
            counts[j] += 1
        end
    end
    return counts
end

g_indeg = in_degrees(g_graph, unique_docs)
c_indeg = in_degrees(c_graph, unique_docs)

# Sort by cosine in-degree to highlight hubs
order = sortperm(c_indeg, rev=true)

println("  $(rpad("Document", 50))  Grassmann  Cosine")
println("  $(repeat("─", 50))  ─────────  ──────")
for i in order
    g = g_indeg[i]
    c = c_indeg[i]
    bar_g = repeat("█", g)
    bar_c = repeat("█", c)
    println("  $(rpad(unique_docs[i], 50))  $(lpad(g, 2)) $bar_g")
    println("  $(rpad("", 50))  $(lpad(c, 2)) $bar_c  (cosine)")
end
println()

# Hub concentration: Gini-like measure
function hub_concentration(degrees)
    sorted = sort(degrees)
    n = length(sorted)
    total = sum(sorted)
    total == 0 && return 0.0
    cumulative = cumsum(sorted) ./ total
    # Area under Lorenz curve
    area = sum(cumulative) / n
    return 1.0 - 2.0 * area  # 0 = perfectly uniform, 1 = one hub gets everything
end

println("  Hub concentration (0=uniform, 1=single hub):")
println("    Grassmann: $(round(hub_concentration(g_indeg); digits=3))")
println("    Cosine:    $(round(hub_concentration(c_indeg); digits=3))")
println()

# ── 2. Bidirectional edges ──────────────────────────────────────────────────

println("=== Bidirectional Edges (mutual nearest neighbors) ===")
println("  Pairs where A→B AND B→A in the k-NN graph (strongest connections).")
println()

function find_bidirectional(graph, names)
    pairs = Set{Tuple{String, String}}()
    n = length(graph)
    for i in 1:n
        for (j, _) in graph[i]
            # Check if j also points to i
            if any(nb == i for (nb, _) in graph[j])
                a, b = minmax(names[i], names[j])
                push!(pairs, (a, b))
            end
        end
    end
    return sort(collect(pairs))
end

g_bidir = find_bidirectional(g_graph, unique_docs)
c_bidir = find_bidirectional(c_graph, unique_docs)

both_bidir = intersect(Set(g_bidir), Set(c_bidir))
g_only_bidir = setdiff(Set(g_bidir), Set(c_bidir))
c_only_bidir = setdiff(Set(c_bidir), Set(g_bidir))

println("  Both metrics agree ($(length(both_bidir))):")
for (a, b) in sort(collect(both_bidir))
    println("    $a ↔ $b")
end

println()
println("  Grassmann only ($(length(g_only_bidir))):")
for (a, b) in sort(collect(g_only_bidir))
    println("    $a ↔ $b")
end

println()
println("  Cosine only ($(length(c_only_bidir))):")
for (a, b) in sort(collect(c_only_bidir))
    println("    $a ↔ $b")
end
println()

# ── 3. Community detection (connected components of mutual k-NN) ────────────

println("=== Communities (connected components of bidirectional edges) ===")
println()

function find_communities(bidir_pairs, all_names)
    # Union-find
    parent = Dict(n => n for n in all_names)
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

    for (a, b) in bidir_pairs
        unite(a, b)
    end

    communities = Dict{String, Vector{String}}()
    for name in all_names
        root = find(name)
        if !haskey(communities, root)
            communities[root] = String[]
        end
        push!(communities[root], name)
    end

    return sort(collect(values(communities)), by=length, rev=true)
end

g_communities = find_communities(g_bidir, unique_docs)
c_communities = find_communities(c_bidir, unique_docs)

println("  Grassmann communities ($(length(g_communities))):")
for (i, comm) in enumerate(g_communities)
    println("    Community $i ($(length(comm))): $(join(comm, ", "))")
end
println()
println("  Cosine communities ($(length(c_communities))):")
for (i, comm) in enumerate(c_communities)
    println("    Community $i ($(length(comm))): $(join(comm, ", "))")
end
println()

# ── 4. Bridge documents ────────────────────────────────────────────────────

println("=== Bridge Documents ===")
println("  Documents whose k-NN neighbors span multiple communities.")
println()

function find_bridges(graph, communities, names)
    # Map each doc to its community index
    doc_comm = Dict{String, Int}()
    for (i, comm) in enumerate(communities)
        for d in comm
            doc_comm[d] = i
        end
    end

    bridges = Tuple{String, Int, Set{Int}}[]
    for i in 1:length(graph)
        name = names[i]
        my_comm = doc_comm[name]
        neighbor_comms = Set{Int}()
        for (j, _) in graph[i]
            push!(neighbor_comms, doc_comm[names[j]])
        end
        # A bridge reaches into communities other than its own
        other_comms = setdiff(neighbor_comms, Set([my_comm]))
        if !isempty(other_comms)
            push!(bridges, (name, my_comm, neighbor_comms))
        end
    end
    return sort(bridges, by=x -> length(x[3]), rev=true)
end

g_bridges = find_bridges(g_graph, g_communities, unique_docs)
c_bridges = find_bridges(c_graph, c_communities, unique_docs)

println("  Grassmann bridges:")
for (name, home, comms) in g_bridges
    comm_strs = ["C$c" for c in sort(collect(comms))]
    println("    $(rpad(name, 50))  home=C$home  reaches=$(join(comm_strs, ","))")
end

println()
println("  Cosine bridges:")
for (name, home, comms) in c_bridges
    comm_strs = ["C$c" for c in sort(collect(comms))]
    println("    $(rpad(name, 50))  home=C$home  reaches=$(join(comm_strs, ","))")
end
println()

# ── 5. Gravitational basins ────────────────────────────────────────────────
# Follow the greedy path from each document — which attractor does it converge to?

println("=== Gravitational Basins ===")
println("  Follow greedy nearest-neighbor until cycle. Which attractor does each doc reach?")
println()

function find_attractor(start, dist_matrix, names; lower_is_closer=true)
    visited = [start]
    current = start
    for _ in 1:length(names)
        scores = [(dist_matrix[current, j], j) for j in 1:length(names) if j != current]
        sort!(scores, by=x -> lower_is_closer ? x[1] : -x[1])
        next = scores[1][2]
        if next in visited
            return (attractor=names[next], cycle_at=names[next], path=names[visited])
        end
        push!(visited, next)
        current = next
    end
    return (attractor=names[current], cycle_at=names[current], path=names[visited])
end

println("  Grassmann basins:")
g_basins = Dict{String, Vector{String}}()
for i in 1:n_docs
    result = find_attractor(i, grassmann_doc, unique_docs; lower_is_closer=true)
    attr = result.attractor
    if !haskey(g_basins, attr)
        g_basins[attr] = String[]
    end
    push!(g_basins[attr], unique_docs[i])
end
for (attr, members) in sort(collect(g_basins), by=x -> length(x[2]), rev=true)
    println("    Attractor: $(rpad(attr, 45))  Basin ($(length(members))): $(join(members, ", "))")
end

println()
println("  Cosine basins:")
c_basins = Dict{String, Vector{String}}()
for i in 1:n_docs
    result = find_attractor(i, cosine_doc, unique_docs; lower_is_closer=false)
    attr = result.attractor
    if !haskey(c_basins, attr)
        c_basins[attr] = String[]
    end
    push!(c_basins[attr], unique_docs[i])
end
for (attr, members) in sort(collect(c_basins), by=x -> length(x[2]), rev=true)
    println("    Attractor: $(rpad(attr, 45))  Basin ($(length(members))): $(join(members, ", "))")
end
println()

# ── 6. Full adjacency list ──────────────────────────────────────────────────

println("=== Full $GRAPH_K-NN Adjacency ===")
println()
println("  Grassmann graph:")
for i in 1:n_docs
    neighbors = join(["$(unique_docs[j]) ($(round(s; digits=3)))" for (j, s) in g_graph[i]], ", ")
    println("    $(rpad(unique_docs[i], 50)) → $neighbors")
end

println()
println("  Cosine graph:")
for i in 1:n_docs
    neighbors = join(["$(unique_docs[j]) ($(round(s; digits=3)))" for (j, s) in c_graph[i]], ", ")
    println("    $(rpad(unique_docs[i], 50)) → $neighbors")
end

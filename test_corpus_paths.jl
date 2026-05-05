#!/usr/bin/env julia
# Corpus path tracing — multi-hop conceptual paths through document space.
#
# Builds a document-to-document distance matrix using chunk-averaged distances,
# then traces greedy paths from each document through nearest unvisited neighbors.
# Compares the paths Grassmann distance finds vs cosine similarity.
#
# Usage:
#   julia --project=. test_corpus_paths.jl

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

println("=== Corpus Path Tracing ===")
println("Documents: $n_docs  Chunks: $n  Dimension: $dim")
println()

# ── Build document-to-document cosine matrix ─────────────────────────────────

println("Computing cosine document matrix...")
cosine_chunk_matrix = embeddings' * embeddings  # n×n

cosine_doc_matrix = Matrix{Float64}(undef, n_docs, n_docs)
for (i, d1) in enumerate(unique_docs)
    for (j, d2) in enumerate(unique_docs)
        if i == j
            cosine_doc_matrix[i, j] = 0.0
        else
            cosine_doc_matrix[i, j] = mean(
                cosine_chunk_matrix[ci, cj]
                for ci in doc_indices[d1], cj in doc_indices[d2]
            )
        end
    end
end
println("  Done.")

# ── Build document-to-document Grassmann matrix ─────────────────────────────
# Use representative chunks (first + middle + last) per doc to keep compute feasible

println("Computing Grassmann document matrix...")

const CONFIG = GrassmannConfig(min(n - 1, 20), 2, :geodesic)

# Precompute all tangent spaces (expensive but done once)
println("  Precomputing tangent spaces for all $n chunks...")
all_ts = Vector{TangentSpace}(undef, n)
for j in 1:n
    point = @view embeddings[:, j]
    nn_idx = knn(point, embeddings, CONFIG.k)
    nn_vecs = embeddings[:, nn_idx]
    all_ts[j] = estimate_tangent_space(point, nn_vecs, CONFIG.p)
    if j % 50 == 0
        print("  [$j/$n]\r")
    end
end
println("  Tangent spaces computed.       ")

# Now compute chunk-to-chunk Grassmann distances and average per doc pair
println("  Computing pairwise document distances...")
grassmann_doc_matrix = Matrix{Float64}(undef, n_docs, n_docs)

for (i, d1) in enumerate(unique_docs)
    for (j, d2) in enumerate(unique_docs)
        if i == j
            grassmann_doc_matrix[i, j] = Inf
        else
            idx1 = doc_indices[d1]
            idx2 = doc_indices[d2]
            # Sample: use up to 5 representative chunks per doc
            sample1 = length(idx1) <= 5 ? idx1 : idx1[round.(Int, range(1, length(idx1), length=5))]
            sample2 = length(idx2) <= 5 ? idx2 : idx2[round.(Int, range(1, length(idx2), length=5))]
            dists = [grassmann_distance(all_ts[ci], all_ts[cj]; distance=CONFIG.distance)
                     for ci in sample1, cj in sample2]
            grassmann_doc_matrix[i, j] = mean(dists)
        end
    end
    print("  [doc $i/$n_docs]\r")
end
println("  Done.                ")
println()

# ── Helper: trace greedy path ────────────────────────────────────────────────

function trace_path(start_idx::Int, dist_matrix::Matrix{Float64}, names::Vector{String};
                    depth=4, lower_is_closer=true)
    path = [(name=names[start_idx], idx=start_idx, score=0.0)]
    visited = Set([start_idx])

    for _ in 1:depth
        current = path[end].idx
        best_idx = 0
        best_score = lower_is_closer ? Inf : -Inf

        for j in 1:size(dist_matrix, 1)
            j in visited && continue
            s = dist_matrix[current, j]
            if lower_is_closer ? (s < best_score) : (s > best_score)
                best_score = s
                best_idx = j
            end
        end

        best_idx == 0 && break
        push!(visited, best_idx)
        push!(path, (name=names[best_idx], idx=best_idx, score=best_score))
    end

    return path
end

# ── Trace and display paths ──────────────────────────────────────────────────

const DEPTH = 4

println("=== Greedy Conceptual Paths (depth=$DEPTH) ===")
println("  Starting from each document, follow nearest unvisited neighbor.")
println()

# Collect for comparison summary
path_divergence = 0
path_total = 0

for (i, doc) in enumerate(unique_docs)
    g_path = trace_path(i, grassmann_doc_matrix, unique_docs; depth=DEPTH, lower_is_closer=true)
    c_path = trace_path(i, cosine_doc_matrix, unique_docs; depth=DEPTH, lower_is_closer=false)

    # Format paths
    g_names = [p.name for p in g_path]
    c_names = [p.name for p in c_path]

    # Check where they diverge
    first_diff = findfirst(k -> k <= length(g_names) && k <= length(c_names) && g_names[k] != c_names[k], 2:min(length(g_names), length(c_names)))

    println("  $(doc)")
    print("    G: $(g_names[1])")
    for k in 2:length(g_path)
        d = round(g_path[k].score; digits=3)
        print(" →[$d] $(g_names[k])")
    end
    println()
    print("    C: $(c_names[1])")
    for k in 2:length(c_path)
        s = round(c_path[k].score; digits=3)
        print(" →[$s] $(c_names[k])")
    end
    println()

    # Count shared vs divergent steps
    for k in 2:min(length(g_names), length(c_names))
        global path_total += 1
        if g_names[k] != c_names[k]
            global path_divergence += 1
        end
    end
    println()
end

println("=== Path Divergence Summary ===")
println("  Steps compared: $path_total")
println("  Divergent steps: $path_divergence ($(round(100 * path_divergence / path_total; digits=1))%)")
println("  Same steps: $(path_total - path_divergence) ($(round(100 * (path_total - path_divergence) / path_total; digits=1))%)")
println()

# ── Shared backbone: edges that both metrics agree on ────────────────────────

println("=== Consensus Edges ===")
println("  Document pairs that BOTH metrics place as nearest neighbor:")
println()

for (i, doc) in enumerate(unique_docs)
    # Grassmann nearest
    g_dists = [(grassmann_doc_matrix[i, j], j) for j in 1:n_docs if j != i]
    sort!(g_dists)
    g_nearest = g_dists[1][2]

    # Cosine nearest
    c_sims = [(cosine_doc_matrix[i, j], j) for j in 1:n_docs if j != i]
    sort!(c_sims, rev=true)
    c_nearest = c_sims[1][2]

    if g_nearest == c_nearest
        println("  $(rpad(doc, 50)) → $(unique_docs[g_nearest])")
    end
end
println()

println("=== Exclusive Edges ===")
println("  Nearest-neighbor connections found ONLY by one metric:")
println()

for (i, doc) in enumerate(unique_docs)
    g_dists = [(grassmann_doc_matrix[i, j], j) for j in 1:n_docs if j != i]
    sort!(g_dists)
    g_nearest = unique_docs[g_dists[1][2]]

    c_sims = [(cosine_doc_matrix[i, j], j) for j in 1:n_docs if j != i]
    sort!(c_sims, rev=true)
    c_nearest = unique_docs[c_sims[1][2]]

    if g_nearest != c_nearest
        println("  $(rpad(doc, 50))  G→ $(rpad(g_nearest, 45))  C→ $c_nearest")
    end
end

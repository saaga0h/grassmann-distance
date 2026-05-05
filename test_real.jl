#!/usr/bin/env julia
# Real embedding test — Grassmann vs cosine on Ollama-generated embeddings.
#
# Loads test_data/real_embeddings.json (produced by scripts/generate_embeddings.jl)
# and compares Grassmann distance ranking against cosine similarity for every
# single-word entry as a query.
#
# Usage:
#   julia --project=. test_real.jl

using GrassmannDistance
using LinearAlgebra
using Statistics
using JSON3

# ── Load embeddings ──────────────────────────────────────────────────────────

const DATA_PATH = joinpath(@__DIR__, "test_data", "real_embeddings.json")

if !isfile(DATA_PATH)
    error("Missing $DATA_PATH — run: OLLAMA_HOST=<host>:<port> julia --project=. scripts/generate_embeddings.jl")
end

data = JSON3.read(read(DATA_PATH, String))
n = length(data.entries)
dim = data.dimension

println("=== Grassmann vs Cosine on Real Embeddings ===")
println("Model: $(data.model)  Dimension: $dim  Entries: $n")
println()

# Build matrices
ids     = [String(e.id) for e in data.entries]
texts   = [String(e.text) for e in data.entries]
clusters = [String(e.cluster) for e in data.entries]
embeddings = Matrix{Float64}(undef, dim, n)
for (j, e) in enumerate(data.entries)
    embeddings[:, j] = collect(Float64, e.embedding)
end

# Normalise (should already be unit, but be safe)
for j in 1:n
    embeddings[:, j] ./= norm(@view embeddings[:, j])
end

# ── Config ───────────────────────────────────────────────────────────────────

const CONFIG = GrassmannConfig(min(n - 1, 20), 2, :geodesic)

# ── Identify single-word entries (queries) and all entries (candidates) ─────

word_indices = findall(c -> c != "phrase", clusters)
phrase_indices = findall(c -> c == "phrase", clusters)

println("Single-word entries: $(length(word_indices))")
println("Phrase entries:      $(length(phrase_indices))")
println("Config: k=$(CONFIG.k), p=$(CONFIG.p), $(CONFIG.distance)")
println()

# ── Per-query comparison ─────────────────────────────────────────────────────

# Accumulate precision stats
grassmann_precision = Float64[]
cosine_precision    = Float64[]

# Interesting pairs: word that appears in a phrase but phrase is semantically different
interesting_words = Set([
    "cherry", "lime", "mandarin", "banana", "pineapple", "orange",
    "carrot", "apple", "tomato", "coconut",
    "sage", "mint", "turkey", "duck", "crab", "peach", "plum",
])

for qi in word_indices
    query = @view embeddings[:, qi]
    query_cluster = clusters[qi]
    query_text = texts[qi]

    # Candidates: everything except the query itself
    cand_idx = setdiff(1:n, qi)
    cand_emb = embeddings[:, cand_idx]
    cand_ids = ids[cand_idx]
    cand_texts = texts[cand_idx]
    cand_clusters = clusters[cand_idx]

    # Grassmann ranking
    grassmann_results = rank_candidates(query, cand_emb, cand_ids, embeddings, CONFIG)

    # Cosine ranking
    cosine_scores = [dot(query, @view cand_emb[:, j]) for j in 1:length(cand_idx)]
    cosine_order = sortperm(cosine_scores, rev=true)

    # Map id -> text/cluster for display
    id_to_text = Dict(zip(cand_ids, cand_texts))
    id_to_cluster = Dict(zip(cand_ids, cand_clusters))

    # Precision within same cluster (top-K where K = cluster size - 1)
    same_cluster_count = count(==(query_cluster), clusters) - 1  # exclude self
    K = same_cluster_count

    grassmann_hits = count(1:K) do i
        id_to_cluster[grassmann_results[i].id] == query_cluster
    end
    cosine_hits = count(1:K) do i
        id_to_cluster[cand_ids[cosine_order[i]]] == query_cluster
    end

    push!(grassmann_precision, grassmann_hits / K)
    push!(cosine_precision, cosine_hits / K)

    # Print detail for interesting words
    if query_text in interesting_words
        println("─── Query: \"$query_text\" (cluster: $query_cluster) ───")
        println()
        println("  Rank  Grassmann                                          Cosine")
        println("  ────  ─────────────────────────────────────────────────  ─────────────────────────────────────────────────")

        top = min(10, length(cand_idx))
        for i in 1:top
            # Grassmann side
            g_id = grassmann_results[i].id
            g_text = id_to_text[g_id]
            g_dist = round(grassmann_results[i].distance; digits=4)
            g_cluster = id_to_cluster[g_id]
            g_mark = g_cluster == query_cluster ? "*" : " "
            g_str = "$(rpad("$g_text", 30)) $(lpad(string(g_dist), 6)) $g_mark"

            # Cosine side
            ci = cosine_order[i]
            c_text = cand_texts[ci]
            c_score = round(cosine_scores[ci]; digits=4)
            c_cluster = cand_clusters[ci]
            c_mark = c_cluster == query_cluster ? "*" : " "
            c_str = "$(rpad("$c_text", 30)) $(lpad(string(c_score), 6)) $c_mark"

            println("  $(lpad(i, 4))  $g_str  $c_str")
        end
        println("  (* = same cluster as query)")
        println()
    end
end

# ── Summary ──────────────────────────────────────────────────────────────────

println("=== Cluster Precision Summary ===")
println("  (fraction of top-K results from same cluster, K = cluster size - 1)")
println()

# Discover all non-phrase clusters dynamically
all_clusters = sort(unique(clusters[word_indices]))
for cluster in all_clusters
    idx = findall(==(cluster), clusters[word_indices])
    g_prec = mean(grassmann_precision[idx])
    c_prec = mean(cosine_precision[idx])
    println("  $(rpad(cluster, 14))  Grassmann: $(round(g_prec * 100; digits=1))%   Cosine: $(round(c_prec * 100; digits=1))%")
end

println()
println("  Overall mean   Grassmann: $(round(mean(grassmann_precision) * 100; digits=1))%   Cosine: $(round(mean(cosine_precision) * 100; digits=1))%")
println()

# ── Phrase proximity analysis ────────────────────────────────────────────────

println("=== Phrase Proximity Analysis ===")
println("  For each phrase, show nearest single word by each metric")
println()

for pi in phrase_indices
    phrase_vec = @view embeddings[:, pi]
    phrase_text = texts[pi]

    # Only compare against single words
    word_cosines = [(texts[wi], dot(phrase_vec, @view embeddings[:, wi])) for wi in word_indices]
    sort!(word_cosines, by=x -> -x[2])

    # Grassmann: build small candidate set from words only
    word_emb = embeddings[:, word_indices]
    word_ids_sub = ids[word_indices]
    word_texts_sub = texts[word_indices]
    g_results = rank_candidates(phrase_vec, word_emb, word_ids_sub, embeddings, CONFIG)

    g_nearest_id = g_results[1].id
    g_nearest_text = word_texts_sub[findfirst(==(g_nearest_id), word_ids_sub)]
    g_dist = round(g_results[1].distance; digits=4)

    c_nearest_text = word_cosines[1][1]
    c_score = round(word_cosines[1][2]; digits=4)

    println("  \"$phrase_text\"")
    println("    Grassmann nearest: $(rpad(g_nearest_text, 12)) (d=$g_dist)")
    println("    Cosine nearest:    $(rpad(c_nearest_text, 12)) (sim=$c_score)")
    println()
end

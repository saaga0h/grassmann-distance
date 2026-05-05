#!/usr/bin/env julia
# Corpus embedding test — Grassmann vs cosine on document chunk embeddings.
#
# Loads test_data/corpus_embeddings.json and tests whether each metric
# correctly identifies same-document chunks and cross-document affinities.
#
# Usage:
#   julia --project=. test_corpus_real.jl

using GrassmannDistance
using LinearAlgebra
using Statistics
using JSON3

# ── Load embeddings ──────────────────────────────────────────────────────────

const DATA_PATH = joinpath(@__DIR__, "test_data", "corpus_embeddings.json")

if !isfile(DATA_PATH)
    error("Missing $DATA_PATH — run: OLLAMA_HOST=<host>:<port> julia --project=. scripts/generate_corpus_embeddings.jl")
end

data = JSON3.read(read(DATA_PATH, String))
n = length(data.entries)
dim = data.dimension

println("=== Grassmann vs Cosine on Document Corpus ===")
println("Model: $(data.model)  Dimension: $dim")
println("Documents: $(data.n_docs)  Chunks: $n")
println()

# Build matrices
ids     = [String(e.id) for e in data.entries]
docs    = [String(e.doc) for e in data.entries]
embeddings = Matrix{Float64}(undef, dim, n)
for (j, e) in enumerate(data.entries)
    embeddings[:, j] = collect(Float64, e.embedding)
end

# Normalise
for j in 1:n
    embeddings[:, j] ./= norm(@view embeddings[:, j])
end

# ── Config ───────────────────────────────────────────────────────────────────

const CONFIG = GrassmannConfig(min(n - 1, 20), 2, :geodesic)

println("Config: k=$(CONFIG.k), p=$(CONFIG.p), $(CONFIG.distance)")
println()

# ── Document-level statistics ────────────────────────────────────────────────

unique_docs = sort(unique(docs))
doc_chunk_counts = Dict(d => count(==(d), docs) for d in unique_docs)

# ── Test 1: Same-document retrieval precision ────────────────────────────────
# For each chunk, rank all other chunks. Measure how many of the top-K
# results come from the same document.

println("=== Test 1: Same-Document Retrieval ===")
println("  For each chunk, what fraction of nearest neighbors share its source doc?")
println()

grassmann_doc_precision = Dict{String, Vector{Float64}}()
cosine_doc_precision    = Dict{String, Vector{Float64}}()

for qi in 1:n
    query = @view embeddings[:, qi]
    query_doc = docs[qi]
    same_doc_count = doc_chunk_counts[query_doc] - 1  # exclude self

    if same_doc_count == 0
        continue  # single-chunk doc, skip
    end

    cand_idx = setdiff(1:n, qi)
    cand_emb = embeddings[:, cand_idx]
    cand_ids = ids[cand_idx]
    cand_docs = docs[cand_idx]

    K = min(same_doc_count, 10)  # cap at 10 to avoid bias from large docs

    # Grassmann
    g_results = rank_candidates(query, cand_emb, cand_ids, embeddings, CONFIG)
    g_id_to_doc = Dict(zip(cand_ids, cand_docs))
    g_hits = count(1:K) do i
        g_id_to_doc[g_results[i].id] == query_doc
    end

    # Cosine
    cosine_scores = [dot(query, @view cand_emb[:, j]) for j in 1:length(cand_idx)]
    cosine_order = sortperm(cosine_scores, rev=true)
    c_hits = count(1:K) do i
        cand_docs[cosine_order[i]] == query_doc
    end

    if !haskey(grassmann_doc_precision, query_doc)
        grassmann_doc_precision[query_doc] = Float64[]
        cosine_doc_precision[query_doc]    = Float64[]
    end
    push!(grassmann_doc_precision[query_doc], g_hits / K)
    push!(cosine_doc_precision[query_doc],    c_hits / K)
end

# Per-document summary
println("  $(rpad("Document", 50))  Chunks  Grassmann  Cosine")
println("  $(repeat("─", 50))  ──────  ─────────  ──────")

g_all = Float64[]
c_all = Float64[]

for doc in unique_docs
    haskey(grassmann_doc_precision, doc) || continue
    g_prec = mean(grassmann_doc_precision[doc])
    c_prec = mean(cosine_doc_precision[doc])
    push!(g_all, g_prec)
    push!(c_all, c_prec)
    n_chunks = doc_chunk_counts[doc]
    winner = g_prec > c_prec ? " ←G" : (c_prec > g_prec ? " ←C" : "")
    println("  $(rpad(doc, 50))  $(lpad(n_chunks, 6))  $(lpad(round(g_prec*100; digits=1), 8))%  $(lpad(round(c_prec*100; digits=1), 5))%$winner")
end

println()
println("  Overall mean:  Grassmann $(round(mean(g_all)*100; digits=1))%   Cosine $(round(mean(c_all)*100; digits=1))%")
println()

# ── Test 2: Cross-document affinity map ──────────────────────────────────────
# For each document, find the most similar *other* document by averaging
# chunk-to-chunk distances.

println("=== Test 2: Nearest Document (chunk-averaged) ===")
println("  For each doc, which other doc has the most similar chunks?")
println()

# Precompute all pairwise cosine similarities
cosine_matrix = embeddings' * embeddings  # n×n

# For Grassmann: compute per-doc centroid distances (full pairwise is too expensive)
# Instead, pick 3 representative chunks per doc and average their rankings

println("  $(rpad("Document", 50))  Grassmann nearest                                    Cosine nearest")
println("  $(repeat("─", 50))  ──────────────────────────────────────────────────  ──────────────────────────────────────────────────")

for doc in unique_docs
    doc_indices = findall(==(doc), docs)

    # Cosine: average similarity to chunks of each other doc
    best_cosine_doc = ""
    best_cosine_sim = -Inf
    for other_doc in unique_docs
        other_doc == doc && continue
        other_indices = findall(==(other_doc), docs)
        avg_sim = mean(cosine_matrix[i, j] for i in doc_indices, j in other_indices)
        if avg_sim > best_cosine_sim
            best_cosine_sim = avg_sim
            best_cosine_doc = other_doc
        end
    end

    # Grassmann: use first chunk of doc as representative query,
    # find nearest chunk from each other doc
    qi = doc_indices[1]
    query = @view embeddings[:, qi]
    other_idx = findall(i -> docs[i] != doc, 1:n)
    other_emb = embeddings[:, other_idx]
    other_ids = ids[other_idx]
    other_docs = docs[other_idx]

    g_results = rank_candidates(query, other_emb, other_ids, embeddings, CONFIG)
    g_id_to_doc = Dict(zip(other_ids, other_docs))
    best_grassmann_doc = g_id_to_doc[g_results[1].id]
    best_grassmann_dist = round(g_results[1].distance; digits=4)

    g_str = "$(rpad(best_grassmann_doc, 45)) d=$(lpad(best_grassmann_dist, 6))"
    c_str = "$(rpad(best_cosine_doc, 45)) s=$(lpad(round(best_cosine_sim; digits=4), 6))"

    println("  $(rpad(doc, 50))  $g_str  $c_str")
end

println()

# ── Test 3: Detailed query examples ──────────────────────────────────────────
# Pick a few interesting chunks and show full top-10 ranking comparison

println("=== Test 3: Detailed Rankings for Selected Queries ===")
println()

# Pick first chunk of a few diverse documents
sample_docs = filter(d -> d in unique_docs, [
    "soul-speed",
    "journal-semantic-space-geometry",
    "infrastructure-pathologist",
    "molecular-density-matrix-gastronomy",
    "transform-framework",
])

for sample_doc in sample_docs
    qi = findfirst(==(sample_doc), docs)
    qi === nothing && continue

    query = @view embeddings[:, qi]

    cand_idx = setdiff(1:n, qi)
    cand_emb = embeddings[:, cand_idx]
    cand_ids = ids[cand_idx]
    cand_docs = docs[cand_idx]

    g_results = rank_candidates(query, cand_emb, cand_ids, embeddings, CONFIG)
    g_id_to_doc = Dict(zip(cand_ids, cand_docs))

    cosine_scores = [dot(query, @view cand_emb[:, j]) for j in 1:length(cand_idx)]
    cosine_order = sortperm(cosine_scores, rev=true)

    println("─── Query: $(ids[qi]) (doc: $sample_doc) ───")
    println()
    println("  Rank  Grassmann                                          Cosine")
    println("  ────  ─────────────────────────────────────────────────  ─────────────────────────────────────────────────")

    for i in 1:min(10, length(cand_idx))
        g_id = g_results[i].id
        g_doc = g_id_to_doc[g_id]
        g_dist = round(g_results[i].distance; digits=4)
        g_same = g_doc == sample_doc ? "*" : " "
        g_str = "$(rpad(g_id, 40)) $(lpad(string(g_dist), 6)) $g_same"

        ci = cosine_order[i]
        c_id = cand_ids[ci]
        c_doc = cand_docs[ci]
        c_score = round(cosine_scores[ci]; digits=4)
        c_same = c_doc == sample_doc ? "*" : " "
        c_str = "$(rpad(c_id, 40)) $(lpad(string(c_score), 6)) $c_same"

        println("  $(lpad(i, 4))  $g_str  $c_str")
    end
    println("  (* = same document as query)")
    println()
end

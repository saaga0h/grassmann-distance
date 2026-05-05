#!/usr/bin/env julia
# Local integration test — no network needed.
# Generates synthetic 4096D embeddings clustered around known subspaces,
# then compares Grassmann distance ranking against cosine similarity.
#
# Usage:
#   julia --project=. test_local.jl

using GrassmannDistance
using LinearAlgebra
using Statistics

# ── Parameters ───────────────────────────────────────────────────────────────

const AMBIENT_DIM = 4096
const N_CLUSTERS = 5
const POINTS_PER_CLUSTER = 40
const SUBSPACE_DIM = 10          # intrinsic dim of each cluster's subspace
const NOISE_SCALE = 0.01         # noise perpendicular to subspace
const CONFIG = GrassmannConfig(20, 2, :geodesic)

# ── Generate synthetic data ──────────────────────────────────────────────────

println("=== Grassmann Distance Local Test ===")
println("Ambient dim: $AMBIENT_DIM")
println("Clusters: $N_CLUSTERS × $POINTS_PER_CLUSTER points")
println("Subspace dim: $SUBSPACE_DIM, Noise: $NOISE_SCALE")
println()

# Each cluster lives near a random SUBSPACE_DIM-dimensional subspace of R^AMBIENT_DIM
n_total = N_CLUSTERS * POINTS_PER_CLUSTER
embeddings = Matrix{Float64}(undef, AMBIENT_DIM, n_total)
labels = Vector{Int}(undef, n_total)
ids = Vector{String}(undef, n_total)

for c in 1:N_CLUSTERS
    # Random orthonormal basis for this cluster's subspace
    basis = Matrix(qr(randn(AMBIENT_DIM, SUBSPACE_DIM)).Q)

    for j in 1:POINTS_PER_CLUSTER
        idx = (c - 1) * POINTS_PER_CLUSTER + j
        # Point = random combination in subspace + small ambient noise
        coords = randn(SUBSPACE_DIM)
        point = basis * coords + NOISE_SCALE * randn(AMBIENT_DIM)
        # Normalize to unit sphere (like real embeddings)
        embeddings[:, idx] = point / norm(point)
        labels[idx] = c
        ids[idx] = "cluster$(c)_point$(j)"
    end
end

println("Generated $n_total embeddings")
println()

# ── Query: pick first point from cluster 1 ───────────────────────────────────

query_idx = 1
query = @view embeddings[:, query_idx]
query_label = labels[query_idx]
println("Query: $(ids[query_idx]) (cluster $query_label)")
println()

# All other points are candidates
candidate_idx = setdiff(1:n_total, query_idx)
candidate_emb = embeddings[:, candidate_idx]
candidate_ids = ids[candidate_idx]
candidate_labels = labels[candidate_idx]

# ── Grassmann ranking ────────────────────────────────────────────────────────

println("Computing Grassmann distance ranking...")
t_grassmann = @elapsed begin
    grassmann_results = rank_candidates(query, candidate_emb, candidate_ids, embeddings, CONFIG)
end
println("  Time: $(round(t_grassmann; digits=3))s")

# ── Cosine similarity ranking (for comparison) ──────────────────────────────

println("Computing cosine similarity ranking...")
t_cosine = @elapsed begin
    cosine_scores = [dot(query, @view candidate_emb[:, j]) for j in 1:size(candidate_emb, 2)]
    cosine_order = sortperm(cosine_scores, rev=true)  # higher = more similar
end
println("  Time: $(round(t_cosine; digits=3))s")
println()

# ── Compare top-K results ────────────────────────────────────────────────────

function cluster_of(name::String)
    # Parse "cluster3_point7" → 3
    m = match(r"cluster(\d+)_", name)
    return parse(Int, m[1])
end

K = 20
println("=== Top $K Results ===")
println()
println("Grassmann ranking (k=$(CONFIG.k), p=$(CONFIG.p), $(CONFIG.distance)):")
println("  Rank  ID                        Cluster  Distance")
same_cluster_grassmann = 0
for i in 1:K
    entry = grassmann_results[i]
    c = cluster_of(entry.id)
    marker = c == query_label ? " ✓" : ""
    c == query_label && (global same_cluster_grassmann += 1)
    println("  $(lpad(i, 4))  $(rpad(entry.id, 24))  $(c)        $(round(entry.distance; digits=4))$marker")
end

println()
println("Cosine similarity ranking:")
println("  Rank  ID                        Cluster  Score")
same_cluster_cosine = 0
for i in 1:K
    j = cosine_order[i]
    c = cluster_of(candidate_ids[j])
    marker = c == query_label ? " ✓" : ""
    c == query_label && (global same_cluster_cosine += 1)
    println("  $(lpad(i, 4))  $(rpad(candidate_ids[j], 24))  $(c)        $(round(cosine_scores[j]; digits=4))$marker")
end

# ── Precision summary ────────────────────────────────────────────────────────

println()
println("=== Precision @ $K (same cluster as query) ===")
println("  Grassmann: $same_cluster_grassmann / $K ($(round(100 * same_cluster_grassmann / K; digits=1))%)")
println("  Cosine:    $same_cluster_cosine / $K ($(round(100 * same_cluster_cosine / K; digits=1))%)")

# ── Distance distribution by cluster ────────────────────────────────────────

println()
println("=== Grassmann distance distribution by cluster ===")
for c in 1:N_CLUSTERS
    dists = [r.distance for r in grassmann_results if cluster_of(r.id) == c]
    marker = c == query_label ? " (query cluster)" : ""
    println("  Cluster $c$marker: mean=$(round(mean(dists); digits=4))  std=$(round(std(dists); digits=4))  min=$(round(minimum(dists); digits=4))")
end

println()
println("=== Cosine similarity distribution by cluster ===")
for c in 1:N_CLUSTERS
    sims = [cosine_scores[j] for j in 1:length(candidate_labels) if candidate_labels[j] == c]
    marker = c == query_label ? " (query cluster)" : ""
    println("  Cluster $c$marker: mean=$(round(mean(sims); digits=4))  std=$(round(std(sims); digits=4))  max=$(round(maximum(sims); digits=4))")
end

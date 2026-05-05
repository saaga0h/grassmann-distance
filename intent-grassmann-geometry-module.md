# Intent: Grassmann Distance Geometry Module for FORGE
## Date: 2026-05-05
## Status: draft — for /intent-bridge

## Core Intent
Replace cosine similarity as the retrieval metric for high-dimensional embedding spaces across the project ecosystem with a geometrically honest alternative based on Grassmann distance. Each track, document, or entity in 4096D embedding space is not a point but a local manifold — a small surface whose structure is estimated from its nearest neighbors via local PCA. Similarity between two entities is then the distance between their local manifolds, measured as the principal angles between their tangent spaces. This is implemented as a FORGE-dispatchable Julia module that accepts candidate vectors and returns geometrically ranked results, making correct high-dimensional geometry available to Calliope, Journal, Minerva, Sibyl, and Jeeves through a single shared interface. The driving insight is that cosine similarity is the wrong tool for this space — it measures angle from origin between points, ignoring local structure entirely, and suffers from concentration of measure at high dimensions where all pairwise distances converge. Grassmann distance measures the angle between local linear approximations of the manifold each entity inhabits, which is geometrically meaningful regardless of ambient dimensionality.

## Context
Emerged from a conversation examining why cosine similarity underperforms in high-dimensional semantic spaces. The standard explanation — that embedding space is a compressed lower-dimensional representation — obscures the real problem: cosine similarity is the wrong metric, not a suboptimal one. At 4096D, concentration of measure collapses the contrast ratio between similar and dissimilar pairs. The industry response — reranking, cross-encoders, hybrid retrieval — is a bandage over a metric that was never correct.

The embedding model in use is qwen3-embedding:8b, which produces 4096D vectors natively and supports Matryoshka Representation Learning (MRL) — meaning the first 1024 dimensions are trained to be maximally informative and can be used as a cheap pre-filter. pgvector's HNSW index currently caps at 2000D, so 4096D vectors are already doing sequential scans for cosine retrieval, removing the index acceleration argument for staying with cosine.

The alternative derives from treating each entity as a local manifold estimated by local PCA over k-nearest neighbors. The tangent space — the principal components of that neighborhood — is a subspace of the ambient 4096D space. Grassmann distance between two such subspaces is the norm of the principal angles between them, computed via SVD of the cross-product of the two tangent space matrices. This is precomputable at ingestion time, making query-time cost only the SVD of a small k×k matrix per candidate.

The tiered architecture: pgvector cosine on MRL-truncated 1024D vectors as a pre-filter to get candidates (fits HNSW index limit, gets acceleration), then Grassmann distance via FORGE Julia module on the candidate set for honest geometric ranking. Data volume is small enough that 1k candidates at 16KB each is ~16MB over 10G — negligible latency.

Julia was chosen over Go for the geometry module because: linear algebra is first-class in Julia, the REPL enables interactive debugging of numerical behavior, GPU acceleration via CUDA.jl is natural, and FORGE already provides the dispatch infrastructure. The module is domain-agnostic — in goes candidate vectors and query context, out comes ranked results — making it reusable across all projects in the ecosystem.

## What This Is Not
- Not a replacement for pgvector or Postgres — cosine pre-filter stays, Postgres remains the store
- Not a new retrieval architecture for each project individually — one shared FORGE module, not per-project implementations
- Not a real-time reranking layer bolted onto existing cosine results — the metric changes, not just the ranking step
- Not a C extension to pgvector — considered and deferred as requiring deep Postgres internals expertise and production-grade C that isn't justified at current data volumes
- Not handling the full library in one pass — the pre-filter must reduce to a manageable candidate set before geometry runs
- Not resolving what "complementarity" means formally for each project — Calliope's density matrix complementarity approach is a separate concern from distance ranking

## Open Questions

**What is k for local PCA neighborhood estimation?**
Too small — underdetermined tangent space, noise dominates. Too large — neighborhoods span multiple local structures, tangent space averages meaninglessly. Likely project-dependent based on library density. *needs an experiment*

**How many principal components per tangent space?**
1 principal component gives a tangent line (Grassmann(n,1)), 2 gives a tangent plane (Grassmann(n,2)). More components capture richer local structure but increase SVD cost at query time. The right number probably depends on intrinsic dimensionality of the local neighborhood. *needs an experiment*

**What does Grassmann distance actually improve in Calliope output quality?**
The mathematical argument is sound but empirical validation against real curation sessions hasn't happened. Need a test that compares cosine-ranked vs Grassmann-ranked candidates on known good sessions. *needs an experiment*

**How are precomputed tangent spaces stored and updated?**
Each entity needs its principal components persisted — likely as a separate column or table. When new tracks are ingested the neighborhood changes and tangent spaces need recomputation. Incremental update strategy vs full recomputation threshold unclear. *probably answered in the codebase*

**What is the FORGE job interface — batch or streaming?**
Sending 1k candidates as a batch is straightforward. Whether FORGE expects a single request-response or supports streaming results back affects latency profile for interactive use in Calliope. *probably answered in the codebase*

**How does MRL truncation interact with tangent space estimation?**
Tangent spaces are estimated in full 4096D. Grassmann distance is computed in full 4096D. The 1024D truncation is only for the cosine pre-filter. This is the intended design but worth verifying that the pre-filter candidate set doesn't systematically exclude entities whose full 4096D manifold structure would rank them highly. *needs an experiment*

**Is Grassmann distance the right metric for complementarity or only for similarity?**
Today's conversation distinguished similarity (proximity) from complementarity (completion). Grassmann distance measures geometric similarity of local structure. For complementarity — finding what fills the gaps — the density matrix approach may be the right tool and Grassmann distance the wrong question. The two use cases may need different metrics. *needs a decision*

## Intent in One Sentence
Replace cosine similarity with Grassmann distance over precomputed local tangent spaces, implemented as a domain-agnostic FORGE-dispatchable Julia module, so that all projects in the ecosystem retrieve against geometrically honest manifold structure rather than directional proximity in a space where that metric is known to fail.

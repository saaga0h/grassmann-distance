# GrassmannDistance — Concepts

> _Conceptual path finding in embedding space via Grassmann manifold geometry._

**As of**: May 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why Cosine Similarity Fails](#2-why-cosine-similarity-fails)
3. [Tangent Spaces as Local Geometry](#3-tangent-spaces-as-local-geometry)
4. [Grassmann Distance](#4-grassmann-distance)
5. [From Distance to Graph](#5-from-distance-to-graph)
6. [Conceptual Paths](#6-conceptual-paths)
7. [Topology as Structure Discovery](#7-topology-as-structure-discovery)
8. [Chunks and Entities](#8-chunks-and-entities)
9. [Design Decisions and Roads Not Taken](#9-design-decisions-and-roads-not-taken)
10. [Relationships Between Concepts](#10-relationships-between-concepts)

---

## 1. Problem Statement

### What This Solves

High-dimensional embedding spaces (4096D from models like qwen3-embedding) encode semantic meaning, but the standard retrieval metric — cosine similarity — is geometrically wrong for this space. It measures angle from origin between two points, ignoring local structure. At high dimensions, concentration of measure collapses contrast between similar and dissimilar pairs, making cosine rankings unreliable.

The industry response — reranking, cross-encoders, hybrid retrieval — treats the symptom. GrassmannDistance treats the cause: the metric itself is wrong.

### Why the Approach Is Non-Obvious

The key insight is that each entity in embedding space is not a point but a local manifold — a small surface whose shape is estimated from its neighbors. Similarity between entities should measure how similar their local geometry is, not how close their individual vectors are. This shifts the problem from point-to-point distance to subspace-to-subspace distance, which lives on the Grassmann manifold — a space where each "point" is a linear subspace of fixed dimension.

The practical consequence: cosine similarity draws a topology where all paths converge to vocabulary hubs. Grassmann distance draws a topology where paths follow domain-coherent conceptual lanes. The difference is not incremental — testing on a 27-document corpus showed 71.3% path divergence between the two metrics.

---

## 2. Why Cosine Similarity Fails

Cosine similarity measures the angle between two vectors from the origin. In low dimensions, this is intuitive. In 4096D, three problems emerge:

**Concentration of measure.** As dimensionality grows, pairwise cosine distances converge. The gap between "similar" and "dissimilar" shrinks, making rankings fragile — small noise flips order.

**Origin dependence.** Cosine is defined relative to the origin, which has no semantic meaning in embedding space. Two documents that are conceptually related but lie in different regions of the space may have small cosine similarity simply because they point in different directions from an arbitrary origin.

**No local structure.** Cosine treats each embedding as an isolated direction. It ignores the neighborhood — the local manifold shape that encodes how an entity relates to its conceptual context. A document about physics surrounded by other physics documents has a different local geometry than a document about physics surrounded by cooking recipes, even if the physics embeddings are identical.

The hub problem is the observable consequence: cosine similarity creates gravitational attractors where meta-documents (broad vocabulary, many weak connections) pull disproportionate traffic. In testing, one hub attracted 48% of a corpus into its gravitational basin (hub concentration: 0.629). Grassmann distance on the same corpus had hub concentration of 0.256 — documents clustered by conceptual affinity, not vocabulary overlap.

---

## 3. Tangent Spaces as Local Geometry

A tangent space is the local linear approximation of the manifold at a point. For each embedding vector, we estimate its tangent space by:

1. Finding the k nearest neighbors (local PCA neighborhood)
2. Centering the neighborhood (subtract mean)
3. Computing SVD of the centered neighborhood matrix
4. Taking the first p left singular vectors as the tangent space basis

The result is a `TangentSpace`: an orthonormal basis matrix (ambient_dim x p) plus the neighborhood centroid. The basis spans the p-dimensional subspace that best approximates the local variation around that point.

**Why p matters.** p=1 gives a tangent line (the dominant direction of local variation). p=2 gives a tangent plane. Higher p captures richer structure but increases computation. The right value depends on the intrinsic dimensionality of the local neighborhood. In practice, p=2 captures enough structure to distinguish local geometries while keeping the SVD at query time small (2x2 cross-product matrix).

**Why k matters.** k controls the neighborhood radius. Too small: underdetermined tangent space, noise dominates. Too large: the neighborhood spans multiple local structures and the tangent space averages them meaninglessly. The default k=20 works for moderate-density regions of embedding space.

---

## 4. Grassmann Distance

The Grassmannian Gr(n, p) is the space of all p-dimensional subspaces of R^n. Each tangent space — a p-dimensional subspace of the ambient 4096D space — is a point on this manifold. Grassmann distance measures how far apart two such points are.

The computation:

1. Given two tangent space bases U (n x p) and V (n x p)
2. Compute the cross-product matrix M = U'V (p x p)
3. SVD of M gives singular values sigma_1, ..., sigma_p
4. Principal angles: theta_i = arccos(sigma_i)
5. Geodesic distance: sqrt(sum(theta_i^2))

The principal angles are the canonical angles between the two subspaces. If the subspaces are identical, all angles are zero and the distance is zero. If they are orthogonal, all angles are pi/2 and the distance is maximal.

**Geodesic vs chordal.** The geodesic distance (sqrt of sum of squared angles) measures arc length along the Grassmannian. The chordal distance (sqrt of sum of squared sines of angles) measures straight-line distance through the ambient space. Geodesic is the default — it respects the manifold's geometry.

**What this measures intuitively.** Two entities have small Grassmann distance when their local neighborhoods "point the same way" — the directions of maximum variation around each entity are aligned. This captures conceptual similarity in a way that is invariant to position in the ambient space and robust to the curse of dimensionality.

---

## 5. From Distance to Graph

A pairwise distance matrix alone is useful for ranking but doesn't reveal structure. The graph layer converts distances into a navigable topology:

1. **Entity distance matrix.** For each pair of entities, compute the average Grassmann distance across representative chunks (see Section 8). This produces an (n_entities x n_entities) symmetric distance matrix.

2. **k-NN adjacency.** For each entity, find its k_graph nearest neighbors by Grassmann distance. This produces a directed graph — A's nearest neighbor may not reciprocate.

3. **GrassmannGraph.** The complete precomputed structure: entities, embeddings, tangent spaces, distance matrix, and k-NN adjacency list. Built once, queried many times.

The k_graph parameter (default: 3) controls graph density. Too low: disconnected components, paths can't reach parts of the corpus. Too high: every entity connects to everything, topology becomes trivial.

---

## 6. Conceptual Paths

The graph enables path finding — tracing multi-hop conceptual trajectories through the corpus.

**Greedy path.** At each hop, move to the nearest unvisited neighbor in the full distance matrix. This produces a conceptual narrative — each step is locally the most natural transition. Greedy paths reveal the dominant conceptual lanes in the corpus.

**Shortest path (Dijkstra).** Minimize total Grassmann distance between source and target. This finds the most efficient conceptual route, which may use unexpected intermediaries.

**Reachability.** BFS over the k-NN adjacency to find all entities within a given number of hops, with cumulative distance.

The difference from cosine: cosine paths converge to the same vocabulary hub within 2-3 hops regardless of starting point. Grassmann paths follow domain-coherent lanes — infrastructure topics stay in the infrastructure lane, theoretical topics stay in theory. The paths are navigable because the distance metric preserves domain boundaries.

---

## 7. Topology as Structure Discovery

Beyond paths, the graph's structure reveals corpus-level organization:

**Bidirectional edges.** Mutual nearest neighbors — A's k-NN includes B and B's k-NN includes A. These are the strongest conceptual bonds in the corpus.

**Communities.** Connected components of the bidirectional-edge subgraph (union-find). Groups of entities that mutually recognize each other as nearest neighbors. These emerge without supervision — the geometry defines the communities.

**Gravitational basins.** For each entity, greedily follow the nearest neighbor chain until reaching a cycle. The cycle's entry point is the attractor. Entities that converge to the same attractor form a basin — a gravitational well in conceptual space. Hub concentration (Gini coefficient over basin sizes) measures how uniformly distributed the basins are. Low concentration means distributed structure; high concentration means hub dominance.

**Bridges.** Entities whose k-NN neighbors span multiple communities. These are the conceptual connectors — topics that link otherwise separate domains.

---

## 8. Chunks and Entities

A document produces multiple embedding vectors (chunks), one per segment. An entity is a named group of chunks — typically one document, but the abstraction is general.

**Why chunks matter.** A single embedding vector compresses an entire document segment into one point. Multiple chunks from the same document sample different aspects of its content. The tangent space is estimated per chunk (each chunk has its own local geometry), but the entity distance averages across representative chunks. This gives a more robust distance estimate than comparing single vectors.

**Contiguous chunk ranges.** In the GrassmannGraph, each entity's chunks occupy a contiguous range in the embedding matrix. This is a caller-enforced invariant — chunks must be ordered by entity. The Entity struct stores a UnitRange{Int} pointing into the shared arrays.

**Representative sampling.** When an entity has many chunks, computing all pairwise Grassmann distances between two entities' chunks is expensive. Instead, we sample evenly-spaced representative chunks (max_chunks parameter, default: 5) and average their pairwise distances.

---

## 9. Design Decisions and Roads Not Taken

### Precomputed graph, not lazy evaluation

The GrassmannGraph precomputes all tangent spaces and the full distance matrix at build time. This is expensive (O(n^2) entity pairs, each requiring SVD) but makes all queries — paths, topology, reachability — instantaneous lookups. The alternative (lazy distance computation) would make build cheap but every query slow. Since this is a batch compute job dispatched by FORGE, build cost is acceptable.

### Entity distance as chunk average, not minimum

We average Grassmann distances across representative chunks rather than taking the minimum. Minimum distance would find the single closest chunk pair, which might be an outlier. Average distance measures the overall geometric relationship between two entities' local manifolds.

### Geodesic over chordal as default

The geodesic distance respects the Grassmannian's intrinsic geometry. Chordal distance is faster (no arccos) but loses curvature information. For small angles the two are nearly identical; the difference matters for entities with very different local geometries, which is exactly where you want the metric to discriminate.

### Julia, not Go or Python

Linear algebra is first-class in Julia — LAPACK-backed SVD, native matrix operations, GPU acceleration via KernelAbstractions.jl. The REPL enables interactive exploration of numerical behavior. Python's NumPy could handle the math but lacks the GPU story without additional frameworks. Go has no native linear algebra worth mentioning.

### GPU for kNN and cross-products, CPU for SVD

The three stages of graph construction have different compute profiles:
1. **kNN** — pairwise distance matrix via matmul. GPU-native (single GEMM operation).
2. **Tangent space SVD** — many small SVDs (dim x k matrices). GPU SVD dispatch through rocSOLVER is unreliable; CPU SVD on small matrices is fast enough.
3. **Entity distance cross-products** — batch dot products across all entity pairs. GPU-native (parallel element-wise multiply and reduce).

### Graph blob as opaque base64

The serialized GrassmannGraph is a base64-encoded Julia Serialization blob. Clients pass it through without inspecting it. This avoids defining a cross-language schema for internal data structures that only Julia needs to read. The tradeoff: graphs are not portable to non-Julia consumers. This is acceptable because all graph operations run inside the Julia worker.

### No incremental graph updates

Adding an entity requires recomputing tangent spaces (neighborhoods change) and the distance matrix. Incremental update is possible but complex — deferred until scale demands it. Current design: rebuild the graph from scratch. At 1000 entities, build takes seconds on GPU.

### FORGE as the only client interface

The module has no HTTP API, no CLI, no REPL entry point for production use. All production interaction goes through MQTT via FORGE. This keeps the module focused — it does geometry, FORGE does orchestration. Development and testing use Julia's REPL and test scripts directly.

---

## 10. Relationships Between Concepts

```
Embeddings (4096D vectors)
    │
    ├── kNN neighborhoods (k nearest neighbors per chunk)
    │       │
    │       └── Local PCA → TangentSpace (orthonormal basis, p dimensions)
    │                           │
    │                           ├── principal_angles(U, V) → angles between subspaces
    │                           │       │
    │                           │       └── grassmann_distance() → scalar distance
    │                           │
    │                           └── entity_distance() → averaged over representative chunks
    │                                       │
    │                                       └── Distance Matrix (n_entities x n_entities)
    │                                               │
    │                                               ├── k-NN Adjacency → GrassmannGraph
    │                                               │       │
    │                                               │       ├── find_greedy_path()
    │                                               │       ├── find_shortest_path()
    │                                               │       ├── reachable()
    │                                               │       │
    │                                               │       ├── bidirectional_edges()
    │                                               │       ├── communities()
    │                                               │       ├── basins()
    │                                               │       ├── bridges()
    │                                               │       └── hub_concentration()
    │                                               │
    │                                               └── Topology Analysis
    │
    └── Chunks → Entities (named groups of contiguous chunks)
```

The dependency flows downward. Embeddings are the raw input. Everything above the distance matrix is precomputed during graph build. Everything below the distance matrix is computed on demand during queries.

# TODO

## Sysimage build with AMDGPU

**Current state**: sysimage build is skipped in `singularity.def`. The worker
JIT-compiles on every cold start (~30s). Acceptable for now since this is a
batch job and startup time is not critical.

**Deferred until GPU server rebuild**: korok needs significantly more local disk
for Singularity scratch and growing model storage before a sysimage build is
practical. The build takes 20-30 min and must be redone after significant code
changes, so doing it on a stable server makes more sense.

**Why it's skipped**: PackageCompiler uses Julia's bundled LLVM to compile the
sysimage. AMDGPU.jl injects AMD GCN bitcode (`amdgcn` intrinsics) into that
pipeline. Julia's generic LLVM can't handle GCN intrinsics and aborts with
`LLVM ERROR: Broken module found`. This happens even when AMDGPU is not in the
package list passed to `create_sysimage` — `using GrassmannDistance` loads it
conditionally. Installing ROCm on the build host does not help because the crash
is inside Julia's own LLVM, not the ROCm runtime.

**Proper solution**: build the sysimage on the GPU node where ROCm's LLVM is
present, then store it on NFS for the container to pick up at runtime.

Steps:
1. On the GPU node, with `/opt/rocm` available:
   ```bash
   singularity exec --bind /opt/rocm:/opt/rocm \
     /nfs/images/grassmann-distance/worker.sif \
     julia --project=/app -e '
       using PackageCompiler
       create_sysimage(:GrassmannDistance;
         sysimage_path="/tmp/sysimage.so",
         precompile_execution_file="/app/test/runtests.jl")'
   ```

2. Copy the resulting `sysimage.so` to NFS:
   ```bash
   cp /tmp/sysimage.so /nfs/images/grassmann-distance/sysimage.so
   ```

3. The container runscript already checks `SYSIMAGE_PATH` and falls back
   gracefully if no sysimage is found — no container rebuild needed.
   Set in the Nomad job:
   ```hcl
   SYSIMAGE_PATH = "/nfs/images/grassmann-distance/sysimage.so"
   ```

4. Automate: add a post-deploy step to the Gitea workflow that dispatches a
   one-off Nomad job to rebuild the sysimage on the GPU node after each image
   publish. The sysimage job can write directly to `/nfs/images/grassmann-distance/`.

## Dimensionality crossover — GD vs cosine discrimination

qwen3-embedding supports MRL (Matryoshka Representation Learning), so we can
get 768D, 1024D, and 4096D embeddings for the same corpus without re-embedding.
Test whether Grassmann distance still outperforms cosine at lower dimensions,
and find the crossover point (if any) where cosine concentration-of-measure
problems disappear and GD stops adding value.

Test plan (via grassmann-bench):
1. Embed foodCorpus at 768D, 1024D, 4096D using qwen3-embedding MRL truncation
2. Build Grassmann graphs at each dimensionality (same k, p parameters)
3. Compare hub concentration, path divergence, community structure across dims
4. Check if cosine topology improves enough at 768D/1024D to close the gap

This matters for practical deployment — if GD only wins at high dimensions,
users with 768D embeddings (the common case) don't benefit.

## REPL Experiments

### Grassmann distance for complementarity
Density matrix implementation already exists for complementarity ranking.
Open question: can Grassmann distance also work for complementarity, or is
it fundamentally a similarity-only metric? The geometry suggests DG measures
structural similarity of local manifolds — complementarity (what fills gaps)
may need a different framing. Worth testing in REPL with real embeddings:
- Compare DG rankings vs density matrix rankings on known complementary pairs
- Check whether large Grassmann distance correlates with complementarity
  or just with dissimilarity (not the same thing)
- If DG can do both, the module interface simplifies significantly

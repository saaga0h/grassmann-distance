# TODO

## Sysimage build with AMDGPU

**Current state**: sysimage build is skipped in `singularity.def`. The worker
JIT-compiles on every cold start (~30s). Acceptable for now since this is a
batch job and startup time is not critical.

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

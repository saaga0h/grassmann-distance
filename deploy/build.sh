#!/bin/bash
# Grassmann Distance Worker — Build Script
#
# Builds the Singularity image and publishes it to NFS.
# Designed to run on the Packer build host or any machine with Singularity installed.
#
# Optional environment variables:
#   GIT_COMMIT - Git commit hash (auto-detected if in a git repo)
#
# NFS layout (fixed):
#   /nfs/images/grassmann-distance/worker.sif  — published image (mirrored storage)
#   /nfs/cache/                                — build-time scratch (stripe-friendly)

set -euo pipefail

WORKER_NAME="grassmann-distance"
IMAGE_DIR="/nfs/images/$WORKER_NAME"
CACHE_DIR="/nfs/cache"

echo "================================================"
echo "Grassmann Distance Worker Build"
echo "================================================"

# Validate NFS mounts
if [ ! -d "$IMAGE_DIR" ] && ! mkdir -p "$IMAGE_DIR" 2>/dev/null; then
    echo "ERROR: /nfs/images not mounted or not writable"
    exit 1
fi
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: /nfs/cache not mounted"
    exit 1
fi

# Validate we can find singularity.def
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$REPO_DIR/singularity.def" ]; then
    echo "ERROR: singularity.def not found at $REPO_DIR/singularity.def"
    exit 1
fi

cd "$REPO_DIR"

# Record git commit
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo 'unknown')}"
echo "Git commit: $GIT_COMMIT"
echo ""

# Build Singularity image
echo "=== Building Singularity image ==="

# Build scratch goes to local disk — NFS xattrs break fakeroot unpacking
BUILD_TMPDIR="$CACHE_DIR/singularity-build-$$"
mkdir -p "$BUILD_TMPDIR"

SIF_PATH="$IMAGE_DIR/worker.sif"

if command -v sudo >/dev/null 2>&1; then
    sudo -E SINGULARITY_TMPDIR="$BUILD_TMPDIR" singularity build "$SIF_PATH" singularity.def
else
    SINGULARITY_TMPDIR="$BUILD_TMPDIR" singularity build "$SIF_PATH" singularity.def
fi

rm -rf "$BUILD_TMPDIR"

if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: Singularity build failed"
    exit 1
fi

SIZE=$(du -h "$SIF_PATH" | cut -f1)
echo "Image: $SIF_PATH ($SIZE)"
echo ""

# Write manifest alongside the image
cat > "$IMAGE_DIR/manifest.json" <<EOF
{
  "worker": "$WORKER_NAME",
  "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_commit": "$GIT_COMMIT",
  "size_mb": $(du -m "$SIF_PATH" | cut -f1)
}
EOF

echo "=== Build complete ==="
ls -lh "$IMAGE_DIR"
cat "$IMAGE_DIR/manifest.json"

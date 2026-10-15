#!/bin/bash -l
# ==============================================================================
# SCRIPT: Set up the ViT training example (Aurora)
# DESCRIPTION: Clones the sc25-dl-tutorial training code at the pinned commit
#              these examples are tested with, applies the DataLoader and
#              XPU model-parallel patches, and builds the venv. Run on a login node (needs internet).
#              Safe to re-run.
# ==============================================================================

set -euo pipefail

# Device-portable branch of NERSC/sc25-dl-tutorial (upstream PR #4,
# branch sunspot-xpu-support). Same commit on every system.
REPO_URL=https://github.com/saforem2/sc25-dl-tutorial.git
COMMIT=7e91316d7af4af0f51bbfc7e9e7194418ae454ee
EZPZ="git+https://github.com/saforem2/ezpz@d2ae5311e00a3ea84ebb4ac6c68e82d729abf24a"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PATCHES=("$EXAMPLE_DIR/sc25-persistent-workers.patch" "$EXAMPLE_DIR/sc25-xpu-mp.patch")

if [[ ! -d "$SC25_DIR/.git" ]]; then
    git clone --quiet --filter=blob:none "$REPO_URL" "$SC25_DIR"
fi

if [[ -n "$(git -C "$SC25_DIR" status --porcelain --untracked-files=no)" ]]; then
    # Local changes are allowed only on the pinned commit (our patches)
    if [[ "$(git -C "$SC25_DIR" rev-parse HEAD)" != "$COMMIT" ]]; then
        echo "error: $SC25_DIR has local changes; not switching commits" >&2
        exit 1
    fi
else
    git -C "$SC25_DIR" fetch --quiet origin "$COMMIT"
    git -C "$SC25_DIR" -c advice.detachedHead=false checkout --quiet "$COMMIT"
fi

# Apply each patch unless already applied (so older checkouts pick up new ones)
for patch in "${PATCHES[@]}"; do
    if git -C "$SC25_DIR" apply --reverse --check "$patch" 2>/dev/null; then
        continue
    fi
    if ! git -C "$SC25_DIR" apply "$patch"; then
        echo "error: $patch does not apply to $SC25_DIR (other local changes?)" >&2
        exit 1
    fi
done

# venv on top of the frameworks env, adding ezpz, ruamel.yaml, tensorboard
# Lmod's scripts trip over `set -u`
set +u
module use /soft/modulefiles
module load "$FRAMEWORKS_MODULE"
set -u
BASE_PYTHON=$FRAMEWORKS_PREFIX/bin/python3
if ! "$BASE_PYTHON" -c 'import torch' 2>/dev/null; then
    echo "error: $BASE_PYTHON can't import torch; set FRAMEWORKS_PREFIX in env.sh" >&2
    echo "  (the conda env $FRAMEWORKS_MODULE activates: module show $FRAMEWORKS_MODULE)" >&2
    exit 1
fi
if [[ ! -x "$VENV/bin/python3" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
fi
if ! "$VENV/bin/python3" -c 'import torch' 2>/dev/null; then
    echo "error: $VENV was built on another python ($(grep '^home' "$VENV/pyvenv.cfg"));" >&2
    echo "  remove it and re-run: rm -rf $VENV && $0" >&2
    exit 1
fi
"$VENV/bin/python3" -m pip install --quiet "$EZPZ" ruamel.yaml tensorboard

mkdir -p "$LOGDIR"


echo "sc25-dl-tutorial at $(git -C "$SC25_DIR" rev-parse --short HEAD) (patched) in $SC25_DIR"
echo "venv in $VENV"
echo "logs will be written to $LOGDIR"

# Site settings shared by setup.sh, submit.sh and run_aurora.sh. Override
# any of these by exporting them before running a script (or with qsub -v).

EXAMPLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Python environment: frameworks module plus a small venv on top (made by setup.sh)
export FRAMEWORKS_MODULE=${FRAMEWORKS_MODULE:-frameworks/2026.1.0}
export FRAMEWORKS_PREFIX=${FRAMEWORKS_PREFIX:-/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0}
export VENV=${VENV:-$EXAMPLE_DIR/venv}

# Checkout of the training code (created by setup.sh)
export SC25_DIR=${SC25_DIR:-$EXAMPLE_DIR/sc25-dl-tutorial}

# ERA5 tutorial data (readable by all Aurora users)
export DATADIR=${DATADIR:-/lus/flare/projects/datasets/sc25_tutorial_data}

# Training outputs (console logs, TensorBoard events)
export LOGDIR=${LOGDIR:-$EXAMPLE_DIR/logs}


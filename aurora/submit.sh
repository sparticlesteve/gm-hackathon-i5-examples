#!/bin/bash -l
#PBS -q debug-scaling
#PBS -l select=4
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -N vit-weather
#PBS -j oe
# ==============================================================================
# SCRIPT: ViT training on ERA5 (Aurora)
# DESCRIPTION: Runs one or more configs from ViT_aurora.yaml. Submit from this
#              directory after running setup.sh.
# USAGE:       qsub -A <project> [-l select=N] [-v CONFIG=<config>] submit_aurora.sh
#   qsub -A <project> submit_aurora.sh                               # dp, 4 nodes
#   qsub -A <project> -l select=1 -q debug -v CONFIG=smoke submit_aurora.sh
#   qsub -A <project> -l select=16 -v CONFIG=dp_16node submit_aurora.sh
# The node count must match the config (see README.md). CONFIG may be a
# space-separated list, run in turn in the same allocation. Optional:
# EXTRA_ARGS="<train.py args>", plus any variable from env.sh.
# ==============================================================================

set -eo pipefail

CONFIG="${CONFIG:-dp}"

# PBS runs a copy of this script from its spool dir, so find the example
# directory from the submission directory.
EXAMPLE_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -f "$EXAMPLE_DIR/env.sh" ]]; then
    echo "error: submit from the aurora directory ($EXAMPLE_DIR has no env.sh)" >&2
    exit 1
fi
source "$EXAMPLE_DIR/env.sh"

if [[ ! -x "$VENV/bin/python3" || ! -d "$SC25_DIR" ]]; then
    echo "error: run ./setup.sh on a login node first" >&2
    exit 1
fi

module use /soft/modulefiles
module load "$FRAMEWORKS_MODULE"
source "$VENV/bin/activate"

status=0
for cfg in ${CONFIG}; do
    bash "$EXAMPLE_DIR/run_aurora.sh" "$cfg" ${EXTRA_ARGS:-} || status=$?
done
# Don't call `exit`: in this login shell it runs ~/.bash_logout, whose output
# (e.g. a failing `clear` with no TERM) can replace the job's exit status
[[ "$status" -eq 0 ]]

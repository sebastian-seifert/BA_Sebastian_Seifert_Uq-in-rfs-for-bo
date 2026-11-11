#!/bin/bash
# Meta-HPO of the proximity hyperparameters: 100 SMAC iterations, each evaluated as one
# Slurm array over the BBSubset-dev tasks (5 seeds). Runs in the foreground; use tmux.
#
#   slurm/meta_hpo.sh [--method a|b|ac|bc] [further options of scripts/run_meta_hpo.py]
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

exec "$PYTHON_BIN" scripts/run_meta_hpo.py --start-iteration 1 --end-iteration 100 \
    --seeds 5 --trials 100 "$@"

#!/bin/bash
# Submit one sweep to Slurm in array chunks.
#
#   slurm/submit.sh <sweep> [--sbatch] [scheduler options]
#
#   sweep      bbsubset-dev | bo-super | extrapolation | timing
#   --sbatch   run the scheduler inside a Slurm job (slurm/master.sbatch) instead of in
#              this shell (then keep the shell alive, e.g. in tmux)
#
# Remaining options go to scripts/slurm_scheduler.py, e.g. --dry-run --once.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

sweep="${1:?usage: slurm/submit.sh <bbsubset-dev|bo-super|extrapolation|timing> [--sbatch] [options]}"
shift
if [ "${1:-}" = "--sbatch" ]; then
    shift
    sbatch --job-name="master_${sweep}" slurm/master.sbatch "$sweep" "$@"
    exit 0
fi

case "$sweep" in
    bbsubset-dev)
        manifest=results/sweep_bbsubset_dev_big_comparison_fix/tasks_all.txt
        [ -f "$manifest" ] || "$PYTHON_BIN" scripts/make_bo_tasks.py
        opts=(--sbatch slurm/bo.sbatch --chunk-size 200 --max-tasks 4000 --batch-tasks 4000
              --trigger-tasks 10 --use-task-offset --job-prefix bo_uq_dev)
        ;;
    bo-super)
        manifest=results/sweep_bo_uq_alignment_super/tasks_all.txt
        [ -f "$manifest" ] || "$PYTHON_BIN" scripts/make_bo_tasks.py --five-methods
        opts=(--sbatch slurm/bo.sbatch --chunk-size 200 --max-tasks 4000 --batch-tasks 4000
              --trigger-tasks 10 --use-task-offset --job-prefix bo_uq_swp)
        ;;
    extrapolation)
        manifest=results/extrapolation_sweep_tasks.txt
        [ -f "$manifest" ] || "$PYTHON_BIN" scripts/make_extrapolation_tasks.py --no-metrics
        opts=(--sbatch slurm/extrapolation.sbatch --chunk-size 200 --max-tasks 4500
              --trigger-jobs 10 --job-prefix extrap_uq)
        ;;
    timing)
        manifest=results/sweep_bbob_timing/tasks.txt
        [ -f "$manifest" ] || "$PYTHON_BIN" scripts/make_timing_tasks.py
        opts=(--sbatch slurm/timing.sbatch --chunk-size 100 --max-tasks 1100
              --trigger-jobs 5 --job-prefix bbob_time)
        ;;
    *)
        echo "unknown sweep: $sweep" >&2
        exit 1
        ;;
esac

exec "$PYTHON_BIN" scripts/slurm_scheduler.py --manifest "$manifest" --concurrency 25 \
    --state-file "$(dirname "$manifest")/scheduler_state_${sweep}.json" "${opts[@]}" "$@"

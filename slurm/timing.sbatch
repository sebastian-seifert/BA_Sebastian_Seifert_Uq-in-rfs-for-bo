#!/bin/bash
#SBATCH --partition=ai
#SBATCH --nodelist=ai-n005
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
# One timed BO run per array task. All tasks run on the same node so timings are comparable.

source slurm/common.sh
run_task_line scripts/run_bo_timed.py

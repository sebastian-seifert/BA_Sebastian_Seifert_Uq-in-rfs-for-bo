#!/bin/bash
#SBATCH --partition=ai
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
# One BO run of a meta-HPO iteration. Submitted by scripts/run_meta_hpo.py with the
# iteration's task file as first argument.

source slurm/common.sh
TASK_FILE="$1"
run_task_line scripts/run_bo.py

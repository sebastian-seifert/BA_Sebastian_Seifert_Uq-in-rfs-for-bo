#!/bin/bash
#SBATCH --partition=ai
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# One BO run per array task (BBSubset-dev sweep and BO super-sweep).

source slurm/common.sh
run_task_line scripts/run_bo.py

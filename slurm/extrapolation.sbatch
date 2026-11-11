#!/bin/bash
#SBATCH --partition=ai
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
# One extrapolation UQ experiment per array task (task lines are full commands).

source slurm/common.sh
run_task_line

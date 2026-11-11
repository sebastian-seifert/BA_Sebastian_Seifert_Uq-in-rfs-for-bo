#!/bin/bash
#SBATCH --partition=ai
#SBATCH --job-name=dyrf_master
#SBATCH --output=logs/%x_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=120:00:00
# Runs the chunked scheduler for one sweep inside a long-lived job. Use slurm/submit.sh.

exec slurm/submit.sh "$@"

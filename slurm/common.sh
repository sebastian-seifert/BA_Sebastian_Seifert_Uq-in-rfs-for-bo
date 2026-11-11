# Shared environment for all Slurm jobs; sourced by the .sbatch files and submit.sh.
#
# The Python interpreter is taken from $DYRF_PYTHON, falling back to `python3` on PATH.
# Activate the project environment before submitting: sbatch exports it to the job.

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}" || exit 1

PYTHON_BIN="${DYRF_PYTHON:-python3}"

# Fixed hash seed: without it, SMAC's candidate ordering and therefore BO trajectories
# differ between runs with the same seed.
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

# One BLAS thread per task avoids oversubscription and keeps timings comparable.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p logs

# Run line (SLURM_ARRAY_TASK_ID + TASK_OFFSET) of $TASK_FILE.
# With a runner script, the line holds its arguments; without one, the line is a full command.
run_task_line() {
    local runner="${1:-}"
    local line_num=$((SLURM_ARRAY_TASK_ID + ${TASK_OFFSET:-0}))
    local line
    line=$(sed -n "${line_num}p" "$TASK_FILE")
    if [ -z "$line" ]; then
        echo "error: line ${line_num} of ${TASK_FILE} is empty" >&2
        exit 1
    fi
    echo "task ${line_num} of ${TASK_FILE}: ${line}"
    if [ -n "$runner" ]; then
        # shellcheck disable=SC2086  # the line is a list of arguments
        "$PYTHON_BIN" "$runner" $line
    else
        line="${line#python3 }"
        line="${line#python }"
        # shellcheck disable=SC2086
        "$PYTHON_BIN" $line
    fi
}

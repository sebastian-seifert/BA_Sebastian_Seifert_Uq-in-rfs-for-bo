# DyRF-BO

Code for the bachelor's thesis *Uncertainty Quantification in Random Forest Surrogates for
Bayesian Optimization* (Sebastian Seifert, Leibniz University Hannover, 2026).

Random forest surrogates, as used in SMAC, estimate predictive uncertainty from the variance
between trees. Axis-aligned splits make every tree constant outside the training support, so this
variance underestimates uncertainty exactly where exploration is needed. The thesis compares two
alternatives, both as standalone uncertainty estimators under controlled extrapolation and inside
Bayesian optimization (BO):

- **Proximity-based uncertainty quantification (PUQ).** Out-of-bag (or cross-validated) residuals
  of the training points are localized through the partitions learned by the forest, either by
  leaf co-occurrence `P_co` or by the tree-path proximity `P_walk(x, x') = exp(-λ · d(x, x'))`,
  where `d` is the path length between the two leaves. Approach A takes the residual quantiles
  of the k most similar points, approach B weights all residuals by proximity. The C variants
  additionally rescale the interval by a local density factor. In BO the lower residual quantile
  is used in the proximity lower confidence bound (PLCB) with an exploration floor ε.
- **Entropy-based epistemic uncertainty.** The entropy decomposition of Shaker & Hüllermeier
  (2020) is extended to regression forests by treating each tree as a Gaussian; the epistemic
  part is converted back to a standard deviation and used in a standard LCB.

## Methods

The optimizer configurations in [`configs/optimizer`](configs/optimizer) define the methods
compared in the BO experiments. The PLCB hyperparameters come from the meta-HPO described below
(thesis, Table 4.1).

| Optimizer ID | Uncertainty in BO | Acquisition | Effective hyperparameters |
| --- | --- | --- | --- |
| `SMAC20_ProximityA_LCB` | top-k residuals, leaf co-occurrence | PLCB | k = 28, ε = 0.081 |
| `SMAC20_ProximityB_LCB` | proximity-weighted residuals, tree path | PLCB | λ = 1.273, ε = 0.156 |
| `SMAC20_ProximityAC_LCB` | as A | PLCB | k = 7, ε = 0.093 |
| `SMAC20_ProximityBC_LCB` | as B | PLCB | λ = 1.494, ε = 0.143 |
| `SMAC20_Proximity{A,B,AC,BC}_LCB_CV` | as above, 5-fold CV residuals instead of OOB | PLCB | as above |
| `SMAC20_Entropy_LCB` | entropy-based epistemic std. | LCB, β = 3.8416 | – |
| `SMAC3_HPOFacade_lcb` | between-tree variance (SMAC default) | LCB, β = 3.8416 | – |
| `SMAC3_HPOFacade_ei` | between-tree variance (SMAC default) | EI | – |

PLCB does not apply the density factor, and A/AC use leaf co-occurrence, so the `decay_lambda`
of A/AC and the `alpha` of AC/BC in the config files have no effect in BO (see Section 4.3.1 of
the thesis). In BO, AC and BC are therefore second tuned configurations of A and B. The density
calibration is evaluated in the standalone extrapolation study.

All methods use SMAC's `HyperparameterOptimizationFacade` and its random forest; only the
uncertainty estimate and the acquisition function differ.

## Repository layout

```
configs/            Hydra configs for CARP-S: optimizers and the benchmark tasks used
src/dyrf_bo/
  uq/               proximity UQ, entropy UQ, and the extractor registry used by the surrogate
  bo/               SMAC surrogate, PLCB acquisition, CARP-S entry point and timed runner
  extrapolation/    extrapolation study: test functions, samplers, evaluator, metrics
  sweeps/           task-manifest generators and the Slurm array scheduler
  analysis/         gathering BO logs, scorecards, CARP-S reports, extrapolation and timing analysis
  meta_hpo/         SMAC-based tuning of the proximity hyperparameters and HyperSHAP analysis
  benchmarks/       registries of the CARP-S benchmark tasks
scripts/            command-line entry points (one per step, see below)
slurm/              Slurm job scripts and the sweep submission wrapper
results/            aggregated results reported in the thesis
```

## Installation

Python 3.10 or newer on Linux (verified with Python 3.12). The package versions used for that
verification are pinned in [`requirements-lock.txt`](requirements-lock.txt).

```bash
git clone <repository-url> DyRF-BO
cd DyRF-BO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
```

Optional extras:

```bash
pip install -e ".[benchmarks]"   # YAHPO Gym and HPOBench, needed for the real-world suites
pip install -e ".[explain]"      # HyperSHAP, needed for scripts/explain_meta_hpo.py
pip install -e ".[gpu]"          # CuPy for GPU-accelerated proximity and entropy computations
```

The BBOB and BBSubset tasks need no data. The YAHPO Gym and HPOBench suites need their surrogate
and tabular data; download them as described in the CARP-S documentation and point
`DYRF_BENCHMARK_DIR` at a directory containing `yahpo-data/` and `hpobench/`.

### Reproducibility

All entry points must run with a fixed hash seed:

```bash
export PYTHONHASHSEED=0
```

Without it, SMAC's internal ordering depends on Python's per-process hash randomization and two
runs with the same seed diverge after the initial design. The Slurm scripts set it (and pin BLAS
to one thread per task) in [`slurm/common.sh`](slurm/common.sh).

## Reproducing the experiments

Every step reads and writes below `results/` (manifests, logs, analysis) and `runs/` (raw CARP-S
run directories). All commands are run from the repository root.

### 1. Bayesian optimization benchmark

A single run, e.g. Proximity BC on one BBSubset-dev task:

```bash
python scripts/run_bo.py --config-dir configs \
    +task=subselection/blackbox/dev/subset_bbob_2_12_0 +optimizer=smac20_proximity_bc_lcb \
    task.optimization_resources.n_trials=100 seed=1 \
    optimizer_id=SMAC20_ProximityBC_LCB optimizer_container_id=SMAC20_ProximityBC_LCB \
    baserundir=runs/example
```

The full sweeps are generated with `scripts/make_bo_tasks.py` and submitted with
`slurm/submit.sh`, which keeps the number of queued array tasks below the cluster limit:

| Sweep | Command | Runs |
| --- | --- | --- |
| BBSubset-dev, all 11 methods | `slurm/submit.sh bbsubset-dev` | 20 tasks × 11 methods × 20 seeds = 4,400 |
| 5 main methods on the remaining suites¹ | `slurm/submit.sh bo-super` | 76,200 |

¹ BBSubset-test, BBOB in 2, 4, 8, 16, 32 and 128 dimensions, YAHPO rbv2_ranger and rbv2_super,
HPOBench ML; methods ProximityAC, ProximityB, Entropy, and the two SMAC baselines.

`slurm/submit.sh <sweep>` runs the scheduler in the current shell (use tmux);
`slurm/submit.sh <sweep> --sbatch` runs it as a long-lived Slurm job instead, and
`--dry-run --once` prints the first submission without submitting. Set `DYRF_PYTHON` to the
environment's interpreter if `python3` on the compute nodes is a different one.

Once the runs are finished, gather the trial logs, build the merged log table, and compute the
scorecards and CARP-S reports (ranks, critical-difference diagrams):

```bash
python scripts/analyze_bo.py --suites sweep_bbsubset_dev_big_comparison \
    --gather --merge --run-scorecards --run-carps
```

`--suites` accepts any of the suite names listed by `python scripts/analyze_bo.py --help`.

### 2. Extrapolation study

Uncertainty estimates of all approaches are evaluated on held-out points of standardized test
functions (sphere, Rosenbrock, Rastrigin, Ackley) for 8 dimensions, 6 training-set sizes,
4 random forest configurations and 30 seeds (23,040 runs). Each run writes per-point predictions
(Parquet) and a JSON summary of the metrics.

```bash
python scripts/run_extrapolation.py --dimension 8 --n-train 136 --function ackley \
    --strategy stratified --surrogate smac_default --seed 0    # single run
slurm/submit.sh extrapolation                                     # full sweep on Slurm
python scripts/run_extrapolation_local.py --workers 8             # or locally
```

Analysis:

```bash
python scripts/aggregate_extrapolation.py
python scripts/analyze_extrapolation_cube.py \
    --input-dir results/extrapolation_uq/raw --output-dir results/extrapolation_uq/analysis
python scripts/plot_extrapolation.py
```

The first command writes the scorecards, the second the stratified slices (`main_global*.csv`,
`stratum_*.csv`) that the plots are drawn from; all three use `results/extrapolation_uq/analysis`.

### 3. Runtime scaling

Wall-clock time and BO overhead of all 11 methods on BBOB function 8 (Rosenbrock, instance 0)
in 2 to 32 dimensions, 100 trials each (20 seeds, 1,100 runs). The job script pins all runs to
one node so timings are comparable; adjust `--nodelist` in [`slurm/timing.sbatch`](slurm/timing.sbatch) for another cluster.

```bash
slurm/submit.sh timing
python scripts/analyze_timing.py
python scripts/plot_timing.py
```

### 4. Hyperparameter tuning (meta-HPO)

The proximity hyperparameters were tuned with an outer SMAC loop: each of 100 iterations runs
the candidate configuration on the 18 BBSubset-dev tasks with 5 seeds and scores it by the mean
normalized regret. Reference bounds for the normalization come from earlier BBSubset-dev runs
(`results/meta_smac_proximity_hpo/reference_bounds.json`, recomputable with
`scripts/extract_reference_bounds.py`).

```bash
slurm/meta_hpo.sh --method a                         # a, b, ac or bc
python scripts/explain_meta_hpo.py --method a        # HyperSHAP interaction analysis
```

`scripts/run_meta_hpo.py --dry-run` replaces the Slurm evaluation by a mock objective to check
the loop locally.

  ## Results

  Aggregated evaluation summaries, metrics, and generated plots are stored in
  ￹[`results/`](results/):

  - **Extrapolation Analysis**:
  ￹[`results/extrapolation_uq/analysis/`](results/extrapolation_uq/analysis/)
  - **Runtime Scaling Analysis**:
  ￹[`results/sweep_bbob_timing/analysis/`](results/sweep_bbob_timing/analysis/)
  - **BO Benchmarks**: Scorecards and CARP-S reports are saved to `results/<suite>_merged/`   
  ￹and `results/global_summary_scorecard.md`.


## Citation

```bibtex
@thesis{seifert2026dyrfbo,
  author = {Sebastian Seifert},
  title  = {Uncertainty Quantification in Random Forest Surrogates for Bayesian Optimization},
  school = {Leibniz University Hannover},
  type   = {Bachelor's thesis},
  year   = {2026}
}
```

Reviewers: Prof. Dr. Marius Lindauer, Prof. Dr. Henning Wachsmuth. Supervisor: Leona Hennig.

## License

[MIT](LICENSE)

"""Outer SMAC loop that tunes the proximity hyperparameters on the BBSubset dev tasks.

Each iteration asks SMAC for a configuration, evaluates it on 18 dev tasks x 5 seeds as a Slurm array
(or with a synthetic loss for ``--dry-run``), and tells SMAC the meta-loss (see ``loss.py``). Progress is
checkpointed after every iteration so a run can be continued with ``--start-iteration``/``--resume``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory import StatusType, TrialInfo, TrialValue

from dyrf_bo.benchmarks.bbsubset import CarpsBBSubsetRegistry
from dyrf_bo.meta_hpo.loss import compute_meta_loss, load_dev_reference_bounds
from dyrf_bo.meta_hpo.space import build_initial_design_with_incumbent, create_proximity_meta_configspace
from dyrf_bo.meta_hpo.tasks import METHOD_OPTIMIZERS, generate_iteration_tasks


def parse_iteration_results(
    iter_dir: Path,
    iter_base_dir: Path,
    dev_tasks: list[str],
) -> dict[str, list[float]]:
    """Best cost of every finished run of one iteration, grouped by dev task.

    Runs are read from CARP-S ``trial_logs.jsonl`` files, then from SMAC ``runhistory.json`` files of runs
    not seen yet, and finally from ``telemetry_*.json`` files.
    """
    results: dict[str, list[float]] = {t: [] for t in dev_tasks}

    def _match_task(target_path: Path) -> str | None:
        path_str = str(target_path)
        for dt in dev_tasks:
            if dt in path_str:
                return dt
        hydra_cfg = target_path.parent / ".hydra" / "config.yaml"
        if hydra_cfg.exists():
            try:
                content = hydra_cfg.read_text()
                for dt in dev_tasks:
                    if dt in content:
                        return dt
            except Exception:
                pass
        return None

    found_runs: set = set()

    if iter_base_dir.exists():
        trial_log_files = sorted(list(iter_base_dir.glob("**/trial_logs.jsonl")))
        for tfile in trial_log_files:
            dt = _match_task(tfile)
            if dt is None:
                continue
            costs = []
            try:
                with open(tfile) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        trial_val = row.get("trial_value", {})
                        status = trial_val.get("status", 1)
                        if status in (1, "SUCCESS", "StatusType.SUCCESS"):
                            cost = trial_val.get("cost")
                            if cost is not None and np.isfinite(float(cost)):
                                costs.append(float(cost))
                if costs:
                    best_c = min(costs)
                    results[dt].append(best_c)
                    found_runs.add((dt, str(tfile.parent)))
            except Exception as e:
                print(f"Warning: Failed parsing {tfile}: {e}")

        runhistory_files = sorted(list(iter_base_dir.glob("**/runhistory.json")))
        for rh_file in runhistory_files:
            dt = _match_task(rh_file)
            if dt is None:
                continue
            run_key = (dt, str(rh_file.parent.parent))
            run_key_direct = (dt, str(rh_file.parent))
            if run_key in found_runs or run_key_direct in found_runs:
                continue
            try:
                with open(rh_file) as f:
                    rh_data = json.load(f)
                costs = [
                    float(entry["cost"])
                    for entry in rh_data.get("data", [])
                    if entry.get("status") in (1, "SUCCESS", "StatusType.SUCCESS")
                    and entry.get("cost") is not None
                    and np.isfinite(float(entry["cost"]))
                ]
                if costs:
                    best_c = min(costs)
                    results[dt].append(best_c)
                    found_runs.add(run_key_direct)
            except Exception as e:
                print(f"Warning: Failed parsing {rh_file}: {e}")

    telemetry_files = []
    if iter_dir.exists():
        telemetry_files.extend(list(iter_dir.glob("telemetry_*.json")))
    if iter_base_dir.exists():
        telemetry_files.extend(list(iter_base_dir.glob("**/telemetry_*.json")))

    for tfile in sorted(telemetry_files):
        dt = _match_task(tfile)
        if dt is None:
            continue
        try:
            with open(tfile) as f:
                tdata = json.load(f)
            inc_cost = tdata.get("cost_inc", tdata.get("best_cost"))
            if inc_cost is None and "trials" in tdata:
                valid_costs = [
                    float(tr.get("cost", tr.get("trial_value", {}).get("cost", np.nan)))
                    for tr in tdata["trials"]
                    if tr.get("cost", tr.get("trial_value", {}).get("cost")) is not None
                ]
                finite_costs = [c for c in valid_costs if np.isfinite(c)]
                if finite_costs:
                    inc_cost = min(finite_costs)
            if inc_cost is not None and np.isfinite(float(inc_cost)):
                results[dt].append(float(inc_cost))
        except Exception as e:
            print(f"Warning: Failed parsing {tfile}: {e}")

    return results


class MetaSmacOrchestrator:
    """Ask/tell loop over iterations ``start_iteration..end_iteration``."""

    def __init__(
        self,
        method: str = "a",
        start_iteration: int = 1,
        end_iteration: int = 100,
        seeds: int = 5,
        trials: int = 100,
        output_dir: str | None = None,
        baserundir: str | None = None,
        resume: bool = False,
        dry_run: bool = False,
        sbatch_script: str = "slurm/meta_hpo_step.sbatch",
    ) -> None:
        self.method = str(method).lower()
        self.start_iteration = start_iteration

        if self.method not in {"a", "b", "ac", "bc"}:
            raise ValueError(
                f"Unknown proximity method: '{self.method}'. Supported methods are 'a', 'b', 'ac', 'bc'."
            )

        self.end_iteration = end_iteration
        self.seeds = seeds
        self.trials = trials

        if output_dir is None:
            if self.method == "a":
                self.output_dir = Path("results/meta_smac_proximity_hpo")
            else:
                self.output_dir = Path(f"results/meta_smac_proximity_{self.method}_hpo")
        else:
            self.output_dir = Path(output_dir)

        if baserundir is None:
            if self.method == "a":
                self.baserundir = Path("runs/meta_smac_proximity_hpo")
            else:
                self.baserundir = Path(f"runs/meta_smac_proximity_{self.method}_hpo")
        else:
            self.baserundir = Path(baserundir)

        self.resume = resume
        self.dry_run = dry_run
        self.sbatch_script = sbatch_script

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baserundir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.output_dir / "meta_smac_checkpoint.json"
        self.leaderboard_file = self.output_dir / "meta_leaderboard.csv"
        self.best_config_file = self.output_dir / "best_config.json"

        self.ref_bounds = load_dev_reference_bounds()
        self.dev_tasks = [
            t.split("/")[-1] for t in CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)
        ]

        self.cs = create_proximity_meta_configspace(method=self.method)
        self.scenario = Scenario(
            configspace=self.cs,
            n_trials=self.end_iteration,
            deterministic=True,
            output_directory=self.output_dir / "smac3_internal",
        )
        self.initial_design = build_initial_design_with_incumbent(self.scenario)

        self.smac_opt = HyperparameterOptimizationFacade(
            scenario=self.scenario,
            target_function=lambda cfg, seed=0: 0.0,
            initial_design=self.initial_design,
            overwrite=not self.resume and not self.checkpoint_file.exists(),
        )

        self.history: list[dict[str, Any]] = []
        if self.resume or (self.start_iteration > 1 and self.checkpoint_file.exists()):
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load the history of earlier iterations and replay it into SMAC."""
        if not self.checkpoint_file.exists():
            return

        with open(self.checkpoint_file) as f:
            data = json.load(f)

        self.history = data.get("history", [])

        for entry in self.history:
            cfg_dict = {}
            for hp_name in self.cs.keys():
                if hp_name in entry:
                    val = entry[hp_name]
                    hp = self.cs[hp_name]
                    if "Integer" in type(hp).__name__:
                        cfg_dict[hp_name] = int(val)
                    else:
                        cfg_dict[hp_name] = float(val)
            cfg = Configuration(self.cs, values=cfg_dict)
            trial_info = TrialInfo(config=cfg, seed=self.scenario.seed)
            trial_value = TrialValue(cost=float(entry["loss_normalized_regret"]), status=StatusType.SUCCESS)
            try:
                self.smac_opt.tell(trial_info, trial_value, save=False)
            except Exception:
                pass

    def _save_checkpoint(self, last_iter: int) -> None:
        """Write checkpoint, best_config.json and meta_leaderboard.csv."""
        if not self.history:
            return

        sorted_history = sorted(self.history, key=lambda x: x["loss_normalized_regret"])
        best_entry = sorted_history[0]

        checkpoint_data = {
            "last_iteration": last_iter,
            "total_evaluated": len(self.history),
            "best_config": best_entry,
            "history": self.history,
        }

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        with open(self.best_config_file, "w") as f:
            json.dump(best_entry, f, indent=2)

        df = pd.DataFrame(self.history)
        df.to_csv(self.leaderboard_file, index=False)

    def _mock_evaluate(self, iteration: int, cfg: dict[str, Any]) -> dict[str, list[float]]:
        """Synthetic costs for ``--dry-run``: grows with the distance to the default configuration."""
        results: dict[str, list[float]] = {}
        err = 0.0
        if "k" in cfg:
            err += ((cfg["k"] - 25) / 25.0) ** 2
        if "decay_lambda" in cfg:
            err += ((cfg["decay_lambda"] - 1.345) / 1.345) ** 2
        if "eps" in cfg:
            target_eps = 0.1678 if self.method == "a" else 0.16
            err += ((cfg["eps"] - target_eps) / target_eps) ** 2
        if "alpha" in cfg:
            err += ((cfg["alpha"] - 1.0) / 1.0) ** 2

        dist_factor = 0.2 + 0.5 * err

        for task in self.dev_tasks:
            bounds = self.ref_bounds.get(task, {"min": 0.0, "max": 1.0})
            y_min = bounds["min"]
            y_max = bounds["max"]
            spread = max(1e-5, y_max - y_min)
            synth_cost = y_min + dist_factor * spread
            results[task] = [synth_cost] * self.seeds
        return results

    def _slurm_evaluate(self, iteration: int, task_file: str, iter_dir: Path) -> dict[str, list[float]]:
        """Submit the iteration's tasks as a Slurm array, wait for it, and collect the results."""
        num_tasks = 18 * self.seeds
        sbatch_cmd = [
            "sbatch",
            "--wait",
            "--parsable",
            f"--array=1-{num_tasks}%25",
            self.sbatch_script,
            str(task_file),
            str(iter_dir),
        ]
        print(f"[Iter {iteration:03d}] Submitting SLURM array: {' '.join(sbatch_cmd)}")
        subprocess.run(sbatch_cmd, check=True)

        iter_base_dir = self.baserundir / f"iter_{iteration:03d}"
        results = parse_iteration_results(iter_dir, iter_base_dir, self.dev_tasks)

        total_runs_found = sum(len(v) for v in results.values())
        expected_runs = len(self.dev_tasks) * self.seeds
        print(
            f"[Iter {iteration:03d}] Evaluation complete: gathered {total_runs_found}/{expected_runs} runs."
        )
        for dt, costs in results.items():
            if len(costs) < self.seeds:
                print(
                    f"[Iter {iteration:03d}] Warning: Task {dt} yielded {len(costs)}/{self.seeds} valid runs."
                )

        return results

    def run(self) -> dict[str, Any]:
        """Run the remaining iterations and return the best history entry."""
        print(
            f"Meta-HPO for proximity method {self.method.upper()}: iterations {self.start_iteration}-"
            f"{self.end_iteration}, {self.seeds} seeds, {self.trials} trials per task, dry_run={self.dry_run}"
        )
        opt_prefix = METHOD_OPTIMIZERS[self.method][1]

        for i in range(self.start_iteration, self.end_iteration + 1):
            iter_dir = self.output_dir / f"iter_{i:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            task_file = iter_dir / "tasks.txt"

            trial_info = self.smac_opt.ask()
            cfg_dict: dict[str, Any] = dict(trial_info.config)

            print(f"\n>>> Iteration {i:03d} / {self.end_iteration:03d}")
            param_repr = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in cfg_dict.items()
            )
            print(f"Proposed: {param_repr}")

            generate_iteration_tasks(
                config=cfg_dict,
                iteration=i,
                seeds=self.seeds,
                trials=self.trials,
                output_dir=str(self.output_dir),
                baserundir=str(self.baserundir),
                output_file=str(task_file),
                method=self.method,
            )

            if self.dry_run:
                task_results = self._mock_evaluate(i, cfg_dict)
            else:
                task_results = self._slurm_evaluate(i, str(task_file), iter_dir)

            meta_loss = compute_meta_loss(task_results, self.ref_bounds)
            print(f"Aggregate Meta-Loss: {meta_loss:.4f}")

            trial_value = TrialValue(cost=float(meta_loss), status=StatusType.SUCCESS)
            self.smac_opt.tell(trial_info, trial_value)

            entry = {
                "iteration": i,
                "config_id": f"cfg_{i:03d}",
                "optimizer_id": f"{opt_prefix}_iter{i:03d}",
                **cfg_dict,
                "loss_normalized_regret": float(meta_loss),
            }
            self.history.append(entry)
            self._save_checkpoint(i)

        sorted_history = sorted(self.history, key=lambda x: x["loss_normalized_regret"])
        best = sorted_history[0]
        best_repr = ", ".join(
            f"{k}={best[k]:.4f}" if isinstance(best[k], float) else f"{k}={best[k]}"
            for k in self.cs.keys()
            if k in best
        )
        print(f"\nBest configuration: {best_repr}")
        print(f"Best meta-loss: {best['loss_normalized_regret']:.4f}")
        print(f"Leaderboard: {self.leaderboard_file}")
        return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-HPO of the proximity hyperparameters with SMAC.")
    parser.add_argument(
        "--method", choices=["a", "b", "ac", "bc"], default="a", help="Proximity variant (default: a)"
    )
    parser.add_argument("--start-iteration", type=int, default=1, help="Start iteration (default: 1)")
    parser.add_argument("--end-iteration", type=int, default=100, help="End iteration (default: 100)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (default: 5)")
    parser.add_argument("--trials", type=int, default=100, help="Trials per task (default: 100)")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults based on method)")
    parser.add_argument("--baserundir", default=None, help="Hydra baserundir (defaults based on method)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Use a synthetic loss instead of Slurm runs")
    args = parser.parse_args()

    orchestrator = MetaSmacOrchestrator(
        method=args.method,
        start_iteration=args.start_iteration,
        end_iteration=args.end_iteration,
        seeds=args.seeds,
        trials=args.trials,
        output_dir=args.output_dir,
        baserundir=args.baserundir,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()

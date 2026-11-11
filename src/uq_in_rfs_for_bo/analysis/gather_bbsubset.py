"""Gathers BBSubset trial logs (dev and held-out test sweeps) into one table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Matched as substrings of run-path parts; later entries win, so _CV variants come after their base.
VALID_OPTIMIZER_IDS = [
    "SMAC3_HPOFacade_lcb",
    "SMAC3_HPOFacade_ei",
    "SMAC20_Entropy_LCB",
    "SMAC20_ProximityA_LCB",
    "SMAC20_ProximityA_LCB_CV",
    "SMAC20_ProximityB_LCB",
    "SMAC20_ProximityB_LCB_CV",
    "SMAC20_ProximityAC_LCB",
    "SMAC20_ProximityAC_LCB_CV",
    "SMAC20_ProximityBC_LCB",
    "SMAC20_ProximityBC_LCB_CV",
]


def gather_trial_logs(runs_base: Path, outdir: Path) -> pd.DataFrame:
    """Collects all BBSubset run logs under `runs_base` into `outdir/logs.{csv,parquet}`."""
    trial_log_files = list(runs_base.glob("**/trial_logs.jsonl"))
    records = []

    if trial_log_files:
        for t_file in trial_log_files:
            run_dir = t_file.parent
            hydra_cfg_file = run_dir / ".hydra" / "config.yaml"

            opt_id = "unknown"
            task_id = "unknown"
            seed = 1

            if hydra_cfg_file.exists():
                try:
                    import yaml

                    with open(hydra_cfg_file) as f:
                        cfg_data = yaml.safe_load(f)
                    if isinstance(cfg_data, dict):
                        opt_id = cfg_data.get("optimizer_id", opt_id)
                        task_id = cfg_data.get("task", {}).get("name", task_id)
                        seed = cfg_data.get("seed", seed)
                except Exception:
                    pass

            # Infer optimizer and task from the run path if the Hydra config is missing.
            if opt_id == "unknown" or task_id == "unknown":
                parts = run_dir.parts
                for p in parts:
                    for v_opt in VALID_OPTIMIZER_IDS:
                        if v_opt in p:
                            opt_id = v_opt
                    if "subset_" in p:
                        task_id = p

            try:
                with open(t_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        records.append(
                            {
                                "task_id": task_id,
                                "optimizer_id": opt_id,
                                "seed": seed,
                                "n_trials": entry.get("trial", entry.get("n_trials", 1)),
                                "trial_value__cost": entry.get(
                                    "cost", entry.get("trial_value__cost", np.nan)
                                ),
                            }
                        )
            except Exception:
                continue

    # Fall back to telemetry_*.json files when no trial_logs.jsonl were found.
    telemetry_files = list(outdir.glob("telemetry_*.json")) + list(runs_base.glob("**/telemetry_*.json"))
    if telemetry_files and not records:
        print(f"Parsing {len(telemetry_files)} telemetry JSON files...")
        for tf in telemetry_files:
            try:
                with open(tf) as f:
                    data = json.load(f)
                opt_id = data.get("optimizer_id", "unknown")
                task_id = data.get("task_id", "unknown")
                seed = data.get("seed", 1)
                for tr in data.get("trials", []):
                    records.append(
                        {
                            "task_id": task_id,
                            "optimizer_id": opt_id,
                            "seed": seed,
                            "n_trials": tr.get("trial", 1),
                            "trial_value__cost": tr.get("cost", np.nan),
                        }
                    )
            except Exception:
                continue

    if not records:
        print(f"No execution records found in {runs_base} or {outdir}.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    grouper = ["task_id", "optimizer_id", "seed"]
    df = df.sort_values(by=grouper + ["n_trials"]).reset_index(drop=True)
    df["trial_value__cost_inc"] = df.groupby(grouper)["trial_value__cost"].cummin()

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "logs.csv", index=False)
    df.to_parquet(outdir / "logs.parquet", index=False)
    print(
        f"Saved {len(df)} records across {df['optimizer_id'].nunique()} optimizers to {outdir / 'logs.parquet'}"
    )
    return df

"""Task lines for one meta-HPO iteration: one proposed configuration on the 18 dev tasks x seeds."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dyrf_bo.benchmarks.bbsubset import CarpsBBSubsetRegistry

# method -> (optimizer config, optimizer id prefix, uncertainty_func)
METHOD_OPTIMIZERS: dict[str, tuple[str, str, str]] = {
    "a": ("smac20_proximity_lcb", "SMAC20_ProximityLCB", "proximity_b"),
    "b": ("smac20_proximity_b_lcb", "SMAC20_ProximityB_LCB", "proximity_b"),
    "ac": ("smac20_proximity_ac_lcb", "SMAC20_ProximityAC_LCB", "proximity_ac"),
    "bc": ("smac20_proximity_bc_lcb", "SMAC20_ProximityBC_LCB", "proximity_bc"),
}


def generate_iteration_tasks(
    config: dict[str, Any],
    iteration: int,
    seeds: int | Sequence[int] = 5,
    trials: int = 100,
    alpha: float = 0.05,
    output_dir: str = "results/meta_smac_proximity_hpo",
    baserundir: str = "runs/meta_smac_proximity_hpo",
    output_file: str | None = None,
    method: str = "a",
) -> list[str]:
    """Hydra argument lines for evaluating ``config`` in iteration ``iteration``.

    ``config`` holds the hyperparameters of the method's search space (see ``space.py``); ``alpha`` is
    the LCB risk level (``level = 1 - alpha``). An int ``seeds`` means seeds 1..seeds. If
    ``output_file`` is given, the lines are also written there.
    """
    m = method.lower()
    if m not in METHOD_OPTIMIZERS:
        raise ValueError(f"Unknown method '{m}'. Supported methods are 'a', 'b', 'ac', 'bc'.")
    opt_name, container_id, uncertainty_func = METHOD_OPTIMIZERS[m]
    opt_id = f"{container_id}_iter{iteration:03d}"

    seeds_list = list(range(1, seeds + 1)) if isinstance(seeds, int) else [int(s) for s in seeds]

    overrides: list[str] = []
    if m in ("a", "ac"):
        k_val = int(config["k"])
        overrides.append(f"++optimizer.acq_func_kwargs.k={k_val}")
        if m == "ac":
            overrides.append(f"++optimizer.acq_func_kwargs.k_warmup={k_val}")
    overrides += [
        f"++optimizer.acq_func_kwargs.level={round(1.0 - alpha, 4)}",
        f"++optimizer.acq_func_kwargs.eps={float(config['eps'])}",
        f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={uncertainty_func}",
        f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda={float(config['decay_lambda'])}",
    ]
    if m in ("ac", "bc"):
        overrides.append(
            f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.alpha={float(config.get('alpha', 1.0))}"
        )
    override_str = " ".join(overrides)

    iter_run_dir = f"{output_dir}/iter_{iteration:03d}"
    iter_base_dir = f"{baserundir}/iter_{iteration:03d}"

    lines: list[str] = []
    for task_arg in CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True):
        task_name = task_arg.split("/")[-1]
        task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg
        for seed in seeds_list:
            telemetry = f"{iter_run_dir}/telemetry_{opt_id}_{task_name}_seed{seed}.json"
            lines.append(
                f"--config-dir configs "
                f"+optimizer={opt_name} "
                f"{override_str} "
                f"{task_cmd} task.optimization_resources.n_trials={trials} "
                f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                f"baserundir={iter_base_dir} "
                f"optimizer_id={opt_id} optimizer_container_id={container_id}"
            )

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for line in lines:
                f.write(line + "\n")

    return lines

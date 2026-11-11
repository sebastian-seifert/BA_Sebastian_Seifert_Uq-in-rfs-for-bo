"""Run a single CARP-S optimization task with the thesis surrogates and acquisition functions.

CARP-S's SMAC3 optimizer can neither build a custom surrogate model from a config nor use the
proximity lower-bound acquisition. `apply_patches` extends it to read `acq_func_name`,
`acq_func_kwargs`, `smac_cfg.model_class` and `smac_cfg.model_kwargs` from the optimizer config,
and adds a few robustness fixes for logging and config resolution. `main` is CARP-S's Hydra entry
point, so all Hydra overrides are passed through unchanged.

Set ``DYRF_BENCHMARK_DIR`` to a directory with ``hpobench/`` and ``yahpo-data/`` subfolders to run
the real-world benchmarks from a shared data location (e.g. on the cluster).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ConfigSpace
import omegaconf
from omegaconf import OmegaConf

UNKNOWN_KEYS = ("benchmark_id", "task_id", "optimizer_id", "optimizer_container_id")


def _configure_benchmark_data() -> None:
    data_dir = os.environ.get("DYRF_BENCHMARK_DIR")
    if not data_dir or not os.path.exists(data_dir):
        return
    os.environ["CARPS_TASK_DATA_DIR"] = data_dir

    import carps.objective_functions.yahpo
    import hpobench
    import yahpo_gym

    hpobench.config_file.data_dir = Path(data_dir, "hpobench")
    carps.objective_functions.yahpo.YAHPO_TASK_DATA_DIR = Path(data_dir, "yahpo-data")
    try:
        yahpo_gym.local_config.init_config(data_path=str(Path(data_dir, "yahpo-data")))
    except Exception:
        pass


def _patch_configspace() -> None:
    """Restore `_sort_hyperparameters`, which newer ConfigSpace versions removed."""
    if hasattr(ConfigSpace.ConfigurationSpace, "_sort_hyperparameters"):
        return

    def _sort_hyperparameters(self):
        return list(self.values()) if hasattr(self, "values") else self.get_hyperparameters()

    ConfigSpace.ConfigurationSpace._sort_hyperparameters = _sort_hyperparameters


def _patch_file_logger() -> None:
    """Create missing parent directories for the environment log and keep `.hydra` on overwrite."""
    import carps.loggers.abstract_logger
    import carps.loggers.file_logger
    import carps.utils.loggingutils
    from carps.loggers.file_logger import get_run_directory, logger

    log_python_env = carps.utils.loggingutils.log_python_env

    def log_python_env_mkdir(log_file="env_log.txt"):
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_python_env(log_file=log_file)

    def file_logger_init(self, overwrite: bool = False, directory=None):
        carps.loggers.abstract_logger.AbstractLogger.__init__(self)
        directory = Path(directory) if directory is not None else get_run_directory()
        assert directory is not None, (
            "Directory must be specified in FileLogger or hydra run dir must be available."
        )
        self.directory = directory

        filename = getattr(self, "_filename", "trial_logs.jsonl")
        if (directory / filename).is_file():
            if not overwrite:
                raise RuntimeError(
                    f"Found previous run at '{directory}'. Stopping run. If you want to overwrite, specify "
                    f"overwrite for the file logger in the config (CARP-S/carps/configs/logger.yaml)."
                )
            logger.info(f"Found previous run. Removing '{directory}'.")
            for root, _dirs, files in os.walk(directory):
                for name in files:
                    path = Path(root) / name
                    if ".hydra" not in str(path):
                        try:
                            path.unlink()
                            logger.debug(f"Removed {path}")
                        except OSError:
                            pass

        carps.utils.loggingutils.log_python_env(log_file=Path(directory) / "env_info.txt")

    carps.utils.loggingutils.log_python_env = log_python_env_mkdir
    carps.loggers.file_logger.FileLogger.__init__ = file_logger_init


def _patch_argparse() -> None:
    """Ignore help-string validation errors raised for Hydra's lazy completion help on Python 3.14."""
    if not hasattr(argparse.ArgumentParser, "_check_help"):
        return
    check_help = argparse.ArgumentParser._check_help

    def lenient_check_help(self, action):
        try:
            check_help(self, action)
        except Exception:
            pass

    argparse.ArgumentParser._check_help = lenient_check_help


def _patch_omegaconf() -> None:
    """Fall back to placeholder values instead of failing on unresolved mandatory keys."""
    select = OmegaConf.select
    to_container = OmegaConf.to_container

    def lenient_select(cfg, key, *args, **kwargs):
        try:
            return select(cfg, key, *args, **kwargs)
        except Exception:
            if key in ("hydra.run.dir", "hydra.sweep.dir", "hydra.sweep.subdir"):
                return "runs/unknown_optimizer/unknown_benchmark/unknown_task/1"
            if key == "conda_env_name":
                return "carps_env"
            if key in UNKNOWN_KEYS:
                return f"unknown_{key}"
            raise

    def lenient_to_container(cfg, resolve=True, **kwargs):
        try:
            return to_container(cfg, resolve=resolve, **kwargs)
        except Exception:
            for key in UNKNOWN_KEYS:
                try:
                    value = OmegaConf.select(cfg, key)
                    if value is None or value == "???":
                        setattr(cfg, key, f"unknown_{key}")
                except Exception:
                    setattr(cfg, key, f"unknown_{key}")
            try:
                value = OmegaConf.select(cfg, "conda_env_name")
                if value is None or value == "???":
                    cfg.conda_env_name = "carps_env"
            except Exception:
                cfg.conda_env_name = "carps_env"
            try:
                return to_container(cfg, resolve=resolve, **kwargs)
            except Exception:
                return to_container(cfg, resolve=False, **kwargs)

    OmegaConf.select = lenient_select
    OmegaConf.to_container = lenient_to_container


def _build_acquisition_function(name: str, kwargs: dict):
    """Instantiate the acquisition function named in the optimizer config (None if unknown)."""
    name = str(name).lower()
    if name in ("proximity_lcb", "proximity_lower_bound"):
        from dyrf_bo.bo.acquisition import ProximityLowerBoundAcquisition

        return ProximityLowerBoundAcquisition(**kwargs)

    import smac.acquisition.function as acquisition

    acq_cls = {"ei": acquisition.EI, "pi": acquisition.PI, "lcb": acquisition.LCB}.get(name)
    if acq_cls is None:
        return None
    if name == "lcb" and "kappa" in kwargs:
        kwargs["beta"] = float(kwargs.pop("kappa")) ** 2
        kwargs["update_beta"] = False
    return acq_cls(**kwargs)


def _patch_smac3_optimizer() -> None:
    """Let CARP-S's SMAC3 optimizer use custom acquisition functions and surrogate models."""
    import carps.optimizers.smac20
    from carps.optimizers.smac20 import SMAC3Optimizer, maybe_inst_add_scenario
    from hydra.utils import get_class
    from smac.scenario import Scenario

    init = SMAC3Optimizer.__init__

    def init_with_acquisition(
        self,
        task,
        smac_cfg,
        loggers=None,
        expects_multiple_objectives=False,
        expects_fidelities=False,
        **kwargs,
    ):
        for key in ("acq_func_name", "acq_func_kwargs", "method"):
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
        init(
            self,
            task=task,
            smac_cfg=smac_cfg,
            loggers=loggers,
            expects_multiple_objectives=expects_multiple_objectives,
            expects_fidelities=expects_fidelities,
        )

    def setup_optimizer(self):
        smac_class = get_class(self.smac_cfg.smac_class)
        if smac_class == get_class("smac.facade.multi_fidelity_facade.MultiFidelityFacade"):
            self.fidelity_enabled = True

        smac_kwargs = {}
        if self.smac_cfg.smac_kwargs is not None:
            smac_kwargs = OmegaConf.to_container(self.smac_cfg.smac_kwargs, resolve=True, enum_to_str=True)

        acq_name = getattr(self, "acq_func_name", None) or getattr(self.smac_cfg, "acq_func_name", None)
        acq_kwargs = {}
        if getattr(self, "acq_func_kwargs", None):
            if isinstance(self.acq_func_kwargs, omegaconf.DictConfig):
                acq_kwargs = OmegaConf.to_container(self.acq_func_kwargs, resolve=True)
            else:
                acq_kwargs = dict(self.acq_func_kwargs)
        elif getattr(self.smac_cfg, "acq_func_kwargs", None):
            acq_kwargs = OmegaConf.to_container(self.smac_cfg.acq_func_kwargs, resolve=True)
        if getattr(self, "method", None) and "method" not in acq_kwargs:
            acq_kwargs["method"] = self.method

        if "acquisition_function" not in smac_kwargs and acq_name:
            acquisition_function = _build_acquisition_function(acq_name, acq_kwargs)
            if acquisition_function is not None:
                smac_kwargs["acquisition_function"] = acquisition_function

        scenario_kwargs = {"configspace": self.configspace}
        scenario_kwargs.update(OmegaConf.to_container(self.smac_cfg.scenario, resolve=True))
        scenario = Scenario(**scenario_kwargs)
        smac_kwargs["scenario"] = scenario

        if "model" not in smac_kwargs and getattr(self.smac_cfg, "model_class", None):
            model_kwargs = {}
            if getattr(self.smac_cfg, "model_kwargs", None) is not None:
                model_kwargs = OmegaConf.to_container(self.smac_cfg.model_kwargs, resolve=True)
            smac_kwargs["model"] = get_class(self.smac_cfg.model_class)(
                configspace=self.configspace, seed=scenario.seed, **model_kwargs
            )

        if "callbacks" not in smac_kwargs:
            smac_kwargs["callbacks"] = []
        elif isinstance(smac_kwargs["callbacks"], dict):
            smac_kwargs["callbacks"] = list(smac_kwargs["callbacks"].values())

        smac_kwargs = maybe_inst_add_scenario(smac_kwargs, "intensifier", scenario)

        if "acquisition_function" in smac_kwargs and "acquisition_maximizer" in smac_kwargs:
            maximizer = smac_kwargs["acquisition_maximizer"](
                configspace=self.configspace, acquisition_function=smac_kwargs["acquisition_function"]
            )
            smac_kwargs["acquisition_maximizer"] = maximizer
            if hasattr(maximizer, "selector") and hasattr(maximizer.selector, "expl2callback"):
                smac_kwargs["callbacks"].append(maximizer.selector.expl2callback)

        for key in ("config_selector", "initial_design", "multi_objective_algorithm"):
            smac_kwargs = maybe_inst_add_scenario(smac_kwargs, key, scenario)

        return smac_class(target_function=self.target_function, **smac_kwargs)

    carps.optimizers.smac20.SMAC3Optimizer.__init__ = init_with_acquisition
    carps.optimizers.smac20.SMAC3Optimizer._setup_optimizer = setup_optimizer


def apply_patches() -> None:
    _configure_benchmark_data()
    _patch_configspace()
    _patch_file_logger()
    _patch_argparse()
    _patch_omegaconf()
    _patch_smac3_optimizer()


def main() -> None:
    apply_patches()
    from carps.run import main as carps_main

    carps_main()


if __name__ == "__main__":
    main()

"""Meta-HPO search spaces for the four proximity variants and an incumbent-first initial design."""

from __future__ import annotations

from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.scenario import Scenario


def create_proximity_meta_configspace(method: str = "a", seed: int = 42) -> ConfigurationSpace:
    """Search space for proximity variant ``method``.

    - 'a':  k, decay_lambda, eps (eps default 0.1678)
    - 'b':  decay_lambda, eps (eps default 0.16)
    - 'ac': k, decay_lambda, eps, alpha
    - 'bc': decay_lambda, eps, alpha

    The defaults (k=25, decay_lambda=1.345, eps, alpha=1.0) are the configuration evaluated first.
    """
    m = method.lower()
    cs = ConfigurationSpace(name=f"proximity_lcb_meta_space_{m}", seed=seed)

    if m == "a":
        k = Integer("k", bounds=(5, 30), default=25)
        decay_lambda = Float("decay_lambda", bounds=(0.2, 2.0), default=1.345)
        eps = Float("eps", bounds=(0.02, 0.20), default=0.1678)
        cs.add([k, decay_lambda, eps])
    elif m == "b":
        decay_lambda = Float("decay_lambda", bounds=(0.2, 2.0), default=1.345)
        eps = Float("eps", bounds=(0.02, 0.20), default=0.16)
        cs.add([decay_lambda, eps])
    elif m == "ac":
        k = Integer("k", bounds=(5, 30), default=25)
        decay_lambda = Float("decay_lambda", bounds=(0.2, 2.0), default=1.345)
        eps = Float("eps", bounds=(0.02, 0.20), default=0.16)
        alpha = Float("alpha", bounds=(0.1, 2.0), default=1.0)
        cs.add([k, decay_lambda, eps, alpha])
    elif m == "bc":
        decay_lambda = Float("decay_lambda", bounds=(0.2, 2.0), default=1.345)
        eps = Float("eps", bounds=(0.02, 0.20), default=0.16)
        alpha = Float("alpha", bounds=(0.1, 2.0), default=1.0)
        cs.add([decay_lambda, eps, alpha])
    else:
        raise ValueError(f"Unknown proximity method: '{method}'. Supported methods are 'a', 'b', 'ac', 'bc'.")

    return cs


class IncumbentFirstInitialDesign(SobolInitialDesign):
    """Sobol initial design whose first configuration is the space's default configuration."""

    def select_configurations(self) -> list[Configuration]:
        configs = super().select_configurations()
        incumbent = self._configspace.get_default_configuration()
        filtered = [c for c in configs if c != incumbent]
        return ([incumbent] + filtered)[: len(configs)]


def build_initial_design_with_incumbent(
    scenario: Scenario, n_configs: int | None = 10, seed: int | None = None
) -> IncumbentFirstInitialDesign:
    return IncumbentFirstInitialDesign(scenario=scenario, n_configs=n_configs, seed=seed or scenario.seed)

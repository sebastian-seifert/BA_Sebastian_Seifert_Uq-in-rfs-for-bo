"""BBOB task registry (CARP-S ``BBOB/cfg_<dim>_<fid>_<iid>`` tasks) and function metadata."""

from __future__ import annotations

from enum import Enum


class BBOBFunctionCategory(str, Enum):
    """BBOB function groups as defined by the COCO benchmark."""

    SEPARABLE = "separable"
    MODERATE_CONDITIONING = "moderate_conditioning"
    ILL_CONDITIONED = "ill_conditioned"
    MULTIMODAL_ADEQUATE_GLOBAL = "multimodal_adequate_global"
    MULTIMODAL_WEAK_GLOBAL = "multimodal_weak_global"


BBOB_FUNCTION_METADATA: dict[int, tuple[str, BBOBFunctionCategory]] = {
    1: ("Sphere", BBOBFunctionCategory.SEPARABLE),
    2: ("Ellipsoidal", BBOBFunctionCategory.SEPARABLE),
    3: ("Rastrigin", BBOBFunctionCategory.SEPARABLE),
    4: ("Buche-Rastrigin", BBOBFunctionCategory.SEPARABLE),
    5: ("Linear Slope", BBOBFunctionCategory.SEPARABLE),
    6: ("Attractive Sector", BBOBFunctionCategory.MODERATE_CONDITIONING),
    7: ("Step Ellipsoidal", BBOBFunctionCategory.MODERATE_CONDITIONING),
    8: ("Rosenbrock", BBOBFunctionCategory.MODERATE_CONDITIONING),
    9: ("Rosenbrock Rotated", BBOBFunctionCategory.MODERATE_CONDITIONING),
    10: ("Ellipsoidal High Cond", BBOBFunctionCategory.ILL_CONDITIONED),
    11: ("Discus", BBOBFunctionCategory.ILL_CONDITIONED),
    12: ("Bent Cigar", BBOBFunctionCategory.ILL_CONDITIONED),
    13: ("Sharp Ridge", BBOBFunctionCategory.ILL_CONDITIONED),
    14: ("Different Powers", BBOBFunctionCategory.ILL_CONDITIONED),
    15: ("Rastrigin Multi", BBOBFunctionCategory.MULTIMODAL_ADEQUATE_GLOBAL),
    16: ("Weierstrass", BBOBFunctionCategory.MULTIMODAL_ADEQUATE_GLOBAL),
    17: ("Schaffer F7 Cond 10", BBOBFunctionCategory.MULTIMODAL_ADEQUATE_GLOBAL),
    18: ("Schaffer F7 Cond 1000", BBOBFunctionCategory.MULTIMODAL_ADEQUATE_GLOBAL),
    19: ("Griewank-Rosenbrock", BBOBFunctionCategory.MULTIMODAL_ADEQUATE_GLOBAL),
    20: ("Schwefel", BBOBFunctionCategory.MULTIMODAL_WEAK_GLOBAL),
    21: ("Gallagher 101 Peaks", BBOBFunctionCategory.MULTIMODAL_WEAK_GLOBAL),
    22: ("Gallagher 21 Peaks", BBOBFunctionCategory.MULTIMODAL_WEAK_GLOBAL),
    23: ("Katsuura", BBOBFunctionCategory.MULTIMODAL_WEAK_GLOBAL),
    24: ("Lunacek bi-Rastrigin", BBOBFunctionCategory.MULTIMODAL_WEAK_GLOBAL),
}


class CarpsBBOBFullRegistry:
    """Hydra task arguments for the 24 BBOB functions x 3 instances per dimension."""

    DIMENSIONS: tuple[int, ...] = (2, 4, 8, 16, 32, 128)
    NUM_FUNCTIONS: int = 24
    NUM_INSTANCES: int = 3

    @classmethod
    def get_tasks_by_dimension(cls, dim: int) -> list[str]:
        """Return the 72 task arguments for one dimension."""
        if dim not in cls.DIMENSIONS:
            raise ValueError(f"Unsupported dimension {dim}. Supported dimensions: {cls.DIMENSIONS}")
        return [
            f"+task=BBOB/cfg_{dim}_{fid}_{iid}"
            for fid in range(1, cls.NUM_FUNCTIONS + 1)
            for iid in range(cls.NUM_INSTANCES)
        ]

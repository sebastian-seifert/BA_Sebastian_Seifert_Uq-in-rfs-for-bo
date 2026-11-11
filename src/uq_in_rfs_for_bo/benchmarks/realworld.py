"""Task lists for the YAHPO rbv2_ranger, rbv2_super and HPOBench tabular-ML suites."""

from __future__ import annotations

import os


class CarpsRealworldRegistry:
    """Task arguments for the real-world suites.

    Task names are read from the installed CARP-S configs when the expected number is found there,
    otherwise the hard-coded lists below are used.
    """

    # fmt: off
    _CANONICAL_RANGER = [
        "cfg_rbv2_ranger_1040", "cfg_rbv2_ranger_1049", "cfg_rbv2_ranger_1050", "cfg_rbv2_ranger_1053",
        "cfg_rbv2_ranger_1056", "cfg_rbv2_ranger_1063", "cfg_rbv2_ranger_1067", "cfg_rbv2_ranger_1068",
        "cfg_rbv2_ranger_11", "cfg_rbv2_ranger_1111", "cfg_rbv2_ranger_1112", "cfg_rbv2_ranger_1114",
        "cfg_rbv2_ranger_1116", "cfg_rbv2_ranger_1119", "cfg_rbv2_ranger_1120", "cfg_rbv2_ranger_1128",
        "cfg_rbv2_ranger_1130", "cfg_rbv2_ranger_1134", "cfg_rbv2_ranger_1138", "cfg_rbv2_ranger_1142",
        "cfg_rbv2_ranger_1146", "cfg_rbv2_ranger_1169", "cfg_rbv2_ranger_12", "cfg_rbv2_ranger_1220",
        "cfg_rbv2_ranger_14", "cfg_rbv2_ranger_1457", "cfg_rbv2_ranger_1461", "cfg_rbv2_ranger_1462",
        "cfg_rbv2_ranger_1464", "cfg_rbv2_ranger_1468", "cfg_rbv2_ranger_1475", "cfg_rbv2_ranger_1478",
        "cfg_rbv2_ranger_1479", "cfg_rbv2_ranger_1480", "cfg_rbv2_ranger_1485", "cfg_rbv2_ranger_1486",
        "cfg_rbv2_ranger_1487", "cfg_rbv2_ranger_1489", "cfg_rbv2_ranger_1494", "cfg_rbv2_ranger_15",
        "cfg_rbv2_ranger_1501", "cfg_rbv2_ranger_1510", "cfg_rbv2_ranger_16", "cfg_rbv2_ranger_18",
        "cfg_rbv2_ranger_181", "cfg_rbv2_ranger_182", "cfg_rbv2_ranger_188", "cfg_rbv2_ranger_22",
        "cfg_rbv2_ranger_23", "cfg_rbv2_ranger_23381", "cfg_rbv2_ranger_23512", "cfg_rbv2_ranger_23517",
        "cfg_rbv2_ranger_24", "cfg_rbv2_ranger_28", "cfg_rbv2_ranger_29", "cfg_rbv2_ranger_3",
        "cfg_rbv2_ranger_30", "cfg_rbv2_ranger_300", "cfg_rbv2_ranger_307", "cfg_rbv2_ranger_31",
        "cfg_rbv2_ranger_312", "cfg_rbv2_ranger_32", "cfg_rbv2_ranger_37", "cfg_rbv2_ranger_375",
        "cfg_rbv2_ranger_377", "cfg_rbv2_ranger_38", "cfg_rbv2_ranger_40496", "cfg_rbv2_ranger_40498",
        "cfg_rbv2_ranger_40499", "cfg_rbv2_ranger_40536", "cfg_rbv2_ranger_40668", "cfg_rbv2_ranger_40670",
        "cfg_rbv2_ranger_40701", "cfg_rbv2_ranger_40927", "cfg_rbv2_ranger_40966", "cfg_rbv2_ranger_40975",
        "cfg_rbv2_ranger_40979", "cfg_rbv2_ranger_40981", "cfg_rbv2_ranger_40982", "cfg_rbv2_ranger_40983",
        "cfg_rbv2_ranger_40984", "cfg_rbv2_ranger_40994", "cfg_rbv2_ranger_41138", "cfg_rbv2_ranger_41142",
        "cfg_rbv2_ranger_41143", "cfg_rbv2_ranger_41146", "cfg_rbv2_ranger_41150", "cfg_rbv2_ranger_41156",
        "cfg_rbv2_ranger_41157", "cfg_rbv2_ranger_41159", "cfg_rbv2_ranger_41161", "cfg_rbv2_ranger_41162",
        "cfg_rbv2_ranger_4134", "cfg_rbv2_ranger_4135", "cfg_rbv2_ranger_4154", "cfg_rbv2_ranger_42",
        "cfg_rbv2_ranger_44", "cfg_rbv2_ranger_4534", "cfg_rbv2_ranger_4538", "cfg_rbv2_ranger_458",
        "cfg_rbv2_ranger_46", "cfg_rbv2_ranger_469", "cfg_rbv2_ranger_470", "cfg_rbv2_ranger_50",
        "cfg_rbv2_ranger_54", "cfg_rbv2_ranger_554", "cfg_rbv2_ranger_6", "cfg_rbv2_ranger_60",
        "cfg_rbv2_ranger_6332",
    ]

    _CANONICAL_SUPER = [
        "cfg_rbv2_super_1040", "cfg_rbv2_super_1049", "cfg_rbv2_super_1050", "cfg_rbv2_super_1053",
        "cfg_rbv2_super_1056", "cfg_rbv2_super_1063", "cfg_rbv2_super_1067", "cfg_rbv2_super_1068",
        "cfg_rbv2_super_11", "cfg_rbv2_super_1111", "cfg_rbv2_super_1112", "cfg_rbv2_super_1114",
        "cfg_rbv2_super_1116", "cfg_rbv2_super_1119", "cfg_rbv2_super_1120", "cfg_rbv2_super_1128",
        "cfg_rbv2_super_1130", "cfg_rbv2_super_1134", "cfg_rbv2_super_1138", "cfg_rbv2_super_1142",
        "cfg_rbv2_super_1146", "cfg_rbv2_super_1169", "cfg_rbv2_super_12", "cfg_rbv2_super_1220",
        "cfg_rbv2_super_14", "cfg_rbv2_super_1457", "cfg_rbv2_super_1461", "cfg_rbv2_super_1462",
        "cfg_rbv2_super_1464", "cfg_rbv2_super_1468", "cfg_rbv2_super_1475", "cfg_rbv2_super_1478",
        "cfg_rbv2_super_1479", "cfg_rbv2_super_1480", "cfg_rbv2_super_1485", "cfg_rbv2_super_1486",
        "cfg_rbv2_super_1487", "cfg_rbv2_super_1489", "cfg_rbv2_super_1494", "cfg_rbv2_super_15",
        "cfg_rbv2_super_1501", "cfg_rbv2_super_1510", "cfg_rbv2_super_16", "cfg_rbv2_super_18",
        "cfg_rbv2_super_181", "cfg_rbv2_super_182", "cfg_rbv2_super_188", "cfg_rbv2_super_22",
        "cfg_rbv2_super_23", "cfg_rbv2_super_23381", "cfg_rbv2_super_23512", "cfg_rbv2_super_23517",
        "cfg_rbv2_super_24", "cfg_rbv2_super_28", "cfg_rbv2_super_29", "cfg_rbv2_super_3",
        "cfg_rbv2_super_30", "cfg_rbv2_super_300", "cfg_rbv2_super_307", "cfg_rbv2_super_31",
        "cfg_rbv2_super_312", "cfg_rbv2_super_32", "cfg_rbv2_super_37", "cfg_rbv2_super_375",
        "cfg_rbv2_super_377", "cfg_rbv2_super_38", "cfg_rbv2_super_40496", "cfg_rbv2_super_40498",
        "cfg_rbv2_super_40499", "cfg_rbv2_super_40536", "cfg_rbv2_super_40668", "cfg_rbv2_super_40670",
        "cfg_rbv2_super_40701", "cfg_rbv2_super_40927", "cfg_rbv2_super_40966", "cfg_rbv2_super_40975",
        "cfg_rbv2_super_40979", "cfg_rbv2_super_40981", "cfg_rbv2_super_40982", "cfg_rbv2_super_40983",
        "cfg_rbv2_super_40984", "cfg_rbv2_super_40994", "cfg_rbv2_super_41138", "cfg_rbv2_super_41142",
        "cfg_rbv2_super_41143", "cfg_rbv2_super_41146", "cfg_rbv2_super_41150", "cfg_rbv2_super_41156",
        "cfg_rbv2_super_41157", "cfg_rbv2_super_41159", "cfg_rbv2_super_41161", "cfg_rbv2_super_41162",
        "cfg_rbv2_super_4134", "cfg_rbv2_super_4135", "cfg_rbv2_super_4154", "cfg_rbv2_super_42",
        "cfg_rbv2_super_44", "cfg_rbv2_super_4534", "cfg_rbv2_super_4538", "cfg_rbv2_super_458",
        "cfg_rbv2_super_46", "cfg_rbv2_super_469", "cfg_rbv2_super_470", "cfg_rbv2_super_50",
        "cfg_rbv2_super_54", "cfg_rbv2_super_6", "cfg_rbv2_super_60", "cfg_rbv2_super_6332",
    ]

    _CANONICAL_HPOBENCH_ML = [
        "cfg_ml_lr_10101", "cfg_ml_lr_12", "cfg_ml_lr_146212", "cfg_ml_lr_146606", "cfg_ml_lr_146818",
        "cfg_ml_lr_146821", "cfg_ml_lr_146822", "cfg_ml_lr_14965", "cfg_ml_lr_167119", "cfg_ml_lr_167120",
        "cfg_ml_lr_168911", "cfg_ml_lr_168912", "cfg_ml_lr_3", "cfg_ml_lr_31", "cfg_ml_lr_3917",
        "cfg_ml_lr_53", "cfg_ml_lr_7592", "cfg_ml_lr_9952", "cfg_ml_lr_9977", "cfg_ml_lr_9981",
        "cfg_ml_nn_10101", "cfg_ml_nn_146818", "cfg_ml_nn_146821", "cfg_ml_nn_146822", "cfg_ml_nn_31",
        "cfg_ml_nn_3917", "cfg_ml_nn_53", "cfg_ml_nn_9952",
        "cfg_ml_rf_10101", "cfg_ml_rf_12", "cfg_ml_rf_146212", "cfg_ml_rf_146606", "cfg_ml_rf_146818",
        "cfg_ml_rf_146821", "cfg_ml_rf_146822", "cfg_ml_rf_14965", "cfg_ml_rf_167119", "cfg_ml_rf_167120",
        "cfg_ml_rf_168911", "cfg_ml_rf_168912", "cfg_ml_rf_3", "cfg_ml_rf_31", "cfg_ml_rf_3917",
        "cfg_ml_rf_53", "cfg_ml_rf_7592", "cfg_ml_rf_9952", "cfg_ml_rf_9977", "cfg_ml_rf_9981",
        "cfg_ml_svm_10101", "cfg_ml_svm_12", "cfg_ml_svm_146212", "cfg_ml_svm_146606", "cfg_ml_svm_146818",
        "cfg_ml_svm_146821", "cfg_ml_svm_146822", "cfg_ml_svm_14965", "cfg_ml_svm_167119", "cfg_ml_svm_167120",
        "cfg_ml_svm_168911", "cfg_ml_svm_168912", "cfg_ml_svm_3", "cfg_ml_svm_31", "cfg_ml_svm_3917",
        "cfg_ml_svm_53", "cfg_ml_svm_7592", "cfg_ml_svm_9952", "cfg_ml_svm_9977", "cfg_ml_svm_9981",
        "cfg_ml_xgboost_10101", "cfg_ml_xgboost_12", "cfg_ml_xgboost_146212", "cfg_ml_xgboost_146606", "cfg_ml_xgboost_146818",
        "cfg_ml_xgboost_146821", "cfg_ml_xgboost_146822", "cfg_ml_xgboost_14965", "cfg_ml_xgboost_167119", "cfg_ml_xgboost_167120",
        "cfg_ml_xgboost_168911", "cfg_ml_xgboost_168912", "cfg_ml_xgboost_3", "cfg_ml_xgboost_31", "cfg_ml_xgboost_3917",
        "cfg_ml_xgboost_53", "cfg_ml_xgboost_7592", "cfg_ml_xgboost_9952", "cfg_ml_xgboost_9977", "cfg_ml_xgboost_9981",
    ]
    # fmt: on

    @classmethod
    def _find_package_tasks(cls, subpath: str, prefix: str) -> list[str]:
        """Sorted task config names in the installed carps package, or [] if unavailable."""
        try:
            import carps

            base_dir = os.path.dirname(carps.__file__)
            target_dir = os.path.join(base_dir, "configs", "task", *subpath.split("/"))
            if os.path.isdir(target_dir):
                tasks = [
                    f[:-5] for f in os.listdir(target_dir) if f.startswith(prefix) and f.endswith(".yaml")
                ]
                return sorted(tasks)
        except Exception:
            pass
        return []

    @classmethod
    def get_ranger_tasks(cls) -> list[str]:
        """119 YAHPO rbv2_ranger tasks."""
        found = cls._find_package_tasks("YAHPO/blackbox", "cfg_rbv2_ranger_")
        items = found if len(found) == 119 else cls._CANONICAL_RANGER
        return [f"task=YAHPO/blackbox/{t}" for t in items]

    @classmethod
    def get_super_tasks(cls) -> list[str]:
        """103 YAHPO rbv2_super tasks."""
        found = cls._find_package_tasks("YAHPO/blackbox", "cfg_rbv2_super_")
        items = found if len(found) == 103 else cls._CANONICAL_SUPER
        return [f"task=YAHPO/blackbox/{t}" for t in items]

    @classmethod
    def get_hpobench_ml_tasks(cls) -> list[str]:
        """88 HPOBench tabular-ML tasks."""
        found = cls._find_package_tasks("HPOBench/blackbox/tabular/ml", "cfg_ml_")
        items = found if len(found) == 88 else cls._CANONICAL_HPOBENCH_ML
        return [f"task=HPOBench/blackbox/tabular/ml/{t}" for t in items]

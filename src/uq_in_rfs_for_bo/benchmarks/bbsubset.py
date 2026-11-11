"""CARP-S BBSubset (blackbox, single-objective) dev and test task lists."""


class CarpsBBSubsetRegistry:
    DEV_TASKS = [
        "+task=subselection/blackbox/dev/subset_bbob_2_12_0",
        "+task=subselection/blackbox/dev/subset_bbob_2_12_1",
        "+task=subselection/blackbox/dev/subset_bbob_2_20_0",
        "+task=subselection/blackbox/dev/subset_bbob_4_6_1",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_ml_lr_146818",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_ml_rf_146212",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_ml_xgboost_146212",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_nas_NavalPropulsionBenchmark",
        "+task=subselection/blackbox/dev/subset_hpobench_blackbox_tabular_nas_SliceLocalizationBenchmark",
        "+task=subselection/blackbox/dev/subset_yahpo_lcbench_168335_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_1462_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_312_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_40498_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_aknn_458_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_glmnet_41157_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_ranger_40927_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_svm_182_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_svm_24_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_xgboost_23512_None",
        "+task=subselection/blackbox/dev/subset_yahpo_rbv2_xgboost_42_None",
    ]

    TEST_TASKS = [
        "+task=subselection/blackbox/test/subset_bbob_2_6_1",
        "+task=subselection/blackbox/test/subset_bbob_2_9_0",
        "+task=subselection/blackbox/test/subset_bbob_2_12_2",
        "+task=subselection/blackbox/test/subset_bbob_8_22_0",
        "+task=subselection/blackbox/test/subset_bbob_16_1_1",
        "+task=subselection/blackbox/test/subset_bbob_16_11_0",
        "+task=subselection/blackbox/test/subset_bbob_32_9_0",
        "+task=subselection/blackbox/test/subset_bbob_32_11_0",
        "+task=subselection/blackbox/test/subset_hpobench_blackbox_tabular_ml_svm_12",
        "+task=subselection/blackbox/test/subset_yahpo_lcbench_167184_None",
        "+task=subselection/blackbox/test/subset_yahpo_rbv2_glmnet_32_None",
        "+task=subselection/blackbox/test/subset_yahpo_rbv2_glmnet_375_None",
        "+task=subselection/blackbox/test/subset_yahpo_rbv2_ranger_29_None",
        "+task=subselection/blackbox/test/subset_yahpo_rbv2_rpart_18_None",
        "+task=subselection/blackbox/test/subset_yahpo_rbv2_rpart_4534_None",
        "+task=subselection/blackbox/test/subset_yahpo_rbv2_svm_1493_None",
        "+task=subselection/blackbox/test/subset_yahpo_rbv2_xgboost_1457_None",
        "+task=subselection/blackbox/test/subset_yahpo_rbv2_xgboost_1493_None",
        "+task=subselection/blackbox/test/subset_yahpo_rbv2_xgboost_1510_None",
        "+task=subselection/blackbox/test/subset_yahpo_rbv2_xgboost_41027_None",
    ]

    @classmethod
    def get_dev_tasks(cls) -> list[str]:
        return list(cls.DEV_TASKS)

    @classmethod
    def get_test_tasks(cls) -> list[str]:
        return list(cls.TEST_TASKS)

    @classmethod
    def get_working_dev_tasks(cls, exclude_nas: bool = True) -> list[str]:
        """Dev tasks without the two HPOBench tabular NAS tasks (18 tasks), used by the meta-HPO."""
        if exclude_nas:
            return [t for t in cls.DEV_TASKS if "tabular_nas" not in t]
        return list(cls.DEV_TASKS)

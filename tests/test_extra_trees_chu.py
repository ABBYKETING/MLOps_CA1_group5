"""Regression checks for Extra Trees validation selection and model export inputs."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import extra_trees_chu as et


class TestExtraTrees(unittest.TestCase):
    def make_frame(self):
        rng = np.random.default_rng(42)
        count = 120
        x = rng.normal(size=count)
        return pd.DataFrame({
            "country_code": ["SG"] * count,
            "order_id": [str(i) for i in range(count)],
            "customer_id": [str(i) for i in range(count)],
            "order_date": (["2023-03-01"] * 40 + ["2023-04-01"] * 40
                           + ["2023-05-01"] * 40),
            "is_fraud": (x > 0).astype(int),
            "signal": x,
            "noise": rng.normal(size=count),
        })

    def test_tuning_ignores_test_features_and_labels(self):
        frame = self.make_frame()
        changed = frame.copy()
        # NaNs would make training fail if Test matrices leaked into tuning.
        changed.loc[80:, ["signal", "noise"]] = np.nan
        changed.loc[80:, "is_fraud"] = 0
        grid = {"max_depth": [2, 4], "class_weight": [None, "balanced"]}
        base = {"n_estimators": 5, "n_jobs": 1, "min_samples_leaf": 1,
                "min_samples_split": 2}
        with contextlib.redirect_stdout(io.StringIO()):
            table, best = et.tune(frame, grid, base)
            other_table, other_best = et.tune(changed, grid, base)
        self.assertEqual(best, other_best)
        pd.testing.assert_frame_equal(table.drop(columns="train_seconds"),
                                      other_table.drop(columns="train_seconds"))
        self.assertTrue(table.val_average_precision.is_monotonic_decreasing)
        self.assertEqual(best["max_depth"], grid["max_depth"][int(table.iloc[0].candidate) % 2])
        self.assertIsInstance(best["max_depth"], int)
        json.dumps(best)  # No NumPy scalars or NaN class-weight values.

    def test_selected_settings_round_trip_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "params.json"
            with self.assertRaises(FileNotFoundError):
                et.load_selected_params(path)
            expected = {**et.DEFAULT_PARAMS, "class_weight": None, "max_depth": 25}
            path.write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(et.load_selected_params(path), expected)

    def test_default_run_uses_committed_selected_settings(self):
        with patch.object(et.mu, "run_safe_and_full", return_value=([], {})) as train:
            et.run(use_mlflow=False)
            self.assertEqual(train.call_args.args[4], et.SELECTED_PARAMS)
            self.assertEqual(train.call_args.args[4]["max_depth"], 25)
            self.assertIsNone(train.call_args.args[4]["class_weight"])
        self.assertEqual(et.DEFAULT_PARAMS["max_depth"], 15)
        self.assertEqual(et.DEFAULT_PARAMS["class_weight"], "balanced")

    def test_invalid_sampling_is_rejected(self):
        for fraction in [0, -0.1, 1.5]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                et.tune(self.make_frame(), sample_frac=fraction)


if __name__ == "__main__":
    unittest.main()

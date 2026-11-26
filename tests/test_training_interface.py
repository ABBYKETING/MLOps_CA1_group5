"""Task #48 contract checks using all eight real member implementations.

No RDS, URLs, shared MLflow service, or original training CSVs are required.
Synthetic data is deliberately separable so the existing 70%-precision
threshold policies can be exercised without weakening those policies.
"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import training_interface as interface
from data_processing import SNAPSHOT_FEATURES


def write_features(folder):
    rng = np.random.default_rng(42)
    n = 2000
    y = np.tile([0, 1], n // 2)
    signal = np.where(y == 1, 2.0, -2.0) + rng.normal(0, 0.15, n)
    safe = pd.DataFrame({
        "country_code": ["SG"] * n,
        "order_id": [str(i) for i in range(n)],
        "customer_id": [f"c{i}" for i in range(n)],
        "order_date": ["2023-03-01"] * 1200 + ["2023-04-01"] * 400 + ["2023-05-01"] * 400,
        "signal": signal,
        "signal_squared": signal ** 2,
        "noise_a": rng.normal(size=n),
        "noise_b": rng.normal(size=n),
        "is_fraud": y,
    })
    safe.to_csv(Path(folder) / "features_base_safe.csv", index=False)
    return safe


class TrainingInterfaceTests(unittest.TestCase):
    def test_registry_has_two_algorithms_for_each_member(self):
        owners = [spec.member for spec in interface.MODEL_REGISTRY.values()]
        self.assertEqual(set(owners), {"emi", "chu", "yueliang", "abby"})
        self.assertTrue(all(owners.count(member) == 2 for member in set(owners)))
        for key in interface.MODEL_REGISTRY:
            self.assertIsInstance(interface.get_model_config(key), dict)

    def test_configs_are_copied_and_overrides_are_checked(self):
        module = importlib.import_module("mlp_model")
        original = deepcopy(module.SELECTED_PARAMS)
        config = interface.get_model_config("emi_mlp", {"max_iter": 3})
        config["hidden_layer_sizes"] = (2,)
        self.assertEqual(module.SELECTED_PARAMS, original)
        with self.assertRaisesRegex(ValueError, "Unknown parameters"):
            interface.get_model_config("emi_mlp", {"not_a_parameter": 1})
        with self.assertRaisesRegex(ValueError, "dictionary"):
            interface.get_model_config("emi_mlp", [])

    def test_invalid_requests_fail_before_a_model_runs(self):
        key = "chu_extra_trees"
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            interface.run_model("missing")
        for versions in [(), ("full",), ("safe", "full"), ("safe", "safe"), ("unknown",), "safe"]:
            with self.subTest(versions=versions), self.assertRaises(ValueError):
                interface.run_model(key, datasets=versions)
        for fraction in [0, -1, 2, float("nan"), float("inf"), True]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                interface.run_model(key, sample_frac=fraction)
        with self.assertRaisesRegex(ValueError, "Sampled development"):
            interface.run_model(key, sample_frac=0.5, use_mlflow=True)

    def test_snapshot_features_in_a_safe_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = write_features(folder)
            frame[SNAPSHOT_FEATURES[0]] = 0
            frame.to_csv(Path(folder) / "features_base_safe.csv", index=False)
            with patch("extra_trees_chu.run") as train:
                with self.assertRaisesRegex(ValueError, "excluded snapshot features"):
                    interface.run_model("chu_extra_trees", data_dir=folder)
                train.assert_not_called()

    def test_cli_refuses_full_before_training(self):
        for versions in [["full"], ["safe", "full"]]:
            with self.subTest(versions=versions), patch.object(
                sys, "argv", ["training_interface.py", "--model", "chu_extra_trees", "--datasets", *versions]
            ), patch.object(interface, "run_model") as train, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    interface.main()
                self.assertEqual(stopped.exception.code, 2)
                train.assert_not_called()

    def test_all_eight_algorithms_train_save_and_reload_safe_only(self):
        evidence = []
        with tempfile.TemporaryDirectory() as folder:
            safe = write_features(folder)
            data = {"safe": safe}
            # No Full file is created: every algorithm must work with Safe alone.
            self.assertFalse((Path(folder) / "features_base_full.csv").exists())
            output = Path(folder) / "models"
            for key, spec in interface.MODEL_REGISTRY.items():
                with self.subTest(model=key):
                    module = importlib.import_module(spec.module)
                    original = deepcopy(getattr(module, spec.parameter_attribute))
                    # Global environment defaults must not override member identity.
                    with patch.dict("os.environ", {"MODEL_DEVELOPER": "wrong_member"}), redirect_stdout(io.StringIO()):
                        results, models = interface.run_model(
                            key, data_dir=folder, output_dir=output,
                            use_mlflow=False,
                        )
                    self.assertEqual([r["dataset_type"] for r in results], ["safe"])
                    self.assertEqual(set(models), {"safe"})
                    self.assertEqual(getattr(module, spec.parameter_attribute), original)
                    if spec.member == "abby":
                        self.assertEqual(results[0]["threshold"], 0.5)
                    elif spec.member == "yueliang":
                        self.assertIn("precision", results[0]["threshold_rule"])
                        self.assertGreaterEqual(results[0]["validation"]["precision"], 0.70)
                    else:
                        self.assertEqual(results[0]["threshold_rule"], "max F1 on validation")
                    for result in results:
                        dataset = result["dataset_type"]
                        self.assertEqual(result["developer"], spec.member)
                        self.assertEqual(result["params"], original)
                        self.assertEqual(result["rows"], {"Train": 1200, "Validation": 400, "Test": 400})
                        self.assertEqual(result["model_key"], key)
                        self.assertEqual(result["feature_count"], 4)
                        self.assertTrue(set(result["features"]).isdisjoint(SNAPSHOT_FEATURES))
                        self.assertNotIn("refund_value", result["features"])
                        for period in ["validation", "test"]:
                            self.assertTrue(all(np.isfinite(value) for value in result[period].values()))
                        artifact_dir = Path(result["artifact_dir"])
                        self.assertEqual(artifact_dir.parent, output.resolve() / key)
                        saved = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
                        self.assertEqual(saved["model_key"], key)
                        self.assertEqual(saved["threshold"], result["threshold"])
                        self.assertNotIn("importance", saved)
                        importance = pd.read_csv(artifact_dir / "feature_importance.csv")
                        self.assertEqual(set(importance.feature), set(result["features"]))
                        restored = joblib.load(artifact_dir / "model.joblib")
                        sample = data[dataset].loc[:9, result["features"]]
                        np.testing.assert_allclose(models[dataset].predict_proba(sample),
                                                   restored.predict_proba(sample), rtol=1e-7, atol=1e-9)
                    evidence.append({"model_key": key, "member": spec.member,
                                     "versions": ["safe"], "reload_passed": True})
        report = Path(__file__).resolve().parents[1] / "outputs" / "task48_contract_verification.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"data": "synthetic, 1200 Train / 400 Validation / 400 Test",
                                      "logging": False, "models": evidence}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

"""LightGBM baseline checks: configuration, held-out isolation and serialization."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from skops.io import dumps, loads

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm_chu as lgb


class TestLightGBMBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
        cls.y = (cls.X["a"].to_numpy() > 0).astype(int)
        cls.params = {**lgb.DEFAULT_PARAMS, "n_estimators": 5, "n_jobs": 1}

    def test_validation_does_not_change_training(self):
        first, _ = lgb.train_model(self.X[:200], self.y[:200], self.X[200:], self.y[200:], self.params)
        other, _ = lgb.train_model(self.X[:200], self.y[:200], self.X[200:] * 100, 1-self.y[200:], self.params)
        np.testing.assert_allclose(first.predict_proba(self.X), other.predict_proba(self.X), atol=0, rtol=0)

    def test_serialization_and_gain_importance(self):
        model, _ = lgb.train_model(self.X[:200], self.y[:200], self.X[200:], self.y[200:], self.params)
        restored = loads(dumps(model), trusted=lgb.TRUSTED_TYPES)
        np.testing.assert_allclose(model.predict_proba(self.X), restored.predict_proba(self.X))
        importance = lgb.feature_importance(model, self.X[200:], self.y[200:], list(self.X))
        self.assertEqual(set(importance.feature), set(self.X))
        self.assertAlmostEqual(float(importance.importance.sum()), 1)
        self.assertTrue(importance.importance.is_monotonic_decreasing)

    def test_baseline_config_is_forwarded_without_mutation(self):
        original = lgb.DEFAULT_PARAMS.copy()
        with patch.object(lgb.mu, "run_safe_and_full", return_value=([], {})) as shared:
            lgb.run(use_mlflow=False)
            self.assertEqual(shared.call_args.args[4], original)
            self.assertEqual(shared.call_args.kwargs["developer"], "chu")
            self.assertEqual(shared.call_args.kwargs["trusted_types"], lgb.TRUSTED_TYPES)
        lgb.create_model({"n_estimators": 5})
        self.assertEqual(lgb.DEFAULT_PARAMS, original)


if __name__ == "__main__":
    unittest.main()

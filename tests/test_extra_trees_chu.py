"""Extra Trees baseline checks: configuration, held-out isolation and serialization."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from skops.io import dumps, loads

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import extra_trees_chu as et


class TestExtraTreesBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
        cls.y = (cls.X["a"].to_numpy() > 0).astype(int)
        cls.params = {**et.DEFAULT_PARAMS, "n_estimators": 5, "n_jobs": 1}

    def test_validation_does_not_change_training(self):
        first, _ = et.train_model(self.X[:200], self.y[:200], self.X[200:], self.y[200:], self.params)
        other, _ = et.train_model(self.X[:200], self.y[:200], self.X[200:] * 100, 1-self.y[200:], self.params)
        np.testing.assert_allclose(first.predict_proba(self.X), other.predict_proba(self.X), atol=0, rtol=0)

    def test_serialization_and_impurity_importance(self):
        model, _ = et.train_model(self.X[:200], self.y[:200], self.X[200:], self.y[200:], self.params)
        restored = loads(dumps(model), trusted=et.TRUSTED_TYPES)
        np.testing.assert_allclose(model.predict_proba(self.X), restored.predict_proba(self.X))
        importance = et.feature_importance(model, self.X[200:], self.y[200:], list(self.X))
        self.assertEqual(set(importance.feature), set(self.X))
        self.assertAlmostEqual(float(importance.importance.sum()), 1)
        self.assertTrue(importance.importance.is_monotonic_decreasing)

    def test_baseline_config_is_forwarded_without_mutation(self):
        original = et.DEFAULT_PARAMS.copy()
        with patch.object(et.mu, "run_safe_and_full", return_value=([], {})) as shared:
            et.run(use_mlflow=False)
            self.assertEqual(shared.call_args.args[4], original)
            self.assertEqual(shared.call_args.kwargs["developer"], "chu")
            self.assertEqual(shared.call_args.kwargs["trusted_types"], et.TRUSTED_TYPES)
        et.create_model({"n_estimators": 5})
        self.assertEqual(et.DEFAULT_PARAMS, original)


if __name__ == "__main__":
    unittest.main()

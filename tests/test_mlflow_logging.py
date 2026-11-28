"""
Integration test for shared MLflow logging utilities.
Uses synthetic data and a local MLflow server only.
"""

import os
import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.mlflow_utils import (
    connect_mlflow,
    start_model_run,
    log_model_parameters,
    log_classification_metrics,
    log_trained_model,
    log_evaluation_artifacts
)


class TestMLflowLogging(unittest.TestCase):

    def test_complete_logging_workflow(self):

        # Ensure testing uses a local server only
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")

        self.assertIn(
            urlparse(tracking_uri).hostname,
            ["localhost", "127.0.0.1"],
            "Integration test must use a local MLflow server."
        )

        # 1. Connect to MLflow
        connect_mlflow(
            experiment_name="SP-buy-Logging-Test"
        )

        # 2. Generate synthetic dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            weights=[0.9, 0.1],
            random_state=42
        )

        X = pd.DataFrame(
            X,
            columns=[
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
                "feature_5"
            ]
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # 3. Define model
        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        threshold = 0.5

        # 4. Start MLflow run
        with start_model_run(
            member="test",
            algorithm="logistic_regression",
            dataset_version="synthetic",
            model_version="v1"
        ) as run:

            run_id = run.info.run_id

            # Train model
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_val)[:, 1]

            # Log parameters
            log_model_parameters(
                model_params=model.get_params(),
                threshold=threshold,
                feature_count=X_train.shape[1]
            )

            # Log metrics
            metrics = log_classification_metrics(
                y_true=y_val,
                y_prob=y_prob,
                threshold=threshold,
                split="val"
            )

            # Log trained model
            model_info = log_trained_model(
                model=model,
                X_sample=X_train.head(5)
            )

            # Log evaluation artifacts
            log_evaluation_artifacts(
                y_true=y_val,
                y_prob=y_prob,
                feature_names=X_train.columns,
                threshold=threshold,
                split="val"
            )

        # 5. Verify stored run
        client = mlflow.MlflowClient()

        stored_run = client.get_run(run_id)
        # Standalone callers retain the original shared helper's default name.
        self.assertEqual(client.get_logged_model(model_info.model_id).name, "model")

        self.assertEqual(
            stored_run.info.status,
            "FINISHED"
        )

        self.assertEqual(
            stored_run.data.params["feature_count"],
            "5"
        )

        self.assertEqual(
            stored_run.data.tags["team_member"],
            "test"
        )

        # Verify all evaluation metrics
        for name, value in metrics.items():
            self.assertAlmostEqual(
                stored_run.data.metrics[name],
                value
            )

        # 6. Verify evaluation artifacts
        artifacts = client.list_artifacts(
            run_id,
            "evaluation"
        )

        artifact_paths = [
            artifact.path for artifact in artifacts
        ]

        self.assertIn(
            "evaluation/val_confusion_matrix.png",
            artifact_paths
        )

        self.assertIn(
            "evaluation/val_pr_curve.png",
            artifact_paths
        )

        # Verify feature list can be downloaded
        with TemporaryDirectory() as temp_dir:

            feature_file = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="metadata/features.json",
                dst_path=temp_dir
            )

            features = json.loads(
                Path(feature_file).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                features["features"],
                list(X_train.columns)
            )

        # 7. Reload and verify trained model
        loaded_model = mlflow.sklearn.load_model(
            model_info.model_uri
        )

        np.testing.assert_array_equal(
            model.predict(X_val),
            loaded_model.predict(X_val)
        )

        np.testing.assert_allclose(
            model.predict_proba(X_val),
            loaded_model.predict_proba(X_val)
        )

        print("\nAll MLflow logging tests passed!")
        print("Run ID:", run_id)
        print("Model URI:", model_info.model_uri)

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    unittest.main()

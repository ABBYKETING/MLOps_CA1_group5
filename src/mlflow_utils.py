"""
Shared MLflow logging utilities for the SP-buy
fraud detection project.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar

import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

from mlflow.models import infer_signature

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)


_RETRAINING_CONTEXT = ContextVar("mlflow_retraining_context", default=None)


@contextmanager
def retraining_run_context(*, experiment_id, parent_run_id, tags, on_start):
    """Attach pipeline identity to existing member loggers for this task only.

    Direct notebook/script calls retain the original team naming and behaviour.
    The parent is created through MlflowClient, so it is not an active fluent run.
    """
    token = _RETRAINING_CONTEXT.set({
        "experiment_id": experiment_id, "parent_run_id": parent_run_id,
        "tags": dict(tags), "on_start": on_start,
    })
    try:
        yield
    finally:
        _RETRAINING_CONTEXT.reset(token)

def connect_mlflow(
    experiment_name="SP-buy-Fraud-Detection"
):
    """
    Connect to the configured MLflow tracking server
    and select the experiment.
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI is not configured."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")


def start_model_run(
    member,
    algorithm,
    dataset_version,
    model_version
):
    """
    Start an MLflow run using the team's
    shared naming convention.
    """

    run_name = (
        f"{member}_{algorithm}_"
        f"{dataset_version}_{model_version}"
    )

    tags = {
        "team_member": member,
        "algorithm": algorithm,
        "dataset_version": dataset_version,
        "model_version": model_version
    }

    context = _RETRAINING_CONTEXT.get()
    options = {}
    if context is not None:
        tags.update(context["tags"])
        options = {"experiment_id": context["experiment_id"],
                   "parent_run_id": context["parent_run_id"]}
    run = mlflow.start_run(
        run_name=run_name,
        tags=tags,
        **options,
    )
    if context is not None:
        try:
            context["on_start"](run.info.run_id)
        except Exception:
            mlflow.end_run(status="FAILED")
            raise
    return run

def log_model_parameters(
    model_params,
    threshold,
    feature_count
):
    """
    Log model hyperparameters and training configuration.
    """

    mlflow.log_params(model_params)

    mlflow.log_params({
        "decision_threshold": threshold,
        "feature_count": feature_count
    })

def log_classification_metrics(
    y_true,
    y_prob,
    threshold=0.5,
    split="val"
):
    """
    Calculate and log binary classification metrics
    using a consistent evaluation method.
    """

    if split not in ("train", "val", "test"):
        raise ValueError(
            "split must be train, val or test."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        f"{split}_accuracy": accuracy_score(
            y_true, y_pred
        ),
        f"{split}_precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        f"{split}_recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        f"{split}_f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        f"{split}_pr_auc": average_precision_score(
            y_true, y_prob
        ),
        f"{split}_roc_auc": roc_auc_score(
            y_true, y_prob
        )
    }

    mlflow.log_metrics(metrics)

    return metrics

def log_trained_model(
    model,
    X_sample,
    model_name="model",
    skops_trusted_types=None
):
    """
    Log a trained scikit-learn model or pipeline
    together with its input and output signature.

    Automated retraining uses the source run's full member/algorithm/dataset/
    version name, including for callers that supply a legacy short model name.
    Direct notebook/script calls keep their existing model_name behaviour.
    """

    if _RETRAINING_CONTEXT.get() is not None:
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("Log the retraining model inside its active model run")
        model_name = active_run.info.run_name

    signature = infer_signature(
        X_sample,
        model.predict(X_sample)
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,
        signature=signature,
        serialization_format="skops",
        skops_trusted_types=skops_trusted_types
    )

    mlflow.set_tag(
        "logged_model_uri",
        model_info.model_uri
    )

    return model_info

def log_evaluation_artifacts(
    y_true,
    y_prob,
    feature_names,
    threshold=0.5,
    split="val"
):
    """
    Log evaluation plots and the feature list.
    """

    if split not in ("train", "val", "test"):
        raise ValueError(
            "split must be train, val or test."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["Not Fraud", "Fraud"],
        cmap="Blues",
        ax=ax
    )

    ax.set_title(
        f"{split.upper()} Confusion Matrix"
    )

    mlflow.log_figure(
        fig,
        f"evaluation/{split}_confusion_matrix.png"
    )

    plt.close(fig)

    # Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 5))

    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_prob,
        ax=ax
    )

    ax.set_title(
        f"{split.upper()} Precision-Recall Curve"
    )

    mlflow.log_figure(
        fig,
        f"evaluation/{split}_pr_curve.png"
    )

    plt.close(fig)

    # Feature List
    mlflow.log_dict(
        {
            "feature_count": len(feature_names),
            "features": list(feature_names)
        },
        "metadata/features.json"
    )

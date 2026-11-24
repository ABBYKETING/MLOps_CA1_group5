"""
Shared MLflow logging utilities for the SP-buy
fraud detection project.
"""

import os

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

    return mlflow.start_run(
        run_name=run_name,
        tags=tags
    )

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
    """

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
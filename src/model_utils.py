"""Shared helpers for PBI 02 model training (Decision Tree and MLP).

Used by ``decision_tree.py`` and ``mlp_model.py`` so that both algorithms
load, split, evaluate and log their runs in exactly the same way.

What this module does NOT do:
- It does not clean data or build features (see ``data_processing.py``).
- It does not decide the split dates (see ``data_split.py``).
"""

import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files; works without a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data_split import create_split_labels, rolling_split_policy, split_summary

RANDOM_SEED = 42
TARGET = "is_fraud"

# Columns kept for tracking and splitting only. They are NOT model inputs.
META_COLS = ["country_code", "order_id", "customer_id", "order_date"]

# Columns that must never be model inputs (leakage or identifiers).
FORBIDDEN_FEATURES = ["refund_value", TARGET, "order_id", "customer_id", "order_date"]

ID_DTYPES = {"country_code": "string", "order_id": "string", "customer_id": "string"}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
# MLflow logging uses the team's shared src/mlflow_utils.py (#38):
# experiment "SP-buy-Fraud-Detection", run name {member}_{algorithm}_{dataset_version}_{model_version},
# metric names val_* / test_* (pr_auc = average precision).
DEFAULT_EXPERIMENT = "SP-buy-Fraud-Detection"
DEFAULT_DEVELOPER = os.environ.get("MODEL_DEVELOPER", "emi")   # {member} in run names (GitHub: emi-psl)
MODEL_VERSION = "v1"


# ---------------------------------------------------------------------------
# 1. Loading and validating data
# ---------------------------------------------------------------------------
def dataset_path(dataset_type, data_dir=None, postfix="base"):
    """Return the path of features_<postfix>_<safe|full>.csv.

    The folder comes from (in order): the data_dir argument, the
    FRAUD_DATA_DIR environment variable, or the repo's data/processed folder.
    """
    if dataset_type not in ("safe", "full"):
        raise ValueError(f"dataset_type must be 'safe' or 'full', got {dataset_type!r}")
    folder = Path(data_dir or os.environ.get("FRAUD_DATA_DIR") or DEFAULT_DATA_DIR)
    path = folder / f"features_{postfix}_{dataset_type}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run notebooks/01_data_cleaning.ipynb first, or pass --data-dir / set FRAUD_DATA_DIR."
        )
    return path


def file_sha256(path):
    """SHA-256 checksum, so every run records exactly which file it used."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def get_feature_columns(df):
    """Every column except metadata and the target, after leakage checks."""
    features = [c for c in df.columns if c not in META_COLS + [TARGET]]
    leaked = [c for c in FORBIDDEN_FEATURES if c in features]
    if leaked:
        raise ValueError(f"Forbidden columns found among features: {leaked}")
    return features


def load_feature_data(path):
    """Load one processed feature CSV and check it is safe to train on."""
    path = Path(path)
    df = pd.read_csv(path, dtype=ID_DTYPES)

    missing = [c for c in META_COLS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must contain only 0 and 1")
    if df.duplicated(["country_code", "order_id"]).any():
        raise ValueError("Duplicate (country_code, order_id) order keys found")

    features = get_feature_columns(df)
    values = df[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing or infinite values found in feature columns")
    # Task #50's sidecar binds the agreed split/provenance to these exact bytes.
    # Existing PBI 2 files without a sidecar retain the historical split.
    metadata_path = path.with_suffix(".metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("schema_version") != 1
                or metadata.get("dataset_type") != "safe"
                or metadata.get("source") != "rds_current_state"):
            raise ValueError("Unsupported retraining data metadata")
        if metadata.get("sha256") != file_sha256(path):
            raise ValueError("Retraining data checksum does not match its metadata")
        if metadata.get("features") != features:
            raise ValueError("Retraining feature schema does not match its metadata")
        if metadata.get("total_rows") != len(df) or metadata.get("feature_count") != len(features):
            raise ValueError("Retraining row/feature counts do not match the data")
        if metadata.get("split") != rolling_split_policy(df):
            raise ValueError("Retraining split metadata does not match the order dates")
        df.attrs["retraining_split"] = metadata["split"]
        if metadata.get("partitions") != split_summary(df):
            raise ValueError("Retraining partition counts do not match the data")
        df.attrs["preparation_metadata"] = metadata
    return df


def check_same_orders(df_a, df_b):
    """Raise an error unless both datasets hold the same orders, in the same order, with the same labels."""
    cols = META_COLS + [TARGET]
    if not df_a[cols].equals(df_b[cols]):
        raise ValueError("Safe and full datasets do not contain identical orders and labels")


def split_data(df, sample_frac=None, seed=RANDOM_SEED):
    """Apply the team's shared chronological split.

    Returns {"Train": (X, y), "Validation": (X, y), "Test": (X, y)} plus the feature list.
    ``sample_frac`` keeps a random fraction of each split, for quick smoke tests only.
    """
    labels = create_split_labels(df)  # shared team function: Train / Validation / Test
    features = get_feature_columns(df)
    parts = {}
    for name in ["Train", "Validation", "Test"]:
        part = df[labels == name]
        if sample_frac:
            part = part.sample(frac=sample_frac, random_state=seed)
        if part[TARGET].nunique() < 2:
            raise ValueError(f"{name} split must contain both fraud and non-fraud orders")
        parts[name] = (part[features].astype("float32"), part[TARGET].to_numpy())
    return parts, features


def class_balance(y):
    """Count and percentage of each class."""
    counts = pd.Series(y).value_counts().sort_index()
    return pd.DataFrame({"orders": counts, "percent": (counts / counts.sum() * 100).round(2)})


# ---------------------------------------------------------------------------
# 2. Threshold and evaluation
# ---------------------------------------------------------------------------
def choose_threshold(y_val, prob_val):
    """Pick the probability cut-off that gives the best fraud F1 on VALIDATION data.

    The test set is never used here, so the final test scores stay unbiased.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, prob_val)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1[:-1]))  # last point has no threshold
    return float(thresholds[best])


def evaluate(y_true, prob, threshold):
    """All agreed metrics for the fraud class (label 1)."""
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "average_precision": average_precision_score(y_true, prob),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def timed_predict_proba(model, X):
    """Return (fraud probabilities, seconds taken)."""
    start = time.perf_counter()
    prob = model.predict_proba(X)[:, 1]
    return prob, time.perf_counter() - start


def permutation_importance_table(model, X, y, n_rows=20000, n_repeats=3, seed=RANDOM_SEED):
    """Model-agnostic importance: how much average precision drops when one feature is shuffled."""
    y = np.asarray(y)
    if len(X) > n_rows:  # use a random subset to keep it fast
        rows = np.random.default_rng(seed).choice(len(X), size=n_rows, replace=False)
        X, y = X.iloc[rows], y[rows]
    result = permutation_importance(
        model, X, y, scoring="average_precision",
        n_repeats=n_repeats, random_state=seed, n_jobs=1,
    )
    return (pd.DataFrame({"feature": X.columns, "importance": result.importances_mean,
                          "std": result.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
# 3. Plots (saved as PNG files, then logged to MLflow)
# ---------------------------------------------------------------------------
def save_confusion_matrix(metrics, title, path):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Not fraud", "Fraud"])
    ax.set_yticks([0, 1], ["Not fraud", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_pr_roc_curves(y_true, prob, title, pr_path, roc_path):
    precision, recall, _ = precision_recall_curve(y_true, prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.axhline(y_true.mean(), linestyle="--", color="grey", label=f"Fraud rate ({y_true.mean():.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title}\nAP = {average_precision_score(y_true, prob):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title}\nROC-AUC = {roc_auc_score(y_true, prob):.3f}")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)


def save_importance_plot(importance, title, path, top_n=15, snapshot_features=()):
    """Horizontal bar chart; snapshot (possibly leaking) features shown in red."""
    top = importance.head(top_n).iloc[::-1]
    colours = ["firebrick" if f in snapshot_features else "steelblue" for f in top["feature"]]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(top) + 1.2))
    ax.barh(top["feature"], top["importance"], color=colours)
    ax.set_title(title + ("\n(red = customer snapshot feature)" if snapshot_features else ""))
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. MLflow
# ---------------------------------------------------------------------------
def setup_mlflow(experiment=DEFAULT_EXPERIMENT, tracking_uri=None):
    """Connect to MLflow. Returns 'shared' or 'local' so runs can be labelled.

    Shared server: set the MLFLOW_TRACKING_URI environment variable (never hardcode it);
    the team's mlflow_utils.connect_mlflow() then connects to it.
    If nothing is set, runs go to a LOCAL SQLite store in mlruns/ for testing only.
    """
    import mlflow
    import mlflow_utils  # team's shared MLflow helpers (#38)

    # MLflow 3.x needs a database store; the local fallback is a SQLite file inside mlruns/ (git-ignored)
    local_folder = (REPO_ROOT / "mlruns").resolve()
    local_uri = f"sqlite:///{(local_folder / 'mlflow.db').as_posix()}"

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    # Note: mlflow.set_tracking_uri() also writes MLFLOW_TRACKING_URI, so a second call
    # in the same session sees the local URI here - compare it, don't just check "is set".
    location = "local" if (not uri or uri == local_uri) else "shared"
    try:
        if location == "shared":
            os.environ["MLFLOW_TRACKING_URI"] = uri       # connect_mlflow reads the variable
            mlflow_utils.connect_mlflow(experiment)
        else:
            local_folder.mkdir(exist_ok=True)
            print(f"[MLflow] No shared server configured -> LOCAL testing store: {local_uri}")
            mlflow.set_tracking_uri(local_uri)
            if mlflow.get_experiment_by_name(experiment) is None:
                mlflow.create_experiment(experiment, artifact_location=(local_folder / "artifacts").as_uri())
            mlflow.set_experiment(experiment)
    except Exception as err:  # server down, wrong URL, etc.
        raise RuntimeError(
            f"Cannot reach MLflow at {uri or local_uri}: {err}\n"
            "Check the server is running, or unset MLFLOW_TRACKING_URI to use a local store."
        ) from err
    return location


# {algorithm} part of the run name, lowercase snake_case like "logistic_regression"
RUN_ALGORITHM_NAMES = {
    "DecisionTreeClassifier": "decision_tree",
    "MLPClassifier": "mlp",
    "LogisticRegression": "logistic_regression",
    "HistGradientBoostingClassifier": "hist_gradient_boosting",
    "ExtraTreesClassifier": "extra_trees",
    "LGBMClassifier": "lightgbm",
}


def mlflow_run_name(result, version=MODEL_VERSION):
    """Team convention {member}_{algorithm}_{dataset}_{version}, e.g. emi_decision_tree_safe_v1."""
    algorithm = RUN_ALGORITHM_NAMES[result["algorithm"]]
    return f"{result['developer']}_{algorithm}_{result['dataset_type']}_{version}"


def log_run_to_mlflow(result, model, X_example, plot_dir, tracking_location, predictions,
                      extra_artifacts=(), trusted_types=()):
    """Log one finished training run using the team's shared mlflow_utils functions.

    ``predictions``: {"val": (y_val, prob_val), "test": (y_test, prob_test)}.
    ``trusted_types``: scikit-learn internal types MLflow's safe (skops) format
    must be told to trust. Safe here because we trained the model ourselves.
    """
    import mlflow
    import mlflow_utils  # team's shared MLflow helpers (#38)

    is_full = result["dataset_type"] == "full"
    threshold = float(result["threshold"])

    # Run name + team tags (team_member, algorithm, dataset_version, model_version)
    with mlflow_utils.start_model_run(
        member=result["developer"],
        algorithm=RUN_ALGORITHM_NAMES[result["algorithm"]],
        dataset_version=result["dataset_type"],
        model_version=MODEL_VERSION,
    ) as run:
        # Extra tags for the leakage investigation
        mlflow.set_tags({
            "model_class": result["algorithm"],
            "pbi": "PBI-02",
            "run_type": "primary",
            "selection_candidate": str(result["selection_candidate"]).lower(),
            "production_eligible": "false",
            "exploratory": str(is_full).lower(),
            "tracking_location": tracking_location,
            "note": ("Contains uncertain customer snapshot features - leakage "
                     "investigation only, never register or promote." if is_full
                     else "Safe feature set - eligible for team model selection."),
        })

        # Hyperparameters + decision_threshold + feature_count (team function)
        mlflow_utils.log_model_parameters(result["params"], threshold, result["feature_count"])
        mlflow.log_params({
            "dataset_sha256": result["dataset_sha256"],
            "train_rows": result["rows"]["Train"],
            "validation_rows": result["rows"]["Validation"],
            "test_rows": result["rows"]["Test"],
            "random_seed": RANDOM_SEED,
            "threshold_rule": result["threshold_rule"],
        })

        # Accuracy, precision, recall, F1, pr_auc, roc_auc + plots (team functions)
        for split, (y_true, prob) in predictions.items():
            mlflow_utils.log_classification_metrics(y_true, prob, threshold=threshold, split=split)
            mlflow_utils.log_evaluation_artifacts(y_true, prob, result["features"],
                                                  threshold=threshold, split=split)
        # Extras not covered by mlflow_utils
        for split, key in [("val", "validation"), ("test", "test")]:
            mlflow.log_metrics({f"{split}_{k}": result[key][k] for k in ("tp", "tn", "fp", "fn")})
        mlflow.log_metrics({
            "train_seconds": result["train_seconds"],
            "inference_seconds_test": result["inference_seconds_test"],
            **result.get("extra_metrics", {}),
        })
        mlflow.log_artifacts(str(plot_dir), artifact_path="plots")   # ROC, importance, training curve
        for path in extra_artifacts:
            mlflow.log_artifact(str(path))

        # Model + signature (team function)
        mlflow_utils.log_trained_model(model, X_example.head(5),
                                       skops_trusted_types=list(trusted_types) or None)
        return run.info.run_id


# ---------------------------------------------------------------------------
# 5. One complete training run (shared by both algorithms)
# ---------------------------------------------------------------------------
def run_training(algorithm, short_name, dataset_type, df, train_fn, params,
                 importance_fn, threshold=None, output_dir=DEFAULT_OUTPUT_DIR,
                 dataset_sha256="", sample_frac=None, mlflow_location=None,
                 developer=DEFAULT_DEVELOPER, snapshot_features=(), trusted_types=(),
                 threshold_fn=None, threshold_rule=None):
    """Split -> train -> choose threshold -> evaluate -> plots -> (optional) MLflow.

    train_fn(X_train, y_train, X_val, y_val, params) must return (model, info_dict)
    where model has predict_proba. If ``threshold`` is given it is reused
    (so a full-feature run uses exactly the same cut-off as its safe run).
    ``threshold_fn(y_val, prob_val)`` lets a model use its own threshold rule
    (default: choose_threshold, max F1); ``threshold_rule`` is its description.
    """
    import joblib

    run_name = f"{short_name}_{dataset_type}"
    parts, features = split_data(df, sample_frac=sample_frac)
    X_train, y_train = parts["Train"]
    X_val, y_val = parts["Validation"]
    X_test, y_test = parts["Test"]

    start = time.perf_counter()
    model, info = train_fn(X_train, y_train, X_val, y_val, params)
    train_seconds = time.perf_counter() - start

    prob_val, _ = timed_predict_proba(model, X_val)
    if threshold is not None:
        threshold_rule = "reused from safe run"
    else:
        threshold = (threshold_fn or choose_threshold)(y_val, prob_val)
        threshold_rule = threshold_rule or "max F1 on validation"
    prob_test, inference_seconds = timed_predict_proba(model, X_test)

    result = {
        "run_name": run_name, "algorithm": algorithm, "dataset_type": dataset_type,
        "developer": developer, "params": params, "features": features,
        "feature_count": len(features), "dataset_sha256": dataset_sha256,
        "rows": {k: len(v[1]) for k, v in parts.items()},
        "threshold": threshold, "threshold_rule": threshold_rule,
        "validation": evaluate(y_val, prob_val, threshold),
        "test": evaluate(y_test, prob_test, threshold),
        "train_seconds": train_seconds, "inference_seconds_test": inference_seconds,
        "extra_metrics": {k: float(v) for k, v in info.get("metrics", {}).items()},
        "selection_candidate": dataset_type == "safe" and sample_frac is None,
        "production_eligible": False,
        "sample_frac": sample_frac,
    }

    # Plots and local copies
    run_dir = Path(output_dir) / run_name
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(result["test"], f"{run_name} - test set", plot_dir / "confusion_matrix_test.png")
    save_pr_roc_curves(y_test, prob_test, f"{run_name} - test set",
                       plot_dir / "precision_recall_test.png", plot_dir / "roc_test.png")
    importance = importance_fn(model, X_val, y_val, features)
    importance.to_csv(run_dir / "feature_importance.csv", index=False)
    save_importance_plot(importance, f"{run_name} - feature importance",
                         plot_dir / "feature_importance.png", snapshot_features=snapshot_features)
    for name, fig_fn in info.get("plots", {}).items():
        fig_fn(plot_dir / name)

    joblib.dump(model, run_dir / "model.joblib")
    with open(run_dir / "result.json", "w") as f:
        json.dump({k: v for k, v in result.items()}, f, indent=2, default=str)

    if mlflow_location:
        result["mlflow_run_id"] = log_run_to_mlflow(
            result, model, X_val, plot_dir, mlflow_location,
            predictions={"val": (y_val, prob_val), "test": (y_test, prob_test)},
            extra_artifacts=[run_dir / "feature_importance.csv"], trusted_types=trusted_types)
    result["importance"] = importance
    return result, model


def comparison_table(results):
    """Side-by-side test metrics for several runs (for the notebooks)."""
    rows = {}
    for r in results:
        t = r["test"]
        rows[r["run_name"]] = {
            "Accuracy": t["accuracy"], "Precision (fraud)": t["precision"],
            "Recall (fraud)": t["recall"], "F1 (fraud)": t["f1"],
            "Average precision": t["average_precision"], "ROC-AUC": t["roc_auc"],
            "TP": t["tp"], "FP": t["fp"], "FN": t["fn"], "TN": t["tn"],
            "Threshold": r["threshold"], "Train seconds": r["train_seconds"],
            "Inference seconds (test)": r["inference_seconds_test"],
            "Selection candidate": "Yes" if r["selection_candidate"] else "No",
            "Production eligible": "Yes" if r["production_eligible"] else "No",
        }
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Safe + full experiment pair and command-line options (shared by both scripts)
# ---------------------------------------------------------------------------
def run_safe_and_full(algorithm, short_name, train_fn, importance_fn, params,
                      datasets=("safe", "full"), data_dir=None, output_dir=DEFAULT_OUTPUT_DIR,
                      sample_frac=None, use_mlflow=True, experiment=DEFAULT_EXPERIMENT,
                      tracking_uri=None, developer=DEFAULT_DEVELOPER, trusted_types=(),
                      threshold_fn=None, threshold_rule=None):
    """Train the same configuration on the safe and then the full dataset.

    The safe run chooses the threshold on validation data; the full run
    reuses it, so the ONLY difference between the two runs is the feature set.
    """
    from data_processing import SNAPSHOT_FEATURES

    if "full" in datasets and "safe" not in datasets:
        raise ValueError("The full run reuses the safe run's threshold, so 'safe' must also be run")

    frames, checksums = {}, {}
    for dataset_type in datasets:
        path = dataset_path(dataset_type, data_dir)
        frames[dataset_type] = load_feature_data(path)
        checksums[dataset_type] = file_sha256(path)
        print(f"Loaded {dataset_type}: {frames[dataset_type].shape}, sha256 {checksums[dataset_type][:12]}...")
    if len(frames) == 2:
        check_same_orders(frames["safe"], frames["full"])
        print("Safe and full datasets contain identical orders and labels.")

    location = setup_mlflow(experiment, tracking_uri) if use_mlflow else None

    results, models, threshold = [], {}, None
    for dataset_type in datasets:
        result, model = run_training(
            algorithm, short_name, dataset_type, frames[dataset_type], train_fn, params,
            importance_fn, threshold=threshold, output_dir=output_dir,
            dataset_sha256=checksums[dataset_type], sample_frac=sample_frac,
            mlflow_location=location, developer=developer,
            snapshot_features=SNAPSHOT_FEATURES, trusted_types=trusted_types,
            threshold_fn=threshold_fn, threshold_rule=threshold_rule,
        )
        threshold = result["threshold"]  # full run reuses the safe threshold
        results.append(result)
        models[dataset_type] = model
        t = result["test"]
        print(f"{result['run_name']}: train {result['train_seconds']:.1f}s | test AP {t['average_precision']:.4f} "
              f"| precision {t['precision']:.4f} recall {t['recall']:.4f} F1 {t['f1']:.4f}")
    return results, models


def build_arg_parser(description, default_params):
    """Command-line options shared by decision_tree.py and mlp_model.py."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", help="Folder with features_base_safe/full.csv "
                        "(default: $FRAUD_DATA_DIR or data/processed)")
    parser.add_argument("--datasets", nargs="+", default=["safe", "full"], choices=["safe", "full"])
    parser.add_argument("--params", default=None,
                        help=f"JSON overriding the default hyperparameters {json.dumps(default_params)}")
    parser.add_argument("--tune", action="store_true",
                        help="Run the small hyperparameter search on safe train/validation first")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Use a fraction of each split (smoke tests only)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
    parser.add_argument("--tracking-uri", default=None, help="Overrides $MLFLOW_TRACKING_URI")
    parser.add_argument("--developer", default=DEFAULT_DEVELOPER, help="Team member name for MLflow")
    return parser


def parse_params(text, defaults):
    """Merge a JSON string from the command line into the default hyperparameters."""
    params = dict(defaults)
    if text:
        try:
            override = json.loads(text)
        except json.JSONDecodeError as err:
            raise ValueError(f"--params must be valid JSON, e.g. '{{\"max_depth\": 8}}': {err}") from err
        unknown = set(override) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown hyperparameters: {sorted(unknown)}. Allowed: {sorted(defaults)}")
        params.update(override)
    return params


def log_tuning_table(table, run_name, experiment=DEFAULT_EXPERIMENT, tracking_uri=None,
                     developer=DEFAULT_DEVELOPER):
    """Log a hyperparameter-search table as its own clearly-labelled MLflow run."""
    import mlflow

    location = setup_mlflow(experiment, tracking_uri)
    with mlflow.start_run(run_name=f"{developer}_{run_name}_{MODEL_VERSION}"):
        mlflow.set_tags({"run_type": "tuning", "pbi": "PBI-02", "dataset_version": "safe",
                         "selection_candidate": "false",
                         "production_eligible": "false", "tracking_location": location,
                         "team_member": developer, "model_version": MODEL_VERSION})
        mlflow.log_table(table, "tuning_results.json")
        mlflow.log_metric("best_val_pr_auc", float(table["val_average_precision"].max()))


def gain_by_month(safe_model, full_model, df_safe, df_full, after=None):
    """Average precision of the safe and full models for each month of unseen data.

    Used for the leakage analysis: if the full model's advantage grows as orders
    get closer to the customer snapshot date, that is consistent with (but does
    not prove) snapshot features containing information from after the order.
    """
    from data_split import TRAIN_END

    after = pd.Timestamp(after) if after is not None else TRAIN_END
    dates = pd.to_datetime(df_safe["order_date"])
    month = dates.dt.to_period("M").astype(str)
    unseen = dates > after
    X_safe = df_safe[get_feature_columns(df_safe)].astype("float32")
    X_full = df_full[get_feature_columns(df_full)].astype("float32")
    rows = []
    for m in sorted(month[unseen].unique()):
        rows_m = unseen & (month == m)
        y = df_safe.loc[rows_m, TARGET]
        ap_safe = average_precision_score(y, safe_model.predict_proba(X_safe[rows_m])[:, 1])
        ap_full = average_precision_score(y, full_model.predict_proba(X_full[rows_m])[:, 1])
        rows.append({"month": m, "orders": int(rows_m.sum()), "fraud_rate_%": round(y.mean() * 100, 2),
                     "safe_AP": ap_safe, "full_AP": ap_full, "gain_full_minus_safe": ap_full - ap_safe})
    return pd.DataFrame(rows)

"""PBI 5 task #48: one entry point for the eight existing member models.

This adapter trains Safe features only and preserves each module's settings,
preprocessing and threshold policy. It runs one algorithm at a time; task #49's
retraining_workflow.py calls it for all models. Task #50 prepares RDS data and
records the agreed split in a checked metadata file; ETL integration is task #52.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
import sys

import pandas as pd

# Existing member modules import helpers from the src directory directly.
SOURCE_DIR = str(Path(__file__).resolve().parent)
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

import model_utils as mu
from data_processing import SNAPSHOT_FEATURES


@dataclass(frozen=True)
class ModelSpec:
    module: str
    member: str
    parameter_attribute: str


# Explicit ownership avoids MODEL_DEVELOPER accidentally attributing all runs
# to one person. Names also distinguish the two Logistic Regression models.
MODEL_REGISTRY = {
    "emi_decision_tree": ModelSpec("decision_tree", "emi", "SELECTED_PARAMS"),
    "emi_mlp": ModelSpec("mlp_model", "emi", "SELECTED_PARAMS"),
    "chu_extra_trees": ModelSpec("extra_trees_chu", "chu", "DEFAULT_PARAMS"),
    "chu_lightgbm": ModelSpec("lightgbm_chu", "chu", "DEFAULT_PARAMS"),
    "yueliang_logistic_regression": ModelSpec("logistic_regression_yueliang", "yueliang", "SELECTED_PARAMS"),
    "yueliang_hist_gradient_boosting": ModelSpec("hist_gradient_boosting_yueliang", "yueliang", "SELECTED_PARAMS"),
    "abby_logistic_regression": ModelSpec("logistic_regression_abby", "abby", "SELECTED_PARAMS"),
    "abby_random_forest": ModelSpec("random_forest_abby", "abby", "SELECTED_PARAMS"),
}


def _get_spec(model_key):
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as error:
        raise ValueError(f"Unknown model {model_key!r}. Choose from {list(MODEL_REGISTRY)}") from error


def get_model_config(model_key, params=None):
    """Return an independent configuration, without changing module defaults."""
    spec = _get_spec(model_key)
    module = importlib.import_module(spec.module)
    config = deepcopy(getattr(module, spec.parameter_attribute))
    if params is not None:
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary of parameter overrides")
        unknown = set(params) - set(config)
        if unknown:
            raise ValueError(f"Unknown parameters for {model_key}: {sorted(unknown)}")
        config.update(deepcopy(params))
    return config


def run_model(model_key, *, data_dir=None, datasets=("safe",),
              output_dir=mu.DEFAULT_OUTPUT_DIR, params=None, sample_frac=None,
              use_mlflow=False, experiment=mu.DEFAULT_EXPERIMENT,
              tracking_uri=None):
    """Run one member algorithm and return ``(results, models)``.

    The input file is features_base_safe.csv with the
    team's metadata and target columns. Each existing module applies its own
    original preprocessing and threshold policy. Full requests are rejected.
    The caller supplies prepared data; this function does not fetch or append
    new records, choose a new chronological split, or promote a model.

    Output directories are isolated by model key. Returned results retain the
    member's original fields and add model_key, model_version, artifact_dir and
    selection_candidate when absent. result.json excludes the importance
    DataFrame, which is stored separately as feature_importance.csv.
    """
    spec = _get_spec(model_key)
    if isinstance(datasets, str):
        raise ValueError("datasets must be the sequence ('safe',)")
    requested = tuple(datasets)
    if requested != ("safe",):
        raise ValueError("PBI 5 trains Safe only; datasets must be ('safe',)")
    versions = ("safe",)
    if sample_frac is not None:
        if (isinstance(sample_frac, bool) or not isinstance(sample_frac, (int, float))
                or not math.isfinite(sample_frac) or not 0 < sample_frac <= 1):
            raise ValueError("sample_frac must be a finite number in (0, 1]")
        if use_mlflow:
            raise ValueError("Sampled development runs must use use_mlflow=False")

    config = get_model_config(model_key, params)
    safe_path = mu.dataset_path("safe", data_dir)
    metadata_path = safe_path.with_suffix(".metadata.json")
    preparation = (json.loads(metadata_path.read_text(encoding="utf-8"))
                   if metadata_path.is_file() else None)
    columns = pd.read_csv(safe_path, nrows=0).columns
    snapshots = sorted(set(columns).intersection(SNAPSHOT_FEATURES))
    if snapshots:
        raise ValueError(f"Safe input contains excluded snapshot features: {snapshots}")
    module = importlib.import_module(spec.module)
    model_output = Path(output_dir).resolve() / model_key
    results, models = module.run(
        params=config, datasets=versions, data_dir=data_dir,
        output_dir=model_output, sample_frac=sample_frac,
        use_mlflow=use_mlflow, experiment=experiment,
        tracking_uri=tracking_uri, developer=spec.member,
    )
    if [result["dataset_type"] for result in results] != list(versions):
        raise RuntimeError(f"{model_key} returned unexpected dataset results")
    if set(models) != {"safe"}:
        raise RuntimeError(f"{model_key} returned models other than Safe")
    for result in results:
        dataset = result["dataset_type"]
        if dataset not in models or not callable(getattr(models[dataset], "predict_proba", None)):
            raise RuntimeError(f"{model_key}/{dataset} did not return a probability model")
        if result["developer"] != spec.member or result["params"] != config:
            raise RuntimeError(f"{model_key} changed its member identity or requested parameters")
        required = {"accuracy", "precision", "recall", "f1", "average_precision", "roc_auc", "tp", "tn", "fp", "fn"}
        for period in ("validation", "test"):
            if not required.issubset(result[period]):
                raise RuntimeError(f"{model_key}/{dataset} is missing standard {period} metrics")
        result["model_key"] = model_key
        if preparation is not None:
            result["data_preparation"] = preparation
        result.setdefault("model_version", mu.MODEL_VERSION)
        result.setdefault("selection_candidate", dataset == "safe" and sample_frac is None)
        # Retain member eligibility policy; sampled runs are never promoted
        # to eligible by standardization.
        if sample_frac is not None:
            result["production_eligible"] = False
            result["selection_candidate"] = False
        artifact_dir = model_output / result["run_name"]
        result["artifact_dir"] = str(artifact_dir)
        for filename in ("model.joblib", "feature_importance.csv"):
            if not (artifact_dir / filename).is_file():
                raise RuntimeError(f"Missing {model_key}/{dataset} artifact: {filename}")
        payload = {key: value for key, value in result.items() if key != "importance"}
        (artifact_dir / "result.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8",
        )
    return results, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model", choices=MODEL_REGISTRY)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default=str(mu.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--datasets", nargs="+", choices=["safe"], default=["safe"],
                        help="PBI 5 supports Safe only (the default)")
    parser.add_argument("--params", help="JSON object overriding existing configuration keys")
    parser.add_argument("--sample-frac", type=float)
    parser.add_argument("--use-mlflow", action="store_true", help="Off by default; logging integration is task #51")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default=mu.DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    if args.list_models:
        for key, spec in MODEL_REGISTRY.items():
            print(f"{key:<36} {spec.module:<36} {spec.member}")
        return
    if not args.model:
        parser.error("give --model or --list-models")
    try:
        overrides = json.loads(args.params) if args.params is not None else None
        results, _ = run_model(
            args.model, data_dir=args.data_dir, datasets=args.datasets,
            output_dir=args.output_dir, params=overrides, sample_frac=args.sample_frac,
            use_mlflow=args.use_mlflow, tracking_uri=args.tracking_uri, experiment=args.experiment,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(mu.comparison_table(results).round(4).to_string())


if __name__ == "__main__":
    main()

"""PBI 02 - Multi-Layer Perceptron (MLP) fraud classifier (MLP_safe and MLP_full).

Run from the repository root, for example:

    python src/mlp_model.py                        # MLP_safe + MLP_full with the selected settings
    python src/mlp_model.py --tune                 # small search on safe train/validation first
    python src/mlp_model.py --sample-frac 0.05 --no-mlflow   # quick smoke test

Key design decisions
--------------------
1. Scaling: neural networks are trained with gradient descent, which works badly
   when features have very different ranges (e.g. account_age_days in the
   hundreds vs 0/1 flags). A StandardScaler is fitted on TRAINING data only and
   saved together with the network in one sklearn Pipeline, so retraining and
   prediction always apply exactly the same scaling.
2. Chronological early stopping: MLPClassifier(early_stopping=True) would take a
   RANDOM 10% of the training rows as its own validation set, which mixes dates.
   Instead we train one epoch at a time with partial_fit and check the team's
   chronological VALIDATION period after each epoch, keeping the best epoch.
3. Class imbalance: MLPClassifier has no class_weight option, but scikit-learn
   1.7 accepts sample_weight. class_weight="balanced" turns into per-row weights
   computed from TRAINING labels only. No oversampling is used.
"""

import copy
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import model_utils as mu

ALGORITHM = "MLPClassifier"
SHORT_NAME = "MLP"

# Settings selected by the validation search (see notebooks/mlp_neural_network_emi.ipynb).
SELECTED_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "tanh",        # tanh beat relu on validation AP (relu peaked after 1 epoch)
    "solver": "adam",
    "alpha": 0.0001,             # L2 regularisation
    "batch_size": 1024,
    "learning_rate_init": 0.001,
    "max_iter": 50,              # maximum epochs
    "n_iter_no_change": 5,       # stop after this many epochs without improvement
    "tol": 0.0001,               # minimum validation AP gain that counts as improvement
    "class_weight": None,        # None or "balanced"
    "random_state": mu.RANDOM_SEED,
}

# Small, practical search: each candidate changes one idea from the starting point.
# Round 1 started from the suggested relu configuration; round 2 was added after
# every relu network peaked after only 1 epoch. The selected settings are above.
CANDIDATES = [
    {"activation": "relu"},                              # suggested starting configuration
    {"activation": "relu", "class_weight": "balanced"},              # imbalance handling
    {"activation": "relu", "hidden_layer_sizes": (32, 16)},          # smaller network
    {"activation": "relu", "hidden_layer_sizes": (128, 64)},         # bigger network
    {"activation": "relu", "alpha": 0.001},                          # stronger regularisation
    {"activation": "tanh"},                                          # different activation
    {"activation": "relu", "learning_rate_init": 0.0001},            # round 2: slower learning
    {"activation": "tanh", "alpha": 0.001},                          # round 2: tanh + regularisation
]


def create_model(params):
    """Untrained MLPClassifier (our own training loop handles epochs and stopping)."""
    return MLPClassifier(
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        activation=params["activation"],
        solver=params["solver"],
        alpha=params["alpha"],
        batch_size=params["batch_size"],
        learning_rate_init=params["learning_rate_init"],
        random_state=params["random_state"],
        shuffle=True,
    )


def train_model(X_train, y_train, X_val, y_val, params):
    """Fit scaler on train, then train the network epoch by epoch with chronological early stopping.

    Returns (Pipeline(scaler, best network), info dict with history and plots).
    """
    scaler = StandardScaler().fit(X_train)              # training data only
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)                   # validation uses the training scaling

    weights = (compute_sample_weight("balanced", y_train)
               if params.get("class_weight") == "balanced" else None)

    mlp = create_model(params)
    best_ap, best_epoch, best_mlp, waited = -1.0, 0, None, 0
    history = []
    for epoch in range(1, params["max_iter"] + 1):
        mlp.partial_fit(X_train_s, y_train, classes=[0, 1], sample_weight=weights)
        val_ap = average_precision_score(y_val, mlp.predict_proba(X_val_s)[:, 1])
        history.append({"epoch": epoch, "train_loss": mlp.loss_, "val_average_precision": val_ap})
        if val_ap > best_ap + params["tol"]:
            best_ap, best_epoch, best_mlp, waited = val_ap, epoch, copy.deepcopy(mlp), 0
        else:
            waited += 1
            if waited >= params["n_iter_no_change"]:
                break

    history = pd.DataFrame(history)
    model = Pipeline([("scaler", scaler), ("mlp", best_mlp)])
    info = {
        "metrics": {
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_validation_average_precision_during_training": best_ap,
            "final_train_loss": float(history["train_loss"].iloc[-1]),
            "stopped_early": float(len(history) < params["max_iter"]),
            "n_weights": sum(w.size for w in best_mlp.coefs_) + sum(b.size for b in best_mlp.intercepts_),
        },
        "history": history,
        "plots": {"training_curve.png": lambda path: save_training_curve(history, best_epoch, path)},
    }
    return model, info


def save_training_curve(history, best_epoch, path):
    """Training loss and validation AP per epoch (convergence check)."""
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(history["epoch"], history["train_loss"], color="steelblue", label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["val_average_precision"], color="darkorange",
             label="Validation AP")
    ax2.set_ylabel("Validation average precision", color="darkorange")
    ax1.axvline(best_epoch, linestyle="--", color="grey")
    ax1.set_title(f"MLP training curve (best epoch = {best_epoch})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def feature_importance(model, X_val, y_val, features):
    """Permutation importance on a validation sample.

    Network weights are NOT a feature importance measure, so we measure how much
    validation average precision drops when each feature is randomly shuffled.
    """
    return mu.permutation_importance_table(model, X_val, y_val)


def tune(df_safe, candidates=CANDIDATES, base_params=SELECTED_PARAMS, sample_frac=None):
    """Train each candidate on SAFE train data and score it on SAFE validation data."""
    parts, _ = mu.split_data(df_safe, sample_frac=sample_frac)
    X_train, y_train = parts["Train"]
    X_val, y_val = parts["Validation"]
    rows, best_params, best_score = [], None, -1.0
    for change in candidates:
        params = {**base_params, **change}
        start = time.perf_counter()
        model, info = train_model(X_train, y_train, X_val, y_val, params)
        seconds = time.perf_counter() - start
        prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, prob)
        rows.append({"change": str(change) if change else "starting point",
                     "hidden_layer_sizes": str(params["hidden_layer_sizes"]),
                     "activation": params["activation"], "alpha": params["alpha"],
                     "class_weight": str(params["class_weight"]),
                     "val_average_precision": score,
                     "val_roc_auc": roc_auc_score(y_val, prob),
                     "epochs_run": info["metrics"]["epochs_run"],
                     "best_epoch": info["metrics"]["best_epoch"],
                     "train_seconds": round(seconds, 1)})
        print(f"  {rows[-1]['change']}: val AP {score:.4f} in {seconds:.0f}s")
        if score > best_score:
            best_score, best_params = score, params
    table = pd.DataFrame(rows).sort_values("val_average_precision", ascending=False)
    return table.reset_index(drop=True), best_params


# Internal types MLflow's safe save format must trust (we trained the model ourselves)
TRUSTED_TYPES = [
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
    "sklearn.neural_network._stochastic_optimizers.SGDOptimizer",
]


def run(params=None, **kwargs):
    """Train MLP_safe and MLP_full (see model_utils.run_safe_and_full for options)."""
    return mu.run_safe_and_full(ALGORITHM, SHORT_NAME, train_model, feature_importance,
                                params or SELECTED_PARAMS, trusted_types=TRUSTED_TYPES, **kwargs)


def main():
    parser = mu.build_arg_parser("Train MLP neural network fraud models (PBI 02).", SELECTED_PARAMS)
    args = parser.parse_args()
    params = mu.parse_params(args.params, SELECTED_PARAMS)

    if args.tune:
        df_safe = mu.load_feature_data(mu.dataset_path("safe", args.data_dir))
        table, params = tune(df_safe, base_params=params, sample_frac=args.sample_frac)
        print(table.to_string(index=False))
        print("Selected:", params)
        if not args.no_mlflow:
            mu.log_tuning_table(table, "mlp_tuning", args.experiment, args.tracking_uri, args.developer)

    start = time.perf_counter()
    results, _ = run(params, datasets=args.datasets, data_dir=args.data_dir,
                     output_dir=args.output_dir, sample_frac=args.sample_frac,
                     use_mlflow=not args.no_mlflow, experiment=args.experiment,
                     tracking_uri=args.tracking_uri, developer=args.developer)
    print(mu.comparison_table(results).round(4).to_string())
    print(f"Total time (load + train + evaluate + log): {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()

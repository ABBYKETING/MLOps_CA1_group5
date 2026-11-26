# MLOps CA1 — Group 5

Fraud Detection Machine Learning Project

## Project Structure

- notebooks/ — EDA and model experiments
- src/ — Reusable Python code
- data/ — Local datasets
- flows/ — Prefect workflows
- sql/ — Analytics queries

## Shared MLflow Experiment Logging

All team members should use the shared logging utilities in `src/mlflow_utils.py` to ensure consistent and comparable model training results.

### Shared Standards

- Experiment name: `SP-buy-Fraud-Detection`
- Run naming convention: `{member}_{algorithm}_{dataset_version}_{model_version}`
- Primary evaluation metric: Precision
- Secondary evaluation metric: Recall
- Additional metrics: Accuracy, F1-score, Average Precision and ROC-AUC
- Validation metrics are used for model improvement and selection.
- Test metrics are used for final evaluation.
- The `refund_value` column must not be used as a prediction feature.

### Available Functions

| Function | Purpose |
|---|---|
| `connect_mlflow()` | Connect to the configured tracking server |
| `start_model_run()` | Create a run with consistent naming and tags |
| `log_model_parameters()` | Record hyperparameters and training configuration |
| `log_classification_metrics()` | Calculate and log classification metrics |
| `log_trained_model()` | Save a trained scikit-learn model and its signature |
| `log_evaluation_artifacts()` | Save evaluation plots and feature metadata |

### Configuration

Configure the `MLFLOW_TRACKING_URI` environment variable before running model training. Do not hard-code server URLs or credentials in source code.

For local testing, use the address of your local MLflow server.

### Testing

Run the integration test from the repository root while a local MLflow server is running:

`python -m unittest discover -s tests -p "test_mlflow_logging.py" -v`

The integration test uses synthetic data to verify parameter logging, metric logging, artifact retrieval, model storage and model reloading.

The current implementation has been successfully tested with scikit-learn Logistic Regression. Other model types must be verified for compatibility before formal experiment logging.

## ETL Pipeline (PBI 4)

Cleans each new batch of `customers_XXX.csv`, `orders_XXX.csv` and `labels_XXX.csv` with the team's rules in `src/data_processing.py` and loads it into the RDS MySQL database. Customers, orders and labels that already exist are updated ("latest wins"). Rows that cannot be used are saved in `rejected_records` with a reason. Each run is recorded in `etl_batches`.

### 1. Set the connection (once per terminal)

Never hard-code these values or commit them.

```powershell
$env:DB_HOST = "<RDS endpoint>"
$env:DB_USER = "<username>"
$env:DB_PASSWORD = "<password>"
$env:DB_NAME = "spbuy"        # optional, default spbuy
```

### 2. Run a new batch (interview demo)

```powershell
python src/etl_pipeline.py --customers <customers URL> --orders <orders URL> --labels <labels URL>
```

The batch name is taken from the file postfix (`XXX`). A summary is written to `outputs/etl/<batch>_summary.json`.

Other options:

| Command | Purpose |
|---|---|
| `python src/etl_pipeline.py --data-dir <folder> --postfix base` | Load local files (the historical base data is already loaded) |
| `python src/etl_pipeline.py ... --dry-run` | Extract and clean only, without touching the database |
| `python src/database.py` | Create the tables from `sql/schema.sql` (safe to re-run) |
| `python src/database.py --check` | Check the connection and tables |

### 3. Check the data after a run

```powershell
python src/check_data_integrity.py
```

This prints row counts, rejected rows by reason and 11 PASS/FAIL checks, and exits with code 1 if any check fails.

### 4. Tests

```powershell
python -m unittest tests.test_etl_pipeline -v
```

The database test runs only when `ETL_TEST_DATABASE` names a separate practice database (for example `spbuy_practice`). It refuses to run against `spbuy`.

### What happens to bad data

- **A bad row** (wrong value, duplicate, future date, too long, malformed line, or a label without its order or customer) is rejected with a reason, and the rest of the batch loads.
- **A bad file** (missing column, empty, not a UTF-8 CSV, unreachable URL) stops the run before anything is loaded.
- **A database error** rolls back the whole batch and marks it `failed`; fix the problem and run the same command again.

## Standardized model training (PBI 5, task #48)

`src/training_interface.py` provides one callable interface and one CLI for the
eight existing member algorithms, training **Safe only** for PBI 5. It delegates to the original training modules;
their settings, preprocessing, early stopping and threshold policies are preserved.
Existing notebooks and script entry points remain usable.

### Reviewed model interfaces

| Model key | Source module in `src/` | Configuration | Preprocessing | Safe threshold rule |
|---|---|---|---|---|
| `emi_decision_tree` | `decision_tree.py` | `SELECTED_PARAMS` | Numeric features | Maximum validation F1 |
| `emi_mlp` | `mlp_model.py` | `SELECTED_PARAMS` | Train-fitted StandardScaler; chronological validation early stopping | Maximum validation F1 |
| `chu_extra_trees` | `extra_trees_chu.py` | `DEFAULT_PARAMS` | Numeric features | Maximum validation F1 |
| `chu_lightgbm` | `lightgbm_chu.py` | `DEFAULT_PARAMS` | Numeric features | Maximum validation F1 |
| `yueliang_logistic_regression` | `logistic_regression_yueliang.py` | `SELECTED_PARAMS` | Train-fitted StandardScaler | Highest recall with validation precision at least 70% |
| `yueliang_hist_gradient_boosting` | `hist_gradient_boosting_yueliang.py` | `SELECTED_PARAMS` | Numeric features; chronological validation early stopping | Highest recall with validation precision at least 70% |
| `abby_logistic_regression` | `logistic_regression_abby.py` | `SELECTED_PARAMS` | Train-fitted StandardScaler | Fixed 0.5 |
| `abby_random_forest` | `random_forest_abby.py` | `SELECTED_PARAMS` | Numeric features | Fixed 0.5 |

The two Logistic Regression implementations are separate entries with their own
member identity and configuration. The PBI 5 adapter rejects Full requests;
exploratory Safe/Full comparisons remain available in the original PBI 2 scripts.
It does not change differing threshold policies to make scores look comparable.

### Input and execution

The only input file required is `features_base_safe.csv` in `--data-dir`
(or `FRAUD_DATA_DIR`, otherwise `data/processed`). These are **prepared features**,
not the raw RDS join. The file includes `country_code`, `order_id`, `customer_id`,
`order_date`, `is_fraud` and numeric feature columns. Shared validation excludes
metadata, labels and `refund_value` from model inputs. The adapter also rejects a
Safe file that contains any excluded snapshot features. Abby's modules retain their original numeric
precision; the other modules retain the shared float32 preparation.

The existing historical split remains unchanged. The team-approved retraining
split and combined RDS feature preparation belong to task #50; running this
interface alone does not cause new orders to enter the training split.

From the repository root using the project's virtual environment:

```powershell
.venv\Scripts\python.exe src/training_interface.py --list-models
.venv\Scripts\python.exe src/training_interface.py --model chu_extra_trees
.venv\Scripts\python.exe src/training_interface.py --model chu_lightgbm --datasets safe
```

Only Safe runs, including when no `--datasets` option is supplied. Requests for
`--datasets full` or both datasets are rejected. `--sample-frac` is optional for development checks
and must retain both classes in every split. Sampled runs cannot enable MLflow.
MLflow is off by default; `--use-mlflow` delegates to the existing member logger.
Shared-server and per-execution logging integration are verified in task #51.

For a caller with `src` on its Python import path:

```python
from training_interface import run_model

results, models = run_model(
    "chu_extra_trees",
    data_dir="data/processed",
    datasets=("safe",),
    output_dir="outputs/task48",
    use_mlflow=False,
)
```

`get_model_config(model_key, params=None)` returns a copy of the member's
configuration. Optional overrides must name existing configuration keys. No
hyperparameter search or manual notebook execution is required.

### Common results and artifacts

`run_model` returns `(results, models)`: a one-item Safe result list and a
dictionary containing the `safe` fitted estimator. Each result includes:

- Model key, algorithm, member, version, parameters, feature names/count and data checksum.
- Split row counts, threshold and its rule, validation/test accuracy, precision,
  recall, F1, average precision, ROC-AUC and TP/TN/FP/FN.
- Training time, test prediction time, additional member metrics and artifact directory.
- Existing member eligibility fields; sampled runs are never made eligible.

Artifacts are saved under
`<output-dir>/<model-key>/<original-run-name>/`: `model.joblib`,
`feature_importance.csv` and normalized `result.json`. Existing member-specific
extras, including local plots where implemented, remain available. Plot sets and
optional training metrics are not required to be identical across modules.
Eligibility fields are review metadata, not automatic registry promotion.

Use a separate output directory for each pipeline execution in the later runner;
reusing the same directory replaces its local artifacts. Batch identifiers and
execution history are tasks #49/#51, not implemented by this single-model adapter.

### Task #48 verification

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_training_interface.py -v
```

The contract tests exercise all eight real member implementations on Safe
synthetic features, without creating any Full input file, using the saved default configurations. They check
member attribution, unchanged settings, threshold policies, metric fields,
separate output paths and prediction equality after reloading all eight saved models.
Additional checks cover parameter-copy isolation, rejection of Full requests,
excluded snapshot features in a Safe input file, and invalid requests.

The tests require no RDS instance or MLflow server. A small verification summary
is written to git-ignored `outputs/task48_contract_verification.json`; temporary
model files are removed by the tests. This verifies the callable interface, not
fraud-model quality, shared cloud logging or the full three-minute ETL requirement.

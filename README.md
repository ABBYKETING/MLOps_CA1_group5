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

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

Existing files without preparation metadata retain the historical split.
Task #50's RDS preparation writes a checked metadata file alongside the CSV;
the shared loader uses its expanding/28-day/28-day split for all eight models.
Running this interface alone does not fetch new records from RDS.

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
The workflow's per-execution logging and verification commands are described in
task #51 below; direct adapter calls retain the member logger's original behaviour.

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
The task #49 workflow below now supplies separate execution directories and status reports.

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

## Automated model retraining workflow (PBI 5, task #49)

`src/retraining_workflow.py` invokes every registered Safe model through the
task #48 interface in one execution. It includes the two algorithms contributed
by each of the four members. No notebook or individual model command is needed.
Models run sequentially, preserving their existing settings and threshold rules.

### One command for all eight models

```powershell
.venv\Scripts\python.exe src/retraining_workflow.py --data-dir data/processed
```

This command requires an already prepared `features_base_safe.csv`. Before any training,
the workflow copies it into the run directory, checks its hash, validates the
features and target, and confirms both classes exist in every split. Task #50's
preparation metadata, when present, is copied and checked with the CSV.
Every model reads that same snapshot, even if the original CSV later changes.
Use `--from-database` below to prepare current RDS records before training.

Options for development or targeted checks:

```powershell
.venv\Scripts\python.exe src/retraining_workflow.py --data-dir data/processed --sample-frac 0.02
.venv\Scripts\python.exe src/retraining_workflow.py --models chu_extra_trees chu_lightgbm
.venv\Scripts\python.exe src/retraining_workflow.py --stop-on-error
```

The default model list contains all eight algorithms. `--models` selects a subset
explicitly. Sampled runs remain development checks and cannot enable MLflow.
Existing member logging can be enabled with `--use-mlflow` and `--tracking-uri`;
task #51 now adds pipeline/model links, execution tags and strict server configuration.

### Calling from another Python stage

With `src` on the import path:

```python
from retraining_workflow import RetrainingFailed, run_retraining

try:
    report = run_retraining(data_dir="data/processed")
except RetrainingFailed as error:
    print(error.summary_path)
    print(error.report["tasks"])
    raise  # stop any subsequent stage that requires successful training
```

This callable entry point can be invoked after successful new-data processing.
Task #50 adds RDS preparation via `--from-database` (see below). The automatic
ETL handoff remains task #52; this runner does not detect arriving data by itself.

### Run status, errors and repeatability

Each invocation receives a generated UTC timestamp/UUID run ID. Optional
`--run-id` values must be safe directory names; an existing run directory is
rejected rather than overwritten. Outputs are stored under:

```text
outputs/retraining/<run-id>/
    summary.json
    input/features_base_safe.csv
    models/<model-key>/<original-run-name>/...
    <failed-model-key>_error.txt       # only when that model fails
    input_error.txt                    # only when input preparation fails
```

`summary.json` records the input hash, row counts, model/member identity, overall
status, per-task status, start/end times, durations, results and artifact paths.
It is updated atomically before and after each model task.

- A task moves from `pending` to `running`, then `completed` or `failed`.
- A model failure records the exception and traceback. Remaining tasks run by
  default so their outcomes can be inspected. The overall run still fails.
- `--stop-on-error` skips the remaining tasks after the first model failure.
- Invalid input skips every training task and records an input-stage failure.
- Any input/model failure raises `RetrainingFailed` with the saved report.
  The CLI exits **1**; success exits **0**; invalid arguments exit **2**.
- Failed task artifacts may be incomplete; a failed report must not be treated
  as a successful pipeline execution or a signal for model promotion.

### Task #49 acceptance evidence

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_retraining_workflow.py -v
```

| Acceptance criterion | Implementation and verification |
|---|---|
| Automatically invoke training scripts | One `run_retraining()` call or CLI command invokes registered functions |
| Include all four members' training tasks | Default registry includes eight Safe models, two per member |
| No manual per-model execution | Real integration test runs all eight through one workflow call |
| Record each task's status | Persistent summary tracks pending, running, completed, failed and skipped states |
| Informative failure messages | Exception details and traceback files are verified by controlled failure tests |
| Repeatable execution | Two real all-model runs are verified, with distinct directories and retained artifacts; reusing a run ID is rejected |

Tests also verify that invalid input prevents training, optional early stopping
skips remaining model tasks, every task reads the same snapshot if the original
file changes, and the CLI reports execution failure with exit code 1.
The integration test uses synthetic Safe data and writes a small verification
summary to `outputs/task49_workflow_verification.json`. It does not assess model
quality, AWS service availability or the final three-minute end-to-end requirement.

## Combined RDS retraining data (PBI 5, task #50)

`src/retraining_data.py` prepares the current combined historical and new labelled
data from RDS. It calls Emi's `database.load_training_data()` once, joining labels
to orders by `(country_code, order_id)` and customers by
`(country_code, customer_id)`. The loader parses both date columns and returns
only labelled orders with matching customer/order details.

PBI 4 already applies the shared cleaning/validation rules before loading RDS.
RDS retains one current record per key: later batches update existing orders,
customers and corrected fraud labels. Preparation reads that combined state;
it does not append the new CSV again, drop duplicate orders silently, or modify
the database. Duplicate order keys in the join stop preparation.

### Agreed chronological retraining split

The latest valid **labelled order date in the data**, rather than today's date,
anchors each run. All orders on the same calendar date stay together:

| Partition | Date range |
|---|---|
| Train | All records before the validation window; history expands over time |
| Validation | The 28 calendar days immediately before the test window |
| Test | The latest 28 calendar days, including the anchor date |

For example, when the latest order date is **31 May 2024**, Train ends on
**5 April**, Validation covers **6 April–3 May**, and Test covers **4–31 May**.
Newer orders initially remain outside model fitting. As later dated orders
arrive, older validation/test records can enter training. Late-arriving records
are assigned by their order date, not the batch arrival date. Rerunning the same
database state gives the same partitions.

An empty partition, invalid order date, or a partition missing either fraud or
non-fraud stops preparation. Windows are never silently widened or replaced by
a random split. Row counts, fraud/non-fraud counts, fraud rates and actual date
ranges are saved so the team can assess whether each window has enough fraud
examples for useful evaluation; having both classes alone does not guarantee
statistically reliable metrics.

All eight models use the same dates and input snapshot in a run. Their existing
settings, preprocessing and validation threshold rules are preserved. No test
rows are passed to model fitting or threshold selection. Test windows change
between runs, so new metrics are not directly comparable to the original PBI 2
scores or to earlier runs on different orders. Historical test records that later
enter training must not be reused as an independent historical benchmark.

### Safe feature preparation and provenance

Preparation reuses `data_processing.build_feature_sets()` with the same PBI 1
feature transformations and fixed category encoding. Only the 31 Safe features
are saved for training. `refund_value` and the ten excluded customer-snapshot
features are absent from prediction inputs. IDs, order dates and the label remain
metadata/target columns, never prediction features.

Each new output directory contains:

```text
features_base_safe.csv
features_base_safe.metadata.json
```

The filename retains `base` for compatibility with the existing model interfaces;
its content is the combined RDS state at preparation time. Keep both files
together. The JSON stores the source type, UTC preparation time, CSV SHA-256,
feature schema, excluded feature names, split boundaries and class counts.
The shared loader verifies these against the actual CSV before attaching the
split policy to the loaded DataFrame. All eight models, including Abby's custom
split functions, use that policy through `create_split_labels()`.

The workflow freezes both files in its run directory, records preparation details
in `summary.json`, and includes them in each local model `result.json`. Existing
directories cannot be overwritten. A database/preparation failure skips every
model and is recorded through the existing workflow error reporting.

RDS holds current records, without historical customer snapshots or historical
label-availability timestamps. This approach cannot reconstruct exactly what was
known at every past prediction time. Safe feature exclusions reduce the identified
risks; they do not prove every remaining feature has point-in-time availability.

### Execution

Set the usual `DB_HOST`, `DB_USER`, `DB_PASSWORD`, optional `DB_PORT` and
`DB_NAME=spbuy` environment variables. RDS must be running and permit your IP.
Do not put credentials in code, commands committed to Git, or preparation metadata.

Prepare a snapshot without training (the output directory must be new):

```powershell
.venv\Scripts\python.exe src/retraining_data.py --output-dir outputs/prepared/demo01
```

Prepare from RDS and run all eight Safe models in one command:

```powershell
.venv\Scripts\python.exe src/retraining_workflow.py --from-database
```

Or train from a previously prepared snapshot and its metadata:

```powershell
.venv\Scripts\python.exe src/retraining_workflow.py --data-dir outputs/prepared/demo01
```

`--from-database` and `--data-dir` are mutually exclusive. Without either flag,
the existing prepared-file default remains available. MLflow is still optional;
these commands do not require it unless `--use-mlflow` is supplied.

For an orchestration caller with `src` on the import path:

```python
from retraining_workflow import run_retraining

report = run_retraining(from_database=True)
```

### Task #50 verification

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_retraining_data.py -v
.venv\Scripts\python.exe -m unittest discover -s tests -p test_database.py -v
```

The new tests exercise the actual shared SQL join against local SQLite and train
all eight real models on synthetic Safe features using the moving split. They
cover composite keys, date boundaries, expanding history after new rows arrive,
corrected labels, excluded features, class counts, unchanged historical defaults,
snapshot preservation, tampered metadata, invalid/insufficient inputs and database
failure reporting. The existing database tests verify join parity and parsed dates.
Local evidence is saved to git-ignored `outputs/task50_data_verification.json`.
These checks do not verify live AWS connectivity, real-data model quality or the
final end-to-end runtime requirement.

### Live RDS verification — 27 November 2026

The read-only preparation check passed against the team's AWS RDS MySQL 8.4.11
instance. It retrieved **949,872** complete labelled orders and saved **31 Safe
features**. Reloading the generated CSV passed the checksum, feature schema,
split boundaries and partition-count validation. Preparation plus CSV reload
took **79.68 seconds** in this execution; this is not an end-to-end pipeline or
all-model training runtime measurement.

| Partition | Actual order dates | Rows | Fraud | Non-fraud |
|---|---|---:|---:|---:|
| Train | 12 November 2022–16 March 2023 | 671,218 | 68,405 | 602,813 |
| Validation | 17 March–13 April 2023 | 134,532 | 18,182 | 116,350 |
| Test | 14 April–11 May 2023 | 144,122 | 17,835 | 126,287 |

The batch log contained only `base` with status `completed`. A separate read-only
preflight of the school's `*_test.csv` files confirmed that none of their order
or label keys was stored yet. At the checked state, ETL would insert 100 customers
and 104 orders; 97 labels have matching parents and 7 would be rejected for
missing customers. These are preflight expectations, not reported database writes.

Local evidence is retained under
`outputs/prepared/rds_check_20261127T094904_155924Z/` in `verification.json` and
`school_batch_preflight.json`, alongside the prepared CSV and metadata.
Outputs remain git-ignored. Live preparation after loading a new school batch
remains to be verified; the existing local tests already cover combined records
and later batches moving the date windows. No RDS rows were changed by these checks.

## Automated MLflow logging (PBI 5, task #51)

The retraining workflow reuses the team's `mlflow_utils.py` and each member's
existing model logger. Enabling `--use-mlflow` creates one parent execution run
and one linked child run for each requested Safe model (eight by default).
Existing model configurations, preprocessing, threshold rules and model-version
labels remain unchanged.

### Shared server configuration

Use the PBI 3 server documented in `mlflow_server/local_deploy/README.md`: a shared
Docker MLflow 3.18.0 server accessed through private Tailscale HTTPS. The host,
Docker and Tailscale must be running. Your individual account needs EDIT access
to the existing `SP-buy-Fraud-Detection` experiment. MLflow credentials are
separate from the AWS RDS credentials.

Set these in the PowerShell session used to run the workflow, replacing the
address and username with the details provided privately by the server owner:

```powershell
$env:MLFLOW_TRACKING_URI = 'https://YOUR_SHARED_TAILSCALE_HOST'
$env:MLFLOW_TRACKING_USERNAME = 'YOUR_INDIVIDUAL_USERNAME'
$env:MLFLOW_TRACKING_PASSWORD = [System.Net.NetworkCredential]::new('', (Read-Host 'MLflow password' -AsSecureString)).Password
```

The workflow requires an explicit configured shared HTTPS address and account.
It checks the server/experiment before data preparation or model training.
Missing settings, connection failures, authentication errors or insufficient
permissions fail the execution; they never silently switch to local storage.
Keep credentials out of URLs, source files and commits.

Prepare the current RDS data and train/log all eight Safe models:

```powershell
.venv\Scripts\python.exe src/retraining_workflow.py --from-database --use-mlflow
```

An existing prepared snapshot can be supplied with `--data-dir` instead of
`--from-database`. Keep its CSV and task #50 metadata together. This mode does not
require RDS to remain running. `--tracking-uri` optionally overrides the tracking
URI environment variable; authentication still comes from environment variables.
Omitting `--use-mlflow` retains local-only training.

### Execution identity and recorded information

The parent run name is `retraining_<pipeline_run_id>`. Child run names retain
`{member}_{algorithm}_safe_{model_version}` and add tags identifying their
`pipeline_run_id`, `model_key`, member, algorithm, model version, tracking location
and optional ETL `batch_name`. MLflow's parent/child links group each execution
in the UI. `v1` continues to describe the existing model configuration; each
newly fitted model has its own new MLflow run ID and model URI.

Each child records:

- Existing model hyperparameters, decision threshold, threshold rule and feature count.
- Validation/Test precision, recall, F1, accuracy, average precision (`pr_auc`),
  ROC-AUC and confusion-matrix counts; existing extra member metrics are retained.
- A trained model in the shared skops format, its signature and approved trust types.
- Validation/Test confusion-matrix and precision-recall plots, feature importance,
  feature names, input provenance and the final model result JSON.

The parent stores input metadata, the execution summary, duration and completed,
failed/skipped model counts. Provenance includes the data checksum, source,
feature schema, row counts and split policy; task #50 inputs also supply the
actual split boundaries and class counts. The full CSV containing customer/order
identifiers is not uploaded. Local summaries include parent/child run IDs,
model URIs and clickable UI links.

Every invocation creates fresh server records. Reusing an existing local run
directory, an existing server pipeline ID, an active MLflow run or
`MLFLOW_RUN_ID` is rejected to prevent accidental resumption/overwriting.
Model/logging failures are recorded; the parent execution is FAILED when any
required task or final logging step fails. Remaining tasks follow the existing
`--stop-on-error` policy. Password/token values are redacted from saved errors.

An optional `--batch-name` identifies the source ETL batch after it has loaded
successfully. `--verification-run` marks a shared-server check as verification,
with `selection_candidate=false`. This prevents synthetic checks from becoming
model-selection candidates. Automatic registration/promotion is not performed.
The original PBI 3 baseline comparison excludes runs tagged with a pipeline ID,
so new retraining runs with the same team names do not invalidate its fixed
16-run historical comparison.

### Verify recorded runs and UI access

After a successful execution, use the printed output directory's summary path:

```powershell
.venv\Scripts\python.exe src/verify_retraining_logging.py --summary outputs/retraining/YOUR_RUN_ID/summary.json
```

This reads the existing server records without creating runs or retraining.
It checks execution/model identities, parameters, metrics, required artifacts,
model descriptors and parent/child membership. It also checks that the server
serves the MLflow UI HTML using the configured account. Evidence is saved beside
the execution in `mlflow_verification.json`.

Open the generated `parent_run_url` in a browser, sign in with your individual
account and expand the child runs. Confirm model metrics/artifacts are visible.
Run another execution and confirm its parent/child IDs differ and the earlier
records remain accessible. HTML/API checks do not replace this visual check on
the actual shared server.

### Isolated local verification

Local HTTP server testing must be explicitly enabled with `--local-mlflow`,
alongside `--use-mlflow` and a loopback HTTP(S) `--tracking-uri`. These runs are
marked as verification and are not selection candidates. SQLite/file URIs are
not accepted as silent replacements for the shared server. Sampled runs still
cannot enable MLflow.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_retraining_tracking.py -v
```

The integration suite starts and stops its own isolated HTTP MLflow server and
temporary SQLite/artifact store. It trains/logs all eight real models twice on
synthetic data, downloads and reloads all eight model artifacts, verifies metrics,
tags, plots, model metadata and unique execution IDs, and exercises configuration,
input, model-artifact and final-report logging failures. It also verifies that
the original shared logger still works outside a pipeline context. No RDS or
shared MLflow instance is contacted. `pytest` is needed separately to run the
existing `tests/test_model_comparison.py` regression suite.

Local evidence is saved to git-ignored `outputs/task51_logging_verification.json`.
For another deployment, repeat the shared-server and browser checks using the
server owner's provided address/account. The completed live check is recorded below.

### Live shared-server verification — 27 November 2026

Task #51 was verified against the team's authenticated shared HTTPS server using
an individual account with EDIT permission on experiment **2**,
`SP-buy-Fraud-Detection`. The existing prepared RDS snapshot supplied **949,872
labelled orders and 31 Safe features**; RDS was not contacted during this check.

| Execution | Parent MLflow run ID | Models logged | Execution time |
|---|---|---:|---:|
| `task51_shared_all_20261127T115644Z` | `78d1407153554b66886e86aa0f593178` | All eight Safe algorithms | 549.63 s |
| `task51_shared_repeat_20261127T115644Z` | `94a3c2e32329444bab78232a8f69306c` | LightGBM repeated on the same snapshot | 81.01 s |

Both executions finished successfully with new parent/child run IDs and trained
model URIs. Read-back checks matched the reported model parameters, thresholds,
Validation/Test metrics, data checksum and required artifacts. A rendered browser
check confirmed the metrics, parameters, artifact trees and confusion-matrix
previews for all eight models and the repeated LightGBM run, plus both execution
pages. These timings cover model execution/logging in this environment, not ETL
or a complete automatically triggered pipeline.

The before/after audit verified that all **23 pre-existing experiment records**
were unchanged, the original **16 official baseline runs** remained available to
the comparison code, and the existing **one registered model, two versions and
champion/challenger aliases** were unchanged. Verification runs have
`selection_candidate=false`; no registry promotion was performed.

The live test exposed a Windows redirected-console issue: MLflow's decorative
Unicode run-link output could raise `UnicodeEncodeError` on cp1252 streams after
successful logging. The workflow now suppresses that SDK output during its
execution and restores the caller's setting afterward; explicit UI URLs remain
in the reports. A strict cp1252 HTTP integration test verifies the fix. The
interrupted attempt remains recorded as FAILED for traceability; the successful
checks use the new execution IDs above. A transient DNS failure during read-back
was resolved by rerunning verification against the completed records, without
retraining the first execution.

The local regression/focused checks total **70 distinct passing tests**, including
all eight models logged twice to an isolated HTTP server and model reloads.
Live evidence and rendered UI screenshots remain local and git-ignored under
`outputs/task51_shared_validation/20261127T114919_011773Z/`, including
`shared_verification.json`, `ui_verification.json`, `before.json` and `after.json`.
No credentials or private tracking hostname are included in these instructions.

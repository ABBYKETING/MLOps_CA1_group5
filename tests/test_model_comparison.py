"""Unit tests for baseline model comparison and selection."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from model_comparison import (
    expected_run_names,
    fetch_official_runs,
)

from model_threshold_evaluation import (
    choose_precision_threshold,
)

from model_selection import (
    select_top_two,
    lock_selection,
)


def make_runs():
    return [
        SimpleNamespace(
            info=SimpleNamespace(
                run_name=name,
                status="FINISHED",
            )
        )
        for name in sorted(expected_run_names())
    ]


def make_client(runs):
    client = Mock()

    client.get_experiment_by_name.return_value = (
        SimpleNamespace(
            name="SP-buy-Fraud-Detection",
            experiment_id="2",
        )
    )

    client.search_runs.return_value = runs

    return client


def make_candidates():
    rows = []

    for i in range(8):
        rows.append({
            "run_name": f"candidate_{i}",
            "run_id": f"run_{i}",
            "model_uri": f"models:/model_{i}",
            "team_member": "test_member",
            "algorithm": f"algorithm_{i}",
            "dataset_version": "safe",
            "dataset_sha256": "test_hash",
            "precision_target": 0.70,
            "target_achieved": True,
            "original_threshold": 0.5,
            "selected_threshold": 0.75,
            "val_precision": 0.75,
            "val_recall": (i + 1) / 10,
            "val_f1": 0.5,
            "val_pr_auc": 0.5,
            "val_alerts": 100,
            "val_tp": 75,
            "val_fp": 25,
            "test_precision": 0.5,
            "test_recall": 0.5,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Official run inventory
# ---------------------------------------------------------

def test_expected_run_names():

    names = expected_run_names()

    assert len(names) == 16
    assert "abby_random_forest_safe_v1" in names
    assert "abby_random_forest_full_v2" in names
    assert "chu_extra_trees_safe_v1" in names

    assert sum("_safe_" in name for name in names) == 8
    assert sum("_full_" in name for name in names) == 8


def test_fetch_all_official_runs():

    runs = make_runs()
    client = make_client(runs)

    result = fetch_official_runs(
        client, "SP-buy-Fraud-Detection"
    )

    assert len(result) == 16


def test_missing_experiment_rejected():

    client = Mock()
    client.get_experiment_by_name.return_value = None

    with pytest.raises(ValueError, match="Experiment not found"):
        fetch_official_runs(client, "missing")


def test_missing_run_rejected():

    client = make_client(make_runs()[:-1])

    with pytest.raises(ValueError, match="Missing or duplicate"):
        fetch_official_runs(client, "SP-buy-Fraud-Detection")


def test_duplicate_run_rejected():

    runs = make_runs()
    runs.append(runs[0])

    client = make_client(runs)

    with pytest.raises(ValueError, match="Missing or duplicate"):
        fetch_official_runs(client, "SP-buy-Fraud-Detection")


def test_unfinished_run_rejected():

    runs = make_runs()
    runs[0].info.status = "RUNNING"

    client = make_client(runs)

    with pytest.raises(ValueError, match="FINISHED"):
        fetch_official_runs(client, "SP-buy-Fraud-Detection")


# ---------------------------------------------------------
# Precision-constrained threshold selection
# ---------------------------------------------------------

def test_threshold_maximises_recall():

    y = np.array([1, 0, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.1])

    threshold = choose_precision_threshold(
        y, probabilities, 0.60
    )

    assert threshold == pytest.approx(0.7)


def test_stricter_precision_target():

    y = np.array([1, 0, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.1])

    threshold = choose_precision_threshold(
        y, probabilities, 0.80
    )

    assert threshold == pytest.approx(0.9)


def test_unachievable_precision_target():

    y = np.array([0, 1, 1])
    probabilities = np.array([0.9, 0.8, 0.7])

    threshold = choose_precision_threshold(
        y, probabilities, 0.90
    )

    assert threshold is None


def test_identical_scores_not_split():

    y = np.array([1, 0, 1])
    probabilities = np.array([0.9, 0.9, 0.2])

    threshold = choose_precision_threshold(
        y, probabilities, 0.80
    )

    assert threshold is None


@pytest.mark.parametrize("target", [0, -0.1, 1.1])
def test_invalid_precision_target(target):

    with pytest.raises(ValueError):
        choose_precision_threshold(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            target,
        )


# ---------------------------------------------------------
# Candidate eligibility and ranking
# ---------------------------------------------------------

def test_top_two_ranked_by_validation_recall():

    candidates = make_candidates()
    selected = select_top_two(candidates)

    assert selected["run_name"].tolist() == [
        "candidate_7",
        "candidate_6",
    ]


def test_test_metrics_do_not_affect_selection():

    candidates = make_candidates()

    candidates.loc[0, "test_precision"] = 1.0
    candidates.loc[0, "test_recall"] = 1.0

    candidates.loc[7, "test_precision"] = 0.0
    candidates.loc[7, "test_recall"] = 0.0

    selected = select_top_two(candidates)

    assert selected["run_name"].tolist() == [
        "candidate_7",
        "candidate_6",
    ]


def test_full_model_rejected():

    candidates = make_candidates()
    candidates.loc[7, "dataset_version"] = "full"

    with pytest.raises(ValueError, match="Only Safe"):
        select_top_two(candidates)


def test_duplicate_candidate_rejected():

    candidates = make_candidates()
    candidates.loc[7, "run_id"] = candidates.loc[6, "run_id"]

    with pytest.raises(ValueError, match="Duplicate"):
        select_top_two(candidates)


def test_wrong_precision_policy_rejected():

    candidates = make_candidates()
    candidates.loc[7, "precision_target"] = 0.60

    with pytest.raises(ValueError, match="precision policy"):
        select_top_two(candidates)


def test_below_precision_target_excluded():

    candidates = make_candidates()
    candidates.loc[7, "val_precision"] = 0.69

    selected = select_top_two(candidates)

    assert selected["run_name"].tolist() == [
        "candidate_6",
        "candidate_5",
    ]


def test_unachievable_candidate_excluded():

    candidates = make_candidates()
    candidates.loc[7, "target_achieved"] = False

    selected = select_top_two(candidates)

    assert selected["run_name"].tolist() == [
        "candidate_6",
        "candidate_5",
    ]


def test_insufficient_eligible_models_rejected():

    candidates = make_candidates()

    candidates.loc[
        candidates.index != 7,
        "target_achieved",
    ] = False

    with pytest.raises(ValueError, match="Fewer than two"):
        select_top_two(candidates)


def test_precision_breaks_recall_tie():

    candidates = make_candidates()

    candidates.loc[6, "val_recall"] = 0.8
    candidates.loc[6, "val_precision"] = 0.80

    selected = select_top_two(candidates)

    assert selected.iloc[0]["run_name"] == "candidate_6"


def test_average_precision_breaks_remaining_tie():

    candidates = make_candidates()

    candidates.loc[6, "val_recall"] = 0.8
    candidates.loc[6, "val_pr_auc"] = 0.65

    selected = select_top_two(candidates)

    assert selected.iloc[0]["run_name"] == "candidate_6"


def test_run_name_breaks_final_tie():

    candidates = make_candidates()

    candidates.loc[6, "val_recall"] = 0.8

    selected = select_top_two(candidates)

    assert selected.iloc[0]["run_name"] == "candidate_6"


# ---------------------------------------------------------
# Locked selection manifest
# ---------------------------------------------------------

def test_selection_manifest_created(tmp_path):

    selected = select_top_two(make_candidates())

    selection = lock_selection(selected, tmp_path)

    path = tmp_path / "top_two_models.json"

    assert path.exists()

    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved == selection
    assert saved["selection_basis"] == "validation_only"
    assert saved["precision_target"] == 0.70
    assert saved["production_eligible"] is False
    assert saved["models"][0]["proposed_role"] == "champion"
    assert saved["models"][1]["proposed_role"] == "challenger"
    assert saved["models"][0]["decision_threshold"] == 0.75

    assert all(
        model["production_eligible"] is False
        for model in saved["models"]
    )


def test_identical_selection_can_be_repeated(tmp_path):

    selected = select_top_two(make_candidates())

    first = lock_selection(selected, tmp_path)
    second = lock_selection(selected, tmp_path)

    assert first == second


def test_existing_selection_cannot_be_overwritten(tmp_path):

    selected = select_top_two(make_candidates())

    lock_selection(selected, tmp_path)

    path = tmp_path / "top_two_models.json"
    original = path.read_bytes()

    changed = selected.copy()
    changed.loc[0, "selected_threshold"] = 0.85

    with pytest.raises(ValueError, match="locked selection differs"):
        lock_selection(changed, tmp_path)

    assert path.read_bytes() == original


def test_invalid_locked_threshold_rejected(tmp_path):

    selected = select_top_two(make_candidates())
    selected.loc[0, "selected_threshold"] = 1.5

    with pytest.raises(ValueError, match="Invalid selected threshold"):
        lock_selection(selected, tmp_path)

    assert not (tmp_path / "top_two_models.json").exists()

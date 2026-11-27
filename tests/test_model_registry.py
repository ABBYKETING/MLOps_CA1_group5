"""Regression tests for baseline model registration and promotion."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from mlflow.entities import LoggedModelStatus

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_registry as registry
import model_promotion as promotion
import model_registry_verification as verification


@pytest.fixture
def selection():
    return registry.load_manifest(registry.DEFAULT_MANIFEST)


@pytest.fixture
def name():
    return registry.REGISTERED_MODEL_NAME


def make_version(name, candidate, number):
    return SimpleNamespace(
        name=name,
        version=str(number),
        model_id=None,
        run_id=candidate["run_id"],
        source=candidate["model_uri"],
        status="READY",
        tags=registry.version_tags(candidate),
    )


@pytest.fixture
def registry_state(selection, name):
    """Create an isolated in-memory registry for unit tests."""

    candidates = selection["models"]

    versions = [
        make_version(name, candidate, index)
        for index, candidate in enumerate(candidates, 1)
    ]

    aliases = {}

    logged_models = {}

    for candidate in candidates:

        model_id = candidate["model_uri"].removeprefix(
            "models:/"
        )

        logged_models[model_id] = SimpleNamespace(
            model_id=model_id,
            source_run_id=candidate["run_id"],
            status=LoggedModelStatus.READY,
            experiment_id="2",
            artifact_location=(
                f"mlflow-artifacts:/2/models/"
                f"{model_id}/artifacts"
            ),
        )

    client = Mock()

    client.search_model_versions.side_effect = (
        lambda **kwargs: list(versions)
    )

    def get_version(model_name, number):

        assert model_name == name

        return next(
            version for version in versions
            if version.version == str(number)
        )

    client.get_model_version.side_effect = get_version

    client.get_logged_model.side_effect = (
        lambda model_id: logged_models[model_id]
    )

    def download_uri(model_name, number):

        version = get_version(model_name, number)

        model_id = version.source.removeprefix(
            "models:/"
        )

        return logged_models[model_id].artifact_location

    client.get_model_version_download_uri.side_effect = (
        download_uri
    )

    client.get_registered_model.side_effect = (
        lambda model_name: SimpleNamespace(
            name=model_name,
            aliases=dict(aliases),
        )
    )

    def assign_alias(model_name, alias, number):

        get_version(model_name, number)
        aliases[alias] = str(number)

    client.set_registered_model_alias.side_effect = (
        assign_alias
    )

    client.get_model_version_by_alias.side_effect = (
        lambda model_name, alias: get_version(
            model_name, aliases[alias]
        )
    )

    return SimpleNamespace(
        client=client,
        versions=versions,
        aliases=aliases,
        logged_models=logged_models,
    )


def test_locked_manifest(selection):

    assert len(selection["models"]) == 2

    assert selection["precision_target"] == 0.70

    assert selection["production_eligible"] is False

    assert [
        model["proposed_role"]
        for model in selection["models"]
    ] == ["champion", "challenger"]

    assert all(
        model["dataset_version"] == "safe"
        for model in selection["models"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_version", "full"),
        ("production_eligible", True),
        ("decision_threshold", -0.1),
        ("decision_threshold", 1.1),
        ("decision_threshold", True),
        ("decision_threshold", float("nan")),
        ("val_precision", 0.69),
        ("val_recall", 0),
        ("rank", 2),
        ("proposed_role", "challenger"),
        ("model_uri", "runs:/invalid/model"),
    ],
)
def test_invalid_manifest_candidate(
    selection, tmp_path, field, value
):

    invalid = copy.deepcopy(selection)
    invalid["models"][0][field] = value

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(invalid),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        registry.load_manifest(path)


def test_duplicate_manifest_run_ids(selection, tmp_path):

    invalid = copy.deepcopy(selection)

    invalid["models"][1]["run_id"] = (
        invalid["models"][0]["run_id"]
    )

    path = tmp_path / "manifest.json"

    path.write_text(
        json.dumps(invalid),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="Duplicate source run IDs"
    ):
        registry.load_manifest(path)


def test_missing_model_id_is_supported(
    selection, registry_state
):
    """Regression: actual MLflow registry returns model_id=None."""

    version = registry_state.versions[0]

    assert version.model_id is None

    registry.verify_version(
        registry_state.client,
        version,
        selection["models"][0],
    )


def test_matching_model_id_is_supported(
    selection, registry_state
):

    candidate = selection["models"][0]
    version = registry_state.versions[0]

    version.model_id = candidate["model_uri"].removeprefix(
        "models:/"
    )

    registry.verify_version(
        registry_state.client,
        version,
        candidate,
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("model_id", "m-wrong", "model ID mismatch"),
        ("source", "models:/m-wrong", "source URI mismatch"),
        ("run_id", "wrong", "source run mismatch"),
        ("status", "FAILED_REGISTRATION", "not READY"),
    ],
)
def test_incorrect_version_identity(
    selection, registry_state, field, value, error
):

    version = registry_state.versions[0]
    setattr(version, field, value)

    with pytest.raises(ValueError, match=error):

        registry.verify_version(
            registry_state.client,
            version,
            selection["models"][0],
        )


def test_incorrect_artifact_location(
    selection, registry_state
):

    registry_state.client.get_model_version_download_uri.side_effect = (
        None
    )

    registry_state.client.get_model_version_download_uri.return_value = (
        "mlflow-artifacts:/incorrect/model"
    )

    with pytest.raises(
        ValueError, match="artifact location mismatch"
    ):

        registry.verify_version(
            registry_state.client,
            registry_state.versions[0],
            selection["models"][0],
        )


@pytest.mark.parametrize(
    "tag,value",
    [
        ("decision_threshold", "0.5"),
        ("production_eligible", "true"),
        ("dataset_version", "full"),
        ("source_model_uri", "models:/m-wrong"),
    ],
)
def test_invalid_version_tags(
    selection, registry_state, tag, value
):

    version = registry_state.versions[0]
    version.tags[tag] = value

    with pytest.raises(ValueError, match="Version tag mismatch"):

        registry.verify_version(
            registry_state.client,
            version,
            selection["models"][0],
        )


def test_missing_version_tag(selection, registry_state):

    version = registry_state.versions[0]

    del version.tags["decision_threshold"]

    with pytest.raises(
        ValueError, match="Version tag mismatch"
    ):

        registry.verify_version(
            registry_state.client,
            version,
            selection["models"][0],
        )


def test_registration_dry_run(
    selection, registry_state, name, monkeypatch
):

    registry_state.versions.pop()

    register = Mock()

    monkeypatch.setattr(
        registry.mlflow, "register_model", register
    )

    plan = registry.plan_registration(
        registry_state.client, name, selection
    )

    assert plan[0][1].version == "1"
    assert plan[1][1] is None

    registry.register_models(
        registry_state.client,
        name,
        plan,
        apply=False,
    )

    register.assert_not_called()


def test_repeat_safe_registration(
    selection, registry_state, name, monkeypatch
):

    register = Mock()

    monkeypatch.setattr(
        registry.mlflow, "register_model", register
    )

    for _ in range(2):

        plan = registry.plan_registration(
            registry_state.client,
            name,
            selection,
        )

        versions = registry.register_models(
            registry_state.client,
            name,
            plan,
            apply=True,
        )

        assert [v.version for v in versions] == ["1", "2"]

    register.assert_not_called()

    assert len(registry_state.versions) == 2


def test_resume_partial_registration(
    selection, registry_state, name, monkeypatch
):

    pending = registry_state.versions.pop()

    def create_model(**kwargs):

        assert kwargs["model_uri"] == pending.source
        assert kwargs["name"] == name
        assert kwargs["tags"] == pending.tags

        registry_state.versions.append(pending)

        return pending

    register = Mock(side_effect=create_model)

    monkeypatch.setattr(
        registry.mlflow, "register_model", register
    )

    plan = registry.plan_registration(
        registry_state.client, name, selection
    )

    versions = registry.register_models(
        registry_state.client,
        name,
        plan,
        apply=True,
    )

    assert [v.version for v in versions] == ["1", "2"]

    register.assert_called_once()

    assert len(registry_state.versions) == 2


def test_duplicate_registered_source_rejected(
    selection, registry_state, name
):

    registry_state.versions.append(
        make_version(name, selection["models"][0], 3)
    )

    with pytest.raises(
        ValueError, match="Duplicate registered versions"
    ):

        registry.plan_registration(
            registry_state.client, name, selection
        )


def test_unexpected_registered_source_rejected(
    selection, registry_state, name
):

    registry_state.versions[1].source = "models:/m-other"

    with pytest.raises(
        ValueError, match="Unexpected registered versions"
    ):

        registry.plan_registration(
            registry_state.client, name, selection
        )


def test_promotion_requires_both_versions(
    selection, registry_state, name
):

    registry_state.versions.pop()

    with pytest.raises(
        ValueError, match="Both selected models"
    ):

        promotion.plan_promotion(
            registry_state.client, name, selection
        )


def test_promotion_dry_run(
    selection, registry_state, name
):

    plan = promotion.plan_promotion(
        registry_state.client, name, selection
    )

    promotion.promote_models(
        registry_state.client, name, plan, apply=False
    )

    assert registry_state.aliases == {}

    registry_state.client.set_registered_model_alias.assert_not_called()


def test_promotion_and_alias_verification(
    selection, registry_state, name
):

    plan = promotion.plan_promotion(
        registry_state.client, name, selection
    )

    promotion.promote_models(
        registry_state.client, name, plan, apply=True
    )

    assert registry_state.aliases == {
        "champion": "1",
        "challenger": "2",
    }

    promotion.verify_aliases(
        registry_state.client, name, selection
    )

    assert len(registry_state.versions) == 2


def test_repeat_safe_promotion(
    selection, registry_state, name
):

    registry_state.aliases.update({
        "champion": "1",
        "challenger": "2",
    })

    plan = promotion.plan_promotion(
        registry_state.client, name, selection
    )

    promotion.promote_models(
        registry_state.client, name, plan, apply=True
    )

    registry_state.client.set_registered_model_alias.assert_not_called()


def test_conflicting_alias_rejected(
    selection, registry_state, name
):

    registry_state.aliases["champion"] = "2"

    with pytest.raises(
        ValueError, match="will not overwrite"
    ):

        promotion.plan_promotion(
            registry_state.client, name, selection
        )

    registry_state.client.set_registered_model_alias.assert_not_called()


def test_alias_change_after_planning_rejected(
    selection, registry_state, name
):

    plan = promotion.plan_promotion(
        registry_state.client, name, selection
    )

    registry_state.aliases["champion"] = "2"

    with pytest.raises(ValueError, match="has changed"):

        promotion.promote_models(
            registry_state.client,
            name,
            plan,
            apply=True,
        )

    assert registry_state.aliases["champion"] == "2"

    registry_state.client.set_registered_model_alias.assert_not_called()


def test_alias_verification_rejects_wrong_version(
    selection, registry_state, name
):

    registry_state.aliases.update({
        "champion": "2",
        "challenger": "1",
    })

    with pytest.raises(
        ValueError, match="unexpected version"
    ):

        promotion.verify_aliases(
            registry_state.client, name, selection
        )


def test_prediction_consistency():

    original = np.array([0.1, 0.4, 0.8, 0.9])

    difference = verification.verify_predictions(
        original,
        original.copy(),
        threshold=0.7,
    )

    assert difference == 0.0


def test_floating_point_difference_accepted():

    original = np.array([0.1, 0.4, 0.8, 0.9])

    registered = original + 1e-15

    difference = verification.verify_predictions(
        original,
        registered,
        threshold=0.7,
    )

    assert 0 < difference < 1e-12


def test_prediction_difference_rejected():

    original = np.array([0.1, 0.4, 0.8])
    registered = np.array([0.1, 0.4, 0.85])

    with pytest.raises(
        ValueError, match="probabilities differ"
    ):

        verification.verify_predictions(
            original, registered, threshold=0.7
        )


def test_threshold_crossing_rejected():

    threshold = 0.7

    original = np.array([threshold - 1e-14])
    registered = np.array([threshold + 1e-14])

    with pytest.raises(
        ValueError, match="classification decisions differ"
    ):

        verification.verify_predictions(
            original, registered, threshold
        )


def test_prediction_shape_mismatch():

    with pytest.raises(
        ValueError, match="Prediction shape mismatch"
    ):

        verification.verify_predictions(
            np.array([0.1, 0.2]),
            np.array([0.1]),
            threshold=0.7,
        )

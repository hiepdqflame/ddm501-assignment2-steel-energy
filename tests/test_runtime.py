"""Failures must not silently reuse corrupt or incompatible evidence."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import joblib
import pytest

from steel_energy.bundle import BundleStore, BundleUnavailable
from steel_energy.integrity import IntegrityError, StudyGuard


def write_bundle(path, value=1):
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump({"value": value}, path / "model.joblib")
    frozen = {
        "model_sha256": hashlib.sha256(
            (path / "model.joblib").read_bytes()
        ).hexdigest(),
        "registered_version": str(value),
        "spec": {"id": "fixture"},
    }
    (path / "frozen.json").write_text(json.dumps(frozen))


def test_cached_bundle_loads_once_for_concurrent_requests(tmp_path):
    write_bundle(tmp_path)
    store = BundleStore(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        objects = list(pool.map(lambda _: store.get()[0], range(32)))
    assert all(obj is objects[0] for obj in objects)
    assert store.status()["load_count"] == 1


def test_valid_version_change_reloads_once(tmp_path):
    write_bundle(tmp_path)
    store = BundleStore(tmp_path)
    assert store.get()[0]["value"] == 1
    write_bundle(tmp_path, 2)
    assert store.get()[0]["value"] == 2
    assert store.status()["load_count"] == 2


def test_manifest_content_detects_version_change_with_identical_metadata(
    tmp_path, monkeypatch
):
    write_bundle(tmp_path)
    store = BundleStore(tmp_path)
    signature = store._stat()
    monkeypatch.setattr(store, "_stat", lambda: signature)
    assert store.get()[0]["value"] == 1
    write_bundle(tmp_path, 2)
    assert store.get()[0]["value"] == 2
    assert store.status()["load_count"] == 2


@pytest.mark.parametrize("damage", ["missing", "checksum", "json"])
def test_corrupted_bundle_is_not_served_from_old_cache(tmp_path, damage):
    write_bundle(tmp_path)
    store = BundleStore(tmp_path)
    store.get()
    if damage == "missing":
        (tmp_path / "model.joblib").unlink()
    elif damage == "checksum":
        (tmp_path / "model.joblib").write_bytes(b"broken")
    else:
        (tmp_path / "frozen.json").write_text("{")
    with pytest.raises(BundleUnavailable):
        store.get()


def test_checksum_valid_but_invalid_serialization_fails_closed(tmp_path):
    write_bundle(tmp_path)
    (tmp_path / "model.joblib").write_bytes(b"not a pickle")
    frozen = json.loads((tmp_path / "frozen.json").read_text())
    frozen["model_sha256"] = hashlib.sha256(b"not a pickle").hexdigest()
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    with pytest.raises(BundleUnavailable):
        BundleStore(tmp_path).get()


def prepared(root):
    directory = root / "end_of_day"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "features.parquet").write_bytes(b"fixture features")
    (directory / "prepared.json").write_text("{}")


def test_guard_detects_completed_artifact_corruption(tmp_path):
    guard = StudyGuard(tmp_path, fingerprint={"runtime": "v1"})
    guard.check()
    prepared(tmp_path)
    guard.checkpoint()
    (tmp_path / "end_of_day/features.parquet").write_bytes(b"changed")
    with pytest.raises(IntegrityError, match="artifact"):
        guard.check()


def test_runtime_change_cannot_reuse_completed_study(tmp_path):
    guard = StudyGuard(tmp_path, fingerprint={"lock": "abc", "numpy": "v1"})
    guard.check()
    prepared(tmp_path)
    guard.checkpoint()
    with pytest.raises(IntegrityError, match="environment"):
        StudyGuard(tmp_path, fingerprint={"lock": "abc", "numpy": "v2"}).check()


def test_unsealed_legacy_evidence_requires_explicit_adoption(tmp_path):
    prepared(tmp_path)
    with pytest.raises(IntegrityError, match="adopt-legacy"):
        StudyGuard(tmp_path, fingerprint={}).check()


def test_failed_partial_stage_can_retry_without_changing_completed_data(tmp_path):
    guard = StudyGuard(tmp_path, fingerprint={})
    guard.check()
    prepared(tmp_path)
    guard.checkpoint()
    partial = tmp_path / "end_of_day/E01-development-predictions.csv"
    partial.write_text("partial attempt")
    guard.check()
    partial.write_text("recomputed attempt")
    guard.checkpoint()
    guard.check()


def test_completed_stage_cannot_gain_an_untracked_file(tmp_path):
    guard = StudyGuard(tmp_path, fingerprint={})
    guard.check()
    prepared(tmp_path)
    guard.checkpoint()
    (tmp_path / "end_of_day/frozen.json").write_text("{}")
    with pytest.raises(IntegrityError, match="unsealed"):
        guard.check()


def test_registry_failure_seals_prepared_stage_and_allows_retry(tmp_path, monkeypatch):
    import sys
    from steel_energy import cli

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["cli", "develop"])
    monkeypatch.setattr("steel_energy.integrity.runtime_fingerprint", lambda: {})

    def failed_development(policy):
        prepared(tmp_path)
        (tmp_path / "end_of_day/E01-development-predictions.csv").write_text("partial")
        raise ConnectionError("MLflow registry unavailable")

    monkeypatch.setattr(cli, "develop", failed_development)
    with pytest.raises(ConnectionError, match="registry unavailable"):
        cli.main()
    StudyGuard(tmp_path, fingerprint={}).check()
    monkeypatch.setattr(cli, "develop", lambda policy: {"retried": True})
    cli.main()
    StudyGuard(tmp_path, fingerprint={}).check()

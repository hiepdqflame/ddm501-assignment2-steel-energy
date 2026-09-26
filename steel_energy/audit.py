"""Consistency checks before explicitly adopting pre-ledger experiment evidence."""

import numpy as np
import pandas as pd

from steel_energy.bundle import BundleStore
from steel_energy.config import ROOT, identity, read_json, settings
from steel_energy.data import build_features, load_meter, temporal_split
from steel_energy.integrity import IntegrityError, digest
from steel_energy.metrics import alert_metrics, regression_metrics
from steel_energy.models import predict
from steel_energy.tracking import environment


def _metrics(actual, expected):
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None, key
        else:
            assert np.isclose(actual[key], value, rtol=1e-10, atol=1e-10), key


def audit_legacy(root):
    """Recompute saved metrics and model parity without fitting or selecting again."""
    checks = []
    runtime = environment()
    try:
        for policy in ("end_of_day", "literal"):
            path = root / policy
            if not (path / "prepared.json").exists():
                continue
            for name in (
                "prepared",
                "development",
                "frozen",
                "final-results",
                "test-opened",
            ):
                file = path / f"{name}.json"
                if file.exists():
                    manifest = read_json(file)
                    assert manifest["identity"] == identity(), f"{name} identity"
                    if "environment" in manifest:
                        for key in ("python", "machine", "packages"):
                            assert manifest["environment"][key] == runtime[key], (
                                f"{name} environment {key}"
                            )
            prepared = read_json(path / "prepared.json")
            assert digest(path / "features.parquet") == prepared["features_sha256"]
            assert (
                digest(ROOT / "data/raw/Steel_industry_data.csv")
                == settings()["dataset_sha256"]
            )
            frame = pd.read_parquet(path / "features.parquet")
            pd.testing.assert_frame_equal(
                frame,
                build_features(
                    load_meter(ROOT / "data/raw/Steel_industry_data.csv", policy)
                ),
            )
            checks.append(f"{policy}: identity, runtime, raw and feature content")
            if (path / "development.json").exists():
                manifest = read_json(path / "development.json")
                results = pd.read_csv(path / "experiment-results.csv")
                folds = pd.read_csv(path / "fold-results.csv")
                assert len(results) == 10 and len(folds) == 30
                for row in manifest["results"]:
                    saved = pd.read_csv(
                        path / f"{row['id']}-development-predictions.csv"
                    )
                    metrics = regression_metrics(
                        saved.energy.to_numpy(), saved.prediction.to_numpy()
                    )
                    _metrics(metrics, {k: row[k] for k in metrics})
                    csv_row = results.loc[results.id.eq(row["id"])].iloc[0]
                    _metrics(metrics, {k: csv_row[k] for k in metrics})
                checks.append(f"{policy}: all ten pooled development metrics")
            if (path / "frozen.json").exists():
                model, frozen = BundleStore(path).get()
                _, valid = temporal_split(frame, *settings()["calibration"])
                values = predict(model, valid, frozen["spec"])
                _metrics(
                    regression_metrics(valid.energy.to_numpy(), values),
                    frozen["calibration_metrics"],
                )
                for row in frozen["calibration_grid"]:
                    _metrics(
                        alert_metrics(
                            valid.energy.to_numpy(),
                            values,
                            frozen["threshold_kwh"],
                            row["margin"],
                        ),
                        {k: v for k, v in row.items() if k != "margin"},
                    )
                checks.append(f"{policy}: frozen checksum and October calibration")
            if (path / "final-results.json").exists():
                final = read_json(path / "final-results.json")
                saved = pd.read_csv(path / "test-predictions.csv")
                _, test = temporal_split(frame, *settings()["test"])
                np.testing.assert_allclose(
                    predict(model, test, frozen["spec"]),
                    saved.prediction,
                    rtol=1e-10,
                    atol=1e-10,
                )
                np.testing.assert_array_equal(
                    test.energy.to_numpy(), saved.energy.to_numpy()
                )
                _metrics(
                    regression_metrics(
                        saved.energy.to_numpy(), saved.prediction.to_numpy()
                    ),
                    final["metrics"],
                )
                _metrics(
                    regression_metrics(
                        saved.energy.to_numpy(), saved.baseline.to_numpy()
                    ),
                    final["baseline_metrics"],
                )
                _metrics(
                    alert_metrics(
                        saved.energy.to_numpy(),
                        saved.prediction.to_numpy(),
                        frozen["threshold_kwh"],
                        frozen["margin_kwh"],
                    ),
                    final["advisory"],
                )
                checks.append(
                    f"{policy}: final predictions, regression and advisory metrics"
                )
        assert checks, "No prepared evidence to adopt"
    except (AssertionError, ValueError, KeyError, OSError) as error:
        raise IntegrityError(f"Legacy audit failed: {error}") from error
    return checks

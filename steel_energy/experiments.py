"""Staged experiments: develop, freeze calibration, then open the locked test."""

import hashlib
import logging
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from threadpoolctl import threadpool_limits

from steel_energy.config import (
    ROOT,
    identity,
    output_root,
    read_json,
    save_json,
    settings,
)
from steel_energy.data import (
    build_features,
    feature_columns,
    load_meter,
    temporal_split,
)
from steel_energy.metrics import alert_metrics, regression_metrics, release_gates
from steel_energy.models import make_model, predict
from steel_energy.tracking import configure, environment, log_metrics

LOG = logging.getLogger(__name__)


def folder(policy: str) -> Path:
    """Partition primary and sensitivity evidence without sharing model selection."""
    if policy not in {"end_of_day", "literal"}:
        raise ValueError("Unknown timestamp policy")
    path = output_root() / policy
    path.mkdir(parents=True, exist_ok=True)
    return path


def checked_manifest(path: Path) -> dict:
    """Refuse to reuse evidence after code, data or configuration changes."""
    manifest = read_json(path)
    if manifest["identity"] != identity():
        raise ValueError(
            "Evidence identity changed. Use a new OUTPUT_DIR; do not overwrite a locked study."
        )
    return manifest


def prepare(policy: str = "end_of_day") -> dict:
    """Validate the exact licensed source and persist a deterministic feature frame."""
    destination = folder(policy)
    raw = ROOT / "data/raw/Steel_industry_data.csv"
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if digest != settings()["dataset_sha256"]:
        raise ValueError("Raw dataset checksum differs from the declared UCI snapshot")
    manifest_path = destination / "prepared.json"
    if manifest_path.exists():
        result = checked_manifest(manifest_path)
        if (
            hashlib.sha256((destination / "features.parquet").read_bytes()).hexdigest()
            != result["features_sha256"]
        ):
            raise ValueError("Prepared features were changed")
        return result
    meter = load_meter(raw, policy)
    features = build_features(meter)
    features.to_parquet(destination / "features.parquet", index=True)
    result = {
        "identity": identity(),
        "policy": policy,
        "raw_rows": len(meter),
        "feature_rows": len(features),
        "features": feature_columns(),
        "features_sha256": hashlib.sha256(
            (destination / "features.parquet").read_bytes()
        ).hexdigest(),
        "first_interval": str(meter.interval_end.min()),
        "last_interval": str(meter.interval_end.max()),
        "timestamp_status": "Unconfirmed interval-end convention; literal sensitivity evaluated separately",
    }
    save_json(manifest_path, result)
    return result


def develop(policy: str = "end_of_day") -> dict:
    """Run all ten configurations on three folds; no November/December scoring."""
    prepare(policy)
    destination, config = folder(policy), settings()
    if (destination / "development.json").exists():
        return checked_manifest(destination / "development.json")
    configure(policy)
    frame = pd.read_parquet(destination / "features.parquet")
    rows, folds = [], []
    for spec in config["models"]:
        LOG.info("%s: evaluating %s", policy, spec["id"])
        predictions, fit_seconds = [], 0.0
        with mlflow.start_run(run_name=f"{spec['id']}-development") as parent:
            mlflow.log_params(
                {**spec, "seed": config["seed"], "timestamp_policy": policy}
            )
            mlflow.set_tags(
                {**identity(), "stage": "development", "test_access": "none"}
            )
            mlflow.log_dict(environment(), "environment.json")
            for start, end in config["folds"]:
                train, valid = temporal_split(frame, start, end)
                cols = feature_columns(spec["features"])
                model = make_model(spec, config["seed"])
                begin = time.perf_counter()
                with threadpool_limits(limits=1):
                    model.fit(train[cols], train.energy)
                    predicted = predict(model, valid, spec)
                elapsed = time.perf_counter() - begin
                fit_seconds += elapsed
                metrics = regression_metrics(valid.energy.to_numpy(), predicted)
                fold = {
                    "id": spec["id"],
                    "month": start[:7],
                    "train_rows": len(train),
                    "last_training_label": str(train.interval_end.max()),
                    "first_issue_time": str(valid.issue_time.min()),
                    "fit_predict_seconds": elapsed,
                    **metrics,
                }
                folds.append(fold)
                with mlflow.start_run(run_name=start[:7], nested=True):
                    mlflow.log_params(
                        {
                            "validation_start": start,
                            "validation_end": end,
                            "train_rows": len(train),
                            "validation_rows": len(valid),
                        }
                    )
                    log_metrics(metrics)
                    mlflow.log_metric("fit_predict_seconds", elapsed)
                part = valid[["interval_start", "energy"]].copy()
                part["prediction"] = predicted
                predictions.append(part)
            joined = pd.concat(predictions)
            metrics = regression_metrics(
                joined.energy.to_numpy(), joined.prediction.to_numpy()
            )
            row = {
                "id": spec["id"],
                "kind": spec["kind"],
                "features": spec["features"],
                **metrics,
                "fit_predict_seconds": fit_seconds,
                "mlflow_run_id": parent.info.run_id,
            }
            rows.append(row)
            joined.to_csv(
                destination / f"{spec['id']}-development-predictions.csv", index=False
            )
            log_metrics(metrics)
            mlflow.log_metric("fit_predict_seconds", fit_seconds)
            mlflow.log_artifact(
                str(destination / f"{spec['id']}-development-predictions.csv")
            )
    results = pd.DataFrame(rows)
    results.to_csv(destination / "experiment-results.csv", index=False)
    pd.DataFrame(folds).to_csv(destination / "fold-results.csv", index=False)
    selected = (
        results.loc[results.kind.ne("baseline")].sort_values(["mae", "id"]).iloc[0]
    )
    baseline = (
        results.loc[results.kind.eq("baseline")].sort_values(["mae", "id"]).iloc[0]
    )
    manifest = {
        "identity": identity(),
        "policy": policy,
        "selected_id": selected.id,
        "baseline_id": baseline.id,
        "selection_metric": "pooled development MAE",
        "results": rows,
        "environment": environment(),
    }
    save_json(destination / "development.json", manifest)
    return manifest


def calibrate() -> dict:
    """Fit only the selected model, calibrate October, and seal the test contract."""
    policy = "end_of_day"
    prepare(policy)
    destination, config = folder(policy), settings()
    development = checked_manifest(destination / "development.json")
    if (destination / "frozen.json").exists():
        return checked_manifest(destination / "frozen.json")
    client = configure(policy)
    frame = pd.read_parquet(destination / "features.parquet")
    train, valid = temporal_split(frame, *config["calibration"])
    spec = next(s for s in config["models"] if s["id"] == development["selected_id"])
    cols = feature_columns(spec["features"])
    model = make_model(spec, config["seed"])
    with threadpool_limits(limits=1):
        model.fit(train[cols], train.energy)
        predicted = predict(model, valid, spec)
    threshold = float(train.energy.quantile(0.9))
    margins = [
        {"margin": m, **alert_metrics(valid.energy.to_numpy(), predicted, threshold, m)}
        for m in config["alert_margins"]
    ]
    eligible = [
        r
        for r in margins
        if r["recall"] is not None
        and r["precision"] is not None
        and r["recall"] >= 0.8
        and r["precision"] >= 0.6
    ]
    margin = min(r["margin"] for r in eligible) if eligible else 0
    joblib.dump(model, destination / "model.joblib")
    with mlflow.start_run(run_name=f"{spec['id']}-frozen-candidate") as run:
        mlflow.log_params(
            {
                **spec,
                "seed": config["seed"],
                "train_rows": len(train),
                "threshold_kwh": threshold,
                "alert_margin_kwh": margin,
            }
        )
        mlflow.set_tags(
            {**identity(), "stage": "calibration", "release": "candidate-only"}
        )
        log_metrics(
            regression_metrics(valid.energy.to_numpy(), predicted), "calibration_"
        )
        mlflow.log_dict(margins, "alert-calibration.json")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(train[cols], model.predict(train[cols])),
            input_example=train[cols].head(2),
            code_paths=[str(ROOT / "steel_energy")],
            pip_requirements=[
                f"{k}=={v}" for k, v in environment()["packages"].items()
            ],
        )
        version = mlflow.register_model(
            f"runs:/{run.info.run_id}/model", config["registered_model"]
        )
        client.set_registered_model_alias(
            config["registered_model"], "candidate", version.version
        )
        client.set_model_version_tag(
            config["registered_model"], version.version, "production_approved", "false"
        )
        frozen = {
            "identity": identity(),
            "spec": spec,
            "baseline_id": development["baseline_id"],
            "threshold_kwh": threshold,
            "margin_kwh": margin,
            "calibration_pass": bool(eligible),
            "calibration_grid": margins,
            "calibration_metrics": regression_metrics(
                valid.energy.to_numpy(), predicted
            ),
            "train_rows": len(train),
            "calibration_rows": len(valid),
            "last_training_label": str(train.interval_end.max()),
            "model_sha256": hashlib.sha256(
                (destination / "model.joblib").read_bytes()
            ).hexdigest(),
            "registered_name": config["registered_model"],
            "registered_version": version.version,
            "run_id": run.info.run_id,
            "environment": environment(),
        }
        mlflow.log_dict(frozen, "frozen-manifest.json")
    save_json(destination / "frozen.json", frozen)
    return frozen


def final_evaluation(confirm: bool = False) -> dict:
    """Score the frozen model without model selection or threshold changes."""
    if not confirm:
        raise ValueError(
            "Locked test requires --confirm-final after development and calibration"
        )
    prepare("end_of_day")
    destination, config = folder("end_of_day"), settings()
    frozen = checked_manifest(destination / "frozen.json")
    if (
        hashlib.sha256((destination / "model.joblib").read_bytes()).hexdigest()
        != frozen["model_sha256"]
    ):
        raise ValueError("Frozen model bytes changed")
    if (destination / "final-results.json").exists():
        return checked_manifest(destination / "final-results.json")
    configure("end_of_day")
    frame = pd.read_parquet(destination / "features.parquet")
    _, test = temporal_split(frame, *config["test"])
    model = joblib.load(destination / "model.joblib")
    baseline_spec = next(
        s for s in config["models"] if s["id"] == frozen["baseline_id"]
    )
    save_json(
        destination / "test-opened.json", {"identity": identity(), "frozen": frozen}
    )
    with threadpool_limits(limits=1):
        predicted = predict(model, test, frozen["spec"])
        baseline = predict(make_model(baseline_spec), test, baseline_spec)
    actual = test.energy.to_numpy()
    metrics, reference = (
        regression_metrics(actual, predicted),
        regression_metrics(actual, baseline),
    )
    high = actual > frozen["threshold_kwh"]
    peak_mae = (
        float(np.abs(actual[high] - predicted[high]).mean()) if high.any() else None
    )
    baseline_peak_mae = (
        float(np.abs(actual[high] - baseline[high]).mean()) if high.any() else None
    )
    advisory = alert_metrics(
        actual, predicted, frozen["threshold_kwh"], frozen["margin_kwh"]
    )
    baseline_advisory = alert_metrics(actual, baseline, frozen["threshold_kwh"], 0)
    output = test[["interval_start", "interval_end", "issue_time", "energy"]].copy()
    output["prediction"], output["baseline"] = predicted, baseline
    output.to_csv(destination / "test-predictions.csv", index=False)
    output["difference"] = np.abs(actual - baseline) - np.abs(actual - predicted)
    daily = output.groupby(output.interval_start.dt.date).difference.mean().to_numpy()
    rng = np.random.default_rng(config["seed"])
    # Paired moving blocks retain within-week dependence better than iid rows.
    bootstrap = []
    for _ in range(2000):
        starts = rng.integers(0, len(daily) - 6, size=int(np.ceil(len(daily) / 7)))
        sample = np.concatenate([daily[s : s + 7] for s in starts])[: len(daily)]
        bootstrap.append(float(sample.mean()))
    low, high_ci = np.quantile(bootstrap, [0.025, 0.975])
    subgroups = []
    for name, mask in {
        "November": test.interval_start.dt.month.eq(11),
        "December": test.interval_start.dt.month.eq(12),
        "weekday": test.interval_start.dt.dayofweek.lt(5),
        "weekend": test.interval_start.dt.dayofweek.ge(5),
        "high_load": high,
    }.items():
        if np.asarray(mask).any():
            subgroups.append(
                {
                    "group": name,
                    **regression_metrics(actual[mask], predicted[mask]),
                    "baseline_mae": regression_metrics(actual[mask], baseline[mask])[
                        "mae"
                    ],
                }
            )
    pd.DataFrame(subgroups).to_csv(destination / "subgroup-results.csv", index=False)
    timings = []
    with threadpool_limits(limits=1):
        for i in range(120):
            start = time.perf_counter()
            predict(model, test.iloc[[i % len(test)]], frozen["spec"])
            timings.append((time.perf_counter() - start) * 1000)
    result = {
        "identity": identity(),
        "selected_id": frozen["spec"]["id"],
        "baseline_id": frozen["baseline_id"],
        "metrics": metrics,
        "baseline_metrics": reference,
        "peak_mae": peak_mae,
        "baseline_peak_mae": baseline_peak_mae,
        "advisory": advisory,
        "baseline_advisory_zero_margin": baseline_advisory,
        "relative_mae_improvement_pct": 100
        * (reference["mae"] - metrics["mae"])
        / reference["mae"],
        "paired_mae_reduction_kwh": float(daily.mean()),
        "moving_block_95_ci_kwh": [float(low), float(high_ci)],
        "inference_p95_ms": float(np.quantile(timings[10:], 0.95)),
        "latency_scope": "warm single-row model call; excludes ingestion, HTTP and scheduling",
        "gates": release_gates(
            metrics["mae"],
            reference["mae"],
            peak_mae,
            baseline_peak_mae,
            advisory,
            frozen["calibration_pass"],
        ),
        "environment": environment(),
    }
    with mlflow.start_run(run_name="locked-test-evaluation"):
        mlflow.set_tags(
            {**identity(), "stage": "locked-test", "candidate_run_id": frozen["run_id"]}
        )
        log_metrics(metrics, "test_")
        log_metrics(reference, "baseline_")
        mlflow.log_dict(result, "final-results.json")
        mlflow.log_artifact(str(destination / "test-predictions.csv"))
        mlflow.log_artifact(str(destination / "subgroup-results.csv"))
    save_json(destination / "final-results.json", result)
    return result

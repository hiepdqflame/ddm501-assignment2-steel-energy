"""Interpretable metrics and conservative release criteria."""

import numpy as np


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    """Return errors in kWh, with undefined percentage denominators as null."""
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if (
        y.shape != prediction.shape
        or y.size == 0
        or not np.isfinite([y, prediction]).all()
    ):
        raise ValueError("Metrics require equally sized finite, nonempty arrays")
    error = prediction - y
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape_pct": float(100 * np.abs(error).sum() / y.sum()) if y.sum() else None,
        "bias": float(error.mean()),
        "n": int(len(y)),
    }


def alert_metrics(
    y: np.ndarray, prediction: np.ndarray, threshold: float, margin: float
) -> dict:
    """Evaluate an advisory against a frozen development-period threshold."""
    actual = np.asarray(y) > threshold
    alert = np.asarray(prediction) + margin > threshold
    tp = int((actual & alert).sum())
    return {
        "precision": tp / int(alert.sum()) if alert.any() else None,
        "recall": tp / int(actual.sum()) if actual.any() else None,
        "true_positives": tp,
        "false_positives": int((~actual & alert).sum()),
        "false_negatives": int((actual & ~alert).sum()),
        "high_load_count": int(actual.sum()),
        "alert_count": int(alert.sum()),
    }


def release_gates(
    mae: float,
    baseline_mae: float,
    peak_mae: float | None,
    baseline_peak_mae: float | None,
    advisory: dict,
    calibration_pass: bool,
) -> dict:
    """Assess the predeclared gates without automatically approving deployment."""
    forecast = bool(
        mae <= 5
        and mae <= 0.9 * baseline_mae
        and peak_mae is not None
        and baseline_peak_mae is not None
        and peak_mae <= baseline_peak_mae
    )
    alert = bool(
        calibration_pass
        and advisory.get("recall") is not None
        and advisory.get("precision") is not None
        and advisory["recall"] >= 0.8
        and advisory["precision"] >= 0.6
    )
    return {
        "forecast_pass": forecast,
        "advisory_pass": alert,
        "offline_gates_pass": forecast and alert,
        "production_eligible": False,
        "reason": "Production additionally requires a local pilot and human approval",
    }

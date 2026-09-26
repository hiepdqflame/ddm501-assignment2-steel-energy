"""A strict historical-replay interface; no machine-control authority."""

import hashlib

import joblib
import numpy as np
import pandas as pd

from steel_energy.config import output_root, read_json
from steel_energy.data import build_features, load_meter
from steel_energy.models import predict


def predict_request(model, frozen: dict, payload: dict) -> dict:
    """Reject incomplete, late or future-contaminated meter histories."""
    start, issued = (
        pd.Timestamp(payload["target_start"]),
        pd.Timestamp(payload["issued_at"]),
    )
    cutoff = start - pd.Timedelta(minutes=15)
    if start.tzinfo is not None or issued.tzinfo is not None:
        raise ValueError("Benchmark request uses naive Asia/Seoul local timestamps")
    if start.minute % 15 or start.second or issued != cutoff + pd.Timedelta(minutes=1):
        raise ValueError("Invalid target alignment or issue deadline")
    history = pd.DataFrame(payload["history"])
    if not {"interval_end", "energy"}.issubset(history.columns) or len(history) < 672:
        raise ValueError("At least 672 consecutive completed readings are required")
    history["interval_end"] = pd.to_datetime(history.interval_end)
    if (
        history.interval_end.dt.tz is not None
        or history.interval_end.max() != cutoff
        or not history.interval_end.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
        or not np.isfinite(history.energy).all()
        or history.energy.lt(0).any()
    ):
        raise ValueError("Nonconsecutive, invalid or future-contaminated history")
    # Unknown intervals only align the target row; feature construction excludes them.
    future = pd.DataFrame(
        {
            "interval_end": [start, start + pd.Timedelta(minutes=15)],
            "energy": [0.0, 0.0],
        }
    )
    features = build_features(pd.concat([history.tail(672), future], ignore_index=True))
    value = float(predict(model, features.tail(1), frozen["spec"])[0])
    return {
        "target_start": str(start),
        "target_end": str(start + pd.Timedelta(minutes=15)),
        "issued_at": str(issued),
        "forecast_kwh": value,
        "average_kw": 4 * value,
        "threshold_exceeded": value + frozen["margin_kwh"] > frozen["threshold_kwh"],
        "advisory_enabled": False,
        "calibration_pass": frozen["calibration_pass"],
        "model_version": frozen["registered_version"],
        "model_sha256": frozen["model_sha256"],
        "production_approved": False,
        "mode": "historical-benchmark-replay",
    }


def load_bundle():
    """Only deserialize the locally generated, checksum-verified model."""
    destination = output_root() / "end_of_day"
    frozen = read_json(destination / "frozen.json")
    path = destination / "model.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen["model_sha256"]:
        raise ValueError("Model checksum mismatch")
    return joblib.load(path), frozen


def demo_request(raw_path) -> dict:
    """Create a replay input solely from readings preceding the target window."""
    meter = load_meter(raw_path)
    cutoff = pd.Timestamp("2018-10-31 23:45")
    history = meter.loc[meter.interval_end.le(cutoff)].tail(672).copy()
    history["interval_end"] = history.interval_end.astype(str)
    return {
        "target_start": "2018-11-01 00:00",
        "issued_at": "2018-10-31 23:46",
        "history": history.to_dict("records"),
    }

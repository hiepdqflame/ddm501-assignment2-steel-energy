"""Serving must enforce the same information boundary as the experiment."""

import numpy as np
import pandas as pd
import pytest

from steel_energy.models import LagRegressor
from steel_energy.serving import predict_request


def request():
    ends = pd.date_range(end="2018-10-31 23:45", periods=672, freq="15min")
    return {
        "target_start": "2018-11-01 00:00",
        "issued_at": "2018-10-31 23:46",
        "history": [
            {"interval_end": str(t), "energy": float(i)} for i, t in enumerate(ends)
        ],
    }


def frozen():
    return {
        "spec": {"id": "E01", "features": "full"},
        "threshold_kwh": 700,
        "margin_kwh": 0,
        "calibration_pass": False,
        "registered_version": "1",
        "model_sha256": "fixture",
    }


def test_prediction_uses_latest_completed_energy_and_never_authorizes_control():
    result = predict_request(LagRegressor(2), frozen(), request())
    assert result["forecast_kwh"] == 671
    assert result["average_kw"] == 2684
    assert result["advisory_enabled"] is False
    assert result["production_approved"] is False


@pytest.mark.parametrize(
    "mutation", ["future", "missing", "nan", "duplicate", "late_issue"]
)
def test_invalid_serving_request_is_rejected(mutation):
    payload = request()
    if mutation == "future":
        payload["history"][-1]["interval_end"] = "2018-11-01 00:00"
    elif mutation == "missing":
        payload["history"].pop(50)
    elif mutation == "nan":
        payload["history"][50]["energy"] = np.nan
    elif mutation == "duplicate":
        payload["history"][50] = payload["history"][51]
    else:
        payload["issued_at"] = "2018-11-01 00:01"
    with pytest.raises(ValueError):
        predict_request(LagRegressor(2), frozen(), payload)

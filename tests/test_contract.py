"""Regression tests for information leakage, time boundaries and metric validity."""

import numpy as np
import pandas as pd
import pytest

from steel_energy.data import (
    build_features,
    feature_columns,
    load_meter,
    temporal_split,
)
from steel_energy.metrics import alert_metrics, regression_metrics, release_gates


def meter_frame(n=800):
    end = pd.date_range("2018-01-01 00:15", periods=n, freq="15min")
    return pd.DataFrame({"interval_end": end, "energy": np.arange(n, dtype=float)})


def test_source_midnight_is_last_interval_of_previous_day(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(
        "date,Usage_kWh\n01/01/2018 23:45,3\n01/01/2018 00:00,4\n02/01/2018 00:15,5\n"
    )
    frame = load_meter(path, "end_of_day")
    assert frame.interval_end.tolist() == list(
        pd.to_datetime(["2018-01-01 23:45", "2018-01-02 00:00", "2018-01-02 00:15"])
    )


@pytest.mark.parametrize(
    "values",
    [
        "01/01/2018 00:15,-1\n",
        "01/01/2018 00:15,NaN\n",
        "01/01/2018 00:15,2\n01/01/2018 00:15,3\n",
        "01/01/2018 00:15,2\n01/01/2018 00:45,3\n",
    ],
)
def test_invalid_meter_data_fails_closed(tmp_path, values):
    path = tmp_path / "bad.csv"
    path.write_text("date,Usage_kWh\n" + values)
    with pytest.raises(ValueError):
        load_meter(path, "end_of_day")


def test_lags_and_rolling_windows_respect_two_interval_gap():
    row = build_features(meter_frame()).loc[700]
    assert row["lag_2"] == 698
    assert row["lag_96"] == 604
    assert row["lag_672"] == 28
    assert row["mean_4"] == 696.5
    assert row["interval_start"] - row["issue_time"] == pd.Timedelta(minutes=14)


def test_future_and_unfinished_readings_cannot_change_prediction_features():
    before = meter_frame()
    after = before.copy()
    after.loc[699:, "energy"] = 999999
    cols = feature_columns("full")
    pd.testing.assert_series_equal(
        build_features(before).loc[700, cols], build_features(after).loc[700, cols]
    )


def test_warmup_and_feature_ablation_are_consistent():
    frame = build_features(meter_frame())
    assert frame.index.min() == 672
    assert len(frame) == 128
    assert "lag_96" not in feature_columns("short")
    assert "mean_4" not in feature_columns("short")
    assert not {"energy", "Load_Type", "CO2(tCO2)", "lag_1"}.intersection(
        feature_columns("full")
    )


def test_training_purges_label_not_available_at_first_validation_issue():
    frame = build_features(meter_frame(96 * 220))
    train, valid = temporal_split(frame, "2018-07-01", "2018-08-01")
    assert valid.interval_start.min() == pd.Timestamp("2018-07-01 00:00")
    assert train.interval_end.max() == pd.Timestamp("2018-06-30 23:45")
    assert (
        train.interval_end + pd.Timedelta(seconds=30) <= valid.issue_time.min()
    ).all()


def test_metrics_are_hand_checked_and_zero_denominators_are_undefined():
    metrics = regression_metrics(np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert metrics["mae"] == 1
    assert metrics["rmse"] == 1
    assert metrics["wape_pct"] == 100
    assert regression_metrics(np.zeros(2), np.ones(2))["wape_pct"] is None
    assert alert_metrics(np.zeros(2), np.zeros(2), 5, 0)["precision"] is None
    assert alert_metrics(np.zeros(2), np.zeros(2), 5, 0)["recall"] is None


def test_release_is_blocked_when_either_forecast_or_advisory_gate_fails():
    gates = release_gates(4.0, 5.0, 3.0, 4.0, {"recall": 0.9, "precision": 0.3}, True)
    assert gates["forecast_pass"] is True
    assert gates["advisory_pass"] is False
    assert gates["production_eligible"] is False

"""Meter validation and features with an explicit prediction-time boundary."""

from pathlib import Path

import numpy as np
import pandas as pd

LAGS = [2, 3, 4, 5, 6, 9, 17, 96, 672]
WINDOWS = [4, 16, 96]
CALENDAR = ["slot_sin", "slot_cos", "weekday_sin", "weekday_cos", "weekend"]


def load_meter(path: Path, policy: str = "end_of_day") -> pd.DataFrame:
    """Validate interval energy; retain the source file's byte-level provenance.

    ``end_of_day`` interprets the source's trailing midnight as next-day 00:00.
    ``literal`` sorts the literal timestamps for the declared sensitivity check.
    Neither convention is claimed to be a confirmed production meter contract.
    """
    raw = pd.read_csv(path)
    if not {"date", "Usage_kWh"}.issubset(raw.columns) or raw.empty:
        raise ValueError("Missing meter columns or empty file")
    times = pd.to_datetime(raw["date"], format="%d/%m/%Y %H:%M", errors="raise")
    energy = pd.to_numeric(raw["Usage_kWh"], errors="raise")
    if not np.isfinite(energy).all() or (energy < 0).any() or times.duplicated().any():
        raise ValueError("Invalid energy or duplicate timestamps")
    if policy == "end_of_day":
        midnight = (times.dt.hour == 0) & (times.dt.minute == 0)
        times = times + pd.to_timedelta(midnight.astype(int), unit="D")
    elif policy != "literal":
        raise ValueError(f"Unknown timestamp policy: {policy}")
    frame = pd.DataFrame({"interval_end": times, "energy": energy})
    if policy == "literal":
        frame = frame.sort_values("interval_end")
    frame = frame.reset_index(drop=True)
    if (
        frame.interval_end.duplicated().any()
        or not frame.interval_end.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
    ):
        raise ValueError("Expected a complete, unique, ordered 15-minute grid")
    return frame


def feature_columns(feature_set: str = "full") -> list[str]:
    """Return a whitelist; target and contemporaneous measurements cannot enter."""
    if feature_set not in {"full", "short"}:
        raise ValueError("Unknown feature set")
    lags = LAGS if feature_set == "full" else LAGS[:-2]
    return (
        CALENDAR
        + [f"lag_{n}" for n in lags]
        + ([f"mean_{n}" for n in WINDOWS] if feature_set == "full" else [])
    )


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct a target-end horizon of 30 minutes with a two-interval gap."""
    out = frame.copy()
    out["interval_start"] = out.interval_end - pd.Timedelta(minutes=15)
    out["issue_time"] = out.interval_end - pd.Timedelta(minutes=29)
    slot = out.interval_start.dt.hour * 4 + out.interval_start.dt.minute / 15
    weekday = out.interval_start.dt.dayofweek
    out["slot_sin"] = np.sin(2 * np.pi * slot / 96)
    out["slot_cos"] = np.cos(2 * np.pi * slot / 96)
    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    out["weekend"] = (weekday >= 5).astype(float)
    for lag in LAGS:
        out[f"lag_{lag}"] = out.energy.shift(lag)
    for window in WINDOWS:
        out[f"mean_{window}"] = out.energy.shift(2).rolling(window).mean()
    return out.dropna(subset=feature_columns()).copy()


def temporal_split(
    frame: pd.DataFrame, start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Purge training labels that would not have arrived at the first issue time."""
    valid = frame.loc[
        frame.interval_start.ge(start) & frame.interval_start.lt(end)
    ].copy()
    if valid.empty:
        raise ValueError("Empty evaluation window")
    train = frame.loc[
        frame.interval_start.lt(start)
        & (frame.interval_end + pd.Timedelta(seconds=30)).le(valid.issue_time.min())
    ].copy()
    if train.empty:
        raise ValueError("Empty training window")
    return train, valid

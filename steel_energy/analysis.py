"""Post-freeze diagnostics: descriptive test errors, October-only importance."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from steel_energy.bundle import BundleStore
from steel_energy.config import output_root, read_json, save_json, settings
from steel_energy.data import temporal_split
from steel_energy.integrity import digest
from steel_energy.models import predict


def analyze():
    root = output_root()
    source, target = root / "end_of_day", root / "engineering"
    target.mkdir(parents=True, exist_ok=True)
    model, frozen = BundleStore(source).get()
    final = read_json(source / "final-results.json")
    frame = pd.read_parquet(source / "features.parquet")
    saved = pd.read_csv(source / "test-predictions.csv", parse_dates=["interval_start"])
    _, test = temporal_split(frame, *settings()["test"])
    np.testing.assert_allclose(saved.energy, test.energy, rtol=0, atol=0)
    np.testing.assert_array_equal(
        saved.interval_start.to_numpy(), test.interval_start.to_numpy()
    )
    saved["ramp_kwh"] = saved.energy.to_numpy() - test.lag_2.to_numpy()
    saved["signed_error_kwh"] = saved.prediction - saved.energy
    saved["missed"] = (saved.energy > frozen["threshold_kwh"]) & (
        saved.prediction + frozen["margin_kwh"] <= frozen["threshold_kwh"]
    )
    missed = saved.loc[saved.missed].copy()
    assert len(missed) == final["advisory"]["false_negatives"]
    runs = saved.missed.ne(saved.missed.shift()).cumsum()
    episodes = saved.loc[saved.missed].groupby(runs).size()
    hourly = (
        saved.assign(
            hour=saved.interval_start.dt.hour,
            high=saved.energy > frozen["threshold_kwh"],
        )
        .groupby("hour")
        .agg(missed=("missed", "sum"), high_load=("high", "sum"))
    )
    missed.to_csv(target / "missed-peaks.csv", index=False)
    hourly.to_csv(target / "missed-peaks-by-hour.csv")
    diagnostics = {
        "scope": "Descriptive locked-test analysis only; no model or alert retuning",
        "missed_intervals": len(missed),
        "missed_episodes": len(episodes),
        "longest_missed_episode_intervals": int(episodes.max()) if len(episodes) else 0,
        "mean_signed_error_kwh": float(missed.signed_error_kwh.mean())
        if len(missed)
        else None,
        "median_ramp_from_last_available_kwh": float(missed.ramp_kwh.median())
        if len(missed)
        else None,
        "rising_misses": int(missed.ramp_kwh.gt(0).sum()),
        "hourly": hourly.reset_index().to_dict("records"),
        "largest_misses": missed.sort_values("signed_error_kwh")
        .head(10)
        .assign(interval_start=lambda x: x.interval_start.astype(str))
        .to_dict("records"),
    }
    _, october = temporal_split(frame, *settings()["calibration"])
    groups = {
        "Time of day": ["slot_sin", "slot_cos"],
        "Day of week": ["weekday_sin", "weekday_cos", "weekend"],
        "Recent energy (30-90 min)": [f"lag_{i}" for i in (2, 3, 4, 5, 6)],
        "Earlier energy (135-255 min)": ["lag_9", "lag_17"],
        "Previous day": ["lag_96"],
        "Previous week": ["lag_672"],
        "Rolling means": ["mean_4", "mean_16", "mean_96"],
    }
    rng = np.random.default_rng(42)
    block_size, repeats = 16, 5
    blocks = np.arange(len(october)).reshape(-1, block_size)
    importance = []
    with threadpool_limits(limits=1):
        base = float(
            np.abs(predict(model, october, frozen["spec"]) - october.energy).mean()
        )
        for label, columns in groups.items():
            deltas = []
            for _ in range(repeats):
                changed = october.copy()
                order = blocks[rng.permutation(len(blocks))].ravel()
                changed[columns] = october[columns].to_numpy()[order]
                deltas.append(
                    float(
                        np.abs(
                            predict(model, changed, frozen["spec"]) - october.energy
                        ).mean()
                    )
                    - base
                )
            importance.append(
                {
                    "group": label,
                    "mae_increase_kwh": float(np.mean(deltas)),
                    "repeat_std_kwh": float(np.std(deltas)),
                    "repeats": repeats,
                }
            )
    importance.sort(key=lambda x: x["mae_increase_kwh"], reverse=True)
    result = {
        "diagnostics": diagnostics,
        "importance": {
            "scope": "October calibration only; frozen model; grouped 4-hour block permutation, 5 repeats, seed 42. Correlated features and off-manifold inputs limit causal interpretation; repeat SD is not a confidence interval.",
            "baseline_mae": base,
            "rows": importance,
        },
        "provenance": {
            name: digest(source / name)
            for name in (
                "frozen.json",
                "model.joblib",
                "test-predictions.csv",
                "final-results.json",
            )
        },
    }
    save_json(target / "analysis.json", result)
    save_json(
        target / "dashboard.json",
        {
            **result,
            "final": final,
            "model": {
                key: frozen[key]
                for key in (
                    "spec",
                    "registered_name",
                    "registered_version",
                    "model_sha256",
                    "threshold_kwh",
                    "margin_kwh",
                    "calibration_pass",
                )
            },
            "experiments": pd.read_csv(source / "experiment-results.csv").to_dict(
                "records"
            ),
            "series": saved.assign(interval_start=saved.interval_start.astype(str))[
                ["interval_start", "energy", "prediction", "baseline", "missed"]
            ].to_dict("records"),
        },
    )
    return result

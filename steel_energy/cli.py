"""Staged CLI shared by Docker, Airflow and local development."""

import argparse
import fcntl
import json
import logging

import mlflow.sklearn
import numpy as np

from steel_energy.config import ROOT, output_root, save_json
from steel_energy.data import build_features, load_meter, feature_columns
from steel_energy.experiments import (
    calibrate,
    checked_manifest,
    develop,
    final_evaluation,
    prepare,
)
from steel_energy.models import predict
from steel_energy.plots import export_figures
from steel_energy.serving import demo_request, load_bundle, predict_request
from steel_energy.tracking import configure
from steel_energy.integrity import StudyGuard


def verify() -> dict:
    """Load the actual registered model and verify round-trip prediction parity."""
    root = output_root()
    prepare()
    final = checked_manifest(root / "end_of_day/final-results.json")
    model, frozen = load_bundle()
    configure("end_of_day")
    registered = mlflow.sklearn.load_model(
        f"models:/{frozen['registered_name']}/{frozen['registered_version']}"
    )
    features = build_features(
        load_meter(ROOT / "data/raw/Steel_industry_data.csv")
    ).head(20)
    expected = predict(model, features, frozen["spec"])
    actual = np.maximum(
        0, registered.predict(features[feature_columns(frozen["spec"]["features"])])
    )
    if not np.allclose(expected, actual, rtol=1e-10, atol=1e-10):
        raise ValueError("Registry serialization changed predictions")
    payload = demo_request(ROOT / "data/raw/Steel_industry_data.csv")
    save_json(root / "demo-request.json", payload)
    save_json(root / "demo-prediction.json", predict_request(model, frozen, payload))
    result = {
        "registry_round_trip": "passed",
        "compared_rows": 20,
        "feature_rows": 34368,
        "test_rows": final["metrics"]["n"],
        "model_version": frozen["registered_version"],
        "production_approved": False,
    }
    save_json(root / "verification.json", result)
    from steel_energy.analysis import analyze

    analyze()
    return result


def main() -> None:
    """Run an explicit stage; serialize writers and preserve frozen experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "develop",
            "calibrate",
            "final",
            "plots",
            "verify",
            "all",
            "adopt-legacy",
            "analyze",
        ],
    )
    parser.add_argument(
        "--policy", choices=["end_of_day", "literal"], default="end_of_day"
    )
    parser.add_argument("--confirm-final", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.stage in {"all", "final"} and not args.confirm_final:
        parser.error(
            "Use --confirm-final to reproduce the predeclared locked-test evaluation"
        )
    with (output_root() / ".pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        guard = StudyGuard(output_root())
        if args.stage == "adopt-legacy":
            print(json.dumps(guard.adopt_legacy(), indent=2))
            return
        guard.check()

        def stage(function, *arguments):
            try:
                return function(*arguments)
            finally:
                # A completed predecessor remains sealed even when a later step fails.
                guard.checkpoint()

        if args.stage == "all":
            stage(prepare)
            stage(develop)
            if args.sensitivity:
                stage(develop, "literal")
            stage(calibrate)
            result = stage(final_evaluation, True)
            stage(export_figures)
            stage(verify)
        elif args.stage == "prepare":
            result = stage(prepare, args.policy)
        elif args.stage == "develop":
            result = stage(develop, args.policy)
        elif args.stage == "calibrate":
            result = stage(calibrate)
        elif args.stage == "final":
            result = stage(final_evaluation, args.confirm_final)
        elif args.stage == "plots":
            stage(export_figures)
            result = {"figures": str(output_root() / "figures")}
        elif args.stage == "analyze":
            from steel_energy.analysis import analyze

            result = stage(analyze)
        else:
            result = stage(verify)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

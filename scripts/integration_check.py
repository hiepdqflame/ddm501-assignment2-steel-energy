"""Real HTTP and registry checks using disposable fixtures, never study evidence."""

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

from steel_energy.config import ROOT, save_json
from steel_energy.data import build_features, load_meter, feature_columns
from steel_energy.models import LagRegressor
from steel_energy.serving import demo_request


def http(base, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base + route, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read()
            return response.status, json.loads(raw) if "json" in response.headers.get(
                "Content-Type", ""
            ) else raw
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def main():
    with tempfile.TemporaryDirectory(prefix="steel-integration-") as directory:
        root = Path(directory)
        dest = root / "end_of_day"
        dest.mkdir()
        joblib.dump(LagRegressor(2), dest / "model.joblib")
        frozen = {
            "model_sha256": hashlib.sha256(
                (dest / "model.joblib").read_bytes()
            ).hexdigest(),
            "registered_version": "smoke-fixture",
            "spec": {"id": "fixture", "features": "full"},
            "threshold_kwh": 80,
            "margin_kwh": 0,
            "calibration_pass": False,
        }
        save_json(dest / "frozen.json", frozen)
        payload = demo_request(ROOT / "data/raw/Steel_industry_data.csv")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        with (root / "api.log").open("w+") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "steel_energy.api:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                env={**os.environ, "OUTPUT_DIR": str(root)},
                stdout=log,
                stderr=log,
            )
            try:
                for _ in range(100):
                    try:
                        if http(base, "/health")[0] == 200:
                            break
                    except OSError:
                        pass
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    time.sleep(0.1)
                assert http(base, "/health")[0] == 200
                assert http(base, "/")[0] == 200
                assert http(base, "/evidence")[0] == 503
                with ThreadPoolExecutor(max_workers=8) as pool:
                    responses = list(
                        pool.map(lambda _: http(base, "/predict", payload), range(24))
                    )
                assert all(
                    code == 200 and not body["production_approved"]
                    for code, body in responses
                )
                assert all(
                    body["forecast_kwh"] == payload["history"][-1]["energy"]
                    for _, body in responses
                )
                assert http(base, "/health")[1]["load_count"] == 1
                invalid = json.loads(json.dumps(payload))
                invalid["history"][-1]["interval_end"] = invalid["target_start"]
                assert http(base, "/predict", invalid)[0] == 422
                invalid["history"] = []
                assert http(base, "/predict", invalid)[0] == 422
                (dest / "model.joblib").write_bytes(b"damaged")
                assert http(base, "/health")[0] == 503
                assert http(base, "/predict", payload)[0] == 503
                joblib.dump(LagRegressor(3), dest / "model.joblib")
                frozen["model_sha256"] = hashlib.sha256(
                    (dest / "model.joblib").read_bytes()
                ).hexdigest()
                frozen["registered_version"] = "recovered-fixture"
                save_json(dest / "frozen.json", frozen)
                code, result = http(base, "/predict", payload)
                assert (
                    code == 200
                    and result["forecast_kwh"] == payload["history"][-2]["energy"]
                )
                assert http(base, "/health")[1]["load_count"] == 2
            finally:
                process.terminate()
                process.wait(timeout=15)

        # A failed registry call must raise, then the real endpoint must still work.
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"
        try:
            MlflowClient(tracking_uri="http://127.0.0.1:1").search_experiments()
        except mlflow.exceptions.MlflowException:
            pass
        else:
            raise AssertionError("Unavailable registry did not fail")
        uri = os.environ["MLFLOW_TRACKING_URI"]
        mlflow.set_tracking_uri(uri)
        name = "integration-smoke-" + uuid.uuid4().hex[:10]
        mlflow.set_experiment(name)
        frame = build_features(
            load_meter(ROOT / "data/raw/Steel_industry_data.csv")
        ).head(20)
        model = LagRegressor(2)
        with mlflow.start_run() as run:
            mlflow.set_tag(
                "purpose", "disposable integration fixture; not reported study"
            )
            mlflow.log_metric("fixture_rows", len(frame))
            mlflow.sklearn.log_model(
                model, "model", pip_requirements=["scikit-learn==1.7.2"]
            )
            version = mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
        restored = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
        np.testing.assert_array_equal(
            model.predict(frame[feature_columns()]),
            restored.predict(frame[feature_columns()]),
        )
        client = MlflowClient()
        client.delete_registered_model(name)
        client.delete_experiment(client.get_experiment_by_name(name).experiment_id)
        print(
            json.dumps(
                {
                    "http_concurrent_requests": 24,
                    "cached_loads_before_update": 1,
                    "bad_request_422": "passed",
                    "corruption_503_and_recovery": "passed",
                    "registry_unavailability_and_recovery": "passed",
                    "registry_round_trip": "passed",
                    "fixture_scope": "temporary integration only",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

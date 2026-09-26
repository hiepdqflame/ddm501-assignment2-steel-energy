"""Measure real HTTP round trips on the Compose network, including request parsing."""

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from steel_energy.config import output_root, save_json
from steel_energy.tracking import environment


def main():
    root = output_root()
    payload = (root / "demo-request.json").read_bytes()
    url = os.getenv("API_URL", "http://api:8000/predict")
    expected = json.loads((root / "demo-prediction.json").read_text())["forecast_kwh"]

    def once(_):
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        begin = time.perf_counter()
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        elapsed = (time.perf_counter() - begin) * 1000
        assert abs(result["forecast_kwh"] - expected) < 1e-8
        return elapsed

    for i in range(10):
        once(i)
    rows = []
    for concurrency in (1, 4, 8):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            durations = list(pool.map(once, range(100)))
        rows.append(
            {
                "concurrency": concurrency,
                "requests": len(durations),
                "p50_ms": float(np.median(durations)),
                "p95_ms": float(np.quantile(durations, 0.95)),
                "max_ms": max(durations),
                "requests_per_second": len(durations) / (time.perf_counter() - start),
            }
        )
    result = {
        "scope": "Warm HTTP round trip, Compose network, 672 readings/request, one Uvicorn worker. Includes parsing, features, model and transport; excludes meter ingestion. Hardware-specific observation, not an SLA.",
        "warmup_requests": 10,
        "payload_bytes": len(payload),
        "rows": rows,
        "environment": environment(),
    }
    save_json(root / "engineering/http-benchmark.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

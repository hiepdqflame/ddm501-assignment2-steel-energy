"""Exercise actual HTTP serving and compare it with the saved replay prediction."""

import json
import os
import urllib.request
from pathlib import Path

root = Path(os.getenv("OUTPUT_DIR", "/app/artifacts/run"))
expected = json.loads((root / "demo-prediction.json").read_text())
request = urllib.request.Request(
    os.getenv("API_URL", "http://api:8000/predict"),
    data=(root / "demo-request.json").read_bytes(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    actual = json.load(response)
if abs(actual["forecast_kwh"] - expected["forecast_kwh"]) > 1e-8:
    raise SystemExit("HTTP prediction differs from the saved model")
print(json.dumps({"http_prediction_parity": "passed", "response": actual}, indent=2))

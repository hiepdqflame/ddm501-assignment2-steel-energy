"""Real MLflow tracking with local SQLite or a Compose tracking service."""

import importlib.metadata
import os
import platform
import subprocess

import mlflow
from mlflow import MlflowClient

from steel_energy.config import ROOT, output_root, settings


def configure(policy: str) -> MlflowClient:
    """Create an experiment whose artifact location is portable over HTTP."""
    root = output_root()
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{root / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    name = f"{settings()['experiment_name']}-{policy}"
    client = MlflowClient()
    if client.get_experiment_by_name(name) is None:
        if uri.startswith("http"):
            client.create_experiment(name)
        else:
            client.create_experiment(
                name, artifact_location=(root / "mlruns" / policy).as_uri()
            )
    mlflow.set_experiment(name)
    return client


def environment() -> dict:
    """Capture measured runtime provenance; do not invent a Git commit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        commit = result.stdout.strip() if result.returncode == 0 else "not-in-git"
    except FileNotFoundError:
        commit = "git-not-installed; use source_sha256"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": commit,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "mlflow", "scipy", "joblib")
        },
    }


def log_metrics(metrics: dict, prefix: str = "") -> None:
    """MLflow cannot store null metrics; JSON artifacts retain their meaning."""
    mlflow.log_metrics(
        {
            prefix + k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (float, int)) and v is not None
        }
    )

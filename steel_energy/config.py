"""Portable configuration and content-addressed experiment identity."""

import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def settings() -> dict:
    """Load YAML and explicit environment overrides; never embed host paths."""
    path = Path(os.getenv("CONFIG_PATH", ROOT / "config/experiment.yaml"))
    config = yaml.safe_load(path.read_text())
    if len(config["models"]) != 10 or len({m["id"] for m in config["models"]}) != 10:
        raise ValueError("Expected ten unique predeclared configurations")
    config["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def output_root() -> Path:
    """Resolve the writeable experiment volume in local and Docker execution."""
    path = Path(os.getenv("OUTPUT_DIR", ROOT / "artifacts/run")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def identity() -> dict:
    """Hash model-relevant source and configuration, independent of filesystem paths."""
    files = [
        ROOT / "steel_energy" / f"{name}.py"
        for name in ("data", "metrics", "models", "experiments", "config", "tracking")
    ]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    config = settings()
    return {
        "source_sha256": digest.hexdigest(),
        "config_sha256": config["config_sha256"],
        "data_sha256": config["dataset_sha256"],
    }


def save_json(path: Path, value: dict | list) -> None:
    """Atomically replace a small stage manifest after successful computation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n"
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    """Read a manifest; a missing predecessor fails the stage explicitly."""
    return json.loads(path.read_text())

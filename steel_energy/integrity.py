"""Supplemental integrity ledger; original experiment manifests stay unchanged."""

import hashlib
import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path

from steel_energy.config import ROOT, identity, read_json, save_json


class IntegrityError(RuntimeError):
    """A completed study cannot be safely reused."""


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_fingerprint():
    packages = {}
    for line in (ROOT / "requirements.lock").read_text().splitlines():
        if "==" in line:
            name = line.split("==", 1)[0]
            packages[name] = importlib.metadata.version(name)
    try:
        packages["greenlet"] = importlib.metadata.version("greenlet")
    except importlib.metadata.PackageNotFoundError:
        packages["greenlet"] = "not-installed"
    return {
        "lock_sha256": digest(ROOT / "requirements.lock"),
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "identity": identity(),
    }


class StudyGuard:
    """Seal successful stages while allowing incomplete stages to be retried."""

    def __init__(self, root, fingerprint=None):
        self.root = Path(root)
        self.path = self.root / "integrity-ledger.json"
        self.fingerprint = runtime_fingerprint() if fingerprint is None else fingerprint

    def _completed(self, names_only=False):
        files = []
        stages = {
            "prepared.json": ["features.parquet"],
            "development.json": ["experiment-results.csv", "fold-results.csv"]
            + [f"E{i:02}-development-predictions.csv" for i in range(1, 11)],
            "frozen.json": ["model.joblib"],
            "test-opened.json": [],
            "final-results.json": [
                "test-opened.json",
                "test-predictions.csv",
                "subgroup-results.csv",
            ],
        }
        for policy in ("end_of_day", "literal"):
            for manifest, companions in stages.items():
                if (self.root / policy / manifest).exists():
                    files.extend(f"{policy}/{name}" for name in [manifest, *companions])
        if names_only:
            return set(files)
        try:
            return {name: digest(self.root / name) for name in sorted(set(files))}
        except OSError as error:
            raise IntegrityError(f"Completed artifact is missing: {error}") from error

    def _verify(self, ledger):
        if ledger["environment"] != self.fingerprint:
            raise IntegrityError("Study environment changed; use a new OUTPUT_DIR")
        for name, expected in ledger["artifacts"].items():
            try:
                if digest(self.root / name) != expected:
                    raise IntegrityError(f"Completed artifact changed: {name}")
            except OSError as error:
                raise IntegrityError(f"Completed artifact missing: {name}") from error

    def check(self):
        current = self._completed(names_only=True)
        if not self.path.exists():
            if current:
                raise IntegrityError(
                    "Existing unsealed study; use explicit adopt-legacy to audit it"
                )
            save_json(
                self.path,
                {"schema": 1, "environment": self.fingerprint, "artifacts": {}},
            )
        ledger = read_json(self.path)
        self._verify(ledger)
        if set(current) != set(ledger["artifacts"]):
            raise IntegrityError(
                "Unexpected unsealed completed artifact; refusing cached reuse"
            )
        return ledger

    def checkpoint(self):
        ledger = read_json(self.path)
        self._verify(ledger)
        ledger["artifacts"] = self._completed()
        save_json(self.path, ledger)

    def adopt_legacy(self):
        """Explicit consistency audit, not proof of historical authenticity."""
        if self.path.exists():
            return self.check()
        from steel_energy.audit import audit_legacy

        audit = audit_legacy(self.root)
        ledger = {
            "schema": 1,
            "environment": self.fingerprint,
            "artifacts": self._completed(),
            "adoption": {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "origin": "explicit legacy audit; no historical manifests rewritten",
                "checks": audit,
            },
        }
        save_json(self.path, ledger)
        return ledger

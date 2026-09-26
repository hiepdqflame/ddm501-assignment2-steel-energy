"""Thread-safe, fail-closed cache for a trusted, locally produced model bundle."""

import hashlib
import json
from pathlib import Path
from threading import RLock

import joblib


class BundleUnavailable(RuntimeError):
    """The serving bundle is absent, changing, or corrupt."""


class BundleStore:
    """Verify once per file revision, then share the immutable model in memory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._lock = RLock()
        self._signature = None
        self._bundle = None
        self._loads = 0

    def _stat(self):
        return tuple(
            (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            for s in (
                (self.directory / name).stat()
                for name in ("frozen.json", "model.joblib")
            )
        )

    def get(self):
        """Never return the previous model after a damaged update is detected."""
        with self._lock:
            try:
                # Some filesystems give rapid writes identical timestamps. The small
                # publication manifest must be compared by content on every access.
                signature = (
                    self._stat(),
                    (self.directory / "frozen.json").read_bytes(),
                )
                if signature == self._signature and self._bundle is not None:
                    return self._bundle
                frozen = json.loads(signature[1])
                path = self.directory / "model.joblib"
                # Hash and deserialize the same open file, then detect concurrent writes.
                with path.open("rb") as stream:
                    if (
                        hashlib.file_digest(stream, "sha256").hexdigest()
                        != frozen["model_sha256"]
                    ):
                        raise ValueError("Model checksum mismatch")
                    stream.seek(0)
                    model = joblib.load(stream)
                if signature != (
                    self._stat(),
                    (self.directory / "frozen.json").read_bytes(),
                ):
                    raise ValueError(
                        "Bundle changed during loading; retry after publication"
                    )
                self._bundle = (model, frozen)
                self._signature = signature
                self._loads += 1
                return self._bundle
            except Exception as error:
                self._bundle, self._signature = None, None
                raise BundleUnavailable(
                    "Frozen bundle unavailable: " + str(error)
                ) from error

    def status(self):
        with self._lock:
            return {"load_count": self._loads, "cached": self._bundle is not None}

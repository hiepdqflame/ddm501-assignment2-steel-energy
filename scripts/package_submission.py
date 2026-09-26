"""Package portable source and evidence, excluding machine-local runtime state."""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "output/submission"
NAME = "DDM501_Assignment2_25MS13293_DoQuangHiep"


def main() -> None:
    """Write a self-contained submission whose fresh artifact directory is empty."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = []
    excluded = {
        ".venv",
        ".venv-airflow",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "artifacts",
        "mlruns",
    }
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if (
            path.is_file()
            and not excluded.intersection(relative.parts)
            and path.suffix not in {".pyc", ".zip"}
        ):
            if path.name not in {".env", ".DS_Store"}:
                paths.append((path, relative))
    manifest = {
        str(relative): hashlib.sha256(path.read_bytes()).hexdigest()
        for path, relative in paths
    }
    with ZipFile(OUTPUT / f"{NAME}.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path, relative in paths:
            archive.write(path, f"{NAME}/{relative}")
        archive.writestr(f"{NAME}/artifacts/.gitkeep", "")
        archive.writestr(
            f"{NAME}/SUBMISSION_SHA256.json", json.dumps(manifest, indent=2) + "\n"
        )
    print(OUTPUT / f"{NAME}.zip")
    print(
        f"Packaged {len(paths)} files; no virtual environments, model caches or registry databases"
    )


if __name__ == "__main__":
    main()

"""Restore the pinned source exports used by the archived recovery scripts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent


def relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Expected a relative archive path without '..'; got {value!r}.")
    return Path(*path.parts)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_archive(manifest: dict) -> None:
    for name, record in manifest["files"].items():
        path = HERE / relative_path(name)
        if digest(path) != record["sha256"]:
            raise ValueError(f"Expected the recorded SHA-256 for archived file {name!r}.")


def restore(repository: Path, name: str, spec: dict, files: dict) -> None:
    target = HERE / relative_path(name)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Expected an unused source-export path: {target}")
    revision = spec["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"Expected a full pinned Git commit: {revision!r}")
    # Create only a new destination. A failed export is retained for inspection.
    with tempfile.TemporaryFile() as archive_file:
        subprocess.run(
            ["git", "-C", str(repository), "archive", "--format=tar", revision],
            stdout=archive_file,
            check=True,
        )
        archive_file.seek(0)
        target.mkdir(parents=True)
        with tarfile.open(fileobj=archive_file, mode="r:") as archive:
            archive.extractall(target, filter="data")
    for destination, source in spec.get("overlays", {}).items():
        origin = HERE / relative_path(source)
        if digest(origin) != files[source]["sha256"]:
            raise ValueError(f"Expected the recorded overlay hash: {source}")
        output = target / relative_path(destination)
        if output.is_symlink():
            raise ValueError(f"Expected a regular overlay destination: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, output)
    for relative, expected in spec.get("checks", {}).items():
        if digest(target / relative_path(relative)) != expected:
            raise ValueError(f"Restored source hash differs: {name}/{relative}")
    receipt = target / "RECOVERY_SOURCE.json"
    receipt.write_text(json.dumps({"export": name, **spec}, indent=2) + "\n")
    print(f"Restored {name} at {revision}", flush=True)


def main() -> None:
    manifest = json.loads((HERE / "source_manifest.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify archived file hashes only")
    parser.add_argument("--only", action="append", choices=sorted(manifest["exports"]))
    parser.add_argument("--repository", type=Path, default=HERE.parents[1])
    args = parser.parse_args()
    verify_archive(manifest)
    if args.check:
        print(f"Verified {len(manifest['files'])} archived source/protocol files")
        return
    names = args.only or list(manifest["exports"])
    # Check every destination before starting, so an existing export never gets replaced.
    for name in names:
        target = HERE / relative_path(name)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Expected an unused source-export path: {target}")
    for name in names:
        restore(args.repository.resolve(), name, manifest["exports"][name], manifest["files"])


if __name__ == "__main__":
    main()

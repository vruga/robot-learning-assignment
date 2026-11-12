"""Verify installed upstream code, assets and frozen controller weights."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from .protocol import UPSTREAM_COMMIT


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_integrity(root: Path | None = None) -> dict:
    import mujoco

    if root is None:
        spec = importlib.util.find_spec("bigym")
        if spec is None or spec.origin is None:
            raise RuntimeError("BiGym2 is not installed; run uv sync --locked")
        root = Path(spec.origin).parent
    manifest = json.loads(Path(__file__).with_name("upstream_manifest.json").read_text())
    failures = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            failures.append(name)
    if mujoco.__version__ != "3.8.1":
        failures.append(f"mujoco version {mujoco.__version__}, expected 3.8.1")
    return {
        "ok": not failures,
        "upstream_commit": UPSTREAM_COMMIT,
        "verified_files": len(manifest["files"]),
        "failures": failures,
        "manifest_sha256": digest(Path(__file__).with_name("upstream_manifest.json")),
    }


def submission_digest(root: Path) -> str:
    """Identify candidate code/config/checkpoints without hashing environments or outputs."""
    ignored = {".git", ".venv", "venv", "__pycache__", "results", "datasets", ".pytest_cache"}
    suffixes = {
        ".py",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".pt",
        ".pth",
        ".onnx",
        ".safetensors",
        ".npz",
        ".lock",
    }
    value = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and not path.is_symlink() and not ignored.intersection(relative.parts):
            if path.suffix in suffixes or (
                path.name.startswith("requirements") and path.suffix == ".txt"
            ):
                value.update(str(relative).encode())
                value.update(digest(path).encode())
    return value.hexdigest()


def main():
    result = check_integrity()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)

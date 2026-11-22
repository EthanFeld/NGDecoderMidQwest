"""Atomic reports, portable single-writer locking, and resume fingerprints."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_reports(output: Path, rows: list[dict[str, Any]]) -> None:
    write_json(output / "results.json", rows)
    buffer = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_text(output / "results.csv", buffer.getvalue())


class OutputLock:
    """OS releases this advisory lock even when the owner is killed."""

    def __init__(self, output: Path):
        self.path = output / ".run.lock"
        self.file: Any = None

    def __enter__(self) -> OutputLock:
        self.file = self.path.open("a+b")
        self.file.seek(0, os.SEEK_END)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.file.close()
            raise ValueError(f"another run owns output directory: {self.path.parent}") from error
        return self

    def __exit__(self, *_: Any) -> None:
        if self.file:
            self.file.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()


def _hash_files(files: list[Path], root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(files):
        hasher.update(path.relative_to(root).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def implementation_identity(decoders: tuple[str, ...]) -> dict[str, Any]:
    root = Path(__file__).parent
    result = {"qecbench_source": _hash_files(list(root.glob("*.py")), root)}
    for spec in decoders:
        if spec in ("pymatching", "zero"):
            continue
        module = importlib.import_module(spec.partition(":")[0])
        package_name = (module.__package__ or module.__name__).split(".")[0]
        package = importlib.import_module(package_name)
        if hasattr(package, "__path__"):
            paths = [Path(p) for p in package.__path__]
            hashes = [_hash_files(list(p.rglob("*.py")), p) for p in paths]
            result[spec] = digest(hashes)
        elif getattr(module, "__file__", None):
            result[spec] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        else:
            raise ValueError(f"decoder {spec!r} needs an on-disk module for resume fingerprinting")
        dependency_identity = getattr(module, "qecbench_implementation_identity", None)
        if callable(dependency_identity):
            result[spec] = {"source": result[spec], "dependencies": dependency_identity()}
    return result


def environment_identity() -> dict[str, Any]:
    versions = {}
    for name in ("numpy", "scipy", "stim", "pymatching", "relay-bp", "ngdecoder-bench"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "source"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_identifier": os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "versions": versions,
    }

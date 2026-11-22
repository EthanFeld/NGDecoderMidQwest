"""Load validated Windows binaries or compile the bundled native source."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

NATIVE = Path(__file__).parent / "native"
VENDOR = NATIVE / "vendor/pymatching_2_4_0"


def sources(kind):
    if kind == "support":
        return [NATIVE / "frontier.c", NATIVE / "path.c"]
    if kind == "routing":
        return [NATIVE / "path_bucket.c"]
    return [
        NATIVE / "matching.cc",
        *sorted(VENDOR.rglob("*.h")),
        *sorted(VENDOR.rglob("*.inl")),
        *sorted((VENDOR / "src").rglob("*.cc")),
    ]


def source_digest(kind):
    return hashlib.sha256(b"".join(p.read_bytes() for p in sources(kind))).hexdigest()


def _library(kind, force=False):
    digest = source_digest(kind)
    manifest_path = NATIVE / "prebuilt/windows/manifest.json"
    if (
        not force
        and kind in ("support", "matching")
        and os.environ.get("TENSOR_DECODER_USE_PREBUILT", "1") != "0"
        and os.name == "nt"
        and ctypes.sizeof(ctypes.c_void_p) == 8
        and platform.machine().lower() in ("amd64", "x86_64")
        and manifest_path.exists()
    ):
        record = json.loads(manifest_path.read_text())[kind]
        prebuilt = manifest_path.parent / record["file"]
        if record["source_sha256"] == digest and prebuilt.exists():
            if hashlib.sha256(prebuilt.read_bytes()).hexdigest() != record["binary_sha256"]:
                raise RuntimeError(f"validated native binary integrity failure: {prebuilt}")
            return prebuilt
    project = Path(__file__).resolve().parents[2]
    default_cache = (
        project / "build"
        if (project / "pyproject.toml").exists()
        else Path.home() / ".cache/tensor_decoder"
    )
    cache = Path(os.environ.get("TENSOR_DECODER_CACHE", str(default_cache)))
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"{kind}_{digest[:16]}{'.dll' if os.name == 'nt' else '.so'}"
    if output.exists() and not force:
        return output
    use_c = kind != "matching"
    tool = "gcc" if use_c else "g++"
    compiler = shutil.which(
        os.environ.get("CC" if use_c else "CXX", tool)
    ) or shutil.which(f"C:/msys64/mingw64/bin/{tool}.exe")
    if compiler is None:
        raise RuntimeError(
            "Native build needs GCC and C++20; set CC/CXX or install MSYS2 on Windows."
        )
    env = os.environ.copy()
    env["PATH"] = str(Path(compiler).parent) + os.pathsep + env.get("PATH", "")
    flags = ["-O3", "-shared"]
    if use_c:
        flags += ["-std=c11", "-Wall", "-Wextra"]
        flags += ["-static-libgcc"] if os.name == "nt" else ["-fPIC"]
        inputs = sources(kind)
    else:
        flags += ["-DNDEBUG", "-std=c++20", "-fno-strict-aliasing"]
        flags += ["-static"] if os.name == "nt" else ["-fPIC"]
        flags += ["-I", str(VENDOR / "src")]
        inputs = [NATIVE / "matching.cc", *sorted((VENDOR / "src").rglob("*.cc"))]
    handle, name = tempfile.mkstemp(dir=cache, suffix=output.suffix)
    os.close(handle)
    temporary = Path(name)
    try:
        subprocess.run(
            [
                compiler,
                *flags,
                *(str(p) for p in inputs),
                "-o",
                str(temporary),
                *(["-lm"] if use_c else []),
            ],
            check=True,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def support_library():
    override = os.environ.get("TENSOR_DECODER_SUPPORT_LIBRARY")
    return Path(override) if override else _library("support")


def matching_library():
    return _library("matching")


def routing_library():
    return _library("routing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="compile from source instead of using validated binaries",
    )
    args = parser.parse_args()
    for kind in ("support", "matching"):
        print(_library(kind, args.force))


if __name__ == "__main__":
    main()

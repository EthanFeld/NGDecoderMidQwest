"""Regenerate the paired deck comparison with separate accuracy and timing phases."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SCRIPT = ROOT / "past_experiments/comparisons/sweep.py"
spec = importlib.util.spec_from_file_location("deck_model_sweep", SCRIPT)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)

import numpy as np
import pymatching
import stim
from pymatching_tensor_decoder import TensorDecoder, build_decoder, qecbench_implementation_identity
from surface_decoders.ibm_relay_bp import qecbench_implementation_identity as relay_identity

# The archived harness supplies paired sampling, physical front-end replay and
# timing aggregation. All tensor construction is explicitly the restored backend.
helper.TensorDecoder = TensorDecoder
helper.build_decoder = build_decoder
helper.qecbench_implementation_identity = qecbench_implementation_identity

RATES = (.001, .002, .003, .004, .005, .006, .007, .0075, .008, .009, .010)
POINTS = [(15, p, 2048) for p in RATES] + [(7, .003, 8192), (7, .0075, 8192)]
SEED_BASE = 94623701
TIME_SHOTS = 192
FPGA_MODE = "fpga_only_zero_on_defer"


def utc():
    return datetime.now(timezone.utc).isoformat()


def name_for(distance, p):
    return f"d{distance}-p{round(p * 1_000_000):05d}"


def point_seed(distance, p):
    return SEED_BASE + distance * 1_000_000 + round(p * 1_000_000)


def relay_count(distance, p, n):
    return 64 if distance == 15 else (n if p == .003 else 1024)


def load_rows(directory):
    rows = []
    for path in sorted((directory / "checkpoints").glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def settings_for(profiles):
    return {
        "circuit": "Surface Telescope joint rotated Z, d rounds, circuit-level noise",
        "source": "cycle_model",
        "backend_namespace": "pymatching_tensor_decoder",
        "points": [{"distance": d, "p": p, "shots": n, "seed": point_seed(d, p),
                    "relay_shots": relay_count(d, p, n)} for d, p, n in POINTS],
        "seed_base": SEED_BASE,
        "requested_time_shots": TIME_SHOTS,
        "batch_size": 256,
        "telescope_batch_timing_prefix": 1024,
        "relay_requested_time_shots_d15": 24,
        "fpga_only_defer_policy": "explicit host reporting proxy of logical zero; no software decode",
        "latency": "model core cycles plus measured host projection/completion; excludes UART, queue wait and sampling",
        "timing_sample": "random accepted/deferred strata weighted back to cohort shares; Relay uses its accuracy prefix",
        "phase_separation": "all accuracy cohorts finish before separately scheduled single-shot timing",
        "versions": {"python": sys.version, "host": platform.platform(), "numpy": np.__version__,
                     "stim": stim.__version__, "pymatching": pymatching.__version__,
                     "relay_bp": importlib.metadata.version("relay-bp")},
        "tensor": qecbench_implementation_identity(),
        "relay": relay_identity(),
        "script_sha256": helper.sha256(Path(__file__)),
        "harness_sha256": helper.sha256(SCRIPT),
        "profiles": {str(d): profile.identity() for d, profile in profiles.items()},
    }


def save_status(directory, phase, **fields):
    helper.atomic_json(directory / "status.json", {"phase": phase, "pid": os.getpid(),
                                                   "updated_utc": utc(), **fields})


def accuracy_point(directory, settings, profile, d, p, n):
    point = name_for(d, p)
    seed = point_seed(d, p)
    context, mapping = helper.build_context(profile, p)
    events, truth = context.circuit.compile_detector_sampler(seed=seed).sample(
        n, bit_packed=True, separate_observables=True,
    )
    truth = np.asarray(truth, np.uint8).reshape(n)
    events.flags.writeable = False
    front = profile.front().decode(helper.project(events, context, mapping))
    artifact = directory / "cohorts" / f"{point}.npz"

    def capture_without_timing(decoders, ctx, indices, rows, accepted, fpga, core,
                               link, prediction, count, time_seed, relay_time_shots):
        clocks = getattr(front, "cycles", None)
        if clocks is None:
            clocks = front.clocks
        np.savez_compressed(
            artifact, events=rows, truth=truth, memory_indices=indices,
            accepted=accepted, fpga_logical=np.asarray(front.logical, np.uint8),
            cycles=np.asarray(clocks, np.int32),
            **{f"prediction_{key}": value for key, value in prediction.items()},
        )
        return {key: (None, 0) for key in prediction}, {}, 0

    original_timing = helper.timing
    helper.timing = capture_without_timing
    started = perf_counter()
    try:
        rows = helper.compare_point(
            profile, context, mapping, events, truth, front, TIME_SHOTS, 256,
            seed=seed, relay_shots=relay_count(d, p, n),
            relay_time_shots=24 if d == 15 else TIME_SHOTS,
            telescope_batch_time_shots=1024,
        )
    finally:
        helper.timing = original_timing
    for row in rows:
        if row["mode"] in ("fpga_only", FPGA_MODE):
            row["mode"] = FPGA_MODE
        row["accuracy_completed_utc"] = utc()
        row["timing_complete"] = False
        row["cohort_npz"] = artifact.relative_to(ROOT).as_posix()
        row["cohort_npz_sha256"] = helper.sha256(artifact)
    helper.atomic_json(directory / "checkpoints" / f"{point}.json", rows)
    helper.write_tables(directory, load_rows(directory), settings)
    print(json.dumps({"phase": "accuracy", "point": point, "seconds": perf_counter() - started,
                      "results": [{k: r[k] for k in ("mode", "errors", "shots")} for r in rows]}), flush=True)


def timing_point(directory, settings, profile, d, p):
    point = name_for(d, p)
    checkpoint = directory / "checkpoints" / f"{point}.json"
    records = json.loads(checkpoint.read_text(encoding="utf-8"))
    context, mapping = helper.build_context(profile, p)
    with np.load(directory / "cohorts" / f"{point}.npz", allow_pickle=False) as data:
        events = data["events"]
        accepted = data["accepted"]
        fpga = data["prediction_fpga_only"]
        core = data["cycles"].astype(float) / profile.clock_mhz
        prediction = {key.removeprefix("prediction_"): data[key] for key in data.files
                      if key.startswith("prediction_")}
        if not np.array_equal(mapping, data["memory_indices"]):
            raise ValueError("detector map changed between accuracy and timing")
    events.flags.writeable = False
    tensor, decoders = helper.build_software(context, include_relay=True)
    seed = point_seed(d, p)
    warm = context.circuit.compile_detector_sampler(seed=seed + 1).sample(32, bit_packed=True)
    warm.flags.writeable = False
    try:
        for decode in decoders.values():
            decode(warm)
        started = perf_counter()
        wall, wire, timed_defers = helper.timing(
            decoders, context, mapping, events, accepted, fpga, core, None,
            prediction, TIME_SHOTS, seed, 24 if d == 15 else TIME_SHOTS,
        )
    finally:
        close = getattr(tensor, "close", None)
        if close is not None:
            close()
    for row in records:
        internal = "fpga_only" if row["mode"] == FPGA_MODE else row["mode"]
        stats, count = wall[internal]
        row["single_shot_us_excluding_uart"] = stats
        row["single_shot_us_including_uart"] = wire.get(internal)
        row["timed_shots"] = count
        row["timed_fpga_defers"] = timed_defers if internal in ("fpga_only", "telescope") else None
        row["timing_complete"] = True
        row["timing_completed_utc"] = utc()
    helper.atomic_json(checkpoint, records)
    helper.write_tables(directory, load_rows(directory), settings)
    print(json.dumps({"phase": "timing", "point": point, "seconds": perf_counter() - started,
                      "median_us": {r["mode"]: r["single_shot_us_excluding_uart"]["median"]
                                    for r in records}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("accuracy", "timing"), required=True)
    parser.add_argument("--output", type=Path, default=BASE / "model")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    for child in (directory, directory / "checkpoints", directory / "cohorts"):
        child.mkdir(parents=True, exist_ok=True)
    profiles = {d: helper.get_profile(d) for d in (7, 15)}
    settings = settings_for(profiles)
    manifest = directory / "manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text(encoding="utf-8")) != settings:
            raise ValueError("resume requires unchanged settings and implementation")
    else:
        helper.atomic_json(manifest, settings)
    print(json.dumps({"phase": args.phase, "pid": os.getpid(), "points": len(POINTS),
                      "namespace": "pymatching_tensor_decoder", "started_utc": utc()}), flush=True)
    for d, p, n in POINTS:
        point = name_for(d, p)
        checkpoint = directory / "checkpoints" / f"{point}.json"
        if args.phase == "accuracy" and checkpoint.exists() and args.resume:
            continue
        if args.phase == "timing":
            if not checkpoint.exists():
                raise ValueError("all accuracy points must exist before timing")
            if all(row.get("timing_complete") for row in json.loads(checkpoint.read_text())):
                continue
        save_status(directory, args.phase, point=point)
        if args.phase == "accuracy":
            accuracy_point(directory, settings, profiles[d], d, p, n)
        else:
            timing_point(directory, settings, profiles[d], d, p)
    save_status(directory, f"{args.phase}_complete", completed_points=len(POINTS))
    print(json.dumps({"phase": f"{args.phase}_complete", "pid": os.getpid(), "completed_utc": utc()}), flush=True)


if __name__ == "__main__":
    main()

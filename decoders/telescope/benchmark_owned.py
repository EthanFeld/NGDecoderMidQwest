"""Test-only paired validation of the owned matcher and retained library references.

PyMatching is allowed here only as a comparison oracle. Production entry points
must not import this module. Fresh samples and retained board replay are labeled
separately, and existing output files are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

SURFACE = Path(__file__).resolve().parent
ROOT = SURFACE.parents[1]
sys.path.insert(0, str(SURFACE / "tensor_decoder/src"))
BOARD = ROOT / "past_experiments/comparisons/board/telescope-real-20261122"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprints():
    files = [
        *SURFACE.glob("*owned*.py"), SURFACE / "telescope.py", SURFACE / "telescope_circuit.py",
    ]
    files.extend((SURFACE / "native").glob("owned*"))
    files.extend((SURFACE / "tensor_decoder/src/tensor_decoder").rglob("*.py"))
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(set(files)) if path.is_file()
    }


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    return {
        "calls": len(values), "mean_us": float(values.mean()),
        "p50_us": float(np.percentile(values, 50)),
        "p99_us": float(np.percentile(values, 99)),
        "p999_us": float(np.percentile(values, 99.9)), "max_us": float(values.max()),
    }


def quality(prediction, truth, controls):
    wrong = prediction[:, 0] != truth
    errors, n = int(wrong.sum()), len(truth)
    rate, z = errors / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    delta = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    packed = np.packbits(prediction[:, 0], bitorder="little").tobytes()
    return {
        "shots": n, "logical_errors": errors, "ler_per_memory_experiment": rate,
        "wilson_ci95": [max(0., center - delta), min(1., center + delta)],
        "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
        "predictions_bit_packed_little_hex": packed.hex(),
        "paired": {
            name: {
                "fixes": int(((control[:, 0] != truth) & ~wrong).sum()),
                "regressions": int(((control[:, 0] == truth) & wrong).sum()),
                "disagreements": int(np.count_nonzero(prediction[:, 0] != control[:, 0])),
            }
            for name, control in controls.items()
        },
    }


def labels(value, count):
    value = np.asarray(value)
    if value.shape != (count, 1) or value.dtype != np.uint8 or not np.isin(value, [0, 1]).all():
        raise ValueError("a decoder returned invalid logical predictions")
    return np.ascontiguousarray(value)


def owned_stats(decoder):
    method = getattr(decoder, "owned_stats", None)
    return copy.deepcopy(method()) if callable(method) else {}


def numeric_delta(before, after):
    result = {}
    for key, value in after.items():
        old = before.get(key)
        if isinstance(value, dict) and isinstance(old, dict):
            result[key] = numeric_delta(old, value)
        elif isinstance(value, (int, float)) and isinstance(old, (int, float)):
            result[key] = value - old
    return result


def matching_free_probe(profile, noise):
    """Run production construction and decoding with matching imports forbidden."""
    import ctypes
    import faulthandler
    import importlib.abc

    faulthandler.dump_traceback_later(60, repeat=True)
    started = perf_counter_ns()

    class BlockMatching(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"pymatching", "networkx", "fusion_blossom"}:
                raise RuntimeError(f"production attempted an outside matching import: {fullname}")
            return None

    sys.meta_path.insert(0, BlockMatching())
    loaded = []
    original_cdll = ctypes.CDLL

    def checked_cdll(name, *args, **kwargs):
        path = Path(str(name)).resolve()
        stem = path.stem.lower()
        if stem == "matching" or stem.startswith("matching_"):
            raise RuntimeError(f"production attempted to load the vendored matcher: {path}")
        loaded.append(str(path))
        return original_cdll(name, *args, **kwargs)

    ctypes.CDLL = checked_cdll
    from telescope import FrontBatch, TelescopeDecoder
    from telescope_circuit import build_context

    image = json.loads(profile.read_bytes())
    context, mapping = build_context(image, noise)
    context_seconds = (perf_counter_ns() - started) / 1e9
    print(f"Dependency probe context ready in {context_seconds:.2f}s", file=sys.stderr, flush=True)
    events = np.zeros((2, (context.num_detectors + 7) // 8), dtype=np.uint8)
    events.flags.writeable = False
    telescope = TelescopeDecoder(context, image, mapping, front=object())
    decoder = telescope.backend
    constructed_seconds = (perf_counter_ns() - started) / 1e9
    print(
        f"Dependency probe default telescope ready in {constructed_seconds:.2f}s",
        file=sys.stderr, flush=True,
    )
    try:
        residual = np.zeros((2, len(mapping)), dtype=bool)
        reply = FrontBatch(
            np.zeros(2, bool), residual, np.zeros(2, np.uint8), np.ones(2, np.int32),
        )
        result = telescope.complete(events, reply)
        labels(result, 2)
        if "pymatching" in sys.modules:
            raise AssertionError("an outside matching library was loaded")
        print(json.dumps({
            "passed": True, "shots": 2, "stats": owned_stats(decoder),
            "loaded_native_libraries": loaded,
            "context_seconds": context_seconds,
            "context_and_constructor_seconds": constructed_seconds,
            "total_seconds": (perf_counter_ns() - started) / 1e9,
        }))
    finally:
        faulthandler.cancel_dump_traceback_later()
        close = getattr(decoder, "close", None)
        if callable(close):
            close()


def dependency_check(profile, noise):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe-dependencies",
         "--profile", str(profile), "--noise", str(noise)],
        check=False, stdout=subprocess.PIPE, text=True, timeout=300,
    )
    if result.returncode:
        raise RuntimeError(
            "The isolated production dependency check failed:\n"
            + result.stdout + (result.stderr or "")
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def board_inputs(args, image, context, mapping):
    from host import REPLY, Inbox, parse_reply, unpack
    from reference import Predecoder
    from surface_model import seal
    from telescope import FrontBatch

    capture = json.loads(args.board_capture.read_bytes())
    columns = {name: i for i, name in enumerate(capture["columns"])}
    run, = [run for run in capture["runs"] if run["p"] == args.noise]
    if not run["complete"] or run["fault"] or len(run["rows"]) != run["planned"]:
        raise ValueError("board replay requires a complete, successful capture")
    tag = seal(image)
    if run.get("image_tag") != tag:
        raise ValueError("the board capture and profile image tags differ")
    with np.load(args.board_joint, allow_pickle=False) as bundle:
        cohorts = np.flatnonzero(bundle["rates"] == args.noise)
        if len(cohorts) != 1 or not np.array_equal(bundle["memory_indices"], mapping):
            raise ValueError("the retained detector order or noise cohort does not match")
        index = int(cohorts[0])
        if int(bundle["seeds"][index]) != run["seed"]:
            raise ValueError("the board and joint-syndrome seeds differ")
        full_events = bundle[f"events_{index}"]
        full_truth = bundle[f"truth_{index}"].reshape(-1)
        if len(full_events) != len(run["rows"]):
            raise ValueError("joint syndromes and board rows have different lengths")
        count = min(args.shots, len(full_events))
        events = np.ascontiguousarray(full_events[:count], dtype=np.uint8)
        truth = np.ascontiguousarray(full_truth[:count], dtype=np.uint8)
    bits = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")
    memory = np.ascontiguousarray(bits[:, mapping], dtype=bool)
    front = FrontBatch(
        np.zeros(count, bool), np.zeros_like(memory), np.zeros(count, np.uint8),
        np.zeros(count, np.int32),
    )
    budget = image.get("growth_cycles", 0)
    max_cycles = Predecoder(image).plan.max_clocks + (budget + 1 if budget else 0)
    first_token = None
    for i, row in enumerate(run["rows"][:count]):
        if row[columns["truth"]] != int(truth[i]):
            raise ValueError(f"retained truth differs at board row {i}")
        wire = bytes.fromhex(row[columns["fpga_wire_hex"]])
        body, = Inbox().feed(wire)
        token, reply = parse_reply(body, tag, image["width"], image["rounds"] + 1, max_cycles)
        if first_token is None:
            first_token = token
        status = row[columns["fpga_status"]]
        residual_hex = None if reply.residual is None else reply.residual.hex()
        if (
            token != (first_token + i) % 65536
            or status not in (1, 3) or REPLY.unpack_from(body)[3] != status
            or reply.cycles != row[columns["core_cycles"]]
            or int(reply.logical) != row[columns["fpga_frame_bit"]]
            or residual_hex != row[columns["fpga_residual_hex"]]
            or (reply.residual is None) != (status == 1)
        ):
            raise ValueError(f"board reply metadata differs from its CRC-checked wire at row {i}")
        front.accepted[i] = status == 1
        front.logical[i] = reply.logical
        front.cycles[i] = reply.cycles
        if reply.residual is not None:
            front.residual[i] = unpack(reply.residual, image["width"], image["rounds"] + 1)
            if np.any(front.residual[i] & ~memory[i]):
                raise ValueError(f"board residual adds a detector at row {i}")
    events.flags.writeable = False
    return events, truth, memory, front, {
        "kind": "retained_physical_board_replay", "fresh_validation": False,
        "seed": int(run["seed"]), "cohort_shots": len(run["rows"]),
        "selection": "first requested shots, without truth-based selection",
        "capture_sha256": sha256(args.board_capture), "joint_sha256": sha256(args.board_joint),
        "crc_checked_replies": count,
    }


def fresh_inputs(args, image, context, mapping):
    from telescope import ReferenceFront

    growth = None
    if image.get("growth_cycles", 0):
        if args.growth_build is None:
            raise ValueError("a fresh growth-profile run requires --growth-build")
        from growth_reference import GrowthModel

        library = args.growth_build / "sim" / (
            "growth_model.dll" if sys.platform == "win32" else "growth_model.so"
        )
        growth = GrowthModel(args.growth_build / "growth_graph.txt", library)
    seed = secrets.randbits(31) if args.seed is None else args.seed
    events, observed = context.circuit.compile_detector_sampler(seed=seed).sample(
        args.shots, bit_packed=True, separate_observables=True,
    )
    events.flags.writeable = False
    bits = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")
    memory = np.ascontiguousarray(bits[:, mapping], dtype=bool)
    front = ReferenceFront(image, growth).decode(memory)
    return events, observed[:, 0], memory, front, {
        "kind": "fresh_joint_stim_samples", "fresh_validation": True, "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--noise", type=float, default=.005)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timing-shots", type=int, default=32)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--warmup-shots", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing-context", default="unspecified host load")
    parser.add_argument("--board", action="store_true")
    parser.add_argument(
        "--only-defers", action="store_true",
        help="keep FPGA accepts and evaluate/time backend completion on defers only",
    )
    parser.add_argument("--board-joint", type=Path, default=BOARD / "d15_joint.npz")
    parser.add_argument("--board-capture", type=Path, default=BOARD / "d15_board.json")
    parser.add_argument("--growth-build", type=Path)
    parser.add_argument("--probe-dependencies", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.shots, args.batch_size, args.timing_repeats) < 1:
        parser.error("shots, batch size and timing repeats must be positive")
    if min(args.timing_shots, args.warmup_shots) < 0:
        parser.error("timing and warmup counts must be nonnegative")
    if args.profile is None:
        args.profile = (
            ROOT / "past_experiments/surface_fpga/experiments/d15-weighted-growth/d15z-growth-cap4000.json"
            if args.board else SURFACE / "profiles/d15z-gap3-81mhz.json"
        )
    if args.probe_dependencies:
        matching_free_probe(args.profile, args.noise)
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if args.output is None:
        args.output = SURFACE / f"tensor_decoder/results/runs/owned_matching_{stamp}.json"
    if args.output.exists():
        parser.error("the output already exists; choose a new filename")
    sources_at_start = source_fingerprints()
    dependency = dependency_check(args.profile, args.noise)
    print("Default telescope dependency check passed", file=sys.stderr, flush=True)

    # These imports are test references; none belongs in the production decoder.
    import pymatching
    import stim
    from tensor_decoder.baseline import CorrelatedMatching

    from owned_backend import OwnedTensorDecoder
    from telescope_circuit import build_context
    from tensor_decoder import TensorDecoder

    image = json.loads(args.profile.read_bytes())
    before = perf_counter_ns()
    context, mapping = build_context(image, args.noise)
    context_us = (perf_counter_ns() - before) / 1000
    inputs = board_inputs if args.board else fresh_inputs
    events, truth, memory, front, source = inputs(args, image, context, mapping)
    del memory
    count = len(events)
    valid = ~front.accepted & (front.cycles > 0)
    factories = {
        "owned_no_hint": OwnedTensorDecoder, "owned_fpga_hint": OwnedTensorDecoder,
        "legacy_tensor": TensorDecoder, "legacy_correlated_matching": CorrelatedMatching,
    }
    decoders, initialization_us = {}, {}
    try:
        for name, factory in factories.items():
            before = perf_counter_ns()
            decoders[name] = factory(context)
            initialization_us[name] = (perf_counter_ns() - before) / 1000
            print(
                f"Initialized {name} in {initialization_us[name] / 1e6:.2f}s",
                file=sys.stderr, flush=True,
            )

        def execute(name, selection):
            selected_events = events[selection]
            residual = front.residual[selection]
            logical = front.logical[selection]
            available = valid[selection]
            count_selected = len(selected_events)
            deferred = None
            if args.only_defers:
                deferred = np.flatnonzero(~front.accepted[selection])
                completed = np.array(logical, dtype=np.uint8, copy=True).reshape(-1, 1)
                if not len(deferred):
                    return completed
                selected_events = np.ascontiguousarray(selected_events[deferred])
                selected_events.flags.writeable = False
                residual, logical, available = (
                    residual[deferred], logical[deferred], available[deferred],
                )
            if name == "owned_fpga_hint":
                returned = decoders[name].decode_with_front(
                    selected_events, residual, logical, available, mapping,
                )
            else:
                returned = decoders[name].decode_batch(selected_events)
            result = labels(returned, len(selected_events))
            if deferred is None:
                return result
            completed[deferred] = result
            return labels(completed, count_selected)

        stats = {"before_warmup": {k: owned_stats(v) for k, v in decoders.items()}}
        warmup_us = {}
        if args.only_defers:
            warmup_selection = np.flatnonzero(~front.accepted)[:args.warmup_shots]
            warmup_count = len(warmup_selection)
        else:
            warmup_count = min(args.warmup_shots, count)
            warmup_selection = slice(0, warmup_count)
        for name in decoders:
            before = perf_counter_ns()
            if warmup_count:
                execute(name, warmup_selection)
            warmup_us[name] = (perf_counter_ns() - before) / 1000
        stats["after_warmup"] = {k: owned_stats(v) for k, v in decoders.items()}
        predictions = {name: np.empty((count, 1), np.uint8) for name in decoders}
        elapsed_us = dict.fromkeys(decoders, 0.)
        names = list(decoders)
        for batch, start in enumerate(range(0, count, args.batch_size)):
            selection = slice(start, min(start + args.batch_size, count))
            for offset in range(len(names)):
                name = names[(batch + offset) % len(names)]
                before = perf_counter_ns()
                result = execute(name, selection)
                elapsed_us[name] += (perf_counter_ns() - before) / 1000
                predictions[name][selection] = result
        stats["after_accuracy_batches"] = {k: owned_stats(v) for k, v in decoders.items()}
        eligible = np.flatnonzero(~front.accepted) if args.only_defers else np.arange(count)
        positions = np.linspace(
            0, max(0, len(eligible) - 1), min(args.timing_shots, len(eligible)), dtype=np.intp,
        )
        indices = eligible[positions]
        timings = {name: [] for name in decoders}
        deferred_timings = {name: [] for name in decoders}
        for repeat in range(args.timing_repeats):
            for j, index in enumerate(indices):
                selection = slice(index, index + 1)
                for offset in range(len(names)):
                    name = names[(j + repeat + offset) % len(names)]
                    before = perf_counter_ns()
                    result = execute(name, selection)
                    duration = (perf_counter_ns() - before) / 1000
                    np.testing.assert_array_equal(result, predictions[name][selection])
                    timings[name].append(duration)
                    if valid[index]:
                        deferred_timings[name].append(duration)
        stats["after_single_shot_timing"] = {k: owned_stats(v) for k, v in decoders.items()}
        controls = {name: predictions[name] for name in (
            "legacy_tensor", "legacy_correlated_matching", "owned_no_hint",
        )}
        telescope_predictions = {
            name: np.where(
                front.accepted[:, None], front.logical[:, None], prediction,
            ).astype(np.uint8)
            for name, prediction in predictions.items()
        }
        sources_at_end = source_fingerprints()
        accuracy_records = {
            name: quality(value, truth, controls) for name, value in predictions.items()
        }
        report = {
            "format": "owned-matching-paired-validation-v1", "created_utc": stamp,
            "source": source, "distance": image["distance"], "rounds": image["rounds"],
            "basis": image["basis"], "physical_noise": args.noise, "shots": count,
            "evaluation_scope": "telescope_completion" if args.only_defers else "full_decoder",
            "backend_shots": int((~front.accepted).sum()) if args.only_defers else count,
            "profile": str(args.profile.resolve()), "profile_sha256": sha256(args.profile),
            "source_sha256": sources_at_start, "source_sha256_at_completion": sources_at_end,
            "sources_changed_during_run": sources_at_start != sources_at_end,
            "dependency_probe": dependency,
            "environment": {
                "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "stim": stim.__version__,
                "pymatching_test_reference_only": pymatching.__version__,
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            },
            "input_sha256": hashlib.sha256(events.tobytes()).hexdigest(),
            "truth_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
            "frontend": {
                "accepted": int(front.accepted.sum()), "deferred": int(valid.sum()),
                "unavailable": int((front.cycles <= 0).sum()),
                "accepted_errors": int(np.count_nonzero(
                    front.logical[front.accepted] != truth[front.accepted],
                )),
                "hint_rule": (
                    "valid reply on FPGA defers only; original full joint syndrome retained"
                ),
            },
            "full_decoder_accuracy": None if args.only_defers else accuracy_records,
            "telescope_completion_accuracy": accuracy_records if args.only_defers else None,
            "telescope_accuracy": {
                name: quality(
                    value, truth, {"legacy_tensor": telescope_predictions["legacy_tensor"]},
                )
                for name, value in telescope_predictions.items()
            },
            "timing_protocol": {
                "host_load_context": args.timing_context,
                "context_and_initialization_excluded": True, "warmup_excluded": True,
                "completed_host_calls": True, "decoder_order": "rotating per batch and shot",
                "batch_size": args.batch_size, "timing_unique_shots": len(indices),
                "timing_repeats": args.timing_repeats, "warmup_shots": warmup_count,
                "timing_indices": indices.tolist(),
                "timing_selection": "FPGA defers only" if args.only_defers else "full population",
                "tail_resolution_calls": len(indices) * args.timing_repeats,
            },
            "context_construction_us": context_us, "decoder_initialization_us": initialization_us,
            "warmup_us": warmup_us,
            "batch_timing": {
                name: {"completed_wall_us": value, "amortized_us_per_shot": value / count}
                for name, value in elapsed_us.items()
            },
            "single_shot_latency": {name: percentiles(v) for name, v in timings.items()},
            "deferred_single_shot_latency": {
                name: percentiles(v) for name, v in deferred_timings.items()
            },
            "single_shot_samples_us": timings, "owned_stats_by_phase": stats,
            "owned_accuracy_batch_stat_changes": {
                name: numeric_delta(
                    stats["after_warmup"][name], stats["after_accuracy_batches"][name],
                )
                for name in decoders
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
        print(json.dumps({
            "output": str(args.output.resolve()), "shots": count, "seed": source["seed"],
            "evaluation_scope": report["evaluation_scope"],
            "errors": {k: v["logical_errors"] for k, v in accuracy_records.items()},
            "single_shot_latency": report["single_shot_latency"],
        }, indent=2), flush=True)
    finally:
        for decoder in decoders.values():
            close = getattr(decoder, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    main()

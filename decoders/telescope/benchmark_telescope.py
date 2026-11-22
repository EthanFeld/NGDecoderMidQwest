"""Benchmark the d15 prior-tilt/tensor telescope on paired full syndromes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tensor_decoder/src"))

import numpy as np  # noqa: E402
import stim  # noqa: E402

from host import unpack  # noqa: E402
from owned_backend import _library  # noqa: E402
from telescope import POLICIES, TENSOR, FrontBatch, TelescopeDecoder  # noqa: E402
from telescope_circuit import build_context  # noqa: E402

PROFILE = ROOT / "profiles/d15z-gap3-81mhz.json"
DEFAULT_OUTPUT = ROOT / "tensor_decoder/results/runs/d15_telescope.json"


def implementation_identity():
    """Fingerprint the engine actually used, without building the reference matcher."""
    paths = [ROOT / "owned_backend.py", ROOT / "telescope.py",
             ROOT / "native/owned_matching.cpp", *sorted((ROOT / "native").glob("owned_*.h"))]
    return {
        "implementation": "owned-correlated-tensor-v1",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths},
        "matching_native_sha256": hashlib.sha256(Path(_library()._name).read_bytes()).hexdigest(),
    }


def percentile(values):
    if not len(values):
        return None
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "p999": float(np.percentile(values, 99.9)),
    }


def accuracy(prediction, truth, baseline):
    wrong = prediction[:, 0] != truth
    prior_wrong = baseline[:, 0] != truth
    n = len(truth)
    errors = int(wrong.sum())
    rate = errors / n
    z = 1.959963984540054
    den = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / den
    delta = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / den
    return {
        "errors": errors, "ler": rate, "ci95": [max(0.0, center - delta), min(1.0, center + delta)],
        "paired_fixes": int((prior_wrong & ~wrong).sum()),
        "paired_regressions": int((~prior_wrong & wrong).sum()),
        "prediction_differences": int((prediction != baseline).sum()),
    }


def timed(call, *args):
    started = perf_counter_ns()
    result = call(*args)
    return result, (perf_counter_ns() - started) / 1000


def board_data(bundle, run, index, telescope, image):
    events = bundle[f"events_{index}"]
    truth = bundle[f"truth_{index}"].reshape(-1)
    if not (
        run["complete"] and not run["fault"]
        and run["seed"] == int(bundle["seeds"][index])
        and run["p"] == float(bundle["rates"][index])
        and run["clock_hz"] == image.get("decoder_clock_mhz", 135) * 1_000_000
        and len(run["rows"]) == len(events) == len(truth)
        and np.array_equal(bundle["memory_indices"], telescope.memory_indices)
    ):
        raise ValueError("board capture and full detector fixture do not describe the same run")
    memory = telescope.memory(events)
    rows = run["rows"]
    expected = telescope.front.decode(memory)
    if not np.array_equal(truth, [r[0] for r in rows]):
        raise ValueError("board capture truth labels are out of sequence")
    accepted = np.array([r[3] is not None for r in rows])
    logical = np.array([r[16] for r in rows], dtype=np.uint8)
    clocks = np.array([r[4] for r in rows], dtype=np.int32)
    residual = np.zeros_like(memory)
    for i in np.flatnonzero(~accepted):
        residual[i] = unpack(bytes.fromhex(rows[i][15]), image["width"], image["rounds"] + 1)
    if not (
        np.array_equal(accepted, expected.accepted)
        and np.array_equal(clocks, expected.cycles)
        and np.array_equal(residual, expected.residual)
        and np.array_equal(logical, expected.logical)
    ):
        raise ValueError("board reply differs from the prior-tilt reference")
    link_us = np.asarray([r[14] if r[14] is not None else r[6] for r in rows])
    return events, truth, FrontBatch(accepted, residual, logical, clocks), link_us


def slice_front(front, index):
    return FrontBatch(front.accepted[index], front.residual[index], front.logical[index], front.cycles[index])


def quality(decoders, events, truth, front, memory, batch_size):
    outputs = {name: [] for name in decoders}
    elapsed = dict.fromkeys(decoders, 0.0)
    stage_counts = {name: np.zeros(3, dtype=np.int64) for name in decoders}
    names = list(decoders)
    for block, start in enumerate(range(0, len(events), batch_size)):
        selection = slice(start, start + batch_size)
        for shift in range(len(names)):
            name = names[(block + shift) % len(names)]
            (result, stages), us = timed(
                decoders[name], events[selection], slice_front(front, selection), memory[selection],
            )
            outputs[name].append(result)
            elapsed[name] += us
            if stages is not None:
                stage_counts[name] += np.bincount(stages, minlength=3)
    predictions = {k: np.concatenate(v) for k, v in outputs.items()}
    raw = predictions["tensor_cpu"]
    report = {
        name: {
            **accuracy(value, truth, raw),
            "amortized_us_per_shot": elapsed[name] / len(events),
            "vs_cpu_telescope_differences": int((value != predictions["telescope_cpu"]).sum()),
            "stage_counts": dict(zip(
                ("prior_tilt", "residual_check", "tensor"), stage_counts[name].tolist(), strict=True,
            )) if name.startswith("telescope") else None,
        }
        for name, value in predictions.items()
    }
    return report, predictions


def replay_accuracy(telescopes, cpu, events, truth, front, memory, batch_size):
    raw = np.concatenate([
        cpu.backend.decode_batch(events[start : start + batch_size])
        for start in range(0, len(events), batch_size)
    ])
    report = {"tensor_cpu": accuracy(raw, truth, raw)}
    plain = None
    for name, telescope in telescopes.items():
        prediction, stages = telescope.route_front(events, front, memory)
        prediction[stages == TENSOR] = raw[stages == TENSOR]
        if plain is None:
            plain = prediction
        report[name] = {
            **accuracy(prediction, truth, raw),
            "vs_plain": accuracy(prediction, truth, plain),
            "stage_counts": dict(zip(
                ("prior_tilt", "residual_check", "tensor"),
                np.bincount(stages, minlength=3).tolist(), strict=True,
            )),
        }
    return report


def single_shot(decoders, projector, events, front, predictions, core_us, uart_us, count):
    names = list(decoders)
    selection = np.linspace(0, len(events) - 1, min(len(events), count), dtype=np.intp)
    observations = {name: np.empty(len(selection)) for name in names}
    projection = np.empty(len(selection))
    for j, i in enumerate(selection):
        segment = slice(i, i + 1)
        syndromes, result = events[segment], slice_front(front, segment)
        projected, projection[j] = timed(projector, syndromes)
        for shift in range(len(names)):
            name = names[(j + shift) % len(names)]
            (label, _), elapsed = timed(decoders[name], syndromes, result, projected)
            if not np.array_equal(label, predictions[name][segment]):
                raise ValueError(f"{name} changed its prediction between batch and single-shot")
            observations[name][j] = elapsed
    stats = {"host_projection_us": percentile(projection), "timed_shots": len(selection)}
    for name, values in observations.items():
        entry = {"host_completion_us": percentile(values)}
        if name.startswith("telescope"):
            point = values + projection
            entry["hybrid_without_uart_us"] = percentile(point + core_us[selection])
            entry["deferred_host_us"] = percentile(values[~front.accepted[selection]])
            if uart_us is not None:
                entry["hybrid_with_uart_us"] = percentile(point + uart_us[selection])
        stats[name] = entry
    return stats


def benchmark(image, args):
    capture = json.loads(args.board_capture.read_bytes())["runs"] if args.board_capture else None
    bundle = np.load(args.board_events, allow_pickle=False) if args.board_events else None
    rates = args.rates or [0.003, 0.005, 0.0075]
    selected = list(enumerate(rates))
    if capture is not None:
        selected = [(i, run["p"]) for i, run in enumerate(capture) if not args.rates or run["p"] in rates]
        if not selected or (args.rates and set(rates) != {p for _, p in selected}):
            raise ValueError("requested rates must be present in the board capture")
    cohorts = []
    for i, rate in selected:
        context, mapping = build_context(image, rate)
        cpu = TelescopeDecoder(context, image, mapping, policy="plain")
        cuda = TelescopeDecoder(context, image, mapping, device=args.cuda_mode, policy="plain") if args.cuda else None
        seed = capture[i]["seed"] if capture else args.seed + i
        if capture:
            events, truth, front, uart_us = board_data(bundle, capture[i], i, cpu, image)
        else:
            events, truth = context.circuit.compile_detector_sampler(seed=seed).sample(
                args.shots, separate_observables=True, bit_packed=True,
            )
            truth = truth[:, 0]
            front = cpu.front.decode(cpu.memory(events))
            uart_us = None
        events.flags.writeable = False
        memory = cpu.memory(events)
        decoders = {
            "tensor_cpu": lambda events, _front, _memory, backend=cpu.backend: (backend.decode_batch(events), None),
            "telescope_cpu": cpu.complete_with_stages,
        }
        telescopes = {"telescope_cpu": cpu}
        if cuda:
            decoders["telescope_cuda"] = cuda.complete_with_stages
        for policy in args.policies:
            if policy != "plain":
                accelerated = cuda or cpu
                name = f"telescope_{policy}_{'cuda' if cuda else 'cpu'}"
                telescope = TelescopeDecoder(
                    context, image, mapping, front=cpu.front, backend=accelerated.backend, policy=policy,
                )
                decoders[name] = telescope.complete_with_stages
                telescopes[name] = telescope
        core_us = front.cycles / image.get("decoder_clock_mhz", 135)
        timing = None
        if args.accuracy_only:
            quality_report = replay_accuracy(telescopes, cpu, events, truth, front, memory, args.batch_size)
        else:
            for decode in decoders.values():
                decode(events[:32], slice_front(front, slice(32)), memory[:32])
            quality_report, predictions = quality(decoders, events, truth, front, memory, args.batch_size)
            if cuda and np.any(predictions["telescope_cuda"] != predictions["telescope_cpu"]):
                raise ValueError("CUDA changed the telescope decisions")
            if args.time_shots:
                timing = single_shot(decoders, cpu.memory, events, front, predictions, core_us, uart_us, args.time_shots)
        raw_weight = memory.sum(axis=1)
        result = {
            "p": rate, "seed": seed, "shots": len(truth), "fpga_accepts": int(front.accepted.sum()),
            "fpga_accept_errors": int((front.accepted & (front.logical != truth)).sum()),
            "fpga_defers": int((~front.accepted).sum()),
            "selected_detectors": len(mapping), "joint_detectors": context.num_detectors,
            "deferred_raw_weight": percentile(raw_weight[~front.accepted]),
            "deferred_residual_weight": percentile(front.residual[~front.accepted].sum(axis=1)),
            "fpga_core_us": percentile(core_us), "fpga_uart_us": percentile(uart_us) if capture else None,
            "quality": quality_report, "timing": timing,
        }
        cohorts.append(result)
        print(json.dumps(result), flush=True)
    return cohorts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--rates", type=float, nargs="+")
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--time-shots", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=51917031)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda-mode", choices=("cuda", "cuda-batch"), default="cuda",
                        help="GPU dispatch mode used with --cuda")
    parser.add_argument("--accuracy-only", action="store_true",
                        help="score routing policies against a shared CPU tensor decode and skip timing")
    parser.add_argument("--policies", choices=POLICIES, nargs="+", default=POLICIES)
    parser.add_argument("--board-capture", type=Path)
    parser.add_argument("--board-events", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.shots < 1 or args.batch_size < 1 or args.time_shots < 0:
        parser.error("shots and batch size must be positive and timed shots must be nonnegative")
    if bool(args.board_capture) != bool(args.board_events):
        parser.error("provide both the board capture and the full detector event fixture")
    if args.cuda and args.accuracy_only:
        parser.error("choose separate CPU accuracy replay and CUDA timing runs")
    if args.output.exists():
        parser.error("choose a new output filename to preserve earlier evidence")
    raw = args.profile.read_bytes()
    report = {
        "frontend_sha256": hashlib.sha256(raw).hexdigest(),
        "telescope_sha256": hashlib.sha256((ROOT / "telescope.py").read_bytes()).hexdigest(),
        "tensor": implementation_identity(),
        "stim": stim.__version__, "numpy": np.__version__, "host": platform.platform(),
        "stage1_source": "COM capture" if args.board_capture else "cycle model",
        "batch_size": args.batch_size, "cuda_mode": args.cuda_mode if args.cuda else None,
        "accuracy_only": args.accuracy_only,
        "cohorts": benchmark(json.loads(raw), args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Verify and time completed CUDA correlated matching on d15 syndromes."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

SURFACE = Path(__file__).resolve().parents[2]
ROOT = SURFACE.parent
CAPTURE = ROOT / "fpga_benchmark/.build/surface_d15_tilt_ram_growth_27_fresh_p005_72621005"
sys.path[:0] = [str(SURFACE), str(SURFACE.parents[1] / "decoders/telescope"), str(SURFACE.parents[1] / "decoders/telescope/tensor_decoder/src"), str(SURFACE.parents[1] / "decoders/pymatching_telescope/src")]

import numpy as np
import pymatching
import stim

from tensor_decoder import DecoderContext, Experiment, qecbench_implementation_identity




def board_inputs(args):
    from compare_telescope import checked_capture
    from pymatching_tensor_decoder.telescope.telescope_circuit import build_context

    image = json.loads(args.profile.read_bytes())
    archive = json.loads(args.capture.read_bytes())
    columns = {name: i for i, name in enumerate(archive["columns"])}
    capture = next(
        (run for run in archive["runs"] if run["mode"] == "surface" and run["p"] == args.noise),
        None,
    )
    if capture is None or not capture["complete"] or capture["fault"]:
        raise ValueError("the requested board cohort must be complete")
    context, indices = build_context(image, args.noise)
    shots = len(capture["rows"])
    packed, labels = context.circuit.compile_detector_sampler(seed=capture["seed"]).sample(
        shots, bit_packed=True, separate_observables=True,
    )
    bits = np.unpackbits(packed, axis=1, count=context.num_detectors, bitorder="little")
    front = checked_capture(capture, columns, args.fixtures, image, bits[:, indices], labels[:, 0])
    unresolved = np.flatnonzero(~front.accepted)
    if not len(unresolved):
        raise ValueError("the board cohort has no deferred shots")
    return context, bits[unresolved], unresolved, capture


def retained_inputs():
    records = ROOT.parent / "decoders/telescope/tensor_decoder/results/validation"
    packed = np.asarray([
        case["events"] for lane in "ab"
        for case in json.loads((records / f"validation_{lane}.json").read_bytes())["cases"]
    ], dtype=np.uint8)
    experiment = Experiment(15, 15, 0.005)
    circuit = experiment.circuit()
    context = DecoderContext(experiment, circuit, circuit.detector_error_model(decompose_errors=True))
    return context, np.unpackbits(packed, axis=1, count=context.num_detectors, bitorder="little")


def exact_edges(matcher, rows):
    return [matcher.decode_to_edges_array(row, enable_correlations=True) for row in rows]


def canonical(edges):
    return sorted((min(a, b), max(a, b)) for a, b in edges)


def compare_edges(reference, candidate, indexes):
    counts = {"shots": 0, "ordered_exact": 0, "canonical_exact": 0, "gpu_unresolved": 0}
    first_nonexact = []
    for expected, actual, index in zip(reference, candidate, indexes, strict=True):
        counts["shots"] += 1
        if actual is None:
            counts["gpu_unresolved"] += 1
        elif np.array_equal(expected, actual):
            counts["ordered_exact"] += 1
            counts["canonical_exact"] += 1
            continue
        elif canonical(expected) == canonical(actual):
            counts["canonical_exact"] += 1
        if len(first_nonexact) < 8:
            first_nonexact.append(int(index))
    counts["different_edge_sets"] = (
        counts["shots"] - counts["canonical_exact"] - counts["gpu_unresolved"]
    )
    counts["first_nonexact_shot_ids"] = first_nonexact
    return counts


def timing_summary(samples, shots, repeats):
    p50, p99, p999 = np.percentile(samples, (50, 99, 99.9))
    return {
        "mean_active_us_per_shot": sum(samples) / (shots * repeats),
        "batch_p50_us": float(p50),
        "batch_p99_us": float(p99),
        "batch_p999_us": float(p999),
        "measured_batches": len(samples),
    }


def benchmark_board(cpu_edges, gpu_edges, bits, board_indexes, args):
    result = {}
    for size in args.batch_sizes:
        calls = (cpu_edges, gpu_edges)
        durations = ([], [])
        cpu_result, gpu_result = [], []
        for repeat in range(args.repeats):
            for index, start in enumerate(range(0, len(bits), size)):
                rows = bits[start : start + size]
                decoded = [None, None]
                for position in range(2):
                    mode = (position + index + repeat) % 2
                    before = perf_counter_ns()
                    decoded[mode] = calls[mode](rows)
                    durations[mode].append((perf_counter_ns() - before) / 1000)
                if repeat == 0:
                    cpu_result.extend(decoded[0])
                    gpu_result.extend(decoded[1])
        result[size] = {
            "cpu": timing_summary(durations[0], len(bits), args.repeats),
            "cuda": timing_summary(durations[1], len(bits), args.repeats),
            "edge_parity": compare_edges(cpu_result, gpu_result, board_indexes),
        }
    return result


def graph_summary(context, graph):
    return {
        "dem_sha256": hashlib.sha256(str(context.dem).encode()).hexdigest(),
        "detectors": context.num_detectors,
        "edges": len(graph.weights),
        "rules": len(graph.rule_targets),
    }


def benchmark(args):
    from tensor_decoder.correlated_graph import correlated_graph

    from tensor_decoder import correlated_cuda

    context, board, board_indexes, capture = board_inputs(args)
    matcher = pymatching.Matching.from_detector_error_model(context.dem, enable_correlations=True)
    graph = correlated_graph(context.dem, matcher)
    gpu = correlated_cuda.CUDACorrelated(graph, max_batch=args.max_device_batch)
    report = {
        "host": platform.platform(),
        "pymatching": pymatching.__version__,
        "stim": stim.__version__,
        "numpy": np.__version__,
        "implementation": qecbench_implementation_identity(),
        "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "capture_sha256": hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        "noise": args.noise,
        "board_seed": capture["seed"],
        "board_shots": len(capture["rows"]),
        "board_defers": len(board),
        "board_graph": graph_summary(context, graph),
        "repeats": args.repeats,
        "max_device_batch": args.max_device_batch,
        "timing_scope": "completed edge extraction including GPU transfers and synchronization",
    }
    native = Path(correlated_cuda.__file__).parent / "native"
    report["cuda_sources"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(native.iterdir()) if path.suffix in (".cu", ".cuh")
    }
    origin = native / "vendor/pymatching_2_4_0/ORIGIN.json"
    report["sparse_blossom_tree"] = json.loads(origin.read_bytes())["tree_sha"]

    def cpu_edges(rows):
        return exact_edges(matcher, rows)

    try:
        cpu_edges(board)
        gpu.edges(board)
        report["timing"] = benchmark_board(cpu_edges, gpu.edges, board, board_indexes, args)
        if args.fresh_shots:
            sampler = context.circuit.compile_detector_sampler(seed=args.fresh_seed)
            fresh = sampler.sample(args.fresh_shots).astype(np.uint8, copy=False)
            report["fresh"] = {
                "seed": args.fresh_seed,
                "edge_parity": compare_edges(cpu_edges(fresh), gpu.edges(fresh), range(len(fresh))),
            }
    finally:
        gpu.close()
    if args.hard:
        hard_context, hard = retained_inputs()
        hard_matcher = pymatching.Matching.from_detector_error_model(
            hard_context.dem, enable_correlations=True,
        )
        hard_graph = correlated_graph(hard_context.dem, hard_matcher)
        hard_gpu = correlated_cuda.CUDACorrelated(hard_graph, max_batch=args.max_device_batch)
        try:
            report["retained_hard"] = {
                "graph": graph_summary(hard_context, hard_graph),
                "edge_parity": compare_edges(
                    exact_edges(hard_matcher, hard), hard_gpu.edges(hard), range(len(hard)),
                ),
            }
        finally:
            hard_gpu.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path(__file__).with_name("d15z-growth-cap4000.json"))
    parser.add_argument("--capture", type=Path, default=Path(f"{CAPTURE}_board.json"))
    parser.add_argument("--fixtures", type=Path, default=Path(f"{CAPTURE}_fixtures.json"))
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64, 128, 221])
    parser.add_argument("--max-device-batch", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hard", action="store_true", help="compare all retained hard shots")
    parser.add_argument("--fresh-shots", type=int, default=0)
    parser.add_argument("--fresh-seed", type=int, default=72691005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.batch_sizes, args.repeats, args.max_device_batch) < 1 or args.fresh_shots < 0:
        parser.error("use positive batch sizes and repeat counts")
    if args.max_device_batch > 256:
        parser.error("--max-device-batch must be in [1, 256]")
    if args.output.exists():
        parser.error("choose a fresh output path")
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for size, batch in report["timing"].items():
        timings = {
            name: round(batch[name]["mean_active_us_per_shot"], 1) for name in ("cpu", "cuda")
        }
        print(size, timings, batch["edge_parity"])


if __name__ == "__main__":
    main()

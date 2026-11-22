"""Measure active tensor decode time on deferred syndromes from FPGA captures."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import stim

SURFACE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SURFACE), str(SURFACE.parents[1] / "decoders/telescope"), str(SURFACE.parents[1] / "decoders/telescope/tensor_decoder/src"), str(SURFACE.parents[1] / "decoders/pymatching_telescope/src")]


def measure(decoders, events, expected, batch_sizes, repeats):
    results = {}
    names = list(decoders)
    for size in batch_sizes:
        measurements = {name: [] for name in names}
        for repeat in range(repeats):
            for index, start in enumerate(range(0, len(events), size)):
                selection = slice(start, start + size)
                for j in range(len(names)):
                    name = names[(index + repeat + j) % len(names)]
                    t0 = perf_counter_ns()
                    labels = decoders[name].decode_batch(events[selection])
                    elapsed_us = (perf_counter_ns() - t0) / 1000
                    np.testing.assert_array_equal(labels, expected[selection])
                    measurements[name].append(elapsed_us)
        results[size] = {
            name: {
                "mean_active_us_per_defer": sum(samples) / (len(events) * repeats),
                "batch_p50_us": float(np.percentile(samples, 50)),
                "batch_p99_us": float(np.percentile(samples, 99)),
                "batch_p999_us": float(np.percentile(samples, 99.9)),
                "measured_batches": len(samples),
            }
            for name, samples in measurements.items()
        }
    return results


def benchmark(args):
    from compare_telescope import checked_capture
    from pymatching_tensor_decoder.telescope.telescope_circuit import build_context
    from tensor_decoder._build import routing_library

    from tensor_decoder import TensorDecoder, qecbench_implementation_identity

    raw_profile = args.profile.read_bytes()
    image = json.loads(raw_profile)
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
    events, labels = context.circuit.compile_detector_sampler(seed=capture["seed"]).sample(
        shots, bit_packed=True, separate_observables=True,
    )
    truth = labels[:, 0]
    memory = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")[:, indices]
    front = checked_capture(capture, columns, args.fixtures, image, memory, truth)
    defer = ~front.accepted
    events = events[defer]
    events.flags.writeable = False
    if not len(events):
        raise ValueError("the board cohort has no deferred shots")

    configs = {
        "cpu": {},
        "cpu_threads": {"solve_workers": args.workers},
    }
    if args.cuda:
        configs.update(
            cuda_batch={"device": "cuda-batch"},
            cuda_threads={"device": "cuda-batch", "solve_workers": args.workers},
        )
    if args.bucket:
        configs.update(
            cpu_bucket={"path_bucket": True},
            cpu_bucket_threads={"path_bucket": True, "solve_workers": args.workers},
        )
        if args.cuda:
            configs["cuda_bucket_threads"] = {
                "device": "cuda-batch", "path_bucket": True, "solve_workers": args.workers,
            }
    if args.cuda_correlated:
        other_options = {
            "device": "cuda-batch" if args.cuda else "cpu",
            "path_bucket": args.bucket,
            "solve_workers": args.workers,
        }
        configs = {
            "cpu": {}, "reference_tail": other_options,
            "cuda_correlated_tail": {**other_options, "cuda_correlated": True},
        }
    decoders = {}
    try:
        for name, options in configs.items():
            decoders[name] = TensorDecoder(context, **options)
        expected, diagnostic = decoders["cpu"].decode_with_diagnostics(events)
        for decoder in decoders.values():
            np.testing.assert_array_equal(decoder.decode_batch(events), expected)
        timing = measure(decoders, events, expected, args.batch_sizes, args.repeats)
    finally:
        for decoder in decoders.values():
            decoder.close()
    identity = qecbench_implementation_identity()
    if args.bucket:
        identity["routing_native_sha256"] = hashlib.sha256(routing_library().read_bytes()).hexdigest()
    if args.cuda_correlated:
        from tensor_decoder._build import NATIVE

        source = b"".join(
            (NATIVE / name).read_bytes()
            for name in ("cuda_mwpm_containers.cuh", "correlated_matching.cu")
        )
        identity["cuda_correlated_sha256"] = hashlib.sha256(source).hexdigest()
    return {
        "p": args.noise,
        "seed": capture["seed"],
        "total_shots": shots,
        "defers": int(defer.sum()),
        "conditioned_solves": int(diagnostic["routed"].sum()),
        "logical_sector_searches": int(diagnostic["competition_checked"].sum()),
        "fpga_accept_errors": int((front.accepted & (front.logical != truth)).sum()),
        "tensor_defer_errors": int((expected[:, 0] != truth[defer]).sum()),
        "workers": args.workers,
        "repeats": args.repeats,
        "profile_sha256": hashlib.sha256(raw_profile).hexdigest(),
        "tensor": identity,
        "host": platform.platform(),
        "stim": stim.__version__,
        "numpy": np.__version__,
        "timing": timing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path(__file__).with_name("d15z-growth-cap4000.json"))
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--noise", type=float, default=.005)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--bucket", action="store_true")
    parser.add_argument("--cuda-correlated", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.batch_sizes, args.workers, args.repeats) < 1 or args.output.exists():
        parser.error("use positive counts and a fresh output path")
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for size, modes in report["timing"].items():
        print(size, {k: round(v["mean_active_us_per_defer"], 1) for k, v in modes.items()})


if __name__ == "__main__":
    main()

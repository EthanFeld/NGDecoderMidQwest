"""Controlled native iteration/ensemble ablations with raw convergence evidence.

Calibration is separate from validation. No observed error or unconverged shot is
discarded; the native decoder never receives logical truth. Latency admission is
reported independently of accuracy-focused configurations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qecbench import DecoderContext, Experiment
from qecbench.decoders import load_factory, validate_predictions
from qecbench.metrics import count_failures, wilson_interval
from qecbench.storage import environment_identity, implementation_identity, write_json
from surface_decoders.ibm_relay_bp import (
    PAPER_S1_CONFIG,
    PAPER_S5_CONFIG,
    TUNED_CONFIG,
    IBMRelayBPDecoder,
)


def iteration_cases():
    cases = {"latency_profile": TUNED_CONFIG}
    for first in (30, 160):
        cases[f"first_{first}"] = replace(TUNED_CONFIG, pre_iter=first)
    for per_leg in (30, 120, 240):
        cases[f"leg_iter_{per_leg}"] = replace(TUNED_CONFIG, set_max_iter=per_leg)
    for sets in (7, 100, 300, 600):
        cases[f"legs_{sets + 1}"] = replace(TUNED_CONFIG, num_sets=sets)
    for solutions in (1, 5, 10):
        cases[f"legs_601_s{solutions}"] = replace(TUNED_CONFIG, num_sets=600, stop_nconv=solutions)
    cases["all_32_legs"] = replace(TUNED_CONFIG, stopping_criterion="all")
    return cases


def paper_cases():
    return {
        "latency_profile": TUNED_CONFIG,
        "xz_paper_s1": replace(PAPER_S1_CONFIG, detector_mode="memory_basis"),
        "xz_paper_s5": replace(PAPER_S5_CONFIG, detector_mode="memory_basis"),
        "xz_paper_s5_alpha09": replace(PAPER_S5_CONFIG, detector_mode="memory_basis", alpha=0.9),
        "xyz_paper_s1": PAPER_S1_CONFIG,
        "xyz_paper_s5": PAPER_S5_CONFIG,
        "xyz_paper_s5_alpha09": replace(PAPER_S5_CONFIG, alpha=0.9),
    }


def data(distance, noise, shots, seed):
    experiment = Experiment(distance, distance, noise)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    events, truth = circuit.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True, bit_packed=True
    )
    events.flags.writeable = False
    return context, events, truth


def latency(decoders, events, probes, warmup=32):
    """Actual packed one-shot calls, serialized with rotating decoder order."""
    names = list(decoders)
    for i in range(warmup):
        for decoder in decoders.values():
            decoder.decode_batch(events[i : i + 1])
    times = {name: np.empty(probes) for name in names}
    for i in range(probes):
        order = names[i % len(names) :] + names[: i % len(names)]
        shot = events[warmup + i : warmup + i + 1]
        for name in order:
            start = time.perf_counter_ns()
            decoders[name].decode_batch(shot)
            times[name][i] = (time.perf_counter_ns() - start) / 1000
    return times


def evaluate(config, context, events, truth):
    decoder = IBMRelayBPDecoder(context, config)
    syndrome = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")
    syndrome = np.ascontiguousarray(syndrome[:, decoder.detector_indices] ^ decoder.detector_offset)
    syndrome.flags.writeable = False
    start = time.perf_counter()
    detailed = decoder.runner.decode_observables_detailed_batch(
        syndrome, parallel=False, progress_bar=False
    )
    detailed_seconds = time.perf_counter() - start
    predictions = np.packbits(
        np.asarray([r.observables for r in detailed], dtype=np.uint8) ^ decoder.observable_offset,
        axis=1,
        bitorder="little",
    )
    iterations = np.asarray([r.iterations for r in detailed], dtype=np.int64)
    converged = np.asarray([r.converged for r in detailed], dtype=bool)
    errors = np.any(predictions != truth, axis=1)
    assert np.all(iterations <= config.max_iterations)
    # Check the detailed instrumentation agrees with the contract path on fixed shots.
    indices = np.unique(
        np.r_[
            np.arange(min(8, len(events))),
            np.flatnonzero(errors)[:8],
            np.flatnonzero(~converged)[:8],
        ]
    )
    if len(indices):
        normal = decoder.decode_batch(np.ascontiguousarray(events[indices]))
        np.testing.assert_array_equal(normal, predictions[indices])
    low, high = wilson_interval(int(errors.sum()), len(events))
    row = {
        "config": asdict(config),
        "total_legs": config.total_legs,
        "maximum_iterations": config.max_iterations,
        "distance": context.experiment.distance,
        "noise": context.experiment.noise,
        "shots": len(events),
        "errors": int(errors.sum()),
        "logical_error_rate": float(errors.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "nonconverged": int((~converged).sum()),
        "errors_nonconverged": int(np.count_nonzero(errors & ~converged)),
        "errors_converged": int(np.count_nonzero(errors & converged)),
        "iterations_mean": float(iterations.mean()),
        "iterations_median": float(np.median(iterations)),
        "iterations_p95": float(np.percentile(iterations, 95)),
        "iterations_p99": float(np.percentile(iterations, 99)),
        "iterations_max": int(iterations.max()),
        "shots_at_maximum_iterations": int(np.count_nonzero(iterations == config.max_iterations)),
        "active_checks": decoder.check_matrix.shape[0],
        "active_variables": decoder.check_matrix.shape[1],
        "detailed_instrumentation_seconds": detailed_seconds,
    }
    return row, {
        "predictions": predictions,
        "iterations": iterations,
        "converged": converged,
        "logical_failures": errors,
    }


def add_timing(row, samples):
    median, p95, p99 = np.percentile(samples, [50, 95, 99])
    row.update(
        latency_samples=len(samples),
        median_us=float(median),
        p95_us=float(p95),
        p99_us=float(p99),
        latency_pass=bool(median <= 1000),
        latency_budget_us=1000,
        latency_budget_metric="median",
    )


def run_calibration(output, stage, cases, points, probes):
    rows = []
    for d, p, shots, seed in points:
        directory = output / stage / f"d{d}_p{p:g}"
        directory.mkdir(parents=True, exist_ok=True)
        context, events, truth = data(d, p, shots, seed)
        _, probe_events, _ = data(d, p, probes + 32, seed + 10000)
        reference = load_factory("pymatching")(context)
        pm = reference.decode_batch(events)
        validate_predictions(pm, shots, context.num_observables)
        np.savez_compressed(
            directory / "samples.npz",
            events=events,
            truth=truth,
            pymatching=pm,
            probe_events=probe_events,
        )
        print(stage, "POINT", d, p, "PM", count_failures(pm, truth), "/", shots, flush=True)
        for name, config in cases.items():
            saved = directory / f"{name}.json"
            if saved.exists():
                row = json.loads(saved.read_text())
            else:
                decoder = IBMRelayBPDecoder(context, config)
                times = latency({"relay": decoder, "pymatching": reference}, probe_events, probes)
                row, raw = evaluate(config, context, events, truth)
                row.update(
                    profile=name,
                    stage=stage,
                    seed=seed,
                    pymatching_errors=count_failures(pm, truth),
                )
                add_timing(row, times["relay"])
                np.savez_compressed(
                    directory / f"{name}.npz",
                    **raw,
                    wall_us=times["relay"],
                    pymatching_wall_us=times["pymatching"],
                )
                write_json(saved, row)
            rows.append(row)
            write_json(output / f"{stage}_results.json", rows)
            print(
                stage,
                name,
                "d",
                d,
                "errors",
                row["errors"],
                "nonconv",
                row["nonconverged"],
                "actual_iter_med",
                row["iterations_median"],
                "max_iter",
                row["maximum_iterations"],
                "median_us",
                round(row["median_us"], 1),
                flush=True,
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("past_experiments/decoder_testing/results/relay-iteration-audit-20261121"),
    )
    parser.add_argument("--stage", choices=["iterations", "paper"], default="iterations")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = iteration_cases() if args.stage == "iterations" else paper_cases()
    points = (
        [(7, 0.007, 1024, 1213001), (9, 0.003, 512, 1213002)]
        if args.stage == "iterations"
        else [(5, 0.007, 1024, 1214001), (7, 0.003, 256, 1214002)]
    )
    identity = {
        "stage": args.stage,
        "cases": {name: asdict(c) for name, c in cases.items()},
        "points": points,
        "probes": 129,
        "warmup": 32,
        "source": implementation_identity(("surface_decoders.ibm_relay_bp:build_tuned_decoder",)),
        "environment": environment_identity(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "procedure": "accuracy calibration; 1 ms median status is independent",
        "instrumentation": (
            "native iterations/success from detailed output; latency uses normal packed path"
        ),
    }
    identity = json.loads(json.dumps(identity))
    manifest = output / f"{args.stage}_manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError(
                "resume requires matching settings, native code, source and environment"
            )
    else:
        write_json(manifest, {"created_utc": datetime.now(UTC).isoformat(), "identity": identity})
    rows = run_calibration(output, args.stage, cases, points, 129)
    flat = [{key: value for key, value in row.items() if key != "config"} for row in rows]
    with (output / f"{args.stage}_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


if __name__ == "__main__":
    main()

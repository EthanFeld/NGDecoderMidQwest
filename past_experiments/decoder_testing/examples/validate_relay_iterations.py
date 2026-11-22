"""Fresh paired validation of frozen iteration and accuracy profiles.

Instrumentation records actual iterations and convergence in bounded chunks.
Budget timing uses separate serialized calls through the packed contract path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from audit_relay_iterations import add_timing, data, latency

from qecbench.decoders import load_factory, validate_predictions
from qecbench.metrics import count_failures, wilson_interval
from qecbench.storage import environment_identity, implementation_identity, write_json
from surface_decoders.ibm_relay_bp import (
    ACCURACY_CONFIG,
    ITERATION_CANDIDATE_CONFIG,
    LONG_SEARCH_CONFIG,
    TUNED_CONFIG,
    IBMRelayBPDecoder,
)

PROFILES = {
    "latency_profile": TUNED_CONFIG,
    "iteration_candidate": ITERATION_CANDIDATE_CONFIG,
    "long_memory_search": LONG_SEARCH_CONFIG,
    "xyz_accuracy": ACCURACY_CONFIG,
}
POINTS = [
    (5, 0.007, 8192, 1229001, 1024),
    (7, 0.003, 8192, 1229002, 1024),
    (9, 0.003, 4096, 1229003, 1024),
    (11, 0.003, 512, 1229004, 512),
]


def score(decoder, context, events, truth, directory, name, config):
    count = len(events)
    partial = directory / f"{name}_partial.npz"
    if partial.exists():
        with np.load(partial) as saved:
            predictions = saved["predictions"]
            iterations = saved["iterations"]
            converged = saved["converged"]
            completed = int(saved["completed"])
            instrument_seconds = float(saved["instrument_seconds"])
    else:
        predictions = np.zeros_like(truth)
        iterations = np.zeros(count, dtype=np.int64)
        converged = np.zeros(count, dtype=bool)
        completed, instrument_seconds = 0, 0.0
    for start in range(completed, count, 128):
        stop = min(start + 128, count)
        syndrome = np.unpackbits(
            events[start:stop], axis=1, count=context.num_detectors, bitorder="little"
        )
        syndrome = np.ascontiguousarray(
            syndrome[:, decoder.detector_indices] ^ decoder.detector_offset
        )
        syndrome.flags.writeable = False
        clock = time.perf_counter()
        detailed = decoder.runner.decode_observables_detailed_batch(
            syndrome, parallel=False, progress_bar=False
        )
        instrument_seconds += time.perf_counter() - clock
        predictions[start:stop] = np.packbits(
            np.asarray([r.observables for r in detailed], dtype=np.uint8)
            ^ decoder.observable_offset,
            axis=1,
            bitorder="little",
        )
        iterations[start:stop] = [r.iterations for r in detailed]
        converged[start:stop] = [r.converged for r in detailed]
        temporary = directory / f"{name}_partial.tmp.npz"
        np.savez_compressed(
            temporary,
            predictions=predictions,
            iterations=iterations,
            converged=converged,
            completed=stop,
            instrument_seconds=instrument_seconds,
        )
        temporary.replace(partial)
        write_json(
            directory / "progress.json",
            {
                "profile": name,
                "distance": context.experiment.distance,
                "shots": stop,
                "target": count,
                "errors": count_failures(predictions[:stop], truth[:stop]),
                "nonconverged": int((~converged[:stop]).sum()),
            },
        )
    failures = np.any(predictions != truth, axis=1)
    assert np.all(iterations <= config.max_iterations)
    check = np.unique(
        np.r_[np.arange(8), np.flatnonzero(failures)[:8], np.flatnonzero(~converged)[:8]]
    )
    np.testing.assert_array_equal(
        decoder.decode_batch(np.ascontiguousarray(events[check])), predictions[check]
    )
    errors = int(failures.sum())
    low, high = wilson_interval(errors, count)
    row = {
        "profile": name,
        "config": asdict(config),
        "distance": context.experiment.distance,
        "noise": context.experiment.noise,
        "shots": count,
        "errors": errors,
        "logical_error_rate": errors / count,
        "ci95_low": low,
        "ci95_high": high,
        "nonconverged": int((~converged).sum()),
        "errors_nonconverged": int(np.count_nonzero(failures & ~converged)),
        "errors_converged": int(np.count_nonzero(failures & converged)),
        "total_legs": config.total_legs,
        "maximum_iterations": config.max_iterations,
        "iterations_median": float(np.median(iterations)),
        "iterations_mean": float(iterations.mean()),
        "iterations_p95": float(np.percentile(iterations, 95)),
        "iterations_p99": float(np.percentile(iterations, 99)),
        "iterations_max": int(iterations.max()),
        "active_checks": decoder.check_matrix.shape[0],
        "active_variables": decoder.check_matrix.shape[1],
        "detailed_instrumentation_seconds": instrument_seconds,
        "resource_estimate": decoder.resource_estimate.to_dict(),
    }
    np.savez_compressed(
        directory / f"{name}_raw.npz",
        predictions=predictions,
        iterations=iterations,
        converged=converged,
        logical_failures=failures,
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("past_experiments/decoder_testing/results/relay-iteration-audit-20261121"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration_files = [
        output / f"{stage}_{kind}.json"
        for stage in ("iterations", "paper")
        for kind in ("manifest", "results")
    ]
    identity = {
        "profiles": {name: asdict(config) for name, config in PROFILES.items()},
        "points": POINTS,
        "calibration_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in calibration_files
        },
        "source": implementation_identity(
            ("surface_decoders.ibm_relay_bp:build_accuracy_decoder",)
        ),
        "environment": environment_identity(),
        "warmup_shots": 64,
        "chunk_shots": 128,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(
            Path(__file__).with_name("audit_relay_iterations.py").read_bytes()
        ).hexdigest(),
        "budget_metric": "median",
        "budget_us": 1000,
        "selection": (
            "iteration candidate minimizes calibration errors among median-passing profiles; "
            "long memory search minimizes all iteration calibration errors; full XYZ paper S5 "
            "minimizes paper-comparison calibration errors; ties use latency"
        ),
        "validation_not_used_for_selection": True,
        "d11_note": "512-shot iteration diagnostic; insufficient precision for small LER claims",
    }
    identity = json.loads(json.dumps(identity))
    manifest = output / "validation_manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError(
                "resume rejected: sources, native dependency, profiles or inputs changed"
            )
        rows = json.loads((output / "validation_results.json").read_text())
    else:
        write_json(manifest, {"created_utc": datetime.now(UTC).isoformat(), "identity": identity})
        shutil.copy2(__file__, output / "validation_runner_snapshot.py")
        shutil.copy2(
            Path(__file__).with_name("audit_relay_iterations.py"),
            output / "calibration_runner_snapshot.py",
        )
        rows = []
        write_json(output / "validation_results.json", rows)
    for d, p, shots, seed, probes in POINTS:
        directory = output / "validation" / f"d{d}_p{p:g}"
        directory.mkdir(parents=True, exist_ok=True)
        context, events, truth = data(d, p, shots, seed)
        _, probe_events, _ = data(d, p, probes + 64, seed + 10000)
        reference = load_factory("pymatching")(context)
        predictions = reference.decode_batch(events)
        validate_predictions(predictions, shots, context.num_observables)
        np.savez_compressed(
            directory / "samples.npz",
            events=events,
            truth=truth,
            pymatching=predictions,
            probe_events=probe_events,
        )
        decoders = {name: IBMRelayBPDecoder(context, config) for name, config in PROFILES.items()}
        decoders["pymatching"] = reference
        timing_file = directory / "timing.npz"
        if timing_file.exists():
            with np.load(timing_file) as raw:
                times = {name: raw[name] for name in decoders}
        else:
            print("PROBING", d, p, flush=True)
            times = latency(decoders, probe_events, probes, warmup=64)
            np.savez_compressed(timing_file, **times)
        for name, config in PROFILES.items():
            if any(r["profile"] == name and r["distance"] == d and r["noise"] == p for r in rows):
                continue
            print(
                "SCORING",
                d,
                p,
                name,
                "median_ms",
                round(float(np.median(times[name])) / 1000, 4),
                flush=True,
            )
            row = score(decoders[name], context, events, truth, directory, name, config)
            add_timing(row, times[name])
            row["seed"] = seed
            rows.append(row)
            write_json(output / "validation_results.json", rows)
            print(
                "RESULT",
                d,
                p,
                name,
                "errors",
                row["errors"],
                "/",
                shots,
                "nonconv",
                row["nonconverged"],
                "mean_iter",
                round(row["iterations_mean"], 1),
                flush=True,
            )
        if not any(
            r["profile"] == "pymatching" and r["distance"] == d and r["noise"] == p for r in rows
        ):
            errors = count_failures(predictions, truth)
            low, high = wilson_interval(errors, shots)
            row = {
                "profile": "pymatching",
                "distance": d,
                "noise": p,
                "shots": shots,
                "errors": errors,
                "logical_error_rate": errors / shots,
                "ci95_low": low,
                "ci95_high": high,
                "seed": seed,
            }
            add_timing(row, times["pymatching"])
            rows.append(row)
            write_json(output / "validation_results.json", rows)
            print("REFERENCE", d, p, "pymatching", errors, "/", shots, flush=True)
    flat = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list, tuple))}
        for row in rows
    ]
    with (output / "validation_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in flat for k in row))
        )
        writer.writeheader()
        writer.writerows(flat)
    print("DONE", output, flush=True)


if __name__ == "__main__":
    main()

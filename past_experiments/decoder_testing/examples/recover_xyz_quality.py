"""One fresh fixed-size quality confirmation after a bounded quality-only cap.

Only points with a complete timing sample qualify. Counts depend on the independent
runtime pilot, never the interrupted quality outcomes. Original evidence is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from qecbench import DecoderContext, Experiment
from qecbench.decoders import load_factory
from qecbench.metrics import count_failures, wilson_interval
from qecbench.storage import implementation_identity, write_json
from surface_decoders.ibm_relay_bp import ACCURACY_CONFIG, IBMRelayBPDecoder


def worker(d, p, seed, shots, directory_text):
    os.environ["RAYON_NUM_THREADS"] = "1"
    directory = Path(directory_text)
    experiment = Experiment(d, d, p)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    events, truth = circuit.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True, bit_packed=True
    )
    events.flags.writeable = False
    matching = load_factory("pymatching")(context)
    reference = matching.decode_batch(events)
    decoder = IBMRelayBPDecoder(context, ACCURACY_CONFIG)
    predictions = np.zeros_like(truth)
    iterations = np.zeros(shots, dtype=np.int64)
    converged = np.zeros(shots, dtype=bool)
    for start in range(0, shots, 64):
        stop = min(shots, start + 64)
        syndrome = np.unpackbits(
            events[start:stop], axis=1, count=context.num_detectors, bitorder="little"
        )
        syndrome = np.ascontiguousarray(syndrome ^ decoder.detector_offset)
        syndrome.flags.writeable = False
        result = decoder.runner.decode_observables_detailed_batch(
            syndrome, parallel=False, progress_bar=False
        )
        predictions[start:stop] = np.packbits(
            np.asarray([r.observables for r in result], dtype=np.uint8) ^ decoder.observable_offset,
            axis=1,
            bitorder="little",
        )
        iterations[start:stop] = [r.iterations for r in result]
        converged[start:stop] = [r.converged for r in result]
    failures = np.any(predictions != truth, axis=1)
    errors = int(failures.sum())
    low, high = wilson_interval(errors, shots)
    assert np.all(iterations <= ACCURACY_CONFIG.max_iterations)
    np.savez_compressed(
        directory / "quality.npz",
        events=events,
        truth=truth,
        pymatching=reference,
        predictions=predictions,
        iterations=iterations,
        converged=converged,
        logical_failures=failures,
    )
    write_json(
        directory / "summary.json",
        {
            "quality_complete": True,
            "shots": shots,
            "errors": errors,
            "logical_error_rate": errors / shots,
            "ci95_low": low,
            "ci95_high": high,
            "nonconverged": int((~converged).sum()),
            "errors_nonconverged": int(np.count_nonzero(failures & ~converged)),
            "iterations_mean": float(iterations.mean()),
            "iterations_median": float(np.median(iterations)),
            "iterations_p95": float(np.percentile(iterations, 95)),
            "iterations_p99": float(np.percentile(iterations, 99)),
            "iterations_max": int(iterations.max()),
            "maximum_iterations": ACCURACY_CONFIG.max_iterations,
            "total_legs": ACCURACY_CONFIG.total_legs,
            "quality_seed": seed,
            "quality_threads": 1,
            "paired_shots": shots,
            "matching_paired_errors": count_failures(reference, truth),
            "quality_source": "fresh independent confirmation after quality runtime cap",
            "quality_directory": str(directory.resolve()),
            "paired_with_primary_matching_sample": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("past_experiments/decoder_testing/results/full-xyz-distance-noise-20261121-bounded"),
    )
    parser.add_argument("--replacement", type=Path, help="complete host-pause recheck directory")
    args = parser.parse_args()
    root = args.directory.resolve()
    master = json.loads((root / "manifest.json").read_text())["identity"]
    original = json.loads((root / "results.json").read_text())
    if len(original) != len(master["distances"]) * len(master["noise"]):
        raise ValueError("finish the primary bounded sweep before independent confirmations")
    if (
        implementation_identity(("surface_decoders.ibm_relay_bp:build_accuracy_decoder",))
        != master["source"]
    ):
        raise ValueError("source/native dependency changed")
    protocol = {
        "kind": "single fixed-size independent quality confirmation",
        "selection": "only quality-incomplete points with complete predetermined serial timings",
        "sample_rule": "clip(floor(30/pilot_cost),8,2048)",
        "quality_execution": "serial detailed native path; avoids per-shot parallel graph clones",
        "maximum_seconds": 90,
        "seed_offset": 109000000,
        "config": asdict(ACCURACY_CONFIG),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_results_sha256": hashlib.sha256((root / "results.json").read_bytes()).hexdigest(),
    }
    write_json(root / "quality_confirmation_manifest.json", protocol)
    points = json.loads(json.dumps(original))
    if args.replacement:
        replacement_root = args.replacement.resolve()
        replacement_manifest = json.loads((replacement_root / "manifest.json").read_text())[
            "identity"
        ]
        for key in master.keys() - {"distances", "noise"}:
            if replacement_manifest[key] != master[key]:
                raise ValueError(f"incompatible host-pause recheck: {key}")
        for replacement in json.loads((replacement_root / "results.json").read_text()):
            index = next(
                i
                for i, point in enumerate(points)
                if point["distance"] == replacement["distance"]
                and point["noise"] == replacement["noise"]
            )
            initial = points[index]
            if (
                initial["decoders"]["pymatching"].get("quality_complete")
                or initial.get("elapsed_seconds", 0) <= 600
                or initial.get("phase_at_cap") != "matching_reference"
            ):
                raise ValueError(
                    "only a host-interrupted incomplete matching point may be replaced"
                )
            replacement["measurement_directory"] = str(
                replacement_root / f"d{replacement['distance']}" / f"p{replacement['noise']:g}"
            )
            replacement["measurement_source"] = "whole-point host-pause recheck; original retained"
            points[index] = replacement
        protocol["replacement_directory"] = str(replacement_root)
        protocol["replacement_results_sha256"] = hashlib.sha256(
            (replacement_root / "results.json").read_bytes()
        ).hexdigest()
        write_json(root / "quality_confirmation_manifest.json", protocol)
    shutil.copy2(__file__, root / "quality_confirmation_snapshot.py")
    records = []
    context = mp.get_context("spawn")
    for point in points:
        native = point["decoders"]["relay_bp"]
        # Normalize a historical metadata bug: total-point expiry alone cannot
        # establish a 60-second lower bound for whichever call was then in flight.
        if (
            str(native.get("reason", "")).startswith("bounded point runtime")
            and "censored_call_lower_bound_seconds" in native
        ):
            native.pop("censored_call_lower_bound_seconds")
            native["censoring_note"] = (
                "individual-call lower bound unavailable for total-point expiry"
            )
        elif "censored_call_lower_bound_seconds" in native:
            native["watchdog_call_limit_seconds"] = native.pop("censored_call_lower_bound_seconds")
            native["censoring_note"] = (
                "watchdog limit observed without a complete call timing; not a measured latency"
            )
        if native.get("quality_complete") or "median_us" not in native:
            continue
        d, p = point["distance"], point["noise"]
        directory = root / "quality_confirmation" / f"d{d}_p{p:g}"
        directory.mkdir(parents=True, exist_ok=True)
        cost = point["plan"]["pilot_cost_upper_seconds"]
        shots = min(2048, max(8, int(30 / cost)))
        seed = point["condition_seed"] + 109000000
        if (
            not (directory / "summary.json").exists()
            and not (directory / "runtime_cap.json").exists()
        ):
            print("CONFIRM", d, p, "fixed shots", shots, flush=True)
            process = context.Process(target=worker, args=(d, p, seed, shots, str(directory)))
            process.start()
            process.join(timeout=90)
            if process.is_alive():
                process.terminate()
                process.join()
                write_json(
                    directory / "runtime_cap.json",
                    {"status": "runtime_cap", "seconds": 90, "planned_shots": shots, "seed": seed},
                )
            elif process.exitcode != 0:
                raise RuntimeError(f"quality confirmation worker failed: {process.exitcode}")
        if (directory / "summary.json").exists():
            summary = json.loads((directory / "summary.json").read_text())
            native["initial_runtime_cap"] = native.get("reason")
            native.update(summary, status="complete_quality_confirmation")
            native["partial_quality_is_not_an_estimate"] = True
            records.append({"distance": d, "noise": p, "status": "confirmed", **summary})
            print("CONFIRMED", d, p, summary["errors"], "/", shots, flush=True)
        else:
            records.append({"distance": d, "noise": p, "status": "confirmation_runtime_cap"})
    write_json(root / "analysis_results.json", points)
    write_json(root / "quality_confirmations.json", records)
    print("DONE", len(records), "independent confirmations attempted", flush=True)


if __name__ == "__main__":
    main()

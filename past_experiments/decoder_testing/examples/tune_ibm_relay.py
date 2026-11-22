"""Offline native Relay-BP calibration; held-out evaluation is a separate stage."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from qecbench import DecoderContext, Experiment
from surface_decoders import IBMRelayBPConfig, IBMRelayBPDecoder
from surface_decoders.ibm_relay_bp import FAST_CONFIG

OUTPUT = Path("past_experiments/decoder_testing/results/relay-tuning-20261121")
TRAIN_POINTS = [(7, 0.003), (9, 0.003), (7, 0.007)]


def dataset(distance, noise, shots, seed):
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


def evaluate(config, data, shots, probes=129):
    rows = []
    for context, events, truth in data:
        decoder = IBMRelayBPDecoder(context, config)
        for i in range(16):
            decoder.decode_batch(events[i : i + 1])
        times = []
        for i in range(probes):
            start = time.perf_counter_ns()
            decoder.decode_batch(events[i : i + 1])
            times.append((time.perf_counter_ns() - start) / 1000)
        median = float(np.median(times))
        row = {
            "distance": context.experiment.distance,
            "noise": context.experiment.noise,
            "median_us": median,
            "p95_us": float(np.percentile(times, 95)),
            "p99_us": float(np.percentile(times, 99)),
            "probes": probes,
            "median_pass": median <= 900,
            "variables": len(decoder.error_priors),
            "checks": decoder.check_matrix.shape[0],
        }
        # Calibration reserves 10% headroom; final admission uses the user's exact 1 ms.
        if row["median_pass"]:
            predictions = decoder.decode_batch(events[:shots])
            row.update(
                shots=shots,
                errors=int(np.count_nonzero(np.any(predictions != truth[:shots], axis=1))),
            )
        rows.append(row)
    return {
        "config": asdict(config),
        "points": rows,
        "accepted": all(r["median_pass"] for r in rows),
        "errors": sum(r["errors"] for r in rows) if all("errors" in r for r in rows) else None,
        "selection_penalty": sum(r.get("errors", shots) for r in rows),
        "max_median_us": max(r["median_us"] for r in rows),
    }


def save(name, rows):
    (OUTPUT / name).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine-ranges", action="store_true")
    args = parser.parse_args()
    if args.refine_ranges:
        return refine_ranges()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    coarse = [dataset(d, p, 1024, 703000 + i) for i, (d, p) in enumerate(TRAIN_POINTS)]
    configs = [
        replace(
            FAST_CONFIG,
            merge_equivalent_faults=True,
            detector_mode="memory_basis",
            pre_iter=60,
            num_sets=15,
            set_max_iter=60,
            gamma0=gamma,
            alpha=alpha,
        )
        for gamma in [0.0, 0.1, 0.25, 0.35, 0.5]
        for alpha in [None, 0.6, 0.8, 0.9, 0.0]
    ]
    configs += [
        replace(configs[0], detector_mode="full", gamma0=g, alpha=a)
        for g in [0.1, 0.35]
        for a in [None, 0.8]
    ]
    rows = []
    for i, config in enumerate(configs):
        row = evaluate(config, coarse, 1024)
        rows.append(row)
        save("coarse.json", rows)
        print(
            "coarse",
            i,
            "g",
            config.gamma0,
            "alpha",
            config.alpha,
            "mode",
            config.detector_mode,
            "accepted",
            row["accepted"],
            "errors",
            row["errors"],
            "max_med_us",
            round(row["max_median_us"], 2),
            flush=True,
        )
    shortlist = sorted(
        [r for r in rows if r["accepted"]], key=lambda r: (r["errors"], r["max_median_us"])
    )[:4]
    fine = [dataset(d, p, 4096, 704000 + i) for i, (d, p) in enumerate(TRAIN_POINTS)]
    fine_configs = []
    for row in shortlist:
        config = IBMRelayBPConfig(**row["config"])
        fine_configs += [replace(config, stop_nconv=s, num_sets=31) for s in [1, 3, 5]]
    fine_rows = []
    for i, config in enumerate(fine_configs):
        row = evaluate(config, fine, 4096, 257)
        fine_rows.append(row)
        save("fine.json", fine_rows)
        print(
            "fine",
            i,
            "g",
            config.gamma0,
            "alpha",
            config.alpha,
            "S",
            config.stop_nconv,
            "accepted",
            row["accepted"],
            "errors",
            row["errors"],
            "max_med_us",
            round(row["max_median_us"], 2),
            flush=True,
        )
    viable = sorted(
        [r for r in fine_rows if r["accepted"]], key=lambda r: (r["errors"], r["max_median_us"])
    )
    if not viable:
        raise RuntimeError("No profile passed calibration median budget")
    save("selected.json", viable[0])
    print("SELECTED", json.dumps(viable[0]), flush=True)


def refine_ranges():
    original = json.loads((OUTPUT / "selected.json").read_text())
    base = IBMRelayBPConfig(**original["config"])
    data = [dataset(d, p, 4096, 704000 + i) for i, (d, p) in enumerate(TRAIN_POINTS)]
    rows = []
    for bounds in [(-0.05, 0.5), (-0.1, 0.66), (-0.15, 0.8), (-0.25, 0.66)]:
        for solutions in [1, 3, 5]:
            config = replace(base, pre_iter=80, gamma_dist_interval=bounds, stop_nconv=solutions)
            row = evaluate(config, data, 4096, 257)
            rows.append(row)
            save("range_trials.json", rows)
            print(
                "range",
                bounds,
                "S",
                solutions,
                "accepted",
                row["accepted"],
                "errors",
                row["errors"],
                "max_med_us",
                round(row["max_median_us"], 2),
                flush=True,
            )
    viable = [r for r in [original, *rows] if r["accepted"]]
    chosen = min(viable, key=lambda r: (r["errors"], r["max_median_us"]))
    save("selected.json", chosen)
    print("SELECTED", json.dumps(chosen), flush=True)


if __name__ == "__main__":
    main()

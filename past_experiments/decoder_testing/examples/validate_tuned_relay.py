"""Held-out noise/distance sweep of a frozen IBM Relay-BP profile.

Preflight resources, then measure isolated-shot median latency before quality.
Stop each decoder's distance sweep at its first resource or median failure, for
each noise probability separately. Tail latency never controls admission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from latency_capped_sweep import matching_estimate, probe_latency

from qecbench import DecoderContext, Experiment, SweepConfig, run_sweep
from qecbench.storage import environment_identity, implementation_identity, write_json
from surface_decoders import ResourceLimits, estimate_ibm_resources, estimate_resources
from surface_decoders.ibm_relay_bp import TUNED_CONFIG

SPECS = {
    "relay_bp": "surface_decoders.ibm_relay_bp:build_tuned_decoder",
    "lut": "surface_decoders.lut:build_decoder",
    "pymatching": "pymatching",
}
DISTANCES = [2, 3, 5, 7, 9, 11, 13, 15, 17]
NOISE = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]


def dump_csv(output, rows):
    flat = []
    for row in rows:
        entry = {key: row[key] for key in ("noise", "distance", "rounds", "decoder", "status")}
        entry["reason"] = row.get("reason", "")
        for category in ("estimate", "measured"):
            for key, value in row.get(category, {}).items():
                if not isinstance(value, (dict, list, tuple)):
                    entry[f"{category}_{key}"] = value
        flat.append(entry)
    fields = list(dict.fromkeys(key for entry in flat for key in entry))
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def estimate(context, name, limits):
    if name == "relay_bp":
        result = estimate_ibm_resources(context, TUNED_CONFIG, limits).to_dict()
        result["scope"] = "conservative unprojected/unmerged graph upper bound"
        return result
    if name == "lut":
        return estimate_resources(context, "lut", limits=limits).to_dict()
    return matching_estimate(context, limits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("past_experiments/decoder_testing/results/relay-tuned-heldout-20261121")
    )
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--distance-shots", type=int, default=20_000)
    parser.add_argument("--latency-samples", type=int, default=2048)
    parser.add_argument("--latency-us", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=805001)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    limits = ResourceLimits()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "distances": DISTANCES,
        "noise": NOISE,
        "rounds": "distance",
        "basis": "z",
        "geometry": "rotated",
        "noise_model": "circuit",
        "shots": args.shots,
        "distance_shots_at_p_003": args.distance_shots,
        "latency_samples": args.latency_samples,
        "latency_warmup": 64,
        "latency_budget_us": args.latency_us,
        "latency_budget_metric": "median",
        "seed": args.seed,
        "limits": asdict(limits),
        "relay_config": asdict(TUNED_CONFIG),
        "environment": environment_identity(),
        "source": implementation_identity(tuple(SPECS.values())),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(
            Path(__file__).with_name("latency_capped_sweep.py").read_bytes()
        ).hexdigest(),
        "selection": {
            "directory": "../relay-tuning-20261121",
            "coarse_seeds": [703000, 703001, 703002],
            "fine_and_range_seeds": [704000, 704001, 704002],
            "heldout_not_used_for_selection": True,
        },
        "procedure": "serial probes first; paired circuit quality on admitted decoders only",
    }
    identity = json.loads(json.dumps(identity))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text())["identity"] != identity:
            raise ValueError(
                "output exists; resume requires identical settings, sources and environment"
            )
        rows = json.loads((output / "results.json").read_text())
    else:
        write_json(
            manifest_path, {"created_utc": datetime.now(UTC).isoformat(), "identity": identity}
        )
        rows = []
        write_json(output / "results.json", rows)
        shutil.copy2(__file__, output / "runner_snapshot.py")
        shutil.copy2(
            Path(__file__).with_name("latency_capped_sweep.py"), output / "probe_snapshot.py"
        )
    for p in NOISE:
        stopped = {}
        for d in DISTANCES:
            prior = [r for r in rows if r["noise"] == p and r["distance"] == d]
            if prior:
                for row in prior:
                    if row["status"] in ("resource_cap", "latency_cap"):
                        stopped[row["decoder"]] = f"d{d}: {row.get('reason', row['status'])}"
                continue
            experiment = Experiment(d, d, p)
            circuit = experiment.circuit()
            context = DecoderContext(
                experiment, circuit, circuit.detector_error_model(decompose_errors=True)
            )
            directory = output / f"p{p:g}" / f"d{d}"
            directory.mkdir(parents=True, exist_ok=True)
            point_rows, active = [], []
            for name in SPECS:
                row = {"noise": p, "distance": d, "rounds": d, "decoder": name}
                if name in stopped:
                    row.update(status="skipped_after_cap", reason=stopped[name])
                else:
                    row["estimate"] = estimate(context, name, limits)
                    if not row["estimate"]["accepted"]:
                        reason = "; ".join(row["estimate"]["reasons"])
                        row.update(status="resource_cap", reason=reason)
                        stopped[name] = f"d{d}: {reason}"
                    else:
                        active.append(name)
                point_rows.append(row)
            measured_path = directory / "latency_summary.json"
            if active:
                if measured_path.exists():
                    latency = json.loads(measured_path.read_text())
                else:
                    latency = probe_latency(
                        context,
                        active,
                        directory,
                        args.latency_samples,
                        64,
                        args.seed,
                        args.latency_us,
                        specs=SPECS,
                    )
                    write_json(measured_path, latency)
                admitted = []
                for row in point_rows:
                    name = row["decoder"]
                    if name not in active:
                        continue
                    row["measured"] = latency[name]
                    if latency[name]["latency_pass"]:
                        row["status"] = "accepted"
                        admitted.append(name)
                    else:
                        reason = (
                            f"median {latency[name]['single_shot_p50_us']:.3f} us "
                            f"> {args.latency_us:g} us"
                        )
                        row.update(status="latency_cap", reason=reason)
                        stopped[name] = f"d{d}: {reason}"
                print(
                    f"PROBED p={p:g} d={d}: "
                    + "; ".join(
                        f"{n} median={latency[n]['single_shot_p50_us']:.1f}us "
                        f"{'PASS' if n in admitted else 'CAP'}"
                        for n in active
                    ),
                    flush=True,
                )
                if admitted:
                    config = SweepConfig(
                        distances=(d,),
                        rounds=d,
                        noise=(p,),
                        decoders=tuple(SPECS[n] for n in admitted),
                        shots=args.distance_shots if p == 0.003 else args.shots,
                        batch_size=1024,
                        warmup_shots=64,
                        workers=1,
                        seed=args.seed,
                        output=str(directory / "quality"),
                    )
                    quality = run_sweep(
                        config, resume=(directory / "quality/manifest.json").exists()
                    )
                    by_spec = {r["decoder"]: r for r in quality}
                    for row in point_rows:
                        if row["decoder"] in admitted:
                            row["measured"].update(by_spec[SPECS[row["decoder"]]])
                    print(
                        f"QUALITY p={p:g} d={d}: "
                        + "; ".join(
                            f"{n} {by_spec[SPECS[n]]['errors']}/{by_spec[SPECS[n]]['shots']}"
                            for n in admitted
                        ),
                        flush=True,
                    )
            rows.extend(point_rows)
            write_json(output / "results.json", rows)
            dump_csv(output, rows)
    print(f"DONE {output}", flush=True)


if __name__ == "__main__":
    main()

"""Measure d11/13/15 beyond the latency stop without changing budget admission.

The user-requested extension is diagnostic. Storage/build caps remain enforced;
latency-failing shots are neither removed nor replaced with another decoder.
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
from surface_decoders import ResourceLimits, estimate_ibm_resources
from surface_decoders.ibm_relay_bp import TUNED_CONFIG

SPECS = {
    "relay_bp": "surface_decoders.ibm_relay_bp:build_tuned_decoder",
    "pymatching": "pymatching",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("past_experiments/decoder_testing/results/relay-d15-projection-d31-20261121"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("past_experiments/decoder_testing/results/relay-tuned-heldout-20261121"),
    )
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--latency-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=906321)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output, baseline = args.output.resolve(), args.baseline.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior_manifest = json.loads((baseline / "manifest.json").read_text())["identity"]
    current = implementation_identity(
        (SPECS["relay_bp"], "surface_decoders.lut:build_decoder", "pymatching")
    )
    if (
        prior_manifest["source"] != current
        or prior_manifest["environment"] != environment_identity()
    ):
        raise ValueError(
            "baseline source/native identity or environment changed; refusing silent reuse"
        )
    prior_rows = json.loads((baseline / "results.json").read_text())
    limits = ResourceLimits()
    identity = {
        "distances": [11, 13, 15],
        "noise": 0.003,
        "rounds": "distance",
        "shots": args.shots,
        "batch_size": 256,
        "latency_samples": args.latency_samples,
        "latency_warmup": 64,
        "latency_budget_us": 1000,
        "latency_budget_metric": "median",
        "seed": args.seed,
        "relay_config": asdict(TUNED_CONFIG),
        "limits": asdict(limits),
        "source": current,
        "environment": environment_identity(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(
            Path(__file__).with_name("latency_capped_sweep.py").read_bytes()
        ).hexdigest(),
        "baseline": str(baseline),
        "baseline_results_sha256": hashlib.sha256(
            (baseline / "results.json").read_bytes()
        ).hexdigest(),
        "baseline_manifest_sha256": hashlib.sha256(
            (baseline / "manifest.json").read_bytes()
        ).hexdigest(),
        "purpose": "diagnostic beyond median stop, not budget admission; no configuration retuning",
    }
    identity = json.loads(json.dumps(identity))
    path = output / "manifest.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text())["identity"] != identity:
            raise ValueError("resume requires identical sources, settings, inputs and environment")
        rows = json.loads((output / "extension_results.json").read_text())
    else:
        write_json(path, {"created_utc": datetime.now(UTC).isoformat(), "identity": identity})
        rows = []
        write_json(output / "extension_results.json", rows)
        shutil.copy2(__file__, output / "extension_runner_snapshot.py")
    for d in [11, 13, 15]:
        if any(row["distance"] == d for row in rows):
            continue
        experiment = Experiment(d, d, 0.003)
        circuit = experiment.circuit()
        context = DecoderContext(
            experiment, circuit, circuit.detector_error_model(decompose_errors=True)
        )
        directory = output / f"d{d}"
        directory.mkdir(parents=True, exist_ok=True)
        estimates = {
            "relay_bp": estimate_ibm_resources(context, TUNED_CONFIG, limits).to_dict(),
            "pymatching": matching_estimate(context, limits),
        }
        names = [name for name in SPECS if estimates[name]["accepted"]]
        latency_path = directory / "latency_summary.json"
        if latency_path.exists():
            latency = json.loads(latency_path.read_text())
        else:
            # Reuse a whole compatible timing point only, with raw evidence copied.
            previous = {
                name: next(
                    (
                        r
                        for r in prior_rows
                        if r["noise"] == 0.003
                        and r["distance"] == d
                        and r["decoder"] == name
                        and "single_shot_p50_us" in r.get("measured", {})
                    ),
                    None,
                )
                for name in names
            }
            if all(previous.values()):
                source = baseline / "p0.003" / f"d{d}"
                latency = json.loads((source / "latency_summary.json").read_text())
                for name in names:
                    latency[name]["timing_reused_from"] = str(source)
                shutil.copy2(source / "latency_samples.npz", directory / "latency_samples.npz")
            else:
                latency = probe_latency(
                    context,
                    names,
                    directory,
                    args.latency_samples,
                    64,
                    args.seed,
                    1000,
                    specs=SPECS,
                )
            write_json(latency_path, latency)
        print(
            f"TIMING d={d}: "
            + "; ".join(
                f"{name} median {latency[name]['single_shot_p50_us'] / 1000:.3f}ms"
                for name in names
            ),
            flush=True,
        )
        quality = run_sweep(
            SweepConfig(
                distances=(d,),
                rounds=d,
                noise=(0.003,),
                decoders=tuple(SPECS[name] for name in names),
                shots=args.shots,
                batch_size=256,
                warmup_shots=32,
                workers=1,
                seed=args.seed,
                output=str(directory / "quality"),
            ),
            resume=(directory / "quality/manifest.json").exists(),
        )
        by_spec = {row["decoder"]: row for row in quality}
        for name in SPECS:
            row = {
                "noise": 0.003,
                "distance": d,
                "rounds": d,
                "decoder": name,
                "estimate": estimates[name],
                "diagnostic_extension": True,
            }
            if name not in names:
                row.update(status="resource_cap", reason="; ".join(estimates[name]["reasons"]))
            else:
                measured = {**latency[name], **by_spec[SPECS[name]]}
                row.update(
                    measured=measured,
                    budget_admitted=measured["latency_pass"],
                    status="accepted"
                    if measured["latency_pass"]
                    else "diagnostic_over_latency_budget",
                )
            rows.append(row)
        write_json(output / "extension_results.json", rows)
        print(
            f"QUALITY d={d}: "
            + "; ".join(
                f"{name} {by_spec[SPECS[name]]['errors']}/{by_spec[SPECS[name]]['shots']}"
                for name in names
            ),
            flush=True,
        )
    flat = []
    for row in rows:
        item = {key: value for key, value in row.items() if key not in ("estimate", "measured")}
        item.update(
            {
                key: value
                for key, value in row.get("measured", {}).items()
                if not isinstance(value, (dict, list, tuple))
            }
        )
        item["peak_working_bytes"] = row["estimate"]["peak_working_bytes"]
        flat.append(item)
    with (output / "extension_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
        writer.writeheader()
        writer.writerows(flat)
    print("DONE", output, flush=True)


if __name__ == "__main__":
    main()

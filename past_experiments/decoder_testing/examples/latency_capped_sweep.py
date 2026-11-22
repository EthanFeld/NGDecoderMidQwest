"""Run paired decoder sweeps with resource preflight and measured latency admission.

The first latency-failing point is retained as a diagnostic. Larger distances are
skipped for that decoder. Admission uses median isolated-shot wall time; p95,
p99, and maximum are diagnostics and do not reject an otherwise passing median.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qecbench import DecoderContext, Experiment, SweepConfig, run_sweep
from qecbench.decoders import load_factory, validate_predictions
from qecbench.runner import batch_seed
from qecbench.storage import environment_identity, implementation_identity, write_json
from surface_decoders import ResourceLimits, estimate_ibm_resources, estimate_resources
from surface_decoders.ibm_relay_bp import ACCURACY_CONFIG, FAST_CONFIG, TUNED_CONFIG
from surface_decoders.model import model_stats

SPECS = {
    "relay_bp": "surface_decoders.ibm_relay_bp:build_fast_decoder",
    "lut": "surface_decoders.lut:build_decoder",
    "pymatching": "pymatching",
}
LABELS = {"relay_bp": "IBM Relay-BP", "lut": "LUT", "pymatching": "PyMatching"}


def relay_profile(name):
    """Return the same frozen configuration for resource planning and execution."""
    config, factory = {
        "fast": (FAST_CONFIG, "build_fast_decoder"),
        "tuned": (TUNED_CONFIG, "build_tuned_decoder"),
        "accuracy": (ACCURACY_CONFIG, "build_accuracy_decoder"),
    }[name]
    return config, {**SPECS, "relay_bp": f"surface_decoders.ibm_relay_bp:{factory}"}


def matching_estimate(context: DecoderContext, limits: ResourceLimits) -> dict:
    """Conservative graph allocation model, not an instrumented native peak.

    Each decomposed component has at least one detector incidence unless it is
    detector-free, so incidences + mechanisms bounds graph edge count. Budget
    256 bytes/edge, 128/node, three graph copies for compilation/runtime scratch.
    This deliberately overcounts merged matching edges. No static latency model.
    """
    if (
        context.dem.num_errors > limits.max_model_mechanisms
        or context.num_detectors > limits.max_detectors
    ):
        return {"accepted": False, "reasons": ["DEM scan cap"], "peak_working_bytes": None}
    stats = model_stats(context)
    edges = stats.edges + stats.mechanisms
    persistent = 4096 + 256 * edges + 128 * (stats.detectors + 2) + 16 * stats.observables
    peak = 3 * persistent + 64 * stats.detectors
    accepted = peak <= limits.max_memory_mib * 1024**2
    return {
        "decoder": "pymatching",
        "stats": asdict(stats),
        "persistent_bytes": persistent,
        "peak_working_bytes": peak,
        "build_operations": None,
        "decode_operations_per_shot": None,
        "estimated_latency_us": None,
        "accepted": accepted,
        "reasons": [] if accepted else [f"graph peak estimate {peak} bytes exceeds memory cap"],
        "model": "256 bytes/edge upper bound + 128/node; 3x graph + detector scratch",
    }


def latency_admission(samples_us: np.ndarray, budget_us: float) -> bool:
    samples_us = np.asarray(samples_us)
    return bool(
        samples_us.ndim == 1
        and samples_us.size > 0
        and np.all(np.isfinite(samples_us))
        and np.all(samples_us >= 0)
        and math.isfinite(budget_us)
        and budget_us > 0
        and np.median(samples_us) <= budget_us
    )


def latency_metrics(samples_us: np.ndarray, budget_us: float) -> dict:
    samples_us = np.asarray(samples_us)
    if (
        samples_us.ndim != 1
        or not samples_us.size
        or not np.all(np.isfinite(samples_us))
        or np.any(samples_us < 0)
    ):
        raise ValueError("latency samples must be a nonempty finite nonnegative vector")
    p50, p95, p99 = np.percentile(samples_us, [50, 95, 99], method="linear")
    return {
        "single_shot_p50_us": float(p50),
        "single_shot_p95_us": float(p95),
        "single_shot_p99_us": float(p99),
        "single_shot_max_us": float(np.max(samples_us)),
        "latency_budget_us": budget_us,
        "latency_budget_metric": "median",
        "single_shot_over_budget_count": int(np.count_nonzero(samples_us > budget_us)),
        "single_shot_over_budget_fraction": float(np.mean(samples_us > budget_us)),
        "latency_pass": latency_admission(samples_us, budget_us),
    }


def probe_latency(context, names, output, samples, warmup, seed, budget_us, specs=None):
    """Serial wall/CPU timings of actual one-shot packed decode calls."""
    sampler = context.circuit.compile_detector_sampler(
        seed=batch_seed(seed, context.experiment, -2)
    )
    events = sampler.sample(shots=samples + warmup, bit_packed=True)
    events.flags.writeable = False
    specs = SPECS if specs is None else specs
    decoders = {name: load_factory(specs[name])(context) for name in names}
    for i in range(warmup):
        for name in names:
            validate_predictions(
                decoders[name].decode_batch(events[i : i + 1]), 1, context.num_observables
            )
    wall = {name: np.empty(samples) for name in names}
    cpu = {name: np.empty(samples) for name in names}
    for i in range(samples):
        shot = events[warmup + i : warmup + i + 1]
        # Rotate decoder order while keeping every one-shot call serialized.
        order = names[i % len(names) :] + names[: i % len(names)]
        for name in order:
            cpu_start = time.thread_time_ns()
            start = time.perf_counter_ns()
            predictions = decoders[name].decode_batch(shot)
            elapsed = time.perf_counter_ns() - start
            cpu_elapsed = time.thread_time_ns() - cpu_start
            validate_predictions(predictions, 1, context.num_observables)
            wall[name][i] = elapsed / 1000
            cpu[name][i] = cpu_elapsed / 1000
    np.savez_compressed(
        output / "latency_samples.npz",
        events=events[warmup:],
        **{f"{n}_wall_us": wall[n] for n in names},
        **{f"{n}_cpu_us": cpu[n] for n in names},
    )
    results = {}
    for name in names:
        times = wall[name]
        result = {
            "latency_samples": samples,
            "latency_warmup_shots": warmup,
            "single_shot_cpu_max_us": float(np.max(cpu[name])),
            **latency_metrics(times, budget_us),
        }
        if name == "pymatching":
            result.update(
                matching_nodes=decoders[name].matching.num_nodes,
                matching_edges=decoders[name].matching.num_edges,
            )
        if hasattr(decoders[name], "check_matrix"):
            result.update(
                active_checks=decoders[name].check_matrix.shape[0],
                active_variables=decoders[name].check_matrix.shape[1],
                active_edges=decoders[name].check_matrix.nnz,
            )
        results[name] = result
    return results


def point(context, names, output, args):
    specs = getattr(args, "decoder_specs", SPECS)
    config = SweepConfig(
        distances=(context.experiment.distance,),
        noise=(args.noise,),
        rounds=context.experiment.rounds,
        bases=("z",),
        geometries=("rotated",),
        decoders=tuple(specs[name] for name in names),
        shots=args.shots,
        batch_size=1024,
        warmup_shots=64,
        workers=1,
        seed=args.seed,
        output=str(output),
    )
    rows = run_sweep(config)
    latency = probe_latency(
        context, names, output, args.latency_samples, 64, args.seed, args.latency_us, specs=specs
    )
    by_spec = {row["decoder"]: row for row in rows}
    return {name: {**by_spec[specs[name]], **latency[name]} for name in names}


def load_reuse(source: Path | None, args, limits) -> dict:
    """Reuse only compatible whole points, retaining original timing provenance."""
    if source is None:
        return {}
    from qecbench.storage import read_json

    manifest = read_json(source / "manifest.json")
    expected = {
        "shots_per_point": args.shots,
        "latency_samples": args.latency_samples,
        "noise": args.noise,
        "seed": args.seed,
        "limits": asdict(limits),
        "environment": environment_identity(),
        "source": implementation_identity(tuple(getattr(args, "decoder_specs", SPECS).values())),
    }
    # JSON normalizes tuples to lists (notably the native gamma interval).
    import json

    expected["relay_config"] = json.loads(
        json.dumps(asdict(getattr(args, "relay_config", FAST_CONFIG)))
    )
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"reuse rejected; incompatible fields: {', '.join(mismatches)}")
    return {
        (r["track"], r["distance"], r["decoder"]): r
        for r in read_json(source / "results.json")
        if "measured" in r
    }


def reuse_point(source, track, distance, names, output, prior, args):
    keys = [(track, distance, name) for name in names]
    if not all(key in prior for key in keys):
        return None
    prior_names = {key[2] for key in prior if key[:2] == (track, distance)}
    if prior_names != set(names):
        return None
    original = source / track / f"d{distance}"
    with np.load(original / "latency_samples.npz", allow_pickle=False) as saved:
        results = {}
        for name, key in zip(names, keys, strict=True):
            measured = dict(prior[key]["measured"])
            measured.pop("latency_budget_misses", None)
            measured.update(latency_metrics(saved[f"{name}_wall_us"], args.latency_us))
            measured["reused_from"] = str(original.resolve())
            results[name] = measured
    shutil.copytree(original, output)
    write_json(output / "median_latency_summary.json", results)
    return results


def write_csv(path: Path, rows: list[dict]):
    flat = []
    for row in rows:
        estimate = row.get("estimate", {})
        measured = row.get("measured", {})
        flat.append(
            {
                "track": row["track"],
                "distance": row["distance"],
                "rounds": row["rounds"],
                "decoder": row["decoder"],
                "status": row["status"],
                "reason": row.get("reason", ""),
                **{
                    key: estimate.get(key)
                    for key in (
                        "persistent_bytes",
                        "peak_working_bytes",
                        "build_operations",
                        "decode_operations_per_shot",
                    )
                },
                **{
                    key: measured.get(key)
                    for key in (
                        "shots",
                        "errors",
                        "logical_error_rate",
                        "ci95_low",
                        "ci95_high",
                        "compile_seconds",
                        "decode_us_per_shot",
                        "decode_shots_per_second",
                        "single_shot_p50_us",
                        "single_shot_p95_us",
                        "single_shot_p99_us",
                        "single_shot_max_us",
                        "single_shot_cpu_max_us",
                        "latency_samples",
                        "latency_budget_metric",
                        "latency_budget_us",
                        "latency_pass",
                        "single_shot_over_budget_count",
                        "single_shot_over_budget_fraction",
                        "reused_from",
                    )
                },
            }
        )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def plot(rows, output, track, latency_us, memory_mib, noise, relay_config=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"relay_bp": "#b35c16", "lut": "#258567", "pymatching": "#2563b4"}
    for name, label in LABELS.items():
        points = [r for r in rows if r["track"] == track and r["decoder"] == name]
        measured = [r for r in points if "measured" in r]
        color = colors[name]
        for ax, field in [(axes[0, 1], "decode_us_per_shot"), (axes[1, 0], "single_shot_p50_us")]:
            ax.plot(
                [r["distance"] for r in measured],
                [r["measured"][field] for r in measured],
                "o-",
                color=color,
                label=label,
            )
        for field, style in [("single_shot_p95_us", "--"), ("single_shot_p99_us", ":")]:
            axes[1, 0].plot(
                [r["distance"] for r in measured],
                [r["measured"][field] for r in measured],
                style,
                color=color,
                alpha=0.65,
                linewidth=1.2,
            )
        rates = []
        for r in measured:
            m = r["measured"]
            value = m["logical_error_rate"] or m["ci95_high"]
            rates.append(value)
            if m["errors"]:
                axes[0, 0].errorbar(
                    r["distance"],
                    value,
                    yerr=[[max(0, value - m["ci95_low"])], [max(0, m["ci95_high"] - value)]],
                    fmt="o",
                    color=color,
                    capsize=3,
                )
            else:
                axes[0, 0].plot(r["distance"], value, "v", color=color, markerfacecolor="none")
        axes[0, 0].plot([r["distance"] for r in measured], rates, "-", color=color, label=label)
        estimated = [
            r for r in points if r.get("estimate", {}).get("peak_working_bytes") is not None
        ]
        axes[1, 1].plot(
            [r["distance"] for r in estimated],
            [r["estimate"]["peak_working_bytes"] / 1024**2 for r in estimated],
            "o-",
            color=color,
            label=label,
        )
        for r in points:
            if r["status"] in ("latency_cap", "resource_cap"):
                if "measured" in r:
                    axes[1, 0].scatter(
                        r["distance"],
                        r["measured"]["single_shot_p50_us"],
                        marker="x",
                        s=170,
                        color="#bf2222",
                        zorder=10,
                    )
                axes[1, 1].annotate(
                    f"{label} stop: d={r['distance']}",
                    xy=(0.02, 0.98 - 0.08 * list(LABELS).index(name)),
                    xycoords="axes fraction",
                    va="top",
                    color=color,
                    fontsize=9,
                )
    axes[1, 0].axhline(
        latency_us, color="#bf2222", ls="--", label=f"{latency_us:g} us median budget"
    )
    axes[1, 1].axhline(memory_mib, color="#bf2222", ls="--", label=f"{memory_mib:g} MiB ceiling")
    for ax, title, ylabel in [
        (axes[0, 0], "Logical error rate (95% Wilson interval)", "Failure probability / shot"),
        (axes[0, 1], "Measured batch decode speed", "Amortized microseconds / shot"),
        (axes[1, 0], "Solo latency: median solid, p95 dashed, p99 dotted", "Microseconds"),
        (axes[1, 1], "Estimated peak decoder memory", "MiB (planning model)"),
    ]:
        ax.set(title=title, xlabel="Code distance", ylabel=ylabel)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=9)
    rounds = "rounds = distance" if track == "distance_rounds" else "rounds = 1 (companion)"
    profile_note = (
        f"\nRelay: {relay_config.total_legs} legs; "
        f"cap {relay_config.max_iterations:,} BP iterations; "
        f"S={relay_config.stop_nconv}; {relay_config.detector_mode}"
        if relay_config is not None
        else ""
    )
    fig.suptitle(
        f"Rotated surface code, Z memory, p={noise:g} circuit noise; {rounds}{profile_note}",
        fontsize=13,
    )
    fig.text(
        0.02,
        0.015,
        "First failing point retained as diagnostic; larger points skipped. "
        "Open triangles: zero errors, upper 95% endpoint.\n"
        "Admission uses median only; p95/p99 may exceed budget. "
        "PyMatching memory uses conservative graph assumptions.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.93 if relay_config is not None else 0.96))
    fig.savefig(output / f"{track}.png", dpi=170)
    fig.savefig(output / f"{track}.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--latency-samples", type=int, default=2048)
    parser.add_argument(
        "--latency-us",
        type=float,
        default=1000,
        help="median isolated-shot latency budget, in microseconds",
    )
    parser.add_argument(
        "--reuse-results",
        type=Path,
        help="reuse compatible completed points, recomputing percentiles from raw probes",
    )
    parser.add_argument("--noise", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument(
        "--relay-profile",
        choices=["fast", "tuned", "accuracy"],
        default="tuned",
        help="native preset: tuned=32 legs (default), accuracy=601 full XYZ legs, fast=8 legs",
    )
    args = parser.parse_args()
    args.relay_config, args.decoder_specs = relay_profile(args.relay_profile)
    print(
        f"Relay profile={args.relay_profile}: {args.relay_config.total_legs} total legs, "
        f"at most {args.relay_config.max_iterations:,} BP iterations, "
        f"S={args.relay_config.stop_nconv}, graph={args.relay_config.detector_mode}",
        flush=True,
    )
    if (
        args.shots < 1
        or args.latency_samples < 1
        or not math.isfinite(args.latency_us)
        or args.latency_us <= 0
    ):
        parser.error("shots, latency samples, and latency budget must be positive")
    limits = ResourceLimits()
    prior = load_reuse(args.reuse_results, args, limits)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "started_utc": datetime.now(UTC).isoformat(),
        "shots_per_point": args.shots,
        "latency_budget_us": args.latency_us,
        "latency_budget_metric": "median",
        "latency_percentile_method": "numpy.percentile(method='linear')",
        "latency_samples": args.latency_samples,
        "noise": args.noise,
        "basis": "z",
        "geometry": "rotated",
        "noise_model": "circuit",
        "seed": args.seed,
        "limits": asdict(limits),
        "environment": environment_identity(),
        "source": implementation_identity(tuple(args.decoder_specs.values())),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "latency_policy": (
            "observed median isolated-shot wall-clock decode time <= budget; p95/p99/max diagnostic"
        ),
        "reused_results": str(args.reuse_results.resolve()) if args.reuse_results else None,
        "static_latency_rate": None,
        "resource_policy": "existing Relay/LUT models; conservative graph model for PyMatching",
        "relay_backend": "IBM authors' pinned native Rust implementation",
        "relay_profile": args.relay_profile,
        "relay_config": asdict(args.relay_config),
        "relay_search_budget": {
            "total_legs": args.relay_config.total_legs,
            "maximum_iterations": args.relay_config.max_iterations,
        },
        "memory_scope": (
            "per-decoder estimates; excludes interpreter, circuit/DEM and harness buffers"
        ),
        "diagnostics": (
            "quality and timing retained at first latency-failing point; larger points skipped"
        ),
    }
    write_json(output / "manifest.json", metadata)
    rows = []
    for track, distances in [
        ("distance_rounds", [2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31]),
        ("one_round", [3, 5]),
    ]:
        stopped = {}
        for distance in distances:
            rounds = distance if track == "distance_rounds" else 1
            current = []
            for name in SPECS:
                row = dict(track=track, distance=distance, rounds=rounds, decoder=name)
                if name in stopped:
                    row.update(status="skipped_after_cap", reason=stopped[name])
                else:
                    row["status"] = "pending"
                current.append(row)
            if any(row["status"] == "pending" for row in current):
                experiment = Experiment(distance, rounds, args.noise)
                circuit = experiment.circuit()
                context = DecoderContext(
                    experiment, circuit, circuit.detector_error_model(decompose_errors=True)
                )
                candidates = []
                for row in current:
                    if row["status"] != "pending":
                        continue
                    name = row["decoder"]
                    estimate = (
                        matching_estimate(context, limits)
                        if name == "pymatching"
                        else (
                            estimate_ibm_resources(context, args.relay_config, limits).to_dict()
                            if name == "relay_bp"
                            else estimate_resources(context, name, limits=limits).to_dict()
                        )
                    )
                    row["estimate"] = estimate
                    if not estimate["accepted"]:
                        reason = "; ".join(estimate["reasons"])
                        row.update(status="resource_cap", reason=reason)
                        stopped[name] = f"d={distance}: {reason}"
                    else:
                        candidates.append(name)
                if candidates:
                    point_output = output / track / f"d{distance}"
                    measured = (
                        reuse_point(
                            args.reuse_results,
                            track,
                            distance,
                            candidates,
                            point_output,
                            prior,
                            args,
                        )
                        if prior
                        else None
                    )
                    if measured is None:
                        measured = point(context, candidates, point_output, args)
                    for row in current:
                        name = row["decoder"]
                        if name not in measured:
                            continue
                        row["measured"] = measured[name]
                        row["status"] = (
                            "pass_observed" if measured[name]["latency_pass"] else "latency_cap"
                        )
                        if row["status"] == "latency_cap":
                            reason = (
                                f"median {measured[name]['single_shot_p50_us']:.3f} us > "
                                f"{args.latency_us:g} us; "
                                f"p95={measured[name]['single_shot_p95_us']:.3f} us; "
                                f"p99={measured[name]['single_shot_p99_us']:.3f} us"
                            )
                            row["reason"] = reason
                            stopped[name] = f"d={distance}: {reason}"
            rows.extend(current)
            write_json(output / "results.json", rows)
            write_csv(output / "results.csv", rows)
            print(
                track,
                f"d={distance}",
                "; ".join(
                    f"{r['decoder']}={r['status']}"
                    + (
                        f" LER={r['measured']['logical_error_rate']:.6g}"
                        f" batch_us={r['measured']['decode_us_per_shot']:.3f}"
                        f" median_us={r['measured']['single_shot_p50_us']:.3f}"
                        f" p95_us={r['measured']['single_shot_p95_us']:.3f}"
                        f" p99_us={r['measured']['single_shot_p99_us']:.3f}"
                        if "measured" in r
                        else ""
                    )
                    for r in current
                ),
                flush=True,
            )
        plot(
            rows,
            output,
            track,
            args.latency_us,
            limits.max_memory_mib,
            args.noise,
            relay_config=args.relay_config,
        )
    metadata["completed_utc"] = datetime.now(UTC).isoformat()
    write_json(output / "manifest.json", metadata)
    print(output, flush=True)


if __name__ == "__main__":
    main()

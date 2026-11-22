"""Bounded full-XYZ accuracy sweep; incomplete samples never become LER estimates.

Independent runtime pilots select fixed per-point sample counts. Normal packed
single-shot latency is serial; only quality instrumentation uses native Rayon.
The 1 ms median budget is marked, not used to omit requested diagnostic distances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

if __package__:
    from .latency_capped_sweep import matching_estimate
else:
    from latency_capped_sweep import matching_estimate

from qecbench import DecoderContext, Experiment
from qecbench.decoders import load_factory, validate_predictions
from qecbench.metrics import count_failures, wilson_interval
from qecbench.storage import environment_identity, implementation_identity
from qecbench.storage import write_json as atomic_write_json
from surface_decoders import ResourceLimits
from surface_decoders.ibm_relay_bp import ACCURACY_CONFIG, IBMRelayBPDecoder, estimate_ibm_resources

DISTANCES = [3, 5, 7, 9, 11, 13, 15]
NOISE = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]


def write_json(path, value):
    """Retry transient Windows reader/indexer locks without weakening atomic saves."""
    for attempt in range(7):
        try:
            return atomic_write_json(path, value)
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.01 * 2**attempt)


def sample_plan(pilot_seconds, remaining_seconds, threads, max_shots, max_probes):
    """Use only an independent runtime pilot, never quality outcomes."""
    pilot = np.asarray(pilot_seconds, dtype=float)
    if not 1 <= len(pilot) <= 4 or not np.all(np.isfinite(pilot)) or np.any(pilot <= 0):
        raise ValueError("need one to four complete positive independent pilot timings")
    if remaining_seconds < 20:
        raise ValueError("insufficient remaining point budget after pilot")
    cost = float(pilot.max()) * (2 if len(pilot) < 4 else 1)
    probe_budget = min(60.0, remaining_seconds * 0.30)
    quality_budget = min(120.0, remaining_seconds * 0.55)
    return {
        "pilot_cost_upper_seconds": cost,
        "pilot_shots": len(pilot),
        "latency_samples": min(max_probes, max(2, int(probe_budget / cost))),
        "quality_shots": min(max_shots, max(8, int(quality_budget * threads / cost))),
        "quality_chunk": max(threads, min(128, max(1, int(2 * threads / cost)))),
        "extra_warmup": min(16, max(0, int(1 / cost))),
        "quality_threads": threads,
        "counts_fixed_before_scoring": True,
    }


def quality_worker_count(requested, native_peak, matching_peak, allowance_bytes):
    """Include original graphs as well as upstream per-worker decoder clones."""
    clone_allowance = allowance_bytes - native_peak - matching_peak
    clones = max(0, int(clone_allowance // native_peak))
    return max(1, min(requested, clones))


def timing_summary(times_us):
    values = np.asarray(times_us, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("timings must be complete finite nonnegative samples")
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {
        "latency_samples": len(values),
        "median_us": float(p50),
        "p95_us": float(p95) if len(values) >= 20 else None,
        "p99_us": float(p99) if len(values) >= 20 else None,
        "median_resolved": len(values) >= 32,
        "p95_resolved": len(values) >= 200,
        "p99_resolved": len(values) >= 1000,
        "median_budget_us": 1000,
        "median_budget_pass": bool(p50 <= 1000),
        "max_us": float(values.max()),
        "tail_note": (
            "empirical percentiles; fewer than 10 expected tail observations is underresolved"
        ),
    }


def quality_summary(predictions, truth):
    errors = count_failures(predictions, truth)
    low, high = wilson_interval(errors, len(truth))
    return {
        "shots": len(truth),
        "errors": errors,
        "logical_error_rate": errors / len(truth),
        "ci95_low": low,
        "ci95_high": high,
        "quality_complete": True,
    }


def watchdog_call_metadata(call_started, observed, limit_seconds):
    """Describe watchdog observation without fabricating a decode latency bound."""
    if call_started is None:
        return {}
    return {
        "watchdog_call_elapsed_seconds": max(0.0, observed - call_started),
        "watchdog_call_limit_seconds": limit_seconds,
        "censoring_note": "no complete call timing; watchdog observation is not a decode latency",
    }


def run_worker(d, p, directory_text, options, channel):
    directory = Path(directory_text)
    started = time.monotonic()
    state = {
        "distance": d,
        "rounds": d,
        "noise": p,
        "status": "running",
        "phase": "initializing",
        "decoders": {"relay_bp": {"status": "pending"}, "pymatching": {"status": "pending"}},
    }
    last_save = 0.0

    def emit(event, **values):
        channel.send({"event": event, **values})

    def save(force=True):
        nonlocal last_save
        if not force and time.monotonic() - last_save < 0.5:
            return
        write_json(directory / "state.json", state)
        last_save = time.monotonic()

    def timed_call(decoder, event, algorithm, index):
        emit("call_start", algorithm=algorithm, index=index, phase=state["phase"])
        clock = time.perf_counter_ns()
        prediction = decoder.decode_batch(event)
        elapsed = (time.perf_counter_ns() - clock) / 1000
        emit("call_done", algorithm=algorithm, index=index)
        return prediction, elapsed

    try:
        os.environ["RAYON_NUM_THREADS"] = str(options["threads"])
        experiment = Experiment(d, d, p)
        circuit = experiment.circuit()
        context = DecoderContext(
            experiment, circuit, circuit.detector_error_model(decompose_errors=True)
        )
        limits = ResourceLimits()
        estimates = {
            "relay_bp": estimate_ibm_resources(context, ACCURACY_CONFIG, limits).to_dict(),
            "pymatching": matching_estimate(context, limits),
        }
        state["estimates"] = estimates
        seed = options["seed"] + 100000 * d + int(round(p * 1e6))
        state["condition_seed"] = seed
        sampler = circuit.compile_detector_sampler(seed=seed + 3000000)
        events, truth = sampler.sample(
            shots=options["max_shots"], separate_observables=True, bit_packed=True
        )
        events.flags.writeable = False
        probe_events = circuit.compile_detector_sampler(seed=seed + 2000000).sample(
            shots=options["max_probes"] + 16, bit_packed=True
        )
        probe_events.flags.writeable = False
        state["phase"] = "matching_reference"
        save()
        emit("phase", phase=state["phase"])
        if not estimates["pymatching"]["accepted"]:
            state["decoders"]["pymatching"] = {
                "status": "resource_cap",
                "reason": estimates["pymatching"]["reasons"],
            }
            raise ValueError("matching preflight unexpectedly rejected an in-range point")
        matching = load_factory("pymatching")(context)
        reference = matching.decode_batch(events)
        validate_predictions(reference, len(events), context.num_observables)
        for i in range(16):
            matching.decode_batch(probe_events[i : i + 1])
        pm_times = np.empty(options["max_probes"])
        for i in range(options["max_probes"]):
            _, pm_times[i] = timed_call(matching, probe_events[16 + i : 17 + i], "pymatching", i)
        np.savez_compressed(
            directory / "samples.npz",
            events=events,
            truth=truth,
            pymatching=reference,
            probe_events=probe_events,
        )
        np.savez_compressed(directory / "pymatching_timing.npz", wall_us=pm_times)
        state["decoders"]["pymatching"] = {
            "status": "complete",
            **quality_summary(reference, truth),
            **timing_summary(pm_times),
        }
        save()
        if not estimates["relay_bp"]["accepted"]:
            state["decoders"]["relay_bp"] = {
                "status": "resource_cap",
                "reason": estimates["relay_bp"]["reasons"],
            }
            state["status"] = "complete_with_resource_cap"
            save()
            emit("complete")
            return
        threads = quality_worker_count(
            options["threads"],
            estimates["relay_bp"]["peak_working_bytes"],
            estimates["pymatching"]["peak_working_bytes"],
            options["parallel_memory_mib"] * 1024**2,
        )
        os.environ["RAYON_NUM_THREADS"] = str(threads)
        decoder = IBMRelayBPDecoder(context, ACCURACY_CONFIG, limits)
        state["phase"] = "independent_pilot"
        save()
        emit("phase", phase=state["phase"])
        pilot_events = circuit.compile_detector_sampler(seed=seed + 1000000).sample(
            shots=4, bit_packed=True
        )
        pilot_events.flags.writeable = False
        pilot_times = []
        for i in range(4):
            _, elapsed = timed_call(decoder, pilot_events[i : i + 1], "relay_bp", i)
            pilot_times.append(elapsed / 1e6)
            state["pilot_seconds"] = pilot_times.copy()
            save()
            if elapsed / 1e6 >= 2:
                break  # A slow independent pilot gets a 2x planning safety factor.
        remaining = options["point_seconds"] - (time.monotonic() - started)
        plan = sample_plan(
            pilot_times, remaining, threads, options["max_shots"], options["max_probes"]
        )
        state["plan"] = plan
        state["phase"] = "relay_latency"
        save()
        emit("phase", phase=state["phase"], plan=plan)
        for i in range(plan["extra_warmup"]):
            timed_call(decoder, probe_events[i : i + 1], "relay_warmup", i)
        times = np.empty(plan["latency_samples"])
        timed_predictions = np.zeros((len(times), truth.shape[1]), dtype=np.uint8)
        for i in range(len(times)):
            prediction, times[i] = timed_call(decoder, probe_events[16 + i : 17 + i], "relay_bp", i)
            timed_predictions[i] = prediction[0]
            if (i + 1) % 16 == 0:
                state["latency_completed"] = i + 1
                save(force=False)
        np.savez_compressed(
            directory / "relay_timing.npz", wall_us=times, predictions=timed_predictions
        )
        state["decoders"]["relay_bp"] = {
            "status": "timing_complete",
            **timing_summary(times),
            "warmup_shots": len(pilot_times) + plan["extra_warmup"],
        }
        save()
        # Only after the serial timing phase finishes do Rayon workers run.
        state["phase"] = "relay_quality_parallel"
        save()
        emit("phase", phase=state["phase"], quality_shots=plan["quality_shots"])
        count = plan["quality_shots"]
        predictions = np.zeros_like(truth[:count])
        iterations = np.zeros(count, dtype=np.int64)
        converged = np.zeros(count, dtype=bool)
        for start in range(0, count, plan["quality_chunk"]):
            stop = min(count, start + plan["quality_chunk"])
            syndrome = np.unpackbits(
                events[start:stop], axis=1, count=context.num_detectors, bitorder="little"
            )
            syndrome = np.ascontiguousarray(syndrome ^ decoder.detector_offset)
            syndrome.flags.writeable = False
            detailed = decoder.runner.decode_observables_detailed_batch(
                syndrome, parallel=threads > 1, progress_bar=False
            )
            predictions[start:stop] = np.packbits(
                np.asarray([r.observables for r in detailed], dtype=np.uint8)
                ^ decoder.observable_offset,
                axis=1,
                bitorder="little",
            )
            iterations[start:stop] = [r.iterations for r in detailed]
            converged[start:stop] = [r.converged for r in detailed]
            state["quality_completed"] = stop
            save(force=False)
        assert np.all(iterations <= ACCURACY_CONFIG.max_iterations)
        failures = np.any(predictions != truth[:count], axis=1)
        np.savez_compressed(
            directory / "relay_quality.npz",
            predictions=predictions,
            iterations=iterations,
            converged=converged,
            logical_failures=failures,
        )
        native = state["decoders"]["relay_bp"]
        native.update(
            status="complete",
            **quality_summary(predictions, truth[:count]),
            nonconverged=int((~converged).sum()),
            errors_nonconverged=int(np.count_nonzero(failures & ~converged)),
            iterations_mean=float(iterations.mean()),
            iterations_median=float(np.median(iterations)),
            iterations_p95=float(np.percentile(iterations, 95)),
            iterations_p99=float(np.percentile(iterations, 99)),
            iterations_max=int(iterations.max()),
            maximum_iterations=ACCURACY_CONFIG.max_iterations,
            total_legs=ACCURACY_CONFIG.total_legs,
            quality_threads=threads,
            paired_shots=count,
            matching_paired_errors=count_failures(reference[:count], truth[:count]),
        )
        state.update(
            status="complete", phase="complete", elapsed_seconds=time.monotonic() - started
        )
        save()
        emit("complete")
    except BaseException:
        state.update(status="error", error=traceback.format_exc())
        save()
        emit("error", error=state["error"])
        raise
    finally:
        channel.close()


def run_point(context, d, p, directory, options):
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=run_worker, args=(d, p, str(directory), options, sender))
    process.start()
    sender.close()
    started = time.monotonic()
    call_started = None
    phase = "initializing"
    reason = None
    event = None
    try:
        while True:
            if receiver.poll(0.1):
                try:
                    event = receiver.recv()
                except EOFError:
                    break
                if event["event"] == "call_start":
                    call_started = time.monotonic()
                elif event["event"] == "call_done":
                    call_started = None
                elif event["event"] == "phase":
                    phase = event["phase"]
                    print(
                        "PHASE",
                        d,
                        p,
                        phase,
                        event.get("plan", event.get("quality_shots", "")),
                        flush=True,
                    )
                elif event["event"] == "complete":
                    break
                elif event["event"] == "error":
                    raise RuntimeError(event["error"])
            now = time.monotonic()
            if now - started > options["point_seconds"]:
                reason = f"bounded point runtime exceeded {options['point_seconds']:g} seconds"
                break
            if call_started is not None and now - call_started > options["call_seconds"]:
                reason = f"single call exceeded {options['call_seconds']:g} seconds in {phase}"
                break
            if not process.is_alive() and not receiver.poll():
                break
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        receiver.close()
    state_path = directory / "state.json"
    if not state_path.exists():
        raise RuntimeError(
            f"worker exited before writing state: d={d},p={p},exit={process.exitcode}"
        )
    state = json.loads(state_path.read_text())
    if reason:
        state.update(
            status="runtime_cap",
            phase_at_cap=phase,
            reason=reason,
            elapsed_seconds=time.monotonic() - started,
        )
        native = state["decoders"]["relay_bp"]
        native["status"] = "runtime_cap"
        native["reason"] = reason
        native["partial_quality_is_not_an_estimate"] = True
        native.update(watchdog_call_metadata(call_started, now, options["call_seconds"]))
        write_json(state_path, state)
    elif state["status"] not in ("complete", "complete_with_resource_cap"):
        raise RuntimeError(f"unexpected worker exit: {state}")
    write_json(directory / "point.json", state)
    return state


def write_results(output, points):
    write_json(output / "results.json", points)
    rows = []
    for point in points:
        for name, measure in point["decoders"].items():
            row = {
                "distance": point["distance"],
                "rounds": point["rounds"],
                "noise": point["noise"],
                "decoder": name,
                "point_status": point["status"],
                **measure,
            }
            estimate = point.get("estimates", {}).get(name, {})
            row.update(
                {
                    key: estimate.get(key)
                    for key in (
                        "persistent_bytes",
                        "peak_working_bytes",
                        "build_operations",
                        "decode_operations_per_shot",
                    )
                }
            )
            rows.append(row)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("past_experiments/decoder_testing/results/full-xyz-distance-noise-20261121-bounded"),
    )
    parser.add_argument("--point-seconds", type=float, default=300)
    parser.add_argument("--call-seconds", type=float, default=60)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="quality workers; serial avoids upstream per-shot graph clones",
    )
    parser.add_argument("--shots", type=int, default=32768)
    parser.add_argument("--probes", type=int, default=1024)
    parser.add_argument("--distances", nargs="+", type=int, default=DISTANCES)
    parser.add_argument("--noise", nargs="+", type=float, default=NOISE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        not math.isfinite(args.point_seconds)
        or not math.isfinite(args.call_seconds)
        or args.point_seconds <= 0
        or args.call_seconds <= 0
        or args.threads < 1
        or args.shots < 8
        or args.probes < 8
        or any(d < 3 or d > 15 for d in args.distances)
    ):
        parser.error("invalid bounded-sweep limits")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    options = {
        "point_seconds": args.point_seconds,
        "call_seconds": args.call_seconds,
        "threads": args.threads,
        "max_shots": args.shots,
        "max_probes": args.probes,
        "seed": 1349001,
        "parallel_memory_mib": 1024,
    }
    identity = {
        "distances": args.distances,
        "noise": args.noise,
        "rounds": "distance",
        "basis": "z",
        "geometry": "rotated",
        "noise_model": "circuit",
        "relay_config": asdict(ACCURACY_CONFIG),
        "resource_limits": asdict(ResourceLimits()),
        "bounded_options": options,
        "median_budget_us": 1000,
        "sampling": (
            "independent 1-4 shot runtime pilot fixes counts; "
            "incomplete quality is never summarized"
        ),
        "latency": "serial normal packed calls; quality workers run only after latency",
        "tail_resolution": "p95 >=200 probes, p99 >=1000; fewer samples marked underresolved",
        "source": implementation_identity(
            ("surface_decoders.ibm_relay_bp:build_accuracy_decoder",)
        ),
        "environment": environment_identity(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resource_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("latency_capped_sweep.py").read_bytes()
        ).hexdigest(),
    }
    identity = json.loads(json.dumps(identity))
    manifest = output / "manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError(
                "resume rejected: settings, source, native dependency or environment changed"
            )
    else:
        write_json(manifest, {"created_utc": datetime.now(UTC).isoformat(), "identity": identity})
        shutil.copy2(__file__, output / "runner_snapshot.py")
    points = []
    context = mp.get_context("spawn")
    for d in args.distances:
        for p in args.noise:
            if (
                implementation_identity(("surface_decoders.ibm_relay_bp:build_accuracy_decoder",))
                != identity["source"]
            ):
                raise ValueError("decoder source/native dependency changed during sweep")
            directory = output / f"d{d}" / f"p{p:g}"
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "point.json").exists():
                point = json.loads((directory / "point.json").read_text())
            else:
                print("POINT", d, p, flush=True)
                point = run_point(context, d, p, directory, options)
            points.append(point)
            write_results(output, points)
            print(
                "RESULT",
                d,
                p,
                point["status"],
                {
                    name: {k: r.get(k) for k in ("errors", "shots", "median_us", "latency_samples")}
                    for name, r in point["decoders"].items()
                },
                flush=True,
            )
    print("DONE", output, flush=True)


if __name__ == "__main__":
    main()

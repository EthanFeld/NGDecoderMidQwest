"""Verify physical FPGA captures and score the paired PyMatching telescope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from base64 import b64decode
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "decoders/pymatching_telescope/src"))

import numpy as np
import pymatching
import stim
from pymatching_tensor_decoder import (
    TensorDecoder,
    build_decoder,
    qecbench_implementation_identity,
)
from pymatching_tensor_decoder.telescope.host import (
    COMMAND,
    REPLY,
    REQUEST,
    Inbox,
    frame,
    parse_reply,
    unpack,
)
from pymatching_tensor_decoder.telescope.surface_model import seal
from pymatching_tensor_decoder.telescope.telescope_circuit import build_context


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(staging, path)


def quality(prediction, truth):
    n = len(truth)
    errors = int(np.count_nonzero(prediction != truth))
    if not n:
        return {"shots": 0, "errors": 0, "ler": None, "ci95": None}
    rate, z = errors / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    delta = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return {"shots": n, "errors": errors, "ler": rate,
            "ci95": [max(0., center - delta), min(1., center + delta)]}


def latency(values, weights=None):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    if weights is None:
        percentiles = np.percentile(values, [50, 99, 99.9])
        mean = np.mean(values)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        order = np.argsort(values, kind="stable")
        cdf = np.cumsum(weights[order]) / weights.sum()
        percentiles = [values[order[min(np.searchsorted(cdf, q), len(values) - 1)]]
                       for q in [.5, .99, .999]]
        mean = np.average(values, weights=weights)
    return {"samples": len(values), "mean": float(mean), "p50": float(percentiles[0]),
            "p99": float(percentiles[1]), "p999": float(percentiles[2])}


def resolve_profile(config, distance, override):
    candidates = []
    if override:
        candidates.append(override)
    recorded = config.get("profile")
    if recorded:
        candidates += [ROOT / recorded, ROOT / "past_experiments/surface_fpga/profiles" / (recorded + ".json")]
    if distance == 15:
        candidates += [ROOT / "decoders/telescope/profiles/d15z-gap3-81mhz.json",
                       ROOT / "past_experiments/surface_fpga/experiments/d15-weighted-growth/d15z-growth-cap4000.json"]
    else:
        candidates += [ROOT / "past_experiments/surface_fpga/profiles/d7z-dual-p007-cut0-81mhz.json"]
    expected_hash = config.get("profile_sha256", config.get("graph_sha256"))
    for path in candidates:
        if path.is_file():
            image = load_json(path)
            if (image["distance"] == distance and seal(image) == config["image_tag"]
                    and (expected_hash is None or sha256(path) == expected_hash)):
                return path, image
    raise ValueError("No profile matches the fixture image tag and hash; supply --profile")


def verify_capture(run, columns, cohort, config, image, events, mapping, truth):
    n = len(run["rows"])
    if run["mode"] != "surface" or run.get("fault") or not run.get("complete") or n != run["planned"]:
        raise ValueError("The selected board cohort is incomplete or has a link fault")
    if (n > len(cohort["shots"]) or n > len(events) or n == 0
            or run["clock_hz"] != config["clock_hz"] or run["baud_rate"] != config["baud_rate"]
            or run.get("image_tag", config["image_tag"]) != config["image_tag"]):
        raise ValueError("Capture count, image, clock or baud differs from its fixture")
    memory = np.unpackbits(events, axis=1, count=events.shape[1] * 8, bitorder="little")[:, mapping]
    accepted = np.empty(n, dtype=bool)
    logical, cycles = np.zeros(n, np.uint8), np.empty(n, np.int32)
    residual = np.zeros((n, len(mapping)), dtype=bool)
    residual_available = np.zeros(n, dtype=bool)
    uart, delivery = np.empty(n), np.empty(n)
    verified_wire, retries, status_mismatch, cycle_mismatch = 0, 0, 0, 0
    first_token = None
    for i, (row, vector) in enumerate(zip(run["rows"], cohort["shots"][:n], strict=True)):
        get = dict(zip(columns, row, strict=True)).__getitem__
        request_wire = b64decode(vector[0], validate=True)
        bodies = list(Inbox().feed(request_wire))
        if len(bodies) != 1:
            raise ValueError(f"Request CRC/frame mismatch at row {i}")
        request = bodies[0]
        opcode, token, tag = REQUEST.unpack_from(request)
        if first_token is None:
            first_token = token
        if ((opcode, tag, token) != (COMMAND["tilt"], config["image_tag"], (first_token + i) & 0xffff)
                or not np.array_equal(unpack(request[REQUEST.size:], config["width"], config["rows"]), memory[i])
                or truth[i] != vector[1] or get("truth") != truth[i]
                or [get(k) for k in ("cpu_prediction", "cpu_us", "reference_status", "reference_cycles")] != vector[2:6]
                or get("request_wire_bytes") != len(request_wire)):
            raise ValueError(f"Capture and original joint input differ at row {i}")
        status, bit, count = (get(k) for k in ("fpga_status", "fpga_prediction", "core_cycles"))
        if status not in (1, 3) or bit not in ((0, 1) if status == 1 else (None,)) or not 0 < count <= config["max_cycles"]:
            raise ValueError(f"Invalid hardware status, logical bit or cycle count at row {i}")
        accepted[i], logical[i], cycles[i] = status == 1, bit or 0, count
        frame_bit = get("fpga_frame_bit") if "fpga_frame_bit" in columns else logical[i]
        if frame_bit not in (0, 1) or status == 1 and frame_bit != bit:
            raise ValueError(f"Invalid FPGA frame bit at row {i}")
        if "fpga_residual_hex" in columns:
            raw = get("fpga_residual_hex")
            if (raw is None) != accepted[i]:
                raise ValueError(f"Inconsistent residual availability at row {i}")
            if raw is not None:
                residual[i] = unpack(bytes.fromhex(raw), config["width"], config["rows"])
                if not residual[i].any() or np.any(residual[i] & ~memory[i]):
                    raise ValueError(f"Invalid hardware residual at row {i}")
                residual_available[i] = True
            logical[i] = frame_bit
        if "fpga_wire_hex" in columns and get("fpga_wire_hex") is not None:
            reply_wire = bytes.fromhex(get("fpga_wire_hex"))
            replies = list(Inbox().feed(reply_wire))
            if len(replies) != 1:
                raise ValueError(f"Reply CRC/frame mismatch at row {i}")
            body = replies[0]
            reply_token, reply = parse_reply(body, tag, config["width"], config["rows"], config["max_cycles"])
            raw = None if reply.residual is None else reply.residual.hex()
            if (reply_wire != frame(body) or reply_token != token or REPLY.unpack_from(body)[3] != status
                    or reply.cycles != count or int(reply.logical) != frame_bit
                    or len(reply_wire) != get("reply_wire_bytes")
                    or "fpga_residual_hex" in columns and raw != get("fpga_residual_hex")):
                raise ValueError(f"Decoded wire reply differs from its capture at row {i}")
            verified_wire += 1
        uart[i] = get("uart_round_trip_us")
        delivery_value = get("delivery_us") if "delivery_us" in columns else None
        delivery[i] = uart[i] if delivery_value is None else delivery_value
        retries += get("retry_count") if "retry_count" in columns else 0
        status_mismatch += vector[4] is not None and status != vector[4]
        cycle_mismatch += vector[5] is not None and count != vector[5]
    if not np.isfinite(uart).all() or np.any(uart <= 0) or np.any(delivery < uart):
        raise ValueError("Invalid captured UART times")
    return {
        "accepted": accepted, "logical": logical, "cycles": cycles, "residual": residual,
        "residual_available": residual_available, "uart": uart, "delivery": delivery,
        "verification": {"request_payloads_paired": n, "truth_labels_paired": n,
                         "reply_frames_crc_verified": verified_wire, "uart_retries": int(retries),
                         "hardware_status_model_mismatches": int(status_mismatch),
                         "hardware_cycle_model_mismatches": int(cycle_mismatch)},
    }


def batched_decode(decoder, events, batch_size):
    output = np.empty(len(events), dtype=np.uint8)
    start = perf_counter_ns()
    for offset in range(0, len(events), batch_size):
        rows = events[offset:offset + batch_size]
        returned = decoder(rows)
        if returned.shape != (len(rows), 1) or not np.isin(returned, [0, 1]).all():
            raise ValueError("Decoder returned invalid logical predictions")
        output[offset:offset + len(rows)] = returned[:, 0]
    return output, (perf_counter_ns() - start) / 1e9


def timing_indices(accepted, count):
    count = min(count, len(accepted))
    if not count:
        return np.array([], dtype=np.intp), np.array([], dtype=float)
    if count == 1:
        return np.array([0], dtype=np.intp), np.array([1.])
    pools = [np.flatnonzero(accepted), np.flatnonzero(~accepted)]
    if not len(pools[0]) or not len(pools[1]):
        indices = np.linspace(0, len(accepted) - 1, count, dtype=np.intp)
        return indices, np.full(len(indices), 1 / len(indices))
    minimum = min(32, max(1, count // 2))
    defer_count = min(len(pools[1]), max(minimum, round(count * len(pools[1]) / len(accepted))))
    accept_count = min(len(pools[0]), max(minimum, count - defer_count))
    defer_count = min(len(pools[1]), max(1, count - accept_count))
    indices, weights = [], []
    for pool, size in zip(pools, [accept_count, defer_count], strict=True):
        indices.extend(pool[np.linspace(0, len(pool) - 1, size, dtype=np.intp)])
        weights.extend([len(pool) / len(accepted) / size] * size)
    order = np.argsort(indices)
    return np.asarray(indices, dtype=np.intp)[order], np.asarray(weights)[order]


def time_decoders(decoders, predictions, events, front, mapping, num_detectors, count, clock_hz):
    indices, weights = timing_indices(front["accepted"], count)
    names = list(decoders) + ["telescope"]
    timings = {name: [] for name in names}
    for index in indices[:4]:
        for decoder in decoders.values():
            decoder(events[index:index + 1])
    for order, index in enumerate(indices):
        rows = events[index:index + 1]
        for step in range(len(names)):
            name = names[(order + step) % len(names)]
            start = perf_counter_ns()
            if name == "telescope":
                memory = np.unpackbits(rows, axis=1, count=num_detectors, bitorder="little")[:, mapping]
                if memory.shape[1] != len(mapping):
                    raise ValueError("Invalid memory projection")
                bit = (int(front["logical"][index]) if front["accepted"][index]
                       else int(decoders["tensor"](rows)[0, 0]))
            else:
                bit = int(decoders[name](rows)[0, 0])
            micros = (perf_counter_ns() - start) / 1000
            if bit != predictions[name][index]:
                raise ValueError(f"Single-shot {name} differs from its full-cohort prediction")
            timings[name].append(micros)
    if not len(indices):
        return None
    core = front["cycles"][indices] * 1e6 / clock_hz
    host = np.asarray(timings["telescope"])
    return {
        "sampling": "accept/defer strata, uniformly spaced within each stratum, weighted to full cohort",
        "quantiles": "weighted empirical CDF; no interpolation", "construction_and_warmup_excluded": True,
        "single_shot_calls": len(indices), "indices": indices.tolist(), "weights": weights.tolist(),
        "host_us": {name: latency(values, weights) for name, values in timings.items()},
        "telescope_without_uart_us": latency(core + host, weights),
        "telescope_with_uart_us": latency(front["delivery"][indices] + host, weights),
        "samples_host_us": timings,
        "samples_fpga_core_us": core.tolist(),
        "samples_uart_delivery_us": front["delivery"][indices].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--joint", required=True, type=Path)
    parser.add_argument("--distance", required=True, choices=[7, 15], type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--time-shots", default=256, type=int, help="single-shot timing samples per completed cohort; 0 disables")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--tensor-workers", default=4, type=int)
    parser.add_argument("--reuse-accuracy", action="store_true", help="reuse verified paired prediction arrays from this output for a separate timing phase")
    parser.add_argument("--first-run-index", default=0, type=int, help="ignore older runs in a browser capture containing appended replay sessions")
    parser.add_argument("--residual-probe-shots", default=0, type=int, help="at p=.005, compare original versus residual fallback on this prefix")
    args = parser.parse_args()
    if min(args.batch_size, args.tensor_workers) < 1 or min(args.time_shots, args.residual_probe_shots, args.first_run_index) < 0:
        parser.error("Invalid worker, batch or sample count")
    fixture_path = args.fixtures or args.joint.with_name(f"d{args.distance}_fixtures.json")
    previous = load_json(args.output) if args.reuse_accuracy and args.output.is_file() else None
    if args.reuse_accuracy and previous is None:
        raise ValueError("--reuse-accuracy requires an existing score output")
    fixture = load_json(fixture_path)
    config = fixture["surface"]
    if (config["distance"], config["basis"], config["rows"]) != (args.distance, "z", args.distance + 1):
        raise ValueError("Fixture does not match the requested distance/basis")
    profile_path, image = resolve_profile(config, args.distance, args.profile)
    capture = load_json(args.capture)
    if capture.get("format") != "fpga-uart-benchmark-v1" or len(set(capture["columns"])) != len(capture["columns"]):
        raise ValueError("Invalid browser capture format or duplicate columns")
    report = {"format": "deck-board-measurements-v2", "created_at": datetime.now(UTC).isoformat(),
              "source": "physical Tang Nano 20K replies; paired CPU replay uses original full joint syndromes",
              "distance": args.distance, "basis": "z", "rounds": args.distance,
              "capture": {"path": str(args.capture), "sha256": sha256(args.capture), "started_at": capture.get("started_at")},
              "fixtures": {"path": str(fixture_path), "sha256": sha256(fixture_path)},
              "joint": {"path": str(args.joint), "sha256": sha256(args.joint)},
              "profile": {"path": str(profile_path.relative_to(ROOT)), "sha256": sha256(profile_path), "image_tag": config["image_tag"]},
              "clock_hz": config["clock_hz"], "baud_rate": config["baud_rate"],
              "runner_sha256": sha256(Path(__file__)),
              "settings": {"device": "cpu", "batch_size": args.batch_size, "tensor_workers": args.tensor_workers,
                           "d15_path_bucket": args.distance == 15, "requested_time_shots": args.time_shots},
              "implementation": qecbench_implementation_identity(),
              "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                              "stim": stim.__version__, "pymatching": pymatching.__version__},
              "cohorts": []}
    if previous:
        for key in ["capture", "fixtures", "joint", "profile"]:
            if previous[key]["sha256"] != report[key]["sha256"]:
                raise ValueError(f"Saved accuracy provenance differs for {key}")
        if previous["implementation"] != report["implementation"]:
            raise ValueError("Saved accuracy decoder implementation differs")
        snapshot = args.output.with_name(args.output.stem + ".accuracy.json")
        if not snapshot.exists():
            snapshot.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
        report["accuracy_reused_from"] = {"path": str(snapshot), "sha256": sha256(snapshot)}
    fixture_cohorts = {(item["p"], item["seed"]): (i, item) for i, item in enumerate(config["cohorts"])}
    completed = [(i, run) for i, run in enumerate(capture["runs"])
                 if i >= args.first_run_index and run.get("complete") and not run.get("fault") and len(run["rows"]) > 0]
    selected = completed[-len(config["cohorts"]):]
    if len({(run["p"], run["seed"]) for _, run in selected}) != len(selected):
        raise ValueError("Latest completed runs repeat a cohort; use --first-run-index to select the new session")
    report["run_selection"] = {"first_run_index": args.first_run_index,
                               "rule": "latest completed, nonempty, fault-free runs, one per configured cohort",
                               "selected_capture_run_indices": [i for i, _ in selected],
                               "input_samples": "retained joint samples paired with newly captured FPGA telemetry"}
    with np.load(args.joint, allow_pickle=False) as joint:
        for run_index, run in selected:
            index, cohort = fixture_cohorts[(run["p"], run["seed"])]
            n = len(run["rows"])
            context, mapping = build_context(image, run["p"])
            if not np.array_equal(joint["memory_indices"], mapping):
                raise ValueError("Joint input map differs from the compiled FPGA model")
            for key, value in [("rates", run["p"]), ("seeds", run["seed"])]:
                if key in joint and joint[key][index] != value:
                    raise ValueError("Joint cohort metadata differs from the capture")
            events = np.ascontiguousarray(joint[f"events_{index}"][:n], dtype=np.uint8)
            truth = np.asarray(joint[f"truth_{index}"][:n], dtype=np.uint8).reshape(-1)
            if events.shape != (n, (context.num_detectors + 7) // 8) or len(truth) != n:
                raise ValueError("Invalid joint input or truth shape")
            events.flags.writeable = False
            front = verify_capture(run, capture["columns"], cohort, config, image, events, mapping, truth)
            print(f"d{args.distance} p={run['p']:g}: verified {n:,} board replies", flush=True)
            tensor = (TensorDecoder(context, solve_workers=args.tensor_workers, path_bucket=True)
                      if args.distance == 15 else build_decoder(context))
            ordinary = pymatching.Matching.from_detector_error_model(context.dem)
            correlated = pymatching.Matching.from_detector_error_model(context.dem, enable_correlations=True)
            ordinary.ensure_num_fault_ids(1)
            correlated.ensure_num_fault_ids(1)
            decoders = {
                "tensor": tensor.decode_batch,
                "pymatching": lambda rows: ordinary.decode_batch(rows, bit_packed_shots=True, bit_packed_predictions=True),
                "correlated_pymatching": lambda rows: correlated.decode_batch(rows, bit_packed_shots=True, bit_packed_predictions=True, enable_correlations=True),
            }
            try:
                predictions, batch_seconds = {}, {}
                if previous:
                    saved_record = next((item for item in previous["cohorts"]
                                         if item["p"] == run["p"] and item["seed"] == run["seed"]), None)
                    if not saved_record or saved_record["capture_run_index"] != run_index or saved_record["shots"] != n:
                        raise ValueError("Saved accuracy does not describe this capture cohort")
                    saved_path = Path(saved_record["paired_arrays"]["path"])
                    if sha256(saved_path) != saved_record["paired_arrays"]["sha256"]:
                        raise ValueError("Saved paired accuracy arrays have changed")
                    with np.load(saved_path, allow_pickle=False) as saved:
                        for key, expected in [("events", events), ("truth", truth), ("accepted", front["accepted"]),
                                              ("fpga_logical", front["logical"]), ("cycles", front["cycles"])]:
                            if not np.array_equal(saved[key], expected):
                                raise ValueError(f"Saved accuracy inputs differ for {key}")
                        predictions = {name: saved["prediction_" + name].copy()
                                       for name in [*decoders, "telescope"]}
                    batch_seconds = saved_record["batch_seconds"]
                else:
                    for name, decoder in decoders.items():
                        predictions[name], batch_seconds[name] = batched_decode(decoder, events, args.batch_size)
                    predictions["telescope"] = np.where(front["accepted"], front["logical"], predictions["tensor"]).astype(np.uint8)
                record = {"p": run["p"], "seed": run["seed"], "shots": n, "complete": True,
                          "capture_run_index": run_index, "capture_run_started_at": run.get("started_at"),
                          "accepted": int(front["accepted"].sum()), "deferred": int((~front["accepted"]).sum()),
                          "defer_rate": float((~front["accepted"]).mean()), "verification": front["verification"],
                          "quality": {name: quality(prediction, truth) for name, prediction in predictions.items()},
                          "fpga_native_accepts": quality(front["logical"][front["accepted"]], truth[front["accepted"]]),
                          "fpga_forced_zero_on_defer": quality(np.where(front["accepted"], front["logical"], 0), truth),
                          "fpga_core_us": latency(front["cycles"] * 1e6 / config["clock_hz"]),
                          "uart_round_trip_us": latency(front["uart"]), "uart_delivery_us": latency(front["delivery"]),
                          "batch_seconds": batch_seconds, "batch_size": args.batch_size,
                          "tensor_type": type(tensor).__module__ + "." + type(tensor).__name__}
                record["timing"] = time_decoders(decoders, predictions, events, front, mapping, context.num_detectors,
                                                 args.time_shots, config["clock_hz"])
                if args.time_shots and args.distance == 15 and run["p"] == .005:
                    pending = np.flatnonzero(~front["accepted"])
                    bit_rows = np.unpackbits(events[pending], axis=1, count=context.num_detectors, bitorder="little")
                    matching_times = []
                    for row in bit_rows:
                        started = perf_counter_ns()
                        tensor.control.path.matching.decode_to_edges_array(row, enable_correlations=True)
                        matching_times.append((perf_counter_ns() - started) / 1000)
                    record["initial_correlated_matching_on_board_defers_us"] = {
                        "timing": latency(matching_times), "samples_us": matching_times,
                        "scope": "one completed decode_to_edges_array call per actual FPGA defer; unpacking, construction and UART excluded",
                        "shots": len(pending),
                    }
                if args.residual_probe_shots and run["p"] == .005:
                    limit = min(n, args.residual_probe_shots)
                    pending = np.flatnonzero(~front["accepted"][:limit])
                    if not front["residual_available"][pending].all():
                        raise ValueError("Residual comparison requires captured residuals for every selected defer")
                    bits = np.unpackbits(events[pending], axis=1, count=context.num_detectors, bitorder="little")
                    bits[:, mapping] = front["residual"][pending]
                    mixed = np.packbits(bits, axis=1, bitorder="little")
                    mixed.flags.writeable = False
                    residual_labels, _ = batched_decode(tensor.decode_batch, mixed, args.batch_size)
                    hybrid = predictions["telescope"][:limit].copy()
                    hybrid[pending] = residual_labels ^ front["logical"][pending]
                    record["residual_probe"] = {
                        "shots": limit, "actual_board_defers": len(pending),
                        "original_tensor_only": quality(predictions["tensor"][:limit], truth[:limit]),
                        "original_syndrome_telescope": quality(predictions["telescope"][:limit], truth[:limit]),
                        "residual_syndrome_telescope": quality(hybrid, truth[:limit]),
                        "method": "replace memory-basis detectors with captured hardware residual; retain conjugate checks; XOR captured frame on defers",
                    }
            finally:
                close = getattr(tensor, "close", None)
                if close:
                    close()
            raw_path = args.output.with_name(args.output.stem + f"-p{round(run['p'] * 1e6):05d}.npz")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw_path, events=events, truth=truth, memory_indices=mapping,
                                accepted=front["accepted"], fpga_logical=front["logical"], cycles=front["cycles"],
                                residual_packed=np.packbits(front["residual"], axis=1, bitorder="little"),
                                residual_available=front["residual_available"],
                                **{"prediction_" + key: value for key, value in predictions.items()})
            record["paired_arrays"] = {"path": str(raw_path), "sha256": sha256(raw_path)}
            report["cohorts"].append(record)
            report["complete"] = len(report["cohorts"]) == len(config["cohorts"])
            atomic_json(args.output, report)
            print(f"d{args.distance} p={run['p']:g}: telescope {record['quality']['telescope']['errors']}/{n:,}; "
                  f"defers {record['deferred']:,}", flush=True)
    if not report["cohorts"]:
        raise ValueError("Capture has no completed cohorts to score")
    if "owned_backend" in sys.modules:
        raise RuntimeError("The independent matching engine must not be loaded in this replay")


if __name__ == "__main__":
    main()

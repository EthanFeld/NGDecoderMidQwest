"""Verify a fresh Chrome FPGA capture and score Telescope on its joint samples."""

import argparse
import hashlib
import json
import os
import sys
from base64 import b64decode
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "past_experiments/surface_fpga"), str(ROOT / "decoders/telescope"), str(ROOT / "decoders/telescope/tensor_decoder/src"), str(ROOT / "decoders/pymatching_telescope/src"), str(ROOT / "past_experiments/decoder_testing/src")]

from host import REPLY, REQUEST, Inbox, frame, parse_reply
from telescope_circuit import build_context

from past_experiments.comparisons.board_replay import check_run, hash_file, load_fixtures, load_json
from past_experiments.comparisons.sweep import get_profile, project, wilson
from tensor_decoder import TensorDecoder, build_decoder

RATES = (.001, .003, .005, .007)
SHOTS = 10_000


def identify(path):
    return {"file": path.name, "sha256": hash_file(path)}


def as_sha256(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def atomic_write(path, value):
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(staging, path)


def verify_wire(rows, shots, columns, config):
    if "fpga_wire_hex" not in columns:
        return 0
    for index, (row, shot) in enumerate(zip(rows, shots, strict=True)):
        reply_wire = bytes.fromhex(row[columns["fpga_wire_hex"]])
        body, = Inbox().feed(reply_wire)
        request, = Inbox().feed(b64decode(shot[0], validate=True))
        _, expected_token, tag = REQUEST.unpack_from(request)
        token, reply = parse_reply(body, tag, config["width"], config["rows"], config["max_cycles"])
        residual = None if reply.residual is None else reply.residual.hex()
        if (
            reply_wire != frame(body)
            or len(reply_wire) != row[columns["reply_wire_bytes"]]
            or token != expected_token
            or REPLY.unpack_from(body)[3] != row[columns["fpga_status"]]
            or reply.cycles != row[columns["core_cycles"]]
            or int(reply.logical) != row[columns["fpga_frame_bit"]]
            or residual != row[columns["fpga_residual_hex"]]
        ):
            raise ValueError(f"Raw FPGA response disagrees with capture at shot {index + 1}")
    return len(rows)


def paired_inputs(joint, index, p, seed, profile, memory, truth):
    context, mapping = build_context(profile.image, p)
    if not np.array_equal(joint["memory_indices"], mapping):
        raise ValueError("Retained detector map differs from the Telescope circuit")
    for key, expected in (("rates", p), ("seeds", seed)):
        if key in joint and joint[key][index] != expected:
            raise ValueError(f"Retained joint metadata differs for {key}")
    n = len(truth)
    events = np.ascontiguousarray(joint[f"events_{index}"][:n])
    labels = joint[f"truth_{index}"][:n].reshape(n)
    if events.dtype != np.uint8 or events.shape != (n, (context.num_detectors + 7) // 8):
        raise ValueError("Retained joint syndromes do not use packed Stim bit order")
    np.testing.assert_array_equal(project(events, context, mapping), memory)
    np.testing.assert_array_equal(labels, truth)
    regenerated, observables = context.circuit.compile_detector_sampler(seed=seed).sample(
        SHOTS, bit_packed=True, separate_observables=True,
    )
    np.testing.assert_array_equal(regenerated[:n], events)
    np.testing.assert_array_equal(observables[:n, 0], labels)
    events.flags.writeable = False
    return context, events


def score(context, events, truth, front, batch_size):
    accepted = front.accepted
    deferred = np.flatnonzero(~accepted)
    prediction = front.logical.copy()
    active_seconds = 0.0
    if len(deferred):
        tensor = (
            TensorDecoder(context, solve_workers=4, path_bucket=True)
            if context.experiment.distance == 15 else build_decoder(context)
        )
        try:
            for start in range(0, len(deferred), batch_size):
                indices = deferred[start:start + batch_size]
                tail = events[indices]
                tail.flags.writeable = False
                before = perf_counter()
                returned = tensor.decode_batch(tail)
                active_seconds += perf_counter() - before
                if returned.shape != (len(indices), 1) or not np.isin(returned, [0, 1]).all():
                    raise ValueError("Tensor fallback returned invalid logical labels")
                prediction[indices] = returned[:, 0]
        finally:
            close = getattr(tensor, "close", None)
            if close is not None:
                close()
    wrong = prediction != truth
    errors = int(wrong.sum())
    low, high = wilson(errors, len(truth))
    return {
        "logical_errors": errors,
        "logical_error_rate": errors / len(truth),
        "ci95_low": low,
        "ci95_high": high,
        "accepted_logical_errors": int((wrong & accepted).sum()),
        "deferred_logical_errors": int(wrong[deferred].sum()),
        "tensor_fallback_shots": len(deferred),
        "tensor_fallback_batch_seconds": active_seconds,
        "predictions_bit_packed_little_hex": np.packbits(prediction, bitorder="little").tobytes().hex(),
        "prediction_sha256": hashlib.sha256(prediction.astype(np.uint8).tobytes()).hexdigest(),
    }


def validate_fixture(config, distance, profile):
    if (config["distance"], config["basis"], config["rows"]) != (distance, "z", distance + 1):
        raise ValueError("The fixture does not match the distance or memory basis")
    if tuple(cohort["p"] for cohort in config["cohorts"]) != RATES:
        raise ValueError("The fixture does not contain the four requested rates in order")
    if any(len(cohort["shots"]) != SHOTS for cohort in config["cohorts"]):
        raise ValueError("The fixture does not contain 10,000 shots per rate")
    expected_baud_rate = {7: 1_000_000, 15: 500_000}[distance]
    if (
        config["clock_hz"] != profile["clock_mhz"] * 1_000_000
        or config["baud_rate"] != expected_baud_rate
        or config["image_tag"] != int(profile["image_tag"], 16)
    ):
        raise ValueError("Fixture clock, baud, or image tag differs from the hardware profile")
    graph_hash = config.get("profile_sha256", config.get("graph_sha256"))
    if graph_hash != profile["profile_sha256"]:
        raise ValueError("Fixture fingerprint differs from the FPGA graph")
    image_path = ROOT / config.get("bitstream", profile["bitstream"])
    if hash_file(image_path) != config["bitstream_sha256"]:
        raise ValueError("Fixture fingerprint differs from the FPGA image")


def run_capture(args, fixture, profile, joint, report):
    capture = load_json(args.capture)
    if capture["format"] != "fpga-uart-benchmark-v1" or not capture["serial"].startswith("Web Serial 8N1"):
        raise ValueError("Capture does not identify the Chrome Web Serial benchmark")
    if len(set(capture["columns"])) != len(capture["columns"]):
        raise ValueError("Capture contains duplicate column names")
    columns = {name: index for index, name in enumerate(capture["columns"])}
    source = identify(args.capture)
    config = fixture["surface"]
    cohorts = {(item["p"], item["seed"]): (i, item) for i, item in enumerate(config["cohorts"])}
    if len({(run["p"], run["seed"]) for run in capture["runs"]}) != len(capture["runs"]):
        raise ValueError("Capture repeats a cohort")

    for run in capture["runs"]:
        p, seed = run["p"], run["seed"]
        entry = cohorts.get((p, seed))
        if entry is None:
            raise ValueError(f"Capture contains an unexpected rate or seed at p={p:g}")
        if run.get("image_tag", config["image_tag"]) != config["image_tag"]:
            raise ValueError(f"Capture reports a different FPGA image tag at p={p:g}")
        index, cohort = entry
        rows = run["rows"]
        n = len(rows)
        if not 0 <= n <= run["planned"] <= SHOTS or (run["complete"] and n != run["planned"]):
            raise ValueError(f"Invalid planned or captured row count at p={p:g}")
        if n == 0:
            continue
        complete = bool(run["complete"] and run["fault"] is None and n == SHOTS)
        run_hash = as_sha256({**run, "columns": capture["columns"]})
        prior = next((item for item in report["cohorts"] if item["p"] == p), None)
        if prior and prior["run_sha256"] == run_hash:
            continue
        if prior and prior["complete"]:
            raise ValueError(f"Existing checkpoint already has a different completed capture for p={p:g}")
        if prior and prior["shots"] >= n:
            raise ValueError(f"New capture does not extend the existing checkpoint at p={p:g}")
        if prior and prior["rows_sha256"] != as_sha256(rows[:prior["shots"]]):
            raise ValueError(f"New capture replaces existing rows at p={p:g}")

        # The archive checker validates the recorded rows as a self-contained range.
        # Preserve the original planned, complete and fault fields for reporting.
        prefix = {**cohort, "shots": cohort["shots"][:n]}
        portion = {**run, "planned": n, "complete": True, "fault": None}
        front, memory, truth, _, stats = check_run(
            portion, capture["columns"], {**config, "cohorts": [prefix]}, profile.image, source["file"],
        )
        wire_rows = verify_wire(rows, prefix["shots"], columns, config)
        context, events = paired_inputs(joint, index, p, seed, profile, memory, truth)
        print(f"d{args.distance} p={p:g} verified {n:,} FPGA replies and {stats['defers']:,} defers", flush=True)
        telescope = score(context, events, truth, front, args.batch_size)
        assert telescope["tensor_fallback_shots"] == stats["defers"]
        record = {
            "p": p, "seed": seed, "shots": n, "complete": complete, "planned_shots": SHOTS,
            "capture_planned_shots": run["planned"], "capture_fault": run["fault"],
            "capture": source, "run_sha256": run_hash, "rows_sha256": as_sha256(rows),
            "capture_started_at": capture.get("started_at"), "capture_exported_at": capture.get("exported_at"),
            "accepted": stats["accepted"], "defers": stats["defers"], "coverage": stats["coverage"],
            "fpga_core_us": stats["fpga_core_us"], "uart_round_trip_us": stats["uart_round_trip_us"],
            "uart_delivery_us": stats["uart_delivery_us"], "uart_retries": stats["uart_retries"],
            "verified_joint_shots": n, "stim_seed_replayed_shots": n,
            "independently_crc_verified_reply_frames": wire_rows,
            "telescope": telescope,
        }
        for field, fixture_index in (("status", 4), ("cycle", 5)):
            name = f"hardware_{field}_mismatches_against_reference"
            record[name] = stats[name] if all(shot[fixture_index] is not None for shot in prefix["shots"]) else None
        report["cohorts"] = sorted(
            [item for item in report["cohorts"] if item["p"] != p] + [record], key=lambda item: item["p"],
        )
        report["completed"] = len(report["cohorts"]) == len(RATES) and all(item["complete"] for item in report["cohorts"])
        report["updated_at"] = datetime.now(UTC).isoformat()
        atomic_write(args.output, report)
        print(f"d{args.distance} p={p:g} Telescope errors {telescope['logical_errors']:,}/{n:,}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=(7, 15), required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("positive batch size required")

    profile = get_profile(args.distance)
    profile_identity = profile.identity()
    fixture = load_fixtures(args.fixtures)
    config = fixture["surface"]
    validate_fixture(config, args.distance, profile_identity)
    report = {
        "format": "telescope-fpga-sweep-v1",
        "source": "physical Tang Nano 20K FPGA, Chrome Web Serial",
        "decoder": "FPGA logical bits on accepts and batched CPU tensor on original joint-syndrome defers",
        "distance": args.distance, "basis": "z", "rounds": args.distance, "rates": list(RATES),
        "profile": profile_identity, "baud_rate": config["baud_rate"],
        "fixture_bitstream": {"file": config.get("bitstream", profile_identity["bitstream"]),
                              "sha256": config["bitstream_sha256"]},
        "fixture": identify(args.fixtures), "sampling_versions": fixture.get("versions"),
        "joint": identify(args.joint), "tensor_batch_size": args.batch_size,
    }
    if args.output.exists():
        existing = load_json(args.output)
        if any(existing[key] != value for key, value in report.items()):
            raise ValueError("Existing checkpoint benchmark provenance differs")
        report = existing
    else:
        report.update(cohorts=[], completed=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.joint, allow_pickle=False) as joint:
        run_capture(args, fixture, profile, joint, report)


if __name__ == "__main__":
    main()

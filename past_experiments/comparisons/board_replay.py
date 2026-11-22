"""Verify archived physical FPGA replies and replay only proven joint-syndrome pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from base64 import b64decode
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SURFACE = ROOT / "past_experiments/surface_fpga"
ARCHIVES = ROOT / "past_experiments/fpga_benchmark" / ".build"
sys.path[:0] = [str(ROOT), str(SURFACE), str(ROOT / "decoders/telescope"), str(ROOT / "decoders/telescope/tensor_decoder/src"), str(ROOT / "decoders/pymatching_telescope/src"), str(ROOT / "past_experiments/decoder_testing/src")]


@dataclass(frozen=True)
class Archive:
    name: str
    distance: int
    capture: str
    fixture: str


ARCHIVE_LIST = (
    Archive("d7_81mhz", 7, "surface_fast81_board.json", "surface_fast81_board.html"),
    Archive(
        "d15_27mhz", 15, "surface_d15_tilt_ram_growth_27_board.json",
        "surface_d15_tilt_ram_growth_27_board.html",
    ),
    Archive(
        "d15_27mhz_joint", 15, "surface_d15_tilt_ram_growth_27_fresh_p005_72621005_board.json",
        "surface_d15_tilt_ram_growth_27_fresh_p005_72621005_fixtures.json",
    ),
)


@dataclass(frozen=True)
class FrontReplay:
    accepted: np.ndarray
    logical: np.ndarray
    cycles: np.ndarray


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_fixtures(path):
    if path.suffix == ".json":
        return load_json(path)
    page = path.read_text(encoding="utf-8-sig")
    fields = re.findall(
        r'<script id="embedded-fixtures" type="application/json">(.*?)</script>',
        page, flags=re.DOTALL,
    )
    if len(fields) != 1:
        raise ValueError(f"{path.name} does not contain one embedded fixture")
    return json.loads(fields[0])


def profile_image(distance):
    if distance == 7:
        path = SURFACE / "profiles/d7z-dual-p007-cut0-81mhz.json"
    elif distance == 15:
        path = SURFACE / "experiments/d15-weighted-growth/d15z-growth-cap4000.json"
    else:
        raise ValueError(f"no archived board profile for d{distance}")
    return load_json(path)


def label_error(prediction, truth):
    if not len(truth):
        return {"shots": 0, "logical_errors": 0, "ler": None}
    errors = int(np.count_nonzero(prediction != truth))
    return {"shots": len(truth), "logical_errors": errors, "ler": errors / len(truth)}


def latency(micros):
    if not len(micros):
        return None
    p50, p99, p999 = np.percentile(micros, (50, 99, 99.9))
    return {"mean": float(np.mean(micros)), "median": float(p50),
            "p99": float(p99), "p999": float(p999), "shots": len(micros)}


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_run(run, columns, fixture, image, source):
    from host import COMMAND, REQUEST, Inbox, unpack
    from surface_model import seal

    identity = (run["p"], run["seed"])
    cohort, = [c for c in fixture["cohorts"] if (c["p"], c["seed"]) == identity]
    fields = ("clock_hz", "baud_rate")
    expected_tag = seal(image)
    if (
        run["mode"] != "surface" or run["fault"] or not run["complete"]
        or len(run["rows"]) != run["planned"] or len(run["rows"]) != len(cohort["shots"])
        or any(run[f] != fixture[f] for f in fields) or fixture["image_tag"] != expected_tag
        or fixture["width"] != image["width"] or fixture["rows"] != image["rounds"] + 1
    ):
        raise ValueError(f"incomplete or mismatched board and fixture at {source} {identity}")
    n = len(run["rows"])
    memory = np.empty((n, image["checks"]), dtype=bool)
    accepted, logical, truth, control = np.empty((4, n), dtype=bool)
    cycles = np.empty(n, dtype=np.int32)
    uart, delivery, control_us = np.empty((3, n))
    statuses, clocks, retries = 0, 0, 0
    first_token = None
    frame_bits = "fpga_frame_bit" in columns
    for index, (row, vector) in enumerate(zip(run["rows"], cohort["shots"], strict=True)):
        wire = b64decode(vector[0], validate=True)
        packet, = Inbox().feed(wire)
        opcode, token, tag = REQUEST.unpack_from(packet)
        if first_token is None:
            first_token = token
        get = dict(zip(columns, row, strict=True)).__getitem__
        if (
            (opcode, tag, token) != (COMMAND["tilt"], expected_tag, (first_token + index) & 0xFFFF)
            or [get(name) for name in ("truth", "cpu_prediction", "cpu_us")] != vector[1:4]
            or [get(name) for name in ("reference_status", "reference_cycles")] != vector[4:6]
            or get("request_wire_bytes") != len(wire)
        ):
            raise ValueError(f"fixture row {index} is inconsistent at {source} {identity}")
        memory[index] = unpack(packet[REQUEST.size:], fixture["width"], fixture["rows"])
        status, bit, count = get("fpga_status"), get("fpga_prediction"), get("core_cycles")
        if status not in (1, 3) or bit not in ((0, 1) if status == 1 else (None,)):
            raise ValueError(f"invalid status or result in row {index} at {source} {identity}")
        if not 0 < count <= fixture["max_cycles"]:
            raise ValueError(f"invalid cycle count in row {index} at {source} {identity}")
        if frame_bits:
            frame, raw = get("fpga_frame_bit"), get("fpga_residual_hex")
            if frame not in (0, 1) or frame != (bit if status == 1 else 0):
                raise ValueError(f"invalid wire frame bit in row {index} at {source} {identity}")
            if (raw is None) != (status == 1):
                raise ValueError(f"missing defer residual in row {index} at {source} {identity}")
            if raw is not None:
                residual = unpack(bytes.fromhex(raw), fixture["width"], fixture["rows"])
                if not residual.any() or np.any(residual & ~memory[index]):
                    raise ValueError(f"invalid defer residual in row {index} at {source} {identity}")
        accepted[index], logical[index], cycles[index] = status == 1, bool(bit), count
        truth[index], control[index], control_us[index] = vector[1:4]
        uart[index] = get("uart_round_trip_us")
        delivery_value = get("delivery_us") if "delivery_us" in columns else None
        delivery[index] = uart[index] if delivery_value is None else delivery_value
        statuses += status != vector[4]
        clocks += count != vector[5]
        retries += get("retry_count")
    if not np.isfinite(uart).all() or np.any(uart <= 0) or np.any(delivery < uart):
        raise ValueError(f"invalid UART timing at {source} {identity}")
    front = FrontReplay(accepted, logical, cycles)
    core = cycles * 1e6 / run["clock_hz"]
    historical_extra = np.where(accepted, 0, control_us)
    stats = {
        "p": run["p"], "seed": run["seed"], "shots": n, "clock_hz": run["clock_hz"],
        "baud_rate": run["baud_rate"], "image_tag": f"{fixture['image_tag']:08x}",
        "request_fixtures_verified": n, "accepted": int(accepted.sum()),
        "defers": int((~accepted).sum()), "coverage": float(accepted.mean()),
        "hardware_status_mismatches_against_reference": statuses,
        "hardware_cycle_mismatches_against_reference": clocks, "uart_retries": retries,
        "fpga_native_accepts": label_error(logical[accepted], truth[accepted]),
        "fpga_forced_wire_bit": label_error(logical, truth),
        "forced_bit_source": "captured frame bit" if frame_bits else "RTL drives zero on defer",
        "native_unresolved_or_wrong": int((~accepted | (logical != truth)).sum()),
        "fpga_core_us": latency(core), "uart_round_trip_us": latency(uart),
        "uart_delivery_us": latency(delivery),
        "historical_projected_pymatching": label_error(control, truth),
        "historical_projected_pymatching_telescope": {
            **label_error(np.where(accepted, logical, control), truth),
            "without_uart_us": latency(core + historical_extra),
            "with_uart_us": latency(delivery + historical_extra),
        },
    }
    return front, memory, truth, delivery, stats


def verify_joint(archive, run, image, memory, truth, sampling_batch):
    from telescope_circuit import build_context

    context, mapping = build_context(image, run["p"])
    shot_count = len(truth)
    source = context.circuit.compile_detector_sampler(seed=run["seed"])
    sampled = sampling_batch if sampling_batch is not None else shot_count
    if sampled < shot_count:
        raise ValueError("the fixture captured more rows than its original Stim batch")
    events, observables = source.sample(sampled, bit_packed=True, separate_observables=True)
    events, observables = events[:shot_count], observables[:shot_count]
    projected = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")
    same_row = np.all(projected[:, mapping] == memory, axis=1)
    same_truth = observables[:, 0] == truth
    proof = {
        "method": "full-joint Stim replay checked against every archived detector payload and truth bit",
        "stim_batch_size": sampled,
        "projection_matches": int(same_row.sum()), "truth_matches": int(same_truth.sum()),
        "all_paired": bool(np.all(same_row & same_truth)),
        "capture_sampling": "joint" if archive.name.endswith("_joint") else "projected Stim",
    }
    if archive.name.endswith("_joint"):
        joint_path = ARCHIVES / archive.fixture.replace("_fixtures.json", "_joint.npz")
        with np.load(joint_path, allow_pickle=False) as joint:
            saved = (
                np.array_equal(events, joint["events"])
                and np.array_equal(mapping, joint["memory_indices"])
                and np.array_equal(truth, joint["truth"])
                and (run["p"], run["seed"]) == (joint["p"].item(), joint["seed"].item())
            )
        if not (proof["all_paired"] and saved):
            raise ValueError(f"recorded joint archive failed literal pairing: {archive.name}")
        proof["retained_joint_file_sha256"] = hash_file(joint_path)
    if proof["all_paired"]:
        events.flags.writeable = False
    return context, mapping, events, proof


def archive_replay(archive, *, time_shots, batch_size, telescope_batch_time_shots, inspect_only):
    capture_path, fixture_path = ARCHIVES / archive.capture, ARCHIVES / archive.fixture
    capture, packed_fixtures = load_json(capture_path), load_fixtures(fixture_path)
    fixture = packed_fixtures["surface"]
    if fixture["distance"] != archive.distance or fixture["basis"] != "z":
        raise ValueError(f"unexpected channel or distance for {archive.name}")
    columns = {key: i for i, key in enumerate(capture["columns"])}
    image = profile_image(archive.distance)
    versions = packed_fixtures.get("versions", {})
    report = {
        "archive": archive.name, "distance": archive.distance, "basis": "z",
        "transport_capture_started_at": capture.get("started_at"),
        "capture": capture_path.relative_to(ROOT).as_posix(), "capture_sha256": hash_file(capture_path),
        "fixture": fixture_path.relative_to(ROOT).as_posix(), "fixture_sha256": hash_file(fixture_path),
        "fixture_sampling": fixture.get("sampling", "full joint Stim circuit"),
        "fixture_versions": versions,
        "host_replay_settings": {
            "requested_timing_shots": time_shots, "batch_size": batch_size,
            "telescope_batch_timing_prefix": telescope_batch_time_shots,
            "relay_bp": "measured in the separate model sweep",
        },
        "cohorts": [],
    }
    for run in capture["runs"]:
        front, memory, truth, delivery, stats = check_run(run, columns, fixture, image, archive.name)
        context, mapping, events, proof = verify_joint(
            archive, run, image, memory, truth, versions.get("batch_size"),
        )
        stats["joint_pairing"] = proof
        if not inspect_only and proof["all_paired"]:
            from past_experiments.comparisons.sweep import compare_point, get_profile

            stats["joint_comparison"] = compare_point(
                get_profile(archive.distance), context, mapping, events, truth, front,
                time_shots, batch_size, link_us=delivery, seed=run["seed"],
                source="board_capture", include_relay=False,
                telescope_batch_time_shots=telescope_batch_time_shots,
            )
        report["cohorts"].append(stats)
        print(
            f"{archive.name} p={run['p']:g} n={len(truth):,} "
            f"deferred={stats['defers']:,}, paired joint rows={proof['projection_matches']:,}, "
            f"joint truth matches={proof['truth_matches']:,}", flush=True,
        )
        yield report


def csv_row(archive, record):
    group = {
        "archive": archive["archive"], "distance": archive["distance"],
        **{key: record[key] for key in ("p", "seed", "shots", "accepted", "defers", "coverage")},
        "fpga_accepted_errors": record["fpga_native_accepts"]["logical_errors"],
        "fpga_accepted_ler": record["fpga_native_accepts"]["ler"],
        "fpga_forced_errors": record["fpga_forced_wire_bit"]["logical_errors"],
        "fpga_forced_ler": record["fpga_forced_wire_bit"]["ler"],
        "native_unresolved_or_wrong": record["native_unresolved_or_wrong"],
        "matches_joint": record["joint_pairing"]["all_paired"],
    }
    for phase in ("fpga_core_us", "uart_round_trip_us", "uart_delivery_us"):
        group.update({f"{phase}_{key}": record[phase][key] for key in ("mean", "median", "p99")})
    return group


def paired_csv_row(archive, record):
    metrics = ("fpga_core_us", "single_shot_us_excluding_uart", "single_shot_us_including_uart")
    flat = {"archive": archive["archive"], **{k: v for k, v in record.items() if k not in metrics}}
    for metric in metrics:
        flat.update({
            f"{metric}_{stat}": record[metric][stat] if record[metric] else None
            for stat in ("mean", "median", "p99", "p999")
        })
    return flat


def write_csv(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def checkpoint(directory, reports):
    rows = [csv_row(archive, record) for archive in reports for record in archive["cohorts"]]
    write_csv(directory / "physical_summary.csv", rows)
    pairs = [
        paired_csv_row(archive, comparison) for archive in reports for record in archive["cohorts"]
        for comparison in record.get("joint_comparison", [])
    ]
    if pairs:
        write_csv(directory / "paired_comparison.csv", pairs)
    report = directory / "physical_replays.json"
    temp = report.with_suffix(".json.tmp")
    temp.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    temp.replace(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="verify archives without running decoders")
    parser.add_argument("--time-shots", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--telescope-batch-time-shots", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "board")
    args = parser.parse_args()
    if min(args.time_shots, args.batch_size, args.telescope_batch_time_shots) < 1:
        parser.error("use positive timing and batch sizes")
    output_names = ("physical_replays.json", "physical_summary.csv", "paired_comparison.csv")
    if not args.inspect:
        if any((args.output / name).exists() for name in output_names):
            parser.error("use a fresh board replay directory")
        args.output.mkdir(exist_ok=True, parents=True)
    reports = []
    for archive in ARCHIVE_LIST:
        report = None
        for report in archive_replay(
            archive, time_shots=args.time_shots, batch_size=args.batch_size, inspect_only=args.inspect,
            telescope_batch_time_shots=args.telescope_batch_time_shots,
        ):
            if not args.inspect:
                checkpoint(args.output, [*reports, report])
        if report is None:
            raise ValueError(f"empty board capture for {archive.name}")
        reports.append(report)


if __name__ == "__main__":
    main()

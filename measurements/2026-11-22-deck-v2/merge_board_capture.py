"""Join validated FPGA capture segments without changing their request order."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from base64 import b64decode
from binascii import crc_hqx
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUEST = struct.Struct("<BHI")
REPLY = struct.Struct("<BHIBHBB")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identity(path):
    return {"path": path.resolve().relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def body(wire):
    if not wire or wire[0] != 0x7e or wire[-1] != 0x7e:
        raise ValueError("Missing wire-frame delimiters")
    data, escape = bytearray(), False
    for value in wire[1:-1]:
        if escape:
            data.append(value ^ 0x20)
            escape = False
        elif value == 0x7d:
            escape = True
        elif value == 0x7e:
            raise ValueError("Multiple or unescaped wire-frame delimiters")
        else:
            data.append(value)
    if escape or len(data) < 2 or crc_hqx(data, 0xffff) != 0:
        raise ValueError("Wire frame has an invalid CRC or escape")
    return bytes(data[:-2])


def verify_row(row, columns, vector, config):
    get = dict(zip(columns, row, strict=True)).__getitem__
    request_wire = b64decode(vector[0], validate=True)
    request = body(request_wire)
    opcode, expected_token, expected_tag = REQUEST.unpack_from(request)
    if opcode != 0x45 or expected_tag != config["image_tag"]:
        raise ValueError("Original request has a different mode or image tag")
    if ([get(k) for k in ["truth", "cpu_prediction", "cpu_us", "reference_status", "reference_cycles"]] != vector[1:6]
            or get("request_wire_bytes") != len(request_wire)):
        raise ValueError("Captured metadata differs from its original request fixture")
    reply_wire = bytes.fromhex(get("fpga_wire_hex"))
    reply = body(reply_wire)
    opcode, token, tag, status, cycles, frame, nbytes = REPLY.unpack_from(reply)
    expected_bytes = config["rows"] * ((config["width"] + 7) // 8) if status == 3 else 0
    if ((opcode, token, tag) != (0x92, expected_token, expected_tag)
            or status not in (1, 3) or frame not in (0, 1) or not 0 < cycles <= config["max_cycles"]
            or len(reply) != REPLY.size + nbytes or nbytes != expected_bytes
            or get("reply_wire_bytes") != len(reply_wire)
            or get("fpga_status") != status or get("core_cycles") != cycles
            or get("fpga_frame_bit") != frame or get("fpga_prediction") != (frame if status == 1 else None)):
        raise ValueError("Wire reply token, result or length differs from its original ordered request")
    residual = reply[REPLY.size:].hex() if nbytes else None
    if residual != get("fpga_residual_hex"):
        raise ValueError("Captured residual differs from the raw FPGA reply")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--resume-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(path.resolve() == args.output.resolve() for path in args.parts):
        raise ValueError("Merged output must be separate from every raw part")
    fixture = load(args.fixtures)
    config = fixture["surface"]
    plan = load(args.resume_plan) if args.resume_plan else None
    if plan and plan["original_fixtures_sha256"] != identity(args.fixtures)["sha256"]:
        raise ValueError("Resume plan identifies different original request fixtures")
    offset_plan = {(item["p"], item["seed"]): item for item in (plan or {}).get("segments", [])}
    cohorts = {(item["p"], item["seed"]): item for item in config["cohorts"]}
    captures = [(path, load(path)) for path in args.parts]
    columns = captures[0][1]["columns"]
    if len(set(columns)) != len(columns) or "fpga_wire_hex" not in columns:
        raise ValueError("Merging requires unique columns and raw FPGA reply bytes")
    collected = {key: [] for key in cohorts}
    transport_events = []
    for part_index, (path, capture) in enumerate(captures):
        if capture["format"] != "fpga-uart-benchmark-v1" or capture["columns"] != columns:
            raise ValueError("Capture-part formats or columns differ")
        for run_index, run in enumerate(capture["runs"]):
            key = run["p"], run["seed"]
            if key not in cohorts:
                raise ValueError("Capture part has an unknown noise/seed cohort")
            rows = run["rows"]
            start = run.get("shot_start")
            if start is None:
                start = offset_plan[key]["shot_start"] if part_index and key in offset_plan else 0
            n = len(rows)
            if (run["mode"] != "surface" or run.get("image_tag", config["image_tag"]) != config["image_tag"]
                    or run["clock_hz"] != config["clock_hz"] or run["baud_rate"] != config["baud_rate"]
                    or n > run["planned"] or start < 0 or start + n > len(cohorts[key]["shots"])
                    or run.get("complete") and n != run["planned"]):
                raise ValueError("Capture segment has an invalid range or hardware identity")
            segment = {"source": identity(path), "run_index": run_index, "shot_start": start,
                       "shot_stop": start + n, "validated_shots": n, "original_planned": run["planned"],
                       "original_complete": run.get("complete"), "original_fault": run.get("fault"),
                       "original_recoveries": run.get("recoveries", []), "warmup_shots": run.get("warmup_shots", 0)}
            if run.get("fault") or run.get("recoveries"):
                transport_events.append({"p": run["p"], "seed": run["seed"], **segment})
            if not n:
                continue
            for position, row in enumerate(rows, start=start):
                try:
                    verify_row(row, columns, cohorts[key]["shots"][position], config)
                except (ValueError, IndexError, struct.error) as error:
                    raise ValueError(f"Invalid segment {path.name} run{run_index} original shot{position}: {error}") from error
            collected[key].append((start, rows, segment, run))

    merged_runs = []
    for cohort in config["cohorts"]:
        key = cohort["p"], cohort["seed"]
        cursor, rows, segments = 0, [], []
        for start, segment_rows, segment, run in sorted(collected[key], key=lambda item: item[0]):
            if start != cursor:
                raise ValueError(f"Gap or duplicate at p={key[0]} seed={key[1]}: expected original row{cursor}, received{start}")
            rows.extend(segment_rows)
            segments.append(segment)
            cursor += len(segment_rows)
        if cursor != len(cohort["shots"]):
            raise ValueError(f"Incomplete merged cohort p={key[0]}: {cursor}/{len(cohort['shots'])}")
        merged_runs.append({"mode": "surface", "image_tag": config["image_tag"], "p": key[0], "seed": key[1],
                            "clock_hz": config["clock_hz"], "baud_rate": config["baud_rate"],
                            "planned": cursor, "warmup_shots": sum(item["warmup_shots"] for item in segments),
                            "complete": True, "fault": None, "recoveries": [], "rows": rows,
                            "capture_segments": segments})
    output = {"format": "fpga-uart-benchmark-v1", "serial": captures[0][1]["serial"],
              "started_at": captures[0][1].get("started_at"), "exported_at": datetime.now(UTC).isoformat(),
              "columns": columns, "runs": merged_runs,
              "merge_provenance": {"method": "contiguous original-shot-index join; every request/reply token and CRC verified",
                                   "fixtures": identity(args.fixtures), "raw_parts": [identity(path) for path, _ in captures],
                                   "resume_plan": identity(args.resume_plan) if args.resume_plan else None,
                                   "retained_transport_events": transport_events,
                                   "request_reply_frames_verified": sum(len(run["rows"]) for run in merged_runs),
                                   "resampled_inputs": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(args.output.suffix + ".tmp")
    staging.write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")
    staging.replace(args.output)
    print(json.dumps({"output": str(args.output), "cohorts": len(merged_runs),
                      "shots": [len(run["rows"]) for run in merged_runs],
                      "retained_transport_events": len(transport_events)}))


if __name__ == "__main__":
    main()

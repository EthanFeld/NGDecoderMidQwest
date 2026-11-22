"""Evaluate FPGA-final growth accepts and the original joint tensor tail."""

import argparse
import json
import sys
from base64 import b64decode
from pathlib import Path
from time import perf_counter_ns

import numpy as np

SURFACE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SURFACE), str(SURFACE.parents[1] / "decoders/telescope"), str(SURFACE.parents[1] / "decoders/telescope/tensor_decoder/src"), str(SURFACE.parents[1] / "decoders/pymatching_telescope/src")]


def checked_capture(capture, columns, fixtures, profile, memory, truth):
    from host import COMMAND, REQUEST, Inbox, unpack
    from surface_model import seal
    from pymatching_tensor_decoder.telescope.decoder import FrontBatch

    dated = json.loads(fixtures.read_bytes())["surface"]
    if dated["image_tag"] != seal(profile):
        raise ValueError("the fixture image tag differs from the loaded profile")
    cohort = next(
        (c for c in dated["cohorts"] if (c["p"], c["seed"]) == (capture["p"], capture["seed"])),
        None,
    )
    rows = capture["rows"]
    if cohort is None or len(cohort["shots"]) < len(rows):
        raise ValueError("the capture and fixture have different cohorts")
    width, layers = profile["width"], profile["rounds"] + 1
    accepted = np.empty(len(rows), dtype=bool)
    residual = np.zeros_like(memory)
    logical = np.empty(len(rows), dtype=np.uint8)
    cycles = np.empty(len(rows), dtype=np.int32)
    for i, (row, fixture) in enumerate(zip(rows, cohort["shots"][:len(rows)], strict=True)):
        body, = Inbox().feed(b64decode(fixture[0], validate=True))
        opcode, _, tag = REQUEST.unpack_from(body)
        expected = unpack(body[REQUEST.size:], width, layers)
        if (
            opcode != COMMAND["tilt"]
            or tag != dated["image_tag"]
            or not np.array_equal(expected, memory[i])
        ):
            raise ValueError(f"FPGA request {i} does not match the joint syndrome")
        controls = [row[columns[name]] for name in ("truth", "cpu_prediction", "cpu_us")]
        oracle = [row[columns[name]] for name in ("reference_status", "reference_cycles")]
        if controls != fixture[1:4] or controls[0] != truth[i] or oracle != fixture[4:6]:
            raise ValueError(f"FPGA reply {i} is out of sequence")
        bit = row[columns["fpga_prediction"]]
        status = row[columns["fpga_status"]]
        count = row[columns["core_cycles"]]
        if (
            status not in (1, 3)
            or bit not in ((0, 1) if status == 1 else (None,))
            or not 0 < count <= dated["max_cycles"]
        ):
            raise ValueError(f"FPGA reply {i} has invalid status, prediction, or cycle count")
        accepted[i] = status == 1
        logical[i] = bit if accepted[i] else 0
        cycles[i] = count
        if "fpga_residual_hex" in columns:
            raw = row[columns["fpga_residual_hex"]]
            if (raw is None) != accepted[i]:
                raise ValueError(f"FPGA reply {i} has inconsistent residual metadata")
            if raw is not None:
                residual[i] = unpack(bytes.fromhex(raw), width, layers)
                if not residual[i].any() or np.any(residual[i] & ~memory[i]):
                    raise ValueError(f"FPGA reply {i} has an invalid deferred syndrome")
    return FrontBatch(accepted, residual, logical, cycles)


def latency_summary(values):
    if not len(values):
        return None
    median, p99, p999 = np.percentile(values, [50, 99, 99.9])
    return {
        "mean": float(np.mean(values)),
        "median": float(median),
        "p99": float(p99),
        "p999": float(p999),
    }


def time_capture(telescope, events, front, predictions, capture, columns, count):
    from pymatching_tensor_decoder.telescope.decoder import FrontBatch

    indices = np.linspace(0, len(events) - 1, min(count, len(events)), dtype=np.intp)
    host_us = np.empty(len(indices))
    for j, i in enumerate(indices):
        sl = slice(i, i + 1)
        row = events[sl]
        reply = FrontBatch(front.accepted[sl], front.residual[sl], front.logical[sl], front.cycles[sl])
        start = perf_counter_ns()
        memory = telescope.memory(row)
        label, _ = telescope.complete_with_stages(row, reply, memory)
        host_us[j] = (perf_counter_ns() - start) / 1000
        np.testing.assert_array_equal(label, predictions[sl])

    core_us = front.cycles[indices] * 1e6 / capture["clock_hz"]
    uart_us = []
    for i in indices:
        row = capture["rows"][i]
        delivery = row[columns["delivery_us"]] if "delivery_us" in columns else None
        uart_us.append(delivery if delivery is not None else row[columns["uart_round_trip_us"]])
    without_uart_us = core_us + host_us
    with_uart_us = np.asarray(uart_us) + host_us
    accepted = front.accepted[indices]
    groups = {"all": np.ones(len(indices), dtype=bool), "fpga_accepts": accepted, "fpga_defers": ~accepted}
    return {
        name: {
            "shots": int(mask.sum()),
            "host_projection_and_completion_us": latency_summary(host_us[mask]),
            "without_uart_us": latency_summary(without_uart_us[mask]),
            "with_uart_us": latency_summary(with_uart_us[mask]),
        }
        for name, mask in groups.items()
    }


def main():
    from growth_reference import GrowthModel
    from pymatching_tensor_decoder.telescope.decoder import TENSOR, ReferenceFront, TelescopeDecoder
    from pymatching_tensor_decoder.telescope.telescope_circuit import build_context

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=72611005)
    parser.add_argument("--noise", type=float, default=.005)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--build", type=Path, required=True, help="rendered graph, topology and matching cycle model")
    parser.add_argument("--time-shots", type=int, default=0, help="measure host projection and completion on captured replies")
    parser.add_argument("--tensor-control", action="store_true", help="score tensor-only decoding offline after timing")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.capture) != bool(args.fixtures):
        parser.error("pass both the board capture and its original fixture")
    if args.time_shots and not args.capture:
        parser.error("timing needs a physical board capture")
    if args.shots < 1 or args.time_shots < 0 or args.out.exists():
        parser.error("use a positive shot count, a nonnegative timing count, and a new output path")

    work = args.build
    profile = json.loads((work / "graph.json").read_bytes())
    ctx, cols = build_context(profile, args.noise)
    captured = None
    seed, shots = args.seed, args.shots
    if args.capture:
        archive = json.loads(args.capture.read_bytes())
        columns = {name: i for i, name in enumerate(archive["columns"])}
        captured = next(
            (c for c in archive["runs"] if c["mode"] == "surface" and c["p"] == args.noise),
            None,
        )
        if (
            captured is None
            or not captured["complete"]
            or captured["fault"]
            or captured["clock_hz"] <= 0
            or not len(captured["rows"]) == captured["planned"] > 0
        ):
            raise ValueError("the paired capture must be complete")
        seed, shots = captured["seed"], len(captured["rows"])

    events, truth = ctx.circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True, bit_packed=True,
    )
    events.flags.writeable = False
    truth = truth[:, 0]
    library = work / "sim" / ("growth_model.dll" if sys.platform == "win32" else "growth_model.so")
    model = GrowthModel(work / "growth_graph.txt", library)
    grower = ReferenceFront(profile, model)
    telescope = TelescopeDecoder(ctx, profile, cols, front=grower)
    memory = telescope.memory(events)
    reference = grower.decode(memory)
    replay = checked_capture(captured, columns, args.fixtures, profile, memory, truth) if captured else reference
    accepted = replay.accepted
    predictions, stages = telescope.complete_with_stages(events, replay, memory)
    np.testing.assert_array_equal(stages == TENSOR, ~accepted)
    np.testing.assert_array_equal(predictions[accepted, 0], replay.logical[accepted])
    wrong = predictions[:, 0] != truth
    output = {
        "noise": args.noise,
        "seed": seed,
        "shots": shots,
        "source": "board" if captured else "model",
        "reference_status_mismatches": int(np.count_nonzero(accepted != reference.accepted)),
        "reference_cycle_mismatches": int(np.count_nonzero(replay.cycles != reference.cycles)),
        "reference_shared_accept_bit_mismatches": int(np.count_nonzero(
            accepted & reference.accepted & (replay.logical != reference.logical)
        )),
        "fpga_telescope": {
            "accepted": int(accepted.sum()),
            "forwarded": int((~accepted).sum()),
            "forwarded_pct": float(100 * (~accepted).mean()),
            "errors": int(wrong.sum()),
            "accepted_errors": int((accepted & wrong).sum()),
            "forwarded_errors": int((~accepted & wrong).sum()),
        },
    }
    if args.time_shots:
        output["clock_hz"] = captured["clock_hz"]
        output["latency"] = time_capture(telescope, events, replay, predictions, captured, columns, args.time_shots)
    if args.tensor_control:
        raw = np.concatenate([
            telescope.backend.decode_batch(events[i : i + 256])[:, 0]
            for i in range(0, shots, 256)
        ])
        np.testing.assert_array_equal(predictions[~accepted, 0], raw[~accepted])
        control_wrong = raw != truth
        output["tensor_only_accuracy_control"] = {
            "errors": int(control_wrong.sum()),
            "telescope_fixes": int((control_wrong & ~wrong).sum()),
            "telescope_regressions": int((~control_wrong & wrong).sum()),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()


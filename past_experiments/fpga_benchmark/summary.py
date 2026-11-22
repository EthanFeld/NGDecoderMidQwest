"""Calculate observed logical error rates and latency from captured UART replies."""

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np


PROBABILITIES = (50, 95, 99)


def wilson(errors, total):
    if not total:
        return None
    z = 1.95996398454
    rate = errors / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    half_width = z * sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return {"errors": errors, "shots": total, "rate": rate,
            "ci95": [max(0, center - half_width), min(1, center + half_width)]}


def latency(values):
    samples = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(samples):
        return None
    percentiles = np.percentile(samples, PROBABILITIES)
    return {
        "shots": len(samples), "mean_us": float(samples.mean()),
        **{f"p{rank}_us": float(value) for rank, value in zip(PROBABILITIES, percentiles)},
        "max_us": float(samples.max()),
    }


def summarize_rows(rows, clock_hz, baud_rate):
    """Use CRC-checked FPGA decisions and the paired CPU fallback for each row."""
    circuit_errors = accepts = wrong_accepts = wrong_defers = cpu_errors = 0
    uart_round_trip, uart_wire, core_latency, crc_tail = [], [], [], []
    full_estimate, cpu_fallback, core_cpu_estimate = [], [], []
    for row in rows:
        (truth, cpu_prediction, cpu_us, fpga_prediction, core_cycles, tail_cycles,
         uart_us, _extended_us, request_bytes, reply_bytes) = row[:10]
        accepted = fpga_prediction is not None
        final_prediction = fpga_prediction if accepted else cpu_prediction
        circuit_errors += final_prediction != truth
        cpu_errors += cpu_prediction != truth
        accepts += accepted
        wrong_accepts += accepted and final_prediction != truth
        wrong_defers += not accepted and final_prediction != truth
        fallback_us = 0 if accepted else cpu_us
        uart_round_trip.append(uart_us)
        uart_wire.append((request_bytes + reply_bytes) * 10e6 / baud_rate)
        full_estimate.append(uart_us + fallback_us)
        if not accepted:
            cpu_fallback.append(fallback_us)
        if core_cycles is not None:
            core_us = core_cycles * 1e6 / clock_hz
            core_latency.append(core_us)
            core_cpu_estimate.append(core_us + fallback_us)
        if tail_cycles is not None:
            crc_tail.append(tail_cycles * 1e6 / clock_hz)

    total = len(rows)
    return {
        "hybrid_ler": wilson(circuit_errors, total), "cpu_only_ler": wilson(cpu_errors, total),
        "fpga_accept_ler": wilson(wrong_accepts, accepts),
        "cpu_defer_ler": wilson(wrong_defers, total - accepts),
        "fpga_accepts": accepts, "cpu_defers": total - accepts,
        "uart_round_trip": latency(uart_round_trip), "uart_line_time": latency(uart_wire),
        "fpga_cycles_at_clock": latency(core_latency), "post_request_fpga": latency(crc_tail),
        "cpu_fallback": latency(cpu_fallback),
        "estimated_end_to_end": latency(full_estimate),
        "estimated_device_and_cpu": latency(core_cpu_estimate),
        "uart_accepts": latency(row[6] for row in rows if row[3] is not None),
        "uart_defers": latency(row[6] for row in rows if row[3] is None),
        "wire_without_telemetry": latency(
            (row[8] + 8) * 10e6 / baud_rate for row in rows if row[7] is not None
        ),
        "uart_with_telemetry_extension": latency(row[7] for row in rows),
        "link_retries": sum(row[13] for row in rows if len(row) >= 14),
        "shots_retried": sum(len(row) >= 14 and row[13] > 0 for row in rows),
        "uart_round_trip_with_recovery": latency(row[14] for row in rows if len(row) >= 15),
        "reference_status_mismatches": sum(
            len(row) >= 12 and row[11] is not None and row[10] != row[11] for row in rows
        ),
        "reference_cycle_mismatches": sum(
            len(row) >= 13 and row[12] is not None and row[4] != row[12] for row in rows
        ),
    }


def add_direct_rtl_timing(runs, direct):
    """Join replayed Tang packets to their measured UART rows in fixture order."""
    by_seed = {(run["p"], run["seed"]): run for run in direct["runs"]}
    clock_hz = direct["clock_hz"]
    for run in runs:
        if run["mode"] != "tang" or not run["rows"]:
            continue
        replay = by_seed[(run["p"], run["seed"])]
        rows = run["rows"]
        simulated_rows = replay["rows"][:len(rows)]
        recorded = list(zip(rows, simulated_rows, strict=True))
        if any(board[10] != sim[0] for board, sim in recorded):
            raise ValueError(f"Tang device and RTL decisions disagree at p={run['p']}")
        no_uart = [sim[1] * 1e6 / clock_hz for _, sim in recorded]
        no_uart_with_cpu = [us + (board[2] if board[3] is None else 0)
                            for us, (board, _) in zip(no_uart, recorded, strict=True)]
        run["tang_rtl_without_uart"] = latency(no_uart)
        run["tang_rtl_without_uart_accepts"] = latency(
            us for us, (board, _) in zip(no_uart, recorded, strict=True) if board[3] is not None
        )
        run["tang_estimated_rtl_plus_cpu_without_uart"] = latency(no_uart_with_cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", nargs="+", type=Path)
    parser.add_argument("--tang-direct-rtl", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    all_runs = []
    for filename in args.measurements:
        captures = json.loads(filename.read_text(encoding="utf-8-sig"))
        runs = captures["runs"]
        if args.tang_direct_rtl:
            add_direct_rtl_timing(runs, json.loads(args.tang_direct_rtl.read_text(encoding="utf-8")))
        for run in runs:
            rows = run["rows"]
            stats = summarize_rows(rows, run["clock_hz"], run["baud_rate"])
            all_runs.append({**{key: value for key, value in run.items() if key != "rows"}, **stats})
    args.output.write_text(json.dumps(all_runs, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

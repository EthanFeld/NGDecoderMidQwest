"""Render the decoder sweep and the separately labeled board replay."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODES = {
    "fpga_only": ("FPGA, zero on defer", "#a94316", "s"),
    "tensor_only": ("Tensor only", "#26458c", "o"),
    "telescope": ("Telescope", "#00736c", "D"),
    "pymatching": ("PyMatching", "#7b5795", "x"),
    "correlated_pymatching": ("Correlated PyMatching", "#575b65", "+"),
    "relay_bp": ("Relay-BP", "#9c6b04", "^"),
}
DISTANCES = (3, 7, 15)
ANCHORS = (.001, .005, .010)
RATES = (.001, .002, .003, .004, .005, .006, .007, .0075, .008, .009, .010)


def mode_key(row):
    return "fpga_only" if row["mode"].startswith("fpga_only") else row["mode"]


def by_distance(rows):
    points = defaultdict(list)
    for row in rows:
        points[row["distance"], mode_key(row)].append(row)
    return {key: sorted(values, key=lambda row: row["p"]) for key, values in points.items()}


def plot(rows, *, latency):
    points = by_distance(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.55), sharey=True)
    for distance, ax in zip(DISTANCES, axes, strict=True):
        for mode, (label, color, marker) in MODES.items():
            samples = points.get((distance, mode), [])
            x = [100 * row["p"] for row in samples]
            means = [row["single_shot_us_excluding_uart"]["mean"] for row in samples]
            rates = [100 * row["ler"] if row["errors"] else float("nan") for row in samples]
            ax.plot(x, means if latency else rates, label=label, color=color,
                    marker=marker, linewidth=1.35, markersize=4.3)
            if not latency:
                zero = [row for row in samples if not row["errors"]]
                ax.scatter([100 * row["p"] for row in zero],
                           [100 * row["ci95_high"] for row in zero],
                           color=color, marker="v", s=19, alpha=.88)
        ax.set_title(f"Distance {distance}", loc="left", fontweight="bold")
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", alpha=.23)
        ax.set_xlabel("Physical error rate (%)")
        ax.set_xticks([.1, .3, .5, .75, 1.0])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean single-shot time (µs)" if latency else "Logical error rate (%)")
    title = "Single-shot decode time excluding UART" if latency else "Logical error rate per memory shot"
    fig.suptitle(title, x=.075, ha="left", fontsize=15, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.075, right=.985, bottom=.245, top=.82, wspace=.16)
    fig.savefig(ROOT / ("latency.png" if latency else "ler.png"), dpi=190)
    plt.close(fig)


def ratio(row):
    return f"{row['errors']:,}/{row['shots']:,} ({row['ler']:.3%})"


def single(row):
    timing = row["single_shot_us_excluding_uart"]
    return f"{timing['mean']:,.2f} / {timing['median']:,.2f}"


def table_header(heading, left, names):
    return [
        f"## {heading}", "",
        "| " + " | ".join((*left, *names)) + " |",
        "| " + " | ".join(["---:"] * (len(left) + len(names))) + " |",
    ]


def model_tables(rows):
    index = {(row["distance"], round(row["p"], 6), mode_key(row)): row for row in rows}
    names = [style[0] for style in MODES.values()]
    lines = []
    for label, format_cell in (
        ("Logical errors at selected noise points", ratio),
        ("Single-shot mean / median at selected noise points (µs)", single),
    ):
        left = ("Distance", "Noise", "FPGA defer rate") if format_cell is ratio else ("Distance", "Noise")
        lines.extend(table_header(label, left, names))
        for distance in DISTANCES:
            for p in ANCHORS:
                entries = [index[distance, p, mode] for mode in MODES]
                fields = [str(distance), f"{p:.3f}"]
                if format_cell is ratio:
                    fields.append(f"{entries[0]['fpga_defer_rate']:.2%}")
                fields.extend(format_cell(row) for row in entries)
                lines.append("| " + " | ".join(fields) + " |")
        lines.append("")
    return lines


def relay_comparison(rows):
    names = [style[0] for style in MODES.values()]
    lines = table_header("Paired errors on the Relay-BP sample at 0.50% noise", ("Distance", "Shots"), names)
    for distance in DISTANCES:
        point = {mode_key(row): row for row in rows if row["distance"] == distance and row["p"] == .005}
        relay = point["relay_bp"]
        counts = []
        for mode in MODES:
            paired = relay.get(f"relay_paired_{mode}_errors")
            if paired is None:
                if point[mode]["shots"] != relay["shots"]:
                    raise ValueError(f"Missing d{distance} Relay-BP paired count for {mode}")
                paired = point[mode]["errors"]
            counts.append(str(paired))
        lines.append("| " + " | ".join((str(distance), f"{relay['shots']:,}", *counts)) + " |")
    return [*lines, ""]


def board_table():
    path = ROOT / "board/physical_replays.json"
    if not path.exists():
        return []
    archives = json.loads(path.read_bytes())
    lines = table_header("Archived physical FPGA results", (
        "Distance", "Noise", "Shots", "Native coverage", "Forced FPGA errors",
        "Core mean (µs)", "UART round-trip mean (ms)", "Full joint input verified",
    ), ())
    for archive in archives:
        for cohort in archive["cohorts"]:
            fields = (
                str(archive["distance"]), f"{cohort['p']:.4f}", f"{cohort['shots']:,}",
                f"{cohort['coverage']:.2%}", str(cohort["fpga_forced_wire_bit"]["logical_errors"]),
                f"{cohort['fpga_core_us']['mean']:.2f}",
                f"{cohort['uart_round_trip_us']['mean'] / 1000:.2f}",
                "Yes" if cohort["joint_pairing"]["all_paired"] else "Unavailable",
            )
            lines.append("| " + " | ".join(fields) + " |")
    lines.extend(("", "[Physical CSV](board/physical_summary.csv) · [Provenance and paired board comparisons](board/physical_replays.json)", ""))
    lines.extend(table_header("Paired telescope on archived board inputs", (
        "Archive", "Distance", "Noise", "Telescope errors", "Tensor-only errors", "Correlated PyMatching errors",
        "Telescope mean excluding UART (µs)", "Telescope mean including UART (ms)",
    ), ()))
    for archive in archives:
        for cohort in archive["cohorts"]:
            point = {mode_key(row): row for row in cohort.get("joint_comparison", ())}
            if not point:
                continue
            telescope = point["telescope"]
            fields = (
                archive["archive"], str(archive["distance"]), f"{cohort['p']:.4f}",
                ratio(telescope), ratio(point["tensor_only"]), ratio(point["correlated_pymatching"]),
                f"{telescope['single_shot_us_excluding_uart']['mean']:,.2f}",
                f"{telescope['single_shot_us_including_uart']['mean'] / 1000:,.2f}",
            )
            lines.append("| " + " | ".join(fields) + " |")
    lines.extend(("", "[All paired board modes and latencies](board/paired_comparison.csv)", ""))
    return lines


def main():
    raw = json.loads((ROOT / "model/results.json").read_bytes())
    rows = raw["results"]
    points = {(row["distance"], round(row["p"], 6), mode_key(row)) for row in rows}
    expected = {(distance, p, mode) for distance in DISTANCES for p in RATES for mode in MODES}
    if len(rows) != len(points) or points != expected:
        raise SystemExit(f"Expected the complete {len(expected)}-row sweep, found {len(rows)} rows")
    for row in rows:
        if not (
            0 <= row["errors"] <= row["shots"]
            and row["ler"] == row["errors"] / row["shots"]
            and row["ci95_low"] - 1e-12 <= row["ler"] <= row["ci95_high"] + 1e-12
        ):
            raise ValueError(f"Inconsistent accuracy record for d{row['distance']} p={row['p']}")
        if row["mode"] == "telescope" and not (
            row["tensor_tail_shots"] == row["fpga_defers"]
            and row["errors"] == row["fpga_accepted_errors"] + row["tensor_tail_errors"]
        ):
            raise ValueError(f"Inconsistent telescope routing at d{row['distance']} p={row['p']}")
    plot(rows, latency=False)
    plot(rows, latency=True)
    notes = [
        "# Decoder comparison", "",
        "This run covers 33 distance/noise points on the paired Surface Telescope circuit.",
        "The model FPGA scores use the selected images' cycle references. Single-shot times",
        "include measured host work and exclude UART. The FPGA-only score uses zero on",
        "a defer. D3 provides a hardware decision for every input, so its FPGA-only and",
        "telescope accuracy curves coincide.", "",
        "Downward triangles on the LER plot show 95% Wilson upper bounds for zero observed errors.",
        "Relay-BP uses the smaller shared subsets identified in the raw results. Use matching",
        "shot counts or the paired-control columns for accuracy comparisons at those points.", "",
        "Latency percentiles summarize the timed samples. P99.9 is close to the observed",
        "maximum at these sample sizes. See the CSV for each mode's timing sample count.", "",
        "![Logical error rate across the noise sweep](ler.png)", "",
        "![Single-shot decode time across the noise sweep](latency.png)", "",
        "[Full CSV](model/results.csv) · [Full records, tails and settings](model/results.json)", "",
    ]
    notes.extend(model_tables(rows))
    notes.extend(relay_comparison(rows))
    notes.extend(board_table())
    notes.append("[Benchmark guide and decoder documentation](README.md)")
    (ROOT / "COMPARISON.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

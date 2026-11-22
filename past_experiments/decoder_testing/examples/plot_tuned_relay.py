"""Render saved held-out results; never sample, time, or select decoder settings."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

LABELS = {"relay_bp": "Tuned IBM Relay-BP", "pymatching": "PyMatching", "lut": "Exact LUT"}
COLORS = {"relay_bp": "#008577", "pymatching": "#3563bb", "lut": "#a352a4"}
DISTANCES = [2, 3, 5, 7, 9, 11, 13, 15, 17]
DCOLORS = {d: plt.cm.viridis(i / 8) for i, d in enumerate(DISTANCES)}
NOISE = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]


def save(fig, output, name):
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ler_curve(ax, points, xfield, color, label=None, linestyle="-", marker="o"):
    points = sorted(
        (r for r in points if "shots" in r.get("measured", {})), key=lambda r: r[xfield]
    )
    positive = [r for r in points if r["measured"]["errors"]]
    scale = 100 if xfield == "noise" else 1
    if positive:
        x = [r[xfield] * scale for r in positive]
        y = np.asarray([r["measured"]["logical_error_rate"] for r in positive])
        lower = [r["measured"]["ci95_low"] for r in positive]
        upper = [r["measured"]["ci95_high"] for r in positive]
        ax.errorbar(
            x,
            y,
            yerr=[y - lower, upper - y],
            color=color,
            label=label,
            linestyle=linestyle,
            marker=marker,
            ms=4.5,
            lw=1.7,
            capsize=2.5,
        )
    zeros = [r for r in points if not r["measured"]["errors"]]
    if zeros:
        ax.scatter(
            [r[xfield] * scale for r in zeros],
            [r["measured"]["ci95_high"] for r in zeros],
            marker="v",
            facecolors="white",
            edgecolors=color,
            s=37,
            zorder=5,
            label=label if not positive else None,
        )


def setup(ax, xlabel, ylabel, logx=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([p * 100 for p in NOISE]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
        ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.grid(True, which="major", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def first_caps(rows, p):
    return [r for r in rows if r["noise"] == p and r["status"] in ("latency_cap", "resource_cap")]


def distance_plot(rows, historical, output):
    selected = [r for r in rows if r["noise"] == 0.003]
    old = [r for r in historical if r["track"] == "distance_rounds" and r["decoder"] == "relay_bp"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for name in LABELS:
        points = [r for r in selected if r["decoder"] == name]
        ler_curve(axes[0], points, "distance", COLORS[name], LABELS[name])
        timed = [r for r in points if "single_shot_p50_us" in r.get("measured", {})]
        for field, style, alpha, width in [
            ("single_shot_p50_us", "-", 1, 2),
            ("single_shot_p95_us", "--", 0.7, 1.2),
            ("single_shot_p99_us", ":", 0.7, 1.2),
        ]:
            axes[1].plot(
                [r["distance"] for r in timed],
                [r["measured"][field] / 1000 for r in timed],
                color=COLORS[name],
                linestyle=style,
                alpha=alpha,
                lw=width,
                marker="o" if field == "single_shot_p50_us" else None,
                ms=4,
            )
        capped = [r for r in timed if r["status"] == "latency_cap"]
        for row in capped:
            axes[1].scatter(
                row["distance"],
                row["measured"]["single_shot_p50_us"] / 1000,
                marker="x",
                color="#bc3434",
                s=85,
                linewidths=2,
                zorder=6,
            )
    ler_curve(axes[0], old, "distance", "#8a6643", "Earlier fast profile (historical)", "--", "s")
    timed_old = [r for r in old if "measured" in r]
    axes[1].plot(
        [r["distance"] for r in timed_old],
        [r["measured"]["single_shot_p50_us"] / 1000 for r in timed_old],
        color="#8a6643",
        linestyle="-.",
        marker="s",
        ms=4,
        lw=1.6,
    )
    axes[1].axhline(1, color="#bc3434", lw=1.3, label="1 ms median budget")
    setup(axes[0], "Code distance d (rounds = d)", "Logical error probability / memory experiment")
    setup(axes[1], "Code distance d (rounds = d)", "Isolated-shot decode time (ms)")
    axes[0].set_ylim(5e-6, 0.1)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].set_title("LER at circuit error rate p = 0.3%", loc="left", weight="bold")
    axes[1].set_title("Measured latency: median controls admission", loc="left", weight="bold")
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#333333", lw=2, marker="o", label="Median"),
            Line2D([0], [0], color="#333333", linestyle="--", label="p95"),
            Line2D([0], [0], color="#333333", linestyle=":", label="p99"),
            Line2D([0], [0], color="#bc3434", lw=1.3, label="1 ms median budget"),
            Line2D(
                [0], [0], color="#bc3434", marker="x", linestyle="none", label="First latency cap"
            ),
        ],
        fontsize=8.5,
        loc="upper left",
    )
    for ax in axes:
        ax.set_xticks([2, 3, 5, 7, 9, 11, 13, 15, 17])
        ax.set_xlim(1.6, 17.5)
    caps = first_caps(rows, 0.003)
    note = "; ".join(
        f"{LABELS[r['decoder']]}: d{r['distance']} {r['status'].replace('_cap', '')} cap"
        for r in caps
    )
    fig.suptitle(
        "Frozen native IBM Relay-BP tuning | fresh held-out comparison", fontsize=15, weight="bold"
    )
    fig.text(
        0.02,
        0.01,
        "20,000 paired shots/point; 95% Wilson intervals; 2,048 serial latency probes.\n"
        "Earlier profile: independent historical sample. " + note,
        fontsize=8.5,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    save(fig, output, "distance_ler_time")


def noise_plot(rows, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for column, name in enumerate(LABELS):
        points = [r for r in rows if r["decoder"] == name]
        for d in DISTANCES:
            series = [r for r in points if r["distance"] == d]
            ler_curve(axes[0, column], series, "noise", DCOLORS[d])
            timed = [r for r in series if "single_shot_p50_us" in r.get("measured", {})]
            if timed:
                axes[1, column].plot(
                    [r["noise"] * 100 for r in timed],
                    [r["measured"]["single_shot_p50_us"] / 1000 for r in timed],
                    marker="o",
                    ms=4,
                    lw=1.7,
                    color=DCOLORS[d],
                )
                capped = [r for r in timed if r["status"] == "latency_cap"]
                axes[1, column].scatter(
                    [r["noise"] * 100 for r in capped],
                    [r["measured"]["single_shot_p50_us"] / 1000 for r in capped],
                    marker="x",
                    color="#c73737",
                    s=75,
                    linewidths=2,
                    zorder=6,
                )
        axes[0, column].set_title(LABELS[name], weight="bold")
        axes[0, column].set_ylim(5e-6, 0.7)
        axes[1, column].set_ylim(0.0025, 30)
        axes[1, column].axhline(1, color="#bc3434", lw=1.2)
        for row in range(2):
            setup(
                axes[row, column],
                "Circuit error rate p (%)",
                "Logical error probability" if row == 0 else "Median decode time (ms)",
                logx=True,
            )
    axes[1, 2].text(
        0.04,
        0.93,
        "d ≥ 3: LUT resource cap\n(no table constructed)",
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=10,
    )
    handles = [
        Line2D([0], [0], color=DCOLORS[d], marker="o", lw=1.5, label=f"d={d}")
        for d in DISTANCES[:-1]
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#333333",
                marker="v",
                markerfacecolor="white",
                linestyle="none",
                label="Zero errors: 95% upper bound",
            ),
            Line2D(
                [0], [0], color="#bc3434", marker="x", linestyle="none", label="First median cap"
            ),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle(
        "Circuit-noise sweep | rounds = distance | rotated Z memory", fontsize=16, weight="bold"
    )
    fig.text(
        0.02,
        0.008,
        "10,000 paired shots/point (20,000 at p=0.3%); 95% Wilson intervals. "
        "No LER estimated beyond each decoder's first cap. 1 ms line gates median only.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    save(fig, output, "circuit_error_rate_ler_time")


def tail_plot(rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4))
    for ax, (field, title) in zip(
        axes,
        [
            ("single_shot_p50_us", "Median (budget metric)"),
            ("single_shot_p95_us", "p95 (diagnostic)"),
            ("single_shot_p99_us", "p99 (diagnostic)"),
        ],
        strict=True,
    ):
        for d in DISTANCES:
            points = [
                r
                for r in rows
                if r["decoder"] == "relay_bp"
                and r["distance"] == d
                and field in r.get("measured", {})
            ]
            if points:
                ax.plot(
                    [r["noise"] * 100 for r in points],
                    [r["measured"][field] / 1000 for r in points],
                    color=DCOLORS[d],
                    marker="o",
                    ms=4,
                    label=f"d={d}",
                )
        ax.axhline(1, color="#bc3434", lw=1)
        setup(ax, "Circuit error rate p (%)", "Isolated-shot decode time (ms)", logx=True)
        ax.set_title(title, weight="bold")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(
        "Tuned IBM Relay-BP | latency distribution across circuit noise", fontsize=15, weight="bold"
    )
    fig.text(
        0.02,
        0.01,
        "2,048 serial one-shot calls/point. Includes first median-failing diagnostic point; "
        "larger distances skipped per noise rate. p95/p99 never control admission.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    save(fig, output, "circuit_error_rate_tail_time")


def nfmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}g}"


def report(rows, historical, output):
    caps = []
    for p in NOISE:
        for row in first_caps(rows, p):
            name = row["decoder"]
            accepted = [
                r["distance"]
                for r in rows
                if r["noise"] == p and r["decoder"] == name and r["status"] == "accepted"
            ]
            caps.append(
                f"<tr><td>{p:g}</td><td>{LABELS[name]}</td>"
                f"<td>{max(accepted) if accepted else '—'}</td>"
                f"<td>{row['distance']}</td><td>{html.escape(row['reason'])}</td></tr>"
            )
    table = []
    for row in rows:
        if row["status"] == "skipped_after_cap":
            continue
        m = row.get("measured", {})
        estimate = row.get("estimate", {})
        peak = estimate.get("peak_working_bytes")
        ler = (
            (
                f"{m['errors']}/{m['shots']} = {m['logical_error_rate']:.5g} "
                f"[{m['ci95_low']:.3g}, {m['ci95_high']:.3g}]"
            )
            if "shots" in m
            else "—"
        )
        values = [
            f"{row['noise']:g}",
            str(row["distance"]),
            LABELS[row["decoder"]],
            row["status"],
            ler,
        ]
        values += [
            nfmt(m.get(field, 0) / 1000, 4) if field in m else "—"
            for field in ["single_shot_p50_us", "single_shot_p95_us", "single_shot_p99_us"]
        ]
        values += [
            nfmt(peak / 1024**2 if peak is not None else None, 5),
            str(m.get("active_variables", "—")),
        ]
        table.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
    comparison = []
    for old in historical:
        if (
            old["track"] != "distance_rounds"
            or old["decoder"] != "relay_bp"
            or "measured" not in old
        ):
            continue
        match = next(
            (
                r
                for r in rows
                if r["noise"] == 0.003
                and r["distance"] == old["distance"]
                and r["decoder"] == "relay_bp"
                and "shots" in r.get("measured", {})
            ),
            None,
        )
        if match:
            before, after = old["measured"], match["measured"]
            comparison.append(
                f"<tr><td>{old['distance']}</td><td>{before['logical_error_rate'] * 100:.3g}%</td>"
                f"<td>{after['logical_error_rate'] * 100:.3g}%</td>"
                f"<td>{before['single_shot_p50_us'] / 1000:.4g}</td>"
                f"<td>{after['single_shot_p50_us'] / 1000:.4g}</td></tr>"
            )
    page = (
        """<!doctype html><meta charset="utf-8"><title>Tuned IBM Relay-BP benchmark</title>
<style>body{font:16px system-ui;max-width:1500px;margin:35px auto;padding:0 25px;color:#20303d}
img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}
td,th{padding:8px;border-bottom:1px solid #d7dfe5;text-align:left}th{background:#edf3f5}
p{max-width:1050px;line-height:1.6}code{background:#edf3f5;padding:2px
5px}.scroll{overflow:auto}</style>
<h1>Tuned IBM Relay-BP: distance, circuit noise, and a 1 ms median budget</h1>
<p>The initial fast profile ran an uncompressed full XYZ graph with at most 240 iterations.
More distance also means more rounds and a larger, harder graph. In a separate 1,024-shot d9
diagnostic,
all 10 logical failures occurred on nonconverged shots. This supports insufficient
search/convergence
as the source of the observed rising LER; small differences between neighboring distances also
have sampling uncertainty.</p>
<p>The IBM authors' pinned native Rust implementation remains unmodified. The new adapter profile
keeps
the memory check basis and exactly merges mechanisms with identical retained detector and logical
columns.
Dropping the other check basis loses cross-basis conditioning; it is an explicit computational
tradeoff,
not a full-information equivalent transformation. Equivalent-column merging itself preserves the
retained
joint distribution exactly. No matching or Gaussian-repair fallback is used.</p>
<p>Frozen profile: pre_iter=80, num_sets=31, set_max_iter=60, gamma0=0.35,
gamma range [-0.25,0.66], alpha=0.9, stop_nconv=3, F32. Offline selection used separate
calibration seeds
703000–703002 and 704000–704002; held-out master seed is 805001. Selection evaluated 53 candidates
over three stages.
The calibration objective and held-out benchmark are distinct; no held-out truth selected these
settings.</p>
<p>Rotated Z surface memory, rounds=d. Circuit error parameter p is applied to the contract's four
Stim noise controls.
Quality uses 20,000 paired circuit shots at p=.003, 10,000 elsewhere. Error bars are two-sided 95%
Wilson intervals;
zero observed failures are shown at their upper bound. Latency is 2,048 serialized isolated-shot
packed decode calls
after 64 warmup calls, including adapter conversion and excluding graph construction/sampling.
Only <b>median ≤1 ms</b> admits a point; p95 and p99 remain diagnostics. Quality is evaluated only
at admitted points.</p>
<img src="distance_ler_time.png" alt="Distance sweep LER and latency">
<h2>Before and after at p=.003</h2>
<p>Earlier fast-profile measurements are historical independent samples on this machine, not
paired with
the new held-out sample. Timings vary with CPU scheduling, frequency and host load. These results
do not
reproduce the paper's hardware, establish a threshold, or guarantee a per-shot real-time
deadline.</p>
<table><tr><th>d</th><th>Earlier LER</th><th>Tuned LER</th><th>Earlier median (ms)</th><th>Tuned
median (ms)</th></tr>
"""
        + "".join(comparison)
        + """</table>
<img src="circuit_error_rate_ler_time.png" alt="Circuit error-rate sweep LER and median latency">
<img src="circuit_error_rate_tail_time.png" alt="Native median p95 and p99 across circuit error
rates">
<h2>Where each sweep stops</h2>
<p>Resource limits: estimated decoder peak ≤256 MiB; build work ≤200 million units; bounded
mechanism/detector scans.
Memory estimates exclude runtime, caller circuit/DEM and full input/output batches. Tuned Relay
estimates conservatively
use the full unprojected, unmerged graph; active variable counts show the reduced decoding graph.
PyMatching's memory
model is also conservative and is not measured RSS. Stops reflect these configured models, not
fundamental limits.
After the first cap, larger distances are skipped for that implementation and noise rate.</p>
<table><tr><th>p</th><th>Decoder</th><th>Last admitted d</th><th>First cap
d</th><th>Reason</th></tr>
"""
        + "".join(caps)
        + """</table>
<h2>Measured results and conservative resources</h2>
<p>Latency columns are milliseconds. LER is per complete memory experiment, not normalized per
round.
Dash means unmeasured, not zero. Active N is reported for Relay after marginalization/merging.</p>
<div class="scroll"><table><tr><th>p</th><th>d</th><th>Decoder</th><th>Status</th><th>LER [95%
CI]</th>
<th>Median ms</th><th>p95 ms</th><th>p99 ms</th><th>Estimated peak MiB</th><th>Active N</th></tr>
"""
        + "".join(table)
        + """</table></div>
<p><a href="results.csv">All results CSV</a> · <a href="results.json">JSON</a> ·
<a href="manifest.json">Frozen configuration, source/native identity and environment</a> ·
<a href="verification.json">Verification checks</a> ·
<a href="../relay-tuning-20261121/manifest.json">Calibration provenance</a> ·
<a href="https://arxiv.org/html/2506.01779v2">IBM authors' paper</a> ·
<a href="https://github.com/trmue/relay/tree/72341ef85cd45779e6483c338aef7ba85094eac1">Pinned
upstream source</a></p>
"""
    )
    (output / "report.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("past_experiments/decoder_testing/results/relay-tuned-heldout-20261121"),
    )
    args = parser.parse_args()
    output = args.directory.resolve()
    rows = json.loads((output / "results.json").read_text())
    baseline = output.parent / "latency-median-1ms-ibm-20261121/results.json"
    historical = json.loads(baseline.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlepad": 12})
    distance_plot(rows, historical, output)
    noise_plot(rows, output)
    tail_plot(rows, output)
    report(rows, historical, output)
    print(output / "report.html")


if __name__ == "__main__":
    main()

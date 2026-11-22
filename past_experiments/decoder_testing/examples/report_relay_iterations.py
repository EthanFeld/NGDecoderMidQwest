"""Render the iteration audit and fresh validation without changing any decoder."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from qecbench.metrics import wilson_interval
from qecbench.storage import write_json

ORDER = [
    "latency_profile",
    "iteration_candidate",
    "long_memory_search",
    "xyz_accuracy",
    "pymatching",
]
COLORS = ["#008578", "#d18b22", "#8953ad", "#ca4d47", "#3267b5"]
TICKS = [
    "XZ 32 legs\nS3 / T60",
    "XZ 32 legs\nS3 / T120",
    "XZ 601 legs\nS5 / T60",
    "XYZ 601 legs\nS5 / T60",
    "PyMatching",
]
LABELS = dict(
    zip(
        ORDER,
        [
            "Earlier latency profile",
            "Per-leg iteration candidate",
            "Long memory-basis search",
            "Full XYZ accuracy profile",
            "PyMatching",
        ],
        strict=True,
    )
)


def save(fig, output, name):
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def decorate(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)


def quality_point(ax, index, row, color):
    if row["errors"]:
        rate = row["logical_error_rate"] * 100
        ax.errorbar(
            index,
            rate,
            yerr=[[rate - row["ci95_low"] * 100], [row["ci95_high"] * 100 - rate]],
            color=color,
            marker="o",
            capsize=4,
            ms=6,
            lw=1.8,
        )
    else:
        ax.scatter(
            index,
            row["ci95_high"] * 100,
            marker="v",
            facecolors="white",
            edgecolors=color,
            s=55,
            zorder=5,
        )


def validation_figure(rows, output):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.7))
    for column, (d, p) in enumerate([(5, 0.007), (7, 0.003), (9, 0.003)]):
        points = {r["profile"]: r for r in rows if r["distance"] == d and r["noise"] == p}
        for i, name in enumerate(ORDER):
            row = points[name]
            quality_point(axes[0, column], i, row, COLORS[i])
            times = [row[key] / 1000 for key in ("median_us", "p95_us", "p99_us")]
            axes[1, column].plot([i, i], [times[0], times[2]], color=COLORS[i], alpha=0.45, lw=1.5)
            for value, marker, size in zip(times, ("o", "s", "^"), (7, 5, 6), strict=True):
                axes[1, column].plot(
                    i, value, color=COLORS[i], marker=marker, ms=size, linestyle="none"
                )
            axes[0, column].annotate(
                f"{row['errors']}/{row['shots']}",
                (i, row["ci95_high"] * 100),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=COLORS[i],
            )
        decorate(axes[0, column], "Logical error rate (%)")
        decorate(axes[1, column], "Isolated-shot decode time (ms)")
        axes[0, column].set_title(f"d={d}, circuit p={p * 100:g}%", weight="bold")
        axes[1, column].axhline(1, color="#be3535", lw=1.3)
        for ax in axes[:, column]:
            ax.set_xlim(-0.5, 4.5)
            ax.set_xticks(range(5))
        axes[0, column].set_xticklabels([])
        axes[1, column].set_xticklabels(TICKS, fontsize=8.5)
        maximum = max(r["ci95_high"] for r in points.values()) * 100
        positive = [r["ci95_low"] * 100 for r in points.values() if r["errors"]]
        axes[0, column].set_ylim(min(positive) / 2 if positive else maximum / 20, maximum * 2.5)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], color="#333", marker=marker, linestyle="none", label=label)
            for marker, label in [("o", "Median"), ("s", "p95"), ("^", "p99")]
        ]
        + [Line2D([0], [0], color="#be3535", label="1 ms median budget")],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.045),
        fontsize=10,
    )
    fig.suptitle(
        "Frozen IBM Relay-BP profiles | fresh paired validation", weight="bold", fontsize=17
    )
    fig.text(
        0.02,
        0.01,
        "Same circuit shots per column; 95% Wilson LER intervals. "
        "1,024 serial latency probes after 64 warmup calls. "
        "Triangle at upper bound = zero observed errors.\n"
        "S = valid solutions sought (repeats count); T = iterations per relay leg. "
        "All first legs use up to 80 iterations. "
        "d11's 512-shot iteration diagnostic is in the report.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    save(fig, output, "accuracy_validation")


def iteration_figure(rows, output):
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    short = [
        "base",
        "pre30",
        "pre160",
        "T30",
        "T120",
        "T240",
        "R8",
        "R101",
        "R301",
        "R601",
        "R601/S1",
        "R601/S5",
        "R601/S10",
        "all32",
    ]
    for column, (d, p) in enumerate([(7, 0.007), (9, 0.003)]):
        series = [r for r in rows if r["distance"] == d and r["noise"] == p]
        x = np.arange(len(series))
        for i, row in enumerate(series):
            quality_point(axes[0, column], i, row, "#008578")
        n = series[0]["shots"]
        pm = series[0]["pymatching_errors"]
        pm_value = pm / n if pm else wilson_interval(pm, n)[1]
        axes[0, column].axhline(
            pm_value * 100,
            color="#3267b5",
            linestyle="--",
            label="PyMatching LER" if pm else "PyMatching zero-error 95% upper bound",
        )
        axes[0, column].legend(fontsize=8)
        axes[0, column].set_title(f"Calibration d={d}, p={p:g}; {n:,} paired shots", weight="bold")
        axes[1, column].plot(
            x,
            [r["maximum_iterations"] for r in series],
            "s--",
            color="#9aa4b0",
            label="Maximum permitted",
        )
        axes[1, column].plot(
            x,
            [r["iterations_mean"] for r in series],
            "o-",
            color="#2b526c",
            label="Actual mean iterations",
        )
        axes[1, column].plot(
            x,
            [r["iterations_median"] for r in series],
            ".-",
            color="#008578",
            label="Actual median iterations",
        )
        axes[1, column].legend(fontsize=8)
        for field, style, label in [
            ("median_us", "o-", "Median"),
            ("p95_us", "s--", "p95"),
            ("p99_us", "^:", "p99"),
        ]:
            axes[2, column].plot(x, [r[field] / 1000 for r in series], style, label=label, ms=4)
        axes[2, column].axhline(1, color="#be3535", lw=1.2)
        axes[2, column].legend(fontsize=8)
        for ax, label in zip(
            axes[:, column], ("LER (%)", "BP iterations / shot", "Decode time (ms)"), strict=True
        ):
            decorate(ax, label)
            ax.set_xticks(x)
        axes[0, column].set_xticklabels([])
        axes[1, column].set_xticklabels([])
        axes[2, column].set_xticklabels(short, rotation=55, ha="right", fontsize=8)
    fig.suptitle(
        "Iteration audit | limits, actual work, quality, latency", weight="bold", fontsize=17
    )
    fig.text(
        0.02,
        0.01,
        "Calibration results select candidates only; fresh validation is separate. "
        "129 serial timing probes/point. Zero-error markers show 95% upper bounds.\n"
        "R = total legs, T = iterations per additional leg, S = valid solutions sought. "
        "The base is 80 + 31×60 iterations, S=3; one control changes per ablation.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    save(fig, output, "iteration_ablation")


def paired_results(rows, output):
    result = []
    for row in rows:
        if row["profile"] == "pymatching":
            continue
        directory = output / "validation" / f"d{row['distance']}_p{row['noise']:g}"
        with np.load(directory / "samples.npz") as saved:
            ref = np.any(saved["pymatching"] != saved["truth"], axis=1)
        with np.load(directory / f"{row['profile']}_raw.npz") as saved:
            candidate = saved["logical_failures"]
        wins = int(np.count_nonzero(ref & ~candidate))
        losses = int(np.count_nonzero(~ref & candidate))
        p = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
        item = {
            "profile": row["profile"],
            "distance": row["distance"],
            "noise": row["noise"],
            "wins_vs_pymatching": wins,
            "losses_vs_pymatching": losses,
            "exact_paired_p_value": p,
            "p_value_scope": "exploratory two-sided exact discordance test",
        }
        if row["profile"] == "xyz_accuracy":
            item["primary_four_point_bonferroni_p_value"] = min(1.0, 4 * p)
        result.append(item)
    write_json(output / "paired_results.json", result)
    return result


def report(rows, calibration, paper, paired, output):
    chunks = [
        "<!doctype html><meta charset='utf-8'><title>IBM Relay-BP iteration audit</title>",
        "<style>body{font:16px system-ui;max-width:1500px;margin:35px auto;padding:0 20px}",
        "p{max-width:1150px;line-height:1.6}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}",
        "td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}"
        "th{background:#edf3f5}</style>",
        "<h1>IBM Relay-BP: iteration audit and independent validation</h1>",
        "<p>The earlier 1 ms profile used a small ensemble and a memory-basis marginal. "
        "Its tuning did not adequately explore per-leg iteration limits or paper-scale ensembles. "
        "This audit varies first-leg iterations, relay-leg iterations, total legs, "
        "solution targets "
        "and stopping separately; it also compares full XYZ decoding with the smaller graph. "
        "The pinned IBM native implementation remains unchanged.</p>",
        "<p>The IBM authors report surface-code accuracy comparable to matching, with a slower "
        "CPU implementation. Their much larger LER gains concern bicycle codes "
        "versus BP+OSD+CS-10. "
        "A valid BP correction satisfies the syndrome but can have the wrong logical action. "
        "Convergence is therefore reported separately from LER. "
        "<a href='https://arxiv.org/html/2506.01779v2'>IBM paper, Sections IV–VI</a>.</p>",
        "<h2>Verified iteration semantics</h2><p>num_sets excludes the first leg. "
        "The cap is pre_iter + num_sets × set_max_iter. "
        "Native actual iterations are recorded per shot. "
        "The native result field max_iter belongs to its returned leg "
        "and must not be read as total work. "
        "Each leg exits on convergence; nconv stopping exits after S valid solutions, "
        "including repeats. "
        "The all mode exhausts legs while retaining per-leg convergence. Direct native tests use "
        "an impossible syndrome to prove exact cap execution, and a uniquely solvable graph to "
        "prove repeated-solution early stopping.</p>",
        "<table><tr><th>Profile</th><th>Graph</th><th>First iterations</th><th>Additional legs</th>"
        "<th>Iterations/leg</th><th>S</th><th>Total BP cap</th></tr>",
    ]
    for name in ORDER[:-1]:
        row = next(r for r in rows if r["profile"] == name)
        c = row["config"]
        chunks.append(
            f"<tr><td>{LABELS[name]}</td><td>{c['detector_mode']}</td>"
            f"<td>{c['pre_iter']}</td><td>{c['num_sets']}</td><td>{c['set_max_iter']}</td>"
            f"<td>{c['stop_nconv']}</td><td>{row['maximum_iterations']:,}</td></tr>"
        )
    chunks += [
        "</table><p>Full XYZ accuracy uses gamma0=.35, gamma range [-.254,.985], alpha=None "
        "(native unscaled min-sum), exact equivalent-column merging and all detector information. "
        "The three memory-basis profiles use alpha=.9, gamma0=.35, range [-.25,.66]; "
        "discarding the other check basis sacrifices cross-basis conditioning. "
        "No matching or Gaussian-repair fallback is present. Alpha=0 has separate native "
        "semantics: an exponential ramp, rather than the alpha=None default of 1.</p>",
        "<img src='accuracy_validation.png' alt='Fresh validation quality and latency'>",
        "<h2>Held-out comparison</h2><p>All profiles receive identical full-circuit samples, "
        "with seeds 1229001–1229004 disjoint from calibration. Profiles were frozen beforehand. "
        "LER includes every unconverged shot. Timings are normal packed one-shot calls, "
        "serialized with rotating decoder order, after 64 warmup calls. 1,024 probes are used "
        "at d5/d7/d9 and 512 at d11. Instrumented detailed decoding is used for convergence "
        "and iteration counts, not for latency. Median alone controls the 1 ms status. "
        "Memory/build caps stay enforced. "
        "The d11 512-shot diagnostic cannot resolve small LERs.</p>",
        "<table><tr><th>d, p</th><th>Profile</th><th>Errors/shots</th><th>LER [95% CI]</th>"
        "<th>Nonconv.</th><th>Actual mean / median iterations</th><th>Median ms</th>"
        "<th>p95 ms</th><th>p99 ms</th><th>Median budget</th></tr>",
    ]
    for row in rows:
        iterations = (
            f"{row['iterations_mean']:.2f} / {row['iterations_median']:g}"
            if "iterations_mean" in row
            else "n/a"
        )
        chunks.append(
            f"<tr><td>{row['distance']}, {row['noise']:g}</td><td>{LABELS[row['profile']]}</td>"
            f"<td>{row['errors']}/{row['shots']}</td><td>{row['logical_error_rate'] * 100:.4g}% "
            f"[{row['ci95_low'] * 100:.3g}%, {row['ci95_high'] * 100:.3g}%]</td>"
            f"<td>{row.get('nonconverged', 'n/a')}</td><td>{iterations}</td>"
            + "".join(f"<td>{row[k] / 1000:.4g}</td>" for k in ("median_us", "p95_us", "p99_us"))
            + f"<td>{'pass' if row['latency_pass'] else 'over 1 ms'}</td></tr>"
        )
    chunks += [
        "</table><h2>Paired accuracy comparisons</h2><p>A win means the candidate is correct "
        "on a shot where PyMatching fails; a loss means the reverse. Exact two-sided discordance "
        "tests are descriptive and exploratory for other candidates. The primary full XYZ "
        "comparison also reports a Bonferroni adjustment across its four validation points.</p>",
        "<table><tr><th>d, p</th><th>Profile</th><th>Wins</th><th>Losses</th>"
        "<th>Exact paired p</th><th>Primary four-point adjusted p</th></tr>",
    ]
    for r in paired:
        adjusted = r.get("primary_four_point_bonferroni_p_value")
        chunks.append(
            f"<tr><td>{r['distance']}, {r['noise']:g}</td><td>{LABELS[r['profile']]}</td>"
            f"<td>{r['wins_vs_pymatching']}</td><td>{r['losses_vs_pymatching']}</td>"
            f"<td>{r['exact_paired_p_value']:.4g}</td>"
            f"<td>{adjusted if adjusted is not None else '—'}</td></tr>"
        )
    chunks += [
        "</table><img src='iteration_ablation.png' alt='Controlled iteration ablation'>",
        "<h2>Calibration tables</h2><p>These samples selected candidates and are not independent "
        "validation. A favorable small-sample calibration difference is insufficient to replace "
        "the latency default. Increasing a cap often leaves the median unchanged because most "
        "shots already stop after S solutions; per-leg length also changes the relay trajectory. "
        "A larger iteration cap does not guarantee monotonically lower LER.</p>",
    ]
    for title, data in [
        ("Controlled iteration sweep", calibration),
        ("Paper search settings / graph strategy", paper),
    ]:
        chunks.append(
            f"<h3>{title}</h3><table><tr><th>d,p</th><th>Profile</th><th>Cap</th>"
            "<th>Errors/shots</th><th>Nonconv.</th><th>Actual mean iterations</th>"
            "<th>Median ms</th><th>p95 ms</th><th>p99 ms</th></tr>"
        )
        for r in data:
            chunks.append(
                f"<tr><td>{r['distance']},{r['noise']:g}</td><td>{html.escape(r['profile'])}</td>"
                f"<td>{r['maximum_iterations']:,}</td><td>{r['errors']}/{r['shots']}</td>"
                f"<td>{r['nonconverged']}</td><td>{r['iterations_mean']:.2f}</td>"
                + "".join(f"<td>{r[k] / 1000:.4g}</td>" for k in ("median_us", "p95_us", "p99_us"))
                + "</tr>"
            )
        chunks.append("</table>")
    chunks += [
        "<p>The paper circuit/noise conditions differ from this contract's Stim preset, so "
        "this is not a paper reproduction. Historical d31 extrapolations remain tied to the "
        "old frozen 32-leg profile; they do not predict this accuracy configuration. "
        "No new high-distance LER forecast is inferred from these limited validation points.</p>",
        "<p><a href='validation_results.csv'>Fresh validation CSV</a> · "
        "<a href='validation_manifest.json'>Frozen profiles, source/native hashes and seeds</a> · "
        "<a href='paired_results.json'>Paired comparison data</a> · "
        "<a href='verification.json'>Audit verification</a></p>",
    ]
    (output / "report.html").write_text("\n".join(chunks), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("past_experiments/decoder_testing/results/relay-iteration-audit-20261121"),
    )
    output = parser.parse_args().directory.resolve()
    rows = json.loads((output / "validation_results.json").read_text())
    if len(rows) != 20:
        raise ValueError("all four validation points and five profiles must complete")
    calibration = json.loads((output / "iterations_results.json").read_text())
    paper = json.loads((output / "paper_results.json").read_text())
    paired = paired_results(rows, output)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    validation_figure(rows, output)
    iteration_figure(calibration, output)
    report(rows, calibration, paper, paired, output)
    with (output / "paired_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in paired for k in r))
        )
        writer.writeheader()
        writer.writerows(paired)
    print(output / "report.html")
    print(json.dumps([r for r in paired if r["profile"] == "xyz_accuracy"], indent=2))


if __name__ == "__main__":
    main()

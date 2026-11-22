"""Plot measurements through d15 and explicitly conditional trends through d31."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from distance_projection import binomial_fit, binomial_predict, power_fit, power_predict
from plot_tuned_relay import COLORS, LABELS, ler_curve, save, setup

from qecbench.storage import write_json

FIELDS = ["single_shot_p50_us", "single_shot_p95_us", "single_shot_p99_us"]


def inputs(output):
    manifest = json.loads((output / "manifest.json").read_text())["identity"]
    baseline = Path(manifest["baseline"])
    prior = json.loads((baseline / "results.json").read_text())
    extension = json.loads((output / "extension_results.json").read_text())
    if {row["distance"] for row in extension} != {11, 13, 15}:
        raise ValueError("extension must finish d11/13/15 before projection")
    points = {
        (r["decoder"], r["distance"]): r
        for r in prior
        if r["noise"] == 0.003 and r["distance"] <= 15 and "shots" in r.get("measured", {})
    }
    points = json.loads(json.dumps(list(points.values())))
    by_key = {(r["decoder"], r["distance"]): r for r in points}
    for row in extension:
        key = row["decoder"], row["distance"]
        if row["decoder"] == "relay_bp":
            by_key[key] = row
        elif "measured" in row:
            # Keep the more precise 20k quality result, but use contemporaneous timing.
            previous = by_key[key]
            latency = json.loads(
                (output / f"d{row['distance']}" / "latency_summary.json").read_text()
            )
            previous["measured"].update(latency[row["decoder"]])
            previous["quality_source"] = str(
                baseline / "p0.003" / f"d{row['distance']}" / "quality"
            )
            previous["timing_source"] = str(output / f"d{row['distance']}")
    return list(by_key.values()), manifest


def models(points):
    result = {}
    for name in ("relay_bp", "pymatching"):
        series = sorted(
            [r for r in points if r["decoder"] == name and 7 <= r["distance"] <= 15],
            key=lambda r: r["distance"],
        )
        distances = [r["distance"] for r in series]
        quality = [r["measured"] for r in series]
        result[name] = {
            "ler": binomial_fit(
                distances, [m["errors"] for m in quality], [m["shots"] for m in quality]
            ),
            "time": {},
        }
        for field in FIELDS:
            result[name]["time"][field] = power_fit(distances, [m[field] / 1000 for m in quality])
        recent = [r for r in series if r["distance"] >= 11]
        try:
            result[name]["ler_recent_sensitivity"] = binomial_fit(
                [r["distance"] for r in recent],
                [r["measured"]["errors"] for r in recent],
                [r["measured"]["shots"] for r in recent],
            )
        except ValueError as error:
            result[name]["ler_recent_unavailable"] = str(error)
    return result


def projection_rows(fits):
    rows = []
    for name, fit in fits.items():
        for d in range(17, 32, 2):
            mean, low, high = binomial_predict(fit["ler"], [d])
            row = {
                "decoder": name,
                "distance": d,
                "data_type": "trend_projection",
                "budget_admitted": False,
                "ler": float(mean[0]),
                "ler_fit95_low": float(low[0]),
                "ler_fit95_high": float(high[0]),
                "reason": "unmeasured and beyond configured resource cap",
            }
            if "ler_recent_sensitivity" in fit:
                mean, low, high = binomial_predict(fit["ler_recent_sensitivity"], [d])
                row.update(
                    ler_recent=float(mean[0]),
                    ler_recent_fit95_low=float(low[0]),
                    ler_recent_fit95_high=float(high[0]),
                )
            previous = 0
            for field, label in zip(FIELDS, ("median", "p95", "p99"), strict=True):
                value, low, high = power_predict(fit["time"][field], [d])
                ordered = max(float(value[0]), previous)
                row[f"{label}_ms"] = ordered
                row[f"{label}_raw_fit_ms"] = float(value[0])
                row[f"{label}_fit95_low_ms"] = float(low[0])
                row[f"{label}_fit95_high_ms"] = float(high[0])
                previous = ordered
            row["quantile_order_adjusted"] = any(
                row[f"{label}_ms"] != row[f"{label}_raw_fit_ms"]
                for label in ("median", "p95", "p99")
            )
            rows.append(row)
    return rows


def figure(points, fits, output, projected):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
    for name in LABELS:
        series = sorted([r for r in points if r["decoder"] == name], key=lambda r: r["distance"])
        ler_curve(axes[0], series, "distance", COLORS[name], LABELS[name])
        timed = [r for r in series if FIELDS[0] in r.get("measured", {})]
        for field, style in zip(FIELDS, ("-", "--", ":"), strict=True):
            axes[1].plot(
                [r["distance"] for r in timed],
                [r["measured"][field] / 1000 for r in timed],
                color=COLORS[name],
                linestyle=style,
                marker="o" if field == FIELDS[0] else None,
                lw=2 if field == FIELDS[0] else 1.2,
                ms=4,
            )
        outside = [r for r in series if not r["measured"]["latency_pass"]]
        if outside:
            axes[0].scatter(
                [r["distance"] for r in outside],
                [
                    r["measured"]["logical_error_rate"]
                    if r["measured"]["errors"]
                    else r["measured"]["ci95_high"]
                    for r in outside
                ],
                facecolors="white",
                edgecolors=COLORS[name],
                s=48,
                zorder=6,
            )
            axes[1].scatter(
                [r["distance"] for r in outside],
                [r["measured"][FIELDS[0]] / 1000 for r in outside],
                marker="x",
                color="#bd3535",
                s=65,
                linewidths=1.8,
                zorder=6,
            )
    if projected:
        forecast = projection_rows(fits)
        grid = np.linspace(15, 31, 150)
        lows = []
        for name, fit in fits.items():
            mean, low, high = binomial_predict(fit["ler"], grid)
            axes[0].plot(grid, mean, "--", color=COLORS[name], lw=2)
            axes[0].fill_between(grid, low, high, color=COLORS[name], alpha=0.12)
            lows.extend(low)
            if "ler_recent_sensitivity" in fit:
                recent, _, _ = binomial_predict(fit["ler_recent_sensitivity"], grid)
                axes[0].plot(grid, recent, ":", color=COLORS[name], lw=1.7)
                lows.extend(recent)
            times = [power_predict(fit["time"][field], grid)[0] for field in FIELDS]
            times[1] = np.maximum(times[1], times[0])
            times[2] = np.maximum(times[2], times[1])
            for index, style in enumerate(("--", "-.", ":")):
                axes[1].plot(
                    grid,
                    times[index],
                    linestyle=style,
                    color=COLORS[name],
                    lw=2 if index == 0 else 1.2,
                )
            _, low, high = power_predict(fit["time"][FIELDS[0]], grid)
            axes[1].fill_between(grid, low, high, color=COLORS[name], alpha=0.12)
            endpoint = next(r for r in forecast if r["distance"] == 31 and r["decoder"] == name)
            time_label = (
                f"{endpoint['median_ms'] / 1000:.2g} s"
                if endpoint["median_ms"] >= 1000
                else f"{endpoint['median_ms']:.2g} ms"
            )
            axes[1].annotate(
                time_label,
                (31, endpoint["median_ms"]),
                xytext=(-5, 8),
                textcoords="offset points",
                ha="right",
                color=COLORS[name],
                fontsize=9,
                weight="bold",
            )
        for ax in axes:
            ax.axvspan(15.5, 31.5, color="#64748b", alpha=0.055, zorder=-5)
            ax.axvline(15.5, color="#8d99a6", lw=1, linestyle="--")
            ax.text(
                0.64,
                0.98,
                "TREND PROJECTIONS",
                transform=ax.transAxes,
                va="top",
                color="#5e6878",
                fontsize=10,
                weight="bold",
            )
        axes[0].set_ylim(max(1e-16, float(min(lows)) / 2), 0.5)
    else:
        axes[0].set_ylim(5e-6, 0.1)
    axes[1].axhline(1, color="#bd3535", lw=1.3)
    setup(axes[0], "Code distance d (rounds = d)", "Logical error probability / memory experiment")
    setup(axes[1], "Code distance d (rounds = d)", "Isolated-shot decode time (ms)")
    axes[0].set_title("Logical error rate", loc="left", weight="bold")
    axes[1].set_title("Median, p95, p99 | 1 ms median budget", loc="left", weight="bold")
    axes[0].legend(fontsize=9, loc="lower left" if projected else "upper right")
    from matplotlib.lines import Line2D

    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#333333", marker="o", label="Median measured"),
            Line2D([0], [0], color="#333333", linestyle="--", label="p95 measured"),
            Line2D([0], [0], color="#333333", linestyle=":", label="p99 measured"),
            Line2D(
                [0], [0], color="#bd3535", marker="x", linestyle="none", label="Median over budget"
            ),
        ],
        fontsize=8.5,
        loc="upper left",
    )
    for ax in axes:
        ax.set_xlim(1.5, 31.7 if projected else 15.6)
        ax.set_xticks(
            [2, 3, 7, 11, 15, 19, 23, 27, 31] if projected else [2, 3, 5, 7, 9, 11, 13, 15]
        )
    title = (
        "Measurements through d15 + trend projections to d31"
        if projected
        else "Measured distance sweep through d15"
    )
    fig.suptitle(f"{title} | circuit noise p = 0.003", weight="bold", fontsize=15)
    note = (
        (
            "Dashed forecast: logit-linear LER fit d7–15; dotted LER: sensitivity fit d11–15. "
            "Latency: power-law fits; bands are conditional 95% fit intervals.\n"
            "Extrapolations are unvalidated; no budget admission beyond measured caps. "
            "LUT: one feasible distance, no distance projection."
        )
        if projected
        else (
            "LER: 20,000 shots/point; Relay d11/13/15: 4,096 fresh diagnostic shots each. "
            "95% Wilson intervals; hollow LER points / red ×: over 1 ms median.\n"
            "Latency: 2,048 probes through d11; 1,024 at d13/d15. "
            "Rotated Z memory, rounds=d; frozen native IBM profile. LUT stops at d3 resource cap."
        )
    )
    resource_path = output / "resources.json"
    if projected and resource_path.exists():
        capped = [
            r
            for r in json.loads(resource_path.read_text())
            if r["distance"] == 17
            and r["decoder"] in ("ibm_relay_bp", "pymatching")
            and not r["accepted"]
        ]
        if len(capped) == 2:
            note += "\nBoth graph models exceed the conservative 256 MiB resource cap at d17."
    fig.text(0.02, 0.015, note, fontsize=8.5, va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.subplots_adjust(left=0.08)
    save(fig, output, "projection_to_d31" if projected else "measured_through_d15")


def write_report(points, fits, forecast, output):
    endpoints = [r for r in forecast if r["distance"] == 31]
    parts = [
        "<!doctype html><meta charset='utf-8'><title>d15 measurements and d31 trends</title>",
        "<style>body{font:16px system-ui;max-width:1450px;margin:35px auto;padding:0 20px;}",
        "p{max-width:1100px;line-height:1.6}img{width:100%}table{border-collapse:collapse}",
        "td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}</style>",
        "<h1>Measured through d15; conditional projections through d31</h1>",
        "<p>Rotated Z surface memory, circuit noise p=.003, rounds=distance. "
        "Frozen IBM native core and tuned adapter settings are unchanged. "
        "The 1 ms median budget remains in force: Relay d11/13/15 measurements "
        "are diagnostics beyond its d11 latency stop, not newly admitted configurations. "
        "Memory/build caps remain enforced; no large-distance decoder is allocated.</p>",
        "<img src='measured_through_d15.png' alt='Measurements through distance 15'>",
        "<p>Relay extension uses 4,096 fresh circuit shots per distance, with paired "
        "PyMatching validation. Plotted PyMatching LER retains the more precise prior "
        "20,000-shot result; d13/d15 timing is contemporaneous with Relay. "
        "Quality CIs are two-sided 95% Wilson intervals; zero failures are upper-bound triangles. "
        "Raw timing vectors contain 2,048 calls through d11 and 1,024 at d13/d15, after "
        "64 warmup calls. Input conversion is included; construction/sampling are excluded.</p>",
        "<img src='projection_to_d31.png' alt='Trend projections through distance 31'>",
        "<h2>Projection assumptions and limits</h2>",
        "<p>Primary LER fit uses binomial likelihood with logit(LER) linear in d, fitted "
        "on d=7,9,11,13,15. Zero-error counts enter the likelihood directly. "
        "A separate dotted sensitivity fit uses only d=11,13,15. Neither slope is forced "
        "downward. Latency fits log(time) linear in log(d), separately for median, p95 and p99. "
        "Projected quantiles are ordered if independent fits cross; raw fits are retained in CSV. "
        "Bands describe parameter uncertainty conditional on these models; they do not cover "
        "unknown convergence floors, changing distance regimes, host load or hardware changes. "
        "These are descriptive extrapolations, not validated LER or hardware forecasts. "
        "No LUT distance projection is fitted from its single feasible point.</p>",
        "<table><tr><th>Decoder</th><th>d31 LER trend [fit interval]</th>"
        "<th>d31 recent-distance sensitivity</th><th>Median ms</th>"
        "<th>p95 ms</th><th>p99 ms</th></tr>",
    ]
    for row in endpoints:
        parts.append(
            f"<tr><td>{LABELS[row['decoder']]}</td>"
            f"<td>{row['ler']:.3g} [{row['ler_fit95_low']:.3g}, {row['ler_fit95_high']:.3g}]</td>"
            f"<td>{row.get('ler_recent', float('nan')):.3g}</td>"
            + "".join(f"<td>{row[f'{metric}_ms']:.4g}</td>" for metric in ("median", "p95", "p99"))
            + "</tr>"
        )
    parts.append("</table><h2>Fit diagnostics</h2><ul>")
    for name, fit in fits.items():
        ler = fit["ler"]
        parts.append(
            f"<li>{LABELS[name]}: binomial deviance {ler['deviance']:.3g} "
            f"on {ler['degrees_of_freedom']} degrees of freedom "
            f"(asymptotic p={ler['deviance_p_value']:.3g}). "
            "A small p indicates that the primary trend misses observed structure.</li>"
        )
    parts.append(
        "</ul><h2>Measured data</h2><table><tr><th>d</th><th>Decoder</th>"
        "<th>Errors/shots</th><th>LER [95% CI]</th><th>Median ms</th><th>p95 ms</th>"
        "<th>p99 ms</th><th>Budget status</th></tr>"
    )
    for row in sorted(points, key=lambda r: (r["distance"], r["decoder"])):
        m = row["measured"]
        parts.append(
            f"<tr><td>{row['distance']}</td><td>{LABELS[row['decoder']]}</td>"
            f"<td>{m['errors']}/{m['shots']}</td><td>{m['logical_error_rate']:.4g} "
            f"[{m['ci95_low']:.3g}, {m['ci95_high']:.3g}]</td>"
            + "".join(f"<td>{m[field] / 1000:.4g}</td>" for field in FIELDS)
            + f"<td>{'pass' if m['latency_pass'] else 'over median budget'}</td></tr>"
        )
    parts.append(
        "</table><p><a href='projections.json'>Model coefficients and counts</a> · "
        "<a href='projection_results.csv'>Projection CSV</a> · "
        "<a href='extension_results.csv'>Diagnostic measurements CSV</a> · "
        "<a href='manifest.json'>Measurement provenance</a> · "
        "<a href='verification.json'>Verification</a></p>"
    )
    resource_path = output / "resources.json"
    if resource_path.exists():
        parts.append(
            "<h2>Resource preflight</h2><p>Conservative model, not measured RSS. "
            "Relay uses the full unprojected/unmerged graph as an upper bound. "
            "Decoder graph/table allocation is skipped for rejected points.</p><ul>"
        )
        for row in json.loads(resource_path.read_text()):
            peak = row.get("peak_working_bytes")
            label = "unknown" if peak is None else f"{peak / 1024**2:,.1f} MiB"
            parts.append(
                f"<li>d{row['distance']} {html.escape(row['decoder'])}: {label}; "
                f"{'accepted' if row['accepted'] else html.escape('; '.join(row['reasons']))}</li>"
            )
        parts.append("</ul>")
    (output / "report.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("past_experiments/decoder_testing/results/relay-d15-projection-d31-20261121"),
    )
    args = parser.parse_args()
    output = args.directory.resolve()
    points, _ = inputs(output)
    fits = models(points)
    forecast = projection_rows(fits)
    write_json(output / "combined_measurements.json", points)
    write_json(
        output / "projections.json",
        {
            "models": fits,
            "projection_rows": forecast,
            "measured_max_distance": 15,
            "projected_distances": list(range(17, 32, 2)),
            "budget_admission_from_projection": False,
            "quantile_order": "max(median,p95) and max(ordered_p95,p99); raw fits retained",
            "limitations": "conditional trend fits, not validated large-distance performance; "
            "fit intervals exclude unknown systematic/model changes",
        },
    )
    with (output / "projection_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in forecast for k in r))
        )
        writer.writeheader()
        writer.writerows(forecast)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure(points, fits, output, projected=False)
    figure(points, fits, output, projected=True)
    write_report(points, fits, forecast, output)
    shutil.copy2(__file__, output / "plot_runner_snapshot.py")
    shutil.copy2(
        Path(__file__).with_name("distance_projection.py"), output / "fit_source_snapshot.py"
    )
    write_json(
        output / "render_manifest.json",
        {
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fit_source_sha256": hashlib.sha256(
                Path(__file__).with_name("distance_projection.py").read_bytes()
            ).hexdigest(),
            "extension_results_sha256": hashlib.sha256(
                (output / "extension_results.json").read_bytes()
            ).hexdigest(),
            "combined_measurements_sha256": hashlib.sha256(
                (output / "combined_measurements.json").read_bytes()
            ).hexdigest(),
        },
    )
    print(output / "report.html")
    print(json.dumps([r for r in forecast if r["distance"] == 31], indent=2))


if __name__ == "__main__":
    main()

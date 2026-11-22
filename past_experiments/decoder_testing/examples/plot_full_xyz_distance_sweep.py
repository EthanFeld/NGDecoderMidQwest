"""Scientific plots and conditional d31 trends for a bounded full-XYZ sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

if __package__:
    from .distance_projection import binomial_fit, binomial_predict, power_fit, power_predict
else:
    from distance_projection import binomial_fit, binomial_predict, power_fit, power_predict

from qecbench.storage import write_json

NAMES = {"relay_bp": "Full XYZ Relay-BP", "pymatching": "PyMatching"}
COLORS = {"relay_bp": "#c54d47", "pymatching": "#2867b5"}
FIELDS = [
    ("median_us", "median_resolved", "Median", "-"),
    ("p95_us", "p95_resolved", "p95", "--"),
    ("p99_us", "p99_resolved", "p99", ":"),
]


def save(fig, output, name):
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rows_for(points, noise, name):
    return sorted(
        [{**p, "m": p["decoders"][name]} for p in points if p["noise"] == noise],
        key=lambda r: r["distance"],
    )


def fit_ler(rows, minimum):
    selected = [r for r in rows if r["distance"] >= minimum and r["m"].get("quality_complete")]
    if len(selected) < 3:
        raise ValueError("fewer than three complete distances")
    if sum(r["m"]["errors"] > 0 for r in selected) < 2:
        raise ValueError("errors observed at fewer than two distances; slope unresolved")
    model = binomial_fit(
        [r["distance"] for r in selected],
        [r["m"]["errors"] for r in selected],
        [r["m"]["shots"] for r in selected],
    )
    model["last_measured_distance"] = max(
        r["distance"] for r in rows if r["m"].get("quality_complete")
    )
    model["fit_support_warning"] = sum(model["errors"]) < 20
    model["weak_distance_support"] = len(selected) < 4 or max(r["distance"] for r in selected) < 11
    return model


def build_models(points, noises):
    models = {}
    forecasts = []
    for p in noises:
        models[f"{p:g}"] = {}
        for name in NAMES:
            series = rows_for(points, p, name)
            entry = {"ler_attempts": [], "time": {}, "unavailable": {}}
            for minimum in [7, 3]:
                try:
                    entry["ler"] = fit_ler(series, minimum)
                    entry["ler"]["minimum_distance_rule"] = minimum
                    break
                except ValueError as error:
                    entry["ler_attempts"].append(
                        {"minimum_distance": minimum, "reason": str(error)}
                    )
            if "ler" not in entry:
                entry["unavailable"]["ler"] = (
                    "distance trend not identified by complete error counts"
                )
            try:
                entry["ler_recent_sensitivity"] = fit_ler(series, 11)
            except ValueError as error:
                entry["unavailable"]["ler_recent_sensitivity"] = str(error)
            for field, resolution, label, _ in FIELDS:
                reliable = [
                    r for r in series if r["m"].get(resolution) and r["m"].get(field) is not None
                ]
                recent = [r for r in reliable if r["distance"] >= 7]
                chosen = recent if len(recent) >= 3 else reliable
                if len(chosen) < 3:
                    entry["unavailable"][field] = (
                        "fewer than three distances with adequate quantile sampling"
                    )
                    continue
                fit = power_fit(
                    [r["distance"] for r in chosen], [r["m"][field] / 1000 for r in chosen]
                )
                fit["qualified_metric"] = label
                fit["units"] = "ms"
                fit["latency_samples"] = [r["m"]["latency_samples"] for r in chosen]
                fit["last_measured_distance"] = max(
                    r["distance"] for r in series if field in r["m"] and r["m"][field] is not None
                )
                fit["last_resolved_distance"] = max(r["distance"] for r in reliable)
                fit["weak_distance_support"] = (
                    len(chosen) < 4 or max(r["distance"] for r in chosen) < 11
                )
                entry["time"][field] = fit
            models[f"{p:g}"][name] = entry
            for d in range(17, 32, 2):
                if "ler" in entry:
                    mean, low, high = binomial_predict(entry["ler"], [d])
                    forecasts.append(
                        {
                            "noise": p,
                            "decoder": name,
                            "distance": d,
                            "metric": "ler",
                            "units": "probability",
                            "kind": "conditional_trend_projection",
                            "value": float(mean[0]),
                            "fit95_low": float(low[0]),
                            "fit95_high": float(high[0]),
                            "fit_distances": entry["ler"]["training_distances"],
                            "weak_distance_support": entry["ler"]["weak_distance_support"],
                            "budget_admitted": False,
                        }
                    )
                for field, fit in entry["time"].items():
                    with np.errstate(over="ignore"):
                        mean, low, high = power_predict(fit, [d])
                    if not np.all(np.isfinite([mean[0], low[0], high[0]])):
                        continue
                    forecasts.append(
                        {
                            "noise": p,
                            "decoder": name,
                            "distance": d,
                            "metric": field,
                            "kind": "conditional_trend_projection",
                            "value": float(mean[0]),
                            "fit95_low": float(low[0]),
                            "fit95_high": float(high[0]),
                            "units": "ms",
                            "fit_distances": fit["training_distances"],
                            "weak_distance_support": fit["weak_distance_support"],
                            "budget_admitted": False,
                        }
                    )
    # Independent percentile fits may cross beyond measurements. Preserve raw fits
    # and order their estimates and bounds; never infer an unavailable percentile.
    for p in noises:
        for name in NAMES:
            for d in range(17, 32, 2):
                previous = np.zeros(3)
                for field, _, _, _ in FIELDS:
                    row = next(
                        (
                            r
                            for r in forecasts
                            if r["noise"] == p
                            and r["decoder"] == name
                            and r["distance"] == d
                            and r["metric"] == field
                        ),
                        None,
                    )
                    if row:
                        raw = np.array([row["value"], row["fit95_low"], row["fit95_high"]])
                        ordered = np.maximum(previous, raw)
                        row["raw_fit_value"], row["raw_fit95_low"], row["raw_fit95_high"] = (
                            raw.tolist()
                        )
                        row["value"], row["fit95_low"], row["fit95_high"] = ordered.tolist()
                        row["quantile_order_adjusted"] = bool(np.any(ordered != raw))
                        row["interval_qualification"] = (
                            "ordered envelope of marginal fit bounds; not simultaneous coverage"
                        )
                        previous = ordered
    return models, forecasts


def ordered_time_curves(entry, distances):
    """Order fitted percentile means without inventing any unavailable tail."""
    result = {}
    previous = np.zeros((3, len(distances)))
    for field, _, _, _ in FIELDS:
        if field not in entry["time"]:
            continue
        with np.errstate(over="ignore"):
            mean, low, high = power_predict(entry["time"][field], distances)
        ordered = np.maximum(np.array([mean, low, high]), previous)
        result[field] = tuple(ordered)
        previous = ordered
    return result


def quality_curve(ax, series, xfield, color, label=None):
    xscale = 100 if xfield == "noise" else 1
    positives = [r for r in series if r["m"].get("quality_complete") and r["m"]["errors"]]
    if positives:
        x = [r[xfield] * xscale for r in positives]
        y = np.asarray([r["m"]["logical_error_rate"] for r in positives])
        low = np.asarray([r["m"]["ci95_low"] for r in positives])
        high = np.asarray([r["m"]["ci95_high"] for r in positives])
        ax.errorbar(
            x,
            y,
            yerr=[y - low, high - y],
            color=color,
            marker="o",
            ms=4,
            lw=1.6,
            capsize=2.5,
            label=label,
        )
    zero = [r for r in series if r["m"].get("quality_complete") and not r["m"]["errors"]]
    if zero:
        ax.scatter(
            [r[xfield] * xscale for r in zero],
            [r["m"]["ci95_high"] for r in zero],
            marker="v",
            facecolors="white",
            edgecolors=color,
            s=35,
            zorder=5,
            label=label if not positives else None,
        )


def axes_style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="major", alpha=0.18)


def distance_figures(points, models, noises, output, projected):
    for metric in ("ler", "time"):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
        for ax, p in zip(axes.flat, noises, strict=True):
            for name in NAMES:
                color = COLORS[name]
                series = rows_for(points, p, name)
                entry = models[f"{p:g}"][name]
                if metric == "ler":
                    quality_curve(ax, series, "distance", color, NAMES[name])
                    if projected and "ler" in entry:
                        fit = entry["ler"]
                        grid = np.linspace(fit["last_measured_distance"], 31, 120)
                        mean, low, high = binomial_predict(fit, grid)
                        ax.plot(grid, mean, "--", color=color, lw=1.7)
                        ax.fill_between(grid, np.maximum(low, 1e-14), high, color=color, alpha=0.10)
                        if "ler_recent_sensitivity" in entry:
                            other, _, _ = binomial_predict(entry["ler_recent_sensitivity"], grid)
                            ax.plot(grid, other, ":", color=color, lw=1, alpha=0.7)
                    absent = [r for r in series if not r["m"].get("quality_complete")]
                    if absent:
                        ax.scatter(
                            [r["distance"] for r in absent],
                            [0.035] * len(absent),
                            transform=ax.get_xaxis_transform(),
                            marker="x",
                            color=color,
                            s=40,
                        )
                else:
                    for field, resolution, _, style in FIELDS:
                        valid = [r for r in series if r["m"].get(field) is not None]
                        x = [r["distance"] for r in series]
                        resolved = [
                            r["m"].get(field) / 1000
                            if r["m"].get(resolution) and r["m"].get(field) is not None
                            else np.nan
                            for r in series
                        ]
                        ax.plot(
                            x,
                            resolved,
                            linestyle=style,
                            color=color,
                            lw=1.7 if field == "median_us" else 1,
                        )
                        for row in valid:
                            resolved_point = row["m"].get(resolution, False)
                            marker = (
                                "o" if field == "median_us" else "s" if field == "p95_us" else "^"
                            )
                            ax.plot(
                                row["distance"],
                                row["m"][field] / 1000,
                                marker=marker,
                                ms=3.7,
                                markerfacecolor=color if resolved_point else "white",
                                markeredgecolor=color,
                                linestyle="none",
                                alpha=1 if resolved_point else 0.5,
                            )
                        if projected and field in entry["time"]:
                            fit = entry["time"][field]
                            grid = np.linspace(fit["last_resolved_distance"], 31, 120)
                            mean, low, high = ordered_time_curves(entry, grid)[field]
                            ax.plot(
                                grid,
                                mean,
                                color=color,
                                linestyle=style,
                                lw=1.7 if field == "median_us" else 1,
                                alpha=0.6,
                            )
                            if field == "median_us":
                                ax.fill_between(
                                    grid,
                                    np.maximum(low, 1e-4),
                                    np.minimum(high, 1e7),
                                    color=color,
                                    alpha=0.10,
                                )
                    absent = [r for r in series if "median_us" not in r["m"]]
                    if absent:
                        ax.scatter(
                            [r["distance"] for r in absent],
                            [0.035] * len(absent),
                            transform=ax.get_xaxis_transform(),
                            marker="x",
                            color=color,
                            s=40,
                        )
            if projected:
                ax.axvspan(15.5, 31.5, color="#64748b", alpha=0.05, zorder=-10)
                ax.axvline(15.5, color="#8a96a5", ls="--", lw=0.8)
            ax.set_title(f"Circuit p = {p * 100:g}%", weight="bold")
            ax.set_xlim(2.4, 31.5 if projected else 15.6)
            ax.set_xticks([3, 7, 11, 15, 19, 23, 27, 31] if projected else [3, 5, 7, 9, 11, 13, 15])
            if metric == "ler":
                axes_style(ax, "Code distance d (rounds=d)", "Logical error probability")
                ax.set_ylim(1e-12 if projected else 1e-5, 1)
                missing = [name for name in NAMES if "ler" not in models[f"{p:g}"][name]]
                if projected and missing:
                    ax.text(
                        0.02,
                        0.98,
                        "LER forecast unresolved: " + ", ".join(NAMES[n] for n in missing),
                        transform=ax.transAxes,
                        va="top",
                        fontsize=7.5,
                    )
                native_ler = models[f"{p:g}"]["relay_bp"].get("ler")
                if projected and native_ler and native_ler["weak_distance_support"]:
                    ax.text(
                        0.02,
                        0.27,
                        "Relay LER forecast: weak support\nfit d="
                        + ",".join(map(str, native_ler["training_distances"])),
                        transform=ax.transAxes,
                        va="top",
                        fontsize=7.7,
                        color="#7a3330",
                    )
            else:
                axes_style(ax, "Code distance d (rounds=d)", "Decode time (ms)")
                ax.axhline(1, color="#bc3030", lw=1.2)
                ax.set_ylim(0.001, 1e7 if projected else 2e5)
                native = rows_for(points, p, "relay_bp")
                gaps95 = [str(r["distance"]) for r in native if not r["m"].get("p95_resolved")]
                gaps99 = [str(r["distance"]) for r in native if not r["m"].get("p99_resolved")]
                ax.text(
                    0.03,
                    0.97,
                    "Relay sparse/unresolved tails:\n95: "
                    + (",".join(gaps95) or "none")
                    + "; 99: "
                    + (",".join(gaps99) or "none"),
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7.7,
                    color="#7a3330",
                )
                median_fit = models[f"{p:g}"]["relay_bp"]["time"].get("median_us")
                if projected and median_fit and median_fit["weak_distance_support"]:
                    ax.text(
                        0.48,
                        0.16,
                        "Relay median forecast: weak support\nfit d="
                        + ",".join(map(str, median_fit["training_distances"])),
                        transform=ax.transAxes,
                        va="top",
                        fontsize=7.7,
                        color="#7a3330",
                    )
        forecast_title = " + conditional d31 projections" if projected else ""
        fig.suptitle(
            ("LER" if metric == "ler" else "Median / p95 / p99 decode time")
            + " | full XYZ accuracy vs PyMatching"
            + forecast_title,
            fontsize=16,
            weight="bold",
        )
        handles = [Line2D([0], [0], color=COLORS[n], label=NAMES[n]) for n in NAMES]
        if metric == "time":
            handles += [
                Line2D([0], [0], color="#333", ls=style, label=label)
                for _, _, label, style in FIELDS
            ]
            handles += [Line2D([0], [0], color="#bc3030", label="1 ms median budget")]
            if projected:
                handles += [
                    Line2D(
                        [0],
                        [0],
                        color="#333",
                        alpha=0.5,
                        label="Forecast: lighter lines, same percentile styles",
                    )
                ]
        else:
            handles += [
                Line2D(
                    [0],
                    [0],
                    color="#555",
                    marker="v",
                    markerfacecolor="white",
                    linestyle="none",
                    label="Zero-error 95% upper bound",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#555",
                    marker="x",
                    linestyle="none",
                    label="Compute cap: no LER",
                ),
            ]
            if projected:
                handles += [
                    Line2D([0], [0], color="#555", ls="--", label="Conditional trend"),
                    Line2D([0], [0], color="#555", ls=":", label="Recent-window sensitivity"),
                ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=(4 if metric == "time" else 3) if projected else (6 if metric == "time" else 4),
            bbox_to_anchor=(0.5, 0.072),
            fontsize=8.5,
        )
        note = (
            "Frozen IBM native profile: 601 legs, S=5, cap 80+600×60=36,080 iterations. "
            "Rotated Z memory, rounds=d. No latency-based distance omission.\n"
        )
        if metric == "ler":
            note += (
                "95% Wilson intervals; open triangles = zero-error upper bounds; "
                "× at axis foot = compute cap, no LER.\n"
                "Rising zero-error bounds reflect smaller sample counts, not observed failures. "
            )
            note += (
                "Projection bands conditional on fitted trend; "
                "no guarantee against a new convergence floor."
            )
        else:
            note += (
                "Serial one-shot timing. Open symbols = sparse samples; "
                "absent tails stay unresolved. "
            )
            note += (
                "Only adequately sampled quantiles enter fits. "
                "Bands clipped to plotting range; full intervals in CSV."
            )
        if projected:
            note += (
                "\nShaded d17–31 region is extrapolation, beyond the configured 256 MiB graph cap."
            )
        fig.text(0.02, 0.012, note, fontsize=8.5, va="bottom")
        fig.tight_layout(rect=(0, 0.17 if projected else 0.14, 1, 0.94))
        save(fig, output, f"{metric}_distance_{'projected_d31' if projected else 'measured_d15'}")


def noise_figure(points, noises, output):
    distances = sorted({p["distance"] for p in points})
    colors = {d: plt.cm.viridis(i / (len(distances) - 1)) for i, d in enumerate(distances)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, name in enumerate(NAMES):
        for d in distances:
            series = sorted(
                [{**p, "m": p["decoders"][name]} for p in points if p["distance"] == d],
                key=lambda r: r["noise"],
            )
            quality_curve(axes[0, col], series, "noise", colors[d])
            x = [r["noise"] * 100 for r in series]
            y = [
                r["m"]["median_us"] / 1000 if r["m"].get("median_resolved") else np.nan
                for r in series
            ]
            axes[1, col].plot(x, y, "o-", color=colors[d], ms=4, label=f"d={d}")
            for row in series:
                if "median_us" in row["m"] and not row["m"].get("median_resolved"):
                    axes[1, col].plot(
                        row["noise"] * 100,
                        row["m"]["median_us"] / 1000,
                        marker="o",
                        linestyle="none",
                        markerfacecolor="white",
                        markeredgecolor=colors[d],
                        ms=4,
                        alpha=0.6,
                    )
        for row in range(2):
            axes_style(
                axes[row, col],
                "Circuit error rate p (%)",
                "Logical error probability" if row == 0 else "Median decode time (ms)",
            )
            axes[row, col].set_xscale("log")
            axes[row, col].xaxis.set_major_locator(FixedLocator([p * 100 for p in noises]))
            axes[row, col].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
            axes[row, col].xaxis.set_minor_locator(FixedLocator([]))
        axes[0, col].set_ylim(1e-5, 1)
        axes[1, col].axhline(1, color="#bc3030", lw=1.2)
        axes[0, col].set_title(NAMES[name], weight="bold")
    handles = [Line2D([0], [0], color=colors[d], marker="o", label=f"d={d}") for d in distances]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle(
        "Circuit-error-rate sweep | full XYZ accuracy profile through d15",
        weight="bold",
        fontsize=16,
    )
    fig.text(
        0.02,
        0.01,
        "Measured data only; native sample counts depend on independent runtime pilots. "
        "Triangles show zero-error upper bounds, not zero underlying LER.\n"
        "Open circles = sparse medians; missing points are compute-capped.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    save(fig, output, "circuit_error_rate_measured")


def coverage(points, noises, output):
    distances = sorted({p["distance"] for p in points})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ax, key, title in zip(
        axes,
        ("shots", "latency_samples"),
        ("Completed Relay quality shots", "Completed Relay timing probes"),
        strict=True,
    ):
        data = np.zeros((len(noises), len(distances)))
        for i, p in enumerate(noises):
            for j, d in enumerate(distances):
                point = next(r for r in points if r["noise"] == p and r["distance"] == d)
                data[i, j] = point["decoders"]["relay_bp"].get(key, 0)
        ax.imshow(
            np.log2(np.maximum(data, 1)),
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=15 if key == "shots" else 10,
        )
        for i in range(len(noises)):
            for j in range(len(distances)):
                ax.text(
                    j,
                    i,
                    f"{int(data[i, j]):,}" if data[i, j] else "capped",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white"
                    if np.log2(max(data[i, j], 1)) / (15 if key == "shots" else 10) > 0.58
                    else "#25364a",
                )
        ax.set_xticks(range(len(distances)), distances)
        ax.set_xlabel("Code distance")
        ax.set_yticks(range(len(noises)), [f"{p:g}" for p in noises])
        ax.set_ylabel("Circuit p")
        ax.set_title(title, weight="bold")
    fig.suptitle(
        "Bounded sampling coverage | no partial-quality rate imputation", weight="bold", fontsize=15
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, output, "sampling_coverage")


def noise_comparison(points, noises, output):
    """Put both decoders on the same error-rate axes for every distance."""
    distances = sorted({p["distance"] for p in points})
    for metric in ("ler", "time"):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8.7))
        for ax, d in zip(axes.flat, distances, strict=False):
            for name, color in COLORS.items():
                series = sorted(
                    [{**p, "m": p["decoders"][name]} for p in points if p["distance"] == d],
                    key=lambda r: r["noise"],
                )
                if metric == "ler":
                    quality_curve(ax, series, "noise", color)
                    absent = [r for r in series if not r["m"].get("quality_complete")]
                else:
                    for field, resolution, _, style in FIELDS:
                        x = [r["noise"] * 100 for r in series]
                        y = [
                            r["m"][field] / 1000
                            if r["m"].get(resolution) and r["m"].get(field) is not None
                            else np.nan
                            for r in series
                        ]
                        ax.plot(x, y, color=color, ls=style, lw=1.5 if field == "median_us" else 1)
                        for row in series:
                            if row["m"].get(field) is None:
                                continue
                            resolved = row["m"].get(resolution)
                            ax.plot(
                                row["noise"] * 100,
                                row["m"][field] / 1000,
                                marker="o"
                                if field == "median_us"
                                else "s"
                                if field == "p95_us"
                                else "^",
                                ms=3.5,
                                linestyle="none",
                                color=color,
                                markerfacecolor=color if resolved else "white",
                                alpha=1 if resolved else 0.5,
                            )
                    absent = [r for r in series if "median_us" not in r["m"]]
                if absent:
                    ax.scatter(
                        [r["noise"] * 100 for r in absent],
                        [0.035] * len(absent),
                        transform=ax.get_xaxis_transform(),
                        marker="x",
                        color=color,
                        s=40,
                    )
            ax.set_title(f"Distance d = {d}", weight="bold")
            axes_style(
                ax,
                "Circuit error rate p (%)",
                "Logical error probability" if metric == "ler" else "Decode time (ms)",
            )
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator([p * 100 for p in noises]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.set_ylim(1e-5 if metric == "ler" else 0.001, 1 if metric == "ler" else 2e5)
            if metric == "time":
                ax.axhline(1, color="#bc3030", lw=1.2)
        legend_ax = axes.flat[-1]
        legend_ax.axis("off")
        handles = [Line2D([0], [0], color=COLORS[n], marker="o", label=NAMES[n]) for n in NAMES]
        if metric == "time":
            handles += [
                Line2D([0], [0], color="#333", ls=style, label=label)
                for _, _, label, style in FIELDS
            ]
            handles += [Line2D([0], [0], color="#bc3030", label="1 ms median budget")]
        legend_ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10)
        legend_ax.text(
            0.04,
            0.30 if metric == "time" else 0.65,
            "Measured data only\n× = compute cap; value unavailable\n"
            + (
                "Open symbols = sparse quantiles\nAbsent tails remain unresolved"
                if metric == "time"
                else "Open triangles = zero-error 95% upper bound\n"
                "Error bars = 95% Wilson intervals"
            ),
            transform=legend_ax.transAxes,
            fontsize=9,
            va="top",
            linespacing=1.6,
        )
        fig.suptitle(
            ("LER" if metric == "ler" else "Median / p95 / p99 decode time")
            + " vs circuit error rate | full XYZ Relay-BP vs PyMatching",
            weight="bold",
            fontsize=15,
        )
        fig.text(
            0.02,
            0.012,
            "Frozen full XYZ profile: 601 legs, S=5, maximum 36,080 iterations. "
            "Rotated Z memory; rounds=d. Bounded samples at costly points; "
            "no partial-quality imputation.",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        save(fig, output, f"{metric}_circuit_error_rate_comparison")


def number(value, scale=1):
    return "unresolved" if value is None else f"{value * scale:.4g}"


def report(points, models, forecasts, output):
    native_samples = [
        p["decoders"]["relay_bp"]
        for p in points
        if p["decoders"]["relay_bp"].get("quality_complete")
    ]
    native_shots = sum(m["shots"] for m in native_samples)
    nonconverged = sum(m["nonconverged"] for m in native_samples)
    maximum_observed_iterations = max((m["iterations_max"] for m in native_samples), default=0)
    pieces = [
        "<!doctype html><meta charset='utf-8'><title>Full XYZ d15 / d31 sweep</title>",
        "<style>body{font:16px system-ui;max-width:1550px;margin:35px auto;padding:0 25px}",
        "p{max-width:1150px;line-height:1.6}img{width:100%}table{border-collapse:collapse;font-size:12px}",
        "td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}"
        ".scroll{overflow:auto}</style>",
        "<h1>Full XYZ accuracy through d15, conditional projections to d31</h1>",
        "<p>Frozen IBM authors' native implementation, full detector information, exact column "
        "merging, gamma0=.35, gamma range [-.254,.985], unscaled min-sum, S=5, "
        "601 total legs and at most 80+600×60=36,080 BP iterations. No memory-basis projection "
        "or matching fallback is used. Rotated Z memory with rounds=d "
        "and the contract's circuit noise.</p>",
        f"<p>Complete native quality samples total <b>{native_shots:,} shots</b>, "
        f"with <b>{nonconverged:,} native nonconvergence flags</b>. Maximum observed "
        f"iteration count is {maximum_observed_iterations:,} of the configured 36,080. "
        "The S=5 stopping rule can finish before the cap. Interrupted jobs are not included "
        "in these convergence totals and cannot establish convergence or LER.</p>",
        "<p>The user selected a bounded diagnostic sweep after runtime pilots exceeded 20 seconds "
        "per call at large noisy points. Each point allows 300 seconds overall and 60 seconds per "
        "serial call. Independent 1–4-shot runtime pilots fix later sample counts; a pilot stops "
        "after a call longer than 2 seconds and uses a 2× planning safety factor. Native quality "
        "uses up to four Rayon workers after all serial timing finishes. Serial/parallel native "
        "outputs and iteration counts were checked for equality. Parallel throughput is never "
        "reported as isolated-shot latency.</p>",
        "<p>PyMatching uses the repository's default MWPM adapter, with correlation "
        "reweighting disabled; this is not the correlated-matching variant. It scores "
        "32,768 full-circuit shots per point. "
        "Relay scores a fixed prefix chosen "
        "by the independent pilot, so its paired comparison uses that same prefix of PyMatching's "
        "sample. If that fixed quality job hits its runtime cap after complete timing, one "
        "smaller, independently seeded fixed-size quality confirmation is allowed, with a "
        "90-second cap. Its count depends only on the original runtime pilot, never the "
        "interrupted outcomes. Confirmations use the equivalent serial detailed native path "
        "to avoid upstream parallel per-shot decoder clones. Their paired matching "
        "predictions are saved; "
        "the primary "
        "PyMatching curve retains the original 32,768-shot sample. Confirmation points are "
        "labelled in the measurement table and CSV. Original capped evidence remains intact. "
        "No partially completed quality batch is turned into an LER estimate. Every "
        "completed Relay sample includes unconverged predictions. Quality intervals are 95% Wilson "
        "intervals; zero errors imply an upper bound, not zero underlying probability. "
        "Rising zero-error upper bounds at costly distances reflect fewer shots, not an "
        "observed increase in failures.</p>",
        "<p>One host pause interrupted d9, p=.007 before its matching reference finished "
        "(4,959 seconds elapsed against a 300-second guard). A separate whole-point recheck "
        "uses the same frozen protocol; interrupted evidence is retained separately. "
        "This host interruption is not a decoder resource or latency failure.</p>",
        "<p>Unrelated floor-analysis CLI/modules were added to the workspace during the run. "
        "The remaining measurements and audit use a private source snapshot whose complete "
        "benchmark and decoder hashes exactly match the original manifest. Workspace changes "
        "were preserved. <a href='source_isolation.json'>Source-isolation verification</a> "
        "and the frozen source tree accompany the results.</p>",
        "<p>Normal packed one-shot calls are timed serially, including adapter overhead on "
        "this host. CPU affinity and frequency were not pinned. Medians with fewer than 32 probes, "
        "p95 with fewer than 200, and p99 with fewer than 1,000 are flagged as sparse. "
        "These thresholds are diagnostic heuristics requiring about ten tail observations, "
        "not precision guarantees. With fewer than 20 probes no p95/p99 is reported. "
        "The red line is the user's <b>1 ms median budget</b>; larger distances are still "
        "attempted as diagnostics. Runtime guards are separate from that admission budget.</p>",
        "<img src='circuit_error_rate_measured.png' "
        "alt='Measured LER and time across circuit errors'>",
        "<img src='ler_circuit_error_rate_comparison.png' "
        "alt='Both decoders LER against circuit error at each distance'>",
        "<img src='time_circuit_error_rate_comparison.png' "
        "alt='Both decoders median p95 p99 against circuit error at each distance'>",
        "<img src='ler_distance_measured_d15.png' alt='LER through distance 15'>",
        "<img src='time_distance_measured_d15.png' alt='Median p95 p99 through distance 15'>",
        "<img src='ler_distance_projected_d31.png' alt='LER conditional projections to 31'>",
        "<img src='time_distance_projected_d31.png' alt='Time conditional projections to 31'>",
        "<h2>Projection assumptions</h2><p>LER uses binomial logit-linear distance fits "
        "on complete "
        "d7–15 data; d3–15 is the fallback if the recent window is unidentified. At least three "
        "complete distances and errors at two distinct distances are required. A separate "
        "d11–15 sensitivity fit is shown when identified. Zero-error counts enter the likelihood "
        "directly. Latency uses log-log power-law fits only on adequately sampled percentiles, "
        "preferring d7–15 and otherwise using all eligible distances. "
        "Missing fits remain unresolved. "
        "Independent timing percentile fits can cross outside their measured range; "
        "displayed estimates and interval endpoints then use an ordered envelope. "
        "Raw fits and bounds remain in the CSV/JSON. These envelopes are not simultaneous "
        "95% coverage statements. "
        "The plotted bands are conditional fit intervals. They do not cover a new BP convergence "
        "floor, larger-distance algorithm failure, hardware changes "
        "or other model misspecification. "
        "High-error extrapolations can exceed random-guessing LER and should be treated as a "
        "warning that this simple trend model is outside its useful regime.</p>",
        "<p>All d17–31 curves are unvalidated extrapolations and confer no budget admission. "
        "Both conservative graph resource models exceed the configured 256 MiB cap at d17. "
        "No d31 decoder is allocated. The full confidence limits, including limits outside "
        "the plotted range, and exact fit windows are stored in the projection CSV/JSON.</p>",
        "<img src='sampling_coverage.png' alt='Actual completed sample counts'>",
        "<h2>Median budget and graph caps</h2><p>Budget admission uses measured median "
        "only. Larger distances remain diagnostic even after the first failure. "
        "Both conservative graph models reject the next tested odd distance, d17, at 256 MiB. "
        "Storage estimates are planning bounds, not measured process RSS.</p>"
        "<table><tr><th>p</th><th>Decoder</th><th>Measured passing distances</th>"
        "<th>First measured median over 1 ms</th><th>Missing median distances</th></tr>",
    ]
    for p in sorted({r["noise"] for r in points}):
        for name in NAMES:
            series = rows_for(points, p, name)
            passing = [r["distance"] for r in series if r["m"].get("median_budget_pass")]
            failed = [
                r["distance"]
                for r in series
                if "median_us" in r["m"] and not r["m"]["median_budget_pass"]
            ]
            missing = [r["distance"] for r in series if "median_us" not in r["m"]]
            pieces.append(
                f"<tr><td>{p:g}</td><td>{NAMES[name]}</td><td>{passing or 'none'}</td>"
                f"<td>{min(failed) if failed else 'not observed'}</td>"
                f"<td>{missing or 'none'}</td></tr>"
            )
    pieces += [
        "</table><h2>Complete measurements and caps</h2><div class='scroll'><table><tr>"
        "<th>d</th><th>p</th><th>Decoder</th><th>Status</th>"
        "<th>Errors/shots</th><th>Paired PM errors/shots (Relay sample)</th><th>LER [95% CI]</th>"
        "<th>Probes</th><th>Median ms</th><th>p95 ms</th><th>p99 ms</th><th>Tail status</th>"
        "<th>Median budget</th><th>Estimated peak MiB</th><th>Reason</th></tr>",
    ]
    for point in sorted(points, key=lambda r: (r["noise"], r["distance"])):
        for name, m in point["decoders"].items():
            estimate = point.get("estimates", {}).get(name, {})
            quality = (f"{m['errors']}/{m['shots']}") if m.get("quality_complete") else "unresolved"
            paired = (
                f"{m['matching_paired_errors']}/{m['paired_shots']}"
                if name == "relay_bp" and m.get("quality_complete")
                else "—"
            )
            ler = (
                (f"{m['logical_error_rate']:.4g} [{m['ci95_low']:.3g}, {m['ci95_high']:.3g}]")
                if m.get("quality_complete")
                else "—"
            )
            tails = "; ".join(
                label + ": " + ("adequate" if m.get(flag) else "sparse/unresolved")
                for _, flag, label, _ in FIELDS[1:]
            )
            peak = estimate.get("peak_working_bytes")
            budget = (
                "unmeasured"
                if "median_us" not in m
                else "empirical pass"
                if m["median_budget_pass"]
                else "over 1 ms"
            )
            values = [
                str(point["distance"]),
                f"{point['noise']:g}",
                NAMES[name],
                m["status"],
                quality,
                paired,
                ler,
                str(m.get("latency_samples", "—")),
            ] + [number(m.get(field), 0.001) for field, _, _, _ in FIELDS]
            values += [tails, budget, number(peak, 1 / 1024**2), str(m.get("reason", ""))]
            pieces.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
    pieces += [
        "</table></div><h2>d31 conditional estimates</h2><table><tr><th>p</th>"
        "<th>Decoder</th><th>Metric</th><th>Value</th><th>Conditional interval</th>"
        "<th>Fit distances</th></tr>"
    ]
    for row in forecasts:
        if row["distance"] == 31:
            pieces.append(
                f"<tr><td>{row['noise']:g}</td><td>{NAMES[row['decoder']]}</td>"
                f"<td>{row['metric'].removesuffix('_us')} ({row['units']})</td>"
                f"<td>{row['value']:.4g}</td>"
                f"<td>[{row['fit95_low']:.4g}, {row['fit95_high']:.4g}]</td>"
                f"<td>{row['fit_distances']}"
                f"{'; weak support' if row.get('weak_distance_support') else ''}</td></tr>"
            )
    pieces += [
        "</table><h2>LER trend adequacy</h2><p>Small goodness-of-fit p-values flag "
        "departure from a single logit-linear distance trend. The approximate diagnostic "
        "is less reliable with sparse counts. Fit intervals do not account for this "
        "model error.</p><table><tr><th>p</th><th>Decoder</th><th>Fit distances</th>"
        "<th>Errors</th><th>Deviance / df</th><th>Approximate p-value</th><th>Flag</th></tr>"
    ]
    for p, by_name in models.items():
        for name, entry in by_name.items():
            if "ler" not in entry:
                continue
            fit = entry["ler"]
            flags = []
            if fit["fit_support_warning"]:
                flags.append("fewer than 20 errors")
            if fit["weak_distance_support"]:
                flags.append("weak distance support")
            if fit["deviance_p_value"] < 0.05:
                flags.append("trend mismatch")
            pieces.append(
                f"<tr><td>{p}</td><td>{NAMES[name]}</td>"
                f"<td>{fit['training_distances']}</td><td>{sum(fit['errors'])}</td>"
                f"<td>{fit['deviance']:.3g} / {fit['degrees_of_freedom']}</td>"
                f"<td>{fit['deviance_p_value']:.3g}</td><td>{'; '.join(flags) or '—'}</td></tr>"
            )
    pieces += ["</table><h2>Unresolved fits</h2><ul>"]
    for p, by_name in models.items():
        for name, entry in by_name.items():
            for metric, reason in entry["unavailable"].items():
                pieces.append(f"<li>p={p}, {NAMES[name]}, {metric}: {html.escape(reason)}</li>")
    pieces += [
        "</ul><p><a href='analysis_results.csv'>Measurements CSV</a> · "
        "<a href='results.csv'>Original bounded sweep CSV</a> · "
        "<a href='projections.csv'>Projection CSV</a> · "
        "<a href='projections.json'>Models and fit diagnostics</a> · "
        "<a href='manifest.json'>Frozen profile, sampling policy and environment</a> · "
        "<a href='verification.json'>Verification</a></p>"
    ]
    (output / "report.html").write_text("\n".join(pieces), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("past_experiments/decoder_testing/results/full-xyz-distance-noise-20261121-bounded"),
    )
    args = parser.parse_args()
    output = args.directory.resolve()
    manifest = json.loads((output / "manifest.json").read_text())["identity"]
    data_path = output / "analysis_results.json"
    if not data_path.exists():
        data_path = output / "results.json"
    points = json.loads(data_path.read_text())
    if len(points) != len(manifest["distances"]) * len(manifest["noise"]):
        raise ValueError(
            "bounded sweep must finish every requested condition before final plotting"
        )
    models, forecasts = build_models(points, manifest["noise"])
    measurement_rows = []
    for point in points:
        for name, measure in point["decoders"].items():
            row = {
                "distance": point["distance"],
                "rounds": point["rounds"],
                "noise": point["noise"],
                "decoder": name,
                "point_status": point["status"],
                "measurement_source": point.get("measurement_source", "primary bounded sweep"),
                "measurement_directory": point.get("measurement_directory", ""),
                **measure,
            }
            row.update(
                {
                    "estimated_" + key: value
                    for key, value in point.get("estimates", {}).get(name, {}).items()
                    if key
                    in (
                        "persistent_bytes",
                        "peak_working_bytes",
                        "build_operations",
                        "decode_operations_per_shot",
                    )
                }
            )
            measurement_rows.append(row)
    with (output / "analysis_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in measurement_rows for k in row))
        )
        writer.writeheader()
        writer.writerows(measurement_rows)
    write_json(
        output / "projections.json",
        {
            "models": models,
            "rows": forecasts,
            "meaning": "conditional trend projections, not validated large-distance performance",
            "sampling_qualification": "only adequate quantiles fit; unresolved tails remain absent",
            "budget_admission_from_projection": False,
        },
    )
    with (output / "projections.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in forecasts for k in row))
        )
        writer.writeheader()
        writer.writerows(forecasts)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    for projected in (False, True):
        distance_figures(points, models, manifest["noise"], output, projected)
    noise_figure(points, manifest["noise"], output)
    noise_comparison(points, manifest["noise"], output)
    coverage(points, manifest["noise"], output)
    report(points, models, forecasts, output)
    shutil.copy2(__file__, output / "plot_snapshot.py")
    shutil.copy2(
        Path(__file__).with_name("distance_projection.py"), output / "fit_source_snapshot.py"
    )
    shutil.copy2(
        Path(__file__).with_name("distance_projection.py"), output / "distance_projection.py"
    )
    write_json(
        output / "render_manifest.json",
        {
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fit_source_sha256": hashlib.sha256(
                Path(__file__).with_name("distance_projection.py").read_bytes()
            ).hexdigest(),
            "results_sha256": hashlib.sha256((output / "results.json").read_bytes()).hexdigest(),
            "plotted_data": data_path.name,
            "plotted_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        },
    )
    print(output / "report.html")


if __name__ == "__main__":
    main()

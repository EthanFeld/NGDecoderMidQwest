"""Export scientific plots without mixing noise models or memory bases."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def plot_results(source: Path, output: Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "plotting needs: python -m pip install -e './past_experiments/decoder_testing[plot]'"
        ) from error

    if source.is_dir():
        source = source / "results.json"
    with source.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file)) if source.suffix == ".csv" else json.load(file)
    if not rows:
        raise ValueError("no result rows to plot")
    panels: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        panels[(row["geometry"], row["basis"], row["noise_model"])].append(row)
    figure, axes = plt.subplots(len(panels), 2, squeeze=False, figsize=(12, 4.3 * len(panels)))
    for index, (panel, panel_rows) in enumerate(sorted(panels.items())):
        error_axis, speed_axis = axes[index]
        groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        for row in panel_rows:
            groups[(row["decoder"], float(row["noise"]))].append(row)
        for (decoder, noise), points in sorted(groups.items()):
            points.sort(key=lambda r: int(r["distance"]))
            distances = [int(r["distance"]) for r in points]
            rates = [float(r["logical_error_rate"]) for r in points]
            highs = [float(r["ci95_high"]) for r in points]
            display_rates = [p if p > 0 else high for p, high in zip(rates, highs, strict=True)]
            (line,) = error_axis.plot(distances, display_rates, label=f"{decoder}, p={noise:g}")
            for d, point, rate, high in zip(distances, points, rates, highs, strict=True):
                if rate > 0:
                    error_axis.errorbar(
                        d,
                        rate,
                        yerr=[[max(0, rate - float(point["ci95_low"]))], [max(0, high - rate)]],
                        fmt="o",
                        color=line.get_color(),
                        capsize=3,
                    )
                else:
                    error_axis.plot(d, high, "v", color=line.get_color(), markerfacecolor="none")
            speeds = [float(r["decode_shots_per_second"] or 0) for r in points]
            speed_axis.plot(distances, speeds, "o-", label=f"{decoder}, p={noise:g}")
        title = f"{panel[0]} memory-{panel[1]}, {panel[2]} noise"
        error_axis.set(
            title=title, xlabel="Code distance", ylabel="Logical failure probability / shot"
        )
        speed_axis.set(title=title, xlabel="Code distance", ylabel="Decoder throughput (shots/s)")
        error_axis.set_yscale("log")
        speed_axis.set_yscale("log")
        for axis in (error_axis, speed_axis):
            axis.grid(True, which="both", alpha=0.2)
            axis.legend(fontsize="small")
    figure.suptitle("Surface-code decoder comparison")
    figure.text(
        0.02,
        0.01,
        "Bars: 95% Wilson interval. Open triangles: zero errors, 95% upper endpoint. "
        "Throughput excludes sampling/compile; batch amortization is not single-shot latency.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output

"""Plot floor intervals without presenting a genie bound as an achieved LER."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def plot_floor(source: Path, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if source.is_dir():
        source = source / "ler_floor.json"
    with source.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file)) if source.suffix == ".csv" else json.load(file)
    if not rows:
        raise ValueError("no LER floor rows to plot")
    if any(row["status"] == "confidence_conflict" for row in rows):
        raise ValueError(
            "floor confidence conflict must be resolved before plotting a valid bracket"
        )
    panels = defaultdict(list)
    for row in rows:
        panels[(row["geometry"], row["basis"], row["noise_model"])].append(row)
    figure, axes = plt.subplots(len(panels), 1, squeeze=False, figsize=(10, 4.5 * len(panels)))
    for index, (panel, points) in enumerate(sorted(panels.items())):
        axis = axes[index, 0]
        groups = defaultdict(list)
        for row in points:
            groups[float(row["noise"])].append(row)
        positives = [
            float(row[key])
            for row in points
            for key in ("minimum_ler_lower", "minimum_ler_upper")
            if float(row[key]) > 0
        ]
        display_zero = min(positives, default=1e-3) / 10
        for noise, group in sorted(groups.items()):
            group.sort(key=lambda row: int(row["distance"]))
            distances = [int(row["distance"]) for row in group]
            lower = [max(display_zero, float(row["minimum_ler_lower"])) for row in group]
            upper = [max(display_zero, float(row["minimum_ler_upper"])) for row in group]
            (line,) = axis.plot(distances, upper, "o-", label=f"p={noise:g}: upper endpoint")
            axis.plot(
                distances,
                lower,
                "--",
                color=line.get_color(),
                label=f"p={noise:g}: analytic lower bound",
            )
            axis.fill_between(distances, lower, upper, color=line.get_color(), alpha=0.12)
            for d, low, row in zip(distances, lower, group, strict=True):
                if row["status"] == "exact":
                    axis.plot(d, low, "*", color=line.get_color(), markersize=11)
                if float(row["minimum_ler_lower"]) == 0:
                    axis.plot(d, display_zero, "v", color=line.get_color(), markerfacecolor="none")
        axis.set(
            title=f"{panel[0]} memory-{panel[1]}, {panel[2]} noise",
            xlabel="Code distance",
            ylabel="Minimum possible LER / memory experiment",
            yscale="log",
        )
        axis.set_xticks(sorted({int(row["distance"]) for row in points}))
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(fontsize="small")
    figure.suptitle("Bayes-optimal LER: exact values or uncertainty brackets")
    figure.text(
        0.02,
        0.015,
        "Shading is an uncertainty bracket, not a measured floor. "
        "Stars: exact joint sum. Open triangles: zero lower endpoint.\n"
        "Upper endpoints use fixed-shot decoder confidence bounds; "
        "analytic endpoints assume the stated DEM and float64 accuracy.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.94))
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output

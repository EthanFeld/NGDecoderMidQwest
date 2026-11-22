"""Generate the deck's numerical appendix from recorded measurements and source."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
EXPORT_DIR = None
LINK_ROOT = ROOT
RATES = [.001, .002, .003, .004, .005, .006, .007, .0075, .008, .009, .010]
BOARD_RATES = [.001, .003, .005, .007]
TIMING_POINTS = [(15, .005), (7, .003), (7, .0075), (15, .003), (15, .0075)]
MODES = ["fpga_only_zero_on_defer", "tensor_only", "telescope", "pymatching", "correlated_pymatching", "relay_bp"]
LABELS = {"fpga_only_zero_on_defer": "FPGA only (zero on defer)", "tensor_only": "Tensor only",
          "telescope": "Telescope", "pymatching": "PyMatching", "correlated_pymatching": "Correlated PyMatching",
          "relay_bp": "Relay-BP"}
PACKAGE = ROOT / "decoders/pymatching_telescope/src/pymatching_tensor_decoder"
PROFILE7 = ROOT / "past_experiments/surface_fpga/profiles/d7z-dual-p007-cut0-81mhz.json"
PROFILE15 = ROOT / "past_experiments/surface_fpga/experiments/d15-weighted-growth/d15z-growth-cap4000.json"
GAP3 = ROOT / "decoders/telescope/profiles/d15z-gap3-81mhz.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(path, label=None):
    return f"[{label or path.name}]({Path(os.path.relpath(path, LINK_ROOT)).as_posix()})"


def number(value, places=3):
    if value is None:
        return "Pending"
    rendered = f"{float(value):,.{places}f}"
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def percentage(value, places=5):
    return "Pending" if value is None else number(float(value) * 100, places) + "%"


def count(value):
    return "Pending" if value is None else f"{int(value):,}"


def ratio(a, b):
    return "Pending" if a is None or b is None or b <= 0 else number(a / b, 2) + "×"


def wilson(errors, shots):
    if not shots:
        return None, None
    z = 1.959963984540054
    p = errors / shots
    denominator = 1 + z * z / shots
    center = (p + z * z / (2 * shots)) / denominator
    delta = z * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / denominator
    return max(0., center - delta), min(1., center + delta)


def plot_ler(row):
    if not row:
        return "Pending"
    if int(row["errors"]) == 0:
        return "< " + percentage(wilson(0, int(row["shots"]))[1]) + " (95% upper)"
    return percentage(int(row["errors"]) / int(row["shots"]))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows]])


def field(row, name):
    return row.get(name) if row else None


def latency_field(row, metric="median"):
    if not row:
        return None
    values = row.get("single_shot_us_excluding_uart")
    if isinstance(values, dict):
        return values.get(metric, values.get("p50") if metric == "median" else None)
    return row.get("single_shot_us_excluding_uart_" + metric)


def config_defaults(path, class_name):
    module = ast.parse(path.read_text(encoding="utf-8-sig"))
    node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return {item.target.id: ast.literal_eval(item.value) for item in node.body
            if isinstance(item, ast.AnnAssign) and item.value is not None}


def geometry(image):
    """Execute only the source's pure stencil geometry, without loading a decoder."""
    source = PACKAGE / "telescope/stencils.py"
    parsed = ast.parse(source.read_text(encoding="utf-8-sig"))
    selected = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {"Shape", "Plan", "geometry"}]
    module = ModuleType("_deck_geometry")
    module.__dict__.update(defaultdict=defaultdict, dataclass=dataclass, mean=mean)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), module.__dict__)
    return module.geometry(image)


def load_model():
    json_path, csv_path = BASE / "model/results.json", BASE / "model/results.csv"
    document = read_json(json_path) or {"settings": {}, "results": []}
    csv_rows = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            csv_rows = list(csv.DictReader(stream))
    rows = document["results"]
    if not rows:
        for original in csv_rows:
            row = {}
            for key, value in original.items():
                if value == "":
                    row[key] = None
                elif value in {"True", "False"}:
                    row[key] = value == "True"
                else:
                    try:
                        row[key] = float(value) if "." in value or "e" in value.lower() else int(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
    index = {(int(row["distance"]), float(row["p"]), row["mode"]): row for row in rows}
    for distance in [7, 15]:
        for rate in RATES:
            key = distance, rate, "fpga_only_zero_on_defer"
            if key not in index and (distance, rate, "fpga_only") in index:
                index[key] = index[distance, rate, "fpga_only"]
    return document, index, len(csv_rows)


def write_board_csv(board):
    """Export only completed, full-population physical cohorts; missing timing stays blank."""
    output = (EXPORT_DIR or BASE) / "board_summary.csv"
    fields = [
        "distance", "p", "seed", "source", "mode", "shots", "errors", "ler", "ci95_low", "ci95_high",
        "accepted", "defers", "coverage", "defer_rate", "capture_run_index", "timing_complete", "timed_shots",
        "fpga_core_us_median", "fpga_core_us_p99", "uart_round_trip_us_median", "uart_delivery_us_median",
        "host_us_median", "host_us_p99", "single_shot_us_excluding_uart_median", "single_shot_us_excluding_uart_p99",
        "single_shot_us_including_uart_median", "single_shot_us_including_uart_p99",
        "telescope_host_only_us_median", "telescope_us_excluding_uart_median", "telescope_us_including_uart_median",
        "initial_matching_on_defers_mean_us", "initial_matching_timed_defers", "capture_sha256", "joint_sha256",
        "board_score_json", "board_score_json_sha256", "paired_arrays", "paired_arrays_sha256",
    ]
    rows = []
    for distance, report in sorted(board.items()):
        if not report:
            continue
        score_path = BASE / f"board/d{distance}/score.json"
        for cohort in report.get("cohorts", []):
            n = cohort.get("shots", 0)
            if not cohort.get("complete") or n != 10000:
                continue
            accepted, deferred = cohort["accepted"], cohort["deferred"]
            if accepted + deferred != n:
                raise ValueError("Physical-board accept/defer counts do not cover the full cohort")
            timing = cohort.get("timing") or {}
            host = timing.get("host_us", {})
            telescope_host = host.get("telescope") or {}
            telescope_without = timing.get("telescope_without_uart_us") or {}
            telescope_with = timing.get("telescope_with_uart_us") or {}
            matching = cohort.get("initial_correlated_matching_on_board_defers_us") or {}
            matching_timing = matching.get("timing") or {}
            modes = {"fpga_only_zero_on_defer": cohort["fpga_forced_zero_on_defer"],
                     "tensor_only": cohort["quality"]["tensor"], "telescope": cohort["quality"]["telescope"],
                     "pymatching": cohort["quality"]["pymatching"],
                     "correlated_pymatching": cohort["quality"]["correlated_pymatching"]}
            for mode, quality in modes.items():
                if quality["shots"] != n or not 0 <= quality["errors"] <= n:
                    raise ValueError("Board quality row is not a full-population measurement")
                ci = quality.get("ci95") or wilson(quality["errors"], n)
                mode_host = host.get("tensor" if mode == "tensor_only" else mode) or {}
                mode_without = telescope_without if mode == "telescope" else mode_host
                mode_with = telescope_with if mode == "telescope" else {}
                rows.append({
                    "distance": distance, "p": cohort["p"], "seed": cohort["seed"], "source": "physical_board",
                    "mode": mode, "shots": n, "errors": quality["errors"], "ler": quality["errors"] / n,
                    "ci95_low": ci[0], "ci95_high": ci[1], "accepted": accepted, "defers": deferred,
                    "coverage": accepted / n, "defer_rate": deferred / n, "capture_run_index": cohort["capture_run_index"],
                    "timing_complete": bool(timing), "timed_shots": timing.get("single_shot_calls", 0),
                    "fpga_core_us_median": (cohort.get("fpga_core_us") or {}).get("p50"),
                    "fpga_core_us_p99": (cohort.get("fpga_core_us") or {}).get("p99"),
                    "uart_round_trip_us_median": (cohort.get("uart_round_trip_us") or {}).get("p50"),
                    "uart_delivery_us_median": (cohort.get("uart_delivery_us") or {}).get("p50"),
                    "host_us_median": mode_host.get("p50"), "host_us_p99": mode_host.get("p99"),
                    "single_shot_us_excluding_uart_median": mode_without.get("p50"),
                    "single_shot_us_excluding_uart_p99": mode_without.get("p99"),
                    "single_shot_us_including_uart_median": mode_with.get("p50"),
                    "single_shot_us_including_uart_p99": mode_with.get("p99"),
                    "telescope_host_only_us_median": telescope_host.get("p50"),
                    "telescope_us_excluding_uart_median": telescope_without.get("p50"),
                    "telescope_us_including_uart_median": telescope_with.get("p50"),
                    "initial_matching_on_defers_mean_us": matching_timing.get("mean"),
                    "initial_matching_timed_defers": matching.get("shots"),
                    "capture_sha256": report["capture"]["sha256"], "joint_sha256": report["joint"]["sha256"],
                    "board_score_json": score_path.relative_to(ROOT).as_posix(), "board_score_json_sha256": digest(score_path),
                    "paired_arrays": cohort["paired_arrays"]["path"],
                    "paired_arrays_sha256": cohort["paired_arrays"]["sha256"],
                })
    staging = output.with_suffix(".csv.tmp")
    with staging.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    staging.replace(output)
    return output, len(rows)


def main():
    global EXPORT_DIR, LINK_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export-dir", type=Path, help="write all regenerated outputs into this directory without modifying stored measurements")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--interim", action="store_true", help="write an explicitly IN PROGRESS draft under measurements while runs finish")
    args = parser.parse_args()
    if args.require_complete and args.interim:
        parser.error("--require-complete and --interim are mutually exclusive")
    if args.export_dir:
        if args.output or args.interim:
            parser.error("--export-dir cannot be combined with --output or --interim")
        EXPORT_DIR = args.export_dir.resolve()
        if EXPORT_DIR == BASE or BASE in EXPORT_DIR.parents:
            parser.error("export destination must be outside the stored measurement directory")
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        args.output = EXPORT_DIR / "results.md"
        shutil.copyfile(BASE / "model/results.csv", EXPORT_DIR / "model_results.csv")
    if args.output is None:
        args.output = ((BASE / "TENSOR_TELESCOPE_DECK_V2_NUMBERS_IN_PROGRESS.md") if args.interim
                       else (ROOT / "TENSOR_TELESCOPE_DECK_V2_NUMBERS.md"))
    LINK_ROOT = args.output.resolve().parent
    document, index, csv_count = load_model()
    source_paths = [BASE / "model/results.json", BASE / "model/results.csv",
                    BASE / "model/timing_scope.json",
                    BASE / "board/d7/score.json", BASE / "board/d15/score.json",
                    BASE / "standard_stim/comparison.json", BASE / "standard_stim/residual_comparison.json"]
    board = {distance: read_json(BASE / f"board/d{distance}/score.json") for distance in [7, 15]}
    standard = read_json(BASE / "standard_stim/comparison.json")
    residual = read_json(BASE / "standard_stim/residual_comparison.json")
    config = config_defaults(PACKAGE / "config.py", "DecoderConfig")
    small = config_defaults(PACKAGE / "_small/policy.py", "CascadeConfig")
    image7, image15 = read_json(PROFILE7), read_json(PROFILE15)
    plan7, plan15 = geometry(image7), geometry(image15)
    expected = [(15, p, mode) for p in RATES for mode in MODES]
    expected += [(7, p, mode) for p in [.003, .0075] for mode in MODES]
    missing = [key for key in expected if key not in index]
    timing_pending = [key for key in expected if key[:2] in TIMING_POINTS
                      and key in index and not index[key].get("timing_complete")]
    board_incomplete = {}
    for distance, item in board.items():
        cohorts = (item or {}).get("cohorts", [])
        board_incomplete[distance] = (not item or not item.get("complete") or len(cohorts) != len(BOARD_RATES)
                                      or {row["p"] for row in cohorts} != set(BOARD_RATES)
                                      or any(not row.get("complete") or row.get("shots") != 10000
                                             or row.get("timing") is None for row in cohorts))
    standard_incomplete = (not standard or not standard.get("complete") or standard.get("shots") != 400000
                           or standard.get("target_shots") != 400000)
    incomplete = bool(missing or timing_pending or standard_incomplete or not residual or not residual.get("complete")
                      or any(board_incomplete.values()))
    if incomplete and not args.interim:
        raise RuntimeError("Measurements are still in progress; final appendix was not written")
    board_csv, board_csv_rows = write_board_csv(board)
    source_paths.append(board_csv)

    def model(distance, rate, mode):
        return index.get((distance, rate, mode))

    def point_rows(distance, rate, include_timing=False):
        result = []
        for mode in MODES:
            row = model(distance, rate, mode)
            cells = [f"d{distance}", percentage(rate), LABELS[mode], count(field(row, "errors")),
                     count(field(row, "shots")), percentage(field(row, "ler"))]
            if include_timing:
                cells += [number(latency_field(row)), count(field(row, "timed_shots"))]
            else:
                cells += [plot_ler(row)]
            result.append(cells)
        return result

    def board_cohort(distance, rate):
        return next((item for item in (board[distance] or {}).get("cohorts", []) if item["p"] == rate), None)

    generated = datetime.now(UTC).isoformat()
    sections = ["# " + ("IN PROGRESS — " if args.interim else "") + "Tensor Telescope — Deck v2 numbers", f"Generated: {generated}",
                "M = newly recorded cycle-model/software cohorts. B = newly captured physical-board replies with paired CPU replay. "
                "LER is per complete memory experiment. All latency values below are microseconds; single-shot medians exclude UART unless stated.",
                "Zero-error measurements retain LER = 0 in the count tables. Log plots use the 95% Wilson upper bound and an upper-limit marker."]
    if incomplete:
        sections.append("IN PROGRESS: `Pending` marks a measurement that has not completed. This draft is not the final numerical appendix.")

    sections += ["## Slide 2 — LER versus speed", "d15, 15 rounds, Z memory, physical p = 0.005; source M.",
                 table(["Distance", "Physical p", "Decoder", "Errors", "Shots", "LER", "Median µs", "Timed shots"],
                       point_rows(15, .005, include_timing=True))]
    tensor_median = latency_field(model(15, .005, "tensor_only"))
    telescope_median = latency_field(model(15, .005, "telescope"))
    sections.append(f"Tensor-only median / telescope median: **{ratio(tensor_median, telescope_median)}**. "
                    f"PyMatching median / telescope median: **{ratio(latency_field(model(15, .005, 'pymatching')), telescope_median)}**.")

    sections += ["## Slide 3 — Pipeline", "A valid FPGA accept supplies the logical bit. Each defer sends the original joint syndrome to the tensor backend."]
    telescope = model(15, .005, "telescope")
    sections.append(table(["M, d15 at p = 0.005", "Value"], [
        ["FPGA defers / shots", f"{count(field(telescope, 'fpga_defers'))} / {count(field(telescope, 'shots'))}"],
        ["FPGA defer rate", percentage(field(telescope, "fpga_defer_rate"))],
        ["Tensor-only → telescope median µs", f"{number(tensor_median)} → {number(telescope_median)}"],
    ]))
    sections.append("Original-versus-residual probe: d15 gap3 rescue, p = 0.005, 4,096 newly sampled Surface joint-circuit shots; cycle-model frontend.")
    probe_rows = []
    for key, label in [("original_telescope", "FPGA accepts + original-syndrome fallback"),
                       ("residual_plus_frame_telescope", "FPGA accepts + residual-syndrome fallback")]:
        values = (residual or {}).get("results", {}).get(key, {})
        probe_rows.append([label, count(values.get("errors")), count(values.get("shots")), percentage(values.get("ler")),
                           count(values.get("errors_on_defers")), count(values.get("errors_on_accepts"))])
    sections.append(table(["Input / routing", "Errors", "Shots", "LER", "Errors on defers", "Errors on accepts"], probe_rows))
    if residual:
        sections.append(f"Probe seed: {residual.get('seed')}; FPGA accepts/defers: "
                        f"{count(residual.get('fpga_accepts'))}/{count(residual.get('fpga_defers'))}. Source: "
                        + link(BASE / "standard_stim/residual_comparison.json") + ".")
    physical_probe = (board_cohort(15, .005) or {}).get("residual_probe")
    if physical_probe:
        sections.append("Physical-board residual replay:")
        sections.append(table(["Input / routing", "Errors", "Shots", "LER"], [
            [name, count(values["errors"]), count(values["shots"]), percentage(values["ler"])]
            for name, values in physical_probe.items() if isinstance(values, dict) and "errors" in values]))

    sections += ["## Slide 4 — Initial FPGA decoding", table(
        ["Distance", "Physical edge families", "Virtual bridge families", "First pass clocks", "Two-pass clocks"],
        [[f"d{distance}", sum(not shape.virtual for shape in plan.shapes), sum(shape.virtual for shape in plan.shapes),
          plan.pass_clocks, plan.max_clocks] for distance, plan in [(7, plan7), (15, plan15)]]),
        "The retry swaps the first two shape priorities. A completed first pass can answer before the retry.",
        "Source: " + link(PACKAGE / "telescope/stencils.py") + " and " + link(PACKAGE / "telescope/reference.py") + "."]

    clock7 = document.get("settings", {}).get("profiles", {}).get("7", {}).get("clock_mhz", image7["decoder_clock_mhz"])
    deadline7 = plan7.max_clocks + image7["growth_cycles"] + 1
    sections += ["## Slide 5 — Growth clusters", r"$w = \log((1-q)/q)$", r"$t = 2\max(2,\lfloor Rw/w_{\min}+b+0.5\rfloor)$",
        "Here b is the boundary bias, applied only to boundary edges; q uses the compiled growth prior.",
        table(["Profile", "Resolution R", "Boundary bias b", "Growth watchdog clocks", "Initial clocks + handoff + watchdog", "Clock MHz", "Core deadline µs"], [
            ["d7 dual growth", image7["growth_resolution"], image7.get("growth_boundary_bias", 0), image7["growth_cycles"],
             deadline7, clock7, number(deadline7 / clock7)],
            ["d15 weighted growth", image15["growth_resolution"], image15.get("growth_boundary_bias", 0), image15["growth_cycles"],
             plan15.max_clocks + image15["growth_cycles"] + 1,
             document.get("settings", {}).get("profiles", {}).get("15", {}).get("clock_mhz", "Pending"),
             number((plan15.max_clocks + image15["growth_cycles"] + 1) / document["settings"]["profiles"]["15"]["clock_mhz"])
             if document.get("settings", {}).get("profiles", {}).get("15", {}).get("clock_mhz") else "Pending"],
        ]), "Sources: " + link(ROOT / "past_experiments/surface_fpga/growth.py") + "; " + link(PROFILE7) + "; " + link(PROFILE15) + "."]

    sections += ["## Slide 6 — Matching", f"Tensor routing uses **path score < {number(config['path_factor'], 0)}d**; "
                 f"at d15 the threshold is **{number(config['path_factor'] * 15, 0)}**."]
    matching = (board_cohort(15, .005) or {}).get("initial_correlated_matching_on_board_defers_us")
    if matching and matching.get("timing"):
        sections.append(table(["Measurement", "Actual FPGA defers", "Mean µs", "Median µs", "p99 µs"], [[
            "B, d15 p = 0.005, completed correlated edge extraction", count(matching["shots"]),
            number(matching["timing"]["mean"]), number(matching["timing"]["p50"]), number(matching["timing"]["p99"]),
        ]]))
    else:
        historical_path = ROOT / "decoders/telescope/tensor_decoder/results/runs/d15_correlated_cuda_resident_72621005.json"
        historical = read_json(historical_path)
        value = (historical or {}).get("timing", {}).get("221", {}).get("cpu", {}).get("mean_active_us_per_shot")
        sections.append(table(["Measurement", "Board defers", "Mean µs / defer", "Scope"], [[
            "Historical CPU correlated matching", count((historical or {}).get("board_defers")), number(value),
            "221-shot batch, three repeats; fresh measurement pending",
        ]]))
        if historical:
            sections.append("Historical source: " + link(historical_path) + ".")
    sections.append("Routing source: " + link(PACKAGE / "decoder.py") + ". Matching uses the PyMatching backend in these measurements.")

    sections += ["## Slide 7 — Tensor and refinement", table(["Rule", "Current setting"], [
        ["d7 bounded frontier beam", small["tensor_beam"]], ["Conditioned correction accepted when gain ≥", config["min_gain"]],
        ["Sector competition when |gap| ≤", config["ambiguous_gap"]],
        ["Also compete when", "conditioned and initial matching predictions disagree"],
        ["Sector choice accepted when |log-sum-exp margin| ≥", config["min_margin"]],
        ["Alternating iterations per seed/sector", config["iterations"]], ["Reverse seed", str(config["reverse_seed"]).lower()],
    ]), r"$m=\operatorname{logsumexp}_{c\in C_1}(-\mathrm{cost}(c))-\operatorname{logsumexp}_{c\in C_0}(-\mathrm{cost}(c))$.",
        "The score sums retained candidate witnesses. Sources: " + link(PACKAGE / "config.py") + "; "
        + link(PACKAGE / "_small/policy.py") + "; " + link(PACKAGE / "decoder.py") + ".",
        "Separate backend validation: standard Stim rotated-memory Z circuit, d15, 15 rounds, p = 0.005."]
    standard_rows = []
    for label, key in [("Tensor only", "decoder_errors"), ("Correlated PyMatching", "correlated_matching_errors")]:
        errors, n = (standard or {}).get(key), (standard or {}).get("shots")
        standard_rows.append([label, count(errors), count(n), percentage(errors / n) if n and errors is not None else "Pending",
                              "Complete" if standard and standard.get("complete") else "In progress"])
    sections.append(table(["Decoder", "Errors", "Shots", "LER", "Status"], standard_rows))
    if standard:
        sections.append(f"Paired fixes/regressions versus correlated matching: **{count(standard.get('fixes'))}/{count(standard.get('regressions'))}**. "
                        f"Target shots: {count(standard.get('target_shots'))}. Seeds: " + ", ".join(map(str, standard.get("seeds", []))) + ".")
        sections.append("Source: " + link(BASE / "standard_stim/comparison.json") + ".")

    sections += ["## Slide 8 — LER versus distance", "Source M; d7 and d15 at physical p = 0.003 and 0.0075. Relay-BP uses the shot count shown in its own row.",
        table(["Distance", "Physical p", "Decoder", "Errors", "Shots", "LER", "Log-plot value / limit"],
              [row for rate in [.003, .0075] for distance in [7, 15] for row in point_rows(distance, rate)])]

    sections += ["## Slide 9 — LER versus physical error rate", "Source M; full d15 sweep. FPGA-only predictions use logical zero on a defer. "
                 "Defer is an FPGA frontend property; software-only baselines decode every input."]
    sweep_rows = []
    for rate in RATES:
        for mode in MODES:
            row = model(15, rate, mode)
            if mode in {"telescope", "fpga_only_zero_on_defer"}:
                defer = f"{count(field(row, 'fpga_defers'))}/{count(field(row, 'shots'))} ({percentage(field(row, 'fpga_defer_rate'))})"
            else:
                defer = "—"
            sweep_rows.append([percentage(rate), LABELS[mode], count(field(row, "errors")), count(field(row, "shots")),
                               percentage(field(row, "ler")), plot_ler(row), defer])
    sections.append(table(["Physical p", "Decoder", "Errors", "Shots", "LER", "Log-plot value / limit", "FPGA defers"], sweep_rows))

    sections += ["## Slide 10 — Defer rate", "M: d15 cycle-model cohorts.", table(
        ["Physical p", "Defers", "Shots", "Defer rate"],
        [[percentage(rate), count(field(model(15, rate, "telescope"), "fpga_defers")),
          count(field(model(15, rate, "telescope"), "shots")), percentage(field(model(15, rate, "telescope"), "fpga_defer_rate"))] for rate in RATES]),
        "B: newly captured physical d15 replies. Status comes from the hardware response."]
    board_rows = []
    for rate in BOARD_RATES:
        values = board_cohort(15, rate)
        q = (values or {}).get("quality", {}).get("telescope", {})
        timing = (values or {}).get("timing") or {}
        board_rows.append([percentage(rate), count(field(values, "deferred")), count(field(values, "shots")),
                           percentage(field(values, "defer_rate")), count(q.get("errors")), percentage(q.get("ler")),
                           number(timing.get("telescope_without_uart_us", {}).get("p50")),
                           number(timing.get("telescope_with_uart_us", {}).get("p50"))])
    sections.append(table(["Physical p", "Defers", "Shots", "Defer rate", "Telescope errors", "Telescope LER", "Median µs excl. UART", "Median µs incl. UART"], board_rows))
    if board[15]:
        sections.append("Board source: " + link(BASE / "board/d15/score.json") + ".")
    sections.append("B: newly captured physical d7 replies.")
    sections.append(table(["Physical p", "Defers", "Shots", "Defer rate", "Telescope errors", "Telescope LER"], [
        [percentage(rate), count(field(board_cohort(7, rate), "deferred")), count(field(board_cohort(7, rate), "shots")),
         percentage(field(board_cohort(7, rate), "defer_rate")),
         count((board_cohort(7, rate) or {}).get("quality", {}).get("telescope", {}).get("errors")),
         percentage((board_cohort(7, rate) or {}).get("quality", {}).get("telescope", {}).get("ler"))] for rate in BOARD_RATES]))

    sections += ["## Slide 11 — Decoding time versus distance", "Source M; completed single-shot calls at p = 0.003 and 0.0075. "
                 "FPGA/telescope values add cycle-model core time to measured CPU work. UART and queue fill are excluded."]
    timing_rows = []
    for rate in [.003, .0075]:
        for distance in [7, 15]:
            pm = latency_field(model(distance, rate, "pymatching"))
            tel = latency_field(model(distance, rate, "telescope"))
            for mode in MODES:
                row = model(distance, rate, mode)
                timing_rows.append([f"d{distance}", percentage(rate), LABELS[mode], number(latency_field(row)),
                                    number(latency_field(row, "p99")), count(field(row, "timed_shots")),
                                    ratio(pm, tel) if mode == "telescope" else "—"])
    sections.append(table(["Distance", "Physical p", "Decoder", "Median µs", "p99 µs", "Timed shots", "PyMatching / telescope median"], timing_rows))
    sections += ["## Sources", "\n".join("- " + link(path) for path in source_paths if path.is_file()),
                 "The model/software sweep uses the Surface joint circuit and the compiled d7/d15 growth profiles. "
                 "The 400,000-shot backend study uses the separate standard Stim circuit. Accuracy, single-shot latency, amortized work and UART transport are separate measurements.",
                 "The board CSV exports five measured modes; Relay-BP is measured in the separate model/software sweep. "
                 "FPGA-only LER uses logical zero on defers. Core and UART columns are actual hardware measurements. "
                 + ("Host/telescope timings are measured separately; FPGA-only host-completion time is unmeasured, and its row reports core timing separately."
                    if not incomplete else "Unmeasured host/telescope timing columns are blank; FPGA-only core time is reported separately from unmeasured host completion."),
                 "The requested model single-shot timing points are d15 at p = 0.003, 0.005 and 0.0075, and d7 at p = 0.003 and 0.0075. "
                 "The complete eleven-rate d15 LER sweep requires accuracy at every rate."]
    args.output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    manifest = {"generated_at": generated, "output": os.path.relpath(args.output, ROOT).replace(os.sep, "/"), "output_sha256": digest(args.output),
                "complete": not incomplete and not args.interim, "interim": args.interim,
                "standard_stim_incomplete": bool(standard_incomplete), "board_incomplete": board_incomplete,
                "model_json_rows": len(document["results"]), "model_csv_rows": csv_count, "board_csv_rows": board_csv_rows,
                "missing_model_points": missing, "model_timing_pending": timing_pending,
                "requested_model_timing_points": TIMING_POINTS,
                "sources": {os.path.relpath(path, ROOT).replace(os.sep, "/"): digest(path) for path in source_paths if path.is_file()},
                "source_constants": {os.path.relpath(path, ROOT).replace(os.sep, "/"): digest(path) for path in [PACKAGE / "config.py", PACKAGE / "_small/policy.py", PACKAGE / "telescope/stencils.py", PROFILE7, PROFILE15]}}
    ((EXPORT_DIR or BASE) / "numbers_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "complete": not incomplete, "model_rows": len(document["results"]),
                      "missing_points": len(missing), "timing_pending": len(timing_pending)}))


if __name__ == "__main__":
    main()

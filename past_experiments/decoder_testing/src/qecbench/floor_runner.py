"""Attach Bayes-floor brackets to ordinary, fixed-shot decoder distance sweeps."""

from __future__ import annotations

import csv
import io
import math
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from scipy.special import betaincinv

from .config import SweepConfig
from .ler_floor import FloorOptions, analyze_dem
from .runner import experiment_id, run_sweep
from .storage import OutputLock, atomic_text, digest, read_json, write_json


def binomial_upper(errors: int, shots: int, alpha: float) -> float:
    """One-sided exact Clopper-Pearson bound, requiring a fixed sample count."""
    if type(errors) is not int or type(shots) is not int or not 0 <= errors <= shots or shots < 1:
        raise ValueError("require integer 0 <= errors <= shots and shots > 0")
    if not 0 < alpha < 1 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and in (0, 1)")
    if errors == shots:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log(alpha) / shots)
    # Inverse survival form avoids losing alpha when it is very small.
    return float(1 - betaincinv(shots - errors, errors + 1, alpha))


def summarize_floor(analysis: dict, rows: list[dict], alpha: float) -> dict:
    if not rows or any(not row["complete"] for row in rows):
        raise ValueError("floor reporting requires completed decoder measurements")
    bounds = [(binomial_upper(r["errors"], r["shots"], alpha), r) for r in rows]
    upper, best = min(bounds, key=lambda pair: (pair[0], pair[1]["decoder"]))
    exact = analysis["exact_ler"]
    lower = analysis["minimum_ler_lower"]
    floor_upper = exact if exact is not None else min(upper, analysis["minimum_ler_upper_model"])
    consistent = lower <= upper + 1e-14
    keys = ("experiment_id", "distance", "rounds", "noise", "basis", "geometry", "noise_model")
    return {
        **{key: best[key] for key in keys},
        "method": analysis["method"],
        "status": ("exact" if exact is not None else "bounded")
        if consistent
        else "confidence_conflict",
        "minimum_ler_estimate": exact,
        "minimum_ler_lower": lower,
        "minimum_ler_upper": floor_upper,
        "bracket_ratio": floor_upper / lower if lower > 0 else None,
        "best_decoder": best["decoder"],
        "best_observed_ler": best["logical_error_rate"],
        "best_decoder_errors": best["errors"],
        "shots": best["shots"],
        "best_decoder_ler_upper": upper,
        "upper_alpha_per_comparison": alpha,
        "confidence_consistent": consistent,
        "logical_paths": analysis["logical_paths"],
        "mechanisms": analysis["mechanisms"],
        "num_detectors": analysis["num_detectors"],
        "num_observables": analysis["num_observables"],
        "exact_skip_reason": analysis["exact_skip_reason"],
    }


def run_floor_sweep(
    config: SweepConfig,
    *,
    options: FloorOptions | None = None,
    confidence: float = 0.95,
    resume: bool = False,
    progress: Callable[[list[dict[str, Any]]], None] | None = None,
    bound_progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Return a per-experiment interval containing the minimum achievable LER.

    Analytic lower bounds assume the supplied independent-mechanism DEM. Upper
    endpoints use achieved decoder performance with simultaneous binomial coverage
    across the predeclared decoder list AND all sweep points. No latency constraint
    or postselection is imposed on the optimum being estimated.
    """
    if config.max_errors:
        raise ValueError("floor confidence bounds require fixed shots; set --max-errors 0")
    if isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and in (0, 1)")
    options = options or FloorOptions(memory_mib=config.memory_mib)
    measured = run_sweep(config, resume=resume, progress=progress)
    comparisons = len(config.experiments()) * len(config.decoders)
    alpha = (1 - confidence) / comparisons
    output = config.output_path
    with OutputLock(output):
        parent = read_json(output / "manifest.json")
        identity = digest(
            {
                "parent": parent["identity"],
                "options": asdict(options),
                "confidence": confidence,
                "measurements": measured,
            }
        )
        cache = output / "floor_checkpoints" / identity[:24]
        result = []
        for experiment in config.experiments():
            identifier = experiment_id(experiment)
            path = cache / f"{identifier}.json"
            if path.exists():
                saved = read_json(path)
                if saved["analysis_identity"] != identity:
                    raise ValueError(f"floor checkpoint identity mismatch: {path}")
                row = saved["result"]
            else:
                started = time.perf_counter()
                # Raw DEM keeps correlated hyperedges; graph decomposition is not
                # an independence assumption and is unnecessary for this analysis.
                analysis = analyze_dem(experiment.circuit().detector_error_model(), options)
                point_rows = [r for r in measured if r["experiment_id"] == identifier]
                row = summarize_floor(analysis, point_rows, alpha)
                row["analysis_seconds"] = time.perf_counter() - started
                write_json(
                    path, {"analysis_identity": identity, "analysis": analysis, "result": row}
                )
            result.append(row)
            if bound_progress:
                bound_progress(row)
            # Keep completed point reports usable if a later large model hits a cap.
            write_json(output / "ler_floor.json", result)
            buffer = io.StringIO(newline="")
            columns = ["experiment_id", *sorted(set(result[0]) - {"experiment_id"})]
            writer = csv.DictWriter(buffer, fieldnames=columns)
            writer.writeheader()
            writer.writerows(result)
            atomic_text(output / "ler_floor.csv", buffer.getvalue())
        write_json(
            output / "ler_floor_manifest.json",
            {
                "analysis_identity": identity,
                "measurement_identity": parent["identity"],
                "options": asdict(options),
                "confidence": confidence,
                "simultaneous_comparisons": comparisons,
                "alpha_per_comparison": alpha,
                "target": "Bayes-optimal per-memory-experiment logical failure, no latency limit",
                "lower_bound": (
                    "exact joint coset sum or independent logical-complement genie pairs"
                ),
                "upper_bound": (
                    "achievable fixed-shot decoder; one-sided Clopper-Pearson + Bonferroni"
                ),
                "numerics": (
                    "float64 analytic bounds with roundoff guard; not formal interval arithmetic"
                ),
                "point_estimate": "reported only when exact joint enumeration fits resource caps",
            },
        )
        return result

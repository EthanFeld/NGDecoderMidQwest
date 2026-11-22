"""Audit a completed bounded sweep against raw samples and frozen provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

if __package__:
    from .latency_capped_sweep import matching_estimate
else:
    from latency_capped_sweep import matching_estimate

from qecbench import DecoderContext, Experiment
from qecbench.metrics import count_failures, wilson_interval
from qecbench.storage import environment_identity, implementation_identity, write_json
from surface_decoders import ResourceLimits
from surface_decoders.ibm_relay_bp import ACCURACY_CONFIG, estimate_ibm_resources


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_quality(measure, predictions, truth):
    assert measure["quality_complete"]
    assert predictions.dtype == truth.dtype == np.uint8
    assert predictions.shape == truth.shape
    errors = count_failures(predictions, truth)
    assert measure["shots"] == len(truth)
    assert measure["errors"] == errors
    assert measure["logical_error_rate"] == errors / len(truth)
    np.testing.assert_allclose(
        [measure["ci95_low"], measure["ci95_high"]], wilson_interval(errors, len(truth))
    )


def verify_timing(measure, times):
    assert measure["latency_samples"] == len(times)
    assert np.all(np.isfinite(times)) and np.all(times >= 0)
    assert measure["median_resolved"] == (len(times) >= 32)
    assert measure["p95_resolved"] == (len(times) >= 200)
    assert measure["p99_resolved"] == (len(times) >= 1000)
    for percentile, field in [(50, "median_us"), (95, "p95_us"), (99, "p99_us")]:
        if percentile > 50 and len(times) < 20:
            assert measure[field] is None
        else:
            np.testing.assert_allclose(measure[field], np.percentile(times, percentile))
    assert measure["median_budget_us"] == 1000
    assert measure["median_budget_pass"] == (measure["median_us"] <= 1000)


def verify_point(root, point, original):
    d, p = point["distance"], point["noise"]
    assert point["rounds"] == d
    assert point["condition_seed"] == 1349001 + 100000 * d + round(p * 1e6)
    directory = Path(point.get("measurement_directory", root / f"d{d}" / f"p{p:g}"))
    saved = np.load(directory / "samples.npz")
    matching = point["decoders"]["pymatching"]
    if "measurement_directory" in point:
        assert not original["decoders"]["pymatching"].get("quality_complete")
        assert original["elapsed_seconds"] > 600
        assert original["phase_at_cap"] == "matching_reference"
        assert point["measurement_source"] == "whole-point host-pause recheck; original retained"
    else:
        assert matching == original["decoders"]["pymatching"]
    assert matching["shots"] == 32768
    verify_quality(matching, saved["pymatching"], saved["truth"])
    verify_timing(matching, np.load(directory / "pymatching_timing.npz")["wall_us"])
    assert matching["latency_samples"] == 1024
    native = point["decoders"]["relay_bp"]
    if "median_us" in native:
        verify_timing(native, np.load(directory / "relay_timing.npz")["wall_us"])
        assert native["latency_samples"] == point["plan"]["latency_samples"]
    else:
        assert native["status"] in ("runtime_cap", "resource_cap")
    record = {"distance": d, "noise": p, "relay_status": native["status"]}
    if native.get("quality_complete"):
        if native["status"] == "complete_quality_confirmation":
            quality_directory = root / "quality_confirmation" / f"d{d}_p{p:g}"
            quality = np.load(quality_directory / "quality.npz")
            truth, reference = quality["truth"], quality["pymatching"]
            expected_n = min(2048, max(8, int(30 / point["plan"]["pilot_cost_upper_seconds"])))
            assert len(truth) == expected_n
            assert native["quality_seed"] == point["condition_seed"] + 109000000
            assert native["quality_threads"] == 1
            assert not native["paired_with_primary_matching_sample"]
            assert not original["decoders"]["relay_bp"].get("quality_complete", False)
            assert (quality_directory / "summary.json").exists()
            record["quality_source"] = "independent confirmation"
        else:
            quality = np.load(directory / "relay_quality.npz")
            n = point["plan"]["quality_shots"]
            truth, reference = saved["truth"][:n], saved["pymatching"][:n]
            assert native["shots"] == n
            record["quality_source"] = "complete original fixed sample"
        verify_quality(native, quality["predictions"], truth)
        assert native["paired_shots"] == len(truth)
        assert native["matching_paired_errors"] == count_failures(reference, truth)
        failures = np.any(quality["predictions"] != truth, axis=1)
        np.testing.assert_array_equal(quality["logical_failures"], failures)
        iterations, converged = quality["iterations"], quality["converged"]
        assert len(iterations) == len(converged) == len(truth)
        assert np.all((iterations >= 0) & (iterations <= ACCURACY_CONFIG.max_iterations))
        assert native["maximum_iterations"] == ACCURACY_CONFIG.max_iterations == 36080
        assert native["total_legs"] == ACCURACY_CONFIG.total_legs == 601
        assert native["nonconverged"] == int((~converged).sum())
        assert native["errors_nonconverged"] == int(np.count_nonzero(failures & ~converged))
        for percentile, field in [
            (50, "iterations_median"),
            (95, "iterations_p95"),
            (99, "iterations_p99"),
        ]:
            np.testing.assert_allclose(native[field], np.percentile(iterations, percentile))
        np.testing.assert_allclose(native["iterations_mean"], iterations.mean())
        assert native["iterations_max"] == int(iterations.max())
        record.update(
            shots=len(truth), errors=int(failures.sum()), nonconverged=int((~converged).sum())
        )
    else:
        assert "logical_error_rate" not in native and "errors" not in native
    assert "censored_call_lower_bound_seconds" not in native
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("past_experiments/decoder_testing/results/full-xyz-distance-noise-20261121-bounded"),
    )
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / "manifest.json").read_text())["identity"]
    points = json.loads((root / "analysis_results.json").read_text())
    original = json.loads((root / "results.json").read_text())
    assert len(points) == len(original) == 42
    assert {(r["distance"], r["noise"]) for r in points} == {
        (d, p) for d in manifest["distances"] for p in manifest["noise"]
    }
    assert manifest["relay_config"] == json.loads(json.dumps(asdict(ACCURACY_CONFIG)))
    assert manifest["source"] == implementation_identity(
        ("surface_decoders.ibm_relay_bp:build_accuracy_decoder",)
    )
    assert manifest["environment"] == environment_identity()
    assert digest(root / "runner_snapshot.py") == manifest["runner_sha256"]
    helper = Path(__file__).with_name("latency_capped_sweep.py")
    assert digest(helper) == manifest["resource_helper_sha256"]
    shutil.copy2(helper, root / "latency_capped_sweep.py")
    lookup = {(r["distance"], r["noise"]): r for r in original}
    records = [verify_point(root, r, lookup[(r["distance"], r["noise"])]) for r in points]
    projections = json.loads((root / "projections.json").read_text())
    for row in projections["rows"]:
        assert row["distance"] in range(17, 32, 2)
        assert row["kind"] == "conditional_trend_projection" and not row["budget_admitted"]
        if row["metric"] == "ler":
            assert 0 <= row["fit95_low"] <= row["value"] <= row["fit95_high"] <= 1
        else:
            assert np.isfinite([row["value"], row["fit95_low"], row["fit95_high"]]).all()
            assert row["units"] == "ms"
            assert 0 <= row["fit95_low"] <= row["value"] <= row["fit95_high"]
            assert len(row["fit_distances"]) >= 3
            for d in row["fit_distances"]:
                measurement = next(
                    r for r in points if r["distance"] == d and r["noise"] == row["noise"]
                )
                resolved = {
                    "median_us": "median_resolved",
                    "p95_us": "p95_resolved",
                    "p99_us": "p99_resolved",
                }[row["metric"]]
                assert measurement["decoders"][row["decoder"]][resolved]
    for p in manifest["noise"]:
        for name in ("relay_bp", "pymatching"):
            for d in range(17, 32, 2):
                rows = {
                    r["metric"]: r["value"]
                    for r in projections["rows"]
                    if r["noise"] == p and r["decoder"] == name and r["distance"] == d
                }
                ordered = [rows[k] for k in ("median_us", "p95_us", "p99_us") if k in rows]
                assert ordered == sorted(ordered)
    resources = []
    for d in (17, 31):
        experiment = Experiment(d, d, 0.003)
        circuit = experiment.circuit()
        context = DecoderContext(
            experiment, circuit, circuit.detector_error_model(decompose_errors=True)
        )
        limits = ResourceLimits()
        for estimate in (
            estimate_ibm_resources(context, ACCURACY_CONFIG, limits).to_dict(),
            matching_estimate(context, limits),
        ):
            assert not estimate["accepted"]
            assert estimate["peak_working_bytes"] > 256 * 1024**2
            resources.append({"distance": d, "noise": 0.003, **estimate})
    write_json(root / "resource_preflight_d31.json", resources)
    raw_hashes = {
        path.relative_to(root).as_posix(): digest(path) for path in sorted(root.rglob("*.npz"))
    }
    write_json(root / "raw_artifact_hashes.json", raw_hashes)
    write_json(
        root / "verification.json",
        {
            "status": "passed",
            "conditions_verified": len(records),
            "points": records,
            "checks": [
                "frozen full XYZ profile and source/native fingerprints",
                "all logical failures and Wilson intervals from raw predictions/truth",
                "fixed original or independently seeded confirmation quality counts",
                "serial quantiles from raw timings; sparse-tail qualifications",
                "1 ms budget uses median only",
                "iterations and convergence partitions",
                "projection source quantiles adequate; extrapolation never admits budget",
                "d17/d31 conservative resource rejection without decoder allocation",
            ],
            "analysis_sha256": digest(root / "analysis_results.json"),
            "projections_sha256": digest(root / "projections.json"),
            "verifier_sha256": digest(Path(__file__)),
            "raw_artifact_hashes_sha256": digest(root / "raw_artifact_hashes.json"),
        },
    )
    shutil.copy2(__file__, root / "verifier_snapshot.py")
    print("VERIFIED", len(records), "conditions", flush=True)


if __name__ == "__main__":
    main()

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

import examples.latency_capped_sweep as sweep
from examples.latency_capped_sweep import latency_admission, latency_metrics, matching_estimate
from qecbench import DecoderContext, Experiment
from surface_decoders import ResourceLimits


def test_median_latency_admission_allows_slow_tail_and_includes_boundary():
    assert latency_admission(np.array([20.0, 1000.0]), 1000)
    assert latency_admission(np.array([20.0] * 1000 + [100000.0]), 1000)
    assert latency_admission(np.array([0.0, 2000.0]), 1000)
    assert not latency_admission(np.array([1000.1] * 501 + [20.0] * 499), 1000)
    assert not latency_admission(np.array([float("nan")]), 1000)
    assert not latency_admission(np.array([]), 1000)


def test_latency_metrics_report_p95_p99_independently_of_budget():
    result = latency_metrics(np.arange(101, dtype=float) * 10, 500)
    assert result["single_shot_p50_us"] == 500
    assert result["single_shot_p95_us"] == 950
    assert result["single_shot_p99_us"] == 990
    assert result["single_shot_over_budget_count"] == 50
    assert result["latency_pass"]
    assert result["latency_budget_metric"] == "median"


def test_reuse_requires_same_experiment_sampling_sources_and_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "environment_identity", lambda: {"machine": "test"})
    monkeypatch.setattr(sweep, "implementation_identity", lambda _: {"source": "test"})
    args = SimpleNamespace(shots=10, latency_samples=8, noise=0.003, seed=123)
    limits = ResourceLimits()
    manifest = {
        "shots_per_point": 10,
        "latency_samples": 8,
        "noise": 0.003,
        "seed": 123,
        "limits": asdict(limits),
        "environment": {"machine": "test"},
        "source": {"source": "test"},
        "relay_config": asdict(sweep.FAST_CONFIG),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "results.json").write_text("[]")
    assert sweep.load_reuse(tmp_path, args, limits) == {}
    args.seed = 124
    with pytest.raises(ValueError, match="seed"):
        sweep.load_reuse(tmp_path, args, limits)


def test_matching_resource_preflight_reports_memory_cap_without_compilation():
    experiment = Experiment(3, 3, 0.003)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    admitted = matching_estimate(context, ResourceLimits())
    assert admitted["accepted"]
    assert admitted["peak_working_bytes"] >= admitted["persistent_bytes"]
    rejected = matching_estimate(context, ResourceLimits(max_memory_mib=0.001))
    assert not rejected["accepted"]
    assert "memory cap" in rejected["reasons"][0]


@pytest.mark.parametrize(
    "name,legs,maximum,factory",
    [
        ("fast", 8, 240, "build_fast_decoder"),
        ("tuned", 32, 1940, "build_tuned_decoder"),
        ("accuracy", 601, 36080, "build_accuracy_decoder"),
    ],
)
def test_latency_profile_uses_matching_resource_and_factory_configuration(
    name, legs, maximum, factory
):
    from qecbench.decoders import load_factory

    config, specs = sweep.relay_profile(name)
    assert (config.total_legs, config.max_iterations) == (legs, maximum)
    assert specs["relay_bp"].endswith(":" + factory)
    experiment = Experiment(3, 3, 0.003)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    decoder = load_factory(specs["relay_bp"])(context)
    assert decoder.config == config

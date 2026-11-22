import numpy as np
import pytest

from examples.full_xyz_distance_sweep import (
    quality_worker_count,
    sample_plan,
    timing_summary,
    watchdog_call_metadata,
)
from qecbench import DecoderContext, Experiment
from surface_decoders.ibm_relay_bp import ACCURACY_CONFIG, IBMRelayBPDecoder


def test_sampling_plan_uses_independent_cost_and_keeps_fixed_positive_limits():
    fast = sample_plan([0.00001, 0.00002, 0.00003, 0.00002], 280, 4, 32768, 1024)
    slow = sample_plan([0.4, 1, 2, 4], 280, 4, 32768, 1024)
    assert fast["quality_shots"] == 32768 and fast["latency_samples"] == 1024
    assert 8 <= slow["quality_shots"] < fast["quality_shots"]
    assert 2 <= slow["latency_samples"] < fast["latency_samples"]
    assert slow["counts_fixed_before_scoring"]
    assert sample_plan([30], 280, 4, 32768, 1024)["latency_samples"] == 2
    with pytest.raises(ValueError, match="complete"):
        sample_plan([], 280, 4, 32768, 1024)
    with pytest.raises(ValueError, match="remaining"):
        sample_plan([0.1] * 4, 1, 4, 32768, 1024)


def test_sparse_tails_are_explicitly_unresolved_instead_of_zero():
    tiny = timing_summary(np.arange(8) + 1000)
    assert tiny["p95_us"] is None and tiny["p99_us"] is None
    assert not tiny["p95_resolved"] and not tiny["p99_resolved"]
    assert not tiny["median_budget_pass"]
    full = timing_summary(np.arange(1024, dtype=float))
    assert full["p95_resolved"] and full["p99_resolved"]
    assert full["median_us"] == 511.5 and full["median_budget_pass"]


def test_point_expiry_during_a_call_does_not_impute_the_single_call_limit():
    metadata = watchdog_call_metadata(293, 300, 60)
    assert metadata["watchdog_call_elapsed_seconds"] == 7
    assert metadata["watchdog_call_limit_seconds"] == 60
    assert "censored_call_lower_bound_seconds" not in metadata
    assert "median_us" not in metadata
    assert watchdog_call_metadata(None, 300, 60) == {}


def test_parallel_memory_includes_original_native_and_matching_graphs():
    mib = 1024**2
    workers = quality_worker_count(4, 236 * mib, 236 * mib, 1024 * mib)
    assert workers == 2
    assert (2 + workers) * 236 * mib <= 1024 * mib
    assert quality_worker_count(4, 256 * mib, 256 * mib, 512 * mib) == 1
    assert quality_worker_count(1, 236 * mib, 236 * mib, 1024 * mib) == 1


def test_parallel_quality_matches_serial_normal_accuracy_profile_and_iteration_counts():
    pytest.importorskip("relay_bp")
    experiment = Experiment(3, 3, 0.007)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    decoder = IBMRelayBPDecoder(context, ACCURACY_CONFIG)
    events = circuit.compile_detector_sampler(seed=133781).sample(shots=41, bit_packed=True)
    events.flags.writeable = False
    syndrome = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")
    syndrome ^= decoder.detector_offset
    syndrome.flags.writeable = False
    serial = decoder.runner.decode_observables_detailed_batch(
        syndrome, parallel=False, progress_bar=False
    )
    parallel = decoder.runner.decode_observables_detailed_batch(
        syndrome, parallel=True, progress_bar=False
    )
    for first, second in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(first.observables, second.observables)
        assert first.iterations == second.iterations and first.converged == second.converged
    packed = np.packbits(
        np.asarray([r.observables for r in parallel], dtype=np.uint8) ^ decoder.observable_offset,
        axis=1,
        bitorder="little",
    )
    np.testing.assert_array_equal(packed, decoder.decode_batch(events))

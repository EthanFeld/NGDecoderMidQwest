"""Native reference parity, contract invariance, and surface-code integration."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from test_surface_decoders import context_for_dem, packed_syndromes, surface_context

from qecbench import SweepConfig, run_sweep
from qecbench.decoders import validate_predictions
from qecbench.storage import implementation_identity
from surface_decoders import (
    IBMRelayBPConfig,
    IBMRelayBPDecoder,
    ResourceLimitError,
    ResourceLimits,
    estimate_ibm_resources,
)
from surface_decoders.cli import main as estimate_main
from surface_decoders.ibm_relay_bp import FAST_CONFIG, TUNED_CONFIG, build_tuned_decoder
from surface_decoders.model import FaultModel

relay_bp = pytest.importorskip("relay_bp")


def native_projection(decoder, events):
    cls = (
        relay_bp.RelayDecoderF32 if decoder.config.precision == "f32" else relay_bp.RelayDecoderF64
    )
    native = cls(
        decoder.check_matrix, error_priors=decoder.error_priors, **decoder.config.native_kwargs()
    )
    syndrome = np.unpackbits(events, axis=1, count=decoder.num_detectors, bitorder="little")
    syndrome = syndrome[:, decoder.detector_indices]
    syndrome ^= decoder.detector_offset
    physical = native.decode_batch(syndrome)
    logical = np.asarray((decoder.observable_matrix @ physical.T).T % 2, dtype=np.uint8)
    logical ^= decoder.observable_offset
    return np.packbits(logical, axis=1, bitorder="little")


def test_published_repetition_code_example():
    from scipy.sparse import csr_matrix

    native = relay_bp.RelayDecoderF32(
        csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)),
        error_priors=np.array([0.003] * 3),
        gamma0=0.65,
        pre_iter=80,
        num_sets=100,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        stop_nconv=5,
    )
    result = native.decode_detailed(np.array([1, 1], dtype=np.uint8))
    assert result.success
    np.testing.assert_array_equal(result.decoding, [0, 1, 0])


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_exact_parity_with_direct_native_reference(precision):
    context = surface_context(rounds=3)
    config = IBMRelayBPConfig(
        pre_iter=8, num_sets=4, set_max_iter=8, stop_nconv=3, precision=precision, chunk_shots=7
    )
    decoder = IBMRelayBPDecoder(context, config)
    events = context.circuit.compile_detector_sampler(seed=876).sample(shots=129, bit_packed=True)
    events.flags.writeable = False
    np.testing.assert_array_equal(decoder.decode_batch(events), native_projection(decoder, events))


@pytest.mark.parametrize(
    "distance,basis,geometry,noise_model",
    [
        (2, "x", "rotated", "circuit"),
        (3, "z", "rotated", "circuit"),
        (3, "x", "unrotated", "phenomenological"),
        (5, "z", "rotated", "data"),
    ],
)
def test_batch_order_warmup_restart_and_chunk_independence(distance, basis, geometry, noise_model):
    context = surface_context(distance, distance, basis, geometry, noise_model)
    config = IBMRelayBPConfig(pre_iter=6, num_sets=4, set_max_iter=6, stop_nconv=3, chunk_shots=7)
    events = context.circuit.compile_detector_sampler(seed=123).sample(shots=37, bit_packed=True)
    events.flags.writeable = False
    original = events.copy()
    decoder = IBMRelayBPDecoder(context, config)
    whole = decoder.decode_batch(events)
    scalar = np.concatenate([decoder.decode_batch(events[i : i + 1]) for i in range(len(events))])
    reverse = decoder.decode_batch(np.ascontiguousarray(events[::-1]))[::-1]
    fresh = IBMRelayBPDecoder(context, replace(config, chunk_shots=32)).decode_batch(events)
    decoder.decode_batch(events[:5])
    repeated = decoder.decode_batch(events)
    for actual in (scalar, reverse, fresh, repeated):
        np.testing.assert_array_equal(actual, whole)
    np.testing.assert_array_equal(events, original)
    validate_predictions(whole, len(events), context.num_observables)


def test_correlated_faults_offsets_and_multi_byte_observables():
    context = context_for_dem("error(1) D2 L1\nerror(0.8) D0 L0 ^ D1 L0 L8")
    events = np.array([[4], [7]], dtype=np.uint8)
    decoder = IBMRelayBPDecoder(context, FAST_CONFIG)
    np.testing.assert_array_equal(decoder.decode_batch(events), [[2, 0], [2, 1]])
    assert decoder.check_matrix.shape == (3, 1)
    assert decoder.error_priors[0] == pytest.approx(0.2)


def test_equivalent_fault_merge_preserves_joint_distribution():
    from itertools import product

    from surface_decoders.model import bit_mask

    model = FaultModel.from_context(
        context_for_dem("error(0.1) D0 L0\nerror(0.2) D0 L0\nerror(0.3) D0 L1\nerror(0.8) D1 L0")
    )

    def distribution(candidate):
        result = np.zeros(16)
        offset = bit_mask(tuple(np.flatnonzero(candidate.detector_offset)))
        offset ^= bit_mask(tuple(np.flatnonzero(candidate.observable_offset))) << 2
        for pattern in product((0, 1), repeat=len(candidate.probabilities)):
            state, probability = offset, 1.0
            for i, on in enumerate(pattern):
                p = candidate.probabilities[i]
                probability *= p if on else 1 - p
                if on:
                    state ^= bit_mask(candidate.detectors[i]) | (
                        bit_mask(candidate.observables[i]) << 2
                    )
            result[state] += probability
        return result

    merged = model.merge_equivalent_faults()
    assert len(merged.probabilities) == 3  # Different logical columns sharing D0 remain separate.
    np.testing.assert_allclose(distribution(merged), distribution(model), rtol=1e-14, atol=1e-16)


def test_merging_tiny_priors_and_one_half_is_numerically_stable():
    model = FaultModel.from_context(
        context_for_dem(
            "error(1e-250) D0 L0\nerror(2e-250) D0 L0\nerror(0.5) D1 L1\nerror(0.2) D1 L1"
        )
    )
    merged = model.merge_equivalent_faults()
    assert merged.probabilities[0] == pytest.approx(3e-250, rel=1e-14, abs=0)
    assert merged.probabilities[1] == 0.5


def test_projection_preserves_correlated_fault_and_logical_labels():
    model = FaultModel.from_context(context_for_dem("error(0.2) D0 L0 ^ D1 L1\nerror(0.3) D1"))
    projected = model.project_detectors(np.array([0]))
    assert projected.detectors == ((0,),)
    assert projected.observables == ((0, 1),)
    np.testing.assert_array_equal(projected.probabilities, [0.2])


@pytest.mark.parametrize(
    "basis,geometry,expected_checks",
    [("x", "rotated", 16), ("z", "rotated", 16), ("x", "unrotated", 24), ("z", "unrotated", 24)],
)
def test_memory_basis_mode_matches_direct_native_and_is_batch_independent(
    basis, geometry, expected_checks
):
    context = surface_context(3, 3, basis, geometry)
    config = replace(FAST_CONFIG, detector_mode="memory_basis", merge_equivalent_faults=True)
    decoder = IBMRelayBPDecoder(context, config)
    assert decoder.check_matrix.shape[0] == expected_checks
    events = context.circuit.compile_detector_sampler(seed=181).sample(shots=37, bit_packed=True)
    events.flags.writeable = False
    actual = decoder.decode_batch(events)
    np.testing.assert_array_equal(actual, native_projection(decoder, events))
    np.testing.assert_array_equal(
        actual,
        np.concatenate([decoder.decode_batch(events[i : i + 1]) for i in range(len(events))]),
    )
    np.testing.assert_array_equal(
        actual, decoder.decode_batch(np.ascontiguousarray(events[::-1]))[::-1]
    )
    validate_predictions(actual, len(events), context.num_observables)


def test_memory_basis_mode_rejects_unrecognized_circuit():
    with pytest.raises(ValueError, match="standard generated"):
        IBMRelayBPDecoder(
            context_for_dem("error(0.1) D0 L0"), replace(FAST_CONFIG, detector_mode="memory_basis")
        )


@pytest.mark.parametrize(
    "basis,geometry", [("x", "rotated"), ("z", "rotated"), ("x", "unrotated"), ("z", "unrotated")]
)
def test_memory_basis_selection_agrees_with_physical_pauli_error_injection(basis, geometry):
    import stim

    from surface_decoders.ibm_relay_bp import memory_detector_indices

    context = surface_context(3, 2, basis, geometry)
    circuit = context.circuit.without_noise().flattened()
    selected = set(memory_detector_indices(context))
    final_measurement = next(inst for inst in reversed(list(circuit)) if inst.name in ("M", "MX"))
    data_qubits = [target.value for target in final_measurement.targets_copy()]
    positions = [i for i, inst in enumerate(circuit) if inst.name == "TICK"]
    for pauli in ("X", "Z"):
        sensitive = set()
        for position in positions:
            for qubit in data_qubits:
                injected = circuit.copy()
                injected.insert(position, stim.CircuitInstruction(f"{pauli}_ERROR", [qubit], [1]))
                events = injected.compile_detector_sampler().sample(shots=1)
                sensitive.update(np.flatnonzero(events[0]))
        if pauli == ("Z" if basis == "x" else "X"):
            assert sensitive == selected
        else:
            assert not (sensitive & selected)


def test_no_custom_repair_on_native_nonconvergence():
    context = context_for_dem("error(0.1) D0 L0\nerror(0.1) D0")
    config = IBMRelayBPConfig(pre_iter=1, num_sets=0, alpha=1, gamma0=0)
    events = np.array([[1]], dtype=np.uint8)
    decoder = IBMRelayBPDecoder(context, config)
    np.testing.assert_array_equal(decoder.decode_batch(events), native_projection(decoder, events))
    strict = IBMRelayBPDecoder(context, replace(config, failure_policy="raise"))
    with pytest.raises(RuntimeError, match="did not converge"):
        strict.decode_batch(events)


def test_strict_converged_output_matches_native():
    context = context_for_dem("error(0.01) D0 L0\nerror(0.02) D1 L8")
    decoder = IBMRelayBPDecoder(context, replace(FAST_CONFIG, failure_policy="raise"))
    events = packed_syndromes(2)
    np.testing.assert_array_equal(decoder.decode_batch(events), native_projection(decoder, events))


def test_noise_free_zero_width_and_empty_batches():
    context = surface_context(noise=0)
    decoder = IBMRelayBPDecoder(context)
    np.testing.assert_array_equal(decoder.decode_batch(np.zeros((5, 1), dtype=np.uint8)), 0)
    assert decoder.decode_batch(np.zeros((0, 1), dtype=np.uint8)).shape == (0, 1)
    empty = IBMRelayBPDecoder(context_for_dem("", detectors=0, observables=0))
    assert empty.decode_batch(np.zeros((3, 0), dtype=np.uint8)).shape == (3, 0)
    with pytest.raises(ValueError, match="support"):
        decoder.decode_batch(np.ones((1, 1), dtype=np.uint8))


def test_padding_and_dtype_validation():
    decoder = IBMRelayBPDecoder(context_for_dem("error(0.1) D0 L0"), FAST_CONFIG)
    for events in (
        np.array([[2]], dtype=np.uint8),
        np.ones((1, 1), dtype=bool),
        np.zeros((4, 1), dtype=np.uint8)[::2],
    ):
        with pytest.raises(ValueError):
            decoder.decode_batch(events)


@pytest.mark.parametrize(
    "limits",
    [
        ResourceLimits(max_memory_mib=0.00001),
        ResourceLimits(max_build_operations=1),
        ResourceLimits(latency_budget_us=1000, operations_per_second=1),
    ],
)
def test_caps_precede_model_and_native_allocation(monkeypatch, limits):
    context = surface_context()
    estimate = estimate_ibm_resources(context, FAST_CONFIG, limits)
    assert not estimate.accepted

    def forbidden(*args):
        raise AssertionError("allocated model before resource admission")

    monkeypatch.setattr(FaultModel, "from_context", forbidden)
    with pytest.raises(ResourceLimitError) as caught:
        IBMRelayBPDecoder(context, FAST_CONFIG, limits)
    assert caught.value.estimate == estimate


def test_decoder_is_nontrivial_and_better_than_zero_on_fixed_surface_sample():
    context = surface_context(rounds=3, noise=0.003)
    events, truth = context.circuit.compile_detector_sampler(seed=213).sample(
        shots=2048, separate_observables=True, bit_packed=True
    )
    predictions = IBMRelayBPDecoder(context, FAST_CONFIG).decode_batch(events)
    assert np.any(predictions)
    assert np.count_nonzero(predictions != truth) < np.count_nonzero(truth)


def test_native_dependency_is_fingerprinted(monkeypatch):
    import surface_decoders.ibm_relay_bp as module

    spec = "surface_decoders.ibm_relay_bp:build_fast_decoder"
    original = implementation_identity((spec,))
    identity = original[spec]["dependencies"]
    assert identity["relay_bp_version"] == "0.2.2"
    assert len(identity["upstream_python_native_sha256"]) == 64
    monkeypatch.setattr(module, "qecbench_implementation_identity", lambda: {"changed": True})
    assert implementation_identity((spec,)) != original


def test_native_resource_cli_uses_correct_profile(capsys):
    assert (
        estimate_main(["estimate", "--distances", "3", "--decoders", "ibm_relay_bp", "--ibm-fast"])
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["ibm_relay_bp_config"]["num_sets"] == 7
    assert data["estimates"][0]["decoder"] == "ibm_relay_bp"
    assert (
        estimate_main(
            [
                "estimate",
                "--distances",
                "3",
                "--decoders",
                "ibm_relay_bp",
                "--ibm-fast",
                "--latency-us",
                "1000",
                "--operations-per-second",
                "1",
            ]
        )
        == 2
    )
    assert not json.loads(capsys.readouterr().out)["estimates"][0]["accepted"]


def test_frozen_tuned_factory_and_resource_profile(capsys):
    context = surface_context(3, 3)
    decoder = build_tuned_decoder(context)
    assert decoder.config == TUNED_CONFIG
    events = context.circuit.compile_detector_sampler(seed=739).sample(shots=41, bit_packed=True)
    events.flags.writeable = False
    np.testing.assert_array_equal(decoder.decode_batch(events), native_projection(decoder, events))
    assert decoder.check_matrix.shape == (16, 55)
    assert (
        estimate_main(["estimate", "--distances", "3", "--decoders", "ibm_relay_bp", "--ibm-tuned"])
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["ibm_relay_bp_config"]["detector_mode"] == "memory_basis"
    assert report["ibm_relay_bp_config"]["stop_nconv"] == 3


def test_fresh_spawn_harness_resume(tmp_path):
    config = SweepConfig(
        distances=(2, 3),
        rounds=1,
        noise=(0.003,),
        decoders=("surface_decoders.ibm_relay_bp:build_fast_decoder", "pymatching"),
        shots=33,
        batch_size=16,
        warmup_shots=3,
        workers=2,
        output=str(tmp_path / "native"),
    )
    rows = run_sweep(config)
    assert len(rows) == 4 and all(row["shots"] == 33 for row in rows)
    assert run_sweep(config, resume=True) == rows

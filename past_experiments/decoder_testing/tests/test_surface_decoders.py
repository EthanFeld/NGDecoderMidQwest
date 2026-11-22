"""Algorithm oracles, frozen packed contract, and resource admission tests."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
import stim

from qecbench import DecoderContext, Experiment, SweepConfig, run_sweep
from qecbench.decoders import validate_predictions
from surface_decoders import (
    LUTConfig,
    LUTDecoder,
    RelayBPConfig,
    RelayBPDecoder,
    ResourceLimitError,
    ResourceLimits,
    estimate_resources,
)
from surface_decoders.cli import main
from surface_decoders.model import FaultModel, mechanisms


def context_for_dem(text, detectors=None, observables=None):
    dem = stim.DetectorErrorModel(text)
    circuit = stim.Circuit("M 0")
    for _ in range(dem.num_detectors if detectors is None else detectors):
        circuit.append("DETECTOR", [stim.target_rec(-1)])
    count = dem.num_observables if observables is None else observables
    if count:
        circuit.append("OBSERVABLE_INCLUDE", [], count - 1)
    return DecoderContext(Experiment(3, 1, 0.01), circuit, dem)


def surface_context(
    distance=3, rounds=1, basis="z", geometry="rotated", noise_model="circuit", noise=0.01
):
    experiment = Experiment(distance, rounds, noise, basis, geometry, noise_model)
    circuit = experiment.circuit()
    return DecoderContext(experiment, circuit, circuit.detector_error_model(decompose_errors=True))


def packed_syndromes(count):
    bits = ((np.arange(1 << count)[:, None] >> np.arange(count)) & 1).astype(np.uint8)
    return np.packbits(bits, axis=1, bitorder="little")


def test_lut_matches_independent_exhaustive_logical_ml_oracle():
    faults = [(0.15, 1, 1), (0.2, 2, 2), (0.3, 3, 3), (0.8, 1, 2), (0.12, 0, 1)]
    text = "\n".join(
        f"error({p}) "
        + " ".join(
            [f"D{i}" for i in range(2) if ds >> i & 1] + [f"L{i}" for i in range(2) if ls >> i & 1]
        )
        for p, ds, ls in faults
    )
    joint = np.zeros((4, 4))
    for selected in itertools.product((0, 1), repeat=len(faults)):
        probability, syndrome, logical = 1.0, 0, 0
        for on, (p, ds, ls) in zip(selected, faults, strict=True):
            probability *= p if on else 1 - p
            if on:
                syndrome ^= ds
                logical ^= ls
        joint[syndrome, logical] += probability
    decoder = LUTDecoder(context_for_dem(text))
    actual = decoder.decode_batch(packed_syndromes(2))[:, 0]
    np.testing.assert_array_equal(actual, np.argmax(joint, axis=1))


def test_lut_sums_degenerate_fault_patterns_instead_of_picking_most_likely_pattern():
    # Syndrome 1: logical 0 receives two weight-one patterns; logical 1 one likelier pattern.
    context = context_for_dem("error(0.15) D0\nerror(0.15) D0\nerror(0.2) D0 L0")
    assert LUTDecoder(context).decode_batch(np.array([[1]], dtype=np.uint8))[0, 0] == 0


def test_dem_separators_remain_correlated_and_cancel_duplicate_targets():
    context = context_for_dem("error(0.2) D0 L0 ^ D1 L0 L8")
    assert list(mechanisms(context.dem)) == [(0.2, (0, 1), (8,))]
    events = np.array([[0], [3]], dtype=np.uint8)
    expected = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    for cls in (LUTDecoder, RelayBPDecoder):
        np.testing.assert_array_equal(cls(context).decode_batch(events), expected)
        with pytest.raises(ValueError, match="support"):
            cls(context).decode_batch(np.array([[1]], dtype=np.uint8))


def test_nested_repeat_offsets_are_expanded_lazily():
    dem = stim.DetectorErrorModel("""repeat 2 {
        error(0.1) D0 L0
        repeat 2 {
            error(0.2) D1
            shift_detectors 1
        }
        shift_detectors 1
    }""")
    assert [ds for _, ds, _ in mechanisms(dem)] == [(0,), (1,), (2,), (3,), (4,), (5,)]


def test_huge_error_free_repeats_skip_expansion_but_keep_detector_shift():
    dem = stim.DetectorErrorModel("""repeat 1000000000 {
        repeat 2 {
            detector D0
            shift_detectors 1
        }
    }
    error(0.1) D0 L0
    """)
    assert list(mechanisms(dem)) == [(0.1, (2000000000,), (0,))]


@pytest.mark.parametrize("cls", [RelayBPDecoder, LUTDecoder])
def test_deterministic_and_complemented_faults_multi_byte_observables(cls):
    context = context_for_dem("error(1) D0 L8\nerror(0.8) D1 L0\nerror(0) D2 L2")
    events = np.array([[1], [3]], dtype=np.uint8)
    expected = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(cls(context).decode_batch(events), expected)


def test_relay_reuses_marginals_in_next_leg():
    context = context_for_dem("error(0.1) D0\nerror(0.1) D0 L0")
    decoder = RelayBPDecoder(context, RelayBPConfig(legs=2, iterations=1, failure_policy="raise"))
    decoder.memory[:] = [[0, 0], [-0.5, 0.5]]
    # First leg ends at zero marginals. Second breaks symmetry only with that handoff.
    np.testing.assert_array_equal(decoder.decode_batch(np.array([[1]], dtype=np.uint8)), [[1]])


def test_lut_extremely_rare_syndromes_do_not_underflow():
    context = context_for_dem("error(1e-250) D0 L0\nerror(1e-250) D1")
    np.testing.assert_array_equal(
        LUTDecoder(context).decode_batch(packed_syndromes(2)), [[0], [1], [0], [1]]
    )


def test_lut_ties_choose_smallest_final_logical_label():
    context = context_for_dem("error(1) L0\nerror(0.5) L0")
    np.testing.assert_array_equal(
        LUTDecoder(context).decode_batch(np.zeros((2, 0), dtype=np.uint8)), [[0], [0]]
    )


def test_lut_batch_chunk_and_call_order_invariance():
    context = surface_context()
    events = context.circuit.compile_detector_sampler(seed=123).sample(shots=37, bit_packed=True)
    events.flags.writeable = False
    original = events.copy()
    decoder = LUTDecoder(context, LUTConfig(chunk_shots=3))
    whole = decoder.decode_batch(events)
    scalar = np.concatenate([decoder.decode_batch(events[i : i + 1]) for i in range(len(events))])
    np.testing.assert_array_equal(whole, scalar)
    np.testing.assert_array_equal(whole, LUTDecoder(context).decode_batch(events))
    np.testing.assert_array_equal(events, original)
    validate_predictions(whole, len(events), context.num_observables)


def test_bp_nonconvergence_policy_is_explicit():
    context = context_for_dem("error(0.1) D0 L0\nerror(0.1) D0")
    events = np.array([[1]], dtype=np.uint8)
    config = RelayBPConfig(legs=1, iterations=1, first_memory=0)
    np.testing.assert_array_equal(RelayBPDecoder(context, config).decode_batch(events), [[1]])
    strict = RelayBPConfig(legs=1, iterations=1, first_memory=0, failure_policy="raise")
    with pytest.raises(RuntimeError, match="did not converge"):
        RelayBPDecoder(context, strict).decode_batch(events)


@pytest.mark.parametrize(
    "distance,basis,geometry,noise_model",
    [
        (2, "x", "rotated", "circuit"),
        (3, "z", "rotated", "circuit"),
        (3, "x", "unrotated", "phenomenological"),
        (5, "z", "rotated", "data"),
    ],
)
def test_relay_contract_and_batch_invariance(distance, basis, geometry, noise_model):
    context = surface_context(distance, distance, basis, geometry, noise_model)
    config = RelayBPConfig(legs=3, iterations=6, chunk_shots=7)
    events = context.circuit.compile_detector_sampler(seed=123).sample(shots=37, bit_packed=True)
    events.flags.writeable = False
    original = events.copy()
    decoder = RelayBPDecoder(context, config)
    whole = decoder.decode_batch(events)
    split = np.concatenate([decoder.decode_batch(events[i : i + 1]) for i in range(len(events))])
    permuted = decoder.decode_batch(np.ascontiguousarray(events[::-1]))[::-1]
    np.testing.assert_array_equal(whole, split)
    np.testing.assert_array_equal(whole, permuted)
    np.testing.assert_array_equal(whole, RelayBPDecoder(context, config).decode_batch(events))
    np.testing.assert_array_equal(events, original)
    validate_predictions(whole, len(events), context.num_observables)


@pytest.mark.parametrize("cls", [RelayBPDecoder, LUTDecoder])
def test_empty_noise_free_and_zero_width_models(cls):
    context = surface_context(noise=0)
    decoder = cls(context)
    assert decoder.decode_batch(np.zeros((0, 1), dtype=np.uint8)).shape == (0, 1)
    np.testing.assert_array_equal(decoder.decode_batch(np.zeros((3, 1), dtype=np.uint8)), 0)
    empty = cls(context_for_dem("", detectors=0, observables=0))
    assert empty.decode_batch(np.zeros((3, 0), dtype=np.uint8)).shape == (3, 0)


@pytest.mark.parametrize("cls", [RelayBPDecoder, LUTDecoder])
def test_input_padding_dtype_and_layout_are_checked(cls):
    decoder = cls(context_for_dem("error(0.1) D0 L0"))
    for events in (
        np.array([[2]], dtype=np.uint8),
        np.array([[1]], dtype=bool),
        np.zeros((3, 2), dtype=np.uint8),
        np.zeros((4, 1), dtype=np.uint8)[::2],
    ):
        with pytest.raises(ValueError):
            decoder.decode_batch(events)


@pytest.mark.parametrize("name,cls", [("relay_bp", RelayBPDecoder), ("lut", LUTDecoder)])
@pytest.mark.parametrize(
    "limits",
    [
        ResourceLimits(max_memory_mib=0.00001),
        ResourceLimits(max_build_operations=1),
        ResourceLimits(latency_budget_us=1, operations_per_second=1),
    ],
)
def test_caps_fail_before_model_allocation(monkeypatch, name, cls, limits):
    context = surface_context()
    estimate = estimate_resources(context, name, limits=limits)
    assert not estimate.accepted

    def forbidden(*args):
        raise AssertionError("large model allocated before resource admission")

    monkeypatch.setattr(FaultModel, "from_context", forbidden)
    with pytest.raises(ResourceLimitError) as caught:
        cls(context, limits=limits)
    assert caught.value.estimate == estimate


def test_lut_distance_cap_and_huge_repeat_scan_cap():
    estimate = estimate_resources(surface_context(rounds=3), "lut")
    assert not estimate.accepted
    assert estimate.joint_state_bits == 25
    assert any("peak" in reason for reason in estimate.reasons)
    context = context_for_dem("repeat 1000000000 {\nerror(0.1) D0 L0\n}")
    estimate = estimate_resources(context, "relay_bp")
    assert not estimate.accepted and "scan cap" in estimate.reasons[0]
    assert estimate_resources(context_for_dem("", detectors=100), "lut").joint_state_bits == 100


def test_configuration_validation():
    with pytest.raises(ValueError, match="requires operations_per_second"):
        ResourceLimits(latency_budget_us=1)
    for values in (
        {"legs": 0},
        {"iterations": True},
        {"memory_min": float("nan")},
        {"solutions": 10},
        {"failure_policy": "guess"},
    ):
        with pytest.raises(ValueError):
            RelayBPConfig(**values)
    with pytest.raises(ValueError):
        LUTConfig(chunk_shots=0)


def test_estimation_cli_reports_accepted_and_rejected_sizes(capsys):
    assert main(["estimate", "--distances", "3", "5", "--rounds", "1"]) == 2
    rows = json.loads(capsys.readouterr().out)["estimates"]
    assert [(r["decoder"], r["accepted"]) for r in rows] == [
        ("relay_bp", True),
        ("lut", True),
        ("relay_bp", True),
        ("lut", False),
    ]
    assert main(["estimate", "--distances", "1000000000"]) == 2
    assert all(not r["accepted"] for r in json.loads(capsys.readouterr().out)["estimates"])


def test_cli_rejects_missing_latency_rate_and_zero_chunk(capsys):
    assert main(["estimate", "--latency-us", "1"]) == 1
    assert "requires operations_per_second" in capsys.readouterr().err
    assert main(["estimate", "--chunk-shots", "0"]) == 1
    assert "chunk_shots" in capsys.readouterr().err


def test_importable_factories_spawned_harness_and_resume(tmp_path):
    config = SweepConfig(
        distances=(2, 3),
        rounds=1,
        noise=(0.01,),
        decoders=("surface_decoders.relay_bp:build_decoder", "surface_decoders.lut:build_decoder"),
        shots=33,
        batch_size=16,
        warmup_shots=3,
        workers=2,
        output=str(tmp_path / "run"),
    )
    rows = run_sweep(config)
    assert len(rows) == 4 and all(row["shots"] == 33 and row["complete"] for row in rows)
    assert run_sweep(config, resume=True) == rows

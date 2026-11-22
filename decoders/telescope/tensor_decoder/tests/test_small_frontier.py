"""Exact small-model oracles, sparse contraction equivalence and packed API."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import stim
from oracle import LUTDecoder

from tensor_decoder import DecoderContext, Experiment
from tensor_decoder._small.frontier import (
    FrontierConfig,
    FrontierTNDecoder,
    compile_frontier,
    reference_contract,
)
from tensor_decoder.build_native import support_library
from tensor_decoder.model import FaultModel, bit_mask


@pytest.fixture
def native_library():
    path = support_library()
    if not path.exists():
        pytest.skip("optional native kernel is not built")
    return path


def context_for_dem(text):
    dem = stim.DetectorErrorModel(text)
    circuit = stim.Circuit("M 0")
    for _ in range(dem.num_detectors):
        circuit.append("DETECTOR", [stim.target_rec(-1)])
    circuit.append("OBSERVABLE_INCLUDE", [], 0)
    return DecoderContext(Experiment(3, 1, 0.01), circuit, dem)


def packed_all(m):
    return np.packbits(
        ((np.arange(1 << m)[:, None] >> np.arange(m)) & 1).astype(np.uint8),
        axis=1,
        bitorder="little",
    )


def test_untruncated_frontier_matches_independent_fault_enumeration(native_library):
    faults = [
        (0.15, 1, 0),
        (0.15, 1, 0),
        (0.2, 1, 1),
        (0.8, 2, 1),
        (0.12, 3, 1),
        (0.11, 4, 0),
        (0.3, 6, 1),
    ]
    text = "\n".join(
        f"error({p}) "
        + " ".join([f"D{i}" for i in range(3) if ds >> i & 1] + (["L0"] if ls else []))
        for p, ds, ls in faults
    )
    joint = np.zeros((8, 2))
    for pattern in itertools.product((0, 1), repeat=len(faults)):
        probability, ds, ls = 1.0, 0, 0
        for on, (p, d, logical) in zip(pattern, faults, strict=True):
            probability *= p if on else 1 - p
            if on:
                ds ^= d
                ls ^= logical
        joint[ds, ls] += probability
    decoder = FrontierTNDecoder(
        context_for_dem(text), FrontierConfig(beam=64, refine_beam=64), native_library
    )
    prediction, diagnostics = decoder.decode_with_diagnostics(packed_all(3))
    np.testing.assert_array_equal(prediction[:, 0], np.argmax(joint, axis=1))
    assert diagnostics["unpruned"].all()
    assert not diagnostics["fallback"].any()


@pytest.mark.parametrize("strength", [0, 0.5, 1])
@pytest.mark.parametrize("beam", [1, 2, 8])
def test_native_truncation_matches_python_sum_product_reference(native_library, strength, beam):
    experiment = Experiment(3, 1, 0.03)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    decoder = FrontierTNDecoder(
        context,
        FrontierConfig(beam=beam, refine_beam=beam, lookahead_strength=strength),
        native_library,
    )
    events = packed_all(8)
    actual, diagnostics = decoder.decode_with_diagnostics(events)
    for i, event in enumerate(events):
        expected, margin, pruned, valid = reference_contract(decoder.plan, event, beam)
        if valid:
            assert actual[i, 0] == expected
            assert diagnostics["pruned_states"][i] == pruned
            assert diagnostics["margin"][i] == pytest.approx(margin, abs=1e-12)
        else:
            assert diagnostics["fallback"][i]


def test_correlated_separators_offsets_and_tie(native_library):
    for text, events, expected in [
        ("error(1) L0\nerror(0.5) D0 L0 ^ D1", [[0], [3]], [[1], [0]]),
        ("error(1) L0\nerror(0.5) L0", [[], []], [[0], [0]]),
    ]:
        decoder = FrontierTNDecoder(context_for_dem(text), library=native_library)
        np.testing.assert_array_equal(
            decoder.decode_batch(np.array(events, dtype=np.uint8)), expected
        )


def test_repeated_measurements_and_batch_independence(native_library):
    experiment = Experiment(3, 3, 0.005)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    decoder = FrontierTNDecoder(context, FrontierConfig(beam=4, refine_beam=16), native_library)
    assert context.num_detectors > 12
    assert decoder.plan.max_active <= 63
    events = circuit.compile_detector_sampler(seed=117).sample(73, bit_packed=True)
    original = events.copy()
    events.flags.writeable = False
    whole = decoder.decode_batch(events)
    scalar = np.concatenate([decoder.decode_batch(events[i : i + 1]) for i in range(len(events))])
    np.testing.assert_array_equal(whole, scalar)
    np.testing.assert_array_equal(decoder.decode_batch(events[::-1].copy()), whole[::-1])
    np.testing.assert_array_equal(events, original)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("width", [70, 127])
def test_two_word_frontier_matches_full_model_likelihoods(
    native_library, monkeypatch, reverse, width
):
    # Opening this correlated mechanism first forces live detector lanes across
    # the word boundary. Each syndrome has exactly two complete fault patterns,
    # so its logical likelihood can be checked without enumerating the fault space.
    import tensor_decoder._small.frontier as module

    probability = 0.02
    text = "error(0.01) L0 " + " ^ ".join(f"D{i}" for i in range(width))
    text += "\n" + "\n".join(f"error({probability}) D{i}" for i in range(width))
    text += "\nerror(1) D65 L0"
    context = context_for_dem(text)

    def forced_order(model, reverse=False):
        suffix = list(range(1, len(model.probabilities)))
        return [0] + (suffix[::-1] if reverse else suffix)

    monkeypatch.setattr(module, "_order_faults", forced_order)
    decoder = FrontierTNDecoder(
        context,
        FrontierConfig(beam=4, refine_beam=4, max_frontier_bits=127, reverse_order=reverse),
        native_library,
    )
    assert decoder.plan.max_active == width
    unpacked = np.random.default_rng(42).integers(0, 2, (16, width), dtype=np.uint8)
    unpacked[:2] = np.arange(2, dtype=np.uint8)[:, None]
    events = np.packbits(unpacked, axis=1, bitorder="little")
    prediction, details = decoder.decode_with_diagnostics(events)
    canonical = unpacked.copy()
    canonical[:, 65] ^= 1
    weight = canonical.sum(axis=1)
    log0 = np.log(0.99) + weight * np.log(probability) + (width - weight) * np.log1p(-probability)
    log1 = np.log(0.01) + (width - weight) * np.log(probability) + weight * np.log1p(-probability)
    np.testing.assert_array_equal(prediction[:, 0], (log1 > log0).astype(np.uint8) ^ 1)
    assert details["unpruned"].all()
    for row, predicted in zip(events, prediction[:, 0], strict=True):
        assert reference_contract(decoder.plan, row, 4)[0] == predicted


def test_high_word_lookahead_pruning_matches_reference(native_library, monkeypatch):
    import tensor_decoder._small.frontier as module

    width = 70
    text = "error(0.02) L0 " + " ^ ".join(f"D{i}" for i in range(width))
    text += "\nerror(0.03) " + " ^ ".join(f"D{i}" for i in range(0, width, 2))
    text += "\nerror(0.05) D64 D66"
    text += "\n" + "\n".join(f"error(0.01) D{i}" for i in range(width))
    monkeypatch.setattr(
        module, "_order_faults", lambda model, reverse=False: list(range(len(model.probabilities)))
    )
    decoder = FrontierTNDecoder(
        context_for_dem(text),
        FrontierConfig(beam=1, refine_beam=1, max_frontier_bits=127),
        native_library,
    )
    bits = np.random.default_rng(112).integers(0, 2, (16, width), dtype=np.uint8)
    events = np.packbits(bits, axis=1, bitorder="little")
    prediction, details = decoder.decode_with_diagnostics(events)
    assert (details["pruned_states"] > 0).all()
    assert not details["fallback"].any()
    for i, row in enumerate(events):
        label, margin, pruned, valid = reference_contract(decoder.plan, row, 1)
        assert valid and prediction[i, 0] == label
        assert details["pruned_states"][i] == pruned
        assert details["margin"][i] == pytest.approx(margin, abs=1e-12)


def test_reverse_order_truncation_matches_reference(native_library):
    context = context_for_dem(
        "error(0.03) D0 D1 L0 ^ D2 D3\nerror(0.1) D0\nerror(0.1) D1\n"
        "error(0.1) D2\nerror(0.1) D3\nerror(0.07) D0 D2"
    )
    decoder = FrontierTNDecoder(
        context, FrontierConfig(beam=1, refine_beam=1, reverse_order=True), native_library
    )
    events = packed_all(4)
    prediction = decoder.decode_batch(events)
    for row, predicted in zip(events, prediction[:, 0], strict=True):
        assert reference_contract(decoder.plan, row, 1)[0] == predicted


def test_frontier_caps_before_large_state_or_schedule_allocation():
    context = context_for_dem("error(0.1) D0 D1 L0")
    with pytest.raises(ValueError, match="resource cap"):
        compile_frontier(context, FrontierConfig(max_detectors=1))
    with pytest.raises(ValueError, match="bit-lane"):
        compile_frontier(context, FrontierConfig(max_frontier_bits=1))


@pytest.mark.parametrize("basis", ["x", "z"])
def test_unpruned_surface_contraction_matches_full_logical_ml_lut(native_library, basis):
    experiment = Experiment(3, 1, 0.01, basis=basis)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    decoder = FrontierTNDecoder(context, FrontierConfig(beam=128, refine_beam=128), native_library)
    events = packed_all(context.num_detectors)
    actual, diagnostics = decoder.decode_with_diagnostics(events)
    assert diagnostics["unpruned"].all()
    np.testing.assert_array_equal(actual, LUTDecoder(context).decode_batch(events))


def test_zero_syndrome_shortcut_bound_against_exhaustive_random_models():
    rng = np.random.default_rng(729)
    for _ in range(60):
        faults = [
            (
                float(rng.choice([0.001, 0.02, 0.15, 0.5, 0.8, 1])),
                int(rng.integers(0, 8)),
                int(rng.integers(0, 2)),
            )
            for _ in range(7)
        ]
        text = "\n".join(
            f"error({p}) "
            + " ".join([f"D{i}" for i in range(3) if ds >> i & 1] + (["L0"] if ls else []))
            for p, ds, ls in faults
        )
        context = context_for_dem(text)
        plan = compile_frontier(context)
        if not plan.zero_margin:
            continue
        original_model = FaultModel.from_context(context)
        syndrome = bit_mask(tuple(np.flatnonzero(original_model.detector_offset)))
        masses = np.zeros(2)
        for pattern in itertools.product((0, 1), repeat=len(faults)):
            p, ds, ls = 1.0, 0, 0
            for on, (prior, d, logical) in zip(pattern, faults, strict=True):
                p *= prior if on else 1 - prior
                if on:
                    ds ^= d
                    ls ^= logical
            if ds == syndrome:
                masses[ls] += p
        assert masses[plan.logical_offset] > masses[1 - plan.logical_offset]
        assert plan.zero_margin <= abs(masses[0] - masses[1]) / sum(masses) + 1e-12

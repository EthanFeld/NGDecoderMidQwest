"""Independent exhaustive checks of projected upper and sparse lower bounds."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from test_small_frontier import context_for_dem, packed_all

from tensor_decoder import DecoderContext, Experiment
from tensor_decoder._small.certificates import compile_certificates, projected_joint
from tensor_decoder._small.frontier import (
    FrontierConfig,
    FrontierTNDecoder,
    _merge_identical_faults,
)
from tensor_decoder.build_native import support_library
from tensor_decoder.model import FaultModel, bit_mask


def oracle(model):
    result = np.zeros((1 << len(model.detector_offset), 2))
    for chosen in itertools.product((0, 1), repeat=len(model.probabilities)):
        mass, syndrome, logical = 1.0, 0, 0
        for on, p, ds, ls in zip(
            chosen, model.probabilities, model.detectors, model.observables, strict=True
        ):
            mass *= p if on else 1 - p
            if on:
                syndrome ^= bit_mask(ds)
                logical ^= int(bool(ls))
        result[syndrome, logical] += mass
    return result


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_projected_marginal_equals_independent_full_distribution(bits):
    context = context_for_dem(
        "error(0.1) D0 L0\nerror(0.3) D1 D2\nerror(0.2) D0 D2 L0\nerror(0.8) D2"
    )
    model = _merge_identical_faults(FaultModel.from_context(context))
    joint = oracle(model)
    checks = tuple(range(bits))
    expected = np.zeros((2, 1 << bits))
    for syndrome, row in enumerate(joint):
        expected[:, syndrome & ((1 << bits) - 1)] += row
    np.testing.assert_allclose(np.exp(projected_joint(model, checks)), expected, atol=1e-14)


def test_cached_labels_and_margins_are_valid_for_random_exhaustive_models():
    rng = np.random.default_rng(20261121)
    audited = 0
    for _ in range(40):
        faults = [
            (
                float(rng.choice([0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 1])),
                int(rng.integers(0, 16)),
                int(rng.integers(0, 2)),
            )
            for _ in range(8)
        ]
        text = "\n".join(
            f"error({p}) "
            + " ".join([f"D{i}" for i in range(4) if ds >> i & 1] + (["L0"] if ls else []))
            for p, ds, ls in faults
        )
        context = context_for_dem(text)
        model = _merge_identical_faults(FaultModel.from_context(context))
        truth = oracle(model)
        cache = compile_certificates(model, context.num_detectors, bits=2)
        for key, label, margin in zip(cache.keys, cache.labels, cache.margins, strict=True):
            row = truth[int(key)]
            assert row[int(label)] > row[1 - int(label)]
            assert margin <= abs(row[0] - row[1]) / sum(row) + 1e-12
            audited += 1
    assert audited > 20


def test_native_cache_path_respects_offsets_and_batch_independence():
    library = support_library()
    if not library.exists():
        pytest.skip("native kernel is not built")
    context = context_for_dem(
        "error(1) D0 L0\nerror(0.03) D0\nerror(0.01) D1 L0\nerror(0.02) D0 D1"
    )
    decoder = FrontierTNDecoder(
        context, FrontierConfig(beam=1, refine_beam=1, certificate_bits=2), library
    )
    events = packed_all(2)
    actual, diagnostics = decoder.decode_with_diagnostics(events)
    model = FaultModel.from_context(context)
    truth = oracle(model)
    offset = bit_mask(tuple(np.flatnonzero(model.detector_offset)))
    for i in np.flatnonzero(diagnostics["certified"]):
        expected = int(np.argmax(truth[i ^ offset])) ^ int(model.observable_offset[0])
        assert actual[i, 0] == expected
    assert diagnostics["certified"].any()
    np.testing.assert_array_equal(
        actual, np.concatenate([decoder.decode_batch(events[i : i + 1]) for i in range(4)])
    )


def test_disabled_cache_does_not_allocate_projected_tables():
    context = context_for_dem("error(0.1) D0 L0")
    model = FaultModel.from_context(context)
    cache = compile_certificates(model, 64, bits=16)
    assert len(cache.keys) == 0


@pytest.mark.parametrize("text", ["", "detector D0", "error(1) D0 L0", "error(0.1) L0"])
def test_empty_and_deterministic_models_compile_certificates(text):
    context = context_for_dem(text)
    model = _merge_identical_faults(FaultModel.from_context(context))
    cache = compile_certificates(model, context.num_detectors, bits=16)
    assert len(cache.keys) == 1
    assert cache.keys[0] == 0
    assert cache.labels[0] == 0


def test_prediction_only_path_preserves_numerical_fallback():
    library = support_library()
    if not library.exists():
        pytest.skip("native kernel is not built")
    context = context_for_dem("error(1e-200) D0 D1\nerror(1e-200) D1 D2\nerror(0.1) D0 D2 L0")
    decoder = FrontierTNDecoder(context, FrontierConfig(certificate_bits=0), library)
    events = np.array([[5]], dtype=np.uint8)
    expected, diag = decoder.decode_with_diagnostics(events)
    assert diag["fallback"][0]
    np.testing.assert_array_equal(decoder.decode_batch(events), expected)


def test_static_certified_shortcuts_preserve_batch_and_call_order_independence():
    library = support_library()
    if not library.exists():
        pytest.skip("native kernel is not built")
    experiment = Experiment(3, 3, 0.005)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    decoder = FrontierTNDecoder(context, FrontierConfig(certificate_bits=12), library)
    events = circuit.compile_detector_sampler(seed=190427).sample(257, bit_packed=True)
    original = events.copy()
    events.flags.writeable = False
    cache = decoder._certified_predictions.copy()
    full = decoder.decode_batch(events)
    with_diagnostics, diag = decoder.decode_with_diagnostics(events)
    np.testing.assert_array_equal(with_diagnostics, full)
    assert diag["certified"].mean() > 0.5
    scalar = np.concatenate([decoder.decode_batch(events[i : i + 1]) for i in range(len(events))])
    np.testing.assert_array_equal(full, scalar)
    np.testing.assert_array_equal(decoder.decode_batch(events[::-1].copy()), full[::-1])
    np.testing.assert_array_equal(events, original)
    assert decoder._certified_predictions == cache


def test_cache_cap_leaves_larger_frontier_decoder_available():
    library = support_library()
    if not library.exists():
        pytest.skip("native kernel is not built")
    experiment = Experiment(5, 5, 0.005)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    decoder = FrontierTNDecoder(
        context, FrontierConfig(beam=1, refine_beam=1, certificate_bits=16), library
    )
    assert len(decoder.plan.certificates.keys) == 0
    events = circuit.compile_detector_sampler(seed=5).sample(2, bit_packed=True)
    np.testing.assert_array_equal(
        decoder.decode_batch(events),
        np.concatenate([decoder.decode_batch(events[i : i + 1]) for i in range(2)]),
    )

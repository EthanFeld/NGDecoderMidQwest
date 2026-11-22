"""Gate safety, correlation baseline attribution, and packed shot independence."""

import numpy as np
import pymatching
import pytest

from tensor_decoder import DecoderContext, Experiment
from tensor_decoder._small.policy import CascadeConfig, CascadeTNDecoder
from tensor_decoder.build_native import support_library


def context_for(distance=5):
    experiment = Experiment(distance, distance, 0.005)
    circuit = experiment.circuit()
    return DecoderContext(experiment, circuit, circuit.detector_error_model(decompose_errors=True))


def native_path():
    path = support_library()
    if not path.exists():
        pytest.skip("optional native kernel is not built")
    return path


@pytest.mark.parametrize("consensus", [False, True])
def test_gate_is_independent_of_batch_order_and_warmup(consensus):
    context = context_for()
    decoder = CascadeTNDecoder(
        context,
        CascadeConfig(tensor_beam=4, reverse_consensus=consensus),
        native_path(),
    )
    sample = context.circuit.compile_detector_sampler(seed=3701).sample(2048, bit_packed=True)
    ordinary = decoder._match(sample, False)
    correlated = decoder._match(sample, True)
    independent = pymatching.Matching.from_detector_error_model(context.dem).decode_batch(
        sample, bit_packed_shots=True, bit_packed_predictions=True
    )
    np.testing.assert_array_equal(decoder._match(sample, False), ordinary)
    np.testing.assert_array_equal(ordinary, independent)
    different = ordinary[:, 0] != correlated[:, 0]
    assert different.any()
    # Keep both gate routes but bound the optional native test's runtime.
    events = np.concatenate((sample[different][:8], sample[~different][:24]))
    original = events.copy()
    events.flags.writeable = False
    prediction, details = decoder.decode_with_diagnostics(events)
    np.testing.assert_array_equal(details["tensor_searched"][:8], True)
    np.testing.assert_array_equal(details["tensor_searched"][8:], False)
    scalar = np.concatenate([decoder.decode_batch(row[None]) for row in events])
    np.testing.assert_array_equal(scalar, prediction)
    np.testing.assert_array_equal(decoder.decode_batch(events[::-1].copy()), prediction[::-1])
    np.testing.assert_array_equal(decoder.decode_batch(events), prediction)
    np.testing.assert_array_equal(events, original)
    assert decoder.decode_batch(events[:0]).shape == (0, 1)


def test_capacity_fallback_is_exactly_correlated_matching_without_native():
    context = context_for()
    decoder = CascadeTNDecoder(
        context,
        CascadeConfig(tensor_max_frontier_bits=1),
        library="this_library_must_not_be_loaded.dll",
    )
    assert decoder.mode == "correlated_matching_only"
    assert "bit-lane" in decoder.tensor_unavailable_reason
    pure = build_correlated_decoder(context)
    events = context.circuit.compile_detector_sampler(seed=53).sample(512, bit_packed=True)
    direct = pymatching.Matching.from_detector_error_model(context.dem, enable_correlations=True)
    expected = direct.decode_batch(
        events, bit_packed_shots=True, bit_packed_predictions=True, enable_correlations=True
    )
    for candidate in (decoder, pure):
        prediction, details = candidate.decode_with_diagnostics(events)
        np.testing.assert_array_equal(prediction, expected)
        assert not details["tensor_searched"].any()


def test_small_policy_retains_certified_adaptive_behavior():
    context = context_for(3)
    decoder = CascadeTNDecoder(context, library=native_path())
    assert decoder.mode == "small_certified_adaptive"
    events = context.circuit.compile_detector_sampler(seed=53).sample(256, bit_packed=True)
    np.testing.assert_array_equal(decoder.decode_batch(events), decoder.small.decode_batch(events))


def test_failed_tensor_search_keeps_correlated_result():
    # Force a native failure on a real disagreement. The frontier's ordinary
    # matching emergency fallback must not overwrite the stronger fast pass.
    context = context_for()
    decoder = CascadeTNDecoder(context, CascadeConfig(tensor_beam=2), native_path())
    sample = context.circuit.compile_detector_sampler(seed=3701).sample(2048, bit_packed=True)
    ordinary, correlated = decoder._match(sample, False), decoder._match(sample, True)
    selected = ordinary[:, 0] != correlated[:, 0]
    events = np.ascontiguousarray(sample[selected][:2])
    assert len(events) == 2

    class FailedTensor:
        def decode_with_diagnostics(self, rows):
            return decoder._match(rows, False), {"fallback": np.ones(len(rows), dtype=bool)}

    decoder.tensor = FailedTensor()
    expected = decoder._match(events, True)
    prediction, details = decoder.decode_with_diagnostics(events)
    np.testing.assert_array_equal(prediction, expected)
    assert details["tensor_failed"].all()


@pytest.mark.parametrize("reverse_fails", [False, True])
def test_reverse_disagreement_or_failure_vetoes_tensor_change(reverse_fails):
    context = context_for()
    decoder = CascadeTNDecoder(context, CascadeConfig(tensor_beam=2), native_path())
    sample = context.circuit.compile_detector_sampler(seed=3701).sample(2048, bit_packed=True)
    plain, correlated = decoder._match(sample, False), decoder._match(sample, True)
    events = np.ascontiguousarray(sample[plain[:, 0] != correlated[:, 0]][:2])

    class Forward:
        def decode_with_diagnostics(self, rows):
            return decoder._match(rows, False), {"fallback": np.zeros(len(rows), dtype=bool)}

    class Reverse:
        def decode_with_diagnostics(self, rows):
            return decoder._match(rows, not reverse_fails), {
                "fallback": np.full(len(rows), reverse_fails, dtype=bool)
            }

    decoder.tensor, decoder.reverse = Forward(), Reverse()
    expected = decoder._match(events, True)
    np.testing.assert_array_equal(decoder.decode_batch(events), expected)
    prediction, details = decoder.decode_with_diagnostics(events)
    np.testing.assert_array_equal(prediction, expected)
    assert details["reverse_searched"].all()
    assert details["tensor_failed"].all() == reverse_fails


@pytest.mark.parametrize("value", [0, 128, True])
def test_invalid_lane_limits(value):
    with pytest.raises(ValueError):
        CascadeConfig(tensor_max_frontier_bits=value)


def test_nonzero_padding_rejected_before_matching():
    context = context_for(3)
    # This model has 24 checks, so use a one-round code with eight plus one
    # declared check to exercise padding through a simple context instead.
    import stim

    dem = stim.DetectorErrorModel("error(0.1) D8 L0")
    circuit = stim.Circuit("M 0\n" + "DETECTOR rec[-1]\n" * 9 + "OBSERVABLE_INCLUDE(0)")
    context = DecoderContext(context.experiment, circuit, dem)
    decoder = build_correlated_decoder(context)
    with pytest.raises(ValueError, match="padding"):
        decoder.decode_batch(np.array([[0, 128]], dtype=np.uint8))


def build_correlated_decoder(context):
    return CascadeTNDecoder(
        context, CascadeConfig(enable_tensor=False, preserve_small_policy=False)
    )

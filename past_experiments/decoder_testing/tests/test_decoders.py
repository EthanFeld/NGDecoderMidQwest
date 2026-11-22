import numpy as np
import pymatching
import pytest

from qecbench import DecoderContext, Experiment, UnpackedAdapter
from qecbench.decoders import load_factory, validate_predictions
from qecbench.metrics import count_failures, wilson_interval


@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("noise", [0, 0.005])
def test_packed_matches_scalar_reference(distance, noise):
    experiment = Experiment(distance, distance, noise)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    events, actual = circuit.compile_detector_sampler(seed=22).sample(
        shots=128, separate_observables=True, bit_packed=True
    )
    events.flags.writeable = False
    packed = load_factory("pymatching")(context).decode_batch(events)
    validate_predictions(packed, 128, context.num_observables)
    matching = pymatching.Matching.from_detector_error_model(context.dem)
    matching.ensure_num_fault_ids(context.num_observables)
    unpacked = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")
    scalar = np.array([matching.decode(row) for row in unpacked], dtype=np.uint8)
    assert np.array_equal(packed, np.packbits(scalar, axis=1, bitorder="little"))
    if noise == 0:
        assert count_failures(packed, actual) == 0


def test_unpacked_adapter_little_endian_and_read_only():
    def callback(events):
        assert events.shape == (2, 10)
        assert not events.flags.writeable
        assert np.array_equal(np.flatnonzero(events[0]), [0, 8])
        return events[:, :9]

    adapter = UnpackedAdapter(callback, num_detectors=10, num_observables=9)
    events = np.array([[1, 1], [2, 0]], dtype=np.uint8)
    assert np.array_equal(adapter.decode_batch(events), events)


@pytest.mark.parametrize(
    "predictions",
    [
        np.array([[2]], dtype=np.uint8),
        np.array([[0.5]]),
        np.array([0], dtype=np.uint8),
    ],
)
def test_adapter_rejects_invalid_output(predictions):
    adapter = UnpackedAdapter(lambda events: predictions, 1, 1)
    with pytest.raises(ValueError):
        adapter.decode_batch(np.zeros((1, 1), dtype=np.uint8))


@pytest.mark.parametrize(
    "predictions",
    [
        [[0]],
        np.array([[0]], dtype=bool),
        np.zeros((1, 2), dtype=np.uint8),
        np.array([[128]], dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8)[:, ::2],
    ],
)
def test_packed_output_validation(predictions):
    with pytest.raises(ValueError):
        validate_predictions(predictions, 1, 1)


def test_failure_metric_counts_shots_across_multiple_observables():
    predicted = np.array([[255, 1], [0, 0], [0, 1]], dtype=np.uint8)
    actual = np.zeros_like(predicted)
    assert count_failures(predicted, actual) == 2


def test_confidence_intervals_include_zero_error_uncertainty():
    low, high = wilson_interval(0, 1000)
    assert low == pytest.approx(0)
    assert 0.003 < high < 0.004
    low, high = wilson_interval(1000, 1000)
    assert 0.996 < low < 0.997
    assert high == pytest.approx(1)
    assert wilson_interval(50, 100) == pytest.approx((0.4038315304, 0.5961684696))

"""FPGA metadata must stay aligned with original full-syndrome tail inputs."""

import numpy as np
from telescope import PRIOR_TILT, TENSOR, FrontBatch, TelescopeDecoder


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def decode_batch(self, events):
        self.calls.append((events.copy(), None))
        return self.predictions(events)

    @staticmethod
    def predictions(events):
        return (((events[:, :1] >> 0) ^ (events[:, :1] >> 4)) & 1).astype(np.uint8)

    def decode_with_front(self, events, residual, logical, valid, memory_indices):
        assert not events.flags.writeable
        self.calls.append((events.copy(), (
            residual.copy(), logical.copy(), valid.copy(), memory_indices.copy(),
        )))
        return self.predictions(events)


class BatchOnlyBackend:
    def __init__(self):
        self.calls = []

    def decode_batch(self, events):
        self.calls.append(events.copy())
        return RecordingBackend.predictions(events)


class RowFront:
    def __init__(self, batch):
        self.batch = batch
        self.index = 0

    def decode(self, memory):
        assert len(memory) == 1
        row = slice(self.index, self.index + 1)
        self.index += 1
        return FrontBatch(*(
            data[row] for data in (
                self.batch.accepted, self.batch.residual,
                self.batch.logical, self.batch.cycles,
            )
        ))


def fixture(backend=None):
    decoder = TelescopeDecoder.__new__(TelescopeDecoder)
    decoder.num_detectors = 8
    decoder.memory_indices = np.array([0, 2, 4, 6], dtype=np.intp)
    decoder.cuts = None
    decoder.backend = RecordingBackend() if backend is None else backend
    events = np.array([[1], [4], [16], [64], [17]], dtype=np.uint8)
    events.flags.writeable = False
    front = FrontBatch(
        np.array([True, False, False, True, False]),
        np.array([
            [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
            [0, 0, 0, 0], [1, 0, 1, 0],
        ], dtype=bool),
        np.array([1, 1, 0, 0, 0], dtype=np.uint8),
        np.array([5, 6, -1, 8, 9], dtype=np.int32),
    )
    decoder.front = RowFront(front)
    return decoder, events, front


def test_original_events_and_fpga_hints_are_forwarded_together():
    decoder, events, front = fixture()
    predictions, stages = decoder.complete_with_stages(events, front)
    np.testing.assert_array_equal(stages, [PRIOR_TILT, TENSOR, TENSOR, PRIOR_TILT, TENSOR])
    np.testing.assert_array_equal(predictions[:, 0], [1, 0, 1, 0, 0])
    supplied, (residual, logical, valid, mapping) = decoder.backend.calls[0]
    np.testing.assert_array_equal(supplied, events[[1, 2, 4]])
    np.testing.assert_array_equal(residual, front.residual[[1, 2, 4]])
    np.testing.assert_array_equal(logical, [1, 0, 0])
    np.testing.assert_array_equal(valid, [True, False, True])
    np.testing.assert_array_equal(mapping, decoder.memory_indices)


def test_stream_keeps_hints_aligned_and_flushes_last_partial_batch():
    decoder, events, front = fixture()
    records = list(decoder.decode_stream(events, defer_batch=2))
    assert {row[0]: row[1] for row in records} == {0: 1, 1: 0, 2: 1, 3: 0, 4: 0}
    assert len(decoder.backend.calls) == 2
    for call, selection in zip(decoder.backend.calls, ([1, 2], [4]), strict=True):
        supplied, (residual, logical, valid, _) = call
        np.testing.assert_array_equal(supplied, events[selection])
        np.testing.assert_array_equal(residual, front.residual[selection])
        np.testing.assert_array_equal(logical, front.logical[selection])
        np.testing.assert_array_equal(valid, front.cycles[selection] > 0)


def test_custom_backend_without_hint_api_uses_original_events():
    decoder, events, front = fixture(BatchOnlyBackend())
    decoder.complete_with_stages(events, front)
    np.testing.assert_array_equal(decoder.backend.calls[0], events[[1, 2, 4]])
    streamed, events, _ = fixture(BatchOnlyBackend())
    assert len(list(streamed.decode_stream(events, 2))) == 5
    np.testing.assert_array_equal(streamed.backend.calls[0], events[[1, 2]])
    np.testing.assert_array_equal(streamed.backend.calls[1], events[[4]])


def test_malformed_hint_shape_uses_full_syndrome_fallback():
    decoder, events, front = fixture()
    front.residual = np.zeros((len(events), 3), dtype=bool)
    decoder.complete_with_stages(events, front)
    supplied, hints = decoder.backend.calls[0]
    assert hints is None
    np.testing.assert_array_equal(supplied, events[[1, 2, 4]])


def test_nonbinary_hint_rows_are_marked_unavailable():
    decoder, events, front = fixture()
    front.residual = front.residual.astype(float)
    front.residual[1, 0] = np.nan
    decoder.complete_with_stages(events, front)
    _, (_, logical, valid, _) = decoder.backend.calls[0]
    np.testing.assert_array_equal(valid, [False, False, True])
    np.testing.assert_array_equal(logical, [0, 0, 0])


def test_guard_skips_malformed_hints_before_using_cut_calculations():
    decoder, events, front = fixture()
    decoder.cuts = object()
    front.residual = np.full(front.residual.shape, np.nan)
    decoder.complete_with_stages(events, front)
    _, (_, _, valid, _) = decoder.backend.calls[0]
    assert not valid.any()

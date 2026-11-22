"""Independent tiny-model probability oracle for tests only."""

import numpy as np

from tensor_decoder.model import FaultModel, bit_mask


class LUTDecoder:
    def __init__(self, context):
        width = context.num_detectors
        assert width <= 16, "test oracle only supports tiny models"
        model = FaultModel.from_context(context)
        size = 1 << (width + 1)
        addresses = np.arange(size)
        mass = np.zeros(size)
        mass[0] = 1
        for p, ds, ls in zip(model.probabilities, model.detectors, model.observables, strict=True):
            flip = bit_mask(ds) | (bit_mask(ls) << width)
            mass = (1 - p) * mass + p * mass[addresses ^ flip]
        offset = bit_mask(tuple(np.flatnonzero(model.detector_offset)))
        offset |= int(model.observable_offset[0]) << width
        mass = mass[addresses ^ offset]
        half = 1 << width
        self.labels = (mass[half:] > mass[:half]).astype(np.uint8)

    def decode_batch(self, events):
        return np.array(
            [[self.labels[int.from_bytes(row.tobytes(), "little")]] for row in events], np.uint8
        )

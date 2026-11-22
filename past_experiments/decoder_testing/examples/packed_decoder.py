"""Runnable native packed decoder template, with no unpacking or per-shot loops."""

import numpy as np
import pymatching

from qecbench import DecoderContext


class Decoder:
    def __init__(self, context: DecoderContext):
        self.matching = pymatching.Matching.from_detector_error_model(context.dem)
        self.matching.ensure_num_fault_ids(context.num_observables)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        return self.matching.decode_batch(
            detection_events, bit_packed_shots=True, bit_packed_predictions=True
        )


def build_decoder(context: DecoderContext) -> Decoder:
    return Decoder(context)

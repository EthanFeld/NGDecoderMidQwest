"""The independent correlated-PyMatching comparison and resource fallback."""

from __future__ import annotations

from .model import validate_input


class CorrelatedMatching:
    def __init__(self, context):
        import pymatching

        if context.num_observables != 1:
            raise ValueError("decoder requires exactly one logical observable")
        self.num_detectors = context.num_detectors
        self.matching = pymatching.Matching.from_detector_error_model(
            context.dem, enable_correlations=True
        )
        self.matching.ensure_num_fault_ids(1)

    def decode_batch(self, events):
        validate_input(events, self.num_detectors)
        return self.matching.decode_batch(
            events, bit_packed_shots=True, bit_packed_predictions=True, enable_correlations=True
        )


def build_decoder(context):
    return CorrelatedMatching(context)

"""Runnable unpacked decoder example; replace the backend with your algorithm."""

import pymatching

from qecbench import DecoderContext, UnpackedAdapter


def build_decoder(context: DecoderContext) -> UnpackedAdapter:
    matching = pymatching.Matching.from_detector_error_model(context.dem)
    matching.ensure_num_fault_ids(context.num_observables)
    return UnpackedAdapter(
        decode=matching.decode_batch,
        num_detectors=context.num_detectors,
        num_observables=context.num_observables,
    )

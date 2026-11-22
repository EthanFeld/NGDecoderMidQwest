"""Factory for the repository's frozen qecbench packed-batch contract."""

import numpy as np

from .compression import compile_graph
from .decoder import SparseDecoder
from .dem import parse_dem
from .geometry import LatticeCodec


class PackedDecoder:
    def __init__(self, decoder):
        self.decoder = decoder
        self.num_detectors = decoder.graph.num_detectors
        self.num_observables = decoder.graph.num_observables

    def decode_batch(self, detection_events):
        data = np.asarray(detection_events)
        if (data.dtype != np.uint8 or data.ndim != 2
                or data.shape[1] != (self.num_detectors + 7) // 8):
            raise ValueError("expected packed little-endian uint8 detector batches")
        unpacked = np.unpackbits(data, axis=1, count=self.num_detectors, bitorder="little")
        predictions = self.decoder.decode_batch(unpacked)
        return np.ascontiguousarray(np.packbits(predictions, axis=1, bitorder="little"))


def build_decoder(context):
    experiment = context.experiment
    if experiment.geometry != "rotated":
        raise ValueError("MORPH-31 supports rotated surface-code memories")
    reference = parse_dem(context.dem)
    codec = LatticeCodec(experiment.distance, experiment.rounds, experiment.basis)
    graph = compile_graph(reference, codec)
    return PackedDecoder(SparseDecoder(graph))

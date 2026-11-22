"""The decoder author API: packed input/output, one factory per experiment."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import stim

from .config import Experiment


@dataclass(frozen=True)
class DecoderContext:
    experiment: Experiment
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel

    @property
    def num_detectors(self) -> int:
        return self.circuit.num_detectors

    @property
    def num_observables(self) -> int:
        return self.circuit.num_observables


class PackedDecoder(Protocol):
    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """Return little-endian packed logical flip predictions, without mutating input."""
        ...


@dataclass
class UnpackedAdapter:
    decode: Callable[[np.ndarray], np.ndarray]
    num_detectors: int
    num_observables: int

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        unpacked = np.unpackbits(
            detection_events, axis=1, count=self.num_detectors, bitorder="little"
        )
        unpacked.flags.writeable = False
        predictions = self.decode(unpacked)
        shape = (len(detection_events), self.num_observables)
        if not isinstance(predictions, np.ndarray) or predictions.shape != shape:
            raise ValueError(f"unpacked decoder must return a NumPy array with shape {shape}")
        if predictions.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)):
            raise ValueError("unpacked decoder predictions must have bool or uint8 dtype")
        if np.any(predictions > 1):
            raise ValueError("unpacked decoder predictions must be binary (0 or 1)")
        return np.packbits(predictions, axis=1, bitorder="little")


class _ZeroDecoder:
    def __init__(self, context: DecoderContext):
        self.width = (context.num_observables + 7) // 8

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        return np.zeros((len(detection_events), self.width), dtype=np.uint8)


class _MatchingDecoder:
    def __init__(self, context: DecoderContext):
        import pymatching

        self.matching = pymatching.Matching.from_detector_error_model(context.dem)
        # Noise-free models may have no edges carrying logical fault IDs.
        self.matching.ensure_num_fault_ids(context.num_observables)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        return self.matching.decode_batch(
            detection_events, bit_packed_shots=True, bit_packed_predictions=True
        )


def load_factory(spec: str) -> Callable[[DecoderContext], PackedDecoder]:
    if spec == "zero":
        return _ZeroDecoder
    if spec == "pymatching":
        return _MatchingDecoder
    module_name, _, factory_name = spec.partition(":")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    if not callable(factory):
        raise ValueError(f"decoder factory {spec!r} is not callable")
    return factory


def validate_predictions(predictions: np.ndarray, shots: int, observables: int) -> None:
    shape = (shots, (observables + 7) // 8)
    if not isinstance(predictions, np.ndarray) or predictions.shape != shape:
        raise ValueError(f"packed decoder must return a NumPy array with shape {shape}")
    if predictions.dtype != np.uint8 or not predictions.flags.c_contiguous:
        raise ValueError("packed decoder predictions must be C-contiguous uint8")
    remainder = observables % 8
    if remainder and np.any(predictions[:, -1] & np.uint8(255 ^ ((1 << remainder) - 1))):
        raise ValueError("packed decoder predictions have nonzero unused padding bits")

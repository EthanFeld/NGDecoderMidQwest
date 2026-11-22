"""Complete DEM faults and packed input validation; no harness dependency."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import stim

from .context import DecoderContext


def mechanisms(
    dem: stim.DetectorErrorModel,
) -> Iterator[tuple[float, tuple[int, ...], tuple[int, ...]]]:
    """Expand repeats lazily; XOR repeated targets across all separator components."""

    def detector_shift(block):
        shift = 0
        for instruction in block:
            if isinstance(instruction, stim.DemRepeatBlock):
                shift += instruction.repeat_count * detector_shift(instruction.body_copy())
            elif instruction.type == "shift_detectors":
                shift += instruction.targets_copy()[0]
        return shift

    def walk(block, offset):
        for instruction in block:
            if isinstance(instruction, stim.DemRepeatBlock):
                body = instruction.body_copy()
                if body.num_errors == 0:
                    offset += instruction.repeat_count * detector_shift(body)
                    continue
                for _ in range(instruction.repeat_count):
                    offset = yield from walk(body, offset)
            elif instruction.type == "shift_detectors":
                offset += instruction.targets_copy()[0]
            elif instruction.type == "error":
                detectors, observables = set(), set()
                for target in instruction.targets_copy():
                    if target.is_relative_detector_id():
                        detectors.symmetric_difference_update((offset + target.val,))
                    elif target.is_logical_observable_id():
                        observables.symmetric_difference_update((target.val,))
                yield (
                    instruction.args_copy()[0],
                    tuple(sorted(detectors)),
                    tuple(sorted(observables)),
                )
        return offset

    yield from walk(dem, 0)


@dataclass
class FaultModel:
    probabilities: np.ndarray
    detectors: tuple[tuple[int, ...], ...]
    observables: tuple[tuple[int, ...], ...]
    detector_offset: np.ndarray
    observable_offset: np.ndarray

    def project_detectors(self, indices: np.ndarray) -> FaultModel:
        """Marginalize discarded detector observations without splitting mechanisms."""
        mapping = {int(old): new for new, old in enumerate(indices)}
        probabilities, columns, observables = [], [], []
        for p, ds, obs in zip(self.probabilities, self.detectors, self.observables, strict=True):
            projected = tuple(mapping[d] for d in ds if d in mapping)
            if projected or obs:
                probabilities.append(p)
                columns.append(projected)
                observables.append(obs)
        return FaultModel(
            np.asarray(probabilities, dtype=np.float64),
            tuple(columns),
            tuple(observables),
            self.detector_offset[indices].copy(),
            self.observable_offset.copy(),
        )

    def merge_equivalent_faults(self) -> FaultModel:
        """Replace independent identical detector/logical columns by their odd parity.

        This preserves the entire joint syndrome/logical distribution, including
        correlations and distinct logical actions sharing a detector column.
        """
        log_products: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
        for probability, detectors, observables in zip(
            self.probabilities, self.detectors, self.observables, strict=True
        ):
            key = (detectors, observables)
            term = -math.inf if probability == 0.5 else math.log1p(-2 * probability)
            log_products[key] = log_products.get(key, 0.0) + term
        probabilities = -0.5 * np.expm1(np.asarray(list(log_products.values())))
        return FaultModel(
            probabilities,
            tuple(key[0] for key in log_products),
            tuple(key[1] for key in log_products),
            self.detector_offset.copy(),
            self.observable_offset.copy(),
        )

    @classmethod
    def from_context(cls, context: DecoderContext) -> FaultModel:
        probabilities, detector_columns, observable_columns = [], [], []
        detector_offset = np.zeros(context.num_detectors, dtype=np.uint8)
        observable_offset = np.zeros(context.num_observables, dtype=np.uint8)
        for p, detectors, observables in mechanisms(context.dem):
            if p > 0.5:
                detector_offset[list(detectors)] ^= 1
                observable_offset[list(observables)] ^= 1
                p = 1 - p
            if p == 0:
                continue
            probabilities.append(p)
            detector_columns.append(detectors)
            observable_columns.append(observables)
        return cls(
            np.asarray(probabilities, dtype=np.float64),
            tuple(detector_columns),
            tuple(observable_columns),
            detector_offset,
            observable_offset,
        )


def bit_mask(indices: tuple[int, ...]) -> int:
    result = 0
    for index in indices:
        result ^= 1 << int(index)
    return result


def validate_input(events: np.ndarray, detectors: int) -> None:
    if (
        not isinstance(events, np.ndarray)
        or events.ndim != 2
        or events.shape[1] != (detectors + 7) // 8
        or events.dtype != np.uint8
        or not events.flags.c_contiguous
    ):
        raise ValueError("detection_events must be C-contiguous packed uint8 with detector width")
    remainder = detectors % 8
    if remainder and (
        int(np.bitwise_or.reduce(events[:, -1], initial=0)) & (255 ^ ((1 << remainder) - 1))
    ):
        raise ValueError("detection_events have nonzero padding bits")


def _column_matrix(columns, rows):
    from scipy.sparse import csr_matrix

    row_indices = np.asarray([row for column in columns for row in column], dtype=np.intp)
    column_indices = np.repeat(np.arange(len(columns)), [len(column) for column in columns])
    return csr_matrix(
        (np.ones(len(row_indices), dtype=np.uint8), (row_indices, column_indices)),
        shape=(rows, len(columns)),
        dtype=np.uint8,
    )


def memory_detector_indices(context: DecoderContext) -> np.ndarray:
    """Identify the logical-memory check basis in standard generated Stim circuits.

    Initialization/final readout expose the memory basis; interior checks at those
    same spatial sites measure that basis. Other-basis observations are discarded.
    This opt-in strategy is restricted to the contract's generated memory circuits.
    """
    expected = context.experiment.circuit().without_noise()
    if context.num_observables != 1 or context.circuit.without_noise() != expected:
        raise ValueError("memory_basis mode requires a standard generated surface-memory circuit")
    coordinates = context.circuit.get_detector_coordinates()
    if not coordinates or any(len(value) != 3 for value in coordinates.values()):
        raise ValueError("memory_basis mode requires three-dimensional detector coordinates")
    first = min(value[2] for value in coordinates.values())
    last = max(value[2] for value in coordinates.values())
    sites = {tuple(value[:2]) for value in coordinates.values() if value[2] == first}
    final_sites = {tuple(value[:2]) for value in coordinates.values() if value[2] == last}
    if first != 0 or last != context.experiment.rounds or sites != final_sites:
        raise ValueError("surface-memory boundary coordinates do not identify a unique check basis")
    return np.asarray(
        [i for i, value in coordinates.items() if tuple(value[:2]) in sites], dtype=np.intp
    )

"""Exact logical maximum-likelihood lookup tables for small detector models."""

from __future__ import annotations

import numpy as np

from qecbench import DecoderContext

from .model import FaultModel, bit_mask, validate_input
from .resources import LUTConfig, ResourceLimits, enforce_resources


class LUTDecoder:
    def __init__(
        self,
        context: DecoderContext,
        config: LUTConfig | None = None,
        limits: ResourceLimits | None = None,
    ):
        self.config = config or LUTConfig()
        self.resource_estimate = enforce_resources(context, "lut", self.config, limits)
        self.num_detectors = m = context.num_detectors
        self.num_observables = k = context.num_observables
        model = FaultModel.from_context(context)
        self.detector_offset = model.detector_offset
        logical_offset = bit_mask(tuple(np.flatnonzero(model.observable_offset)))
        addresses = np.arange(1 << (m + k), dtype=np.int64)
        log_probability = np.full(len(addresses), -np.inf, dtype=np.float64)
        log_probability[0] = 0
        scratch = np.empty_like(log_probability)
        for p, detectors, observables in zip(
            model.probabilities, model.detectors, model.observables, strict=True
        ):
            mask = bit_mask(detectors) | (bit_mask(observables) << m)
            # Convolve one independent mechanism, retaining its joint logical action.
            np.take(log_probability, addresses ^ mask, out=scratch)
            scratch += np.log(p)
            log_probability += np.log1p(-p)
            np.logaddexp(log_probability, scratch, out=log_probability)
        joint = log_probability.reshape(1 << k, 1 << m)
        # Sort final logical labels so ties choose their smallest integer value.
        best = np.argmax(joint[np.arange(1 << k) ^ logical_offset], axis=0)
        self.support = np.isfinite(np.max(joint, axis=0))
        predictions = ((best[:, None] >> np.arange(k)) & 1).astype(np.uint8)
        self.table = np.packbits(predictions, axis=1, bitorder="little")
        self.address_weights = np.left_shift(np.uint64(1), np.arange(m, dtype=np.uint64))

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        validate_input(detection_events, self.num_detectors)
        output = np.empty((len(detection_events), (self.num_observables + 7) // 8), dtype=np.uint8)
        for start in range(0, len(detection_events), self.config.chunk_shots):
            stop = start + self.config.chunk_shots
            syndrome = np.unpackbits(
                detection_events[start:stop], axis=1, count=self.num_detectors, bitorder="little"
            )
            syndrome ^= self.detector_offset
            addresses = syndrome @ self.address_weights
            if not np.all(self.support[addresses]):
                raise ValueError("syndrome is outside the detector error model's support")
            output[start:stop] = self.table[addresses]
        return output


def build_decoder(context: DecoderContext) -> LUTDecoder:
    return LUTDecoder(context)

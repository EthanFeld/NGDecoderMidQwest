"""Correlation-aware fast pass and lazy tensor search on matching disagreements.

The gate is a deterministic function of the shot and the fixed DEM. Agreement
between matching passes is a heuristic, not a maximum-likelihood certificate.
PyMatching supplies the two matching passes; this module supplies the search gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model import validate_input
from .frontier import FrontierCapacityError, FrontierConfig, FrontierTNDecoder


@dataclass(frozen=True)
class CascadeConfig:
    tensor_beam: int = 128
    tensor_max_detectors: int = 1024
    tensor_max_mechanisms: int = 30000
    tensor_max_frontier_bits: int = 127
    preserve_small_policy: bool = True
    enable_tensor: bool = True
    reverse_consensus: bool = True

    def __post_init__(self):
        if type(self.tensor_beam) is not int or not 1 <= self.tensor_beam <= 1024:
            raise ValueError("tensor_beam must be an integer in [1,1024]")
        for name in ("tensor_max_detectors", "tensor_max_mechanisms"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.tensor_max_frontier_bits) is not int
            or not 1 <= self.tensor_max_frontier_bits <= 127
        ):
            raise ValueError("tensor_max_frontier_bits must be an integer in [1,127]")
        for name in ("preserve_small_policy", "enable_tensor", "reverse_consensus"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


class CascadeTNDecoder:
    """Bound median cost using matching, with bounded TN search for ambiguity.

    Small models retain the existing certified-cache/adaptive policy. Larger
    models run ordinary and correlated matching on one graph, then contract only
    their disagreements. By default a proposed tensor change must also agree
    with a contraction in the opposite detector order. Unsupported search sizes
    use correlated matching alone.
    No sampled truths, mutable syndrome cache, or online fitting are used.
    """

    def __init__(self, context, config: CascadeConfig | None = None, library=None):
        if context.num_observables != 1:
            raise ValueError("cascade prototype requires exactly one logical observable")
        self.config = config = config or CascadeConfig()
        self.num_detectors = context.num_detectors
        self.small = self.tensor = self.reverse = self.matching = None
        self.tensor_unavailable_reason = None
        if config.enable_tensor and config.preserve_small_policy and context.num_detectors <= 32:
            self.small = FrontierTNDecoder(
                context,
                FrontierConfig(certificate_bits=16, refine_beam=32, refine_margin=0.95),
                library,
            )
            self.mode = "small_certified_adaptive"
            return
        try:
            import pymatching

            self.matching = pymatching.Matching.from_detector_error_model(
                context.dem, enable_correlations=True
            )
        except TypeError as exc:
            raise RuntimeError(
                "cascade requires PyMatching >=2.3; install the scalable extra"
            ) from exc
        self.matching.ensure_num_fault_ids(1)
        if not config.enable_tensor:
            self.tensor_unavailable_reason = "tensor search disabled by configuration"
        elif (
            context.num_detectors > config.tensor_max_detectors
            or context.dem.num_errors > config.tensor_max_mechanisms
        ):
            self.tensor_unavailable_reason = "configured tensor model resource limit"
        else:
            try:
                self.tensor = FrontierTNDecoder(
                    context,
                    FrontierConfig(
                        beam=config.tensor_beam,
                        refine_beam=config.tensor_beam,
                        max_frontier_bits=config.tensor_max_frontier_bits,
                        max_detectors=config.tensor_max_detectors,
                        max_mechanisms=config.tensor_max_mechanisms,
                    ),
                    library,
                )
                if config.reverse_consensus:
                    self.reverse = FrontierTNDecoder(
                        context,
                        FrontierConfig(
                            beam=config.tensor_beam,
                            refine_beam=config.tensor_beam,
                            max_frontier_bits=config.tensor_max_frontier_bits,
                            max_detectors=config.tensor_max_detectors,
                            max_mechanisms=config.tensor_max_mechanisms,
                            reverse_order=True,
                        ),
                        library,
                    )
            except FrontierCapacityError as exc:
                self.tensor = self.reverse = None
                self.tensor_unavailable_reason = str(exc)
        self.mode = "matching_disagreement_tensor" if self.tensor else "correlated_matching_only"

    def _match(self, events, correlated):
        return self.matching.decode_batch(
            events,
            bit_packed_shots=True,
            bit_packed_predictions=True,
            enable_correlations=correlated,
        )

    def _execute(self, events, diagnostics):
        if self.small is not None:
            if not diagnostics:
                return self.small.decode_batch(events)
            output, details = self.small.decode_with_diagnostics(events)
            return output, {
                "tensor_searched": ~details["certified"],
                "tensor_failed": details["fallback"],
                "certified": details["certified"],
                "reverse_searched": np.zeros(len(events), dtype=bool),
            }
        validate_input(events, self.num_detectors)
        output = self._match(events, True)
        count = len(events)
        ordinary = None
        if not diagnostics and count == 1:
            if self.tensor is None:
                return output
            ordinary = self._match(events, False)
            if ordinary[0, 0] == output[0, 0]:
                return output
        selected = np.zeros(count, dtype=bool)
        failed = np.zeros(count, dtype=bool)
        reverse_searched = np.zeros(count, dtype=bool)
        if self.tensor is not None and count:
            if ordinary is None:
                ordinary = self._match(events, False)
            selected = ordinary[:, 0] != output[:, 0]
            if np.any(selected):
                rows = np.flatnonzero(selected)
                prediction, details = self.tensor.decode_with_diagnostics(
                    np.ascontiguousarray(events[selected])
                )
                # The tensor wrapper's emergency fallback is ordinary matching.
                # Keep the correlated result when contraction failed or underflowed.
                failed[rows] = details["fallback"]
                good = ~details["fallback"]
                if self.reverse is not None:
                    changes = good & (prediction[:, 0] != output[rows, 0])
                    if np.any(changes):
                        reverse_rows = np.flatnonzero(changes)
                        reverse_searched[rows[changes]] = True
                        alternative, reverse_details = self.reverse.decode_with_diagnostics(
                            np.ascontiguousarray(events[rows[changes]])
                        )
                        failed[rows[changes]] |= reverse_details["fallback"]
                        good[reverse_rows] = ~reverse_details["fallback"] & (
                            alternative[:, 0] == prediction[reverse_rows, 0]
                        )
                output[rows[good]] = prediction[good]
        if not diagnostics:
            return output
        return output, {
            "tensor_searched": selected,
            "tensor_failed": failed,
            "certified": np.zeros(count, dtype=bool),
            "reverse_searched": reverse_searched,
        }

    def decode_batch(self, events):
        if self.small is not None:
            return self.small.decode_batch(events)
        return self._execute(events, False)

    def decode_with_diagnostics(self, events):
        return self._execute(events, True)

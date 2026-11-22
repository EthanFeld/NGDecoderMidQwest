"""Batched min-sum Relay-BP with disordered memory and bounded GF(2) repair."""

from __future__ import annotations

import numpy as np

from qecbench import DecoderContext

from .model import FaultModel, SyndromeBasis, validate_input
from .resources import RelayBPConfig, ResourceLimits, enforce_resources


class RelayBPDecoder:
    def __init__(
        self,
        context: DecoderContext,
        config: RelayBPConfig | None = None,
        limits: ResourceLimits | None = None,
    ):
        self.config = config or RelayBPConfig()
        self.resource_estimate = enforce_resources(context, "relay_bp", self.config, limits)
        self.model = model = FaultModel.from_context(context)
        self.num_detectors = context.num_detectors
        self.num_observables = context.num_observables
        self.weights = np.log1p(-model.probabilities) - np.log(model.probabilities)
        self.prior = self.weights.copy()
        np.clip(self.prior, -self.config.llr_clip, self.config.llr_clip, out=self.prior)
        n = len(self.prior)
        edge_variables = np.repeat(np.arange(n), [len(ds) for ds in model.detectors])
        edge_checks = np.asarray([d for ds in model.detectors for d in ds], dtype=np.intp)
        order = np.argsort(edge_checks, kind="stable")
        self.edge_variables = edge_variables[order]
        counts = np.bincount(edge_checks, minlength=self.num_detectors)
        self.active_checks = np.flatnonzero(counts)
        self.check_starts = np.cumsum(counts[self.active_checks]) - counts[self.active_checks]
        self.edge_groups = np.repeat(np.arange(len(self.active_checks)), counts[self.active_checks])
        self.variable_order = np.argsort(self.edge_variables, kind="stable")
        counts = np.bincount(self.edge_variables, minlength=n)
        self.active_variables = np.flatnonzero(counts)
        self.variable_starts = (
            np.cumsum(counts[self.active_variables]) - counts[self.active_variables]
        )
        observable_variables = [[] for _ in range(self.num_observables)]
        for variable, observables in enumerate(model.observables):
            for observable in observables:
                observable_variables[observable].append(variable)
        self.observable_variables = tuple(
            np.asarray(vs, dtype=np.intp) for vs in observable_variables
        )
        rng = np.random.Generator(np.random.PCG64(self.config.seed))
        self.memory = rng.uniform(
            self.config.memory_min, self.config.memory_max, size=(self.config.legs, n)
        )
        self.memory[0] = self.config.first_memory
        self.basis = SyndromeBasis(model) if self.config.failure_policy == "repair" else None

    def _action(self, decisions: np.ndarray) -> np.ndarray:
        result = np.zeros((len(decisions), self.num_observables), dtype=np.uint8)
        for observable, variables in enumerate(self.observable_variables):
            if len(variables):
                result[:, observable] = np.logical_xor.reduce(decisions[:, variables], axis=1)
        return result

    def _residual(self, syndrome: np.ndarray, decisions: np.ndarray) -> np.ndarray:
        residual = syndrome.copy()
        if len(self.edge_variables):
            residual[:, self.active_checks] ^= np.logical_xor.reduceat(
                decisions[:, self.edge_variables], self.check_starts, axis=1
            )
        return residual

    def _decode_chunk(self, syndrome: np.ndarray) -> np.ndarray:
        config = self.config
        shots, n = len(syndrome), len(self.prior)
        marginals = np.broadcast_to(self.prior, (shots, n)).copy()
        best_action = np.zeros((shots, self.num_observables), dtype=np.uint8)
        best_weight = np.full(shots, np.inf)
        solutions = np.zeros(shots, dtype=np.int64)
        for memory in self.memory:
            live = solutions < config.solutions
            if not np.any(live):
                break
            # Each leg resets edge messages to priors, but inherits final marginals.
            messages = np.broadcast_to(
                self.prior[self.edge_variables], (shots, len(self.edge_variables))
            ).copy()
            for _ in range(config.iterations):
                bias = (1 - memory) * self.prior + memory * marginals
                incoming = np.zeros_like(marginals)
                if len(self.edge_variables):
                    magnitudes = np.abs(messages)
                    minima = np.minimum.reduceat(magnitudes, self.check_starts, axis=1)
                    at_min = magnitudes == minima[:, self.edge_groups]
                    min_counts = np.add.reduceat(at_min.astype(np.int64), self.check_starts, axis=1)
                    second = np.minimum.reduceat(
                        np.where(at_min, np.inf, magnitudes), self.check_starts, axis=1
                    )
                    outgoing = np.where(
                        at_min & (min_counts[:, self.edge_groups] == 1),
                        second[:, self.edge_groups],
                        minima[:, self.edge_groups],
                    )
                    np.minimum(outgoing, config.llr_clip, out=outgoing)
                    negative = messages < 0
                    parity = np.logical_xor.reduceat(negative, self.check_starts, axis=1)
                    signs = (
                        syndrome[:, self.active_checks][:, self.edge_groups]
                        ^ parity[:, self.edge_groups]
                        ^ negative
                    )
                    outgoing *= np.where(signs, -1, 1)
                    incoming[:, self.active_variables] = np.add.reduceat(
                        outgoing[:, self.variable_order], self.variable_starts, axis=1
                    )
                    total = bias + incoming
                    next_messages = total[:, self.edge_variables] - outgoing
                    np.clip(next_messages, -config.llr_clip, config.llr_clip, out=next_messages)
                    messages[live] = next_messages[live]
                else:
                    total = bias
                np.clip(total, -config.llr_clip, config.llr_clip, out=total)
                marginals[live] = total[live]
                decisions = marginals < 0
                valid = live & ~np.any(self._residual(syndrome, decisions), axis=1)
                # Row reductions keep arithmetic independent of other shots/batch shape.
                weights = np.sum(np.where(decisions, self.weights, 0), axis=1)
                improved = valid & (weights < best_weight)
                if np.any(improved):
                    best_action[improved] = self._action(decisions[improved])
                    best_weight[improved] = weights[improved]
                solutions[valid] += 1
                live[valid] = False
                if not np.any(live):
                    break
        failed = solutions == 0
        if np.any(failed):
            if self.basis is None:
                raise RuntimeError(
                    f"Relay-BP did not converge for {np.count_nonzero(failed)} shot(s)"
                )
            decisions = marginals[failed] < 0
            best_action[failed] = self.basis.repair(
                self._residual(syndrome[failed], decisions), self._action(decisions)
            )
        return best_action ^ self.model.observable_offset

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        validate_input(detection_events, self.num_detectors)
        output = np.empty((len(detection_events), (self.num_observables + 7) // 8), dtype=np.uint8)
        for start in range(0, len(detection_events), self.config.chunk_shots):
            stop = start + self.config.chunk_shots
            syndrome = np.unpackbits(
                detection_events[start:stop], axis=1, count=self.num_detectors, bitorder="little"
            )
            syndrome ^= self.model.detector_offset
            output[start:stop] = np.packbits(
                self._decode_chunk(syndrome), axis=1, bitorder="little"
            )
        return output


def build_decoder(context: DecoderContext) -> RelayBPDecoder:
    return RelayBPDecoder(context)

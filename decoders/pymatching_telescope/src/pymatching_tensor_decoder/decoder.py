"""The validated two-root logical-sector decoder. Public factory: build_decoder."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from threading import local
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
from scipy.sparse import vstack

from ._small.policy import CascadeTNDecoder
from .baseline import CorrelatedMatching
from .conditioned import ConditionedMatching
from .config import DecoderConfig
from .matching import FixedMatching
from .model import validate_input
from .path import PathProxy


class _Candidate(NamedTuple):
    row: np.ndarray
    pairs: np.ndarray
    base: int
    score: float
    routed: bool
    weights: np.ndarray | None


def sector_model(model):
    logical = np.asarray(model.observable_matrix.sum(axis=0)).ravel() != 0
    if not logical.any() or np.any(logical & (np.diff(model.check_matrix.indptr) != 1)):
        raise ValueError("logical sector requires logical edges on a boundary")
    matrix = vstack((model.check_matrix, model.observable_matrix), format="csc")
    return SimpleNamespace(
        indices=np.arange(matrix.shape[0]),
        check_matrix=matrix,
        memory_keys=model.memory_keys,
        other_keys=model.other_keys,
    )


class SectorSearch:
    def __init__(
        self, context, config=None, memory=None, *, device="cpu", matching_engine=None,
    ):
        self.config = config or DecoderConfig()
        self.matching_engine = matching_engine
        conditioned = (
            ConditionedMatching if matching_engine is None else matching_engine.conditioned
        )
        fixed = FixedMatching if matching_engine is None else matching_engine.fixed
        self.memory = memory or conditioned(context)
        self.other = conditioned(context, target_other=True)
        self.joint = conditioned(context, joint_check=True)
        self.memory_fixed = fixed(self.memory)
        self.other_fixed = fixed(self.other)
        self.sector = sector_model(self.memory)
        self.sector_fixed = fixed(self.sector)
        self.logical = np.array([label for _, label in self.memory.memory_keys], np.uint8)
        self.memory_weights = self.memory_fixed.weights
        self.other_weights = self.other_fixed.weights
        self.batched_weights = None
        if device != "cpu":
            from .gpu import CUDAModel

            self.batched_weights = CUDAModel(self.memory).weights
            if device == "cuda":
                self.memory_weights = self.batched_weights
                self.other_weights = CUDAModel(self.other).weights

    @staticmethod
    def initial_solve(memory, sector, syndrome, pairs, weights, make_weights):
        parity = memory.prepare(pairs)
        if weights is None:
            weights = make_weights(parity)
        candidate, cost, gain = memory.solve_details(syndrome, weights)
        proposed = int(candidate[0])
        # The unconstrained solve already gives the cost of its own sector.
        augmented = np.empty(len(syndrome) + 1, dtype=np.uint8)
        augmented[:-1] = syndrome
        augmented[-1] = 1 - proposed
        _, opposite, _ = sector.solve_details(augmented, weights)
        gap = opposite - cost if proposed == 0 else cost - opposite
        return proposed, gain, gap

    @staticmethod
    def global_edges(chosen, indices, auxiliary=None):
        chosen = chosen.copy()
        if auxiliary is not None:
            reverse = chosen[:, 0] == auxiliary
            chosen[reverse] = chosen[reverse, ::-1]
        result = np.empty_like(chosen)
        result[:, 0] = indices[chosen[:, 0]]
        boundary = (chosen[:, 1] < 0) | (chosen[:, 1] >= len(indices))
        result[:, 1] = -1
        result[~boundary, 1] = indices[chosen[~boundary, 1]]
        return result

    def memory_proposal(self, syndrome, other_edges, sector):
        weights = self.memory_weights(self.memory_fixed.prepare(other_edges))
        augmented = np.r_[syndrome[self.memory.indices], np.uint8(sector)]
        chosen = self.sector_fixed.solve_edges(augmented, weights)
        return self.global_edges(chosen, self.memory.indices, len(self.memory.indices))

    def other_proposal(self, syndrome, edges):
        weights = self.other_weights(self.other_fixed.prepare(edges))
        chosen = self.other_fixed.solve_edges(syndrome[self.other.indices], weights)
        return self.global_edges(chosen, self.other.indices)

    def evaluate(self, syndrome, baseline_edges):
        syndrome = np.asarray(syndrome)
        original_other = baseline_edges[~np.isin(baseline_edges[:, 0], self.memory.indices)]
        scores, seen = [[], []], set()

        def add(chain, sector):
            events = self.memory_fixed.events(chain)
            # This dense binary key is bijective with the old canonical edge set.
            # It avoids Python edge loops, and never hashes a syndrome to an answer.
            key = events.tobytes()
            if key in seen:
                return
            seen.add(key)
            try:
                _, cost = self.joint.joint_matching.decode(events, return_weight=True)
            except ValueError:
                return
            scores[sector].append(-float(cost))

        events = self.memory_fixed.events(baseline_edges)
        base_label = int(np.dot(self.logical, events[: len(self.logical)])) & 1
        add(baseline_edges, base_label)
        roots = [original_other]
        if self.config.reverse_seed:
            alternative = self.other_proposal(syndrome, baseline_edges)
            old_key = self.other_fixed.events(original_other).tobytes()
            if self.other_fixed.events(alternative).tobytes() != old_key:
                roots.append(alternative)
        for sector in (0, 1):
            for root in roots:
                other_edges = root
                for _ in range(self.config.iterations):
                    memory_edges = self.memory_proposal(syndrome, other_edges, sector)
                    combined = np.concatenate((memory_edges, other_edges))
                    add(combined, sector)
                    next_other = self.other_proposal(syndrome, combined)
                    add(np.concatenate((memory_edges, next_other)), sector)
                    if np.array_equal(next_other, other_edges):
                        break
                    other_edges = next_other
        masses = np.array([np.logaddexp.reduce(s) if s else -np.inf for s in scores])
        margin = float(masses[1] - masses[0]) if np.isfinite(masses).all() else float("nan")
        return {"prediction": int(margin > 0), "margin": margin, "states": len(seen)}


class TensorDecoder:
    def __init__(
        self, context, config=None, *, device="cpu", solve_workers=1, path_bucket=False,
        cuda_correlated=False, matching_engine=None,
    ):
        if device not in ("cpu", "cuda", "cuda-batch"):
            raise ValueError("tensor decoder device must be 'cpu', 'cuda', or 'cuda-batch'")
        if type(solve_workers) is not int or solve_workers < 1:
            raise ValueError("solve_workers must be a positive integer")
        if matching_engine is not None and cuda_correlated:
            raise ValueError(
                "a supplied matching engine cannot use the PyMatching-derived CUDA core"
            )
        self.device = device
        self.config = config or DecoderConfig()
        self.matching_engine = matching_engine
        self.control = _Control(context, self.config, path_bucket, matching_engine=matching_engine)
        self.num_detectors = context.num_detectors
        self._path_bucket = path_bucket
        self.search = (
            SectorSearch(
                context, self.config, self.control.conditioned, device=device,
                matching_engine=matching_engine,
            )
            if self.control.conditioned is not None
            else None
        )
        if device != "cpu" and self.search is None:
            raise ValueError(
                f"CUDA requires a conditional model: {self.control.unavailable_reason}"
            )
        self._correlated_cuda = None
        if cuda_correlated:
            from .correlated_cuda import CUDACorrelated
            from .correlated_graph import correlated_graph

            if self.search is None:
                raise ValueError("CUDA correlated matching requires a conditional model")
            graph = correlated_graph(context.dem, self.control.path.matching)
            self._correlated_cuda = CUDACorrelated(
                graph, max_batch=min(self.config.chunk_shots, 256),
            )
        self._solvers = local()
        self._solve_workers = solve_workers
        self._pool = (
            ThreadPoolExecutor(max_workers=solve_workers)
            if solve_workers > 1 and self.search is not None
            else None
        )

    def _routes(self, bits, diagnostics):
        matches = (
            self._correlated_cuda.edges(bits) if self._correlated_cuda is not None
            else [None] * len(bits)
        )
        for row, pairs in zip(bits, matches, strict=True):
            if pairs is None:
                pairs = self.control.path.matching.decode_to_edges_array(
                    row, enable_correlations=True
                )
            if self._path_bucket and not diagnostics:
                base, routed = self.control.path.route_edges(pairs, self.control.path_threshold)
                score = np.nan
            else:
                base, score = self.control.path.score_edges(pairs)
                routed = score < self.control.path_threshold
            yield row, pairs, base, score, routed

    def _initial_candidates(self, bits, diagnostics):
        routes = self._routes(bits, diagnostics)
        weights = None
        minimum = 4 if self.device == "cuda-batch" else 1
        if self.device != "cpu" and len(bits) >= max(2, minimum):
            routes = list(routes)
            candidates = [pairs for _, pairs, _, _, routed in routes if routed]
            if len(candidates) >= minimum:
                search = self.search
                # prepare borrows one parity buffer and records the baseline,
                # which _execute restores before solving each candidate.
                parities = [search.memory_fixed.prepare(pairs).copy() for pairs in candidates]
                weights = iter(search.batched_weights(np.stack(parities)))
        for row, pairs, base, score, routed in routes:
            precomputed = next(weights) if routed and weights is not None else None
            yield _Candidate(row, pairs, base, score, routed, precomputed)

    def _solve_candidate(self, candidate, worker=False):
        row, pairs, _, _, _, weights = candidate
        search = self.search
        if worker:
            # Each mutable matching workspace belongs to one worker.
            matchers = getattr(self._solvers, "matchers", None)
            if matchers is None:
                fixed = (
                    FixedMatching if self.matching_engine is None else self.matching_engine.fixed
                )
                matchers = fixed(search.memory), fixed(search.sector)
                self._solvers.matchers = matchers
            memory, sector = matchers
            # CUDA chunk weights, if present, were produced on the owner thread.
            make_weights = memory.weights
        else:
            memory, sector = search.memory_fixed, search.sector_fixed
            make_weights = search.memory_weights
        return search.initial_solve(
            memory, sector, row[search.memory.indices], pairs, weights, make_weights,
        )

    def _solve_chunk(self, bits, diagnostics):
        candidates = self._initial_candidates(bits, diagnostics)
        parallel_results = None
        if self._pool is not None and len(bits) >= 4:
            candidates = list(candidates)
            routed = [candidate for candidate in candidates if candidate.routed]
            if len(routed) >= 4:
                size = (len(routed) + self._solve_workers - 1) // self._solve_workers
                groups = (routed[i : i + size] for i in range(0, len(routed), size))
                parallel_results = chain.from_iterable(
                    self._pool.map(self._solve_on_worker, groups)
                )
        for candidate in candidates:
            solution = None
            if candidate.routed:
                solution = (
                    next(parallel_results) if parallel_results is not None
                    else self._solve_candidate(candidate)
                )
            yield candidate, solution

    def _solve_on_worker(self, candidates):
        return [self._solve_candidate(candidate, worker=True) for candidate in candidates]

    def _execute(self, events, diagnostics):
        validate_input(events, self.num_detectors)
        if self.search is None:
            return self.control.execute_fallback(events, diagnostics)
        output = np.empty((len(events), 1), np.uint8)
        if diagnostics:
            details = {
                key: np.zeros(len(events), dtype=bool)
                for key in ("routed", "changed", "competition_checked")
            }
            for key in ("path_score", "gain", "competition_margin", "conditioned_gap"):
                details[key] = np.full(len(events), np.nan)
            details["competition_states"] = np.zeros(len(events), np.int16)
            for key in ("baseline_prediction", "legacy_prediction", "candidate_prediction"):
                details[key] = np.empty_like(output)
        search = self.search
        for start in range(0, len(events), self.control.config.chunk_shots):
            bits = np.unpackbits(
                events[start : start + self.control.config.chunk_shots],
                axis=1,
                count=self.num_detectors,
                bitorder="little",
            )
            for offset, (candidate, solution) in enumerate(self._solve_chunk(bits, diagnostics)):
                row, pairs, base, score, routed, _ = candidate
                i = start + offset
                label = legacy = proposed = base
                gain = gap = margin = np.nan
                checked, states = False, 0
                if routed:
                    proposed, gain, gap = solution
                    if gain >= self.control.config.min_gain:
                        label = proposed
                    legacy = label
                    if abs(gap) <= self.config.ambiguous_gap or legacy != base:
                        result = search.evaluate(row, pairs)
                        checked, margin, states = True, result["margin"], result["states"]
                        if np.isfinite(margin) and abs(margin) >= self.config.min_margin:
                            label = result["prediction"]
                output[i, 0] = label
                if diagnostics:
                    details["routed"][i], details["changed"][i] = routed, legacy != base
                    details["path_score"][i], details["gain"][i] = score, gain
                    details["conditioned_gap"][i] = gap
                    details["competition_checked"][i] = checked
                    details["competition_margin"][i], details["competition_states"][i] = (
                        margin,
                        states,
                    )
                    details["baseline_prediction"][i, 0] = base
                    details["legacy_prediction"][i, 0] = legacy
                    details["candidate_prediction"][i, 0] = proposed
        return (output, details) if diagnostics else output

    def decode_batch(self, events):
        return self._execute(events, False)

    def decode_with_diagnostics(self, events):
        return self._execute(events, True)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
        if self._correlated_cuda is not None:
            self._correlated_cuda.close()


class _Control:
    def __init__(self, context, config, path_bucket=False, *, matching_engine=None):
        self.config = config
        self.path_threshold = config.path_factor * context.experiment.distance
        self.conditioned = self.path = self.fallback = None
        self.unavailable_reason = None
        if (
            context.num_detectors > config.max_detectors
            or context.dem.num_errors > config.max_mechanisms
            or context.dem.num_errors == 0
        ):
            self.unavailable_reason = "conditional model resource cap or empty fault model"
            self.fallback = (
                CorrelatedMatching(context) if matching_engine is None
                else matching_engine.correlated(context)
            )
        else:
            if matching_engine is None:
                self.path = PathProxy(context, bucket=True) if path_bucket else PathProxy(context)
                self.conditioned = ConditionedMatching(context)
            else:
                self.path = PathProxy(
                    context, bucket=path_bucket, matching=matching_engine.correlated(context),
                )
                self.conditioned = matching_engine.conditioned(context)

    def execute_fallback(self, events, diagnostics):
        output = self.fallback.decode_batch(events)
        if not diagnostics:
            return output
        details = {
            key: np.zeros(len(events), dtype=bool)
            for key in ("routed", "changed", "competition_checked")
        }
        for key in ("path_score", "gain", "competition_margin", "conditioned_gap"):
            details[key] = np.full(len(events), np.nan)
        details["competition_states"] = np.zeros(len(events), dtype=np.int16)
        for key in ("baseline_prediction", "legacy_prediction", "candidate_prediction"):
            details[key] = output.copy()
        return output, details


def build_decoder(context):
    if context.num_detectors <= 1024:
        return CascadeTNDecoder(context)
    return TensorDecoder(context)

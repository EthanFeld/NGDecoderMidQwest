#!/usr/bin/env python3
"""Executable reference for local FPGA matching and the CPU handoff."""

from dataclasses import dataclass
import math

import numpy as np

from exact import logical_map
from rescue import ResidualCheck
from stencils import geometry, maps


@dataclass
class Result:
    residual: np.ndarray
    logical: np.ndarray
    clocks: np.ndarray
    rescued: np.ndarray | None = None

    @property
    def accepted(self):
        return ~self.residual.any(axis=1)


class Predecoder:
    def __init__(self, image, bridges=True):
        compact = image.get('growth_compact', False)
        self.plan = plan = geometry(image, with_bridges=bridges and not compact)
        self.lookup = logical_map(image) if plan.lookup else None
        self.patch = ResidualCheck(image) if image.get('frontend') in ('rescue', 'cluster') and not compact else None
        self.rescue_gap3_cuts = image.get('rescue_gap3_cuts')
        self.cut_guard = image.get('growth_cut_guard', False)
        self.cut_limit = image.get('growth_cut_limit', self.patch.plan.raw_limit) if self.cut_guard else None
        self.packed_bits = 1 << np.arange(image['checks']) if plan.lookup else None
        self.stages = []
        for shape in plan.shapes:
            push, pull, priority = maps(shape, plan.width, image['coordinates'])
            origin = np.array([pull.get(i, plan.width) for i in range(plan.width)])
            destination = np.array([push.get(i, plan.width) for i in range(plan.width)])
            allowed = np.zeros((plan.rows, plan.width), dtype=np.bool_)
            for t, i, _ in shape.links:
                allowed[t, i] = True
            self.stages.append((shape.skip, allowed, origin, destination,
                                np.array([(priority >> i) & 1 for i in range(plan.width)], dtype=np.bool_)))
        self.boundary = self.bits(plan.boundary)
        self.logical = self.bits(plan.logical)

    def bits(self, words):
        return np.array([[(word >> i) & 1 for i in range(self.plan.width)] for word in words], dtype=np.bool_)

    def pass_once(self, syndrome, swap=False):
        shape = (syndrome.shape[0], self.plan.rows, self.plan.width+1)
        r = np.zeros(shape, dtype=np.bool_)
        r[:, :, :-1] = syndrome.reshape(shape[0], self.plan.rows, self.plan.width)
        order = list(range(len(self.stages)))
        if swap:
            order[0], order[1] = order[1], order[0]
        for i in order:
            skip, allowed, origin, destination, priority = self.stages[i]
            for t in range(self.plan.rows-skip):
                a, b = r[:, t, :], r[:, t+skip, :]
                candidate = a[:, :-1] & b[:, origin] & allowed[t]
                mask = np.zeros_like(a)
                mask[:, :-1] = candidate
                if skip == 0:
                    candidate &= priority | ~(mask[:, origin] | mask[:, destination])
                    mask[:, :-1] = candidate
                a[:, :-1] ^= candidate
                b[:, :-1] ^= mask[:, destination]
        r = r[:, :, :-1]
        logical = np.logical_xor.reduce((r & self.logical).reshape(shape[0], -1), axis=1)
        interior = r & ~self.boundary
        return r.reshape(syndrome.shape), interior.reshape(syndrome.shape), logical

    def weak_answers(self, syndrome, result):
        weak = np.zeros(len(syndrome), dtype=bool)
        if self.cut_guard:
            accepted = result.accepted
            cuts = self.patch.cut_count(syndrome[accepted], result.logical[accepted])
            weak[accepted] = cuts > self.cut_limit
        return weak

    def decode(self, syndrome, tilt=True, apply_cut_guard=True):
        clocks = np.full(syndrome.shape[0], self.plan.pass_clocks, dtype=np.int32)
        if self.lookup is not None:
            return Result(np.zeros_like(syndrome), self.lookup[syndrome @ self.packed_bits], clocks)

        residual, interior, logical = self.pass_once(syndrome)
        missed = interior.any(axis=1)
        rescued = np.zeros(len(missed), dtype=np.bool_)
        if tilt and missed.any():
            retry = np.flatnonzero(missed)
            alternate, alt_interior, label = self.pass_once(syndrome[retry], swap=True)
            shape = (-1, self.plan.rows, self.plan.width)
            left = interior[retry].reshape(shape).any(axis=2).sum(axis=1)
            other = alt_interior.reshape(shape).any(axis=2).sum(axis=1)
            if self.patch is not None:
                # Keep the original two-pass answers. Rescue only remaining
                # defers whose two provisional boundary parities agree.
                candidates = np.flatnonzero((other != 0) & (logical[retry] == label))
                start = retry[candidates]
                guess = label[candidates]
                if start.size:
                    # The cut check must agree on both branches even when
                    # a separating tile partition is required for only one.
                    first_cuts = self.patch.cut_count(residual[start], guess)
                    second_cuts = self.patch.cut_count(alternate[candidates], guess)
                    limit = self.patch.plan.residual_limit
                    allowed = (first_cuts <= limit) & (second_cuts <= limit)
                    if self.patch.plan.raw_limit is not None:
                        allowed &= self.patch.cut_count(syndrome[start], guess) <= self.patch.plan.raw_limit
                    first = self.patch.fit(interior[start[allowed]])
                    second = self.patch.fit(alt_interior[candidates[allowed]])
                    rescued[start[allowed]] = (first & second) if self.patch.plan.both else (first | second)
                    if self.rescue_gap3_cuts is not None:
                        raw_limit, residual_limit = self.rescue_gap3_cuts
                        extra = ~rescued[start] & (first_cuts <= residual_limit) & (second_cuts <= residual_limit)
                        extra_indexes = np.flatnonzero(extra)
                        extra[extra_indexes] &= (
                            self.patch.cut_count(syndrome[start[extra_indexes]], guess[extra_indexes]) <= raw_limit
                        )
                        first = self.patch.fit(interior[start[extra]], gap=3)
                        second = self.patch.fit(alt_interior[candidates[extra]], gap=3)
                        rescued[start[extra]] = first & second
            prefer = other < left
            indexes = retry[prefer]
            residual[indexes] = alternate[prefer]
            logical[indexes] = label[prefer]
            clocks[retry] *= 2
            missed[indexes] = other[prefer] != 0
            missed[rescued] = False
        residual[~missed] = False
        logical[missed] = False
        result = Result(residual, logical, clocks, rescued)
        if tilt and apply_cut_guard and self.cut_guard:
            weak = self.weak_answers(syndrome, result)
            residual[weak] = syndrome[weak]
            logical[weak] = False
        return result


def sample(image, count, rng):
    syndrome = np.zeros((count, image['checks']), dtype=np.bool_)
    actual = np.zeros(count, dtype=np.bool_)
    for endpoints, logical, p in image['faults']:
        selected = rng.choice(count, size=rng.binomial(count, p), replace=False)
        for d in endpoints:
            syndrome[selected, d] ^= True
        if logical:
            actual[selected] ^= True
    return syndrome, actual


def matching(image):
    from pymatching import Matching
    graph = Matching()
    for endpoints, logical, p in image['faults']:
        faults = {0} if logical else set()
        weight = math.log((1-p)/p)
        if len(endpoints) == 1:
            graph.add_boundary_edge(endpoints[0], fault_ids=faults, weight=weight, error_probability=p)
        else:
            graph.add_edge(*endpoints, fault_ids=faults, weight=weight, error_probability=p)
    return graph

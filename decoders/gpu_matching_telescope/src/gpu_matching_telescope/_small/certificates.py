"""Compile proven logical labels for likely syndromes using projected TN bounds.

Small fault sets give lower bounds on the full joint class mass. Summing over
unobserved detector checks gives upper bounds. A label is cached only when the
lower bound beats the competing upper bound. No sampled answers are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..model import bit_mask


@dataclass
class CertificateCache:
    keys: np.ndarray
    labels: np.ndarray
    margins: np.ndarray
    candidate_syndromes: int
    projection_bits: int

    @property
    def nbytes(self):
        return self.keys.nbytes + self.labels.nbytes + self.margins.nbytes


def projection_checks(model, detectors, count, reverse=False):
    masks = np.array([bit_mask(ds) for ds in model.detectors], dtype=np.uint64)
    labels = np.array([bool(ls) for ls in model.observables], dtype=bool)
    left = masks[~labels]
    right = masks[labels]
    # A hitting set favors checks distinguishing opposite-label 0/1-fault paths.
    differences = np.concatenate((right, (left[:, None] ^ right[None, :]).reshape(-1)))
    differences = differences[differences != 0]
    priorities = np.zeros(detectors)
    for p, ds in zip(model.probabilities, model.detectors, strict=True):
        priorities[list(ds)] += p
    selected = []
    remaining = list(range(detectors))
    if reverse:
        remaining.reverse()
    while len(selected) < count:
        best = max(
            remaining,
            key=lambda d: (
                int(np.count_nonzero(differences & (np.uint64(1) << np.uint64(d)))),
                priorities[d],
            ),
        )
        selected.append(best)
        remaining.remove(best)
        differences = differences[(differences & (np.uint64(1) << np.uint64(best))) == 0]
    return tuple(selected)


def projected_joint(model, checks):
    width = len(checks)
    position = {d: i for i, d in enumerate(checks)}
    addresses = np.arange(1 << (width + 1), dtype=np.int64)
    log_mass = np.full(len(addresses), -np.inf)
    log_mass[0] = 0
    # Merge equal projected actions before contracting, including identity terms.
    actions = {}
    for p, ds, ls in zip(model.probabilities, model.detectors, model.observables, strict=True):
        mask = sum(1 << position[d] for d in ds if d in position) | (int(bool(ls)) << width)
        if mask:
            actions[mask] = actions.get(mask, 0.0) + (-np.inf if p == 0.5 else np.log1p(-2 * p))
    for mask, log_bias in actions.items():
        p = -np.expm1(log_bias) / 2
        log_mass = np.logaddexp(log_mass + np.log1p(-p), log_mass[addresses ^ mask] + np.log(p))
    return log_mass.reshape(2, 1 << width)


def compile_certificates(model, detectors, bits=16, order=2, projections=2):
    if not 1 <= bits <= 16 or order not in (1, 2) or not 1 <= projections <= 4:
        raise ValueError("certificate bounds require 1..16 bits, fault order 1/2, 1..4 projections")
    empty = CertificateCache(
        np.zeros(0, dtype=np.uint64),
        np.zeros(0, dtype=np.uint8),
        np.zeros(0),
        0,
        min(bits, detectors),
    )
    # Explicitly bound this optional compile-time acceleration; contraction itself
    # retains the larger frontier model limits and needs no such cache.
    n = len(model.probabilities)
    if detectors > 63 or n > 512:
        return empty
    masks = [bit_mask(ds) for ds in model.detectors]
    labels = [int(bool(ls)) for ls in model.observables]
    log_zero = float(np.log1p(-model.probabilities).sum())
    odds = np.log(model.probabilities) - np.log1p(-model.probabilities)
    lower = {0: [log_zero, -np.inf]}

    def add(syndrome, logical, weight):
        row = lower.setdefault(syndrome, [-np.inf, -np.inf])
        row[logical] = np.logaddexp(row[logical], weight)

    for i in range(n):
        add(masks[i], labels[i], log_zero + odds[i])
        if order == 2:
            for j in range(i):
                add(masks[i] ^ masks[j], labels[i] ^ labels[j], log_zero + odds[i] + odds[j])
    keys = np.array(sorted(lower), dtype=np.uint64)
    lows = np.array([lower[int(key)] for key in keys])
    uppers = np.zeros_like(lows)
    width = min(bits, detectors)
    seen = set()
    for projection in range(projections):
        checks = projection_checks(model, detectors, width, reverse=bool(projection % 2))
        identity = frozenset(checks)
        if identity in seen:
            continue
        seen.add(identity)
        marginal = projected_joint(model, checks)
        indices = np.zeros(len(keys), dtype=np.int64)
        for i, d in enumerate(checks):
            indices |= ((keys >> np.uint64(d)) & 1).astype(np.int64) << i
        uppers = np.minimum(uppers, marginal[:, indices].T)
    guard = 1e-9 * (n + 1)
    first = lows[:, 0] - guard > uppers[:, 1] + guard
    second = lows[:, 1] - guard > uppers[:, 0] + guard
    valid = first | second
    prediction = second[valid].astype(np.uint8)
    margin = np.zeros(np.count_nonzero(valid))
    for i, (row, hi, label) in enumerate(zip(lows[valid], uppers[valid], prediction, strict=True)):
        delta = float(row[label] - hi[1 - label] - 2 * guard)
        margin[i] = math.tanh(delta / 2)
    return CertificateCache(keys[valid], prediction, margin, len(keys), width)

#!/usr/bin/env python3
"""Cross-check fault probabilities with a separate stabilizer circuit analyzer."""

import math
from pathlib import Path
import sys

import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surface_model import generate, layout


def circuit(distance, basis, rounds, p):
    plan = layout(distance)
    c = stim.Circuit()
    qdata = list(range(len(plan.data)))
    ndata = len(qdata)
    nrec = 0

    def prepare(b, qubits):
        c.append('RX' if b == 'x' else 'R', qubits)
        c.append('Z_ERROR' if b == 'x' else 'X_ERROR', qubits, p)

    def measure(b, qubits):
        nonlocal nrec
        start = nrec
        c.append('MX' if b == 'x' else 'M', qubits, p)
        nrec += len(qubits)
        return list(range(start, nrec))

    def rec(indices):
        return [stim.target_rec(i - nrec) for i in indices]

    prepare(basis, qdata)
    previous = {}
    current = {}
    for t in range(rounds):
        c.append('DEPOLARIZE1', qdata, p)
        for b in ('x', 'z'):
            prepare(b, [ndata + i for i in plan.memory(b)])
        for tick in plan.ticks:
            qubits = [q for pair in tick for q in pair]
            c.append('CX', qubits)
            c.append('DEPOLARIZE2', qubits, p)
        for b in ('x', 'z'):
            members = plan.memory(b)
            addresses = measure(b, [ndata+i for i in members])
            if b == basis:
                current = dict(zip(members, addresses, strict=True))
        for i in plan.memory(basis):
            outcomes = [current[i]] + ([previous[i]] if t else [])
            _, x, y = plan.checks[i]
            c.append('DETECTOR', rec(outcomes), [x, y, t])
        previous = current
    final = measure(basis, qdata)
    for i in plan.memory(basis):
        _, x, y = plan.checks[i]
        c.append('DETECTOR', rec([current[i], *[final[q] for q in plan.support[i]]]), [x, y, rounds])
    c.append('OBSERVABLE_INCLUDE', rec([final[q] for q in plan.observable(basis)]), 0)
    return c


def verify(distance, basis):
    rounds, p = distance, .001
    image = generate(distance, basis, rounds, p)
    c = circuit(distance, basis, rounds, p)
    expected = {(tuple(edges), label): rate for edges, label, rate in image['faults']}
    actual = {}
    assert len(c.shortest_graphlike_error()) == distance
    for item in c.detector_error_model().flattened():
        if item.type != 'error':
            continue
        endpoints = [t.val for t in item.targets_copy() if t.is_relative_detector_id()]
        label = int(any(t.is_logical_observable_id() for t in item.targets_copy()))
        actual[tuple(endpoints), label] = item.args_copy()[0]
    assert expected.keys() == actual.keys(), (distance, basis, len(expected), len(actual),
                                               list(expected.keys() - actual.keys())[:8],
                                               list(actual.keys() - expected.keys())[:8])
    for key, rate in expected.items():
        assert math.isclose(rate, actual[key], rel_tol=1e-10, abs_tol=1e-13), (key, rate, actual[key])
    print(f'd{distance} {basis.upper()} | Stim matched all {len(expected)} fault probabilities and distance')


if __name__ == '__main__':
    for d in (3, 7, 15):
        for b in ('x', 'z'):
            verify(d, b)

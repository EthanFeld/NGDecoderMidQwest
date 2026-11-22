#!/usr/bin/env python3
"""Construct rotated CSS circuits and their projected, graphlike fault models."""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import permutations, product
import json
from math import sqrt
from pathlib import Path
import zlib

CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class Layout:
    distance: int
    data: tuple
    checks: tuple
    support: tuple
    ticks: tuple

    def memory(self, basis):
        return tuple(i for i, (kind, _, _) in enumerate(self.checks) if kind == basis)

    def observable(self, basis):
        along = 1 if basis == 'z' else 0
        return tuple(i for i, xy in enumerate(self.data) if xy[along] == 0)


def face(distance, x, y):
    return tuple((x+dx, y+dy) for dx, dy in CORNERS
                 if 0 <= x+dx < distance and 0 <= y+dy < distance)


def schedule(checks, support, indices):
    orders = list(permutations(CORNERS))
    x_orders = [o for o in orders if o[-1][1] == o[-2][1]]
    z_orders = [o for o in orders if o[-1][0] == o[-2][0]]
    adjacent = [(i, j, set(support[i]) & set(support[j]))
                for i, (a, _, _) in enumerate(checks) for j in range(i)
                if a != checks[j][0] and set(support[i]) & set(support[j])]
    for x_order, z_order in product(x_orders, z_orders):
        orders_by_type = {'x': x_order, 'z': z_order}
        ticks = [[] for _ in CORNERS]
        slot_by_check = []
        hits = [set() for _ in CORNERS]
        for i, (basis, x, y) in enumerate(checks):
            slots = {}
            for t, (dx, dy) in enumerate(orders_by_type[basis]):
                q = indices.get((x+dx, y+dy))
                if q is None:
                    continue
                slots[q] = t
                hits[t].add(q)
                anc = len(indices)+i
                ticks[t].append((anc, q) if basis == 'x' else (q, anc))
            slot_by_check.append(slots)
        if any(len(active) != len(tick) for active, tick in zip(hits, ticks, strict=True)):
            continue
        # Overlap at two data sites must put the X ancilla on the same side
        # of the Z ancilla in the interaction order at both sites.
        if any(sum(slot_by_check[i][q] < slot_by_check[j][q] for q in both) == 1
               for i, j, both in adjacent):
            continue
        return tuple(tuple(tick) for tick in ticks)
    raise RuntimeError('no collision-free hook-safe schedule')


def layout(distance):
    data = tuple((x, y) for y in range(distance) for x in range(distance))
    indices = {q: i for i, q in enumerate(data)}
    checks = []
    for y in range(-1, distance):
        for x in range(-1, distance):
            kind = 'x' if (x+y) % 2 == 0 else 'z'
            bulk = 0 <= x < distance-1 and 0 <= y < distance-1
            x_half = kind == 'x' and 0 <= x < distance-1 and y in (-1, distance-1)
            z_half = kind == 'z' and 0 <= y < distance-1 and x in (-1, distance-1)
            if bulk or x_half or z_half:
                checks.append((kind, x, y))
    support = tuple(tuple(indices[q] for q in face(distance, x, y)) for _, x, y in checks)
    plan = Layout(distance, data, tuple(checks), support, schedule(checks, support, indices))
    if len(checks) != distance*distance-1:
        raise RuntimeError('wrong number of stabilizers')
    for basis in ('x', 'z'):
        other = 'z' if basis == 'x' else 'x'
        same, conjugate = plan.memory(basis), plan.memory(other)
        if len(same) != (distance*distance-1)//2:
            raise RuntimeError('unbalanced CSS check types')
        if any(len(set(support[a]) & set(support[b])) % 2 for a in same for b in conjugate):
            raise RuntimeError('CSS stabilizers do not commute')
        if any(len(set(plan.observable(basis)) & set(support[b])) % 2 for b in conjugate):
            raise RuntimeError('logical string does not commute with stabilizers')
    return plan


def projected_model(plan, basis, rounds, p):
    memory = plan.memory(basis)
    ndata, width = len(plan.data), len(memory)
    detectors = width*(rounds+1)
    logical = 1 << detectors
    check_lookup = {check: i for i, check in enumerate(memory)}
    probability = {}

    def add(response, rate):
        if response:
            old = probability.get(response, 0.0)
            probability[response] = old + rate - 2*old*rate

    response = [0] * (ndata + len(plan.checks))
    for q in plan.observable(basis):
        response[q] ^= logical
    for i, c in enumerate(memory):
        for q in plan.support[c]:
            response[q] ^= 1 << (rounds*width+i)
    for q in range(ndata):
        add(response[q], p)  # Final data readout.

    # One component of a two-qubit Pauli depolarizing fault has each of
    # its three nonzero outcomes with probability 4p/15. Three independent
    # XOR factors of rate a give exactly the same distribution when
    # a(1-a) = 4p/15.
    two_qubit = (1 - sqrt(1 - 16*p/15))/2
    for t in reversed(range(rounds)):
        for a in range(len(plan.checks)):
            i = check_lookup.get(a)
            toggles = 0 if i is None else (1 << (t*width+i)) | (1 << ((t+1)*width+i))
            response[ndata+a] = toggles
            add(toggles, p)
        for tick in reversed(plan.ticks):
            for c, q in tick:
                left, right = response[c], response[q]
                for outcome in (left, right, left ^ right):
                    add(outcome, two_qubit)
                if basis == 'z':
                    response[c] ^= right
                else:
                    response[q] ^= left
        for a in range(len(plan.checks)):
            if a in check_lookup:
                add(response[ndata+a], p)
            response[ndata+a] = 0
        for q in range(ndata):
            add(response[q], 2*p/3)
    for q in range(ndata):
        add(response[q], p)  # Data initialization in the memory basis.

    faults = []
    for word, rate in sorted(probability.items()):
        bits = word & (logical-1)
        ids = []
        while bits:
            low = bits & -bits
            ids.append(low.bit_length()-1)
            bits ^= low
        if len(ids) not in (1, 2):
            raise RuntimeError(f'projected fault is not graphlike ({ids})')
        if len(ids) == 2 and ids[1]//width - ids[0]//width > 1:
            raise RuntimeError('one fault connects more than two consecutive time slices')
        faults.append([ids, int(bool(word & logical)), rate])

    return {
        'distance': plan.distance,
        'basis': basis,
        'rounds': rounds,
        'width': width,
        'checks': detectors,
        'coordinates': [[plan.checks[i][1], plan.checks[i][2]] for i in memory],
        'p': p,
        'faults': faults,
    }


def fault_distance(image):
    # Interior logical labels are a binary vertex potential for these
    # planar memories. The only nontrivial graph cycles then use a boundary.
    n = image['checks']
    adjacency = [[] for _ in range(n+1)]
    for endpoints, label, _ in image['faults']:
        a = endpoints[0]
        b = endpoints[1] if len(endpoints) == 2 else n
        adjacency[a].append((b, label))
        adjacency[b].append((a, label))
    gauge = {}
    for root in range(n):
        if root in gauge:
            continue
        gauge[root] = 0
        stack = [root]
        while stack:
            a = stack.pop()
            for b, value in adjacency[a]:
                if b == n:
                    continue
                wanted = gauge[a] ^ value
                if b not in gauge:
                    gauge[b] = wanted
                    stack.append(b)
                elif gauge[b] != wanted:
                    raise RuntimeError('nontrivial entirely interior graph cycle')
    todo = deque([(n, 0, 0)])
    seen = {(n, 0)}
    while todo:
        a, parity, length = todo.popleft()
        for b, label in adjacency[a]:
            state = (b, parity ^ label)
            if state == (n, 1):
                return length+1
            if state not in seen:
                seen.add(state)
                todo.append((*state, length+1))
    raise RuntimeError('no undetectable logical path')


def generate(distance, basis, rounds, p):
    if distance not in (3, 7, 15) or basis not in ('x', 'z') or not 1 <= rounds <= distance or not 0 < p < .1:
        raise ValueError('choose distance 3, 7 or 15, X or Z, 1 to distance rounds, and 0 < p < .1')
    image = projected_model(layout(distance), basis, rounds, p)
    image['fault_distance'] = fault_distance(image)
    if image['fault_distance'] != distance:
        raise RuntimeError('spacelike circuit fault distance differs from the code distance')
    return image


def seal(image):
    return zlib.crc32(json.dumps(image, sort_keys=True, separators=(',', ':')).encode())


def growth_precision(text):
    value = float(text)
    if not 2 <= value <= 4:
        raise argparse.ArgumentTypeError('growth resolution must be between 2 and 4')
    return int(value) if value.is_integer() else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance', type=int, choices=(3, 7, 15), default=3)
    parser.add_argument('--basis', choices=('x', 'z'), default='z')
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--p', type=float, default=.001)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frontend', choices=('auto', 'map', 'tilt', 'rescue', 'cluster'), default='auto')
    parser.add_argument('--rescue-gap3-cuts', nargs=2, type=int, metavar=('RAW', 'RESIDUAL'),
                        help='allow three-round residual rescue under these crossing limits')
    parser.add_argument('--growth-p', type=float, default=.007)
    parser.add_argument('--growth-resolution', type=growth_precision, default=2)
    parser.add_argument('--growth-engine', choices=('serial', 'dual-port'), default='serial')
    parser.add_argument('--growth-cut-guard', action='store_true')
    parser.add_argument('--growth-cut-limit', type=int, choices=range(8))
    parser.add_argument('--growth-boundary-bias', type=float, default=0)
    parser.add_argument('--growth-skip-empty', action='store_true')
    parser.add_argument('--growth-compact', action='store_true')
    parser.add_argument('--growth-cycle-cap', type=int,
                        help='override the cluster watchdog in fabric cycles')
    parser.add_argument('--clock-mhz', type=int, default=135)
    args = parser.parse_args()
    rounds = args.distance if args.rounds is None else args.rounds
    image = generate(args.distance, args.basis, rounds, args.p)
    if args.frontend == 'map' and args.distance != 3:
        parser.error('the MAP table is supported at distance 3')
    if args.frontend in ('rescue', 'cluster') and args.distance == 3:
        parser.error('distance 3 uses the exact MAP table or the plain tilt')
    if args.growth_cut_guard and (args.frontend != 'cluster' or args.distance != 7):
        parser.error('growth-cut-guard requires the distance-7 cluster front end')
    if args.growth_cut_limit is not None and not args.growth_cut_guard:
        parser.error('growth-cut-limit requires growth-cut-guard')
    if not -1 <= args.growth_boundary_bias <= 1:
        parser.error('growth-boundary-bias must be between -1 and 1')
    if args.growth_compact and (args.frontend != 'cluster' or args.distance != 15):
        parser.error('growth-compact requires the distance-15 cluster front end')
    if args.growth_cycle_cap is not None and args.frontend != 'cluster':
        parser.error('growth-cycle-cap requires the cluster front end')
    if args.distance == 3 and args.frontend != 'tilt':
        image['frontend'] = 'map'
    if args.distance in (7, 15) and args.frontend in ('auto', 'rescue'):
        image['frontend'] = 'rescue'
    if args.rescue_gap3_cuts is not None:
        if args.distance != 15 or image.get('frontend') != 'rescue':
            parser.error('rescue-gap3-cuts requires the distance-15 rescue front end')
        if any(not 0 <= cut <= args.distance for cut in args.rescue_gap3_cuts):
            parser.error('rescue-gap3-cuts limits must be between 0 and the code distance')
        if not 0 < args.clock_mhz <= 189:
            parser.error('rescue clock must be between 1 and 189 MHz')
        image['rescue_gap3_cuts'] = args.rescue_gap3_cuts
        image['decoder_clock_mhz'] = args.clock_mhz
    if args.frontend == 'cluster':
        from growth import GROW_BUDGET, growth_cycles
        from stencils import geometry
        if args.growth_compact:
            image['growth_compact'] = True
        if args.growth_p not in GROW_BUDGET[args.distance]:
            parser.error('growth-p must be .002, .003, .005, or .007')
        if not 0 < args.clock_mhz <= 189:
            parser.error('growth clock must be between 1 and 189 MHz')
        cap = args.growth_cycle_cap
        if cap is None:
            cap = growth_cycles(image, args.growth_p, args.clock_mhz)
        elif not 1 <= cap <= 65534-geometry(image, with_bridges=not args.growth_compact).max_clocks:
            parser.error('growth-cycle-cap must be positive and fit in the 16-bit reply cycle count')
        if cap < 1:
            parser.error('the original deadline leaves no clustering cycles at this clock')
        image.update(frontend='cluster', growth_p=args.growth_p,
                     growth_resolution=args.growth_resolution,
                     growth_cycles=cap, decoder_clock_mhz=args.clock_mhz)
        if args.growth_engine != 'serial':
            image['growth_engine'] = args.growth_engine
        if args.growth_cut_guard:
            image['growth_cut_guard'] = True
        if args.growth_cut_limit is not None:
            image['growth_cut_limit'] = args.growth_cut_limit
        if args.growth_boundary_bias:
            image['growth_boundary_bias'] = args.growth_boundary_bias
        if args.growth_skip_empty:
            image['growth_skip_empty'] = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'graph.json').write_text(json.dumps(image, separators=(',', ':'))+'\n')
    print(f"d{args.distance} {args.basis.upper()} | {rounds} rounds | {image['checks']} detectors | "
          f"{len(image['faults'])} graph edges | graph fault distance {image['fault_distance']}")


if __name__ == '__main__':
    main()

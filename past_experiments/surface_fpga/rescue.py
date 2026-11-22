#!/usr/bin/env python3
"""Compile and check local zero-logical repairs of selected residuals."""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Patch:
    tiles: tuple
    strips: tuple
    excluded: tuple[int, ...]
    gap: int
    residual_limit: int
    raw_limit: int | None
    both: bool


def compile_patch(image):
    d, w, rows = image['distance'], image['width'], image['rounds']+1
    coords = image['coordinates']
    axis = int(image['basis'] == 'z')
    if d == 7:
        parallel, normal, gap, margin = 3, 3, 2, 1
        residual_limit, raw_limit, both = 2, 3, True
    elif d == 15:
        parallel, normal, gap, margin = 5, 5, 2, 1
        residual_limit, raw_limit, both = 6, None, False
    else:
        raise ValueError('local residual rescue supports distance 7 and 15')

    # Get the actual spatial edges available across the entire window.
    space, time, boundary = Counter(), Counter(), [set() for _ in range(rows)]
    for ends, label, _ in image['faults']:
        t, a = divmod(ends[0], w)
        if len(ends) == 1:
            if (coords[a][axis], label) not in ((0, 1), (d-2, 0)):
                raise ValueError('invalid boundary for the strip projection')
            boundary[t].add(a)
        else:
            u, b = divmod(ends[1], w)
            if label or abs(coords[a][axis]-coords[b][axis]) > 1:
                raise ValueError('strip projection requires local zero-logical edges')
            if u == t:
                space[a, b] += 1
            elif u == t+1 and a == b:
                time[a] += 1
    if not all(group == boundary[0] for group in boundary):
        raise ValueError('boundary sites must be stationary during the memory window')
    neighbors = [set() for _ in coords]
    for (a, b), uses in space.items():
        if uses == rows:
            neighbors[a].add(b)
            neighbors[b].add(a)

    depth = [w] * w
    pending = deque(sorted(boundary[0]))
    for a in pending:
        depth[a] = 0
    while pending:
        a = pending.popleft()
        for b in neighbors[a]:
            if depth[b] > depth[a]+1:
                depth[b] = depth[a]+1
                pending.append(b)
    excluded = tuple(a for a, value in enumerate(depth) if 0 < value < margin)
    strips = tuple(tuple(i for i, xy in enumerate(coords) if xy[axis] == j) for j in range(d-1))
    if not all(strips):
        raise ValueError('missing position in normal cut sequence')

    # Each tile is connected by in-slice physical edges. A tile has one
    # representative that can carry odd parity through time along measurement
    # edges. Compile only partitions with a witness for those vertical paths.
    views = []
    for u, v in product(range(parallel), range(normal)):
        if d == 15 and (u+v) % 2:
            continue
        parts = defaultdict(set)
        for a, xy in enumerate(coords):
            if depth[a] >= margin:
                parts[((xy[1-axis]+u)//parallel, (xy[axis]+v)//normal)].add(a)
        tiles = []
        for members in parts.values():
            while members:
                root = min(members)
                members.remove(root)
                work = [root]
                for a in work:
                    extra = members & neighbors[a]
                    members -= extra
                    work.extend(sorted(extra))
                if not any(time[a] == rows-1 for a in work):
                    raise ValueError('tile has no full vertical measurement path')
                tiles.append(tuple(sorted(work)))
        views.append(tuple(sorted(tiles)))
    return Patch(tuple(dict.fromkeys(views)), strips, excluded, gap, residual_limit, raw_limit, both)


class ResidualCheck:
    def __init__(self, image):
        self.plan = compile_patch(image)
        self.width = image['width']
        self.rows = image['rounds']+1
        self.distance = image['distance']

    def cut_count(self, vector, label):
        rows = vector.reshape(-1, self.rows, self.width)
        parity = np.stack([np.bitwise_xor.reduce(rows[:, :, cols], axis=(1, 2)) for cols in self.plan.strips],
                          axis=1)
        cross = np.bitwise_xor.accumulate(parity, axis=1, dtype=np.uint8).sum(axis=1)
        return np.where(label, self.distance-cross, cross)

    def fit(self, interior, gap=None):
        gap = self.plan.gap if gap is None else gap
        r = interior.reshape(-1, self.rows, self.width)
        remaining = (np.flatnonzero(~r[:, :, self.plan.excluded].any(axis=(1, 2)))
                     if self.plan.excluded else np.arange(len(r)))
        accepted = np.zeros(len(r), dtype=np.bool_)
        for partition in self.plan.tiles:
            if not remaining.size:
                break
            trial = r[remaining]
            parity = np.stack([np.bitwise_xor.reduce(trial[:, :, tile], axis=2) for tile in partition], axis=2)
            carried = np.bitwise_xor.accumulate(parity, axis=1, dtype=np.uint8).astype(np.bool_)
            g = gap
            is_long = carried[:, g:, :].copy()
            for delay in range(1, g+1):
                is_long &= carried[:, g-delay:self.rows-delay, :]
            valid = ~(carried[:, -1, :].any(axis=1) | is_long.any(axis=(1, 2)))
            accepted[remaining[valid]] = True
            remaining = remaining[~valid]
        return accepted


def emit_patch(image, folder):
    p = compile_patch(image)
    width = image['width']

    def mask(indices):
        return f"{width}'h{sum(1 << i for i in indices):x}"

    def decoder(name, values, bits):
        lines = [f'function automatic [{bits-1}:0] {name}(input integer select_id);', '    case (select_id)']
        lines += [f'        {i}: {name} = {word};' for i, word in enumerate(values)]
        lines += [f"        default: {name} = '0;", '    endcase', 'endfunction']
        return '\n'.join(lines)+'\n'

    flat = [mask(tile) for view in p.tiles for tile in view]
    starts = [0]
    for view in p.tiles:
        starts.append(starts[-1] + len(view))
    definitions = {
        'SC_PATCH_TILES': len(flat), 'SC_PATCH_VIEWS': len(p.tiles), 'SC_PATCH_GAP': p.gap,
        'SC_PATCH_CUTS': len(p.strips), 'SC_PATCH_RES_MAX': p.residual_limit,
        'SC_PATCH_RAW_MAX': p.raw_limit or 0, 'SC_PATCH_USE_RAW': int(p.raw_limit is not None),
        'SC_PATCH_BOTH': int(p.both), 'SC_PATCH_EXCLUDE': mask(p.excluded),
    }
    if 'rescue_gap3_cuts' in image:
        raw, residual = image['rescue_gap3_cuts']
        definitions.update(SC_PATCH_GAP3_RAW_MAX=raw, SC_PATCH_GAP3_RES_MAX=residual)
    structs = '\n'.join(f'`define {key} {val}' for key, val in definitions.items())+'\n'
    structs += decoder('tile_mask', flat, width)
    structs += decoder('strip_mask', [mask(s) for s in p.strips], width)
    structs += decoder('tile_start', starts, len(flat).bit_length())
    (folder/'sc_patch.vh').write_text(structs)
    return p

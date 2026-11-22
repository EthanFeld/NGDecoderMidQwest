#!/usr/bin/env python3
"""Map the surface-code graph to constant spatial permutations and time masks."""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
import argparse

from surface_model import seal


@dataclass(frozen=True)
class Shape:
    delta: tuple[int, int, int]
    links: tuple[tuple[int, int, int], ...]
    prior: float
    virtual: bool = False

    @property
    def skip(self):
        return self.delta[0]


@dataclass(frozen=True)
class Plan:
    width: int
    rows: int
    shapes: tuple[Shape, ...]
    boundary: tuple[int, ...]
    logical: tuple[int, ...]
    lookup: bool

    @property
    def pass_clocks(self):
        return 2 if self.lookup else self.rows + 1 + sum(1 + shape.skip for shape in self.shapes) + 3

    @property
    def max_clocks(self):
        return self.pass_clocks if self.lookup else 2*self.pass_clocks


def geometry(image, with_bridges=True):
    coords, width, rounds = image['coordinates'], image['width'], image['rounds']
    planes = rounds + 1
    by_shape = defaultdict(list)
    adjacent = defaultdict(list)
    boundary = [0]*planes
    logical = [0]*planes

    def displacement(a, b):
        ta, i = divmod(a, width)
        tb, j = divmod(b, width)
        x, y = coords[i]
        u, v = coords[j]
        return tb-ta, u-x, v-y

    for ends, label, p in image['faults']:
        a = ends[0]
        t, site = divmod(a, width)
        if len(ends) == 1:
            boundary[t] |= 1 << site
            logical[t] |= label << site
            continue
        b = ends[1]
        if label:
            raise ValueError('design requires logical labels to be supported on boundary edges')
        by_shape[displacement(a, b)].append((a, b, p))
        adjacent[a].append((b, p))
        adjacent[b].append((a, p))

    def Family(delta, edges, virtual=False):
        return Shape(delta, tuple((a//width, a%width, b%width) for a, b, _ in edges),
                     mean(p for _, _, p in edges), virtual)

    native = sorted((Family(delta, edges) for delta, edges in by_shape.items()),
                    key=lambda s: (-s.prior, s.delta))
    if len(native) < 2:
        raise ValueError('prior-tilt pair requires at least two physical shapes')

    # The masks use only two-edge paths assembled from the most likely
    # classes. There is no extrapolation across edges missing at boundaries.
    bridges = []
    if with_bridges:
        allowed_shapes = {shape.delta for shape in native[:3]}
        visible = {(a, b) for group in by_shape.values() for a, b, _ in group}
        eligible = defaultdict(dict)
        for v, options in adjacent.items():
            cheap = [(u, p) for u, p in options if displacement(min(u, v), max(u, v)) in allowed_shapes]
            for i, (a, pa) in enumerate(cheap):
                for b, pb in cheap[i+1:]:
                    a0, b0 = sorted((a, b))
                    if (a0, b0) in visible:
                        continue
                    delta = displacement(a0, b0)
                    dt, dx, dy = delta
                    if (dt == 2 and (dx or dy)) or dt > 2:
                        continue
                    # Separate a short measurement gap or a straight
                    # spatial bridge from mixed or non-collinear hooks.
                    if dt == 0 and (abs(dx) != 2 or abs(dy) != 2):
                        continue
                    if dt == 1 and (abs(dx) != 1 or abs(dy) != 1):
                        continue
                    edges = eligible[delta]
                    edges[a0, b0] = max(edges.get((a0, b0), 0), pa*pb)
        bridges = sorted((Family(delta, [(a, b, p) for (a, b), p in sorted(e.items())], True)
                          for delta, e in eligible.items()), key=lambda s: (-s.prior, s.delta))
    return Plan(width, planes, tuple(native + bridges), tuple(boundary), tuple(logical),
                image.get('frontend') == 'map')


def maps(shape, width, coords):
    push, pull = {}, {}
    for _, a, b in shape.links:
        if (a in pull and pull[a] != b) or (b in push and push[b] != a):
            raise ValueError('shape displacement is not injective')
        pull[a] = b
        push[b] = a
    degree = 0
    priority = 0
    if shape.skip == 0:
        _, dx, dy = shape.delta
        axis, degree = (0, abs(dx)) if dx else (1, abs(dy))
        priority = sum(1 << i for i in pull if (coords[i][axis]//degree) % 2 == 0)
    return push, pull, priority


def emit(image, folder):
    compact = image.get('growth_compact', False)
    plan = geometry(image, with_bridges=not compact)
    w, rows = plan.width, plan.rows
    id_bits = (rows).bit_length()
    masks, inputs, outputs, priorities = [], [], [], []
    for shape in plan.shapes:
        push, pull, priority = maps(shape, w, image['coordinates'])

        def shuffle(lookup):
            return '{' + ', '.join(f'word[{lookup[i]}]' if i in lookup else "1'b0"
                                   for i in reversed(range(w))) + '}'

        inputs.append(shuffle(pull))
        outputs.append(shuffle(push))
        priorities.append(f"{w}'h{priority:x}")
        per_time = [0]*(rows+1)
        for t, a, _ in shape.links:
            per_time[t+shape.skip] |= 1 << a
        masks.append(per_time)

    def func(name, values, ports):
        lines = [f'function automatic [{w-1}:0] {name}(input integer family{ports});',
                 '    case (family)']
        for i, expression in enumerate(values):
            lines.append(f'        {i}: {name} = {expression};')
        lines.extend([f"        default: {name} = '0;", '    endcase', 'endfunction'])
        return '\n'.join(lines)+'\n'

    def masks_func(name, values):
        # Group equal masks so the common interior compiles to one condition.
        output = []
        for per_time in values:
            cases = defaultdict(list)
            for t, bits in enumerate(per_time):
                cases[bits].append(t)
            terms = []
            for bits, times in cases.items():
                if not bits:
                    continue
                allowed = sum(1 << t for t in times)
                terms.append(f"(({rows+1}'h{allowed:x} >> t) & {rows+1}'d1 ? {w}'h{bits:x} : {w}'d0)")
            output.append(' | '.join(terms) or f"{w}'d0")
        return func(name, output, f', input logic [{id_bits-1}:0] t')

    header = {
        'SC_WIDTH': w,
        'SC_ROWS': rows,
        'SC_SHAPES': len(plan.shapes),
        'SC_TAG': f"32'h{seal(image):08x}",
        'SC_MAX_CLOCKS': plan.max_clocks,
        'SC_SKIPS': f"{2*len(plan.shapes)}'h{sum(s.skip << (2*i) for i, s in enumerate(plan.shapes)):x}",
    }
    if plan.lookup:
        from exact import emit_map
        map_file = folder / 'surface_map.hex'
        header['SC_LOOKUP'] = 1
        header['SC_MAP_FILE'] = json.dumps(map_file.as_posix())
        emit_map(image, map_file)
    elif image.get('frontend') in ('rescue', 'cluster') and not compact:
        from rescue import emit_patch
        header['SC_RESCUE'] = 1
        patch = emit_patch(image, folder)
        print(f'{len(patch.tiles)} tile partitions | residual spans at most {patch.gap} rounds')
    if image.get('frontend') == 'cluster':
        from growth import compile_growth
        header['SC_GROWTH'] = 1
        if image.get('growth_cut_guard'):
            header['SC_GROW_CUT_GUARD'] = 1
            if 'growth_cut_limit' in image:
                header['SC_GROW_CUT_LIMIT'] = image['growth_cut_limit']
        header['SC_MAX_CLOCKS'] += image['growth_cycles']+1
        compile_growth(image, folder)
    (folder/'sc_image.vh').write_text(''.join(f'`define {k} {v}\n' for k, v in header.items()))
    text = func('pull_back', inputs, f', input logic [{w-1}:0] word')
    text += func('push_forward', outputs, f', input logic [{w-1}:0] word')
    text += func('even_edges', priorities, '')
    text += masks_func('enabled_edges', masks)
    text += masks_func('boundary_bits', [plan.boundary])
    text += masks_func('logical_bits', [plan.logical])
    (folder/'sc_stencils.vh').write_text(text)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    image = json.loads((args.image/'graph.json').read_text())
    plan = emit(image, args.image)
    print(f'{len(plan.shapes)} edge shapes, {sum(s.virtual for s in plan.shapes)} two-fault shapes')
    print('prior order', ' '.join(str(s.delta) for s in plan.shapes))

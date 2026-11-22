#!/usr/bin/env python3
"""Compile a weighted physical-edge grower for the bounded FPGA fallback."""

from collections import defaultdict
from math import log

from stencils import geometry, maps
from surface_model import generate


# Budgets are additional clocks relative to the packed rescue images.
# Each uses floor(135 MHz * 10 us * measured old defer fraction).
GROW_BUDGET = {
    7: {.002: 62, .003: 137, .005: 347, .007: 585},
    15: {.002: 361, .003: 752, .005: 1264, .007: 1345},
}


def growth_cycles(image, p, clock_mhz):
    """Keep the original 135 MHz deadline plus the permitted extension."""
    base = geometry(image).max_clocks
    total = (base + GROW_BUDGET[image['distance']][p])*clock_mhz//135
    front = geometry(image, with_bridges=False).max_clocks if image.get('growth_compact') else base
    return total - front - 1  # One clock for the handoff.


def compile_growth(image, folder):
    if 'growth_p' in image and image['growth_p'] != image['p']:
        tuned = generate(image['distance'], image['basis'], image['rounds'], image['growth_p'])
        image = {**image, 'faults': tuned['faults'], 'p': tuned['p']}
    plan = geometry(image, with_bridges=False)
    w, rows = plan.width, plan.rows
    tbits, sbits = rows.bit_length(), (w-1).bit_length()
    quant = image['growth_resolution']
    dual_port = image.get('growth_engine') == 'dual-port'
    short_updates = dual_port and not image.get('growth_compact', False)
    shapes = plan.shapes[:4] if image.get('growth_compact') else plan.shapes
    family_of = {shape.delta: i for i, shape in enumerate(shapes)}
    families = len(shapes)+1
    depth = w*rows
    most = max(f[2] for f in image['faults'])
    min_cost = log((1-most)/most)
    boundary_bias = image.get('growth_boundary_bias', 0)
    limits = [2*max(2, int(log((1-p)/p)*quant/min_cost + .5 +
                            (boundary_bias if len(ends) == 1 else 0)))
              for ends, _, p in image['faults']]
    cap = max(limits)
    ages = cap.bit_length()
    initial_age = 1
    if image.get('growth_skip_empty'):
        # Each active endpoint grows by one per tick. A terminal edge has
        # one active end at most, so it takes its full cost to mature.
        first_hit = min(weight if len(ends) == 1 else weight//2
                        for (ends, _, _), weight in zip(image['faults'], limits, strict=True))
        initial_age = first_hit-1
    codes = [[0]*rows for _ in range(families)]
    boundary = [0]*rows
    for (ends, label, _), level in zip(image['faults'], limits, strict=True):
        t, a = divmod(ends[0], w)
        if len(ends) == 1:
            g = families-1
            boundary[t] |= label << a
        else:
            u, b = divmod(ends[1], w)
            x, y = image['coordinates'][a]
            xx, yy = image['coordinates'][b]
            g = family_of.get((u-t, xx-x, yy-y))
            if g is None:
                continue
            t = u
        codes[g][t] |= (level//2 - 1) << (ages*a)

    def declarations(name, size, fields, body):
        return (f'function automatic [{size-1}:0] {name}({fields});\n'+body+
                f"\nendfunction\n")

    def choose(name, width, fields, values):
        body = ['    case (key)']
        body += [f"        {key}: {name} = {width}'h{value:x};" for key, value in values.items()]
        body += [f"        default: {name} = '0;", '    endcase']
        return declarations(name, width, fields, '\n'.join(body))

    headers = {
        'SC_GROW_GROUPS': families, 'SC_GROW_AGES': ages, 'SC_GROW_MAXAGE': cap,
        'SC_GROW_ROWS': rows, 'SC_GROW_SBITS': sbits, 'SC_GROW_TBITS': tbits,
        'SC_GROW_CYCLES': image['growth_cycles'],
        'SC_GROW_INITIAL_AGE': initial_age,
        'SC_GROW_PREFETCH_ROWS': int(w <= 32 and (dual_port or image.get('decoder_clock_mhz', 0) >= 108)),
        'SC_GROW_DUAL_PORT': int(dual_port),
        'SC_GROW_SHORT_UPDATES': int(short_updates),
    }
    head = ''.join(f'`define {k} {v}\n' for k, v in headers.items())
    chunks = ['    case (key)']
    for g, times in enumerate(codes):
        unique = defaultdict(list)
        for t, bits in enumerate(times):
            unique[bits].append(t)
        masks = [f"(({rows+1}'h{sum(1<<t for t in time):x} >> t) & {rows+1}'d1 ? {ages*w}'h{bits:x} : {ages*w}'d0)"
                 for bits, time in unique.items() if bits]
        chunks.append(f'        {g}: edge_weight = '+(' | '.join(masks) or "'0")+';')
    chunks += ["        default: edge_weight = '0;", '    endcase']
    head += declarations('edge_weight', ages*w, f'input integer key, input logic [{tbits-1}:0] t', '\n'.join(chunks))
    head += choose('boundary_type', w, f'input logic [{tbits-1}:0] key', dict(enumerate(boundary)))
    head += choose('past_edge', 1, f'input logic [{(families-1).bit_length()-1}:0] key',
                   {i:s.skip for i,s in enumerate(shapes)})

    source = []
    for s in shapes:
        _, pull, _ = maps(s, w, image['coordinates'])
        source.append(pull)
    source.append({})

    chunks = ['    case (key)']
    for g, pull in enumerate(source):
        lookup = ', '.join(f'word[{ages*pull[i]}+:{ages}]' if i in pull else f"{ages}'d0"
                           for i in reversed(range(w)))
        chunks.append(f'        {g}: pull_age = {{{lookup}}};')
    chunks += ["        default: pull_age = '0;", '    endcase']
    head += declarations('pull_age', ages*w, f'input integer key, input logic [{ages*w-1}:0] word', '\n'.join(chunks))

    chunks = ['    case (key)']
    for g, pull in enumerate(source):
        lookup = ', '.join(f'word[{pull[i]}]' if i in pull else "1'b0"
                           for i in reversed(range(w)))
        chunks.append(f'        {g}: pull_activity = {{{lookup}}};')
    chunks += ["        default: pull_activity = '0;", '    endcase']
    head += declarations('pull_activity', w, f'input integer key, input logic [{w-1}:0] word', '\n'.join(chunks))

    partner_masks = {}
    for g, pull in enumerate(source):
        offsets = defaultdict(int)
        for i, j in pull.items():
            offsets[j-i] |= 1 << i
        partner_masks[g] = offsets
    group_bits = (families-1).bit_length()
    inputs = f'input logic [{group_bits-1}:0] key, input logic [{sbits-1}:0] site'
    if w > 32:
        # Select the increment first. This shares the address adder between
        # the edge families and avoids a separate adder for each translation.
        chunks = ['    case (key)']
        for g, offsets in partner_masks.items():
            terms = [f"(({w}'h{bits:x} >> site) & {w}'d1 ? {sbits}'d{delta % (1<<sbits)} : {sbits}'d0)"
                     for delta, bits in sorted(offsets.items()) if delta]
            chunks.append(f'        {g}: partner_offset = '+(' | '.join(terms) or "'0")+';')
        chunks += ["        default: partner_offset = '0;", '    endcase']
        head += declarations('partner_offset', sbits, inputs, '\n'.join(chunks))
        head += declarations('partner', sbits, inputs, '    partner = site + partner_offset(key, site);')
    else:
        chunks = ['    case (key)']
        for g, offsets in partner_masks.items():
            terms = [f"(({w}'h{bits:x} >> site) & {w}'d1 ? site + {sbits}'d{delta % (1<<sbits)} : {sbits}'d0)"
                     for delta, bits in sorted(offsets.items())]
            chunks.append(f'        {g}: partner = '+(' | '.join(terms) or "'0")+';')
        chunks += ["        default: partner = '0;", '    endcase']
        head += declarations('partner', sbits, inputs, '\n'.join(chunks))

    (folder/'sc_growth.vh').write_text(head)

    # Keep this t-major order as a software fixture for the FPGA engine.
    with (folder/'growth_graph.txt').open('w') as out:
        edges = []
        for g, per_time in enumerate(codes):
            past = g < len(shapes) and shapes[g].skip
            for t, word in enumerate(per_time):
                for i in range(w):
                    encoded = (word >> (i*ages)) & ((1 << ages)-1)
                    if encoded:
                        weight = 2*(encoded+1)
                        u = (t-past)*w+i
                        v = -1 if g == families-1 else t*w+source[g][i]
                        flip = (boundary[t] >> i) & 1 if v < 0 else 0
                        edges.append((t,g,i,u,v,flip,weight))
        print(depth, len(edges), rows, w, cap, int(dual_port), int(short_updates), initial_age, file=out)
        for _, _, _, a, b, flip, level in sorted(edges):
            print(a, b, flip, level, file=out)

#!/usr/bin/env python3
"""Make native and byte-level regression fixtures for a board image."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from host import COMMAND, REQUEST, REPLY, frame, pack
from growth_reference import GrowthModel
from reference import Predecoder, sample
from surface_model import generate as generate_graph, seal


def generate(directory):
    image = json.loads((directory/'graph.json').read_text())
    width, depth = image['width'], (image['width']+7)//8
    original = Predecoder(image)
    if original.lookup is not None:
        (directory/'map_expected.hex').write_text(''.join(f'{int(label)}\n' for label in original.lookup))
    rng = np.random.default_rng(5115)
    shots = [np.zeros(image['checks'], dtype=np.bool_), np.ones(image['checks'], dtype=np.bool_)]
    single = range(len(image['faults'])) if image['distance'] == 3 else rng.choice(len(image['faults']), 130, replace=False)
    for edge in single:
        defect = np.zeros(image['checks'], dtype=np.bool_)
        fault = image['faults'][edge]
        defect[fault[0]] = True
        out = original.decode(defect[None, :], apply_cut_guard=False)
        assert out.accepted[0] and out.logical[0] == bool(fault[1]), (edge, fault)
        shots.append(defect)
    shots.extend(sample(image, 320, rng)[0])
    if original.patch is not None:
        stress = generate_graph(image['distance'], image['basis'], image['rounds'], .003)
        shots.extend(sample(stress, 192, rng)[0])
    model = GrowthModel(directory/'growth_graph.txt') if image.get('frontend') == 'cluster' else None
    if original.cut_guard:
        stress = generate_graph(image['distance'], image['basis'], image['rounds'], image['growth_p'])
        shots.extend(sample(stress, 128, rng)[0])
    shots.extend(rng.random((128, image['checks'])) < .12)
    first_guard_fixture = len(shots)
    if original.cut_guard:
        # Exercise the guarded reply even when clustering times out and
        # the UART payload must come from the original syndrome bank.
        extra, _ = sample(stress, 2_048, np.random.default_rng(1536))
        initial = original.decode(extra, apply_cut_guard=False)
        weak = extra[original.weak_answers(extra, initial)]
        outcomes = model.decode(weak, image['growth_cycles'])
        for accepted in (True, False):
            picks = np.flatnonzero((outcomes[:, 1] == 0) == accepted)
            if picks.size:
                shots.append(weak[picks[0]])
    modes = np.array([(i != 2 and i % 5 != 4) for i in range(len(shots))])
    modes[first_guard_fixture:] = True
    case_array = np.array(shots)
    flat_result = original.decode(case_array, tilt=False)
    full_result = original.decode(case_array)
    weak_cases = (original.weak_answers(case_array, original.decode(case_array, apply_cut_guard=False))
                  if original.cut_guard else np.zeros(len(shots), dtype=bool))
    rescue_cases = modes & full_result.rescued if full_result.rescued is not None else np.zeros(len(shots), bool)
    growth_cases = np.zeros(len(shots), dtype=np.uint8)
    if model is not None:
        touched, outcomes = model.recover(full_result, case_array, image['growth_cycles'])
        growth_cases[touched] = 1 + outcomes[:, 1]
    tag = seal(image)
    replies = []
    with (directory/'core.txt').open('w') as out:
        print(len(shots), file=out)
        for i, syndrome in enumerate(shots):
            result = full_result if modes[i] else flat_result
            residual, clocks, logical = result.residual[i], int(result.clocks[i]), int(result.logical[i])
            status = 3 if residual.any() else 1
            assert not (residual & ~syndrome).any()
            packed_s, packed_r = pack(syndrome, width), pack(residual, width)
            print(int(modes[i]), clocks, int(status == 3), logical, file=out)
            for payload in (packed_s, packed_r):
                print(' '.join(f'{int.from_bytes(payload[j:j+depth], "little"):x}'
                               for j in range(0, len(payload), depth)), file=out)
            token = 0x7d7e+i
            op = COMMAND['tilt' if modes[i] else 'flat']
            request = frame(REQUEST.pack(op, token, tag)+packed_s)
            if i == 0:
                malformed = bytearray(request)
                malformed[1] ^= 1
                request = bytes(malformed)+request
            body = packed_r if status == 3 else b''
            reply = frame(REPLY.pack(0x92, token, tag, status, clocks, logical, len(body))+body)
            replies.append((op, status, clocks, request, reply, rescue_cases[i],
                            growth_cases[i] if modes[i] else 0, weak_cases[i] and modes[i]))

    cases = [replies[0], replies[1]]
    for opcode in COMMAND.values():
        for status in (1, 3):
            cases.extend([entry for entry in replies if entry[:2] == (opcode, status)][:4])
    if not original.plan.lookup:
        cases.extend([entry for entry in replies if entry[1] == 1 and entry[2] == original.plan.max_clocks][:3])
        cases.extend([entry for entry in replies if entry[5]][:3])
        for state in (1, 2, 3):
            cases.extend([entry for entry in replies if entry[6] == state][:3])
        for status in (1, 3):
            cases.extend([entry for entry in replies if entry[1] == status and entry[7]][:2])
    with (directory/'uart.txt').open('w') as out:
        print(len(cases), file=out)
        for _, _, _, sent, received, _, _, _ in cases:
            print(len(sent), len(received), file=out)
            print(sent.hex(' '), file=out)
            print(received.hex(' '), file=out)
    print(f'{len(shots)} direct and {len(cases)} UART fixtures with up to {int(full_result.clocks.max())} FPGA clocks')
    if original.patch is not None:
        print(f'{int(rescue_cases.sum())} native fixtures exercise the residual rescue')
    if image.get('frontend') == 'cluster':
        print(f'{int(np.count_nonzero((growth_cases != 0) & modes))} native fixtures exercise clustering or timeout')
    if original.cut_guard:
        print(f'{int(np.count_nonzero(weak_cases & modes))} native fixtures route a weak tilt answer to clustering')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    generate(args.image)

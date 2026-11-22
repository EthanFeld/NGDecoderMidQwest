#!/usr/bin/env python3
"""Measure paired end-to-end LER and FPGA defer rate away from calibration."""

import argparse
from collections import Counter
import json
from math import sqrt
from pathlib import Path

import numpy as np

from reference import Predecoder, matching, sample
from surface_model import generate


def rate(errors, trials):
    z = 1.959963984540054
    ratio = errors/trials
    denom = 1 + z*z/trials
    mid = (ratio + z*z/(2*trials))/denom
    width = z*sqrt((ratio*(1-ratio) + z*z/(4*trials))/trials)/denom
    return {'events': errors, 'rate': ratio, 'wilson_95': [max(0, mid-width), min(1, mid+width)]}


def sweep(distance, basis, noise, shots, seed, calibration=.001, batch=1024):
    physical = generate(distance, basis, distance, noise)
    firmware = generate(distance, basis, distance, calibration)
    firmware['frontend'] = 'map' if distance == 3 else 'rescue'
    pre = Predecoder(firmware)
    cpu = matching(physical)
    rng = np.random.default_rng(seed)
    counts = Counter()
    cycles = Counter()
    prompt = Predecoder({**physical, 'frontend': 'map'}) if distance == 3 else None

    for start in range(0, shots, batch):
        n = min(shots-start, batch)
        syndrome, actual = sample(physical, n, rng)
        candidate = pre.decode(syndrome)
        rescued = candidate.rescued if candidate.rescued is not None else np.zeros(n, bool)
        accepted = candidate.accepted
        old_accept = accepted & ~rescued
        mwpm = cpu.decode_batch(syndrome)[:, 0].astype(bool)
        new = np.where(accepted, candidate.logical, mwpm)
        old = np.where(old_accept, candidate.logical, mwpm)
        prior_error, new_error = old != actual, new != actual

        counts['defer_before'] += int(np.count_nonzero(~old_accept))
        counts['defer_after'] += int(np.count_nonzero(~accepted))
        counts['ler_before'] += int(prior_error.sum())
        counts['ler_after'] += int(new_error.sum())
        counts['ler_mwpm'] += int(np.count_nonzero(mwpm != actual))
        counts['newly_wrong'] += int(np.count_nonzero(new_error & ~prior_error))
        counts['newly_right'] += int(np.count_nonzero(prior_error & ~new_error))
        counts['wrong_among_rescues'] += int(np.count_nonzero(rescued & new_error))
        counts['rescued'] += int(rescued.sum())
        counts['wrong_fpga_before'] += int(np.count_nonzero(old_accept & prior_error))
        counts['wrong_fpga_after'] += int(np.count_nonzero(accepted & new_error))
        if prompt is not None:
            counts['ler_matched_map'] += int(np.count_nonzero(prompt.decode(syndrome).logical != actual))
        cycles.update(candidate.clocks.tolist())

    assert counts['rescued'] == counts['defer_before'] - counts['defer_after']
    assert counts['ler_after'] - counts['ler_before'] == counts['newly_wrong'] - counts['newly_right']
    return {
        'distance': distance, 'rounds': distance, 'basis': basis, 'p': noise,
        'fpga_prior_p': calibration, 'cpu_prior_p': noise, 'shots': shots, 'seed': seed, 'batch': batch,
        **{k: rate(counts[k], shots) for k in ('defer_before', 'defer_after', 'ler_before', 'ler_after', 'ler_mwpm')},
        'paired': {k: counts[k] for k in ('rescued', 'newly_wrong', 'newly_right', 'wrong_among_rescues',
                                         'wrong_fpga_before', 'wrong_fpga_after')},
        'ler_matched_map': rate(counts['ler_matched_map'], shots) if prompt is not None else None,
        'cycles': dict(sorted(cycles.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance', type=int, choices=(3, 7, 15), required=True)
    parser.add_argument('--basis', choices=('x', 'z', 'both'), default='both')
    parser.add_argument('--p', type=float, nargs='+', default=(.002, .003, .005, .007))
    parser.add_argument('--fpga-prior', type=float, default=.001)
    parser.add_argument('--shots', type=int, default=100000)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=720260000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.shots < 1 or args.batch < 1:
        parser.error('shots and batch size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    bases = ('x', 'z') if args.basis == 'both' else (args.basis,)
    for p in args.p:
        for basis in bases:
            seed = args.seed+args.distance*1_000_000+round(p*1_000_000)*10+(basis == 'x')
            results = sweep(args.distance, basis, p, args.shots, seed, args.fpga_prior, args.batch)
            name = f'd{args.distance}{basis}-p{round(p*1_000_000):06d}.json'
            (args.output/name).write_text(json.dumps(results, indent=2)+'\n')
            print(f'{name} | {args.shots} shots | '
                  f"defer {100*results['defer_before']['rate']:.3f}% to {100*results['defer_after']['rate']:.3f}% | "
                  f"LER {100*results['ler_before']['rate']:.5f}% to {100*results['ler_after']['rate']:.5f}%", flush=True)


if __name__ == '__main__':
    main()

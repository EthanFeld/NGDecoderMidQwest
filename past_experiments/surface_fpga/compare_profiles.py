#!/usr/bin/env python3
"""Compare two FPGA profiles on the same fault draws and CPU fallback."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from growth_reference import GrowthModel
from reference import Predecoder, matching, sample
from surface_model import generate
from sweep import rate


class Profile:
    def __init__(self, path):
        self.image = json.loads(path.read_text())
        self.front = Predecoder(self.image)
        self.budget = self.image.get('growth_cycles', 0)
        self.growth = None
        if self.budget:
            graph = path.with_suffix('.growth.txt')
            self.growth = GrowthModel(graph if graph.is_file() else path.with_name('growth_graph.txt'))
        self.outcomes = Counter()
        self.clocks = Counter()

    def decode(self, syndrome, truth, cpu_wrong):
        result = self.front.decode(syndrome)
        tilt = result.accepted
        if self.growth is not None:
            self.growth.recover(result, syndrome, self.budget)
        fpga = result.accepted
        cluster = fpga & ~tilt
        wrong = np.where(fpga, result.logical != truth, cpu_wrong)
        self.outcomes.update({
            'tilt_answers': int(tilt.sum()), 'tilt_errors': int((tilt & wrong).sum()),
            'cluster_answers': int(cluster.sum()), 'cluster_errors': int((cluster & wrong).sum()),
            'cpu_answers': int((~fpga).sum()), 'cpu_errors': int((~fpga & wrong).sum()),
            'logical_errors': int(wrong.sum()),
        })
        self.clocks.update(result.clocks.tolist())
        return wrong

    def summarize(self, shots):
        counts = self.outcomes
        fpga_errors = counts['tilt_errors'] + counts['cluster_errors']
        fpga_answers = counts['tilt_answers'] + counts['cluster_answers']
        assert fpga_answers + counts['cpu_answers'] == shots
        assert fpga_errors + counts['cpu_errors'] == counts['logical_errors']
        percentiles = {}
        total = 0
        for clocks, count in sorted(self.clocks.items()):
            total += count
            for q in (50, 95, 99):
                if total*100 >= shots*q and str(q) not in percentiles:
                    percentiles[str(q)] = clocks
        return {
            'image': {k: v for k, v in self.image.items() if k.startswith(('growth_', 'decoder_clock'))},
            'counts': dict(counts),
            'ler': rate(counts['logical_errors'], shots),
            'defer_rate': rate(counts['cpu_answers'], shots),
            'fpga_ler_among_answers': rate(fpga_errors, fpga_answers) if fpga_answers else None,
            'cycle_percentiles': percentiles, 'max_cycles': max(self.clocks),
            'clock_histogram': sorted(self.clocks.items()),
        }


def compare(before, after, p, shots, seed):
    a, b = Profile(before), Profile(after)
    fields = ('distance', 'basis', 'rounds')
    dimensions = tuple(a.image[name] for name in fields)
    if dimensions != tuple(b.image[name] for name in fields):
        raise ValueError('profiles must use the same distance, basis and rounds')
    physical = generate(*dimensions, p)
    mwpm = matching(physical)
    rng = np.random.default_rng(seed)
    counts = Counter()
    for offset in range(0, shots, 1024):
        syndrome, truth = sample(physical, min(shots-offset, 1024), rng)
        cpu_wrong = mwpm.decode_batch(syndrome)[:, 0].astype(bool) != truth
        old = a.decode(syndrome, truth, cpu_wrong)
        new = b.decode(syndrome, truth, cpu_wrong)
        counts['cpu_only_errors'] += int(cpu_wrong.sum())
        counts['regressed'] += int((~old & new).sum())
        counts['corrected'] += int((old & ~new).sum())
    assert b.outcomes['logical_errors'] - a.outcomes['logical_errors'] == counts['regressed'] - counts['corrected']
    return {
        **dict(zip(fields, dimensions, strict=True)), 'p': p, 'shots': shots, 'seed': seed,
        'paired': dict(counts), 'before': a.summarize(shots), 'after': b.summarize(shots),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--p', type=float, required=True)
    parser.add_argument('--shots', type=int, default=100_000)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.shots < 1:
        parser.error('shots must be positive')
    result = compare(args.before, args.after, args.p, args.shots, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for label in ('before', 'after'):
        print(label, result[label]['counts'])
    print(result['paired'])


if __name__ == '__main__':
    main()

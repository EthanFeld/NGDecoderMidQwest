#!/usr/bin/env python3
"""Replay the same shots through FPGA variants and through matching."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from reference import Predecoder, matching, sample


def evaluate(image, count, seed, clock, batch):
    cpu = matching(image)
    local = Predecoder(image)
    rng = np.random.default_rng(seed)
    stats = {name: Counter() for name in ('fpga_only_flat', 'fpga_with_tilt', 'fpga_and_cpu_on_residual',
                                          'fpga_and_cpu_on_original', 'cpu_on_original')}
    input_weights, residual_weights = [], []
    core_cycles = Counter()
    full_subset_s = residual_s = 0.0
    for start in range(0, count, batch):
        length = min(batch, count-start)
        syndrome, truth = sample(image, length, rng)
        flat, tilted = local.decode(syndrome, tilt=False), local.decode(syndrome)
        result = cpu.decode_batch(syndrome)[:, 0].astype(bool)
        a0, a1 = flat.accepted, tilted.accepted
        misses = ~a1
        assert not np.any(tilted.residual & ~syndrome)
        for dest, mode, accepted in (('fpga_only_flat', flat, a0), ('fpga_with_tilt', tilted, a1)):
            stats[dest]['accepted'] += int(accepted.sum())
            stats[dest]['wrong_accepted'] += int((accepted & (mode.logical != truth)).sum())
        stats['cpu_on_original']['wrong'] += int((result != truth).sum())
        stats['fpga_and_cpu_on_original']['wrong'] += int((np.where(a1, tilted.logical, result) != truth).sum())
        direct = tilted.logical.copy()
        if misses.any():
            raw = syndrome[misses]
            reduced = tilted.residual[misses]
            start_cpu = perf_counter()
            solved = cpu.decode_batch(reduced)[:, 0].astype(bool)
            residual_s += perf_counter()-start_cpu
            start_cpu = perf_counter()
            cpu.decode_batch(raw)
            full_subset_s += perf_counter()-start_cpu
            direct[misses] ^= solved
            input_weights.extend(raw.sum(axis=1).tolist())
            residual_weights.extend(reduced.sum(axis=1).tolist())
        stats['fpga_and_cpu_on_residual']['wrong'] += int((direct != truth).sum())
        core_cycles.update(tilted.clocks.tolist())
    return {
        'distance': image['distance'], 'basis': image['basis'], 'rounds': image['rounds'],
        'frontend': image.get('frontend', 'tilt'),
        'p': image['p'], 'shots': count, 'seed': seed, 'detectors': image['checks'],
        'fault_edges': len(image['faults']), 'graph_fault_distance': image['fault_distance'],
        'counts': stats, 'fpga_clock_mhz': clock, 'fpga_cycle_histogram': dict(sorted(core_cycles.items())),
        'fpga_worst_us': max(core_cycles)/clock,
        'median_active_before_defer': float(np.median(input_weights)) if input_weights else None,
        'median_active_after_defer': float(np.median(residual_weights)) if residual_weights else None,
        'mean_cpu_matching_us_on_original_defers': full_subset_s*1e6/len(input_weights) if input_weights else None,
        'mean_cpu_matching_us_on_residual_defers': residual_s*1e6/len(input_weights) if input_weights else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--shots', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=507915)
    parser.add_argument('--clock', type=float, default=135)
    parser.add_argument('--batch', type=int, default=1024)
    args = parser.parse_args()
    if min(args.shots, args.clock, args.batch) <= 0:
        parser.error('positive shot count, clock, and batch required')
    print(json.dumps(evaluate(json.loads(args.graph.read_text()), args.shots, args.seed, args.clock, args.batch),
                     indent=2))


if __name__ == '__main__':
    main()

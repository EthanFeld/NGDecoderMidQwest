#!/usr/bin/env python3
"""Measure the existing fallback and bounded full-syndrome clustering on paired shots."""

from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np

from growth import GROW_BUDGET, compile_growth, growth_cycles
from growth_reference import GrowthModel
from reference import Predecoder, matching, sample
from surface_model import generate, growth_precision
from sweep import rate


def evaluate(distance, basis, p, resolution, shots, seed, output, clock_mhz=135, horizon=6000,
             engine='serial', cut_guard=False, compact=False, cut_limit=None,
             boundary_bias=0, skip_empty=False):
    actual = generate(distance, basis, distance, p)
    old_image = generate(distance, basis, distance, .001)
    profile = {**old_image, 'frontend': 'cluster', 'decoder_clock_mhz': clock_mhz,
               'growth_resolution': resolution, 'growth_p': p, 'growth_engine': engine,
               'growth_cut_guard': cut_guard, 'growth_compact': compact,
               'growth_boundary_bias': boundary_bias, 'growth_skip_empty': skip_empty}
    if cut_limit is not None:
        profile['growth_cut_limit'] = cut_limit
    cap = growth_cycles(profile, p, clock_mhz)
    if cap < 1:
        raise ValueError('the original deadline leaves no clustering cycles at this clock')
    profile['growth_cycles'] = cap
    suffix = profile_suffix(engine, cut_guard, compact, cut_limit, boundary_bias, skip_empty)
    folder = output / f'd{distance}{basis}-p{p:g}-q{resolution:g}{suffix}'
    folder.mkdir(parents=True, exist_ok=True)
    compile_growth(profile, folder)
    deep = GrowthModel(folder/'growth_graph.txt')
    fast = Predecoder(profile)
    old = Predecoder({**old_image, 'frontend': 'rescue'}) if compact else None
    cpu = matching(actual)
    rng = np.random.default_rng(seed)
    counts = Counter()
    clocks = Counter()
    recovered_at = Counter()
    wrong_delta_at = Counter()
    for n in range(0, shots, 1024):
        size = min(1024, shots-n)
        s, truth = sample(actual, size, rng)
        head = fast.decode(s, apply_cut_guard=False)
        prior = old.decode(s) if old is not None else head
        original_defers = ~prior.accepted
        weak = fast.weak_answers(s, head)
        misses = ~head.accepted | weak
        answer = cpu.decode_batch(s)[:, 0].astype(bool)
        v = deep.decode(s[misses], horizon)
        agree = v[:, 1] == 0
        within = v[:, 2] < cap
        keep = agree & within
        raw = answer[misses] != truth[misses]
        deep_error = v[:, 0].astype(bool) != truth[misses]
        original_wrong = np.where(original_defers, answer, prior.logical) != truth
        old_on_retries = original_wrong[misses]
        new_on_retries = np.where(keep, deep_error, raw)
        fast_wrong = (head.logical != truth) & ~misses
        fast_errors = int(np.count_nonzero(fast_wrong))
        counts['cpu_mwpm_wrong'] += int(np.count_nonzero(answer!=truth))
        counts['old_defer'] += int(original_defers.sum())
        counts['old_wrong'] += int(original_wrong.sum())
        counts['growth_attempts'] += int(misses.sum())
        counts['wrong_if_cpu_on_retries'] += fast_errors + int(raw.sum())
        counts['reconsidered'] += int(weak.sum())
        counts['reconsidered_tilt_wrong'] += int(np.count_nonzero(weak & original_wrong))
        counts['reconsidered_defer'] += int(np.count_nonzero(weak[misses] & ~keep))
        counts['reconsidered_final_wrong'] += int(np.count_nonzero(weak[misses] & new_on_retries))
        counts['new_defer'] += int((~keep).sum())
        found_wrong = int(np.count_nonzero(keep & deep_error))
        cpu_wrong = int(np.count_nonzero(~keep & raw))
        counts['tilt_wrong'] += fast_errors
        counts['clustering_wrong'] += found_wrong
        counts['cpu_defer_wrong'] += cpu_wrong
        counts['new_wrong'] += fast_errors + found_wrong + cpu_wrong
        counts['newly_wrong'] += (int(np.count_nonzero(new_on_retries & ~old_on_retries)) +
                                  int(np.count_nonzero(fast_wrong & ~original_wrong)))
        counts['newly_right'] += (int(np.count_nonzero(~new_on_retries & old_on_retries)) +
                                  int(np.count_nonzero(~misses & ~fast_wrong & original_wrong)))
        counts['timeouts'] += int(np.count_nonzero(~within))
        counts['ambiguous'] += int(np.count_nonzero(within & (v[:, 1] == 2)))
        counts['relaxed_defer'] += int(np.count_nonzero(~agree))
        counts['relaxed_wrong'] += fast_errors + int(np.where(agree, deep_error, raw).sum())
        decoder_clocks = head.clocks[misses] + 1 + v[:, 2]
        recovered_at.update(decoder_clocks[agree].tolist())
        wrong_delta_at.update(decoder_clocks[agree & deep_error & ~raw].tolist())
        wrong_delta_at.subtract(decoder_clocks[agree & ~deep_error & raw].tolist())
        clocks.update(head.clocks[~misses].tolist())
        clocks.update((head.clocks[misses] + 1 + np.minimum(cap, v[:, 2])).tolist())
    hist = {}
    for cutoff in (50, 95, 99):
        needed = (shots*cutoff + 99)//100
        total = 0
        for latency, count in sorted(clocks.items()):
            total += count
            if total >= needed:
                hist[str(cutoff)] = latency
                break
    assert counts['new_defer'] == counts['timeouts']+counts['ambiguous']
    assert counts['new_wrong']-counts['old_wrong'] == counts['newly_wrong']-counts['newly_right']
    return {
        'distance': distance, 'basis': basis, 'p': p, 'rounds': distance, 'shots': shots, 'seed': seed,
        'growth_resolution': resolution, 'fpga_prior_p': .001, 'growth_prior_p': p,
        'growth_engine': engine, 'growth_cut_guard': cut_guard, 'growth_compact': compact,
        'growth_cut_limit': fast.cut_limit, 'growth_boundary_bias': boundary_bias,
        'growth_skip_empty': skip_empty, 'growth_initial_age': deep.initial_age,
        'cap_extra_cycles': cap+1, 'cap_total_cycles': fast.plan.max_clocks+cap+1,
        'max_board_clock_mhz_used': clock_mhz,
        'budget_at_135mhz_cycles': GROW_BUDGET[distance][p],
        'stat': {k:rate(counts[k], shots) for k in ('old_defer','new_defer','old_wrong','new_wrong','cpu_mwpm_wrong')},
        'paired': dict(counts),
        'cycle_percentiles': hist,
        'cycles_max': max(clocks),
        'cycle_model_horizon': horizon,
        'recovery_curve_reference': 'tilt on unretried shots and original-syndrome matching on retried shots',
        'recovery_by_total_cycle': [[cycle, n, wrong_delta_at[cycle]]
                                    for cycle, n in sorted(recovered_at.items())],
    }


def profile_suffix(engine, cut_guard, compact, cut_limit, boundary_bias, skip_empty):
    parts = []
    if engine != 'serial':
        parts.append(engine)
    if cut_guard:
        parts.append('cut-guard' if cut_limit is None else f'cut-{cut_limit}')
    if compact:
        parts.append('compact')
    if boundary_bias:
        parts.append(f'boundary-{boundary_bias:g}')
    if skip_empty:
        parts.append('skip-empty')
    return ''.join(f'-{part}' for part in parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('distance', type=int, choices=(7,15))
    parser.add_argument('basis', choices=('x','z','both'))
    parser.add_argument('p', type=float, choices=(.002,.003,.005,.007), nargs='+')
    parser.add_argument('--resolution', type=growth_precision, default=2)
    parser.add_argument('--engine', choices=('serial', 'dual-port'), default='serial')
    parser.add_argument('--cut-guard', action='store_true')
    parser.add_argument('--cut-limit', type=int, choices=range(8))
    parser.add_argument('--boundary-bias', type=float, default=0)
    parser.add_argument('--skip-empty', action='store_true')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--shots', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=912450000)
    parser.add_argument('--clock-mhz', type=int, default=135)
    parser.add_argument('--output', type=Path, default=Path('build/adaptive'))
    args = parser.parse_args()
    if args.cut_guard and args.distance != 7:
        parser.error('cut-guard requires distance 7')
    if args.cut_limit is not None and not args.cut_guard:
        parser.error('cut-limit requires cut-guard')
    if not -1 <= args.boundary_bias <= 1:
        parser.error('boundary-bias must be between -1 and 1')
    if args.compact and args.distance != 15:
        parser.error('compact requires distance 15')
    bases = ('x','z') if args.basis == 'both' else (args.basis,)
    for p in args.p:
        for basis in bases:
            seed = args.seed+args.distance*100000+round(p*1000000)*2+(basis=='x')
            report = evaluate(args.distance, basis, p, args.resolution, args.shots, seed,
                              args.output, args.clock_mhz, engine=args.engine, cut_guard=args.cut_guard,
                              compact=args.compact, cut_limit=args.cut_limit, boundary_bias=args.boundary_bias,
                              skip_empty=args.skip_empty)
            suffix = profile_suffix(args.engine, args.cut_guard, args.compact, args.cut_limit,
                                    args.boundary_bias, args.skip_empty)
            name = f"d{args.distance}{basis}-p{p:g}-q{args.resolution:g}{suffix}.json"
            (args.output/name).write_text(json.dumps(report, indent=2)+'\n')
            print(name, report['paired'], report['cycle_percentiles'], flush=True)


if __name__ == '__main__':
    main()

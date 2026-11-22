"""Run the frozen Relay-BP with one native decoder per persistent process."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import relay_bp

from run_relay_bp import (
    HERE, RATES, SHOTS, build_context, deduplicate, detailed_decode, get_profile,
    ibm_relay_bp, names, sha256, validate_samples, wilson, write_json,
)


def initialize(config, matrix, observables, priors, detector_offset, observable_offset, ready):
    global runner
    native = relay_bp.RelayDecoderF32(matrix, error_priors=priors, **config.native_kwargs())
    runner = SimpleNamespace(
        config=config, num_detectors=matrix.shape[0], detector_offset=detector_offset,
        observable_offset=observable_offset,
        runner=relay_bp.ObservableDecoderRunner(native, observables, include_decode_result=False),
    )
    ready.wait(timeout=120)


def decode_worker(rows):
    return detailed_decode(runner, rows, parallel=False)


def decode_pool(pool, rows, chunk):
    slices = (rows[start:start + chunk] for start in range(0, len(rows), chunk))
    split_results = pool.map(decode_worker, slices, chunksize=1)
    return tuple(np.concatenate(outputs) for outputs in zip(*split_results, strict=True))


def run_cohort(args, decoder, events, truth, seed, noise, checkpoint, inherited):
    unique, inverse, frequency = deduplicate(events)
    count = len(unique)
    key = f'15:{noise}'
    prior = checkpoint.get(key, inherited.get(key, {}))
    done = prior.get('done', 0)
    assert 0 <= done <= count
    total_active = prior.get('active_seconds', 0.0)
    inherited_rows = prior.get('inherited_rows', done)
    prediction = np.empty(count, dtype=np.uint8)
    convergence = np.empty(count, dtype=bool)
    iterations = np.empty(count, dtype='<u2')
    if done:
        prediction[:done] = np.frombuffer(bytes.fromhex(prior['predictions']), dtype=np.uint8)
        convergence[:done] = np.frombuffer(bytes.fromhex(prior['convergence']), dtype=bool)
        iterations[:done] = np.frombuffer(bytes.fromhex(prior['iterations']), dtype='<u2')

    pilot = np.linspace(0, count - 1, min(16, count), dtype=np.intp)
    if done:
        pilot = np.union1d(pilot, np.linspace(0, done - 1, min(8, done), dtype=np.intp))
    probes = unique[pilot]
    probes.flags.writeable = False
    expected = detailed_decode(decoder, probes, parallel=False)
    np.testing.assert_array_equal(decoder.decode_batch(probes)[:, 0], expected[0])
    for known, reference in zip((prediction, convergence, iterations), expected, strict=True):
        np.testing.assert_array_equal(known[pilot[pilot < done]], reference[pilot < done])

    ctx = get_context('spawn')
    ready = ctx.Barrier(args.workers + 1)
    init_args = (
        decoder.config, decoder.check_matrix, decoder.observable_matrix, decoder.error_priors,
        decoder.detector_offset, decoder.observable_offset, ready,
    )
    print(f'd15 p={noise:g}: {count:,} unique, {done:,} checkpointed', flush=True)
    if done < count:
        with ctx.Pool(args.workers, initializer=initialize, initargs=init_args) as pool:
            ready.wait(timeout=120)
            actual = decode_pool(pool, probes[::-1].copy(), chunk=4)
            for reference, ordered in zip(expected, actual, strict=True):
                np.testing.assert_array_equal(reference, ordered[::-1])
            for start in range(done, count, args.checkpoint_rows):
                stop = min(start + args.checkpoint_rows, count)
                before = perf_counter()
                prediction[start:stop], convergence[start:stop], iterations[start:stop] = decode_pool(
                    pool, unique[start:stop], chunk=args.chunk,
                )
                elapsed = perf_counter() - before
                total_active += elapsed
                checkpoint[key] = {
                    'done': stop, 'active_seconds': total_active, 'inherited_rows': inherited_rows,
                    'predictions': prediction[:stop].tobytes().hex(),
                    'convergence': convergence[:stop].tobytes().hex(),
                    'iterations': iterations[:stop].tobytes().hex(),
                }
                write_json(args.checkpoint, checkpoint)
                print(f'd15 p={noise:g}: {int(frequency[:stop].sum()):,}/{SHOTS:,} shots, '
                      f'{stop - start:,} unique in {elapsed:.1f}s', flush=True)

    full = prediction[inverse]
    errors = int(np.count_nonzero(full != truth))
    low, high = wilson(errors, SHOTS)
    return {
        'distance': 15, 'p': noise, 'seed': seed, 'shots': SHOTS,
        'logical_errors': errors, 'logical_error_rate': errors / SHOTS,
        'ci95_low': low, 'ci95_high': high, 'unique_syndromes': count,
        'nonconverged_shots': int(np.count_nonzero(~convergence[inverse])),
        'average_iterations': float(iterations[inverse].mean()),
        'native_parallel_active_seconds': total_active, 'checkpointed_prefix_rows': inherited_rows,
        'predictions_bit_packed_little_hex': np.packbits(full, bitorder='little').tobytes().hex(),
        'prediction_sha256': hashlib.sha256(full.tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rates', nargs='+', choices=RATES, type=float, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--checkpoint-rows', type=int, default=384)
    parser.add_argument('--output', type=Path, default=HERE / 'relay_bp_persistent.json')
    parser.add_argument('--checkpoint', type=Path, default=HERE / '.relay_persistent_checkpoint.json')
    parser.add_argument('--inherit-checkpoint', type=Path)
    args = parser.parse_args()
    if min(args.workers, args.chunk, args.checkpoint_rows) < 1 or len(set(args.rates)) != len(args.rates):
        parser.error('Use positive worker and chunk sizes and distinct rates')
    os.environ['RAYON_NUM_THREADS'] = '1'
    joint_path, score_path = names(15)
    src_hash = {p.name: sha256(p) for p in (joint_path, score_path)}
    settings = {
        'decoder': ibm_relay_bp.qecbench_implementation_identity(),
        'config': asdict(ibm_relay_bp.FAST_CONFIG), 'sources_sha256': src_hash,
        'execution': 'pinned native detailed serial decode per persistent worker process',
        'workers': args.workers, 'chunk': args.chunk, 'checkpoint_rows': args.checkpoint_rows,
        'rates': args.rates, 'runner_sha256': sha256(Path(__file__)),
    }
    settings = json.loads(json.dumps(settings))
    score = json.loads(score_path.read_bytes())
    if score['joint']['sha256'] != src_hash[joint_path.name] or not score['completed']:
        raise ValueError('Stored joint cohorts differ from the verified run')
    inherited = {}
    if args.inherit_checkpoint:
        inherited = json.loads(args.inherit_checkpoint.read_bytes())
        compatible = inherited['settings']
        if (any(compatible[key] != settings[key] for key in ('decoder', 'config'))
                or any(compatible['sources_sha256'].get(name) != digest for name, digest in src_hash.items())):
            raise ValueError('Cannot continue a different decoder or inputs')
        settings['inherited_checkpoint_sha256'] = sha256(args.inherit_checkpoint)
    checkpoint = {'settings': settings}
    report = {'settings': settings, 'cohorts': []}
    for path, default in ((args.checkpoint, checkpoint), (args.output, report)):
        if path.exists():
            existing = json.loads(path.read_bytes())
            if existing['settings'] != settings:
                raise ValueError(f'Checkpoint or report provenance differs for {path.name}')
            default.update(existing)

    completed = {row['p'] for row in report['cohorts']}
    profile = get_profile(15)
    with np.load(joint_path, allow_pickle=False) as joint:
        for noise in args.rates:
            if noise in completed:
                continue
            index = RATES.index(noise)
            context, indices = build_context(profile.image, noise)
            events, truth, seed = validate_samples(joint, index, context, indices, score['cohorts'][index])
            decoder = ibm_relay_bp.build_fast_decoder(context)
            result = run_cohort(args, decoder, events, truth, seed, noise, checkpoint, inherited)
            report['cohorts'].append(result)
            report['completed'] = len(report['cohorts']) == len(args.rates)
            report['updated_at'] = datetime.now(UTC).isoformat()
            write_json(args.output, report)
            print(f'd15 p={noise:g}: {result["logical_errors"]:,}/{SHOTS:,} logical errors', flush=True)


if __name__ == '__main__':
    main()

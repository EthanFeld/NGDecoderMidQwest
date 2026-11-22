"""Run the frozen Relay-BP baseline on the joint samples of the live FPGA sweep."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
SURFACE = ROOT / 'past_experiments/surface_fpga'
sys.path[:0] = [str(ROOT), str(SURFACE), str(ROOT / "decoders/telescope"), str(ROOT / "decoders/telescope/tensor_decoder/src"), str(ROOT / "decoders/pymatching_telescope/src"), str(ROOT / "past_experiments/decoder_testing/src")]

from past_experiments.comparisons.sweep import get_profile, wilson
from surface_decoders import ibm_relay_bp
from telescope_circuit import build_context

RATES = (.001, .003, .005, .007)
DISTANCES = (7, 15)
SHOTS = 10_000


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, report):
    staging = path.with_suffix('.tmp')
    staging.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    staging.replace(path)


def names(distance):
    return HERE / f'd{distance}_joint.npz', HERE / f'd{distance}_score.json'


def validate_samples(joint, index, context, memory_map, score):
    p, seed = float(joint['rates'][index]), int(joint['seeds'][index])
    events = np.ascontiguousarray(joint[f'events_{index}'])
    truth = joint[f'truth_{index}'].reshape(-1)
    if (p, seed, len(events), len(truth)) != (score['p'], score['seed'], SHOTS, SHOTS):
        raise ValueError('Joint sample identity does not match the verified FPGA cohort')
    if not score['complete'] or score['verified_joint_shots'] != SHOTS:
        raise ValueError('Expected a fully verified FPGA cohort')
    np.testing.assert_array_equal(joint['memory_indices'], memory_map)
    expected, observables = context.circuit.compile_detector_sampler(seed=seed).sample(
        SHOTS, bit_packed=True, separate_observables=True,
    )
    np.testing.assert_array_equal(events, expected)
    np.testing.assert_array_equal(truth, observables[:, 0])
    events.flags.writeable = False
    return events, truth, seed


def detailed_decode(decoder, rows, parallel):
    syndrome = np.unpackbits(rows, axis=1, count=decoder.num_detectors, bitorder='little')
    syndrome ^= decoder.detector_offset
    syndrome.flags.writeable = False
    decoded = decoder.runner.decode_observables_detailed_batch(
        syndrome, parallel=parallel, progress_bar=False,
    )
    prediction = np.fromiter((row.observables[0] for row in decoded), dtype=np.uint8, count=len(rows))
    prediction ^= decoder.observable_offset[0]
    converged = np.fromiter((row.converged for row in decoded), dtype=np.bool_, count=len(rows))
    iterations = np.fromiter((row.iterations for row in decoded), dtype=np.uint16, count=len(rows))
    if np.any(prediction > 1) or np.any(iterations > decoder.config.max_iterations):
        raise ValueError('Invalid native Relay-BP label or iteration telemetry')
    return prediction, converged, iterations


def deduplicate(events):
    _, first_rows, inverse, frequency = np.unique(
        events, axis=0, return_index=True, return_inverse=True, return_counts=True,
    )
    order = np.argsort(first_rows)
    to_ordered = np.empty_like(order)
    to_ordered[order] = np.arange(len(order))
    return events[first_rows[order]], to_ordered[inverse], frequency[order]


def run_cohort(args, distance, p, context, events, truth, seed, checkpoint, checkpoint_key):
    decoder = ibm_relay_bp.build_fast_decoder(context)
    if decoder.config.detector_mode != 'full' or decoder.num_observables != 1 or decoder.runner is None:
        raise ValueError('Expected the native full joint Relay-BP runner')
    unique, inverse, frequency = deduplicate(events)
    pilot = unique[np.linspace(0, len(unique) - 1, min(12, len(unique)), dtype=np.intp)]
    pilot.flags.writeable = False
    serial = decoder.decode_batch(pilot)[:, 0]
    parallel, converged, iterations = detailed_decode(decoder, pilot[::-1].copy(), True)
    np.testing.assert_array_equal(serial, parallel[::-1])
    serial_detailed = detailed_decode(decoder, pilot, False)
    for ordered, reverse in zip(serial_detailed, (parallel, converged, iterations), strict=True):
        np.testing.assert_array_equal(ordered, reverse[::-1])

    previous = checkpoint.get(checkpoint_key, {})
    unique_count = len(unique)
    prediction = np.empty(unique_count, dtype=np.uint8)
    convergence = np.empty(unique_count, dtype=bool)
    steps = np.empty(unique_count, dtype=np.uint16)
    done = previous.get('done', 0)
    active_seconds = previous.get('active_seconds', 0.0)
    if done:
        prediction[:done] = np.frombuffer(bytes.fromhex(previous['predictions']), dtype=np.uint8)
        convergence[:done] = np.frombuffer(bytes.fromhex(previous['convergence']), dtype=bool)
        steps[:done] = np.frombuffer(bytes.fromhex(previous['iterations']), dtype='<u2')
    print(f'd{distance} p={p:g}: {unique_count:,} unique inputs across {SHOTS:,} shots, {done:,} checkpointed', flush=True)
    last_progress = perf_counter()
    for start in range(done, unique_count, args.batch_size):
        stop = min(start + args.batch_size, unique_count)
        before = perf_counter()
        prediction[start:stop], convergence[start:stop], steps[start:stop] = detailed_decode(
            decoder, unique[start:stop], True,
        )
        active_seconds += perf_counter() - before
        checkpoint[checkpoint_key] = {
            'done': stop, 'active_seconds': active_seconds,
            'predictions': prediction[:stop].tobytes().hex(),
            'convergence': convergence[:stop].tobytes().hex(),
            'iterations': steps[:stop].astype('<u2', copy=False).tobytes().hex(),
        }
        write_json(args.checkpoint, checkpoint)
        if perf_counter() - last_progress >= 15 or stop == unique_count:
            shown = int(frequency[:stop].sum())
            print(f'd{distance} p={p:g}: {stop:,}/{unique_count:,} native solves, {shown:,}/{SHOTS:,} shots, '
                  f'{active_seconds:.1f} s active', flush=True)
            last_progress = perf_counter()

    full_prediction = prediction[inverse]
    errors = int(np.count_nonzero(full_prediction != truth))
    low, high = wilson(errors, SHOTS)
    return {
        'distance': distance, 'p': p, 'seed': seed, 'shots': SHOTS,
        'logical_errors': errors, 'logical_error_rate': errors / SHOTS,
        'ci95_low': low, 'ci95_high': high,
        'unique_syndromes': unique_count, 'nonconverged_shots': int(np.count_nonzero(~convergence[inverse])),
        'average_iterations': float(steps[inverse].mean()),
        'native_parallel_active_seconds': active_seconds,
        'predictions_bit_packed_little_hex': np.packbits(full_prediction, bitorder='little').tobytes().hex(),
        'prediction_sha256': hashlib.sha256(full_prediction.tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=384)
    parser.add_argument('--output', type=Path, default=HERE / 'relay_bp.json')
    parser.add_argument('--checkpoint', type=Path, default=HERE / '.relay_bp_checkpoint.json')
    args = parser.parse_args()
    if min(args.threads, args.batch_size) < 1:
        parser.error('positive threads and batch size required')
    os.environ['RAYON_NUM_THREADS'] = str(args.threads)

    settings = {
        'source': 'original packed joint syndromes and ground truth from the live Chrome FPGA sweep',
        'decoder': ibm_relay_bp.qecbench_implementation_identity(),
        'config': asdict(ibm_relay_bp.FAST_CONFIG),
        'execution': 'pinned native detailed parallel decode, identical syndromes decoded once per cohort',
        'threads': args.threads, 'batch_size': args.batch_size,
        'stim': stim.__version__, 'numpy': np.__version__,
        'sources_sha256': {path.name: sha256(path) for distance in DISTANCES for path in names(distance)},
        'adapter_sha256': sha256(Path(ibm_relay_bp.__file__)),
        'runner_sha256': sha256(Path(__file__)),
    }
    settings = json.loads(json.dumps(settings))
    report = {'settings': settings, 'cohorts': []}
    if args.output.exists():
        report = json.loads(args.output.read_bytes())
        if report['settings'] != settings:
            raise ValueError('Existing result was produced with different software or settings')
    checkpoint = {'settings': settings}
    if args.checkpoint.exists():
        checkpoint = json.loads(args.checkpoint.read_bytes())
        if checkpoint['settings'] != settings:
            raise ValueError('Existing checkpoint was produced with different software or settings')
    completed = {(row['distance'], row['p']) for row in report['cohorts']}
    for distance in DISTANCES:
        input_path, score_path = names(distance)
        board_score = json.loads(score_path.read_bytes())
        if board_score['joint']['sha256'] != settings['sources_sha256'][input_path.name]:
            raise ValueError('Exact joint sample file fingerprint differs from the validated hardware sweep')
        profile = get_profile(distance)
        with np.load(input_path, allow_pickle=False) as joint:
            for index, p in enumerate(RATES):
                if (distance, p) in completed:
                    continue
                context, mapping = build_context(profile.image, p)
                events, truth, seed = validate_samples(joint, index, context, mapping, board_score['cohorts'][index])
                row = run_cohort(args, distance, p, context, events, truth, seed, checkpoint, f'{distance}:{p}')
                report['cohorts'].append(row)
                report['completed'] = len(report['cohorts']) == len(DISTANCES) * len(RATES)
                report['updated_at'] = datetime.now(UTC).isoformat()
                write_json(args.output, report)
                print(f'd{distance} p={p:g} Relay-BP {row["logical_errors"]:,}/{SHOTS:,}, '
                      f'{row["nonconverged_shots"]:,} nonconverged', flush=True)


if __name__ == '__main__':
    main()

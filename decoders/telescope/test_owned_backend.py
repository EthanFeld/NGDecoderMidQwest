"""End-to-end checks for the telescope's independent matching backend."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from owned_backend import OwnedCorrelatedMatching, OwnedTensorDecoder

from tensor_decoder import DecoderConfig, DecoderContext, Experiment


@pytest.fixture(scope="module")
def small_model():
    experiment = Experiment(distance=5, rounds=5, noise=0.005)
    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True),
    )
    events = circuit.compile_detector_sampler(seed=72611907).sample(128, bit_packed=True)
    events.flags.writeable = False
    bits = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")
    return context, events, bits


def difficult_rows(small_model):
    context, events, bits = small_model
    selected = np.argsort(bits.sum(axis=1), kind="stable")[-12:]
    chosen = events[selected].copy()
    chosen.flags.writeable = False
    return context, chosen


def edge_syndrome(pairs, count):
    syndrome = np.zeros(count, dtype=np.uint8)
    for a, b in pairs:
        assert 0 <= a < count
        assert b == -1 or 0 <= b < count
        syndrome[a] ^= 1
        if b >= 0:
            syndrome[b] ^= 1
    return syndrome


def test_correlated_corrections_match_both_observed_bases(small_model):
    context, events, bits = small_model
    matching = OwnedCorrelatedMatching(context)
    logical = []
    for row in bits:
        pairs = matching.decode_to_edges_array(row, enable_correlations=True)
        assert pairs.shape[1:] == (2,)
        np.testing.assert_array_equal(edge_syndrome(pairs, context.num_detectors), row)
        label = 0
        for a, b in pairs:
            key = (int(a),) if b < 0 else tuple(sorted((int(a), int(b))))
            assert key in matching.global_key
            label ^= int(matching.labels[matching.global_key[key]])
        logical.append(label)
    prediction = matching.decode_batch(events)
    assert prediction.dtype == np.uint8
    assert prediction.shape == (128, 1)
    np.testing.assert_array_equal(prediction[:, 0], logical)
    # Reversing the shot order must not alter cached-graph decisions.
    reversed_events = events[::-1].copy()
    reversed_events.flags.writeable = False
    np.testing.assert_array_equal(matching.decode_batch(reversed_events), prediction[::-1])


def test_tensor_predictions_do_not_depend_on_chunks_workers_or_shot_order(small_model):
    context, events = difficult_rows(small_model)
    baseline = OwnedTensorDecoder(
        context, DecoderConfig(chunk_shots=1, path_factor=1000.0),
    )
    parallel = OwnedTensorDecoder(
        context, DecoderConfig(chunk_shots=4, path_factor=1000.0), solve_workers=3,
    )
    try:
        expected = baseline.decode_batch(events)
        np.testing.assert_array_equal(parallel.decode_batch(events), expected)
        permutation = np.random.default_rng(82317).permutation(len(events))
        shuffled = events[permutation].copy()
        shuffled.flags.writeable = False
        np.testing.assert_array_equal(parallel.decode_batch(shuffled), expected[permutation])
        np.testing.assert_array_equal(parallel.decode_batch(events), expected)
        singles = np.concatenate([parallel.decode_batch(events[i:i + 1]) for i in range(len(events))])
        np.testing.assert_array_equal(singles, expected)
    finally:
        baseline.close()
        parallel.close()


def test_fpga_hints_preserve_predictions_and_alignment(small_model):
    context, events = difficult_rows(small_model)
    decoder = OwnedTensorDecoder(context, DecoderConfig(chunk_shots=4, path_factor=1000.0))
    try:
        expected = decoder.decode_batch(events)
        mapping = decoder.control.conditioned.indices
        raw = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")[:, mapping]
        residual = raw.copy()
        residual[:, ::2] = 0
        logical, valid = np.zeros(len(events), np.uint8), np.ones(len(events), bool)
        actual = decoder.decode_with_front(events, residual, logical, valid, mapping)
        np.testing.assert_array_equal(actual, expected)
        stats = decoder.owned_stats()
        assert stats["fpga_hints_received"] == len(events)
        assert stats["fpga_hints_used"] == len(events)
        order = np.arange(len(events))[::-1]
        shuffled = events[order].copy()
        shuffled.flags.writeable = False
        reordered = decoder.decode_with_front(
            shuffled, residual[order], logical[order], valid[order], mapping,
        )
        np.testing.assert_array_equal(reordered, actual[order])
        singles = np.concatenate([
            decoder.decode_with_front(
                events[i:i + 1], residual[i:i + 1], logical[i:i + 1], valid[i:i + 1], mapping,
            ) for i in range(len(events))
        ])
        np.testing.assert_array_equal(singles, actual)
    finally:
        decoder.close()


def test_invalid_and_nonzero_frame_hints_fall_back_to_original_syndrome(small_model):
    context, events = difficult_rows(small_model)
    decoder = OwnedTensorDecoder(context)
    try:
        expected = decoder.decode_batch(events)
        mapping = decoder.control.conditioned.indices
        raw = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")[:, mapping]
        residual = raw.astype(float)
        valid = np.zeros(len(events), bool)
        logical = np.zeros(len(events), float)
        valid[:6] = True
        residual[0, 0] = np.nan
        outside = np.flatnonzero(raw[1] == 0)[0]
        residual[1, outside] = 1
        logical[2] = 1
        logical[3] = 2
        residual[4, 0] = 2
        logical[5] = np.nan
        np.testing.assert_array_equal(
            decoder.decode_with_front(events, residual, logical, valid, mapping), expected,
        )
        assert decoder.owned_stats()["fpga_hints_received"] == 0
        np.testing.assert_array_equal(
            decoder.decode_with_front(
                events, raw, np.ones(len(events), np.uint8), np.ones(len(events), bool), mapping,
            ), expected,
        )
        np.testing.assert_array_equal(
            decoder.decode_with_front(
                events, raw, np.zeros(len(events), np.uint8), np.ones(len(events), bool), mapping[::-1],
            ), expected,
        )
        np.testing.assert_array_equal(
            decoder.decode_with_front(events, raw[:, :-1], logical, valid, mapping), expected,
        )
        assert decoder.owned_stats()["fpga_hints_received"] == 0
    finally:
        decoder.close()


def test_empty_tensor_and_correlated_batches(small_model):
    context, events, _ = small_model
    empty = events[:0]
    decoder = OwnedTensorDecoder(context)
    try:
        result = decoder.decode_batch(empty)
        assert result.shape == (0, 1) and result.dtype == np.uint8
        mapping = decoder.control.conditioned.indices
        result = decoder.decode_with_front(
            empty, np.empty((0, len(mapping)), bool), np.empty(0, np.uint8),
            np.empty(0, bool), mapping,
        )
        assert result.shape == (0, 1) and result.dtype == np.uint8
        result = OwnedCorrelatedMatching(context).decode_batch(empty)
        assert result.shape == (0, 1) and result.dtype == np.uint8
    finally:
        decoder.close()


def test_fresh_process_runs_when_external_matching_imports_and_binaries_are_blocked():
    surface = Path(__file__).resolve().parent
    script = textwrap.dedent(r'''
        import ctypes
        import importlib.abc
        import json
        from pathlib import Path
        import sys

        forbidden = {"pymatching", "networkx", "fusion_blossom", "fusionblossom"}
        class NoExternalMatching(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in forbidden:
                    raise AssertionError("external matching import: " + fullname)
                return None
        sys.meta_path.insert(0, NoExternalMatching())
        original_cdll = ctypes.CDLL
        loaded = []
        def checked_cdll(name, *args, **kwargs):
            base = Path(str(name)).name.lower()
            if (base in {"matching.dll", "matching.so"} or base.startswith("matching_")
                    or "pymatching" in str(name).lower()):
                raise AssertionError("external matching binary: " + str(name))
            loaded.append(base)
            return original_cdll(name, *args, **kwargs)
        ctypes.CDLL = checked_cdll

        import numpy as np
        import telescope
        from owned_backend import OwnedCorrelatedMatching, OwnedTensorDecoder
        from tensor_decoder import DecoderContext, Experiment
        assert telescope.TensorDecoder is OwnedTensorDecoder
        assert telescope.CorrelatedMatching is OwnedCorrelatedMatching
        experiment = Experiment(distance=5, rounds=5, noise=0.005)
        circuit = experiment.circuit()
        context = DecoderContext(
            experiment, circuit, circuit.detector_error_model(decompose_errors=True),
        )
        sampled = circuit.compile_detector_sampler(seed=72611537).sample(96, bit_packed=True)
        weights = np.unpackbits(sampled, axis=1, bitorder="little").sum(axis=1)
        events = sampled[np.argsort(weights)[-3:]].copy()
        assert (np.unpackbits(events, axis=1).sum(axis=1) > 0).all()
        events.flags.writeable = False
        decoder = OwnedTensorDecoder(context)
        try:
            labels = decoder.decode_batch(events)
            assert labels.shape == (3, 1) and np.isin(labels, [0, 1]).all()
            assert not any(name.split(".")[0] in forbidden for name in sys.modules)
            assert any(name.startswith("owned_matching_") for name in loaded)
            print(json.dumps({"shots": len(labels), "nonzero": 3, "loaded": loaded}))
        finally:
            decoder.close()
    ''')
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=surface, env=env,
        capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    assert evidence["shots"] == evidence["nonzero"] == 3

"""Path-level equivalence and independent sector-optimum checks."""

from __future__ import annotations

import itertools
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymatching
import pytest
from scipy.sparse import csc_matrix

from tensor_decoder import DecoderConfig, DecoderContext, Experiment, TensorDecoder
from tensor_decoder.build_native import support_library
from tensor_decoder.conditioned import ConditionedMatching
from tensor_decoder.decoder import SectorSearch
from tensor_decoder.matching import FixedMatching


def context():
    e = Experiment(3, 3, 0.02)
    c = e.circuit()
    return DecoderContext(e, c, c.detector_error_model(decompose_errors=True))


@pytest.fixture(autouse=True)
def cpp_backend(request):
    if shutil.which("g++") is None and shutil.which("C:/msys64/mingw64/bin/g++.exe") is None:
        pytest.skip("optional fixed-topology backend needs a C++20 compiler")
    if (
        request.node.originalname
        in (
            "test_cached_search_matches_all_reference_diagnostics_and_call_order",
            "test_search_seeds_preserve_shot_independence",
        )
        and not Path(support_library()).is_file()
    ):
        pytest.skip("optional original native library is not built")


def test_native_paths_match_pymatching_with_negative_and_tied_weights():
    matrix = csc_matrix(np.array([[1, 0, 1], [0, 1, 1]], np.uint8))
    obs = csc_matrix(np.array([[1, 0, 0]], np.uint8))
    model = SimpleNamespace(
        check_matrix=matrix,
        indices=np.arange(2),
        memory_keys=(((0,), True), ((1,), False), ((0, 1), False)),
        other_keys={},
    )
    fixed = FixedMatching(model)
    rng = np.random.default_rng(79283)
    patterns = [np.zeros(3), np.ones(3), -np.ones(3), *rng.normal(0, 4, (40, 3))]
    expected = []
    for weights in patterns:
        matching = pymatching.Matching.from_check_matrix(
            matrix, weights=weights, faults_matrix=obs, use_virtual_boundary_node=True
        )
        for flags in itertools.product((0, 1), repeat=2):
            syndrome = np.array(flags, np.uint8)
            chosen = matching.decode_to_edges_array(syndrome)
            np.testing.assert_array_equal(fixed.solve_edges(syndrome, weights), chosen)
            expected.append((syndrome, weights, chosen))
    for syndrome, weights, chosen in reversed(expected):
        np.testing.assert_array_equal(fixed.solve_edges(syndrome, weights), chosen)


def test_single_opposite_sector_solve_reproduces_both_optima():
    ctx = context()
    model = ConditionedMatching(ctx)
    reference = model
    fast = SectorSearch(ctx)
    cm = pymatching.Matching.from_detector_error_model(ctx.dem, enable_correlations=True)
    for row in ctx.circuit.compile_detector_sampler(seed=62839).sample(32):
        pairs = cm.decode_to_edges_array(row, enable_correlations=True)
        expected = reference.sector_costs(row, pairs)
        weights = fast.memory_fixed.weights(fast.memory_fixed.prepare(pairs))
        label, cost, _ = fast.memory_fixed.solve_details(row[model.indices], weights)
        other = 1 - int(label[0])
        _, opposite, _ = fast.sector_fixed.solve_details(
            np.r_[row[model.indices], np.uint8(other)], weights
        )
        assert cost == expected[int(label[0])]
        assert opposite == expected[other]


def test_joint_event_workspace_preserves_parity_and_duplicate_cancellation():
    ctx = context()
    model = ConditionedMatching(ctx)
    fixed = FixedMatching(model)
    cm = pymatching.Matching.from_detector_error_model(ctx.dem, enable_correlations=True)
    for row in ctx.circuit.compile_detector_sampler(seed=52193).sample(24):
        edges = cm.decode_to_edges_array(row, enable_correlations=True)
        pairs = np.concatenate((edges, edges[:2], edges[:2]))
        expected = np.zeros(len(model.memory_keys) + len(model.other_keys), np.uint8)
        for a, b in pairs:
            ds = (int(a),) if b < 0 else tuple(sorted((int(a), int(b))))
            if int(a) in model.memory:
                expected[model.edge_lookup[ds]] ^= 1
            else:
                expected[len(model.memory_keys) + model.other_keys[ds]] ^= 1
        np.testing.assert_array_equal(fixed.events(pairs), expected)


def test_reverse_seed_config_rejects_non_boolean():
    with pytest.raises(ValueError):
        DecoderConfig(reverse_seed=1)


@pytest.mark.parametrize("reverse_seed", [False, True])
def test_search_seeds_preserve_shot_independence(reverse_seed):
    ctx = context()
    decoder = TensorDecoder(ctx, DecoderConfig(reverse_seed=reverse_seed))
    decoder.control.path_threshold = np.inf
    events = ctx.circuit.compile_detector_sampler(seed=15481).sample(12, bit_packed=True)
    events.flags.writeable = False
    expected, info = decoder.decode_with_diagnostics(events)
    np.testing.assert_array_equal(decoder.decode_batch(events[::-1].copy()), expected[::-1])
    for i in (0, 7, 11):
        actual, details = decoder.decode_with_diagnostics(events[i : i + 1])
        np.testing.assert_array_equal(actual, expected[i : i + 1])
        np.testing.assert_array_equal(
            details["competition_margin"], info["competition_margin"][i : i + 1]
        )

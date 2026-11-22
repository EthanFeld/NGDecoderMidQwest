import itertools
from types import SimpleNamespace

import numpy as np
import pymatching
import pytest
from scipy.sparse import csc_matrix

from tensor_decoder.matching import FixedMatching, audit_vendor


@pytest.fixture
def cpp_backend():
    return None


def test_pinned_upstream_sources():
    assert audit_vendor() == 37


def test_fixed_solver_matches_exhaustive_objective_and_pymatching(cpp_backend):
    matrix = csc_matrix(np.array([[1, 0, 1], [0, 1, 1]], np.uint8))
    observable = csc_matrix(np.array([[1, 0, 0]], np.uint8))
    model = SimpleNamespace(
        check_matrix=matrix,
        indices=np.arange(2),
        memory_keys=(((0,), True), ((1,), False), ((0, 1), False)),
        other_keys={},
    )
    fixed = FixedMatching(model)
    fixed.prepare(np.empty((0, 2), np.int64))
    rng = np.random.default_rng(33491)
    weights_list = [np.zeros(3), np.ones(3), -np.ones(3), *rng.normal(0, 5, (50, 3))]
    for weights in weights_list:
        reference = pymatching.Matching.from_check_matrix(
            matrix, weights=weights, faults_matrix=observable, use_virtual_boundary_node=True
        )
        for syndrome in itertools.product((0, 1), repeat=2):
            syndrome = np.array(syndrome, np.uint8)
            expected_label, expected_cost = reference.decode(syndrome, return_weight=True)
            label, gain = fixed.solve(syndrome, weights)
            candidates = []
            for flags in itertools.product((0, 1), repeat=3):
                flags = np.array(flags, np.uint8)
                if np.array_equal(matrix @ flags % 2, syndrome):
                    candidates.append(float(weights @ flags))
            assert -gain == pytest.approx(min(candidates), abs=5e-6)
            assert -gain == pytest.approx(expected_cost, abs=1e-10)
            np.testing.assert_array_equal(label, expected_label)
    fixed.close()
    fixed.close()

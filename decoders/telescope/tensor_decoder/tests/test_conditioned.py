import itertools

import numpy as np
import pytest
import stim

from tensor_decoder import DecoderContext, Experiment
from tensor_decoder.conditioned import ConditionedMatching
from tensor_decoder.model import FaultModel


def tiny_conditioned(monkeypatch):
    import tensor_decoder.conditioned as module

    dem = stim.DetectorErrorModel(
        "error(0.10) D0 L0 ^ D2\nerror(0.04) D1 ^ D2\n"
        "error(0.03) D0 L0 ^ D3\nerror(0.12) D1 ^ D3\n"
        "error(0.05) D0 L0\nerror(0.06) D2\nerror(0.02) D0 D1 ^ D2 D3"
    )
    c = stim.Circuit("M 0\n" + "DETECTOR rec[-1]\n" * 4 + "OBSERVABLE_INCLUDE(0)")
    context = DecoderContext(Experiment(3, 1, 0.01), c, dem)
    monkeypatch.setattr(module, "memory_detector_indices", lambda ctx: np.array([0, 1]))
    return context, ConditionedMatching(context)


@pytest.mark.parametrize("target_other", [False, True])
def test_conditioned_tensor_marginals_match_full_fault_enumeration(monkeypatch, target_other):
    context, conditional = tiny_conditioned(monkeypatch)
    if target_other:
        conditional = ConditionedMatching(context, target_other=True)
    model = FaultModel.from_context(context)
    joint = np.zeros((1 << len(conditional.memory_keys), 1 << len(conditional.other_keys)))
    for flags in itertools.product((0, 1), repeat=len(model.probabilities)):
        probability, a_mask, b_mask = 1.0, 0, 0
        for flag, p, ds, _ls in zip(
            flags, model.probabilities, model.detectors, model.observables, strict=True
        ):
            probability *= p if flag else 1 - p
            if flag:
                md = tuple(d for d in ds if d in conditional.memory)
                od = tuple(d for d in ds if d not in conditional.memory)
                if md:
                    a_mask ^= 1 << conditional.edge_lookup[md]
                if od:
                    b_mask ^= 1 << conditional.other_keys[od]
        joint[a_mask, b_mask] += probability
    ordered_other = tuple(conditional.other_keys)
    for b_mask in range(joint.shape[1]):
        edges = np.asarray(
            [
                (ds[0], ds[1] if len(ds) == 2 else -1)
                for i, ds in enumerate(ordered_other)
                if b_mask >> i & 1
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        weights = conditional.weights(edges)
        actual = 1 / (1 + np.exp(weights))
        expected = [
            sum(joint[a, b_mask] for a in range(joint.shape[0]) if a >> i & 1)
            / joint[:, b_mask].sum()
            for i in range(len(conditional.memory_keys))
        ]
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=1e-12)
        np.testing.assert_array_equal(conditional.weights(edges, 0), conditional.prior_weights)


def test_conditional_gain_matches_independent_minimum_weight_search(monkeypatch):
    _, conditional = tiny_conditioned(monkeypatch)
    syndrome = np.array([1, 0, 0, 1], dtype=np.uint8)
    edges = np.array([[0, -1], [2, -1], [2, 3]])
    weights = conditional.weights(edges)
    candidates = []
    for pattern in itertools.product((0, 1), repeat=len(weights)):
        flags = np.asarray(pattern, dtype=np.uint8)
        if np.array_equal(np.asarray(conditional.check_matrix @ flags).ravel() % 2, syndrome[:2]):
            label = int(np.asarray(conditional.observable_matrix @ flags).ravel()[0] % 2)
            candidates.append((float(weights @ flags), label))
    best = min(weight for weight, _ in candidates)
    prediction, gain = conditional.decode_with_gain(syndrome, edges)
    assert any(
        abs(weight - best) < 1e-10 and label == prediction[0] for weight, label in candidates
    )
    baseline = weights[conditional.edge_lookup[(0,)]]
    # Matching's returned weight uses finite internal weight resolution.
    assert gain == pytest.approx(baseline - best, abs=1e-5)


def test_conditioned_sector_costs_match_exhaustive_both_logical_classes(monkeypatch):
    _, conditional = tiny_conditioned(monkeypatch)
    for other_mask in range(1 << len(conditional.other_keys)):
        edges = np.asarray(
            [
                (ds[0], ds[1] if len(ds) == 2 else -1)
                for i, ds in enumerate(conditional.other_keys)
                if other_mask >> i & 1
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        weights = conditional.weights(edges)
        for observed in itertools.product((0, 1), repeat=2):
            syndrome = np.asarray([*observed, 0, 0], dtype=np.uint8)
            expected = [float("inf"), float("inf")]
            for flags in itertools.product((0, 1), repeat=len(weights)):
                flags = np.asarray(flags, dtype=np.uint8)
                actual = np.asarray(conditional.check_matrix @ flags).ravel() % 2
                if np.array_equal(actual, observed):
                    label = int(np.asarray(conditional.observable_matrix @ flags).ravel()[0] % 2)
                    expected[label] = min(expected[label], float(weights @ flags))
            np.testing.assert_allclose(
                conditional.sector_costs(syndrome, edges), expected, atol=2e-5
            )
            # The sector comparison does not depend on a particular memory path.
            augmented = np.concatenate((edges, np.array([[0, -1], [1, -1]])))
            np.testing.assert_array_equal(
                conditional.sector_costs(syndrome, edges),
                conditional.sector_costs(syndrome, augmented),
            )

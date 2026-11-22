import numpy as np
import pytest

from morph31.cost_model import Operations
from morph31.decoder import SparseDecoder
from morph31.graph import correction_syndrome
from morph31.motif import (
    CONFIDENCE_LEVELS,
    NEIGHBORS12,
    MotifPolicy,
    confidence_code,
    signature,
)
from morph31.training import train_policy


def test_training_and_rom_export(models, tmp_path):
    circuit, dem, _, graph = models(5)
    training = circuit.compile_detector_sampler(seed=3101).sample(128)
    first = train_policy(graph, dem, training, min_support=2)
    second = train_policy(graph, dem, training, min_support=2)
    assert np.array_equal(first.table, second.table)
    assert first.table.nbytes == 4096
    assert np.count_nonzero(first.table)
    path = tmp_path / "policy.rom"
    first.save(path)
    assert path.stat().st_size == 4096
    restored = MotifPolicy.load(path)
    assert np.array_equal(first.table, restored.table)
    assert len(path.with_suffix(".hex").read_text().splitlines()) == 4096
    heldout = circuit.compile_detector_sampler(seed=3102).sample(50)
    for syndrome in heldout:
        result = SparseDecoder(graph, policy=restored).decode(syndrome)
        actual, mask = correction_syndrome(result.correction, graph.num_detectors)
        assert np.array_equal(actual, syndrome)
        assert result.logical_mask == mask
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        MotifPolicy.load(path)


def test_adversarial_policy_guards_and_fallback(models):
    _, _, _, graph = models()
    syndrome = np.ones(graph.num_detectors, dtype=np.uint8)
    # Invalid action 14 must abstain because graph degree < 15.
    invalid = MotifPolicy(np.full(4096, 255, dtype=np.uint8), {})
    residual, edges, count = invalid.predecode(graph, syndrome, Operations())
    assert np.array_equal(residual, syndrome) and edges == () and count == 0
    # An aggressively wrong policy still produces a legal complete correction.
    aggressive = MotifPolicy(np.full(4096, 241, dtype=np.uint8), {})
    result = SparseDecoder(graph, policy=aggressive).decode(syndrome)
    actual, mask = correction_syndrome(result.correction, graph.num_detectors)
    assert np.array_equal(actual, syndrome)
    assert result.logical_mask == mask
    residual, pre, _ = aggressive.predecode(graph, syndrome, Operations())
    endpoints = [v for edge in pre for v in (edge.u, edge.v) if v >= 0]
    assert len(endpoints) == len(set(endpoints))
    assert set(np.unique(residual)) <= {0, 1}


def test_features_and_confidence_abstention(models):
    _, _, _, graph = models()
    syndrome = np.ones(graph.num_detectors, dtype=np.uint8)
    assert all(0 <= signature(graph, v, syndrome) < 4096
               for v in range(graph.num_detectors))
    policy = MotifPolicy(np.full(4096, 17, dtype=np.uint8), {}, min_confidence=0.95)
    assert policy.predecode(graph, syndrome, Operations())[2] == 0
    with pytest.raises(ValueError):
        MotifPolicy(np.zeros(10, dtype=np.uint8), {})


def test_full_neighborhood_features_confidence_and_export(models, tmp_path):
    circuit, dem, _, graph = models(5)
    events = circuit.compile_detector_sampler(seed=121).sample(128)
    policy = train_policy(graph, dem, events, feature_version=NEIGHBORS12,
                          min_support=2, min_confidence=.95)
    path = tmp_path / "v2.rom"
    policy.save(path)
    restored = MotifPolicy.load(path)
    assert restored.feature_version == NEIGHBORS12
    assert restored.metadata["confidence_levels"] == list(CONFIDENCE_LEVELS)
    for confidence in (.95, .97, .99, .999, 1):
        code = confidence_code(confidence, NEIGHBORS12)
        assert CONFIDENCE_LEVELS[code] <= confidence
    assert confidence_code(.991, NEIGHBORS12) == 11
    for syndrome in events[:10]:
        for vertex in np.flatnonzero(syndrome):
            neighbors = graph.neighbors(int(vertex))
            key = signature(graph, int(vertex), syndrome, neighbors, feature_version=NEIGHBORS12)
            assert 0 <= key < 4096
            for bit, neighbor in enumerate(neighbors):
                assert ((key >> bit) & 1) == (
                    neighbor.vertex == -1 or syndrome[neighbor.vertex])
        result = SparseDecoder(graph, policy=restored).decode(syndrome)
        assert np.array_equal(correction_syndrome(result.correction, graph.num_detectors)[0],
                              syndrome)


def test_v2_quantized_threshold_and_excess_degree_abstention(models):
    _, _, _, graph = models()
    syndrome = np.ones(graph.num_detectors, dtype=np.uint8)
    metadata = {"feature_version": NEIGHBORS12}
    high = MotifPolicy(np.full(4096, (11 << 4) | 1, dtype=np.uint8), metadata, .99)
    low = MotifPolicy(np.full(4096, (10 << 4) | 1, dtype=np.uint8), metadata, .99)
    assert high.predecode(graph, syndrome, Operations())[2] > 0
    assert low.predecode(graph, syndrome, Operations())[2] == 0
    neighbors = graph.neighbors(0)
    assert signature(graph, 0, syndrome, neighbors * 13, feature_version=NEIGHBORS12) is None

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from morph31.compression import CompressedGraph, compile_graph, verify_equivalence
from morph31.geometry import LatticeCodec
from morph31.graph import Edge, ExplicitGraph


@pytest.mark.parametrize("d,rounds,basis", [(3, 1, "x"), (3, 2, "z"), (5, 5, "x"),
                                          (5, 5, "z"), (7, 3, "x"), (11, 11, "z")])
def test_exact_graph_and_binary_roundtrip(models, d, rounds, basis):
    _, _, reference, compressed = models(d, rounds, basis)
    assert verify_equivalence(reference, compressed) == len(reference.edges)
    restored = CompressedGraph.from_bytes(compressed.to_bytes())
    assert verify_equivalence(reference, restored) == len(reference.edges)
    for v, coordinates in enumerate(reference.coordinates):
        x, y, t = compressed.codec.coordinate(v)
        assert tuple(coordinates) == (x * 2, y * 2, t)
        assert compressed.codec.detector(x, y, t) == v
        view = compressed.neighborhood(v)
        neighbors = compressed.neighbors(v)
        assert [view.endpoint(i) for i in range(len(view))] == [n.vertex for n in neighbors]
        assert [view.neighbor(i) for i in range(len(view))] == neighbors
        assert compressed.neighbors_with_cost(v) == (neighbors, compressed.neighbor_cost(v))


def test_lazy_neighbor_action_bounds(models):
    _, _, _, graph = models()
    view = graph.neighborhood(0)
    with pytest.raises(IndexError):
        view.neighbor(-1)
    with pytest.raises(IndexError):
        view.endpoint(len(view))


def test_parallel_logical_labels_and_multiplicity_survive(models):
    _, _, graph, compressed = models()
    edge = next(e for e in graph.edges if e.u >= 0)
    parallel = replace(edge, logical_mask=1 ^ edge.logical_mask, multiplicity=7)
    reference = ExplicitGraph(graph.num_detectors, 1, graph.coordinates, [*graph.edges, parallel])
    result = compile_graph(reference, compressed.codec)
    assert verify_equivalence(reference, result) == len(reference.edges)
    assert Counter(result.iter_edges())[parallel] == 1


def test_malformed_rom_and_ordering_rejected(models):
    _, _, graph, compressed = models()
    with pytest.raises(ValueError, match="truncated"):
        CompressedGraph.from_bytes(compressed.to_bytes()[:-1])
    with pytest.raises(ValueError, match="unsupported"):
        CompressedGraph.from_bytes(b"badmagic" + compressed.to_bytes()[8:])
    graph = ExplicitGraph(graph.num_detectors, 1, graph.coordinates[::-1], graph.edges)
    with pytest.raises(ValueError, match="ordering"):
        compile_graph(graph, compressed.codec)


def test_invisible_logical_is_retained_and_not_silently_exported(models):
    _, _, graph, compressed = models()
    invisible = Edge(-1, -1, 0.01, 1)
    reference = ExplicitGraph(graph.num_detectors, 1, graph.coordinates,
                              [*graph.edges, invisible])
    result = compile_graph(reference, compressed.codec)
    assert verify_equivalence(reference, result) == len(reference.edges)
    with pytest.raises(ValueError, match="undetectable"):
        result.to_bytes()


@pytest.mark.parametrize("basis", ["x", "z"])
def test_target_codec_is_15_bits(models, basis):
    codec = LatticeCodec(31, 31, basis)
    addresses = set()
    for detector in range(codec.num_detectors):
        xyz = codec.coordinate(detector)
        assert codec.detector(*xyz) == detector
        address = codec.pack(*xyz)
        assert 0 <= address < 32768
        assert codec.unpack(address) == xyz
        addresses.add(address)
    assert len(addresses) == 29760


@pytest.mark.target
@pytest.mark.parametrize("basis", ["x", "z"])
def test_distance31_exact_graph(models, basis):
    _, _, reference, compressed = models(31, 31, basis)
    assert reference.num_detectors == 29760
    assert verify_equivalence(reference, compressed) == len(reference.edges)
    restored = CompressedGraph.from_bytes(compressed.to_bytes())
    assert verify_equivalence(reference, restored) == len(reference.edges)
    assert len(restored.to_bytes()) <= 8192
    assert restored.statistics(reference)["weight_quantization_error"] == 0


def test_zero_noise_graph(models):
    _, _, reference, graph = models(noise=0)
    assert not reference.edges
    assert verify_equivalence(reference, CompressedGraph.from_bytes(graph.to_bytes())) == 0
    assert np.array_equal(graph.probabilities, [])

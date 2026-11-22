from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pymatching
import pytest

from morph31.decoder import DecodeFailure, SparseDecoder
from morph31.graph import correction_syndrome
from morph31.integration import build_decoder
from morph31.sparse_memory import Limits, MemoryOverflow


@pytest.mark.parametrize("distance,basis", [(3, "x"), (3, "z"), (5, "x"), (5, "z")])
def test_no_error_and_every_single_graph_fault(models, distance, basis):
    _, _, reference, graph = models(distance, basis=basis)
    decoder = SparseDecoder(graph)
    zero = np.zeros(graph.num_detectors, dtype=np.uint8)
    assert decoder.decode(zero).logical_mask == 0
    boundary = temporal = logical = False
    for edge in reference.edges:
        syndrome, mask = correction_syndrome([edge], graph.num_detectors)
        result = decoder.decode(syndrome)
        actual, result_mask = correction_syndrome(result.correction, graph.num_detectors)
        assert np.array_equal(actual, syndrome)
        assert result.logical_mask == result_mask == mask
        boundary |= edge.u == -1
        logical |= bool(mask)
        if edge.u >= 0:
            temporal |= reference.coordinates[edge.u, 2] != reference.coordinates[edge.v, 2]
    assert boundary and temporal and logical


@pytest.mark.parametrize("basis", ["x", "z"])
def test_full_circuit_samples_and_batch_independence(models, basis):
    circuit, dem, _, graph = models(5, basis=basis, noise=0.01)
    events, truth = circuit.compile_detector_sampler(seed=3101).sample(
        100, separate_observables=True)
    decoder = SparseDecoder(graph)
    prediction = decoder.decode_batch(events)
    assert np.array_equal(prediction[:8], decoder.decode_batch(events[:8]))
    reference = pymatching.Matching.from_detector_error_model(dem).decode_batch(events)
    assert reference.shape == prediction.shape == truth.shape
    # Logical equivalence is not imposed for multiple faults; syndrome validity is.
    for syndrome in events[:20]:
        result = decoder.decode(syndrome)
        assert np.array_equal(correction_syndrome(result.correction, graph.num_detectors)[0],
                              syndrome)


def test_dense_arbitrary_syndromes_and_logical_chain(models):
    _, _, reference, graph = models()
    decoder = SparseDecoder(graph)
    random = np.random.default_rng(11)
    for syndrome in random.integers(0, 2, (100, graph.num_detectors), dtype=np.uint8):
        result = decoder.decode(syndrome)
        assert np.array_equal(correction_syndrome(result.correction, graph.num_detectors)[0],
                              syndrome)
    # Find a logical boundary-to-boundary chain by lifting paths into parity space.
    todo = [(-1, 0, ())]
    seen = {(-1, 0)}
    adjacency = {-1: [e for e in reference.edges if e.u == -1]}
    for vertex in range(graph.num_detectors):
        adjacency[vertex] = [n.edge for n in graph.neighbors(vertex)]
    while todo:
        vertex, mask, path = todo.pop(0)
        for edge in adjacency[vertex]:
            other = edge.v if vertex == edge.u else edge.u
            state = other, mask ^ edge.logical_mask
            if state in seen:
                continue
            seen.add(state)
            chain = (*path, edge)
            if state == (-1, 1):
                syndrome, parity = correction_syndrome(chain, graph.num_detectors)
                assert not np.any(syndrome) and parity == 1
                assert decoder.decode(syndrome).logical_mask == 0
                return
            todo.append((*state, chain))
    pytest.fail("no logical chain found")


def test_bounds_and_inconsistent_components_fail_visibly(models):
    _, _, _, graph = models()
    syndrome = np.ones(graph.num_detectors, dtype=np.uint8)
    with pytest.raises(MemoryOverflow):
        SparseDecoder(graph, Limits(working_bytes=1)).decode(syndrome)
    with pytest.raises(MemoryOverflow, match="clusters"):
        SparseDecoder(graph, Limits(max_clusters=1)).decode(syndrome)
    with pytest.raises(DecodeFailure, match="limit"):
        SparseDecoder(graph, Limits(max_expansions=1)).decode(syndrome)
    with pytest.raises(ValueError, match="binary"):
        SparseDecoder(graph).decode(syndrome * 2)
    _, _, _, empty = models(noise=0)
    with pytest.raises(DecodeFailure, match="exhausted"):
        SparseDecoder(empty).decode(syndrome)


def test_frozen_harness_contract(models):
    circuit, dem, _, graph = models(5, basis="z")
    events = circuit.compile_detector_sampler(seed=4).sample(9).astype(np.uint8)
    packed = np.packbits(events, axis=1, bitorder="little")
    packed.flags.writeable = False
    context = SimpleNamespace(dem=dem, experiment=SimpleNamespace(
        geometry="rotated", distance=5, rounds=5, basis="z"))
    decoder = build_decoder(context)
    output = decoder.decode_batch(packed)
    expected = SparseDecoder(graph).decode_batch(events)
    assert output.dtype == np.uint8 and output.flags.c_contiguous
    assert np.array_equal(np.unpackbits(output, axis=1, count=1, bitorder="little"), expected)
    assert not np.any(output & 254)


def test_uniform_weight_label_is_not_mwpm(models):
    _, _, _, graph = models()
    result = SparseDecoder(graph).decode(np.zeros(graph.num_detectors))
    assert result.operations.neighbor_requests == 0
    assert result.memory["peak_working_bytes"] > 0
    with pytest.raises(ValueError):
        Limits(max_seconds=0)
    assert replace(Limits(), working_bytes=1024).working_bytes == 1024

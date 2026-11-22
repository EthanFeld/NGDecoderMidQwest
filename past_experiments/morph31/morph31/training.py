"""Small supervised decision counts; one final ROM, no epochs/checkpoints."""

import time
from dataclasses import asdict
from hashlib import sha256

import numpy as np
import pymatching

from .motif import FEATURE_VERSION, MotifPolicy, confidence_code, signature


def train_policy(graph, dem, syndromes, *, min_support=3, min_confidence=0.95,
                 metadata=None, max_seconds=30.0, feature_version=FEATURE_VERSION):
    if min_support < 1:
        raise ValueError("minimum support must be positive")
    syndromes = np.asarray(syndromes)
    if (syndromes.ndim != 2 or syndromes.shape[1] != graph.num_detectors
            or np.any((syndromes != 0) & (syndromes != 1))):
        raise ValueError("training syndromes must be a binary shots by detectors matrix")
    start = time.perf_counter()
    teacher = pymatching.Matching.from_detector_error_model(dem)
    counts = np.zeros((4096, 16), dtype=np.uint32)
    exposures = 0
    for syndrome in syndromes:
        if time.perf_counter() - start > max_seconds:
            raise TimeoutError("bounded motif training exceeded its time budget")
        chosen = {}
        ambiguous = set()
        for a, b in teacher.decode_to_edges_array(syndrome):
            a, b = int(a), int(b)
            data = (teacher.get_boundary_edge_data(a) if b == -1
                    else teacher.get_edge_data(a, b))
            mask = sum(1 << int(bit) for bit in data["fault_ids"])
            for vertex, other in ((a, b), (b, a)):
                if vertex == -1:
                    continue
                if vertex in chosen:
                    ambiguous.add(vertex)
                chosen[vertex] = (other, mask)
        for vertex in map(int, np.flatnonzero(syndrome)):
            neighbors = graph.neighbors(vertex)
            key = signature(graph, vertex, syndrome, neighbors, feature_version=feature_version)
            if key is None:
                continue
            action = 0
            if vertex in chosen and vertex not in ambiguous:
                other, mask = chosen[vertex]
                if other == -1 or syndrome[other]:
                    legal = [index + 1 for index, n in enumerate(neighbors)
                             if n.vertex == other and n.edge.logical_mask == mask]
                    if len(legal) == 1 and legal[0] < 16:
                        action = legal[0]
            counts[key, action] += 1
            exposures += 1
    table = np.zeros(4096, dtype=np.uint8)
    for key, row in enumerate(counts):
        support = int(row.sum())
        if support < min_support:
            continue
        action = int(np.argmax(row))
        confidence = int(row[action]) / support
        if action and confidence >= min_confidence:
            table[key] = (confidence_code(confidence, feature_version) << 4) | action
    info = {**(metadata or {}), "feature_version": feature_version,
            "teacher": "PyMatching from decomposed DEM (uncorrelated mode)",
            "pymatching": pymatching.__version__, "shots": len(syndromes),
            "training_data_sha256": sha256(np.packbits(syndromes).tobytes()).hexdigest(),
            "observed_signatures": int(np.count_nonzero(counts.sum(axis=1))),
            "active_rom_entries": int(np.count_nonzero(table & 15)),
            "exposures": exposures, "min_support": min_support,
            "training_seconds": time.perf_counter() - start}
    return MotifPolicy(table, info, min_confidence)


def training_metadata(experiment, seed):
    return {"experiment": asdict(experiment), "seed": seed}

"""A 4096-byte weightless local policy with legal-edge and conflict guards."""

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from .graph import BOUNDARY

FEATURE_VERSION = "m31-local8-boundary2-family1-time1-v1"
NEIGHBORS12 = "m31-neighbors12-v2"
CONFIDENCE_LEVELS = (0, .5, .75, .875, .9, .925, .95, .96,
                     .97, .98, .985, .99, .995, .997, .999, 1)


def neighbor_signature(endpoints, syndrome):
    if len(endpoints) > 12:
        return None
    return sum(int(other == BOUNDARY or syndrome[other]) << bit
               for bit, other in enumerate(endpoints))


def signature(graph, vertex, syndrome, neighbors=None, *, feature_version=FEATURE_VERSION):
    neighbors = graph.neighbors(vertex) if neighbors is None else neighbors
    if feature_version == NEIGHBORS12:
        return neighbor_signature([n.vertex for n in neighbors], syndrome)
    value, bit = 0, 0
    for neighbor in neighbors:
        if neighbor.vertex != BOUNDARY and bit < 8:
            value |= int(syndrome[neighbor.vertex]) << bit
            bit += 1
    x, y, t = graph.codec.coordinate(vertex)
    d = graph.codec.distance
    value |= int(x in (0, d) or y in (0, d)) << 8
    value |= int(any(n.vertex == BOUNDARY for n in neighbors)) << 9
    value |= ((x + y) & 1) << 10
    value |= int(t in (0, graph.codec.rounds)) << 11
    return value


def confidence_code(confidence, version):
    if version == NEIGHBORS12:
        return int(np.searchsorted(CONFIDENCE_LEVELS, confidence, side="right") - 1)
    return int(confidence * 15)


@dataclass
class MotifPolicy:
    table: np.ndarray
    metadata: dict
    min_confidence: float = 0.95

    def __post_init__(self):
        self.table = np.asarray(self.table)
        if self.table.shape != (4096,) or self.table.dtype != np.uint8:
            raise ValueError("motif ROM must be exactly 4096 uint8 entries")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("confidence threshold must be in [0, 1]")
        if self.metadata.get("feature_version", FEATURE_VERSION) not in (
            FEATURE_VERSION, NEIGHBORS12
        ):
            raise ValueError("incompatible motif feature encoding")

    @property
    def feature_version(self):
        return self.metadata.get("feature_version", FEATURE_VERSION)

    def predecode(self, graph, syndrome, ops):
        residual = syndrome.copy()
        correction = []
        version = self.feature_version
        threshold = (int(np.searchsorted(CONFIDENCE_LEVELS, self.min_confidence))
                     if version == NEIGHBORS12 else math.ceil(15 * self.min_confidence))
        # Features use the original syndrome, so policy inference is independent
        # of preceding acceptances; the current residual enforces conflicts.
        for vertex in map(int, np.flatnonzero(syndrome)):
            if not residual[vertex]:
                continue
            if version == NEIGHBORS12:
                view = graph.neighborhood(vertex)
                endpoints = [view.endpoint(i) for i in range(len(view))]
                count, generation_cost = len(view), view.cost
                key = neighbor_signature(endpoints, syndrome)
            else:
                neighbors, generation_cost = graph.neighbors_with_cost(vertex)
                count = len(neighbors)
                key = signature(graph, vertex, syndrome, neighbors)
            ops.neighbor_requests += 1
            ops.neighbor_generation += generation_cost
            feature_bits = 12 if version == NEIGHBORS12 else 8
            ops.memory_reads += 6 + count + min(feature_bits, count)
            if key is None:
                continue
            entry = int(self.table[key])
            ops.ml_lookups += 1
            action, confidence = (entry & 15) - 1, entry >> 4
            if action < 0 or action >= count or confidence < threshold:
                continue
            other = endpoints[action] if version == NEIGHBORS12 else neighbors[action].vertex
            # The action is resolved against the REAL current graph; invalid
            # indices abstain. Pairs require two still-active endpoints.
            if other != BOUNDARY and not residual[other]:
                continue
            if version == NEIGHBORS12:
                candidate = view.neighbor(action)
                # Reread the chosen descriptor and logical/probability fields;
                # this work is charged in addition to endpoint-only probing.
                ops.neighbor_generation += 1
                ops.memory_reads += 2
            else:
                candidate = neighbors[action]
            residual[vertex] ^= 1
            if candidate.vertex != BOUNDARY:
                residual[candidate.vertex] ^= 1
            correction.append(candidate.edge)
            ops.edge_examinations += 1
            ops.memory_writes += 2
        return residual, tuple(correction), len(correction)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = self.table.tobytes()
        path.write_bytes(raw)
        encoding = ("high nibble indexes confidence_levels; low nibble action+1; zero abstains"
                    if self.feature_version == NEIGHBORS12 else
                    "high nibble confidence/15; low nibble action+1; zero low nibble abstains")
        metadata = {**self.metadata, "feature_version": self.feature_version,
                    "min_confidence": self.min_confidence, "sha256": sha256(raw).hexdigest(),
                    "rom_bytes": 4096, "entry": encoding}
        if self.feature_version == NEIGHBORS12:
            metadata["confidence_levels"] = CONFIDENCE_LEVELS
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
        # One byte per line, suitable for a future ROM initialization flow.
        path.with_suffix(".hex").write_text("".join(f"{value:02x}\n" for value in self.table))

    @classmethod
    def load(cls, path):
        path = Path(path)
        raw = path.read_bytes()
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if sha256(raw).hexdigest() != metadata["sha256"]:
            raise ValueError("motif ROM checksum mismatch")
        return cls(np.frombuffer(raw, dtype=np.uint8).copy(), metadata,
                   metadata["min_confidence"])

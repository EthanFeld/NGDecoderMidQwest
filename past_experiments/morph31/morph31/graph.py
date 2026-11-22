"""Explicit independent graphlike projection; parallel logical labels stay distinct."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOUNDARY = -1


@dataclass(frozen=True, order=True)
class Edge:
    u: int
    v: int
    probability: float
    logical_mask: int = 0
    multiplicity: int = 1

    @property
    def weight(self):
        if self.probability == 0:
            return math.inf
        if self.probability == 1:
            return -math.inf
        return math.log1p(-self.probability) - math.log(self.probability)

    @property
    def key(self):
        return self.u, self.v, self.logical_mask


@dataclass(frozen=True)
class Neighbor:
    vertex: int
    edge: Edge


class ExplicitGraph:
    def __init__(self, num_detectors, num_observables, coordinates, edges, metadata=None):
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        self.edges = tuple(sorted(edges, key=lambda e: e.key))
        self.metadata = metadata or {}
        self._adjacency = [[] for _ in range(num_detectors)]
        seen = set()
        for edge in self.edges:
            if edge.key in seen:
                raise ValueError("identical edge keys must be explicitly aggregated")
            seen.add(edge.key)
            if not -1 <= edge.u < num_detectors or not -1 <= edge.v < num_detectors:
                raise ValueError("edge endpoint outside graph")
            if edge.u == edge.v and edge.u != BOUNDARY:
                raise ValueError("detector self-loops must be reduced over GF(2)")
            if not np.isfinite(edge.probability) or not 0 <= edge.probability <= 1:
                raise ValueError("invalid probability")
            if not 0 <= edge.logical_mask < 2**num_observables:
                raise ValueError("logical label outside graph observables")
            if edge.u != BOUNDARY:
                self._adjacency[edge.u].append(Neighbor(edge.v, edge))
            if edge.v != BOUNDARY:
                self._adjacency[edge.v].append(Neighbor(edge.u, edge))

    def neighbors(self, vertex):
        if not 0 <= vertex < self.num_detectors:
            raise ValueError("invalid detector")
        return self._adjacency[vertex]

    def iter_edges(self):
        return iter(self.edges)

    @property
    def packed_bytes(self):
        # Canonical explicit edge array + CSR incidence + float64 coordinates.
        # This is an analytical packed layout, not Python object heap usage.
        incidences = sum(len(a) for a in self._adjacency)
        return len(self.edges) * 15 + incidences * 4 + (self.num_detectors + 1) * 4 + (
            self.coordinates.nbytes
        )

    def save(self, path: str | Path):
        import json

        np.savez_compressed(
            path,
            counts=np.array([self.num_detectors, self.num_observables], dtype=np.int64),
            coordinates=self.coordinates,
            endpoints=np.array([(e.u, e.v) for e in self.edges], dtype=np.int32).reshape(-1, 2),
            probabilities=np.array([e.probability for e in self.edges], dtype=np.float64),
            masks=np.array([e.logical_mask for e in self.edges], dtype=np.uint64),
            multiplicities=np.array([e.multiplicity for e in self.edges], dtype=np.uint32),
            metadata=np.array(json.dumps(self.metadata)),
        )

    @classmethod
    def load(cls, path):
        import json

        with np.load(path, allow_pickle=False) as data:
            return cls(
                *map(int, data["counts"]), data["coordinates"],
                [Edge(int(u), int(v), float(p), int(m), int(n)) for (u, v), p, m, n in zip(
                    data["endpoints"], data["probabilities"], data["masks"],
                    data["multiplicities"], strict=True
                )], json.loads(str(data["metadata"]))
            )


def correction_syndrome(edges, num_detectors):
    syndrome = np.zeros(num_detectors, dtype=np.uint8)
    mask = 0
    for edge in edges:
        if edge.u != BOUNDARY:
            syndrome[edge.u] ^= 1
        if edge.v != BOUNDARY:
            syndrome[edge.v] ^= 1
        mask ^= edge.logical_mask
    return syndrome, mask

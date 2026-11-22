"""Exact neighborhood templates and shared parity-striped interval maps.

The runtime stores no full adjacency list or per-detector template map. A time
map selects a layer; two tiny interval maps classify y and x, splitting parity
before run-length encoding. Translation uses 15-bit lattice addresses.
"""

import struct
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import LatticeCodec
from .graph import BOUNDARY, Edge, Neighbor

_BOUNDARY_DELTA = -32768
_DESCRIPTOR = np.dtype([("delta", "<i2"), ("prob", "<u2"), ("mask", "u1"), ("n", "<u2")])
_HEADER = struct.Struct("<8sBBBB8I")
_MAGIC = b"M31GR001"


@dataclass(frozen=True, slots=True)
class Neighborhood:
    """One classified ROM range; endpoints need no Edge/weight materialization."""

    graph: object
    vertex: int
    address: int
    start: int
    stop: int
    cost: int

    def __len__(self):
        return self.stop - self.start

    def _descriptor_id(self, index):
        if not 0 <= index < len(self):
            raise IndexError("neighbor action outside template")
        return int(self.graph.templates[self.start + index])

    def endpoint(self, index):
        descriptor_id = self._descriptor_id(index)
        delta = int(self.graph.descriptors["delta"][descriptor_id])
        if delta == _BOUNDARY_DELTA:
            return BOUNDARY
        return self.graph.codec.detector(*self.graph.codec.unpack(self.address + delta))

    def neighbor(self, index):
        other = self.endpoint(index)
        d = self.graph.descriptors[self._descriptor_id(index)]
        u, v = sorted((self.vertex, other))
        edge = Edge(u, v, float(self.graph.probabilities[int(d["prob"])]),
                    int(d["mask"]), int(d["n"]))
        return Neighbor(other, edge)


class CompressedGraph:
    def __init__(self, codec, num_observables, probabilities, descriptors, templates,
                 template_offsets, strips, strip_offsets, rows, layers, times, invisible=()):
        self.codec = codec
        self.num_detectors = codec.num_detectors
        self.num_observables = num_observables
        self.probabilities = np.asarray(probabilities, dtype="<f8")
        self.descriptors = np.asarray(descriptors, dtype=_DESCRIPTOR)
        self.templates = np.asarray(templates, dtype="<u2")
        self.template_offsets = np.asarray(template_offsets, dtype="<u2")
        self.strips = np.asarray(strips, dtype="<u2").reshape(-1, 2)
        self.strip_offsets = np.asarray(strip_offsets, dtype="<u2")
        self.rows = np.asarray(rows, dtype="<u2").reshape(-1, 2)
        self.layers = np.asarray(layers, dtype="<u2").reshape(-1, 2)
        self.times = np.asarray(times, dtype="<u2")
        self.invisible = tuple(invisible)

    def _strip(self, strip, index):
        start, end = map(int, self.strip_offsets[strip:strip + 2])
        # Each entry is (exclusive end, value); maps normally have 1-5 runs.
        position = bisect_right(self.strips[start:end, 0], index)
        return int(self.strips[start + position, 1])

    def template_id(self, x, y, t):
        layer = int(self.times[t])
        row = self._strip(int(self.layers[layer, y & 1]), y >> 1)
        return self._strip(int(self.rows[row, x & 1]), x >> 1)

    def neighborhood(self, vertex):
        x, y, t = self.codec.coordinate(vertex)
        layer = int(self.times[t])
        s = int(self.layers[layer, y & 1])
        row = self._strip(s, y >> 1)
        r = int(self.rows[row, x & 1])
        tid = self._strip(r, x >> 1)
        start, stop = map(int, self.template_offsets[tid:tid + 2])
        comparisons = sum(
            int(self.strip_offsets[i + 1] - self.strip_offsets[i]).bit_length() for i in (s, r)
        )
        return Neighborhood(self, vertex, self.codec.pack(x, y, t), start, stop,
                            5 + comparisons + stop - start)

    def neighbors_with_cost(self, vertex):
        view = self.neighborhood(vertex)
        result = []
        for descriptor_id in self.templates[view.start:view.stop]:
            d = self.descriptors[int(descriptor_id)]
            delta = int(d["delta"])
            other = (BOUNDARY if delta == _BOUNDARY_DELTA else self.codec.detector(
                *self.codec.unpack(view.address + delta)))
            u, v = sorted((vertex, other))
            edge = Edge(u, v, float(self.probabilities[int(d["prob"])]),
                        int(d["mask"]), int(d["n"]))
            result.append(Neighbor(other, edge))
        return result, view.cost

    def neighbor_cost(self, vertex):
        x, y, t = self.codec.coordinate(vertex)
        layer = int(self.times[t])
        s = int(self.layers[layer, y & 1])
        row = self._strip(s, y >> 1)
        r = int(self.rows[row, x & 1])
        # Count interval binary-search comparisons plus the fixed map accesses.
        comparisons = sum(
            int(self.strip_offsets[i + 1] - self.strip_offsets[i]).bit_length() for i in (s, r)
        )
        tid = self._strip(r, x >> 1)
        degree = int(self.template_offsets[tid + 1] - self.template_offsets[tid])
        return 5 + comparisons + degree

    def neighbors(self, vertex):
        x, y, t = self.codec.coordinate(vertex)
        address = self.codec.pack(x, y, t)
        template = self.template_id(x, y, t)
        start, end = map(int, self.template_offsets[template:template + 2])
        result = []
        for descriptor_id in self.templates[start:end]:
            descriptor = self.descriptors[int(descriptor_id)]
            delta = int(descriptor["delta"])
            if delta == _BOUNDARY_DELTA:
                other = BOUNDARY
            else:
                other = self.codec.detector(*self.codec.unpack(address + delta))
            u, v = sorted((vertex, other))
            edge = Edge(u, v, float(self.probabilities[int(descriptor["prob"])]),
                        int(descriptor["mask"]), int(descriptor["n"]))
            result.append(Neighbor(other, edge))
        return result

    def iter_edges(self):
        yield from self.invisible
        for vertex in range(self.num_detectors):
            for neighbor in self.neighbors(vertex):
                if neighbor.vertex == BOUNDARY or vertex < neighbor.vertex:
                    yield neighbor.edge

    def to_bytes(self):
        if self.invisible:
            raise ValueError("ROM export of undetectable logical mechanisms is unsupported")
        counts = [len(self.probabilities), len(self.descriptors), len(self.template_offsets),
                  len(self.templates), len(self.strip_offsets), len(self.strips),
                  len(self.rows), len(self.layers)]
        return _HEADER.pack(_MAGIC, self.codec.distance, self.codec.rounds,
                            self.codec.basis == "x", self.num_observables, *counts) + b"".join(
            a.tobytes() for a in self._arrays()
        )

    def _arrays(self):
        return (self.probabilities, self.descriptors, self.template_offsets, self.templates,
                self.strip_offsets, self.strips, self.rows, self.layers, self.times)

    @classmethod
    def from_bytes(cls, data):
        if len(data) < _HEADER.size:
            raise ValueError("truncated graph ROM")
        magic, d, rounds, basis, observables, *counts = _HEADER.unpack_from(data)
        if magic != _MAGIC or basis not in (0, 1):
            raise ValueError("unsupported graph ROM")
        specs = [("<f8", counts[0]), (_DESCRIPTOR, counts[1]), ("<u2", counts[2]),
                 ("<u2", counts[3]), ("<u2", counts[4]), ("<u2", counts[5] * 2),
                 ("<u2", counts[6] * 2), ("<u2", counts[7] * 2), ("<u2", rounds + 1)]
        offset, arrays = _HEADER.size, []
        for dtype, size in specs:
            nbytes = np.dtype(dtype).itemsize * size
            if offset + nbytes > len(data):
                raise ValueError("truncated graph ROM arrays")
            arrays.append(np.frombuffer(data, dtype=dtype, count=size, offset=offset).copy())
            offset += nbytes
        if offset != len(data):
            raise ValueError("trailing data in graph ROM")
        probabilities, descriptors, toff, templates, soff, strips, rows, layers, times = arrays
        graph = cls(LatticeCodec(d, rounds, "x" if basis else "z"), observables,
                    probabilities, descriptors, templates, toff, strips, soff, rows, layers, times)
        graph._validate_tables()
        return graph

    def _validate_tables(self):
        def offsets_valid(offsets, length):
            return (len(offsets) >= 1 and offsets[0] == 0 and offsets[-1] == length
                    and np.all(offsets[1:] >= offsets[:-1]))
        if not offsets_valid(self.template_offsets, len(self.templates)):
            raise ValueError("invalid template offsets")
        if not offsets_valid(self.strip_offsets, len(self.strips)):
            raise ValueError("invalid interval offsets")
        if len(self.times) != self.codec.rounds + 1:
            raise ValueError("invalid time map")
        for values, bound in [(self.templates, len(self.descriptors)),
                              (self.descriptors["prob"], len(self.probabilities)),
                              (self.times, len(self.layers)),
                              (self.rows, len(self.strip_offsets) - 1),
                              (self.layers, len(self.strip_offsets) - 1)]:
            if values.size and np.max(values) >= bound:
                raise ValueError("graph ROM table reference out of range")
        if not np.all(np.isfinite(self.probabilities)) or np.any(
            (self.probabilities < 0) | (self.probabilities > 1)
        ):
            raise ValueError("invalid graph ROM probabilities")
        for i in range(len(self.strip_offsets) - 1):
            run = self.strips[self.strip_offsets[i]:self.strip_offsets[i + 1]]
            if not len(run) or run[-1, 0] != self.codec.side // 2 or np.any(
                run[1:, 0] <= run[:-1, 0]
            ):
                raise ValueError("invalid graph ROM interval")
        # Validate both map levels, including all references in reachable strips.
        for layer in self.layers:
            for s in layer:
                a, b = self.strip_offsets[int(s):int(s) + 2]
                if np.any(self.strips[a:b, 1] >= len(self.rows)):
                    raise ValueError("invalid row reference")
        for row in self.rows:
            for s in row:
                a, b = self.strip_offsets[int(s):int(s) + 2]
                if np.any(self.strips[a:b, 1] >= len(self.template_offsets) - 1):
                    raise ValueError("invalid template reference")

    def save(self, path):
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path):
        return cls.from_bytes(Path(path).read_bytes())

    def statistics(self, reference=None):
        size = len(self.to_bytes()) if not self.invisible else sum(a.nbytes for a in self._arrays())
        costs = [self.neighbor_cost(v) for v in range(self.num_detectors)]
        result = {
            "graph_rom_bytes": size,
            "templates": len(self.template_offsets) - 1,
            "edge_descriptors": len(self.descriptors),
            "weight_classes": len(self.probabilities),
            "exception_table_bytes": 0,
            "classifier_bytes": sum(a.nbytes for a in (
                self.strips, self.strip_offsets, self.rows, self.layers, self.times)),
            "average_neighbor_generation_operations": float(np.mean(costs)),
            "neighbor_operation_definition": "map accesses + interval comparisons + emitted edges; "
                                             "coordinate arithmetic excluded",
            "weight_quantization_error": 0.0,
            "coordinate_bits": 2 * self.codec.width + self.codec.rounds.bit_length(),
        }
        if reference is not None:
            result.update(explicit_packed_bytes=reference.packed_bytes,
                          compression_ratio=reference.packed_bytes / size,
                          edges=len(reference.edges))
        return result


def _intern(table, value):
    if value not in table:
        table[value] = len(table)
    return table[value]


def compile_graph(reference, codec):
    codec.validate(reference)
    if reference.num_observables > 8:
        raise ValueError("compact graph ROM supports at most 8 observables")
    probabilities = sorted({e.probability for e in reference.edges})
    probability_ids = {p: i for i, p in enumerate(probabilities)}
    descriptors, templates = {}, {(): 0}
    labels = np.zeros((codec.rounds + 1, codec.side, codec.side), dtype=np.uint16)
    for vertex in range(reference.num_detectors):
        address = codec.address(vertex)
        local = []
        for neighbor in reference.neighbors(vertex):
            edge = neighbor.edge
            delta = (_BOUNDARY_DELTA if neighbor.vertex == BOUNDARY
                     else codec.address(neighbor.vertex) - address)
            if not -32768 < delta < 32768 and delta != _BOUNDARY_DELTA:
                raise OverflowError("edge delta exceeds signed 16-bit ROM layout")
            if edge.multiplicity > 65535:
                raise OverflowError("edge multiplicity exceeds ROM layout")
            descriptor = (delta, probability_ids[edge.probability], edge.logical_mask,
                          edge.multiplicity)
            local.append(descriptor)
        # Stable geometry-based action ordering; independent of detector enumeration.
        ids = tuple(_intern(descriptors, item) for item in sorted(local))
        tid = _intern(templates, ids)
        x, y, t = codec.coordinate(vertex)
        labels[t, y, x] = tid

    strips, rows, layers = {}, {}, {}

    def stripe(sequence):
        runs = []
        for index, value in enumerate(sequence):
            value = int(value)
            if runs and runs[-1][1] == value:
                runs[-1] = (index + 1, value)
            else:
                runs.append((index + 1, value))
        return _intern(strips, tuple(runs))

    times = []
    for layer in labels:
        row_ids = [_intern(rows, (stripe(row[::2]), stripe(row[1::2]))) for row in layer]
        times.append(_intern(layers, (stripe(row_ids[::2]), stripe(row_ids[1::2]))))

    def flatten(dictionary):
        data, offsets = [], [0]
        for sequence in dictionary:
            data.extend(sequence)
            offsets.append(len(data))
        return data, offsets

    template_data, template_offsets = flatten(templates)
    strip_data, strip_offsets = flatten(strips)
    for table in (probabilities, descriptors, templates, strips, rows, layers, template_data,
                  strip_data):
        if len(table) > 65535:
            raise OverflowError("compiler table exceeds uint16 layout")
    graph = CompressedGraph(codec, reference.num_observables, probabilities,
                            list(descriptors), template_data, template_offsets,
                            strip_data, strip_offsets, list(rows), list(layers), times,
                            [e for e in reference.edges if e.u == e.v == BOUNDARY])
    graph._validate_tables()
    return graph


def verify_equivalence(reference, compressed):
    # Counters intentionally check multiplicity and logical-distinct parallel edges.
    expected = Counter(reference.iter_edges())
    actual = Counter(compressed.iter_edges())
    if expected != actual:
        missing, extra = expected - actual, actual - expected
        raise AssertionError(f"graph mismatch: {sum(missing.values())} missing, "
                             f"{sum(extra.values())} extra edges; "
                             f"example {next(iter(missing or extra), None)}")
    # Directed incidence matters too: the decoder must see each edge from both ends.
    for v in range(reference.num_detectors):
        if Counter(reference.neighbors(v)) != Counter(compressed.neighbors(v)):
            raise AssertionError(f"neighbor mismatch at detector {v}")
    return len(expected)

"""Cheap correction-path routing score for larger surface-memory experiments."""

from __future__ import annotations

import ctypes as ct

import numpy as np

from ._build import routing_library
from .model import memory_detector_indices, validate_input
from .native import BytePointer, DoublePointer, support_library

IntPointer = ct.POINTER(ct.c_int32)


class PathProxy:
    def __init__(self, context, library=None, *, bucket=False, matching=None):
        if context.num_observables != 1:
            raise ValueError("path proxy requires one observable")
        self.num_detectors = context.num_detectors
        indices = memory_detector_indices(context)
        self.mapping = np.full(self.num_detectors, -1, dtype=np.int32)
        self.mapping[indices] = np.arange(len(indices), dtype=np.int32)
        self.nodes = len(indices) + 2
        self.boundary_logical = np.zeros(self.num_detectors, dtype=np.uint8)
        if matching is None:
            import pymatching

            matching = pymatching.Matching.from_detector_error_model(
                context.dem, enable_correlations=True
            )
        self.matching = matching
        self.matching.ensure_num_fault_ids(1)
        adjacency = [[] for _ in range(self.nodes)]
        weights = []
        for a, b, data in self.matching.edges():
            logical = bool(data["fault_ids"])
            if data["fault_ids"] - {0} or (logical and b is not None):
                raise ValueError("path proxy requires logical edges on a boundary")
            if self.mapping[a] < 0:
                if logical:
                    raise ValueError("logical edge outside memory check basis")
                continue
            if b is not None and self.mapping[b] < 0:
                raise ValueError("matching edge couples check bases")
            weight = data["weight"]
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("path proxy requires finite nonnegative graph weights")
            self.boundary_logical[a] = logical if b is None else self.boundary_logical[a]
            u = int(self.mapping[a])
            v = self.nodes - 1 - logical if b is None else int(self.mapping[b])
            edge = len(weights)
            weights.append(weight)
            adjacency[u].append((v, edge))
            adjacency[v].append((u, edge))
        if not weights or not self.boundary_logical.any():
            raise ValueError("path proxy has no logical boundary")
        self.weights = np.asarray(weights, dtype=np.float64)
        self.start = np.asarray([0, *np.cumsum([len(row) for row in adjacency])], dtype=np.int32)
        self.target = np.asarray([v for row in adjacency for v, _ in row], dtype=np.int32)
        self.ids = np.asarray([edge for row in adjacency for _, edge in row], dtype=np.int32)
        self.lib = ct.CDLL(str(library or support_library()))
        try:
            version = self.lib.path_proxy_abi_version
        except AttributeError as exc:
            raise RuntimeError("native library lacks path routing; rebuild it") from exc
        version.argtypes, version.restype = [], ct.c_int
        if version() != 1:
            raise RuntimeError("path routing ABI mismatch; rebuild it")
        self.lib.path_proxy.argtypes = (
            [ct.c_int] * 4
            + [IntPointer, BytePointer, IntPointer, IntPointer, IntPointer, DoublePointer]
            + [ct.POINTER(ct.c_int64), DoublePointer, BytePointer]
        )
        self.lib.path_proxy.restype = ct.c_int
        self._bucket = None
        if bucket:
            quantized = np.floor(self.weights * 64)
            positive = quantized[self.weights > 0]
            if len(positive) and positive.min() >= 1 and positive.max() <= np.iinfo(np.uint32).max:
                self._bucket_weights = quantized.astype(np.uint32)
                self._min_bucket_weight = int(positive.min())
                self._bucket_lib = ct.CDLL(str(routing_library()))
                self._bucket = self._bucket_lib.path_bucket_route
                self._bucket.argtypes = (
                    [ct.c_int] * 4 + [ct.c_void_p] * 6
                    + [ct.c_uint32, ct.c_void_p, ct.c_double, BytePointer, BytePointer]
                )
                self._bucket.restype = ct.c_int
                self._bucket_args = (
                    self.num_detectors, self.nodes, len(self.weights),
                    self.mapping.ctypes.data, self.boundary_logical.ctypes.data,
                    self.start.ctypes.data, self.target.ctypes.data, self.ids.ctypes.data,
                    self._bucket_weights.ctypes.data,
                )

    @property
    def nbytes(self):
        return sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray))

    def score_edges(self, pairs):
        pairs = np.ascontiguousarray(pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("correction edges must have shape [edges,2]")
        score, label = ct.c_double(), ct.c_uint8()
        status = self.lib.path_proxy(
            self.num_detectors,
            self.nodes,
            len(self.weights),
            len(pairs),
            self.mapping.ctypes.data_as(IntPointer),
            self.boundary_logical.ctypes.data_as(BytePointer),
            self.start.ctypes.data_as(IntPointer),
            self.target.ctypes.data_as(IntPointer),
            self.ids.ctypes.data_as(IntPointer),
            self.weights.ctypes.data_as(DoublePointer),
            pairs.ctypes.data_as(ct.POINTER(ct.c_int64)),
            ct.byref(score),
            ct.byref(label),
        )
        if status:
            raise RuntimeError(f"native path routing returned {status}")
        return label.value, score.value

    def route_edges(self, pairs, threshold):
        """Return the label and the exact threshold decision without an exact score."""
        if self._bucket is not None:
            pairs = np.ascontiguousarray(pairs, dtype=np.int64)
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError("correction edges must have shape [edges,2]")
            original, nodes, edges, *arrays = self._bucket_args
            routed, label = ct.c_uint8(), ct.c_uint8()
            status = self._bucket(
                original, nodes, edges, len(pairs), *arrays, self._min_bucket_weight,
                pairs.ctypes.data, threshold, ct.byref(routed), ct.byref(label),
            )
            if status == 0:
                return label.value, bool(routed.value)
            if status != 1:
                raise RuntimeError(f"bucket path routing returned {status}")
        label, score = self.score_edges(pairs)
        return label, score < threshold

    def decode_with_score(self, events):
        validate_input(events, self.num_detectors)
        bits = np.unpackbits(events, axis=1, count=self.num_detectors, bitorder="little")
        output = np.empty((len(events), 1), dtype=np.uint8)
        scores = np.empty(len(events))
        for i, row in enumerate(bits):
            pairs = self.matching.decode_to_edges_array(row, enable_correlations=True)
            output[i, 0], scores[i] = self.score_edges(pairs)
        return output, scores

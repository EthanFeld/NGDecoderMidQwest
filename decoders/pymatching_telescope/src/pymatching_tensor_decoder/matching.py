"""Fixed-topology wrapper for the pinned PyMatching core."""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np

from ._build import matching_library as build_library


class FixedMatching:
    def __init__(self, model):
        self.model = model
        self.library_path = build_library()
        self.lib = ct.CDLL(str(self.library_path))
        pointer = ct.c_void_p
        self.lib.latency_create.argtypes = [
            ct.c_int,
            ct.c_int,
            pointer,
            pointer,
            pointer,
            ct.c_int,
            ct.c_int,
            pointer,
            pointer,
        ]
        self.lib.latency_create.restype = pointer
        self.lib.latency_destroy.argtypes = [pointer]
        self.lib.latency_destroy.restype = None
        self.lib.latency_prepare.argtypes = [pointer, pointer, ct.c_int, pointer]
        self.lib.latency_prepare.restype = ct.c_int
        self.lib.latency_solve.argtypes = [pointer, pointer, pointer, pointer, pointer, pointer]
        self.lib.latency_solve.restype = ct.c_int
        self.lib.latency_solve_edges.argtypes = [pointer] * 4 + [ct.c_int, pointer]
        self.lib.latency_solve_edges.restype = ct.c_int
        self.lib.latency_events.argtypes = [pointer, pointer, ct.c_int, pointer]
        self.lib.latency_events.restype = ct.c_int
        self.lib.latency_error.argtypes = []
        self.lib.latency_error.restype = ct.c_char_p
        n, m = len(model.indices), len(model.memory_keys)
        u, v = np.empty(m, np.int32), np.empty(m, np.int32)
        labels = np.asarray([label for _, label in model.memory_keys], dtype=np.uint8)
        for j in range(m):
            nodes = model.check_matrix.indices[
                model.check_matrix.indptr[j] : model.check_matrix.indptr[j + 1]
            ]
            u[j], v[j] = nodes[0], nodes[1] if len(nodes) > 1 else -1
        keys, ids = [], []
        for j, (ds, _) in enumerate(model.memory_keys):
            keys.append((int(ds[0]) << 32) | (int(ds[1]) + 1 if len(ds) > 1 else 0))
            ids.append(j + 1)
        for ds, j in model.other_keys.items():
            keys.append((int(ds[0]) << 32) | (int(ds[1]) + 1 if len(ds) > 1 else 0))
            ids.append(-j - 1)
        keys, ids = np.asarray(keys, np.uint64), np.asarray(ids, np.int32)
        self.handle = self.lib.latency_create(
            n,
            m,
            u.ctypes.data,
            v.ctypes.data,
            labels.ctypes.data,
            len(model.other_keys),
            len(keys),
            keys.ctypes.data,
            ids.ctypes.data,
        )
        if not self.handle:
            raise RuntimeError(self.lib.latency_error().decode())
        self.parity = np.empty(len(model.other_keys) + 1, np.uint8)
        self.event_buffer = np.empty(len(model.memory_keys) + len(model.other_keys) + 1, np.uint8)
        self.edge_buffer = None

    def prepare(self, correction_edges):
        pairs = np.ascontiguousarray(correction_edges, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("correction edges must have shape (edges,2)")
        if self.lib.latency_prepare(
            self.handle, pairs.ctypes.data, len(pairs), self.parity.ctypes.data
        ):
            raise RuntimeError(self.lib.latency_error().decode())
        return self.parity

    def weights(self, parity, strength=1.0):
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("conditioning strength must be in [0,1]")
        model = self.model
        q = np.where(parity[model.pair_other], model.q1, model.q0)
        bias = np.multiply.reduceat(1 - 2 * q, model.starts)
        probability = np.clip((1 - bias) / 2, 1e-12, 1 - 1e-12)
        conditioned = np.log1p(-probability) - np.log(probability)
        return model.prior_weights + strength * (conditioned - model.prior_weights)

    def _inputs(self, syndrome, weights):
        syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if syndrome.shape != (len(self.model.indices),) or weights.shape != (
            len(self.model.memory_keys),
        ):
            raise ValueError("wrong fixed matching input dimensions")
        return syndrome, weights

    def solve_details(self, syndrome, weights):
        syndrome, weights = self._inputs(syndrome, weights)
        label, weight, gain = ct.c_uint8(), ct.c_double(), ct.c_double()
        if self.lib.latency_solve(
            self.handle,
            weights.ctypes.data,
            syndrome.ctypes.data,
            ct.byref(label),
            ct.byref(weight),
            ct.byref(gain),
        ):
            raise RuntimeError(self.lib.latency_error().decode())
        return np.array([label.value], np.uint8), weight.value, gain.value

    def solve(self, syndrome, weights):
        label, _, gain = self.solve_details(syndrome, weights)
        return label, gain

    def solve_edges(self, syndrome, weights):
        syndrome, weights = self._inputs(syndrome, weights)
        if self.edge_buffer is None:
            self.edge_buffer = np.empty((len(self.model.memory_keys), 2), np.int64)
        count = ct.c_int()
        if self.lib.latency_solve_edges(
            self.handle,
            weights.ctypes.data,
            syndrome.ctypes.data,
            self.edge_buffer.ctypes.data,
            len(self.edge_buffer),
            ct.byref(count),
        ):
            raise RuntimeError(self.lib.latency_error().decode())
        return self.edge_buffer[: count.value].copy()

    def events(self, correction_edges):
        """Borrow a joint parity workspace; valid until the next events() call."""
        pairs = np.ascontiguousarray(correction_edges, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("correction edges must have shape (edges,2)")
        if self.lib.latency_events(
            self.handle, pairs.ctypes.data, len(pairs), self.event_buffer.ctypes.data
        ):
            raise RuntimeError(self.lib.latency_error().decode())
        return self.event_buffer[:-1]

    def close(self):
        if getattr(self, "handle", None):
            self.lib.latency_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def audit_vendor():
    root = Path(__file__).parent / "native/vendor/pymatching_2_4_0"
    manifest = json.loads((root / "ORIGIN.json").read_text())
    for item in manifest["files"]:
        data = (root / item["path"]).read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != item["git_blob_sha"]:
            raise RuntimeError(f"upstream source integrity failure: {item['path']}")
    return len(manifest["files"])

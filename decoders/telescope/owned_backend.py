"""The telescope's own correlated matching engine; no external decoder code.

NumPy/SciPy provide arrays and model matrices. The compiled solver uses only the
C++ standard library and the original sources in native/owned_*. No PyMatching
package, binary, headers, or vendored sources are loaded by this backend.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import os
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
from tensor_decoder.conditioned import ConditionedMatching
from tensor_decoder.model import FaultModel, validate_input

from tensor_decoder import TensorDecoder

_NATIVE = Path(__file__).resolve().parent / "native"
_LIB = None


def _library():
    global _LIB
    if _LIB is not None:
        return _LIB
    sources = [
        _NATIVE / "owned_matching.cpp", _NATIVE / "owned_mwpm.h",
        _NATIVE / "owned_queue.h",
        _NATIVE / "owned_workers.h",
        _NATIVE / "owned_certificate.h",
        _NATIVE / "owned_correlated.h",
    ]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    cache = Path(
        os.environ.get(
            "TELESCOPE_NATIVE_CACHE",
            str(Path(tempfile.gettempdir()) / "telescope-owned"),
        )
    )
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / (
        "owned_matching_" + digest[:20] + (".dll" if os.name == "nt" else ".so")
    )
    if not output.exists():
        compiler = shutil.which(os.environ.get("CXX", "g++")) or shutil.which(
            "C:/msys64/mingw64/bin/g++.exe"
        )
        if compiler is None:
            raise RuntimeError(
                "The in-repo telescope matcher needs a C++20 compiler; set CXX."
            )
        fd, name = tempfile.mkstemp(dir=cache, suffix=output.suffix)
        os.close(fd)
        temporary = Path(name)
        env = os.environ.copy()
        env["PATH"] = str(Path(compiler).parent) + os.pathsep + env.get("PATH", "")
        try:
            subprocess.run(
                [
                    compiler,
                    "-O3",
                    "-DNDEBUG",
                    "-std=c++20",
                    "-pthread",
                    "-shared",
                    *(["-static"] if os.name == "nt" else ["-fPIC"]),
                    str(sources[0]),
                    "-o",
                    str(temporary),
                ],
                check=True,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                os.replace(temporary, output)
            except PermissionError:
                # Another process may have published and loaded this same hash.
                if not output.exists():
                    raise
        finally:
            temporary.unlink(missing_ok=True)
    lib = ct.CDLL(str(output))
    ptr = ct.c_void_p
    lib.owned_create.argtypes = [ct.c_int, ct.c_int, ptr, ptr, ct.c_int]
    lib.owned_create.restype = ptr
    lib.owned_destroy.argtypes = [ptr]
    lib.owned_solve.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr]
    lib.owned_solve.restype = ct.c_int
    lib.owned_hint.argtypes = [ptr, ptr, ptr, ptr, ptr]
    lib.owned_hint.restype = ct.c_int
    lib.owned_stats.argtypes = [ptr, ptr]
    lib.owned_reference.argtypes = [ptr, ptr]
    lib.owned_reference.restype = ct.c_int
    lib.owned_profile.argtypes = [ptr, ptr]
    lib.owned_profile.restype = None
    lib.owned_profile_count.argtypes = []
    lib.owned_profile_count.restype = ct.c_int
    lib.owned_partial_cache_budget.argtypes = [ptr, ct.c_int64]
    lib.owned_partial_cache_budget.restype = None
    lib.owned_lazy_metric.argtypes = [ptr, ct.c_int]
    lib.owned_lazy_metric.restype = None
    lib.owned_path_workers.argtypes = [ptr, ct.c_int]
    lib.owned_path_workers.restype = ct.c_int
    lib.owned_path_worker_count.argtypes = [ptr]
    lib.owned_path_worker_count.restype = ct.c_int
    lib.owned_correlated_create.argtypes = [
        ptr, ptr, ptr, ptr, ct.c_int, ptr, ptr, ptr,
        ct.c_int, ptr, ptr, ptr, ct.c_int,
    ]
    lib.owned_correlated_create.restype = ptr
    lib.owned_correlated_solve.argtypes = [ptr, ptr, ptr, ct.c_int, ptr, ct.POINTER(ct.c_int)]
    lib.owned_correlated_solve.restype = ct.c_int
    lib.owned_correlated_destroy.argtypes = [ptr]
    lib.owned_correlated_destroy.restype = None
    lib.owned_correlated_basis_workers.argtypes = [ptr, ct.c_int]
    lib.owned_correlated_basis_workers.restype = ct.c_int
    lib.owned_correlated_basis_worker_count.argtypes = [ptr]
    lib.owned_correlated_basis_worker_count.restype = ct.c_int
    lib.owned_correlated_snapshot.argtypes = [ptr, ptr, ptr]
    lib.owned_correlated_snapshot.restype = None
    lib.owned_error.restype = ct.c_char_p
    _LIB = lib
    return lib


def _binary(values, size, name):
    array = np.asarray(values)
    if array.shape != (size,):
        raise ValueError(f"{name} must be a binary vector of length {size}")
    if array.dtype == np.uint8:
        valid = array.max(initial=0) <= 1
    elif array.dtype == np.bool_:
        valid = True
    else:
        valid = ((array == 0) | (array == 1)).all()
    if not valid:
        raise ValueError(f"{name} must be a binary vector of length {size}")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _worker_count(count):
    if (
        not isinstance(count, (int, np.integer))
        or isinstance(count, (bool, np.bool_))
        or not 1 <= count <= 32
    ):
        raise ValueError("the shortest-path worker count must be an integer from 1 to 32")
    return int(count)


def _basis_worker_count(count):
    if (
        not isinstance(count, (int, np.integer))
        or isinstance(count, (bool, np.bool_))
        or count not in (1, 2)
    ):
        raise ValueError("the check-basis worker count must be one or two")
    return int(count)


class OwnedGraph:
    """Exact signed-weight minimum T-join with a virtual boundary."""

    def __init__(self, n, u, v, cache=False):
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("invalid detector count")
        u, v = np.asarray(u), np.asarray(v)
        if (
            u.ndim != 1
            or v.shape != u.shape
            or not np.isfinite(u).all()
            or not np.isfinite(v).all()
        ):
            raise ValueError("invalid endpoint arrays")
        if (
            (u != np.floor(u)).any()
            or (v != np.floor(v)).any()
            or ((u < 0) | (u >= n)).any()
            or ((v < -1) | (v >= n) | (u == v)).any()
        ):
            raise ValueError("invalid graph endpoint")
        self.n, self.m = int(n), len(u)
        self.u, self.v = (
            np.ascontiguousarray(u, dtype=np.int32),
            np.ascontiguousarray(v, dtype=np.int32),
        )
        self.lib = _library()
        self.handle = self.lib.owned_create(
            self.n, self.m, self.u.ctypes.data, self.v.ctypes.data, int(cache)
        )
        if not self.handle:
            raise RuntimeError(self.lib.owned_error().decode())
        self.lib.owned_lazy_metric(self.handle, 1)
        self._reference = None

    def set_reference(self, reference):
        if not self.handle or (reference is not None and not reference.handle):
            raise RuntimeError("matching graph is closed")
        if self.lib.owned_reference(
            self.handle, None if reference is None else reference.handle
        ):
            raise ValueError(self.lib.owned_error().decode())
        self._reference = reference

    def set_partial_cache_budget(self, byte_budget):
        if not self.handle:
            raise RuntimeError("matching graph is closed")
        if (
            not isinstance(byte_budget, (int, np.integer))
            or isinstance(byte_budget, (bool, np.bool_))
            or not 0 <= byte_budget <= np.iinfo(np.int64).max
        ):
            raise ValueError("the shortest-path cache budget must be a nonnegative integer")
        self.lib.owned_partial_cache_budget(self.handle, int(byte_budget))

    def set_path_workers(self, count):
        if not self.handle:
            raise RuntimeError("matching graph is closed")
        if self.lib.owned_path_workers(self.handle, _worker_count(count)):
            raise RuntimeError(self.lib.owned_error().decode())

    def path_worker_count(self):
        if not self.handle:
            raise RuntimeError("matching graph is closed")
        return self.lib.owned_path_worker_count(self.handle)

    def _weights(self, weights):
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if weights.shape != (self.m,) or not np.isfinite(weights).all():
            raise ValueError("invalid graph weights")
        return weights

    def solve(self, syndrome, weights, seed=None):
        if not self.handle:
            raise RuntimeError("matching graph is closed")
        if self._reference is not None and not self._reference.handle:
            self.set_reference(None)
        syndrome, weights = (
            _binary(syndrome, self.n, "syndrome"),
            self._weights(weights),
        )
        if seed is not None:
            seed = _binary(seed, self.m, "seed")
        chosen = np.empty(self.m, np.uint8)
        cost = ct.c_double()
        if self.lib.owned_solve(
            self.handle,
            syndrome.ctypes.data,
            weights.ctypes.data,
            None if seed is None else seed.ctypes.data,
            chosen.ctypes.data,
            ct.byref(cost),
        ):
            raise RuntimeError(self.lib.owned_error().decode())
        return chosen, cost.value

    def hint(self, syndrome, removed, weights):
        if not self.handle:
            raise RuntimeError("matching graph is closed")
        if self._reference is not None and not self._reference.handle:
            self.set_reference(None)
        syndrome = _binary(syndrome, self.n, "syndrome")
        removed = _binary(removed, self.n, "removed syndrome")
        if np.any(removed & ~syndrome):
            raise ValueError(
                "removed detectors must be a subset of the original syndrome"
            )
        weights = self._weights(weights)
        selected = np.empty(self.m, np.uint8)
        if self.lib.owned_hint(
            self.handle,
            syndrome.ctypes.data,
            removed.ctypes.data,
            weights.ctypes.data,
            selected.ctypes.data,
        ):
            raise RuntimeError(self.lib.owned_error().decode())
        return selected

    def stats(self):
        if not self.handle:
            raise RuntimeError("matching graph is closed")
        result = np.empty(14, np.int64)
        self.lib.owned_stats(self.handle, result.ctypes.data)
        return dict(
            zip(
                (
                    "solves",
                    "certificates",
                    "blossoms",
                    "dijkstra_searches",
                    "cache_hits",
                    "seeded_solves",
                    "seed_certificates",
                    "initial_pairs",
                    "hinted_pairs",
                    "augmentations",
                    "sparse_rounds",
                    "priced_edges",
                    "dense_fallbacks",
                    "incremental_searches",
                ),
                map(int, result),
                strict=True,
            )
        )

    def profile(self):
        if not self.handle:
            raise RuntimeError("matching graph is closed")
        names = (
            "setup_ns", "certificate_ns", "metric_ns", "preferred_ns",
            "matching_ns", "trace_ns", "hint_ns", "shortest_ns",
            "matching_build_ns", "matching_solve_ns", "matching_price_ns",
            "matching_prepare_ns",
        )
        count = self.lib.owned_profile_count()
        result = np.empty(count, dtype=np.int64)
        self.lib.owned_profile(self.handle, result.ctypes.data)
        return {
            names[i] if i < len(names) else f"extra_{i}_ns": int(value)
            for i, value in enumerate(result)
        }

    def close(self):
        if getattr(self, "handle", None):
            self.lib.owned_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def _endpoints(matrix):
    matrix = matrix.tocsc()
    u, v = [], []
    for j in range(matrix.shape[1]):
        nodes = matrix.indices[matrix.indptr[j] : matrix.indptr[j + 1]]
        if not 1 <= len(nodes) <= 2:
            raise ValueError("matching requires one or two checks per edge")
        u.append(int(nodes[0]))
        v.append(int(nodes[1]) if len(nodes) == 2 else -1)
    return np.asarray(u, np.int32), np.asarray(v, np.int32)


class _Joint:
    def __init__(self, matrix, weights):
        self.graph = OwnedGraph(matrix.shape[0], *_endpoints(matrix))
        # Only static-model shortest-path workspaces are retained, never answers.
        self.graph.set_partial_cache_budget(64 * 1024 * 1024)
        self.weights = weights

    def ensure_num_fault_ids(self, count):
        if count != 1:
            raise ValueError("the telescope supports one logical observable")

    def decode(self, events, return_weight=False):
        _, cost = self.graph.solve(events, self.weights)
        label = np.zeros(1, np.uint8)
        return (label, cost) if return_weight else label


class _Fixed:
    """Compatibility surface for the tensor's original fixed-topology solver."""

    def __init__(self, model, cache=False):
        self.model = model
        self.u, self.v = _endpoints(model.check_matrix)
        self.graph = OwnedGraph(len(model.indices), self.u, self.v, cache=cache)
        self.logical = np.asarray(
            [label for _, label in model.memory_keys], dtype=np.uint8
        )
        self.lookup = {}
        for i, (ds, _) in enumerate(model.memory_keys):
            self.lookup[tuple(ds)] = i
        self.other = model.other_keys
        self.memory_count = len(model.memory_keys)
        self.baseline = np.zeros(self.memory_count, np.uint8)
        keys, identifiers = [], []
        for ds, index in self.lookup.items():
            keys.append((int(ds[0]) << 32) | (int(ds[1]) + 1 if len(ds) == 2 else 0))
            identifiers.append(index)
        for ds, index in self.other.items():
            keys.append((int(ds[0]) << 32) | (int(ds[1]) + 1 if len(ds) == 2 else 0))
            identifiers.append(self.memory_count + index)
        order = np.argsort(np.asarray(keys, dtype=np.uint64), kind="stable")
        self._edge_keys = np.asarray(keys, dtype=np.uint64)[order]
        self._edge_identifiers = np.asarray(identifiers, dtype=np.intp)[order]

    @staticmethod
    def _key(a, b):
        return (int(a),) if b < 0 else tuple(sorted((int(a), int(b))))

    def events(self, pairs):
        parity = np.zeros(self.memory_count + len(self.other), np.uint8)
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if not len(pairs):
            return parity
        a, b = pairs[:, 0], pairs[:, 1]
        left = np.where(b < 0, a, np.minimum(a, b))
        right = np.where(b < 0, 0, np.maximum(a, b) + 1)
        if np.any(left < 0):
            raise ValueError("correction edge absent from the physical model")
        keys = (left.astype(np.uint64) << 32) | right.astype(np.uint64)
        positions = np.searchsorted(self._edge_keys, keys)
        if np.any(positions >= len(self._edge_keys)) or not np.array_equal(
            self._edge_keys[positions], keys,
        ):
            raise ValueError("correction edge absent from the physical model")
        np.bitwise_xor.at(parity, self._edge_identifiers[positions], 1)
        return parity

    def prepare(self, pairs):
        parity = self.events(pairs)
        self.baseline = parity[: self.memory_count].copy()
        result = np.empty(len(self.other) + 1, dtype=np.uint8)
        result[:-1] = parity[self.memory_count :]
        result[-1] = 0
        return result

    def weights(self, parity):
        model = self.model
        q = np.where(parity[model.pair_other], model.q1, model.q0)
        probability = np.clip(
            (1 - np.multiply.reduceat(1 - 2 * q, model.starts)) / 2, 1e-12, 1 - 1e-12
        )
        return np.log1p(-probability) - np.log(probability)

    def solve_details(self, syndrome, weights):
        chosen, cost = self.graph.solve(syndrome, weights)
        label = np.array([int(np.dot(chosen, self.logical)) & 1], np.uint8)
        return label, cost, float(np.dot(self.baseline, weights)) - cost

    def solve_edges(self, syndrome, weights):
        chosen, _ = self.graph.solve(syndrome, weights)
        selected = np.flatnonzero(chosen)
        return np.column_stack((self.u[selected], self.v[selected])).astype(np.int64)


class _Engine:
    def __init__(self, path_workers=4, basis_workers=2):
        self.path_workers = _worker_count(path_workers)
        self.basis_workers = _basis_worker_count(basis_workers)
        self.models = {}
        self.correlated_model = None
        self.fixed_models = []

    def conditioned(self, context, joint_check=False, target_other=False):
        key = (joint_check, target_other)
        if key not in self.models:
            self.models[key] = ConditionedMatching(
                context,
                joint_check=joint_check,
                target_other=target_other,
                joint_matching_factory=_Joint,
            )
        return self.models[key]

    def fixed(self, model):
        fixed = _Fixed(model)
        self.fixed_models.append(fixed)
        return fixed

    def correlated(self, context):
        if self.correlated_model is None:
            self.correlated_model = OwnedCorrelatedMatching(context, engine=self)
        return self.correlated_model


class OwnedCorrelatedMatching:
    """Original two-pass matching with complete-fault cross-basis correlations."""

    def __init__(self, context, *, engine=None, path_workers=4, basis_workers=2):
        self.context = context
        self.num_detectors = context.num_detectors
        self._fused_handle = None
        self.engine = engine or _Engine(path_workers=path_workers, basis_workers=basis_workers)
        self.memory = self.engine.conditioned(context)
        self.other = self.engine.conditioned(context, target_other=True)
        self.prior = [_Fixed(self.memory, cache=True), _Fixed(self.other, cache=True)]
        self.updated = [_Fixed(self.memory), _Fixed(self.other)]
        for before, after in zip(self.prior, self.updated, strict=True):
            after.graph.set_reference(before.graph)
            after.graph.set_path_workers(self.engine.path_workers)
        self.models = [self.memory, self.other]
        self.offset = len(self.memory.memory_keys)
        self.base_weights = np.r_[self.memory.prior_weights, self.other.prior_weights]
        self.labels = np.r_[
            self.prior[0].logical, np.zeros(len(self.other.memory_keys), np.uint8)
        ]
        self.global_edges = []
        self.global_key = {}
        for model in self.models:
            for ds, logical in model.memory_keys:
                self.global_key[tuple(ds)] = len(self.global_edges)
                self.global_edges.append(
                    (ds[0], ds[1] if len(ds) == 2 else -1, bool(logical))
                )
        self._global_pairs = np.asarray(
            [(a, b) for a, b, _ in self.global_edges], dtype=np.int64,
        ).reshape(-1, 2)
        # Independent equivalent complete mechanisms combine by XOR, not addition.
        grouped = {}
        physical = FaultModel.from_context(context)
        memory_nodes = set(map(int, self.memory.indices))
        for p, ds in zip(physical.probabilities, physical.detectors, strict=True):
            left = tuple(d for d in ds if d in memory_nodes)
            right = tuple(d for d in ds if d not in memory_nodes)
            if left and right:
                a, b = self.global_key[left], self.global_key[right]
                grouped[a, b] = grouped.get((a, b), 0.0) + np.log1p(-2 * p)
        causes, targets, chances = [], [], []
        priors = np.r_[self.memory.prior_probability, self.other.prior_probability]
        for (a, b), bias in grouped.items():
            chance = -np.expm1(bias) / 2
            for source, target in ((a, b), (b, a)):
                causes.append(source)
                targets.append(target)
                chances.append(chance)
        self.causes = np.asarray(causes, np.intp)
        self.targets = np.asarray(targets, np.intp)
        probability = np.clip(np.asarray(chances) / priors[self.causes], 1e-12, 0.5)
        self.rule_weights = np.log1p(-probability) - np.log(probability)
        self._native_indices = tuple(
            np.ascontiguousarray(model.indices, dtype=np.int32) for model in self.models
        )
        self._native_causes = np.ascontiguousarray(self.causes, dtype=np.int32)
        self._native_targets = np.ascontiguousarray(self.targets, dtype=np.int32)
        self._fused_lib = self.prior[0].graph.lib
        self._fused_handle = self._fused_lib.owned_correlated_create(
            self.prior[0].graph.handle, self.prior[1].graph.handle,
            self.updated[0].graph.handle, self.updated[1].graph.handle,
            self.num_detectors, self._native_indices[0].ctypes.data,
            self._native_indices[1].ctypes.data, self.base_weights.ctypes.data,
            len(self.causes), self._native_causes.ctypes.data,
            self._native_targets.ctypes.data, self.rule_weights.ctypes.data,
            self.engine.basis_workers,
        )
        if not self._fused_handle:
            raise RuntimeError(self._fused_lib.owned_error().decode())
        self.front_hints = deque()
        self.hints_received = self.hints_used = 0

    def ensure_num_fault_ids(self, count):
        if count != 1:
            raise ValueError("the telescope supports one logical observable")

    def edges(self):
        return [
            (
                a,
                None if b < 0 else b,
                {"weight": float(w), "fault_ids": {0} if logical else set()},
            )
            for (a, b, logical), w in zip(
                self.global_edges, self.base_weights, strict=True
            )
        ]

    def _pairs(self, selected):
        return self._global_pairs[np.flatnonzero(selected)]

    def decode_to_edges_array(self, row, enable_correlations=True):
        if not self._fused_handle:
            raise RuntimeError("correlated matching is closed")
        row = _binary(row, self.num_detectors, "joint syndrome")
        hint = self.front_hints.popleft() if self.front_hints else None
        if hint is not None:
            self.hints_received += 1
            try:
                hint = _binary(hint, len(self.memory.indices), "removed syndrome")
            except ValueError:
                hint = None
        selected = np.empty(len(self.base_weights), dtype=np.uint8)
        used = ct.c_int()
        if self._fused_lib.owned_correlated_solve(
            self._fused_handle, row.ctypes.data,
            None if hint is None else hint.ctypes.data,
            int(bool(enable_correlations)), selected.ctypes.data, ct.byref(used),
        ):
            raise RuntimeError(self._fused_lib.owned_error().decode())
        self.hints_used += used.value
        return self._pairs(selected)

    def set_basis_workers(self, count):
        if not self._fused_handle:
            raise RuntimeError("correlated matching is closed")
        count = _basis_worker_count(count)
        if self._fused_lib.owned_correlated_basis_workers(self._fused_handle, count):
            raise RuntimeError(self._fused_lib.owned_error().decode())
        self.engine.basis_workers = count

    def basis_worker_count(self):
        if not self._fused_handle:
            raise RuntimeError("correlated matching is closed")
        return self._fused_lib.owned_correlated_basis_worker_count(self._fused_handle)

    def last_correlated_state(self):
        if not self._fused_handle:
            raise RuntimeError("correlated matching is closed")
        first = np.empty(len(self.base_weights), dtype=np.uint8)
        weights = np.empty_like(self.base_weights)
        self._fused_lib.owned_correlated_snapshot(
            self._fused_handle, first.ctypes.data, weights.ctypes.data,
        )
        return first, weights

    def decode_batch(self, events):
        validate_input(events, self.context.num_detectors)
        bits = np.unpackbits(
            events, axis=1, count=self.context.num_detectors, bitorder="little"
        )
        predictions = np.empty((len(events), 1), np.uint8)
        for i, row in enumerate(bits):
            pairs = self.decode_to_edges_array(row)
            label = 0
            for a, b in pairs:
                label ^= int(self.labels[self.global_key[_Fixed._key(a, b)]])
            predictions[i, 0] = label
        return predictions

    def close(self):
        if getattr(self, "_fused_handle", None):
            self._fused_lib.owned_correlated_destroy(self._fused_handle)
            self._fused_handle = None
        # Join path workers before releasing the prior graphs they reference.
        for model in (*self.updated, *self.prior):
            model.graph.close()

    def __del__(self):
        # Release native basis workers for standalone matchers and engine cycles.
        if getattr(self, "_fused_handle", None):
            self._fused_lib.owned_correlated_destroy(self._fused_handle)
            self._fused_handle = None


class OwnedTensorDecoder(TensorDecoder):
    """The main telescope tensor rule, backed entirely by our own matcher."""

    def __init__(
        self,
        context,
        config=None,
        *,
        device="cpu",
        solve_workers=1,
        path_bucket=False,
        cuda_correlated=False,
        path_workers=4,
        basis_workers=2,
    ):
        if cuda_correlated:
            raise ValueError(
                "the in-repo matcher runs on CPU; CUDA conditioning remains available"
            )
        self.owned_engine = _Engine(path_workers=path_workers, basis_workers=basis_workers)
        super().__init__(
            context,
            config,
            device=device,
            solve_workers=solve_workers,
            path_bucket=path_bucket,
            matching_engine=self.owned_engine,
        )

    def close(self):
        super().close()
        engine = self.owned_engine
        matching = engine.correlated_model
        if matching is not None:
            matching.close()
        graphs = (
            [] if matching is None
            else [model.graph for model in (*matching.updated, *matching.prior)]
        )
        graphs.extend(model.graph for model in engine.fixed_models)
        for model in engine.models.values():
            graph = getattr(getattr(model, "joint_matching", None), "graph", None)
            if graph is not None:
                graphs.append(graph)
        seen = set()
        for graph in graphs:
            if id(graph) not in seen:
                seen.add(id(graph))
                graph.close()

    def decode_with_front(self, events, residual, logical, valid, memory_indices):
        validate_input(events, self.num_detectors)
        matching = self.owned_engine.correlated_model
        if matching is None:
            return self.decode_batch(events)
        residual, logical, valid = (
            np.asarray(residual),
            np.asarray(logical),
            np.asarray(valid),
        )
        mapping = np.asarray(memory_indices)
        n = len(events)
        expected = matching.memory.indices
        if (
            residual.shape != (n, len(expected))
            or logical.shape != (n,)
            or valid.shape != (n,)
            or not np.array_equal(mapping, expected)
        ):
            return self.decode_batch(events)
        raw = np.unpackbits(
            events, axis=1, count=self.num_detectors, bitorder="little"
        )[:, expected]
        hints = []
        for i in range(n):
            safe = (
                valid[i] == 1
                and np.isin(residual[i], [0, 1]).all()
                and logical[i] == 0
                and not np.any(residual[i] > raw[i])
            )
            hints.append(
                np.bitwise_xor(raw[i], residual[i].astype(np.uint8)) if safe else None
            )
        matching.front_hints = deque(hints)
        try:
            return self.decode_batch(events)
        finally:
            matching.front_hints.clear()

    def owned_stats(self):
        matching = self.owned_engine.correlated_model
        graphs = (
            []
            if matching is None
            else [f.graph for f in (*matching.prior, *matching.updated)]
        )
        graphs += [f.graph for f in self.owned_engine.fixed_models]
        joint = self.owned_engine.models.get((True, False))
        if joint is not None:
            graphs.append(joint.joint_matching.graph)
        totals = {}
        for graph in graphs:
            for key, value in graph.stats().items():
                totals[key] = totals.get(key, 0) + value
        if matching is not None:
            totals.update(
                fpga_hints_received=matching.hints_received,
                fpga_hints_used=matching.hints_used,
            )
        return totals

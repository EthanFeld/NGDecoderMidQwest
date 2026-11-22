"""Native two-pass correlated sparse blossom for batches of syndromes."""

from __future__ import annotations

import ctypes as ct
from functools import cache
from pathlib import Path

import numpy as np

from .correlated_source import vendor_sources
from .gpu import CUDAError, _compile, _device_ptr, _integer, _pointer, _size, _Workspace

_NATIVE = Path(__file__).parent / "native"
_STACK_LIMIT = 0  # CU_LIMIT_STACK_SIZE
_TOPOLOGY = (
    "offsets", "neighbors", "edge_ids", "endpoints", "weights", "rule_offsets",
    "rule_targets", "rule_weights", "logical",
)


class _GraphView(ct.Structure):
    _fields_ = [("nodes", _integer), ("edges", _integer)] + [
        (field, _device_ptr) for field in (*_TOPOLOGY, "edge_slots")
    ]


def _edge_slots(graph):
    """Map each undirected edge to its two directed slots, or one at a boundary."""
    slots = np.full((len(graph.weights), 2), -1, dtype=np.int32)
    nodes = np.repeat(np.arange(graph.num_detectors), np.diff(graph.offsets))
    first = nodes == graph.endpoints[graph.edge_ids, 0]
    slots[graph.edge_ids[first], 0] = np.flatnonzero(first)
    slots[graph.edge_ids[~first], 1] = np.flatnonzero(~first)
    return slots


@cache
def _ptx(architecture):
    core, headers = vendor_sources()
    source = b"\n".join((
        (_NATIVE / "cuda_mwpm_containers.cuh").read_bytes(),
        core,
        (_NATIVE / "correlated_matching.cu").read_bytes(),
    ))
    return _compile(
        architecture, source, headers,
        (b"--device-as-default-execution-space", b"--std=c++20", b"-DNDEBUG"),
    )


class CUDACorrelated(_Workspace):
    """GPU matching on a fixed DEM. A None row requests the pinned CPU decoder."""

    def __init__(self, graph, *, max_batch=256, device=0, block_size=1):
        if type(max_batch) is not int or not 1 <= max_batch <= 256:
            raise ValueError("CUDA correlated batch must be in [1, 256]")
        if block_size not in (1, 2, 4, 8, 16, 32):
            raise ValueError("CUDA correlated block size must be a power of two in [1, 32]")
        self.max_batch = max_batch
        self._block_size = block_size
        self.nodes, self.num_edges = len(graph.offsets) - 1, len(graph.weights)
        self._limit = min(self.num_edges, self.nodes)
        self._stride = (2 * 1024 * 1024 + 256 * self.nodes + 32 * len(graph.edge_ids) + 15) & ~15
        self._function = _pointer()
        super().__init__(device)
        try:
            ptx = _ptx(self._dev.architecture)
            with self._dev.current():
                # Alternating tree and blossom traversal are recursive.
                stack = _size()
                self._dev.get_limit(ct.byref(stack), _STACK_LIMIT)
                if stack.value < 16 * 1024:
                    self._dev.set_limit(_STACK_LIMIT, 16 * 1024)
                self._dev.load_module(ct.byref(self._module), ptx)
                self._dev.get_function(
                    ct.byref(self._function), self._module, b"correlated_matching",
                )
                self._view = _GraphView(self.nodes, self.num_edges, *(
                    self._copy(getattr(graph, name)).value for name in _TOPOLOGY
                ), self._copy(_edge_slots(graph)).value)
                self._scratch = self._allocate(self._stride * self.max_batch)
                self._shots = self._copy(np.zeros(self.max_batch, dtype=np.uint64))
                self._syndromes = self._allocate(self.nodes * self.max_batch)
                self._sizes = self._allocate(4 * self.max_batch)
                self._edges = self._allocate(8 * self._limit * self.max_batch)
                self._host_syndromes = self._allocate_host(ct.c_uint8, self.nodes)
                self._host_sizes = self._allocate_host(ct.c_int32, 1)
                self._host_edges = self._allocate_host(ct.c_int32, 2 * self._limit)
                self._dev.create_stream(ct.byref(self._stream), 1)
        except BaseException:
            self.close()
            raise

    def edges(self, syndromes):
        if self._closed:
            raise CUDAError("CUDA correlated matcher is closed")
        if syndromes.ndim != 2 or syndromes.shape[1] != self.nodes or syndromes.dtype != np.uint8:
            raise ValueError("expected one uint8 column per detector")
        correction = []
        with self._dev.current():
            for start in range(0, len(syndromes), self.max_batch):
                rows = syndromes[start : start + self.max_batch]
                count = len(rows)
                incoming = self._host_syndromes[:count]
                sizes, edges = self._host_sizes[:count], self._host_edges[:count]
                incoming[...] = rows
                self._dev.upload_async(
                    self._syndromes, incoming.ctypes.data, incoming.nbytes, self._stream,
                )
                args = (
                    self._view, self._syndromes, _integer(count), self._scratch,
                    _size(self._stride), self._shots, self._sizes, self._edges,
                    _integer(self._limit),
                )
                pointers = (_pointer * len(args))(*(ct.addressof(arg) for arg in args))
                blocks = (count + self._block_size - 1) // self._block_size
                self._dev.launch(
                    self._function, blocks, 1, 1, self._block_size, 1, 1, 0,
                    self._stream, pointers, None,
                )
                self._dev.download_async(sizes.ctypes.data, self._sizes, sizes.nbytes, self._stream)
                self._dev.download_async(edges.ctypes.data, self._edges, edges.nbytes, self._stream)
                self._dev.sync_stream(self._stream)
                correction.extend(
                    None if size < 0 else row[: 2 * size].astype(np.int64).reshape(-1, 2)
                    for [size], row in zip(sizes, edges, strict=True)
                )
        return correction

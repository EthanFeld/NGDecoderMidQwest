"""Optional CUDA evaluation of the tensor decoder's conditioned weights.

For models with compact tables, the GPU selects exact NumPy-computed weights
from static parity tables. Larger models use ordered FP64 products on CUDA
and the same NumPy logarithms on the host. CUDA loads on first use.
"""

from __future__ import annotations

import ctypes as ct
import os
import sys
from contextlib import contextmanager
from ctypes.util import find_library
from functools import cache
from pathlib import Path

import numpy as np

_SOURCE = rb"""
extern "C" __global__ void parity_products(
    const int *starts, const int *other, const double *factor0,
    const double *factor1, const unsigned char *parities, double *bias,
    int edges, int stride
) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= edges) return;
    int shot = blockIdx.y;
    const unsigned char *parity = parities + stride * shot;
    double product = 1.0;
    for (int k = starts[edge]; k < starts[edge + 1]; ++k) {
        double factor = parity[other[k]] ? factor1[k] : factor0[k];
        product *= factor;
    }
    bias[shot * edges + edge] = product;
}

extern "C" __global__ void parity_weights(
    const int *starts, const int *other, const int *offsets,
    const double *table, const unsigned char *parities, double *weights,
    int edges, int stride
) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= edges) return;
    int shot = blockIdx.y;
    const unsigned char *parity = parities + stride * shot;
    unsigned int mask = 0;
    int start = starts[edge];
    for (int k = start; k < starts[edge + 1]; ++k) {
        mask |= static_cast<unsigned int>(parity[other[k]] != 0) << (k - start);
    }
    weights[shot * edges + edge] = table[offsets[edge] + mask];
}
"""

# Cap each static FP64 table at 64 MiB. Models beyond this budget use the
# product kernel without materializing the exponential parity combinations.
_MAX_TABLE_ENTRIES = 8 * 1024 * 1024

_pointer = ct.c_void_p
_integer = ct.c_int
_unsigned = ct.c_uint
_size = ct.c_size_t
_device_ptr = ct.c_uint64


class CUDAError(RuntimeError):
    """CUDA is unavailable or could not complete the requested work."""


def _bind(lib, name, args, *, nvrtc=False):
    fn = getattr(lib, name)
    fn.argtypes = args
    fn.restype = _integer

    def checked(*values):
        status = fn(*values)
        if status:
            if nvrtc:
                details = lib.nvrtcGetErrorString(status)
            else:
                msg = ct.c_char_p()
                lib.cuGetErrorString(status, ct.byref(msg))
                details = msg.value
            message = details.decode(errors="replace") if details else f"status {status}"
            raise CUDAError(f"{name}: {message}")

    return checked


def _nvrtc():
    root = os.getenv("CUDA_PATH") or os.getenv("CUDA_HOME")
    if sys.platform == "win32":
        if root:
            for path in (Path(root) / "bin" / "x64", Path(root) / "bin"):
                names = sorted(path.glob("nvrtc64_*.dll"), reverse=True)
                if names:
                    with os.add_dll_directory(path):
                        return ct.CDLL(str(names[0]))
        raise CUDAError("NVRTC DLL was not found. Set CUDA_PATH to the installed CUDA toolkit.")
    library = str(Path(root) / "lib64" / "libnvrtc.so") if root else find_library("nvrtc")
    return ct.CDLL(library or "libnvrtc.so")


def _compile(architecture, source=_SOURCE, headers=None, options=()):
    lib = _nvrtc()
    lib.nvrtcGetErrorString.argtypes = [_integer]
    lib.nvrtcGetErrorString.restype = ct.c_char_p
    program = _pointer()
    char_list = ct.POINTER(ct.c_char_p)
    create = _bind(
        lib,
        "nvrtcCreateProgram",
        [ct.POINTER(_pointer), ct.c_char_p, ct.c_char_p, _integer, char_list, char_list],
        nvrtc=True,
    )
    compile_program = _bind(lib, "nvrtcCompileProgram", [_pointer, _integer, char_list], nvrtc=True)
    destroy = _bind(lib, "nvrtcDestroyProgram", [ct.POINTER(_pointer)], nvrtc=True)
    log_size = _bind(lib, "nvrtcGetProgramLogSize", [_pointer, ct.POINTER(_size)], nvrtc=True)
    log_data = _bind(lib, "nvrtcGetProgramLog", [_pointer, _pointer], nvrtc=True)
    ptx_size = _bind(lib, "nvrtcGetPTXSize", [_pointer, ct.POINTER(_size)], nvrtc=True)
    ptx_data = _bind(lib, "nvrtcGetPTX", [_pointer, _pointer], nvrtc=True)
    headers = headers or {}
    names = (ct.c_char_p * len(headers))(*headers)
    sources = (ct.c_char_p * len(headers))(*headers.values())
    create(ct.byref(program), source, b"tensor_decoder.cu", len(headers), sources, names)
    try:
        arguments = (
            f"--gpu-architecture=compute_{architecture}".encode(), b"--fmad=false", *options,
        )
        flags = (ct.c_char_p * len(arguments))(*arguments)
        size = _size()
        try:
            compile_program(program, len(flags), flags)
        except CUDAError as exc:
            log_size(program, ct.byref(size))
            log = ct.create_string_buffer(size.value)
            log_data(program, log)
            raise CUDAError(log.value.decode(errors="replace")) from exc
        ptx_size(program, ct.byref(size))
        ptx = ct.create_string_buffer(size.value)
        ptx_data(program, ptx)
        return ptx
    finally:
        destroy(ct.byref(program))


class _Device:
    def __init__(self, ordinal):
        lib = ct.WinDLL("nvcuda.dll") if sys.platform == "win32" else ct.CDLL("libcuda.so.1")
        lib.cuGetErrorString.argtypes = [_integer, ct.POINTER(ct.c_char_p)]
        lib.cuGetErrorString.restype = _integer
        get_device = _bind(lib, "cuDeviceGet", [ct.POINTER(_integer), _integer])
        get_attribute = _bind(
            lib, "cuDeviceGetAttribute", [ct.POINTER(_integer), _integer, _integer]
        )
        retain = _bind(lib, "cuDevicePrimaryCtxRetain", [ct.POINTER(_pointer), _integer])
        self.release = _bind(lib, "cuDevicePrimaryCtxRelease_v2", [_integer])
        self.push = _bind(lib, "cuCtxPushCurrent_v2", [_pointer])
        self.pop = _bind(lib, "cuCtxPopCurrent_v2", [ct.POINTER(_pointer)])
        self.get_limit = _bind(lib, "cuCtxGetLimit", [ct.POINTER(_size), _integer])
        self.set_limit = _bind(lib, "cuCtxSetLimit", [_integer, _size])
        self.alloc = _bind(lib, "cuMemAlloc_v2", [ct.POINTER(_device_ptr), _size])
        self.free = _bind(lib, "cuMemFree_v2", [_device_ptr])
        self.alloc_host = _bind(lib, "cuMemAllocHost_v2", [ct.POINTER(_pointer), _size])
        self.free_host = _bind(lib, "cuMemFreeHost", [_pointer])
        self.upload = _bind(lib, "cuMemcpyHtoD_v2", [_device_ptr, _pointer, _size])
        self.upload_async = _bind(
            lib, "cuMemcpyHtoDAsync_v2", [_device_ptr, _pointer, _size, _pointer]
        )
        self.download_async = _bind(
            lib, "cuMemcpyDtoHAsync_v2", [_pointer, _device_ptr, _size, _pointer]
        )
        self.create_stream = _bind(lib, "cuStreamCreate", [ct.POINTER(_pointer), _unsigned])
        self.sync_stream = _bind(lib, "cuStreamSynchronize", [_pointer])
        self.destroy_stream = _bind(lib, "cuStreamDestroy_v2", [_pointer])
        self.launch = _bind(
            lib,
            "cuLaunchKernel",
            [_pointer] + [_unsigned] * 7 + [_pointer, ct.POINTER(_pointer), ct.POINTER(_pointer)],
        )
        self.load_module = _bind(lib, "cuModuleLoadData", [ct.POINTER(_pointer), _pointer])
        self.get_function = _bind(
            lib, "cuModuleGetFunction", [ct.POINTER(_pointer), _pointer, ct.c_char_p]
        )
        self.unload_module = _bind(lib, "cuModuleUnload", [_pointer])
        _bind(lib, "cuInit", [_unsigned])(0)
        self.dev = _integer()
        get_device(ct.byref(self.dev), ordinal)
        major, minor = _integer(), _integer()
        get_attribute(ct.byref(major), 75, self.dev)  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        get_attribute(ct.byref(minor), 76, self.dev)  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        self.architecture = f"{major.value}{minor.value}"
        ptx = _compile(self.architecture)
        self.context, self.module = _pointer(), _pointer()
        self.product_function, self.lookup_function = _pointer(), _pointer()
        retain(ct.byref(self.context), self.dev)
        try:
            with self.current():
                self.load_module(ct.byref(self.module), ptx)
                self.get_function(ct.byref(self.product_function), self.module, b"parity_products")
                self.get_function(ct.byref(self.lookup_function), self.module, b"parity_weights")
        except BaseException:
            self.release(self.dev)
            raise

    @contextmanager
    def current(self):
        self.push(self.context)
        try:
            yield
        finally:
            self.pop(ct.byref(_pointer()))


@cache
def _device(ordinal):
    try:
        return _Device(ordinal)
    except (OSError, AttributeError) as exc:
        raise CUDAError(f"could not load CUDA driver or NVRTC: {exc}") from exc


def _condition_weights(bias, prior):
    probability = np.clip((1 - bias) / 2, 1e-12, 1 - 1e-12)
    conditioned = np.log1p(-probability) - np.log(probability)
    return prior + (conditioned - prior)


def _parity_table(starts, other, factor0, factor1, prior):
    """Deduplicate equal ordered factor sequences, keeping their exact weights."""
    switched = factor0 != factor1
    mask_starts = [0]
    offsets = []
    signatures = {}
    table_size = 0
    for begin, end, weight in zip(starts[:-1], starts[1:], prior, strict=True):
        zero, one = factor0[begin:end], factor1[begin:end]
        key = (zero.tobytes(), one.tobytes(), weight.tobytes())
        degree = int(np.count_nonzero(switched[begin:end]))
        if key not in signatures:
            count = 1 << degree
            if table_size + count > _MAX_TABLE_ENTRIES:
                return None
            signatures[key] = (table_size, zero, one, weight)
            table_size += count
        offsets.append(signatures[key][0])
        mask_starts.append(mask_starts[-1] + degree)

    table = np.empty(table_size, dtype=np.float64)
    for start, zero, one, weight in signatures.values():
        # Append the odd branch for each changing factor so bit k is the
        # k-th changing factor. Each entry keeps the original multiply order.
        product = np.ones(1, dtype=np.float64)
        for even, odd in zip(zero, one, strict=True):
            if even == odd:
                product *= even
            else:
                product = np.concatenate((product * even, product * odd))
        table[start : start + len(product)] = _condition_weights(product, weight)

    return (
        np.asarray(mask_starts, dtype=np.int32),
        np.append(other[switched], np.int32(0)),
        np.asarray(offsets, dtype=np.int32),
        table,
    )


class _Workspace:
    def __init__(self, ordinal):
        self._dev = _device(ordinal)
        self._buffers = []
        self._host_buffers = []
        self._stream = _pointer()
        self._module = _pointer()
        self._closed = False

    def _allocate(self, size):
        ptr = _device_ptr()
        self._dev.alloc(ct.byref(ptr), size)
        self._buffers.append(ptr)
        return ptr

    def _copy(self, array):
        ptr = self._allocate(max(1, array.nbytes))
        if array.nbytes:
            self._dev.upload(ptr, array.ctypes.data, array.nbytes)
        return ptr

    def _allocate_host(self, cell, width):
        ptr = _pointer()
        count = self.max_batch * width
        self._dev.alloc_host(ct.byref(ptr), count * ct.sizeof(cell))
        self._host_buffers.append(ptr)
        array = np.ctypeslib.as_array((cell * count).from_address(ptr.value))
        return array.reshape(self.max_batch, width)

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        with self._dev.current():
            if self._stream.value:
                self._dev.destroy_stream(self._stream)
            for ptr in self._buffers:
                self._dev.free(ptr)
            for ptr in self._host_buffers:
                self._dev.free_host(ptr)
            if self._module.value:
                self._dev.unload_module(self._module)

    def __del__(self):
        try:
            self.close()
        except CUDAError:
            pass  # CUDA may have shut down at interpreter teardown.


class CUDAModel(_Workspace):
    """Reuse device tables for one fixed conditioning model and its parity batches."""

    def __init__(self, model, *, device=0, max_batch=64):
        if type(max_batch) is not int or max_batch < 1:
            raise ValueError("max_batch must be a positive integer")
        super().__init__(device)
        self.prior = model.prior_weights
        self.edges = len(model.starts)
        self.stride = len(model.other_keys) + 1
        self.max_batch = min(max_batch, 65_535)  # CUDA grid Y dimension
        # The absent opposite-basis edge uses the final false parity entry.
        starts = np.append(model.starts, len(model.q0)).astype(np.int32, copy=False)
        other = np.where(model.pair_other < 0, self.stride - 1, model.pair_other).astype(
            np.int32, copy=False
        )
        factor0 = 1 - 2 * model.q0
        factor1 = 1 - 2 * model.q1
        static = _parity_table(starts, other, factor0, factor1, self.prior)
        self.table_bytes = static[-1].nbytes if static is not None else 0
        self._lookup = static is not None
        if self._lookup:
            self._function = self._dev.lookup_function
        else:
            self._function = self._dev.product_function
            self._unconditioned_bias = np.multiply.reduceat(factor0, model.starts)
            self._unconditioned_weights = _condition_weights(self._unconditioned_bias, self.prior)
            static = (starts, other, factor0, factor1)
        try:
            with self._dev.current():
                self._static = tuple(self._copy(array) for array in static)
                self._parities = self._allocate(self.stride * self.max_batch)
                self._products = self._allocate(self.edges * self.max_batch * 8)
                self._host_parities = self._allocate_host(ct.c_uint8, self.stride)
                self._host_bias = self._allocate_host(ct.c_double, self.edges)
                self._dev.create_stream(ct.byref(self._stream), 1)  # CU_STREAM_NON_BLOCKING
        except BaseException:
            self.close()
            raise

    def weights(self, parities):
        """Return host-complete FP64 weights for one or more uint8 parity rows."""
        if self._closed:
            raise CUDAError("conditioned weight model is closed")
        if not isinstance(parities, np.ndarray):
            raise ValueError("parities must be a NumPy array")
        single = parities.ndim == 1
        if single:
            parities = parities[None, :]
        if parities.ndim != 2 or parities.shape[1] != self.stride or parities.dtype != np.uint8:
            raise ValueError("expected uint8 parities with one element per opposite edge plus one")
        weights = np.empty((len(parities), self.edges), dtype=np.float64)
        if len(parities):
            with self._dev.current():
                for start in range(0, len(parities), self.max_batch):
                    source = parities[start : start + self.max_batch]
                    destination = weights[start : start + len(source)]
                    kernel_args = (
                        *self._static,
                        self._parities,
                        self._products,
                        _integer(self.edges),
                        _integer(self.stride),
                    )
                    pointers = (_pointer * len(kernel_args))(
                        *(ct.addressof(arg) for arg in kernel_args)
                    )
                    host_parities = self._host_parities[: len(source)]
                    host_bias = self._host_bias[: len(source)]
                    host_parities[...] = source
                    self._dev.upload_async(
                        self._parities, host_parities.ctypes.data, source.nbytes, self._stream
                    )
                    grid = (self.edges + 127) // 128
                    self._dev.launch(
                        self._function,
                        grid,
                        len(source),
                        1,
                        128,
                        1,
                        1,
                        0,
                        self._stream,
                        pointers,
                        None,
                    )
                    self._dev.download_async(
                        host_bias.ctypes.data, self._products, host_bias.nbytes, self._stream
                    )
                    self._dev.sync_stream(self._stream)
                    if self._lookup:
                        destination[...] = host_bias
                    else:
                        destination[...] = self._unconditioned_weights
                        changed = np.flatnonzero(host_bias != self._unconditioned_bias)
                        prior = self.prior[changed % self.edges]
                        destination.ravel()[changed] = _condition_weights(
                            host_bias.ravel()[changed], prior
                        )
        return weights[0] if single else weights


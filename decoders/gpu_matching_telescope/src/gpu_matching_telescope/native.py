"""Shared ABI types for the current path and private small-code kernels."""

import ctypes as ct

from ._build import support_library as support_library

BytePointer = ct.POINTER(ct.c_uint8)
DoublePointer = ct.POINTER(ct.c_double)

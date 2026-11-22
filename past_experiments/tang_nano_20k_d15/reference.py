"""Load the C++17 reference model for software validation."""

import ctypes as C
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from gen_policy import ROOT, generate
from transport import PROJECTED_BYTES


REASONS = ("accepted", "event_limit", "one_hop_span", "two_hop_span",
           "unpaired_interior", "boundary_depth", "boundary_policy_disagreement",
           "extended_crossings", "accepted_three_hop_pair", "accepted_dense",
           "accepted_boundary_extension", "accepted_wide_even", "extension_budget")


def build_reference(work):
    work = Path(work).resolve()
    generate(work / "policy.h", "#")
    windows = sys.platform == "win32"
    library = work / ("cluster_reference.dll" if windows else "cluster_reference.so")
    flags = ["-static"] if windows else ["-fPIC"]
    compiler = os.environ.get("CXX", "c++")
    subprocess.run([compiler, "-O3", "-std=c++17", "-shared", *flags, "-I", str(work),
                    str(ROOT / "reference.cpp"), "-o", str(library)], check=True)
    return library


class Reference:
    def __init__(self, library):
        self._module = C.CDLL(str(library))
        self._decode = self._module.decode_batch
        self._prior = self._module.prior_batch
        for entry in (self._decode, self._prior):
            entry.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p]
            entry.restype = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        # Windows keeps the DLL file locked until it is unloaded.
        if sys.platform == "win32":
            from _ctypes import FreeLibrary

            self._decode = self._prior = None
            FreeLibrary(self._module._handle)
            self._module = None

    def decode(self, projected, *, prior=False):
        data = np.ascontiguousarray(projected)
        if data.ndim != 2 or data.shape[1] != PROJECTED_BYTES or data.dtype != np.uint8:
            raise ValueError("The reference needs a batch of packed projected FPGA input rows.")
        codes, reasons = np.empty((2, len(data)), dtype=np.uint8)
        entry = self._prior if prior else self._decode
        entry(data.ctypes.data, len(data), codes.ctypes.data, reasons.ctypes.data)
        return codes, reasons

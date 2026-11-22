#!/usr/bin/env python3
"""Drive a cycle model of the bounded cluster engine."""

import ctypes
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
WINDOWS = sys.platform == 'win32'
MODEL_PATH = ROOT / 'build' / ('growth_model.dll' if WINDOWS else 'growth_model.so')


def build_model(compiler):
    MODEL_PATH.parent.mkdir(exist_ok=True)
    flags = ['-static'] if WINDOWS else ['-fPIC']
    subprocess.run([*compiler, '-O3', *flags, '-shared', '-std=c++20', '-Wall',
                    '-Wextra', '-Wpedantic', str(ROOT / 'tools' / 'growth_model.cpp'),
                    '-o', str(MODEL_PATH)], check=True)


class GrowthModel:
    def __init__(self, fixture, library=None):
        with fixture.open() as stream:
            self.vertices, count, _, self.width, self.max_age, *options = map(int, stream.readline().split())
            self.dual_port, self.short_updates, self.initial_age = options + [0, 0, 1][len(options):]
            self.edges = np.ascontiguousarray(np.loadtxt(stream, dtype=np.int32).reshape(count, 4))
        model_path = MODEL_PATH if library is None else Path(library).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f'{model_path} is missing. Run python growth_reference.py in surface-telescope.')
        self.library = ctypes.CDLL(str(model_path))
        self.model = self.library.simulate_growth_from_age
        I, P = ctypes.c_int, ctypes.c_void_p
        self.model.argtypes = [I, I, I, I, P, I, P, I, I, I, I, P]
        self.model.restype = None

    def decode(self, syndrome, cap):
        data = np.ascontiguousarray(syndrome, dtype=np.uint8)
        if data.ndim != 2 or data.shape[1] != self.vertices or cap < 1:
            raise ValueError('wrong syndrome dimensions or cycle cap')
        result = np.zeros((len(data), 3), dtype=np.int32)
        self.model(self.vertices, self.width, self.max_age, len(self.edges), self.edges.ctypes.data,
                   len(data), data.ctypes.data, cap, self.dual_port, self.short_updates,
                   self.initial_age, result.ctypes.data)
        return result

    def recover(self, result, syndrome, cap):
        missed = np.flatnonzero(~result.accepted)
        decoded = self.decode(syndrome[missed], cap)
        result.clocks[missed] += decoded[:, 2]+1
        chosen = decoded[:, 1] == 0
        completed = missed[chosen]
        result.residual[completed] = False
        result.logical[completed] = decoded[chosen, 0].astype(bool)
        return missed, decoded


if __name__ == '__main__':
    if len(sys.argv) > 1:
        compiler = sys.argv[1:]
    else:
        compiler = [os.environ.get('CXX', 'c++')]
    build_model(compiler)

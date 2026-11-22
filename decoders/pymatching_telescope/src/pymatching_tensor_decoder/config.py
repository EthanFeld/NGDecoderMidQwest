"""Defaults of the validated 400,000-shot decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecoderConfig:
    path_factor: float = 3.0
    min_gain: float = 8.0
    chunk_shots: int = 256
    max_detectors: int = 4096
    max_mechanisms: int = 100000
    ambiguous_gap: float = 8.0
    min_margin: float = 3.0
    iterations: int = 2
    reverse_seed: bool = True

    def __post_init__(self):
        for name in ("path_factor", "min_gain", "ambiguous_gap", "min_margin"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("chunk_shots", "max_detectors", "max_mechanisms"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.iterations) is not int or not 1 <= self.iterations <= 8:
            raise ValueError("iterations must be an integer in [1,8]")
        if type(self.reverse_seed) is not bool:
            raise ValueError("reverse_seed must be boolean")

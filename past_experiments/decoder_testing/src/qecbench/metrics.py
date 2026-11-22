"""Vectorized logical failure accounting and explicitly batch-amortized timing."""

from __future__ import annotations

import math

import numpy as np


def count_failures(predictions: np.ndarray, actual: np.ndarray) -> int:
    """Count shots with any logical mismatch, not the number of flipped bits."""
    if predictions.shape != actual.shape:
        raise ValueError("prediction and observable shapes differ")
    return int(np.count_nonzero(np.any(predictions != actual, axis=1)))


def wilson_interval(errors: int, shots: int) -> tuple[float, float]:
    if not 0 <= errors <= shots or shots <= 0:
        raise ValueError("require 0 <= errors <= shots and shots > 0")
    z = 1.959963984540054
    rate = errors / shots
    denominator = 1 + z * z / shots
    center = (rate + z * z / (2 * shots)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / shots + z * z / (4 * shots * shots)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize(errors: int, shots: int, decode_seconds: float) -> dict[str, float | None]:
    low, high = wilson_interval(errors, shots)
    return {
        "logical_error_rate": errors / shots,
        "ci95_low": low,
        "ci95_high": high,
        "decode_shots_per_second": shots / decode_seconds if decode_seconds > 0 else None,
        "decode_us_per_shot": 1e6 * decode_seconds / shots,
    }

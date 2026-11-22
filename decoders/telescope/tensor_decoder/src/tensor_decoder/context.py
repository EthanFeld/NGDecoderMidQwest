"""Standalone circuit description and the decoder context interface."""

from __future__ import annotations

import math
from dataclasses import dataclass

import stim


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


@dataclass(frozen=True)
class Experiment:
    distance: int
    rounds: int
    noise: float
    basis: str = "z"
    geometry: str = "rotated"
    noise_model: str = "circuit"

    def __post_init__(self) -> None:
        _integer("distance", self.distance, 2)
        _integer("rounds", self.rounds, 1)
        if isinstance(self.noise, bool) or not isinstance(self.noise, (int, float)):
            raise ValueError("noise must be a finite probability in [0, 1]")
        if not math.isfinite(self.noise) or not 0 <= self.noise <= 1:
            raise ValueError("noise must be a finite probability in [0, 1]")
        object.__setattr__(self, "noise", float(self.noise))
        if self.basis not in ("x", "z"):
            raise ValueError("basis must be 'x' or 'z'")
        if self.geometry not in ("rotated", "unrotated"):
            raise ValueError("geometry must be 'rotated' or 'unrotated'")
        if self.noise_model not in ("circuit", "phenomenological", "data"):
            raise ValueError("noise_model must be 'circuit', 'phenomenological', or 'data'")

    def circuit(self) -> stim.Circuit:
        kwargs = {"before_round_data_depolarization": self.noise}
        if self.noise_model in ("circuit", "phenomenological"):
            kwargs["before_measure_flip_probability"] = self.noise
        if self.noise_model == "circuit":
            kwargs["after_clifford_depolarization"] = self.noise
            kwargs["after_reset_flip_probability"] = self.noise
        return stim.Circuit.generated(
            f"surface_code:{self.geometry}_memory_{self.basis}",
            distance=self.distance,
            rounds=self.rounds,
            **kwargs,
        )


@dataclass(frozen=True)
class DecoderContext:
    experiment: Experiment
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel

    @property
    def num_detectors(self) -> int:
        return self.circuit.num_detectors

    @property
    def num_observables(self) -> int:
        return self.circuit.num_observables

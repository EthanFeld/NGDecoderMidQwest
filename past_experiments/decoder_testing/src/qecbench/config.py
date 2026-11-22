"""Validated, serializable experiment and execution configuration."""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import stim


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def resolve_rounds(spec: str | int, distance: int) -> int:
    if type(spec) is int:
        result = spec
    elif isinstance(spec, str) and spec in ("d", "distance"):
        result = distance
    elif isinstance(spec, str) and re.fullmatch(r"[1-9][0-9]*d", spec):
        result = int(spec[:-1]) * distance
    elif isinstance(spec, str) and spec.isdecimal():
        result = int(spec)
    else:
        raise ValueError("rounds must be a positive integer, 'distance', or a multiplier like '2d'")
    _integer("rounds", result, 1)
    return result


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
class SweepConfig:
    distances: tuple[int, ...] = (3, 5, 7)
    noise: tuple[float, ...] = (0.001, 0.003, 0.01)
    bases: tuple[str, ...] = ("z",)
    geometries: tuple[str, ...] = ("rotated",)
    rounds: str | int = "distance"
    noise_model: str = "circuit"
    decoders: tuple[str, ...] = ("pymatching",)
    shots: int = 100_000
    min_shots: int = 0
    max_errors: int = 0
    batch_size: int = 8192
    memory_mib: float = 256
    warmup_shots: int = 32
    workers: int = 1
    seed: int = 123
    output: str = "past_experiments/decoder_testing/results/run"

    def __post_init__(self) -> None:
        for name in ("distances", "noise", "bases", "geometries", "decoders"):
            value = getattr(self, name)
            if isinstance(value, str) or not isinstance(value, (tuple, list)) or not value:
                raise ValueError(f"{name} must be a nonempty list")
            normalized = tuple(value)
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{name} contains duplicates")
            object.__setattr__(self, name, normalized)
        for name, minimum in (
            ("shots", 1),
            ("min_shots", 0),
            ("max_errors", 0),
            ("batch_size", 1),
            ("warmup_shots", 0),
            ("workers", 1),
            ("seed", 0),
        ):
            _integer(name, getattr(self, name), minimum)
        if self.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
        if self.min_shots > self.shots:
            raise ValueError("min_shots cannot exceed shots")
        if isinstance(self.memory_mib, bool) or not isinstance(self.memory_mib, (int, float)):
            raise ValueError("memory_mib must be finite and positive")
        if not math.isfinite(self.memory_mib) or self.memory_mib <= 0:
            raise ValueError("memory_mib must be finite and positive")
        if not isinstance(self.output, str) or not self.output.strip():
            raise ValueError("output must be a nonempty path string")
        for decoder in self.decoders:
            if not isinstance(decoder, str) or not decoder:
                raise ValueError("decoder names must be nonempty strings")
            if decoder not in ("pymatching", "zero"):
                module, separator, factory = decoder.partition(":")
                if not separator or not module or not factory or ":" in factory:
                    raise ValueError(f"unknown decoder {decoder!r}; use 'module:factory'")
        # Validate every point before creating any output files or workers.
        for _ in self.experiments():
            pass

    def experiments(self) -> tuple[Experiment, ...]:
        return tuple(
            Experiment(d, resolve_rounds(self.rounds, d), p, b, g, self.noise_model)
            for d, p, b, g in itertools.product(
                self.distances, self.noise, self.bases, self.geometries
            )
        )

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> SweepConfig:
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
        return cls(**values)

    def identity(self) -> dict[str, Any]:
        values = asdict(self)
        # Output location and process scheduling do not alter samples or predictions.
        values.pop("output")
        values.pop("workers")
        return values

    @property
    def output_path(self) -> Path:
        return Path(self.output).expanduser().resolve()

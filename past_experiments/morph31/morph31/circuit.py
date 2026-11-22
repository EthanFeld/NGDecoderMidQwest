"""Reproducible circuit/DEM and full-circuit sample caches (not edge sampling)."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path

import numpy as np
import stim


@dataclass(frozen=True)
class Experiment:
    distance: int = 5
    rounds: int = 5
    basis: str = "x"
    noise: float = 0.001

    def __post_init__(self):
        if (not isinstance(self.distance, Integral) or isinstance(self.distance, bool)
                or not 3 <= self.distance <= 31 or self.distance % 2 != 1):
            raise ValueError("distance must be odd, between 3 and 31")
        if (not isinstance(self.rounds, Integral) or isinstance(self.rounds, bool)
                or not 1 <= self.rounds <= 31):
            raise ValueError("rounds must be between 1 and 31 for the 15-bit codec")
        if self.basis not in ("x", "z"):
            raise ValueError("basis must be x or z")
        if not np.isfinite(self.noise) or not 0 <= self.noise < 0.5:
            raise ValueError("noise must be finite and in [0, 0.5)")

    @property
    def key(self):
        description = {**asdict(self), "stim": stim.__version__, "schema": 1}
        return sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:20]

    def generate(self):
        return stim.Circuit.generated(
            f"surface_code:rotated_memory_{self.basis}",
            distance=self.distance,
            rounds=self.rounds,
            after_clifford_depolarization=self.noise,
            before_round_data_depolarization=self.noise,
            before_measure_flip_probability=self.noise,
            after_reset_flip_probability=self.noise,
        )


def load_experiment(experiment: Experiment, cache: str | Path):
    folder = Path(cache) / experiment.key
    folder.mkdir(parents=True, exist_ok=True)
    circuit_path, dem_path = folder / "circuit.stim", folder / "model.dem"
    if circuit_path.exists() and dem_path.exists():
        circuit = stim.Circuit.from_file(circuit_path)
        dem = stim.DetectorErrorModel.from_file(dem_path)
    else:
        circuit = experiment.generate()
        dem = circuit.detector_error_model(decompose_errors=True)
        circuit.to_file(circuit_path)
        dem.to_file(dem_path)
        (folder / "experiment.json").write_text(
            json.dumps({**asdict(experiment), "stim": stim.__version__}, indent=2) + "\n"
        )
    # Detector count comes from Stim; ancilla count is never used for this.
    if circuit.num_detectors != dem.num_detectors:
        raise ValueError("cached circuit and DEM disagree")
    return circuit, dem, folder


def sample_dataset(circuit, folder: Path, shots: int, seed: int):
    if shots < 1 or seed < 0 or seed >= 2**64:
        raise ValueError("shots must be positive and seed must be an unsigned 64-bit integer")
    path = folder / f"samples-{shots}-{seed}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            detections, observables = data["detections"], data["observables"]
    else:
        detections, observables = circuit.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True
        )
        np.savez_compressed(path, detections=detections, observables=observables)
    if detections.shape != (shots, circuit.num_detectors):
        raise ValueError("sample cache shape mismatch")
    if observables.shape != (shots, circuit.num_observables):
        raise ValueError("observable cache shape mismatch")
    return detections, observables

import math

import pytest

from qecbench import Experiment, SweepConfig
from qecbench.config import resolve_rounds
from qecbench.runner import batch_seed, effective_batch_size


@pytest.mark.parametrize("spec,expected", [("distance", 5), ("d", 5), ("2d", 10), ("7", 7), (9, 9)])
def test_rounds(spec, expected):
    assert resolve_rounds(spec, 5) == expected


@pytest.mark.parametrize("spec", [0, -1, "0d", "-1", "1.5d", True, 1.5])
def test_invalid_rounds(spec):
    with pytest.raises(ValueError):
        resolve_rounds(spec, 5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distances": []},
        {"distances": [1]},
        {"distances": [3, 3]},
        {"distances": [True]},
        {"noise": [math.nan]},
        {"noise": [math.inf]},
        {"noise": [-0.1]},
        {"noise": [1.1]},
        {"noise": [True]},
        {"bases": ["y"]},
        {"geometries": ["toric"]},
        {"noise_model": "typo"},
        {"shots": 0},
        {"batch_size": 0},
        {"workers": 0},
        {"seed": -1},
        {"seed": 2**63},
        {"memory_mib": 0},
        {"memory_mib": math.inf},
        {"memory_mib": True},
        {"shots": 5, "min_shots": 6},
        {"decoders": ["typo"]},
        {"decoders": ["mod:factory:extra"]},
        {"decoders": "pymatching"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SweepConfig(**kwargs)


def test_unknown_config_key():
    with pytest.raises(ValueError, match="unknown config keys"):
        SweepConfig.from_mapping({"shotz": 100})


@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("geometry", ["rotated", "unrotated"])
@pytest.mark.parametrize("noise_model", ["circuit", "phenomenological", "data"])
def test_all_circuit_presets(basis, geometry, noise_model):
    experiment = Experiment(3, 3, 0.003, basis, geometry, noise_model)
    circuit = experiment.circuit()
    assert circuit.num_observables == 1
    assert circuit.num_detectors > 0
    circuit.detector_error_model(decompose_errors=True)
    text = str(circuit)
    assert ("DEPOLARIZE2" in text) == (noise_model == "circuit")


def test_grid_batch_cap_and_seed():
    config = SweepConfig(distances=[3, 5], noise=[0.001, 0.01], bases=["x", "z"])
    assert len(config.experiments()) == 8
    experiment = config.experiments()[0]
    assert batch_seed(123, experiment, 0) == batch_seed(123, experiment, 0)
    assert batch_seed(123, experiment, 0) != batch_seed(123, experiment, 1)
    constrained = SweepConfig(memory_mib=0.01, batch_size=8192)
    size = effective_batch_size(constrained, 1000, 1)
    assert 1 <= size < constrained.batch_size
    with pytest.raises(ValueError, match="too small"):
        effective_batch_size(SweepConfig(memory_mib=1e-9), 1000, 1)

import numpy as np
import pytest
import stim

from morph31.circuit import Experiment, load_experiment, sample_dataset
from morph31.dem import parse_dem
from morph31.graph import ExplicitGraph


def test_correlations_parallel_labels_and_multiplicity():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0 ^ D2
        error(0.2) D0 D1 L0
        error(0.3) D0 D1
        error(0.4) L0
        detector(2,0,0) D0
        detector(4,2,1) D1
        detector(2,2,1) D2
    """)
    graph = parse_dem(dem)
    by_key = {e.key: e for e in graph.edges}
    assert by_key[0, 1, 1].probability == pytest.approx(0.26)
    assert by_key[0, 1, 1].multiplicity == 2
    assert by_key[0, 1, 0].probability == pytest.approx(0.3)
    assert by_key[-1, -1, 1].probability == pytest.approx(0.4)
    assert by_key[-1, 2, 0].probability == pytest.approx(0.1)
    assert graph.metadata["correlated_error_instructions"] == 1
    assert not graph.metadata["correlations_preserved"]


def test_repeat_shifts_and_cancellation():
    dem = stim.DetectorErrorModel("""
        repeat 2 {
            error(0.1) D0 L0 ^ D0 L0
            error(0.2) D0 D0 D1 L0 L0
            detector(2,0,0) D0
            detector(4,2,0) D1
            shift_detectors(0,0,1) 2
        }
    """)
    graph = parse_dem(dem)
    assert [(e.u, e.v, e.logical_mask) for e in graph.edges] == [(-1, 1, 0), (-1, 3, 0)]
    assert np.array_equal(graph.coordinates[2], [2, 0, 1])


def test_reject_non_graphlike():
    with pytest.raises(ValueError, match="non-graphlike"):
        parse_dem(stim.DetectorErrorModel("error(0.1) D0 D1 D2"))


def test_explicit_cache_and_full_circuit_sampling(models, tmp_path):
    _, _, graph, _ = models()
    path = tmp_path / "graph.npz"
    graph.save(path)
    loaded = ExplicitGraph.load(path)
    assert loaded.edges == graph.edges
    assert np.array_equal(loaded.coordinates, graph.coordinates)
    assert loaded.metadata == graph.metadata
    experiment = Experiment()
    circuit, dem, folder = load_experiment(experiment, tmp_path)
    a = sample_dataset(circuit, folder, 32, 31)
    b = sample_dataset(circuit, folder, 32, 31)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert a[0].shape == (32, dem.num_detectors)
    assert load_experiment(experiment, tmp_path)[0] == circuit


@pytest.mark.parametrize("kwargs", [dict(distance=4), dict(rounds=32), dict(basis="y"),
                                   dict(noise=float("nan")), dict(noise=0.5)])
def test_invalid_experiment(kwargs):
    with pytest.raises(ValueError):
        Experiment(**kwargs)

import pytest

from morph31.circuit import Experiment
from morph31.compression import compile_graph
from morph31.dem import parse_dem
from morph31.geometry import LatticeCodec


@pytest.fixture(scope="session")
def models():
    cache = {}

    def model(d=3, rounds=None, basis="x", noise=0.001):
        rounds = d if rounds is None else rounds
        key = d, rounds, basis, noise
        if key not in cache:
            circuit = Experiment(d, rounds, basis, noise).generate()
            dem = circuit.detector_error_model(decompose_errors=True)
            graph = parse_dem(dem)
            compressed = compile_graph(graph, LatticeCodec(d, rounds, basis))
            cache[key] = circuit, dem, graph, compressed
        return cache[key]

    return model

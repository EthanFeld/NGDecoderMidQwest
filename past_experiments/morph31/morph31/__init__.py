"""MORPH-31: implicit graphs, guarded sparse decoding, and tiny motif policies."""

from .circuit import Experiment
from .compression import CompressedGraph, compile_graph
from .decoder import SparseDecoder

__all__ = ["CompressedGraph", "Experiment", "SparseDecoder", "compile_graph"]

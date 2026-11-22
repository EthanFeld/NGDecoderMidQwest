"""Stable public interface for surface-code decoder benchmarks."""

from .config import Experiment, SweepConfig
from .decoders import DecoderContext, PackedDecoder, UnpackedAdapter
from .runner import run_sweep

__version__ = "0.1.0"
__all__ = [
    "DecoderContext",
    "Experiment",
    "PackedDecoder",
    "SweepConfig",
    "UnpackedAdapter",
    "run_sweep",
]

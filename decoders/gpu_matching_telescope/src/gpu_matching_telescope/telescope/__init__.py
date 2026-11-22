"""FPGA telescope with the CUDA correlated matching tensor tail."""

from .decoder import FrontBatch, ReferenceFront, SerialFront, TelescopeDecoder

__all__ = ["FrontBatch", "ReferenceFront", "SerialFront", "TelescopeDecoder"]

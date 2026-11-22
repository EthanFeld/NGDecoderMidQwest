"""Restored PyMatching tensor decoder.

Supported factory: pymatching_tensor_decoder:build_decoder.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ._build import matching_library, support_library
from .config import DecoderConfig as DecoderConfig
from .context import DecoderContext as DecoderContext
from .context import Experiment as Experiment
from .decoder import TensorDecoder as TensorDecoder
from .decoder import build_decoder as build_decoder

__version__ = "1.0.0"


def qecbench_implementation_identity():
    folder = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*.py")):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(path.read_bytes())
    return {
        "implementation": "validated-d15-sector-search-v2",
        "python_sha256": digest.hexdigest(),
        "path_native_sha256": hashlib.sha256(support_library().read_bytes()).hexdigest(),
        "matching_native_sha256": hashlib.sha256(matching_library().read_bytes()).hexdigest(),
    }

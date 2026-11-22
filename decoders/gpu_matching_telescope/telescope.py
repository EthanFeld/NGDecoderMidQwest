"""Run the Telescope with native CUDA correlated matching."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gpu_matching_telescope.telescope.decoder import main  # noqa: E402

if __name__ == "__main__":
    main()

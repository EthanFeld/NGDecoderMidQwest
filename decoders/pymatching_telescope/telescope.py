"""Run the restored PyMatching telescope without changing the independent decoder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pymatching_tensor_decoder.telescope.decoder import main  # noqa: E402

if __name__ == "__main__":
    main()

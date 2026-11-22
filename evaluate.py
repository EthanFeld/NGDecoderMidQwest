"""Regenerate the published benchmark tables from stored measurements (standard library only)."""

import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"),
                        help="destination for results.md, two CSV tables and provenance JSON (default: evaluation)")
    args = parser.parse_args()
    script = Path(__file__).resolve().parent / "measurements/2026-11-22-deck-v2/write_numbers.py"
    sys.argv = [str(script), "--export-dir", str(args.output_dir), "--require-complete"]
    runpy.run_path(str(script), run_name="__main__")
    print(f"Results: {args.output_dir.resolve() / 'results.md'}")
    print("Source: stored measurements; no simulation, decoder execution or FPGA connection.")


if __name__ == "__main__":
    main()

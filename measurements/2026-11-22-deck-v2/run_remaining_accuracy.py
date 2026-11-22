"""Run independent remaining accuracy cohorts without shared table writes."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("deck_model_runner", BASE / "regenerate_model.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", choices=("d15-p10000", "d7"), required=True)
    args = parser.parse_args()
    directory = BASE / "model"
    settings = json.loads((directory / "manifest.json").read_text())
    points = [(15, .010, 2048)] if args.points == "d15-p10000" else [(7, .003, 8192), (7, .0075, 8192)]
    # Each worker publishes atomic per-point checkpoints only. The coordinator
    # aggregates the shared CSV/JSON after the workers and original process exit.
    runner.helper.write_tables = lambda *args, **kwargs: None
    print(json.dumps({"phase": "independent_accuracy", "pid": os.getpid(),
                      "points": args.points, "started_utc": runner.utc()}), flush=True)
    for d, p, n in points:
        checkpoint = directory / "checkpoints" / f"{runner.name_for(d, p)}.json"
        if checkpoint.exists():
            continue
        runner.accuracy_point(directory, settings, runner.helper.get_profile(d), d, p, n)
    print(json.dumps({"phase": "independent_accuracy_complete", "pid": os.getpid(),
                      "points": args.points, "completed_utc": runner.utc()}), flush=True)


if __name__ == "__main__":
    main()

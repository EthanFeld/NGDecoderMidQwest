"""Measure the five latency points requested for the comparison slides."""

import importlib.util
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("deck_requested_timing", BASE / "regenerate_model.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

REQUESTED = [(15, .003), (15, .005), (15, .0075), (7, .003), (7, .0075)]


def main():
    directory = BASE / "model"
    settings = json.loads((directory / "manifest.json").read_text())
    scope = {
        "requested_timing_points": [{"distance": d, "p": p} for d, p in REQUESTED],
        "accuracy_points": 13,
        "requested_time_shots": runner.TIME_SHOTS,
        "relay_requested_time_shots_d15": 24,
        "unmeasured_latency_fields": "blank",
        "runner_sha256": runner.helper.sha256(Path(__file__)),
        "updated_utc": runner.utc(),
    }
    requested_names = {runner.name_for(d, p) for d, p in REQUESTED}
    scope["additional_already_measured_points"] = [
        path.stem for path in sorted((directory / "checkpoints").glob("*.json"))
        if path.stem not in requested_names
        and all(row.get("timing_complete") for row in json.loads(path.read_text()))
    ]
    runner.helper.atomic_json(directory / "timing_scope.json", scope)
    settings["requested_timing_points"] = scope["requested_timing_points"]
    settings["timing_scope_file"] = "measurements/2026-11-22-deck-v2/model/timing_scope.json"
    print(json.dumps({"phase": "requested_timing", "pid": os.getpid(),
                      "started_utc": runner.utc(), "requested_points": len(REQUESTED)}), flush=True)
    for d, p in REQUESTED:
        point = runner.name_for(d, p)
        checkpoint = directory / "checkpoints" / f"{point}.json"
        if all(row.get("timing_complete") for row in json.loads(checkpoint.read_text())):
            continue
        runner.save_status(directory, "timing_requested", point=point)
        runner.timing_point(directory, settings, runner.helper.get_profile(d), d, p)
    rows = runner.load_rows(directory)
    runner.helper.write_tables(directory, rows, settings)
    runner.save_status(directory, "timing_complete", accuracy_points=13,
                       requested_timing_points=len(REQUESTED), completed_rows=len(rows))
    scope["completed_utc"] = runner.utc()
    runner.helper.atomic_json(directory / "timing_scope.json", scope)
    print(json.dumps({"phase": "requested_timing_complete", "pid": os.getpid(),
                      "completed_utc": runner.utc()}), flush=True)


if __name__ == "__main__":
    main()

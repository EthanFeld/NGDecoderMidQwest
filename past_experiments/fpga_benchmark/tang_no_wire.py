"""Replay Tang UART requests through the RTL packet engine at one word per clock."""

import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("runner", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    design = json.loads(args.fixtures.read_text(encoding="utf-8"))["tang"]
    with tempfile.TemporaryDirectory(dir=args.output.resolve().parent) as folder:
        wire_path = Path(folder) / "requests.bin"
        with wire_path.open("wb") as wire:
            for cohort in design["cohorts"]:
                for shot in cohort["shots"]:
                    request = base64.b64decode(shot[0], validate=True)
                    if len(request) != 232:
                        raise ValueError("Tang UART request is not 232 bytes")
                    wire.write(request)
        output = subprocess.check_output([args.runner.resolve(), wire_path], text=True)

    timing_rows = iter(output.splitlines())
    runs = []
    for cohort in design["cohorts"]:
        rows = []
        mismatches = 0
        for shot in cohort["shots"]:
            result = [int(value) for value in next(timing_rows).split(",")]
            rows.append(result)
            if len(shot) > 4 and shot[4] is not None:
                mismatches += result[0] != shot[4]
        runs.append(
            {
                "p": cohort["p"],
                "seed": cohort["seed"],
                "rows": rows,
                "status_mismatches": mismatches,
            }
        )
    if next(timing_rows, None) is not None:
        raise ValueError("Simulator returned more results than input shots")

    result = {
        "clock_hz": design["clock_hz"],
        "row_schema": ["status", "cycles_from_first_word", "cycles_from_last_crc_word"],
        "runs": runs,
    }
    args.output.write_text(
        json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    for run in runs:
        print(
            f"p={run['p']:.4f}: {len(run['rows'])} valid RTL replies, {run['status_mismatches']} status mismatches"
        )


if __name__ == "__main__":
    main()

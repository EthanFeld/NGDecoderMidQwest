"""Command-line entry points; TOML defaults with explicit CLI overrides."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from .config import SweepConfig
from .runner import run_sweep


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Surface-code decoder distance sweeps")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="sample, decode, and report a parameter sweep")
    run.add_argument("--config", type=Path, help="flat TOML configuration file")
    run.add_argument("--resume", action="store_true", help="continue compatible committed batches")
    options = argparse.SUPPRESS
    run.add_argument("--distances", nargs="+", type=int, default=options)
    run.add_argument("--noise", nargs="+", type=float, default=options)
    run.add_argument("--bases", nargs="+", choices=["x", "z"], default=options)
    run.add_argument("--geometries", nargs="+", choices=["rotated", "unrotated"], default=options)
    run.add_argument("--rounds", default=options, help="distance (default), 2d, or a fixed integer")
    run.add_argument(
        "--noise-model", choices=["circuit", "phenomenological", "data"], default=options
    )
    run.add_argument(
        "--decoders", nargs="+", default=options, help="pymatching, zero, module:factory"
    )
    for option in (
        "shots",
        "min-shots",
        "max-errors",
        "batch-size",
        "warmup-shots",
        "workers",
        "seed",
    ):
        run.add_argument(f"--{option}", type=int, default=options)
    run.add_argument("--memory-mib", type=float, default=options, help="buffer budget per worker")
    run.add_argument("--output", default=options, help="new output directory")
    floor = commands.add_parser(
        "floor",
        parents=[run],
        add_help=False,
        help="bound the minimum achievable logical error rate for each experiment",
    )
    floor.add_argument(
        "--exact-bits",
        type=int,
        default=18,
        help="maximum compressed joint state bits for exact LER (0 disables)",
    )
    floor.add_argument(
        "--max-exact-work", type=int, default=100_000_000, help="cap mechanisms times joint states"
    )
    floor.add_argument(
        "--paths", type=int, default=32, help="maximum independent logical witnesses"
    )
    floor.add_argument(
        "--chain-bins",
        type=int,
        default=16384,
        help="quantized likelihood bins per logical witness",
    )
    floor.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="simultaneous upper-bound confidence across the sweep",
    )
    plot = commands.add_parser(
        "plot", help="plot error rate and decoder throughput versus distance"
    )
    plot.add_argument("results", type=Path, help="run directory, results.json, or results.csv")
    plot.add_argument("--output", type=Path, required=True, help="PNG, SVG, or PDF output path")
    floor_plot = commands.add_parser("plot-floor", help="plot optimal-LER uncertainty brackets")
    floor_plot.add_argument("results", type=Path, help="floor run directory, JSON, or CSV")
    floor_plot.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command in ("plot", "plot-floor"):
            if command == "plot":
                from .plotting import plot_results as plot_function
            else:
                from .floor_plotting import plot_floor as plot_function
            path = plot_function(args["results"], args["output"])
            print(f"Plot: {path}")
            return 0
        config_path = args.pop("config")
        resume = args.pop("resume")
        floor_options = None
        if command == "floor":
            floor_options = {
                key: args.pop(key)
                for key in ("exact_bits", "max_exact_work", "paths", "chain_bins")
            }
            confidence = args.pop("confidence")
        values = {}
        if config_path:
            with config_path.open("rb") as file:
                values = tomllib.load(file)
        values.update(args)
        if command == "floor":
            values.setdefault("output", "past_experiments/decoder_testing/results/ler-floor")
        config = SweepConfig.from_mapping(values)
        count = len(config.experiments())
        print(
            f"{count} experiments; {len(config.decoders)} decoders; "
            f"workers={min(config.workers, count)}"
        )

        def progress(rows: list[dict]) -> None:
            first = rows[0]
            rates = ", ".join(f"{r['decoder']}={r['errors']}/{r['shots']}" for r in rows)
            print(
                f"d={first['distance']} p={first['noise']:g} {first['geometry']} "
                f"memory-{first['basis']}: {rates}",
                flush=True,
            )

        if command == "floor":
            from .floor_runner import run_floor_sweep
            from .ler_floor import FloorOptions

            def bound_progress(row: dict) -> None:
                print(
                    f"Floor d={row['distance']} p={row['noise']:g} {row['basis']}: "
                    f"[{row['minimum_ler_lower']:.6g}, {row['minimum_ler_upper']:.6g}] "
                    f"({row['status']}, {row['logical_paths']} witnesses)",
                    flush=True,
                )

            rows = run_floor_sweep(
                config,
                resume=resume,
                progress=progress,
                bound_progress=bound_progress,
                options=FloorOptions(**floor_options, memory_mib=config.memory_mib),
                confidence=confidence,
            )
            print(f"{len(rows)} floor rows: {config.output_path / 'ler_floor.csv'}")
            if any(not row["confidence_consistent"] for row in rows):
                print(
                    "Confidence conflict: inspect analytic lower and measured upper bounds.",
                    file=sys.stderr,
                )
                return 2
            return 0
        rows = run_sweep(config, resume=resume, progress=progress)
        print(f"{len(rows)} result rows: {config.output_path / 'results.csv'}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; committed batches can be resumed.", file=sys.stderr)
        return 130
    except (ValueError, OSError, ImportError, AttributeError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        cause = error.__cause__
        while cause is not None:
            print(f"  Caused by: {cause}", file=sys.stderr)
            cause = cause.__cause__
        return 1

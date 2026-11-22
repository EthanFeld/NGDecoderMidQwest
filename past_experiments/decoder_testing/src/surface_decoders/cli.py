"""Read-only resource planning over explicitly requested experiment sizes."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, replace

from qecbench import DecoderContext, Experiment
from qecbench.config import resolve_rounds

from .ibm_relay_bp import (
    FAST_CONFIG,
    PAPER_S1_CONFIG,
    PAPER_S5_CONFIG,
    TUNED_CONFIG,
    IBMRelayBPConfig,
    estimate_ibm_resources,
)
from .resources import LUTConfig, RelayBPConfig, ResourceLimits, estimate_resources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Surface decoder resource preflight (no LUT builds)"
    )
    parser.add_argument("command", choices=["estimate"])
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--rounds", default="distance")
    parser.add_argument("--noise", nargs="+", type=float, default=[0.001])
    parser.add_argument("--bases", nargs="+", choices=["x", "z"], default=["z"])
    parser.add_argument(
        "--geometries", nargs="+", choices=["rotated", "unrotated"], default=["rotated"]
    )
    parser.add_argument(
        "--noise-model", choices=["circuit", "phenomenological", "data"], default="circuit"
    )
    parser.add_argument(
        "--decoders",
        nargs="+",
        choices=["relay_bp", "ibm_relay_bp", "lut"],
        default=["relay_bp", "lut"],
    )
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument(
        "--ibm-fast",
        action="store_true",
        help="estimate the native eight-leg/30-iteration comparison profile",
    )
    profiles.add_argument(
        "--ibm-tuned", action="store_true", help="estimate the frozen memory-basis profile"
    )
    profiles.add_argument("--ibm-paper-s1", action="store_true", help="301-leg full XYZ profile")
    profiles.add_argument("--ibm-paper-s5", action="store_true", help="601-leg full XYZ profile")
    parser.add_argument("--memory-mib", type=float, default=256)
    parser.add_argument("--latency-us", type=float)
    parser.add_argument("--operations-per-second", type=float)
    parser.add_argument("--max-build-operations", type=int, default=200_000_000)
    parser.add_argument("--max-model-mechanisms", type=int, default=1_000_000)
    parser.add_argument("--max-detectors", type=int, default=100_000)
    parser.add_argument("--legs", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--solutions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--chunk-shots", type=int, help="override each decoder's default chunk size"
    )
    args = parser.parse_args(argv)
    try:
        limits = ResourceLimits(
            args.memory_mib,
            args.latency_us,
            args.operations_per_second,
            args.max_build_operations,
            args.max_model_mechanisms,
            args.max_detectors,
        )
        relay = RelayBPConfig(
            legs=args.legs,
            iterations=args.iterations,
            solutions=args.solutions,
            seed=args.seed,
            chunk_shots=32 if args.chunk_shots is None else args.chunk_shots,
        )
        lut = LUTConfig(chunk_shots=4096 if args.chunk_shots is None else args.chunk_shots)
        ibm = replace(
            PAPER_S5_CONFIG
            if args.ibm_paper_s5
            else PAPER_S1_CONFIG
            if args.ibm_paper_s1
            else TUNED_CONFIG
            if args.ibm_tuned
            else FAST_CONFIG
            if args.ibm_fast
            else IBMRelayBPConfig(),
            seed=args.seed,
            chunk_shots=32 if args.chunk_shots is None else args.chunk_shots,
        )
        # Validate all requested points before generating even the first circuit.
        experiments = [
            Experiment(d, resolve_rounds(args.rounds, d), p, b, g, args.noise_model)
            for d, p, b, g in itertools.product(
                args.distances, args.noise, args.bases, args.geometries
            )
        ]
        rows = []
        for experiment in experiments:
            d = experiment.distance
            checks = d * d - 1 if experiment.geometry == "rotated" else 2 * d * (d - 1)
            # Conservative bound prevents giant circuit/DEM generation in this pathway.
            bound = checks * (experiment.rounds + 1)
            if bound > limits.max_detectors:
                for decoder in args.decoders:
                    rows.append(
                        {
                            "experiment": asdict(experiment),
                            "decoder": decoder,
                            "accepted": False,
                            "reasons": [
                                f"detector upper bound {bound} exceeds generation cap "
                                f"{limits.max_detectors}"
                            ],
                        }
                    )
                continue
            circuit = experiment.circuit()
            context = DecoderContext(
                experiment, circuit, circuit.detector_error_model(decompose_errors=True)
            )
            for decoder in args.decoders:
                estimate = (
                    estimate_ibm_resources(context, ibm, limits)
                    if decoder == "ibm_relay_bp"
                    else estimate_resources(
                        context, decoder, relay if decoder == "relay_bp" else lut, limits
                    )
                )
                rows.append({"experiment": asdict(experiment), **estimate.to_dict()})
        print(
            json.dumps(
                {
                    "limits": asdict(limits),
                    "relay_bp_config": asdict(relay),
                    "ibm_relay_bp_config": asdict(ibm),
                    "ibm_search_budget": {
                        "total_legs": ibm.total_legs,
                        "maximum_iterations": ibm.max_iterations,
                        "stopping_criterion": ibm.stopping_criterion,
                    },
                    "lut_config": asdict(lut),
                    "estimates": rows,
                    "latency_model": (
                        "worst-case operation count / supplied platform rate; "
                        "not a wall-clock guarantee"
                    ),
                    "memory_scope": (
                        "per decoder/worker; excludes harness circuit, DEM, "
                        "batches and Python runtime"
                    ),
                },
                indent=2,
                allow_nan=False,
            )
        )
        return 0 if all(row["accepted"] for row in rows) else 2
    except (ValueError, OverflowError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

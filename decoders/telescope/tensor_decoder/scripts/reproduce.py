"""Paired current-decoder/CM study; all sampled shots contribute to LER."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from tensor_decoder import (
    DecoderConfig,
    DecoderContext,
    Experiment,
    build_decoder,
    qecbench_implementation_identity,
)
from tensor_decoder.baseline import CorrelatedMatching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots-per-seed", type=int, default=512)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[2360427, 2460427, 2560427, 2660427]
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("results/runs/reproduction.json"))
    args = parser.parse_args()
    if min(args.shots_per_seed, args.batch_size) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive counts and distinct seeds are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    e = Experiment(15, 15, 0.005)
    c = e.circuit()
    ctx = DecoderContext(e, c, c.detector_error_model(decompose_errors=True))
    decoder, baseline = build_decoder(ctx), CorrelatedMatching(ctx)
    report = {
        "complete": False,
        "shots": 0,
        "target_shots": args.shots_per_seed * len(args.seeds),
        "config": asdict(DecoderConfig()),
        "experiment": asdict(e),
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "identity": qecbench_implementation_identity(),
        "decoder_errors": 0,
        "correlated_matching_errors": 0,
        "fixes": 0,
        "regressions": 0,
        "streams": [],
    }
    start_time = time.perf_counter()

    def save():
        report["seconds"] = time.perf_counter() - start_time
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for seed in args.seeds:
        sampler = c.compile_detector_sampler(seed=seed)
        digest = hashlib.sha256()
        for start in range(0, args.shots_per_seed, args.batch_size):
            events, truth = sampler.sample(
                min(args.batch_size, args.shots_per_seed - start),
                separate_observables=True,
                bit_packed=True,
            )
            events.flags.writeable = False
            digest.update(events.tobytes())
            prediction = decoder.decode_batch(events)
            reference = baseline.decode_batch(events)
            bad, ref_bad = prediction[:, 0] != truth[:, 0], reference[:, 0] != truth[:, 0]
            report["decoder_errors"] += int(bad.sum())
            report["correlated_matching_errors"] += int(ref_bad.sum())
            report["fixes"] += int(np.count_nonzero(ref_bad & ~bad))
            report["regressions"] += int(np.count_nonzero(~ref_bad & bad))
            report["shots"] += len(events)
            save()
            print(
                seed,
                start + len(events),
                report["decoder_errors"],
                report["correlated_matching_errors"],
                flush=True,
            )
        report["streams"].append(
            {"seed": seed, "shots": args.shots_per_seed, "events_sha256": digest.hexdigest()}
        )
    report["complete"] = True
    report["ler"] = report["decoder_errors"] / report["shots"]
    total = report["fixes"] + report["regressions"]
    report["paired_exact_p"] = (
        float(binomtest(report["fixes"], total, 0.5).pvalue) if total else None
    )
    save()


if __name__ == "__main__":
    main()

"""Measure completed single-shot calls with rotating decoder order."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensor_decoder import (
    DecoderContext,
    Experiment,
    build_decoder,
    qecbench_implementation_identity,
)
from tensor_decoder.baseline import CorrelatedMatching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2860427)
    parser.add_argument("--output", type=Path, default=Path("results/runs/latency.json"))
    args = parser.parse_args()
    if min(args.shots, args.repeats) < 1:
        parser.error("positive counts are required")
    e = Experiment(15, 15, 0.005)
    c = e.circuit()
    ctx = DecoderContext(e, c, c.detector_error_model(decompose_errors=True))
    functions = {
        "decoder": build_decoder(ctx).decode_batch,
        "correlated_matching": CorrelatedMatching(ctx).decode_batch,
    }
    probes = c.compile_detector_sampler(seed=args.seed).sample(args.shots, bit_packed=True)
    probes.flags.writeable = False
    for fn in functions.values():
        fn(probes[:32])
    samples = {name: [] for name in functions}
    names = list(functions)
    for repeat in range(args.repeats):
        for i in range(len(probes)):
            for k in range(len(names)):
                name = names[(i + repeat + k) % len(names)]
                before = time.perf_counter_ns()
                functions[name](probes[i : i + 1])
                samples[name].append((time.perf_counter_ns() - before) / 1000)
    report = {
        "seed": args.seed,
        "unique_probes": args.shots,
        "repeats": args.repeats,
        "identity": qecbench_implementation_identity(),
        "samples_us": samples,
        "latency": {
            name: {
                "median_us": float(np.median(v)),
                "p99_us": float(np.percentile(v, 99)),
                "p999_us": float(np.percentile(v, 99.9)),
            }
            for name, v in samples.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["latency"], indent=2))


if __name__ == "__main__":
    main()

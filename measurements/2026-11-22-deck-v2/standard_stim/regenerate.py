"""Fresh paired d15 standard-Stim accuracy study with per-batch checkpoints."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PACKAGE = ROOT / "decoders/pymatching_telescope/src"
sys.path.insert(0, str(PACKAGE))

import numpy as np
import pymatching
import scipy
import stim
from pymatching_tensor_decoder import (
    DecoderConfig, DecoderContext, Experiment, build_decoder, qecbench_implementation_identity,
)
from pymatching_tensor_decoder.baseline import CorrelatedMatching
from scipy.stats import binomtest

SEEDS = (94628701, 94628702, 94628703, 94628704)
SHOTS_PER_SEED = 100_000
BATCH_SIZE = 4096
PAUSE = OUT / "PAUSE_FOR_TIMING"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def interval(errors, shots):
    if not shots:
        return None
    z = 1.959963984540054
    p = errors / shots
    denominator = 1 + z*z/shots
    center = (p + z*z/(2*shots)) / denominator
    radius = z * math.sqrt(p*(1-p)/shots + z*z/(4*shots*shots)) / denominator
    return [center-radius, center+radius]


def worker(seed):
    start_time = time.perf_counter()
    path = OUT / f"seed_{seed}.json"
    experiment = Experiment(15, 15, .005)
    circuit = experiment.circuit()
    context = DecoderContext(experiment, circuit, circuit.detector_error_model(decompose_errors=True))
    decoder, baseline = build_decoder(context), CorrelatedMatching(context)
    sampler = circuit.compile_detector_sampler(seed=seed)
    record = {
        "seed": seed, "pid": os.getpid(), "complete": False, "paused": False,
        "shots": 0, "target_shots": SHOTS_PER_SEED, "batch_size": BATCH_SIZE,
        "decoder_errors": 0, "correlated_matching_errors": 0, "fixes": 0,
        "regressions": 0, "both_wrong": 0, "chunks": [],
    }
    events_hash, truth_hash = sha256(), sha256()
    try:
        write_json(path, record)
        for start in range(0, SHOTS_PER_SEED, BATCH_SIZE):
            if PAUSE.exists():
                record["paused"] = True
                write_json(path, record)
                while PAUSE.exists():
                    time.sleep(1)
                record["paused"] = False
                write_json(path, record)
            count = min(BATCH_SIZE, SHOTS_PER_SEED-start)
            events, truth = sampler.sample(count, separate_observables=True, bit_packed=True)
            events.flags.writeable = False
            events_hash.update(events.tobytes())
            truth_hash.update(truth.tobytes())
            prediction = decoder.decode_batch(events)
            reference = baseline.decode_batch(events)
            bad, ref_bad = prediction[:, 0] != truth[:, 0], reference[:, 0] != truth[:, 0]
            counts = {
                "decoder_errors": int(bad.sum()),
                "correlated_matching_errors": int(ref_bad.sum()),
                "fixes": int(np.count_nonzero(ref_bad & ~bad)),
                "regressions": int(np.count_nonzero(~ref_bad & bad)),
                "both_wrong": int(np.count_nonzero(ref_bad & bad)),
            }
            for key, value in counts.items():
                record[key] += value
            record["shots"] += count
            record["chunks"].append({"start": start, "shots": count, **counts})
            record["events_sha256"] = events_hash.hexdigest()
            record["truth_sha256"] = truth_hash.hexdigest()
            record["seconds"] = time.perf_counter()-start_time
            write_json(path, record)
            print(f"seed={seed} shots={record['shots']} tensor_errors={record['decoder_errors']} correlated_errors={record['correlated_matching_errors']}", flush=True)
        record["complete"] = True
        write_json(path, record)
    except BaseException as error:
        record["failure"] = repr(error)
        write_json(path, record)
        raise
    finally:
        decoder.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "comparison.json"
    if output.exists() or any((OUT / f"seed_{seed}.json").exists() for seed in SEEDS):
        raise FileExistsError("Fresh study output already exists; do not overwrite")
    package = PACKAGE / "pymatching_tensor_decoder"
    report = {
        "complete": False, "factory": "pymatching_tensor_decoder:build_decoder",
        "experiment": asdict(Experiment(15, 15, .005)), "config": asdict(DecoderConfig()),
        "circuit": "stim.Circuit.generated surface_code:rotated_memory_z",
        "seeds": list(SEEDS), "shots_per_seed": SHOTS_PER_SEED,
        "target_shots": len(SEEDS)*SHOTS_PER_SEED, "batch_size": BATCH_SIZE,
        "identity": qecbench_implementation_identity(),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
                          for p in sorted(package.rglob("*.py"))},
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "stim": stim.__version__,
                     "pymatching": pymatching.__version__, "scipy": scipy.__version__},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        "interpretation": "Per complete memory experiment; accuracy study elapsed time is not decoder latency",
    }
    start_time = time.perf_counter()
    spawn = mp.get_context("spawn")
    processes = [spawn.Process(target=worker, args=(seed,)) for seed in SEEDS]
    for process in processes:
        process.start()
    report["coordinator_pid"] = os.getpid()
    report["worker_pids"] = [p.pid for p in processes]
    print(json.dumps({"coordinator_pid": os.getpid(), "worker_pids": report["worker_pids"],
                      "output": str(output), "pause_file": str(PAUSE)}), flush=True)
    while True:
        streams = []
        for seed in SEEDS:
            path = OUT / f"seed_{seed}.json"
            if path.exists():
                streams.append(json.loads(path.read_text()))
        report["streams"] = streams
        for key in ("shots", "decoder_errors", "correlated_matching_errors", "fixes", "regressions", "both_wrong"):
            report[key] = sum(stream[key] for stream in streams)
        report["seconds"] = time.perf_counter()-start_time
        report["complete"] = len(streams) == len(SEEDS) and all(s["complete"] for s in streams)
        if report["shots"]:
            report["ler"] = report["decoder_errors"]/report["shots"]
            report["correlated_matching_ler"] = report["correlated_matching_errors"]/report["shots"]
            report["ler_wilson95"] = interval(report["decoder_errors"], report["shots"])
            report["correlated_matching_ler_wilson95"] = interval(report["correlated_matching_errors"], report["shots"])
        write_json(output, report)
        if not any(process.is_alive() for process in processes):
            break
        time.sleep(2)
    for process in processes:
        process.join()
    report["worker_exit_codes"] = [p.exitcode for p in processes]
    total = report["fixes"]+report["regressions"]
    report["paired_exact_p"] = float(binomtest(report["fixes"], total, .5).pvalue) if total else None
    write_json(output, report)
    if not report["complete"] or any(p.exitcode for p in processes):
        raise RuntimeError("A standard-circuit worker did not complete")
    print(json.dumps({key: report[key] for key in ("complete", "shots", "decoder_errors", "correlated_matching_errors", "fixes", "regressions", "ler", "seconds")}), flush=True)


if __name__ == "__main__":
    main()

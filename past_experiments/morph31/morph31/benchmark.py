"""Bounded, cached graph checks, paired accuracy tests, and operation estimates."""

import argparse
import json
import math
import platform
import sys
import time
import tomllib
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import pymatching
import stim

from .circuit import Experiment, load_experiment, sample_dataset
from .compression import CompressedGraph, compile_graph, verify_equivalence
from .cost_model import CycleModel
from .decoder import SparseDecoder
from .dem import parse_dem
from .geometry import LatticeCodec
from .graph import ExplicitGraph
from .motif import FEATURE_VERSION, NEIGHBORS12, MotifPolicy
from .sparse_memory import Limits
from .training import train_policy, training_metadata


def source_fingerprint(names=None):
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py")) if names is None else [root / n for n in names]
    digest = sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_graph(experiment, cache, verify=False):
    circuit, dem, folder = load_experiment(experiment, cache)
    version = source_fingerprint(["graph.py", "dem.py", "geometry.py", "compression.py"])[:12]
    explicit_path, rom_path = folder / f"graph-{version}.npz", folder / f"graph-{version}.rom"
    if explicit_path.exists() and rom_path.exists():
        reference, compressed = ExplicitGraph.load(explicit_path), CompressedGraph.load(rom_path)
    else:
        reference = parse_dem(dem)
        compressed = compile_graph(reference, LatticeCodec(
            experiment.distance, experiment.rounds, experiment.basis))
        verify_equivalence(reference, compressed)
        reference.save(explicit_path)
        compressed.save(rom_path)
    if verify:
        verify_equivalence(reference, compressed)
    return circuit, dem, reference, compressed, folder


def wilson(errors, shots):
    if shots <= 0 or not 0 <= errors <= shots:
        raise ValueError("invalid error count")
    z = 1.959963984540054
    p, z2 = errors / shots, z**2
    center = (p + z2 / (2 * shots)) / (1 + z2 / shots)
    half = z * math.sqrt(p * (1 - p) / shots + z2 / (4 * shots**2)) / (1 + z2 / shots)
    return [max(0, center - half), min(1, center + half)]


def accuracy(predictions, truth):
    failed = np.any(predictions != truth, axis=1)
    errors, shots = int(failed.sum()), len(failed)
    return {"shots": shots, "logical_errors": errors, "logical_error_rate": errors / shots,
            "ci95_wilson": wilson(errors, shots)}


def paired_accuracy(candidate, reference, truth):
    candidate_failed = np.any(candidate != truth, axis=1)
    reference_failed = np.any(reference != truth, axis=1)
    added = int(np.sum(candidate_failed & ~reference_failed))
    removed = int(np.sum(~candidate_failed & reference_failed))
    discordant = added + removed
    # Exact two-sided sign/McNemar test on discordant logical outcomes.
    probability = (min(1.0, 2 * sum(math.comb(discordant, k)
                                   for k in range(min(added, removed) + 1)) / 2**discordant)
                   if discordant else 1.0)
    return {"added_failures": added, "removed_failures": removed,
            "both_failed": int(np.sum(candidate_failed & reference_failed)),
            "logical_error_rate_difference": (added - removed) / len(truth),
            "added_failure_rate_ci95_wilson": wilson(added, len(truth)),
            "mcnemar_exact_two_sided_p": probability,
            "interpretation": "A large p-value does not establish non-inferiority; "
                              "zero added failures still has a positive confidence bound."}


def paired_operation_reduction(baseline, candidate):
    baseline, candidate = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    ratio = float(np.mean(candidate) / np.mean(baseline))
    reduction = 1 - ratio
    if len(baseline) < 2:
        interval = None
    else:
        # Paired delta-method interval for the ratio of means, not the mean
        # per-shot ratio. Correlated input difficulty cancels in the residual.
        se = float(np.std(candidate - ratio * baseline, ddof=1)
                   / (math.sqrt(len(baseline)) * np.mean(baseline)))
        interval = [reduction - 1.959963984540054 * se,
                    reduction + 1.959963984540054 * se]
    return {"operation_reduction_fraction": reduction,
            "operation_reduction_ci95_paired_delta": interval,
            "operation_interval_note": "Approximate paired delta-method interval on fixed-size "
                                       "samples; not a worst-case or distribution-free bound."}


def environment():
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "stim": stim.__version__,
            "pymatching": pymatching.__version__, "source_sha256": source_fingerprint()}


def run_benchmark(experiment, cache, *, shots=100, seed=3101, policy=None,
                  limits=None, max_seconds=45.0, verify=False, progress_every=0):
    if max_seconds <= 0:
        raise ValueError("experiment time budget must be positive")
    start = time.perf_counter()
    circuit, dem, reference, graph, folder = prepare_graph(experiment, cache, verify)
    prepared = time.perf_counter()
    detections, truth = sample_dataset(circuit, folder, shots, seed)
    matching = pymatching.Matching.from_detector_error_model(dem)
    t = time.perf_counter()
    reference_predictions = matching.decode_batch(detections)
    matching_seconds = time.perf_counter() - t
    decoders = {"conventional": SparseDecoder(graph, limits)}
    if policy is not None:
        decoders["motif"] = SparseDecoder(graph, limits, policy)
    predictions = {name: [] for name in decoders}
    results = {name: [] for name in decoders}
    elapsed = {name: 0.0 for name in decoders}
    completed = 0
    for syndrome in detections:
        if time.perf_counter() - start > max_seconds:
            break
        for name, decoder in decoders.items():
            t = time.perf_counter()
            result = decoder.decode(syndrome)
            elapsed[name] += time.perf_counter() - t
            # Each correction has already passed syndrome validation. Large
            # comparisons retain its logical result/counters, not every Edge.
            result.correction = ()
            results[name].append(result)
            predictions[name].append([(result.logical_mask >> i) & 1
                                      for i in range(graph.num_observables)])
        completed += 1
        if progress_every and completed % progress_every == 0:
            print(json.dumps({"basis": experiment.basis, "noise": experiment.noise,
                              "completed_shots": completed, "requested_shots": shots,
                              "elapsed_seconds": time.perf_counter() - start}), flush=True)
    if not completed:
        raise TimeoutError("experiment budget exhausted before any paired shot completed")
    true = truth[:completed]
    ref = reference_predictions[:completed]
    report = {
        "experiment": asdict(experiment), "seed": seed, "requested_shots": shots,
        "completed_shots": completed, "bounded_run_truncated": completed != shots,
        "environment": environment(), "cache_key": experiment.key,
        "sample_sha256": sha256(detections[:completed].tobytes() + true.tobytes()).hexdigest(),
        "num_detectors": dem.num_detectors, "num_observables": dem.num_observables,
        "data_qubits": experiment.distance**2,
        "syndrome_ancillas": experiment.distance**2 - 1,
        "stim_num_qubits_including_index_gaps": circuit.num_qubits,
        "graph": graph.statistics(reference), "projection": reference.metadata,
        "prepare_seconds": prepared - start,
        "reference": {**accuracy(ref, true), "decoder": "PyMatching, uncorrelated DEM mode",
                      "decode_seconds_full_requested_batch": matching_seconds},
        "decoders": {},
        "accuracy_note": "Full-circuit samples; small pilot intervals do not establish "
                         "accuracy improvements or equivalence. Conventional is unweighted.",
    }
    for name, records in results.items():
        predicted = np.array(predictions[name], dtype=np.uint8)
        operations = [r.operations for r in records]
        model_bytes = 4096 if name == "motif" else 0
        working = max(r.memory["peak_working_bytes"] for r in records)
        report["decoders"][name] = {
            **accuracy(predicted, true),
            "disagreements_with_pymatching": int(np.any(predicted != ref, axis=1).sum()),
            "paired_vs_pymatching": paired_accuracy(predicted, ref, true),
            "decode_seconds": elapsed[name],
            "mean_operations": float(np.mean([o.total for o in operations])),
            "mean_operation_counts": {key: float(np.mean([o.report()[key] for o in operations]))
                                      for key in operations[0].report()},
            "peak_working_bytes": working, "graph_rom_bytes": len(graph.to_bytes()),
            "model_rom_bytes": model_bytes,
            "peak_total_packed_bytes": working + len(graph.to_bytes()) + model_bytes,
            "peak_records": {key: max(r.memory["peak_records"].get(key, 0) for r in records)
                             for key in records[0].memory["peak_records"]},
            "mean_ml_accepted": float(np.mean([r.ml_accepted for r in records])),
            "cycles": CycleModel().summarize(operations, experiment.rounds),
            "limits": asdict(limits or Limits()),
            "memory_note": "Analytical packed layout, not measured Python heap; growth storage "
                           "is reused in recovery. Static ROMs are additional to working memory.",
        }
    if policy is not None:
        conventional = np.array(predictions["conventional"])
        motif = np.array(predictions["motif"])
        report["motif_comparison"] = {
            **paired_operation_reduction([r.operations.total for r in results["conventional"]],
                                         [r.operations.total for r in results["motif"]]),
            **paired_accuracy(motif, conventional, true),
            "default": "conventional",
            "decision": "Keep conventional default; this pilot alone cannot establish "
                        "accuracy preservation. Policy remains explicitly opt-in.",
            "policy_metadata": policy.metadata,
            "policy_sha256": sha256(policy.table.tobytes()).hexdigest(),
        }
    report["wall_seconds"] = time.perf_counter() - start
    return report


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("graph", "benchmark", "train"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--distance", type=int)
        command.add_argument("--rounds", type=int)
        command.add_argument("--basis", choices=["x", "z"])
        command.add_argument("--noise", type=float)
        command.add_argument("--cache", type=Path)
        command.add_argument("--output", type=Path)
        command.add_argument("--max-seconds", type=float)
        if name == "graph":
            command.add_argument("--rom", type=Path)
        else:
            command.add_argument("--seed", type=int)
            command.add_argument("--shots", type=int)
            command.add_argument("--working-kib", type=int, default=64)
            command.add_argument("--max-expansions", type=int)
        if name == "benchmark":
            command.add_argument("--policy", type=Path)
            command.add_argument("--verify", action="store_true")
            command.add_argument("--progress-every", type=int, default=0)
        if name == "train":
            command.add_argument("--train-shots", type=int, default=512)
            command.add_argument("--min-support", type=int, default=3)
            command.add_argument("--confidence", type=float, default=0.95)
            command.add_argument("--rom", type=Path, default=Path("results/motif.rom"))
            command.add_argument("--features", choices=["local8", "neighbors12"],
                                 default="local8")
    args = parser.parse_args(argv)
    config = {}
    if args.config:
        with args.config.open("rb") as handle:
            config = tomllib.load(handle)

    def option(name, default):
        value = getattr(args, name, None)
        return config.get(name, default) if value is None else value

    distance = option("distance", 5)
    experiment = Experiment(distance, option("rounds", distance), option("basis", "x"),
                            option("noise", 0.001))
    cache = option("cache", "data/cache")
    output = option("output", f"results/{args.command}.json")
    budget = option("max_seconds", 45.0)
    if args.command == "graph":
        start = time.perf_counter()
        _, _, reference, graph, _ = prepare_graph(experiment, cache, verify=True)
        # Verify the exact binary that will be consumed, including incidence.
        verify_equivalence(reference, CompressedGraph.from_bytes(graph.to_bytes()))
        if args.rom:
            args.rom.parent.mkdir(parents=True, exist_ok=True)
            graph.save(args.rom)
        report = {"experiment": asdict(experiment), "environment": environment(),
                  "num_detectors": graph.num_detectors, "graph": graph.statistics(reference),
                  "projection": reference.metadata, "equivalence_verified": True,
                  "rom_roundtrip_verified": True, "wall_seconds": time.perf_counter() - start}
        if report["wall_seconds"] > budget:
            report["time_budget_exceeded"] = True
    else:
        limits = Limits(working_bytes=args.working_kib * 1024,
                        max_expansions=option("max_expansions", 100000),
                        max_seconds=min(budget, 10.0))
        shots, seed = option("shots", 100), option("seed", 3101)
        policy = None
        if args.command == "train":
            circuit, dem, _, graph, folder = prepare_graph(experiment, cache)
            training, _ = sample_dataset(circuit, folder, args.train_shots, seed)
            policy = train_policy(graph, dem, training, min_support=args.min_support,
                                  min_confidence=args.confidence,
                                  metadata=training_metadata(experiment, seed), max_seconds=budget,
                                  feature_version=(NEIGHBORS12 if args.features == "neighbors12"
                                                   else FEATURE_VERSION))
            policy.save(args.rom)
            seed += 1  # Held-out circuit sample cache, never the training stream.
        elif args.policy:
            policy = MotifPolicy.load(args.policy)
        report = run_benchmark(experiment, cache, shots=shots, seed=seed, policy=policy,
                               limits=limits, max_seconds=budget,
                               verify=getattr(args, "verify", False),
                               progress_every=getattr(args, "progress_every", 0))
    _write_report(output, report)
    if args.command == "graph":
        print(json.dumps({"output": str(output), **report["graph"]}, indent=2))
    else:
        print(json.dumps({"output": str(output), "completed_shots": report["completed_shots"],
                          "decoders": {k: {n: v[n] for n in (
                              "logical_errors", "mean_operations", "peak_working_bytes")}
                                       for k, v in report["decoders"].items()},
                          "motif_comparison": report.get("motif_comparison")}, indent=2))
    return 0

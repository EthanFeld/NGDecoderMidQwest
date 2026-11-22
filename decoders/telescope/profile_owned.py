"""Small owned-matcher development traces and exact packet replay; no unit tests.

This helper never imports an outside decoder. Capture uses a frozen baseline
Python/native snapshot. Replay uses the current owned implementation on the same
packets, preserving graph caches and references, without sampling new shots.
"""

from __future__ import annotations

import argparse
import cProfile
import ctypes as ct
import gc
import hashlib
import importlib.abc
import importlib.util
import json
import os
import pstats
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter_ns

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

SURFACE = Path(__file__).resolve().parent
ROOT = SURFACE.parents[1]
sys.path.insert(0, str(SURFACE / "tensor_decoder/src"))
PROFILE_FIELDS = (
    "setup", "certificate", "metric", "preferred", "matching", "trace", "hint", "shortest",
    "matching_build", "matching_solve", "matching_price", "matching_prepare",
)


class BlockOutsideDecoders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"pymatching", "networkx", "fusion_blossom"}:
            raise RuntimeError(f"outside decoder import is forbidden: {fullname}")
        return None


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def native_profile(graph):
    try:
        function = graph.lib.owned_profile
    except AttributeError:
        return None
    function.argtypes = [ct.c_void_p, ct.c_void_p]
    function.restype = None
    count_function = graph.lib.owned_profile_count
    count_function.argtypes = []
    count_function.restype = ct.c_int
    count = count_function()
    if not 8 <= count <= 64:
        raise RuntimeError("unsupported native profiling counter count")
    values = np.zeros(count, dtype=np.int64)
    function(graph.handle, values.ctypes.data)
    return {name: int(values[i]) for i, name in enumerate(PROFILE_FIELDS[:count])}


def enable_lazy(graph):
    function = graph.lib.owned_lazy_metric
    function.argtypes = [ct.c_void_p, ct.c_int]
    function.restype = None
    function(graph.handle, 1)


def set_phase_pricing(graph, enabled):
    function = graph.lib.owned_phase_pricing
    function.argtypes = [ct.c_void_p, ct.c_int]
    function.restype = None
    function(graph.handle, int(enabled))


def set_seed_metric(graph, enabled):
    function = graph.lib.owned_seed_metric
    function.argtypes = [ct.c_void_p, ct.c_int]
    function.restype = None
    function(graph.handle, int(enabled))


def certificate_rejections(graph):
    try:
        function = graph.lib.owned_certificate_rejections
    except AttributeError:
        return 0
    function.argtypes = [ct.c_void_p]
    function.restype = ct.c_int64
    return int(function(graph.handle))


def library_identity(backend):
    path = Path(backend._library()._name)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def path_worker_counts(decoder):
    result = {}
    for name, graph in decoder_graphs(decoder).items():
        if name.startswith("updated_"):
            value = getattr(graph, "path_worker_count", None)
            result[name] = int(value() if callable(value) else value) if value is not None else None
    return result


def basis_worker_count(decoder):
    model = decoder.owned_engine.correlated_model
    value = getattr(model, "basis_worker_count", None)
    if value is None:
        value = getattr(model, "basis_workers", None)
    return int(value() if callable(value) else value) if value is not None else None


class CorrelationComparison:
    """Observe model outputs after decoding; expected data never enters a solve."""

    def __init__(self, model, trace, saved, phase):
        if not callable(getattr(model, "last_correlated_state", None)):
            raise RuntimeError("the correlated backend has no diagnostic snapshot getter")
        self.model, self.saved, self.phase = model, saved, phase
        self.original = model.decode_to_edges_array
        self.index, self.differences = 0, []
        graphs = {row["id"]: row for row in trace["graphs"]}
        self.expected = {}
        for packet in trace["packets"]:
            role = graphs[packet["graph"]]["role"]
            if packet["phase"] == phase and packet["kind"] == "solve" and role.startswith(
                "correlated_",
            ):
                self.expected.setdefault(packet["shot"], {})[role] = packet["prefix"]
        pairs = []
        for role, indices in (
            ("correlated_prior_memory", model.memory.indices),
            ("correlated_prior_other", model.other.indices),
        ):
            graph, = [row for row in trace["graphs"] if row["role"] == role]
            u, v = saved[graph["name"] + "_u"], saved[graph["name"] + "_v"]
            rows = np.full((len(u), 2), -1, np.int64)
            rows[:, 0] = indices[u]
            rows[v >= 0, 1] = indices[v[v >= 0]]
            pairs.append(rows)
        self.pairs = np.concatenate(pairs)

    def __enter__(self):
        self.model.decode_to_edges_array = self.decode
        return self

    def __exit__(self, *exception):
        self.model.decode_to_edges_array = self.original

    def decode(self, row, *args, **kwargs):
        result = self.original(row, *args, **kwargs)
        index, self.index = self.index, self.index + 1
        prefixes = self.expected[index]
        prior, weights, final = [], [], []
        for basis in ("memory", "other"):
            first = prefixes["correlated_prior_" + basis]
            second = prefixes.get("correlated_updated_" + basis, first)
            prior.append(self.saved[first + "_chosen"])
            weights.append(self.saved[second + "_weights"])
            final.append(self.saved[second + "_chosen"])
        first_expected, weights_expected = np.concatenate(prior), np.concatenate(weights)
        pairs_expected = self.pairs[np.flatnonzero(np.concatenate(final))]
        first, updated = self.model.last_correlated_state()
        checks = {
            "first_pass": np.array_equal(first, first_expected),
            "updated_weight_bits": np.array_equal(
                np.asarray(updated, np.float64).view(np.uint64), weights_expected.view(np.uint64),
            ),
            "final_edge_pairs": np.array_equal(result, pairs_expected),
        }
        for key, agrees in checks.items():
            if not agrees:
                self.differences.append({"shot": index, "component": key})
        return result

    def summary(self):
        return {"rows_checked": self.index, "rows_expected": len(self.expected),
                "differences": self.differences,
                "first_pass_weights_and_edges_exact": not self.differences
                and self.index == len(self.expected)}


def decode_diagnostic_phase(decoder, events, saved, mapping, phase, trace, compare):
    observer = CorrelationComparison(
        decoder.owned_engine.correlated_model, trace, saved, phase,
    ) if compare else None
    with observer if observer is not None else nullcontext():
        if phase == "raw":
            prediction, details = decoder.decode_with_diagnostics(events)
        else:
            original_batch = decoder.decode_batch
            decoder.decode_batch = decoder.decode_with_diagnostics
            try:
                prediction, details = decoder.decode_with_front(
                    events, saved["front_residual"], saved["front_logical"],
                    saved["front_valid"], mapping,
                )
            finally:
                decoder.decode_batch = original_batch
    return prediction, details, observer.summary() if observer is not None else None


def profile_summary(profiler):
    stats = pstats.Stats(profiler).stats
    rows = [
        {"file": file, "line": line, "function": function, "calls": nc,
         "primitive_calls": cc, "self_seconds": tt, "cumulative_seconds": cum}
        for (file, line, function), (cc, nc, tt, cum, _) in stats.items()
    ]
    return {
        "by_self": sorted(rows, key=lambda row: row["self_seconds"], reverse=True)[:40],
        "by_cumulative": sorted(rows, key=lambda row: row["cumulative_seconds"], reverse=True)[:40],
    }


def frozen_backend(artifact):
    source = artifact / "baseline_sources/surface-telescope/owned_backend.py"
    specification = importlib.util.spec_from_file_location("profile_frozen_owned", source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    module._NATIVE = source.parent / "native"
    os.environ["TELESCOPE_NATIVE_CACHE"] = str(artifact / "baseline_library")
    module._library()
    return module


def retained_board_rows(count, mapping):
    if not count:
        return None
    from host import unpack

    folder = ROOT / "past_experiments/comparisons/board/telescope-real-20261122"
    capture = json.loads((folder / "d15_board.json").read_bytes())
    columns = {name: i for i, name in enumerate(capture["columns"])}
    run, = [run for run in capture["runs"] if run["p"] == .007]
    choices, all_candidates = [], []
    # Choose by residual weight and frame parity only, never by truth/error.
    for logical in (0, 1):
        candidates = []
        for index, row in enumerate(run["rows"]):
            if row[columns["fpga_status"]] != 3 or row[columns["fpga_frame_bit"]] != logical:
                continue
            raw = row[columns["fpga_residual_hex"]]
            if raw:
                residual = unpack(bytes.fromhex(raw), 112, 16)
                candidates.append((int(residual.sum()), index, residual))
                all_candidates.append((int(residual.sum()), index, residual))
        if candidates:
            choices.append(max(candidates, key=lambda item: (item[0], -item[1])))
    for candidate in sorted(all_candidates, key=lambda item: (-item[0], item[1])):
        if len(choices) >= count:
            break
        if not any(candidate[1] == row[1] for row in choices):
            choices.append(candidate)
    if len(choices) < count:
        raise ValueError("not enough distinct physical-board defers for profiling")
    choices = choices[:count]
    with np.load(folder / "d15_joint.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["memory_indices"], mapping):
            raise ValueError("retained board detector order differs")
        cohort, = np.flatnonzero(saved["rates"] == .007)
        indices = [index for _, index, _ in choices]
        events = saved[f"events_{cohort}"][indices].copy()
    residual = np.stack([r for _, _, r in choices])
    logical = np.asarray([run["rows"][i][columns["fpga_frame_bit"]] for i in indices], np.uint8)
    cycles = np.asarray([run["rows"][i][columns["core_cycles"]] for i in indices], np.int32)
    return events, residual, logical, cycles, indices


def capture(args):
    from telescope import ReferenceFront
    from telescope_circuit import build_context

    artifact = args.artifact.resolve()
    if (artifact / "trace.json").exists() or (artifact / "packets.npz").exists():
        raise ValueError("this artifact already has a captured trace")
    backend = frozen_backend(artifact)
    image = json.loads(args.profile.read_bytes())
    started = perf_counter_ns()
    context, mapping = build_context(image, .005)
    print(f"Surface context ready in {(perf_counter_ns()-started)/1e9:.2f}s", flush=True)
    events = context.circuit.compile_detector_sampler(seed=args.seed).sample(
        args.shots, bit_packed=True,
    )
    memory = np.unpackbits(
        events, axis=1, count=context.num_detectors, bitorder="little",
    )[:, mapping]
    front = ReferenceFront(image).decode(np.ascontiguousarray(memory, dtype=bool))
    board = retained_board_rows(args.board_rows, mapping)
    board_indices = []
    if board is not None:
        board_events, residual, logical, cycles, board_indices = board
        events = np.concatenate((events, board_events))
        front.residual = np.concatenate((front.residual, residual))
        front.logical = np.concatenate((front.logical, logical))
        front.accepted = np.concatenate((front.accepted, np.zeros(len(board_events), bool)))
        front.cycles = np.concatenate((front.cycles, cycles))
    events = np.ascontiguousarray(events, dtype=np.uint8)
    events.flags.writeable = False
    valid = ~front.accepted & (front.cycles > 0)
    decoder = backend.OwnedTensorDecoder(context)
    print(f"Frozen baseline ready in {(perf_counter_ns()-started)/1e9:.2f}s", flush=True)

    arrays = {
        "events": events, "memory_indices": mapping, "front_residual": front.residual,
        "front_logical": front.logical, "front_valid": valid,
        "front_accepted": front.accepted, "front_cycles": front.cycles,
    }
    graph_ids, graph_rows, packets = {}, [], []
    roles = {}
    matching = decoder.owned_engine.correlated_model
    for i, basis in enumerate(("memory", "other")):
        roles[id(matching.prior[i].graph)] = f"correlated_prior_{basis}"
        roles[id(matching.updated[i].graph)] = f"correlated_updated_{basis}"
    if decoder.search is not None:
        for name, fixed in (
            ("tensor_memory", decoder.search.memory_fixed),
            ("tensor_other", decoder.search.other_fixed),
            ("tensor_sector", decoder.search.sector_fixed),
            ("tensor_joint", decoder.search.joint.joint_matching),
        ):
            roles[id(fixed.graph)] = name

    def register(graph):
        key = id(graph)
        if key in graph_ids:
            return graph_ids[key]
        index = len(graph_rows)
        graph_ids[key] = index
        name = f"g{index:03d}"
        arrays[name + "_u"] = graph.u.copy()
        arrays[name + "_v"] = graph.v.copy()
        row = {"id": index, "name": name, "n": graph.n, "m": graph.m,
               "role": roles.get(key, "unknown"),
               "cache": roles.get(key, "").startswith("correlated_prior"), "reference": None}
        graph_rows.append(row)
        if graph._reference is not None:
            row["reference"] = register(graph._reference)
        return index

    phase = {"name": "", "shot": -1}
    original_solve, original_hint = backend.OwnedGraph.solve, backend.OwnedGraph.hint
    original_correlated = backend.OwnedCorrelatedMatching.decode_to_edges_array

    def record(graph, kind, syndrome, weights, seed, call):
        index = len(packets)
        prefix = f"p{index:05d}"
        graph_id = register(graph)
        arrays[prefix + "_syndrome"] = np.asarray(syndrome, np.uint8).copy()
        arrays[prefix + "_weights"] = np.asarray(weights, np.float64).copy()
        if seed is not None:
            arrays[prefix + "_seed"] = np.asarray(seed, np.uint8).copy()
        before = perf_counter_ns()
        returned = call()
        elapsed = perf_counter_ns() - before
        chosen, cost = returned if kind == "solve" else (returned, None)
        arrays[prefix + "_chosen"] = chosen.copy()
        packets.append({
            "index": index, "prefix": prefix, "graph": graph_id, "kind": kind,
            "phase": phase["name"], "shot": phase["shot"], "has_seed": seed is not None,
            "cost": cost, "elapsed_ns": elapsed,
        })
        return returned

    def solve(graph, syndrome, weights, seed=None):
        return record(graph, "solve", syndrome, weights, seed,
                      lambda: original_solve(graph, syndrome, weights, seed))

    def hint(graph, syndrome, removed, weights):
        return record(graph, "hint", syndrome, weights, removed,
                      lambda: original_hint(graph, syndrome, removed, weights))

    def correlated(model, row, enable_correlations=True):
        phase["shot"] += 1
        return original_correlated(model, row, enable_correlations=enable_correlations)

    backend.OwnedGraph.solve, backend.OwnedGraph.hint = solve, hint
    backend.OwnedCorrelatedMatching.decode_to_edges_array = correlated
    phases = {}
    try:
        for name in ("raw", "fpga_hint"):
            phase.update(name=name, shot=-1)
            profiler = cProfile.Profile()
            before = perf_counter_ns()
            profiler.enable()
            if name == "raw":
                prediction, details = decoder.decode_with_diagnostics(events)
            else:
                original_batch = decoder.decode_batch
                decoder.decode_batch = decoder.decode_with_diagnostics
                try:
                    prediction, details = decoder.decode_with_front(
                        events, front.residual, front.logical, valid, mapping,
                    )
                finally:
                    decoder.decode_batch = original_batch
            profiler.disable()
            elapsed = perf_counter_ns() - before
            profiler.dump_stats(str(artifact / f"baseline_{name}.pstats"))
            arrays[name + "_prediction"] = prediction
            for key, value in details.items():
                arrays[name + "_diagnostic_" + key] = value
            phases[name] = {
                "elapsed_ns": elapsed, "cprofile": profile_summary(profiler),
                "stats": decoder.owned_stats(),
            }
            print(f"Captured {name}: {len(packets)} cumulative graph packets", flush=True)
    finally:
        backend.OwnedGraph.solve, backend.OwnedGraph.hint = original_solve, original_hint
        backend.OwnedCorrelatedMatching.decode_to_edges_array = original_correlated
        decoder.close()

    np.savez_compressed(artifact / "packets.npz", **arrays)
    write_json(artifact / "trace.json", {
        "purpose": "development profiling only; no truth or LER estimate",
        "fresh_shots": args.shots, "seed": args.seed, "retained_board_rows": board_indices,
        "board_noise": .007, "model_noise": .005, "profile": str(args.profile.resolve()),
        "num_detectors": context.num_detectors, "dem_errors": context.dem.num_errors,
        "graphs": graph_rows, "packets": packets, "phases": phases,
        "instrumented_capture": True, "raw_phase_precedes_hint_and_warms_model_caches": True,
        "outside_decoder_imported": "pymatching" in sys.modules,
    })
    print(json.dumps({"artifact": str(artifact), "graphs": len(graph_rows),
                      "packets": len(packets), "dem_errors": context.dem.num_errors}), flush=True)


def paired_latency(candidate, artifact, context, mapping, saved, count, normal_cases):
    baseline_backend = frozen_backend(artifact)
    baseline = baseline_backend.OwnedTensorDecoder(context)
    count = min(count, normal_cases)
    indices = np.linspace(0, normal_cases - 1, count, dtype=np.intp)
    events = np.ascontiguousarray(saved["events"])
    events.flags.writeable = False
    decoders = {"baseline": baseline, "candidate": candidate}
    names = ("baseline_raw", "candidate_raw", "baseline_hint", "candidate_hint")
    samples = {name: [] for name in names}

    def execute(name, selection):
        decoder_name, phase = name.split("_")
        decoder = decoders[decoder_name]
        rows = np.ascontiguousarray(events[selection])
        rows.flags.writeable = False
        if phase == "raw":
            return decoder.decode_batch(rows)
        return decoder.decode_with_front(
            rows, saved["front_residual"][selection], saved["front_logical"][selection],
            saved["front_valid"][selection], mapping,
        )

    differences = []
    try:
        for name in names:
            execute(name, indices)
        for position, index in enumerate(indices):
            for offset in range(len(names)):
                name = names[(position + offset) % len(names)]
                before = perf_counter_ns()
                prediction = execute(name, slice(index, index + 1))
                samples[name].append(perf_counter_ns() - before)
                phase = "raw" if name.endswith("_raw") else "fpga_hint"
                if not np.array_equal(prediction, saved[phase + "_prediction"][index:index + 1]):
                    differences.append([name, int(index)])
    finally:
        baseline.close()
    return {
        "indices": indices.tolist(), "new_samples": 0, "warmup_cases_per_variant": count,
        "constructor_excluded": True, "warmup_excluded": True,
        "variant_order_rotates": True, "host_load": "uncontrolled; paired same-process calls",
        "prediction_differences": differences, "samples_ns": samples,
        "latency_ms": {
            name: {"median": float(np.median(values) / 1e6),
                   "p99": float(np.percentile(values, 99) / 1e6)}
            for name, values in samples.items()
        },
    }


def decoder_graphs(decoder):
    matching = decoder.owned_engine.correlated_model
    graphs = {}
    for i, basis in enumerate(("memory", "other")):
        graphs["prior_" + basis] = matching.prior[i].graph
        graphs["updated_" + basis] = matching.updated[i].graph
    for name, model in (
        ("tensor_memory", decoder.search.memory_fixed),
        ("tensor_other", decoder.search.other_fixed),
        ("tensor_sector", decoder.search.sector_fixed),
        ("tensor_joint", decoder.search.joint.joint_matching),
    ):
        graphs[name] = model.graph
    return graphs


def candidate_latencies(decoders, mapping, saved, count, normal_cases):
    count = min(count, normal_cases)
    indices = np.linspace(0, normal_cases - 1, count, dtype=np.intp)
    events = np.ascontiguousarray(saved["events"])
    events.flags.writeable = False
    residual, logical, valid = (
        saved["front_residual"], saved["front_logical"], saved["front_valid"],
    )
    expected = {phase: saved[phase + "_prediction"] for phase in ("raw", "fpga_hint")}
    graphs = {name: decoder_graphs(decoder) for name, decoder in decoders.items()}

    def snapshot():
        return {
            mode: {name: {"stats": graph.stats(), "stage_ns": native_profile(graph),
                          "certificate_rejections": certificate_rejections(graph)}
                   for name, graph in items.items()}
            for mode, items in graphs.items()
        }

    def execute(mode, phase, selection):
        decoder = decoders[mode]
        rows = np.ascontiguousarray(events[selection])
        rows.flags.writeable = False
        if phase == "raw":
            return decoder.decode_batch(rows)
        return decoder.decode_with_front(
            rows, residual[selection], logical[selection], valid[selection], mapping,
        )

    for mode in decoders:
        for phase in ("raw", "fpga_hint"):
            execute(mode, phase, indices)
    before = snapshot()
    times = {mode: {"raw": [], "fpga_hint": []} for mode in decoders}
    differences = {mode: [] for mode in decoders}
    variants = [(mode, phase) for mode in decoders for phase in ("raw", "fpga_hint")]
    for position, index in enumerate(indices):
        for offset in range(len(variants)):
            mode, phase = variants[(position + offset) % len(variants)]
            started = perf_counter_ns()
            result = execute(mode, phase, slice(index, index + 1))
            times[mode][phase].append(perf_counter_ns() - started)
            if not np.array_equal(result, expected[phase][index:index + 1]):
                differences[mode].append([phase, int(index)])
    after = snapshot()
    return {
        mode: {
            "indices": indices.tolist(), "new_samples": 0, "warmup_cases_per_variant": count,
            "constructor_excluded": True, "warmup_excluded": True, "input_io_excluded": True,
            "mode_and_phase_order_rotates": True,
            "prediction_differences": differences[mode], "samples_ns": times[mode],
            "native_counter_deltas": {
                name: {
                    **{kind: {key: after[mode][name][kind][key] - value
                              for key, value in before[mode][name][kind].items()}
                       for kind in ("stats", "stage_ns")},
                    "certificate_rejections": after[mode][name]["certificate_rejections"]
                    - before[mode][name]["certificate_rejections"],
                }
                for name in graphs[mode]
            },
            "latency_ms": {
                phase: {"median": float(np.median(values) / 1e6),
                        "p99": float(np.percentile(values, 99) / 1e6)}
                for phase, values in times[mode].items()
            },
        }
        for mode in decoders
    }


def candidate_latency(decoder, mapping, saved, count, normal_cases):
    return candidate_latencies(
        {"candidate": decoder}, mapping, saved, count, normal_cases,
    )["candidate"]


def replay_decoder(args):
    import owned_backend as backend
    from telescope_circuit import build_context

    artifact = args.replay.resolve()
    trace = json.loads((artifact / "trace.json").read_bytes())
    output = args.output or artifact / "candidate_decoder_replay.json"
    if output.exists():
        raise ValueError("the decoder replay output already exists")
    image = json.loads(Path(trace["profile"]).read_bytes())
    context, mapping = build_context(image, trace["model_noise"])
    original_init = backend.OwnedGraph.__init__

    def initialize(graph, *values, **kwargs):
        original_init(graph, *values, **kwargs)
        enable_lazy(graph)

    if args.lazy:
        backend.OwnedGraph.__init__ = initialize
    try:
        options = {} if args.path_workers is None else {"path_workers": args.path_workers}
        if args.basis_workers is not None:
            options["basis_workers"] = args.basis_workers
        decoder = backend.OwnedTensorDecoder(context, **options)
    finally:
        backend.OwnedGraph.__init__ = original_init
    differences, phases, paired, candidate_times = [], {}, None, None
    try:
        with np.load(artifact / "packets.npz", allow_pickle=False) as saved:
            events = np.ascontiguousarray(saved["events"])
            events.flags.writeable = False
            if args.candidate_timing:
                candidate_times = candidate_latency(
                    decoder, mapping, saved, args.candidate_timing, trace["fresh_shots"],
                )
                print(json.dumps({
                    "candidate_latency_ms": candidate_times["latency_ms"],
                }), flush=True)
            if args.paired_timing:
                paired = paired_latency(
                    decoder, artifact, context, mapping, saved, args.paired_timing,
                    trace["fresh_shots"],
                )
                print(json.dumps({"paired_latency_ms": paired["latency_ms"]}), flush=True)
            for phase in ("raw", "fpga_hint"):
                profiler = cProfile.Profile()
                started = perf_counter_ns()
                profiler.enable()
                prediction, details, correlation_check = decode_diagnostic_phase(
                    decoder, events, saved, mapping, phase, trace, args.compare_correlation,
                )
                profiler.disable()
                elapsed = perf_counter_ns() - started
                if not np.array_equal(prediction, saved[phase + "_prediction"]):
                    differences.append(phase + ":prediction")
                for key, value in details.items():
                    expected = saved[phase + "_diagnostic_" + key]
                    if not np.array_equal(value, expected, equal_nan=True):
                        differences.append(phase + ":" + key)
                if correlation_check and not correlation_check[
                    "first_pass_weights_and_edges_exact"
                ]:
                    differences.append(phase + ":correlation_snapshots")
                profiler.dump_stats(str(output.with_suffix(f".{phase}.pstats")))
                phases[phase] = {
                    "elapsed_ns": elapsed, "stats": decoder.owned_stats(),
                    "cprofile": profile_summary(profiler),
                    "correlation_validation": correlation_check,
                }
                print(f"Decoder replay {phase}: {elapsed / 1e9:.3f}s", flush=True)
    finally:
        decoder.close()
    write_json(output, {
        "fixture": str(artifact), "shots_reused": trace["fresh_shots"]
        + len(trace["retained_board_rows"]), "new_samples": 0,
        "differences": differences, "predictions_and_diagnostics_exact": not differences,
        "phases": phases, "outside_decoder_imported": "pymatching" in sys.modules,
        "lazy_metric_override": args.lazy,
        "production_defaults": not args.lazy and args.path_workers is None
        and args.basis_workers is None
        and args.phase_pricing is None and args.seed_metric is None,
        "path_workers_override": args.path_workers,
        "phase_pricing_override": args.phase_pricing,
        "seed_metric_override": args.seed_metric,
        "basis_workers_override": args.basis_workers,
        "candidate_library": library_identity(backend), "paired_latency": paired,
        "candidate_latency": candidate_times,
    })
    print(json.dumps({"output": str(output), "differences": differences}), flush=True)


def replay(args):
    import owned_backend as backend

    artifact = args.replay.resolve()
    trace = json.loads((artifact / "trace.json").read_bytes())
    report_path = args.output or artifact / "candidate_replay.json"
    if report_path.exists():
        raise ValueError("the replay output already exists")
    records, chosen_differences, cost_differences, strict_cost_differences = [], [], [], []
    with np.load(artifact / "packets.npz", allow_pickle=False) as bundle:
        saved = {key: bundle[key] for key in bundle.files}
        for repeat in range(args.repeats):
            graphs = {}
            try:
                for row in trace["graphs"]:
                    graphs[row["id"]] = backend.OwnedGraph(
                        row["n"], saved[row["name"] + "_u"], saved[row["name"] + "_v"],
                        cache=row["cache"],
                    )
                    if args.lazy:
                        enable_lazy(graphs[row["id"]])
                        if row["role"] == "tensor_joint":
                            graphs[row["id"]].set_partial_cache_budget(args.joint_cache)
                for row in trace["graphs"]:
                    if row["reference"] is not None:
                        graphs[row["id"]].set_reference(graphs[row["reference"]])
                        if args.path_workers is not None:
                            graphs[row["id"]].set_path_workers(args.path_workers)
                before = perf_counter_ns()
                for packet in trace["packets"]:
                    graph, prefix = graphs[packet["graph"]], packet["prefix"]
                    seed = saved[prefix + "_seed"] if packet["has_seed"] else None
                    started = perf_counter_ns()
                    if packet["kind"] == "solve":
                        chosen, cost = graph.solve(
                            saved[prefix + "_syndrome"], saved[prefix + "_weights"], seed,
                        )
                    else:
                        chosen = graph.hint(
                            saved[prefix + "_syndrome"], seed, saved[prefix + "_weights"],
                        )
                    elapsed = perf_counter_ns() - started
                    if packet["kind"] == "solve" and not np.isclose(
                        cost, packet["cost"], rtol=1e-12, atol=1e-10,
                    ):
                        cost_differences.append([repeat, packet["index"], cost, packet["cost"]])
                    if packet["kind"] == "solve" and cost != packet["cost"]:
                        strict_cost_differences.append(
                            [repeat, packet["index"], cost, packet["cost"]],
                        )
                    if not np.array_equal(chosen, saved[prefix + "_chosen"]):
                        chosen_differences.append([repeat, packet["index"]])
                    records.append({
                        "repeat": repeat, "packet": packet["index"], "ns": elapsed,
                        "cost": cost if packet["kind"] == "solve" else None,
                    })
                elapsed = perf_counter_ns() - before
                summary = {
                    row["role"]: {"stats": graphs[row["id"]].stats(),
                                  "stage_ns": native_profile(graphs[row["id"]]),
                                  "certificate_rejections": certificate_rejections(
                                      graphs[row["id"]],
                                  )}
                    for row in trace["graphs"]
                }
                print(f"Replay {repeat}: {elapsed / 1e9:.3f}s", flush=True)
            finally:
                for graph in graphs.values():
                    graph.close()
    write_json(report_path, {
        "fixture": str(artifact), "repeats": args.repeats, "packets": records,
        "chosen_differences": chosen_differences, "cost_differences": cost_differences,
        "strict_cost_differences": strict_cost_differences,
        "strict_cost_difference_count": len(strict_cost_differences),
        "exact_selected_bits": not chosen_differences, "equal_costs": not cost_differences,
        "last_repeat_graphs": summary, "outside_decoder_imported": "pymatching" in sys.modules,
        "lazy_metric": args.lazy, "candidate_library": library_identity(backend),
        "joint_partial_cache_bytes": args.joint_cache if args.lazy else 0,
        "path_workers_override": args.path_workers,
        "phase_pricing_override": args.phase_pricing,
        "seed_metric_override": args.seed_metric,
    })
    print(json.dumps({"output": str(report_path), "chosen_differences": len(chosen_differences),
                      "cost_differences": len(cost_differences)}), flush=True)


def paired_libraries(args):
    import owned_backend as backend

    artifact = args.replay.resolve()
    trace = json.loads((artifact / "trace.json").read_bytes())
    output = args.output or artifact / "paired_libraries.json"
    if output.exists():
        raise ValueError("the paired library output already exists")
    candidate = backend._library()
    original_cdll, previous = ct.CDLL, backend._LIB
    try:
        backend._LIB = None
        ct.CDLL = lambda name, *values, **kwargs: original_cdll(
            str(args.compare_library), *values, **kwargs,
        )
        comparison = backend._library()
    finally:
        ct.CDLL, backend._LIB = original_cdll, previous
    libraries = {"comparison": comparison, "candidate": candidate}
    groups, results, differences = {}, [], []
    with np.load(artifact / "packets.npz", allow_pickle=False) as bundle:
        saved = {key: bundle[key] for key in bundle.files}
    try:
        for name, library in libraries.items():
            backend._LIB = library
            graphs = groups[name] = {}
            for row in trace["graphs"]:
                graph = graphs[row["id"]] = backend.OwnedGraph(
                    row["n"], saved[row["name"] + "_u"], saved[row["name"] + "_v"],
                    cache=row["cache"],
                )
                enable_lazy(graph)
                if row["role"] == "tensor_joint":
                    graph.set_partial_cache_budget(args.joint_cache)
            for row in trace["graphs"]:
                if row["reference"] is not None:
                    graphs[row["id"]].set_reference(graphs[row["reference"]])
        backend._LIB = previous
        for index, packet in enumerate(trace["packets"]):
            prefix = packet["prefix"]
            syndrome, weights = saved[prefix + "_syndrome"], saved[prefix + "_weights"]
            seed = saved[prefix + "_seed"] if packet["has_seed"] else None
            names = ("comparison", "candidate") if index % 2 == 0 else ("candidate", "comparison")
            for name in names:
                graph = groups[name][packet["graph"]]
                started = perf_counter_ns()
                if packet["kind"] == "solve":
                    chosen, cost = graph.solve(syndrome, weights, seed)
                else:
                    chosen, cost = graph.hint(syndrome, seed, weights), None
                elapsed = perf_counter_ns() - started
                if not np.array_equal(chosen, saved[prefix + "_chosen"]):
                    differences.append([name, index, "chosen"])
                if packet["kind"] == "solve" and cost != packet["cost"]:
                    differences.append([name, index, "strict_cost", cost, packet["cost"]])
                results.append({"library": name, "packet": index, "ns": elapsed})
        summary = {
            name: {
                row["role"]: {"stats": graphs[row["id"]].stats(),
                              "stage_ns": native_profile(graphs[row["id"]])}
                for row in trace["graphs"]
            }
            for name, graphs in groups.items()
        }
    finally:
        backend._LIB = previous
        for graphs in groups.values():
            for graph in graphs.values():
                graph.close()
    totals = {
        name: sum(row["ns"] for row in results if row["library"] == name) / 1e9
        for name in libraries
    }
    write_json(output, {
        "fixture": str(artifact), "new_samples": 0, "packets_per_library": len(trace["packets"]),
        "order": "alternating library order for each packet", "records": results,
        "differences": differences, "exact_selected_bits_and_strict_costs": not differences,
        "graphs": summary, "library_call_seconds": totals,
        "libraries": {
            name: {"path": str(library._name),
                   "sha256": hashlib.sha256(Path(library._name).read_bytes()).hexdigest()}
            for name, library in libraries.items()
        },
    })
    print(json.dumps({"output": str(output), "differences": differences,
                      "library_call_seconds": totals}), flush=True)


def worker_sweep(args, phase_compare=False, basis_compare=False):
    import owned_backend as backend
    from telescope_circuit import build_context

    artifact = args.replay.resolve()
    trace = json.loads((artifact / "trace.json").read_bytes())
    output = args.output or artifact / "path_workers.json"
    if output.exists():
        raise ValueError("the worker comparison output already exists")
    image = json.loads(Path(trace["profile"]).read_bytes())
    context, mapping = build_context(image, trace["model_noise"])
    decoders, modes = {}, {}
    if basis_compare:
        configurations = [(f"basis_{n}", args.path_workers or 4, args.phase_pricing, n)
                          for n in (1, 2)]
    elif phase_compare:
        configurations = [("phase_off", args.path_workers or 4, False, args.basis_workers),
                          ("phase_on", args.path_workers or 4, True, args.basis_workers)]
    else:
        configurations = [(f"workers_{n}", n, args.phase_pricing, args.basis_workers)
                          for n in (1, 2, 4)]
    try:
        for name, workers, pricing, basis_workers in configurations:
            before = perf_counter_ns()
            original_init = backend.OwnedGraph.__init__

            def initialize(graph, *values, _pricing=pricing, _original=original_init, **kwargs):
                _original(graph, *values, **kwargs)
                if _pricing is not None:
                    set_phase_pricing(graph, _pricing)

            backend.OwnedGraph.__init__ = initialize
            try:
                options = {"path_workers": workers}
                if basis_workers is not None:
                    options["basis_workers"] = basis_workers
                decoder = backend.OwnedTensorDecoder(context, **options)
            finally:
                backend.OwnedGraph.__init__ = original_init
            decoders[name] = decoder
            modes[name] = {
                "requested_workers": workers,
                "phase_pricing_override": pricing,
                "requested_basis_workers": basis_workers,
                "configured_basis_workers": basis_worker_count(decoder),
                "configured_workers": path_worker_counts(decoder),
                "construction_ns": perf_counter_ns() - before,
            }
            print(f"Constructed {name}", flush=True)
        with np.load(artifact / "packets.npz", allow_pickle=False) as bundle:
            saved = {key: bundle[key] for key in bundle.files}
        latency = candidate_latencies(
            decoders, mapping, saved, args.candidate_timing or 8, trace["fresh_shots"],
        )
        events = np.ascontiguousarray(saved["events"])
        events.flags.writeable = False
        for name, decoder in decoders.items():
            modes[name]["candidate_latency"] = latency[name]
            differences, phases = [], {}
            for phase in ("raw", "fpga_hint"):
                profiler = cProfile.Profile()
                started = perf_counter_ns()
                profiler.enable()
                prediction, details, correlation_check = decode_diagnostic_phase(
                    decoder, events, saved, mapping, phase, trace, args.compare_correlation,
                )
                profiler.disable()
                elapsed = perf_counter_ns() - started
                if not np.array_equal(prediction, saved[phase + "_prediction"]):
                    differences.append(phase + ":prediction")
                for key, value in details.items():
                    if not np.array_equal(
                        value, saved[phase + "_diagnostic_" + key], equal_nan=True,
                    ):
                        differences.append(phase + ":" + key)
                if correlation_check and not correlation_check[
                    "first_pass_weights_and_edges_exact"
                ]:
                    differences.append(phase + ":correlation_snapshots")
                profiler.dump_stats(str(output.with_suffix(f".{name}.{phase}.pstats")))
                phases[phase] = {
                    "elapsed_ns": elapsed, "stats": decoder.owned_stats(),
                    "cprofile": profile_summary(profiler),
                    "correlation_validation": correlation_check,
                }
            modes[name].update(
                differences=differences, predictions_and_diagnostics_exact=not differences,
                phases=phases,
            )
            print(json.dumps({name: {"latency_ms": latency[name]["latency_ms"],
                                     "differences": differences}}), flush=True)
    finally:
        for decoder in decoders.values():
            decoder.close()
        decoders.clear()
        decoder = None
        gc.collect()
    write_json(output, {
        "fixture": str(artifact), "new_samples": 0, "one_context": True,
        "unique_saved_cases": trace["fresh_shots"] + len(trace["retained_board_rows"]),
        "worker_counts": sorted({row[1] for row in configurations}),
        "basis_worker_counts": [1, 2] if basis_compare else None,
        "phase_pricing_values": [False, True] if phase_compare else None,
        "comparison_kind": "check_bases" if basis_compare else (
            "phase_pricing" if phase_compare else "path_workers"
        ),
        "mode_and_phase_order_rotates": True,
        "candidate_library": library_identity(backend), "modes": modes,
        "outside_decoder_imported": "pymatching" in sys.modules,
    })
    print(json.dumps({"output": str(output), "completed_modes": list(modes)}), flush=True)


def main():
    sys.meta_path.insert(0, BlockOutsideDecoders())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decoder", action="store_true", help="replay full saved decoder inputs")
    parser.add_argument("--lazy", action="store_true", help="opt in to native lazy metric replay")
    parser.add_argument("--joint-cache", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--paired-timing", type=int, default=0)
    parser.add_argument("--candidate-timing", type=int, default=0)
    parser.add_argument("--compare-library", type=Path)
    parser.add_argument("--path-workers", type=int, choices=range(1, 9))
    parser.add_argument("--worker-sweep", action="store_true")
    parser.add_argument("--phase-pricing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--phase-sweep", action="store_true")
    parser.add_argument("--seed-metric", action=argparse.BooleanOptionalAction)
    parser.add_argument("--basis-workers", type=int, choices=(1, 2))
    parser.add_argument("--basis-sweep", action="store_true")
    parser.add_argument("--compare-correlation", action="store_true")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--board-rows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=51930101)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile", type=Path, default=SURFACE / "profiles/d15z-gap3-81mhz.json")
    args = parser.parse_args()
    if not 1 <= args.shots <= 16 or not 0 <= args.board_rows <= 2 or not 1 <= args.repeats <= 3:
        parser.error("profiling is limited to16 freshshots,2 retainedrows and3 replaycycles")
    if args.lazy and not args.replay:
        parser.error("--lazy is only available when replaying an existing trace")
    if not 0 <= args.joint_cache <= 256 * 1024 * 1024:
        parser.error("the joint route cache budget must be between0and256MiB")
    if not 0 <= args.paired_timing <= 8:
        parser.error("paired timing is limited to8 existing normal-case inputs")
    if not 0 <= args.candidate_timing <= 8:
        parser.error("candidate timing is limited to8 existing normal-case inputs")
    if args.phase_pricing is not None or args.seed_metric is not None:
        import owned_backend as backend

        original_init = backend.OwnedGraph.__init__

        def initialize(graph, *values, **kwargs):
            original_init(graph, *values, **kwargs)
            if args.phase_pricing is not None:
                set_phase_pricing(graph, args.phase_pricing)
            if args.seed_metric is not None:
                set_seed_metric(graph, args.seed_metric)

        backend.OwnedGraph.__init__ = initialize
    if args.replay and args.basis_sweep:
        worker_sweep(args, basis_compare=True)
    elif args.replay and args.phase_sweep:
        worker_sweep(args, phase_compare=True)
    elif args.replay and args.worker_sweep:
        worker_sweep(args)
    elif args.replay and args.compare_library:
        paired_libraries(args)
    elif args.replay and args.decoder:
        replay_decoder(args)
    elif args.replay:
        replay(args)
    elif args.artifact:
        capture(args)
    else:
        parser.error("pass --artifact for baseline capture or --replay for an existing fixture")


if __name__ == "__main__":
    main()

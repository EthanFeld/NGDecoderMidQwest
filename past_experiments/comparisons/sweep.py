"""Paired rotated-Z Surface Telescope sweep with model-cycle and host timing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[2]
SURFACE = ROOT / "past_experiments/surface_fpga"
sys.path[:0] = [str(ROOT), str(SURFACE), str(ROOT / "decoders/telescope"), str(ROOT / "decoders/telescope/tensor_decoder/src"), str(ROOT / "decoders/pymatching_telescope/src"), str(ROOT / "past_experiments/decoder_testing/src")]

import numpy as np
import pymatching
import stim
from surface_decoders.ibm_relay_bp import build_fast_decoder

from tensor_decoder import (
    TensorDecoder,
    build_decoder,
    qecbench_implementation_identity,
)


RATES = (.001, .002, .003, .004, .005, .006, .007, .0075, .008, .009, .010)
MODES = ("fpga_only", "tensor_only", "telescope", "pymatching", "correlated_pymatching", "relay_bp")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Profile:
    path: Path
    bitstream: Path
    clock_mhz: int
    image: dict
    topology: Path | None = None
    library: Path | None = None

    def front(self):
        from growth_reference import GrowthModel
        from pymatching_tensor_decoder.telescope.decoder import ReferenceFront

        growth = GrowthModel(self.topology, self.library) if self.topology else None
        return ReferenceFront(self.image, growth)

    def identity(self):
        from surface_model import seal

        return {
            "profile": self.path.relative_to(ROOT).as_posix(),
            "profile_sha256": sha256(self.path),
            "bitstream": self.bitstream.relative_to(ROOT).as_posix(),
            "bitstream_sha256": sha256(self.bitstream),
            "image_tag": f"{seal(self.image):08x}",
            "clock_mhz": self.clock_mhz,
            "growth_topology_sha256": sha256(self.topology) if self.topology else None,
            "growth_model_sha256": sha256(self.library) if self.library else None,
        }


def get_profile(distance):
    suffix = ".dll" if sys.platform == "win32" else ".so"
    work = SURFACE / "build/d15-tilt-growth-full-ram"
    if distance == 15:
        path = SURFACE / "experiments/d15-weighted-growth/d15z-growth-cap4000.json"
        bitstream = path.with_name("surface-d15z-tilt-growth-27mhz.fs")
        return Profile(path, bitstream, 27, json.loads(path.read_bytes()),
                       work / "growth_graph.txt", work / "sim" / f"growth_model{suffix}")
    stem, clock = {3: ("d3z", 135), 7: ("d7z-dual-p007-cut0-81mhz", 81)}[distance]
    path = SURFACE / "profiles" / f"{stem}.json"
    image = json.loads(path.read_bytes())
    if distance == 3:
        return Profile(path, SURFACE / "bitstreams/surface-d3z-135mhz.fs", clock, image)
    bitstream = SURFACE / "bitstreams" / f"surface-{stem}.fs"
    return Profile(path, bitstream, clock, image, path.with_suffix(".growth.txt"),
                   SURFACE / "build" / f"growth_model{suffix}")


def build_context(profile, p):
    from pymatching_tensor_decoder.telescope.telescope_circuit import build_context as joint_context

    return joint_context(profile.image, p)


def project(events, context, memory_indices):
    unpacked = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")
    return np.ascontiguousarray(unpacked[:, memory_indices], dtype=bool)


def build_software(context, include_relay=True):
    tensor = (
        TensorDecoder(context, solve_workers=4, path_bucket=True)
        if context.experiment.distance == 15
        else build_decoder(context)
    )
    decoders = {"tensor_only": tensor.decode_batch}
    for name, correlated in (("pymatching", False), ("correlated_pymatching", True)):
        matching = pymatching.Matching.from_detector_error_model(context.dem, enable_correlations=correlated)
        matching.ensure_num_fault_ids(1)
        decoders[name] = partial(
            matching.decode_batch, bit_packed_shots=True, bit_packed_predictions=True,
            enable_correlations=correlated,
        )
    if include_relay:
        decoders["relay_bp"] = build_fast_decoder(context).decode_batch
    return tensor, decoders


def timed(call, *args):
    start = perf_counter_ns()
    result = call(*args)
    return result, (perf_counter_ns() - start) / 1000


def telescope_decode(decode, events, accepted, fpga_labels):
    labels = fpga_labels.copy()
    rejected = ~accepted
    if rejected.any():
        tail = events[rejected]
        tail.flags.writeable = False
        labels[rejected] = decode(tail)[:, 0]
    return labels


def wilson(errors, n):
    if n == 0:
        return None, None
    p = errors / n
    z = 1.959963984540054
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    span = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / den
    return max(0, center - span), min(1, center + span)


def time_stats(values, weights=None):
    values = np.asarray(values)
    if not len(values):
        return None
    if weights is None:
        weights = np.ones(len(values))
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / np.sum(weights)
    limits = np.searchsorted(cdf, [.5, .99, .999])
    return {
        "mean": float(np.average(values, weights=weights)),
        **dict(zip(("median", "p99", "p999"), map(float, values[order[limits]]), strict=True)),
    }


def time_subset(accepted, requested, seed):
    """Oversample rare defers and weight both strata back to cohort size."""
    rng = np.random.default_rng(seed)
    selected, weights = [], []
    for mask in (accepted, ~accepted):
        pool = np.flatnonzero(mask)
        if not len(pool):
            continue
        size = min(len(pool), max(min(32, requested // 4 + 1), round(requested * len(pool) / len(accepted))))
        selected.extend(rng.choice(pool, size, replace=False))
        weights.extend([len(pool) / size] * size)
    order = rng.permutation(len(selected))
    return np.asarray(selected)[order], np.asarray(weights)[order]


def timing(decoders, context, mapping, events, accepted, fpga, core, link, prediction, count, seed,
           relay_time_shots):
    samples, weights = time_subset(accepted, count, seed)
    extra_relay = "relay_bp" in decoders and (len(prediction["relay_bp"]) < len(events) or relay_time_shots < count)
    software = {name: decode for name, decode in decoders.items() if name != "relay_bp" or not extra_relay}
    names = ("fpga_only", "telescope", *software)
    values = {name: np.empty(len(samples)) for name in names}
    with_link = {name: np.empty(len(samples)) for name in ("fpga_only", "telescope")} if link is not None else {}
    decode = decoders["tensor_only"]

    def fpga_host(row, status, label):
        project(row, context, mapping)
        return label

    def telescope_host(row, status, label):
        project(row, context, mapping)
        return telescope_decode(decode, row, status, label)

    calls = {"fpga_only": fpga_host, "telescope": telescope_host}
    for j, index in enumerate(samples):
        segment = slice(index, index + 1)
        row = events[segment]
        for shift in range(len(names)):
            name = names[(j + shift) % len(names)]
            if name in calls:
                answer, elapsed = timed(calls[name], row, accepted[segment], fpga[segment])
                values[name][j] = elapsed + core[index]
                if with_link:
                    with_link[name][j] = elapsed + link[index]
            else:
                answer, elapsed = timed(decoders[name], row)
                values[name][j] = elapsed
            if int(np.ravel(answer)[0]) != int(prediction[name][index]):
                raise ValueError(f"{name} changed its decision between batch and single-shot at row {index}")
    metric = {name: (time_stats(times, weights), len(samples)) for name, times in values.items()}
    if extra_relay:
        relay_samples, relay_weights = time_subset(accepted[: len(prediction["relay_bp"])], relay_time_shots, seed)
        relay_us = []
        for index in relay_samples:
            answer, elapsed = timed(decoders["relay_bp"], events[index : index + 1])
            if int(answer[0, 0]) != int(prediction["relay_bp"][index]):
                raise ValueError(f"relay_bp changed its decision between batch and single-shot at row {index}")
            relay_us.append(elapsed)
        metric["relay_bp"] = time_stats(relay_us, relay_weights), len(relay_samples)
    wire = {name: time_stats(values, weights) for name, values in with_link.items()}
    return metric, wire, int((~accepted[samples]).sum())


def compare_point(profile, context, memory_indices, events, truth, fpga_result, time_shots,
                  batch_size, link_us=None, seed=0, source="cycle_model", *, include_relay=True,
                  relay_shots=None, relay_time_shots=None, telescope_batch_time_shots=None):
    """Decode and time one cohort, including actual board cycles if supplied."""
    n = len(events)
    events.flags.writeable = False
    truth = np.asarray(truth, dtype=np.uint8).reshape(n)
    accepted = np.asarray(fpga_result.accepted, dtype=bool).reshape(n)
    clocks = getattr(fpga_result, "cycles", None)
    if clocks is None:
        clocks = fpga_result.clocks
    core = np.asarray(clocks, dtype=float).reshape(n) / profile.clock_mhz
    if np.any(core <= 0):
        raise ValueError("compare_point requires a valid core reply for every shot")
    frames = np.asarray(fpga_result.logical, dtype=np.uint8).reshape(n)
    fpga = np.where(accepted, frames, 0).astype(np.uint8)
    link = None if link_us is None else np.asarray(link_us, dtype=float).reshape(n)
    if link is not None and (not np.isfinite(link).all() or np.any(link < core)):
        raise ValueError("UART round-trip times must include FPGA core time")

    tensor, decoders = build_software(context, include_relay=include_relay)
    modes = tuple(name for name in MODES if include_relay or name != "relay_bp")
    relay_n = min(relay_shots, n) if relay_shots is not None else n
    relay_time_shots = relay_time_shots if relay_time_shots is not None else time_shots
    fpga_only_name = "fpga_only" if accepted.all() else "fpga_only_zero_on_defer"
    prediction = {name: np.empty(relay_n if name == "relay_bp" else n, dtype=np.uint8) for name in modes}
    prediction["fpga_only"][:] = fpga
    batched = dict.fromkeys(modes, 0.0)
    tail_count = int((~accepted).sum())
    telescope_timed = min(telescope_batch_time_shots, n) if telescope_batch_time_shots else n
    try:
        # Warm on separately generated syndromes rather than using the timing rows.
        warm = context.circuit.compile_detector_sampler(seed=(seed + 1) % (2**64)).sample(
            min(32, n), bit_packed=True,
        )
        warm.flags.writeable = False
        for decode in decoders.values():
            decode(warm)
        project(warm, context, memory_indices)
        names = (*decoders, "fpga_only", "telescope")
        decode = decoders["tensor_only"]
        for block, start in enumerate(range(0, n, batch_size)):
            sl = slice(start, start + batch_size)
            row = events[sl]
            for shift in range(len(names)):
                name = names[(block + shift) % len(names)]
                if name == "fpga_only":
                    _, delta = timed(project, row, context, memory_indices)
                elif name == "telescope":
                    if start >= telescope_timed:
                        continue
                    part = slice(start, min(sl.stop, telescope_timed))
                    _, projection = timed(project, events[part], context, memory_indices)
                    answer, duration = timed(
                        telescope_decode, decode, events[part], accepted[part], fpga[part],
                    )
                    prediction[name][part] = answer
                    delta = projection + duration
                elif name == "relay_bp":
                    if start >= relay_n:
                        continue
                    relay_slice = slice(start, min(sl.stop, relay_n))
                    answer, delta = timed(decoders[name], events[relay_slice])
                    prediction[name][relay_slice] = answer[:, 0]
                else:
                    answer, delta = timed(decoders[name], row)
                    prediction[name][sl] = answer[:, 0]
                batched[name] += delta
        expected = np.where(accepted, fpga, prediction["tensor_only"])
        if not np.array_equal(prediction["telescope"][:telescope_timed], expected[:telescope_timed]):
            raise ValueError("telescope altered an accepted logical bit or changed the tensor prediction")
        prediction["telescope"][telescope_timed:] = expected[telescope_timed:]
        wall, wire, timed_defers = timing(
            decoders, context, memory_indices, events, accepted, fpga, core, link,
            prediction, time_shots, seed, relay_time_shots,
        )
    finally:
        close = getattr(tensor, "close", None)
        if close is not None:
            close()

    accepted_errors = int(((fpga != truth) & accepted).sum())
    distance = context.experiment.distance
    records = []
    pm_wrong = prediction["pymatching"] != truth
    tensor_wrong = prediction["tensor_only"] != truth
    for name in modes:
        count = len(prediction[name])
        wrong = prediction[name] != truth[:count]
        errors = int(wrong.sum())
        low, high = wilson(errors, count)
        has_fpga = name in ("fpga_only", "telescope")
        amortized_shots = telescope_timed if name == "telescope" else count
        amortized_host = batched[name] / amortized_shots
        amortized_core = float(core.mean()) if has_fpga else 0
        host_stats, timed_shots = wall[name]
        same_cohort = {
            f"relay_paired_{other}_errors": int((values[:count] != truth[:count]).sum())
            for other, values in prediction.items() if other != name
        } if name == "relay_bp" else {}
        records.append({
            "distance": distance, "p": context.experiment.noise, "seed": seed,
            "source": source, "mode": fpga_only_name if name == "fpga_only" else name,
            "shots": count, "errors": errors, "ler": errors / count, "ci95_low": low, "ci95_high": high,
            "fixes_vs_pymatching": int((pm_wrong[:count] & ~wrong).sum()),
            "regressions_vs_pymatching": int((~pm_wrong[:count] & wrong).sum()),
            "fixes_vs_tensor": int((tensor_wrong[:count] & ~wrong).sum()),
            "regressions_vs_tensor": int((~tensor_wrong[:count] & wrong).sum()),
            "fpga_accepts": int(accepted.sum()) if has_fpga else None,
            "fpga_accepted_errors": accepted_errors if has_fpga else None,
            "fpga_defers": tail_count if has_fpga else None,
            "fpga_defer_rate": tail_count / n if has_fpga else None,
            "fpga_native_deferred_nonzero_frames": int(np.count_nonzero(frames[~accepted])) if has_fpga else None,
            "fpga_partial_ler_lower": accepted_errors / n if name == "fpga_only" else None,
            "fpga_partial_ler_upper": (accepted_errors + tail_count) / n if name == "fpga_only" else None,
            "fpga_accepted_only_ler": accepted_errors / int(accepted.sum()) if has_fpga and accepted.any() else None,
            "tensor_tail_shots": tail_count if name == "telescope" else (n if name == "tensor_only" else None),
            "tensor_tail_errors": int((wrong & ~accepted).sum()) if name == "telescope" else None,
            "tensor_mode": getattr(tensor, "mode", "d15_sector_search") if name in ("tensor_only", "telescope") else None,
            "tensor_unavailable_reason": getattr(tensor, "tensor_unavailable_reason", None) if name in ("tensor_only", "telescope") else None,
            "fpga_clock_mhz": profile.clock_mhz if has_fpga else None,
            "fpga_core_us": time_stats(core) if has_fpga else None,
            "timed_shots": timed_shots, "timed_fpga_defers": timed_defers if has_fpga else None,
            "single_shot_us_excluding_uart": host_stats,
            "single_shot_us_including_uart": wire.get(name),
            "amortized_timing_shots": amortized_shots,
            "amortized_active_us_per_shot_excluding_uart": amortized_host + amortized_core,
            "amortized_host_us_per_shot": amortized_host,
            **same_cohort,
        })
    return records


def atomic_json(path, content):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_tables(directory, rows, settings):
    ordered = sorted(rows, key=lambda row: (row["distance"], row["p"], row["mode"]))
    atomic_json(directory / "results.json", {"settings": settings, "results": ordered})
    if not ordered:
        return
    data = []
    metrics = ("fpga_core_us", "single_shot_us_excluding_uart", "single_shot_us_including_uart")
    for row in ordered:
        flat = {k: v for k, v in row.items() if k not in metrics}
        for metric in metrics:
            for stat in ("mean", "median", "p99", "p999"):
                flat[f"{metric}_{stat}"] = row[metric][stat] if row[metric] else None
        data.append(flat)
    temp = directory / "results.csv.tmp"
    with temp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for row in data for k in row)))
        writer.writeheader()
        writer.writerows(data)
    temp.replace(directory / "results.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "past_experiments/comparisons/model")
    parser.add_argument("--distances", nargs="+", type=int, choices=(3, 7, 15), default=[3, 7, 15])
    parser.add_argument("--rates", nargs="+", type=float, default=list(RATES))
    parser.add_argument("--shots-small", type=int, default=8192)
    parser.add_argument("--shots-d15", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--time-shots", type=int, default=192)
    parser.add_argument("--relay-shots-d7", type=int, default=1024)
    parser.add_argument("--relay-shots-d15", type=int, default=64)
    parser.add_argument("--relay-time-shots-d15", type=int, default=24)
    parser.add_argument("--shots-d7-high", type=int, default=4096, help="paired shots for d7 p >= .008")
    parser.add_argument("--telescope-batch-time-shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=87223019)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-capped-relay", action="store_true")
    parser.add_argument("--continue-high-noise-budget", action="store_true")
    args = parser.parse_args()
    if min(args.shots_small, args.shots_d15, args.batch_size, args.time_shots,
           args.relay_shots_d7, args.relay_shots_d15, args.relay_time_shots_d15,
           args.shots_d7_high, args.telescope_batch_time_shots) < 1:
        parser.error("shot counts and batch size must be positive")
    if not all(.001 <= p <= .01 for p in args.rates) or len(set(args.rates)) != len(args.rates):
        parser.error("noise rates must be unique and in [.001, .010]")
    profiles = {d: get_profile(d) for d in args.distances}
    settings = {
        "circuit": "Surface Telescope joint rotated Z, d rounds, circuit-level noise",
        "source": "cycle_model", "rates": args.rates, "distances": args.distances,
        "shots_small": args.shots_small, "shots_d15": args.shots_d15,
        "requested_time_shots": args.time_shots, "batch_size": args.batch_size, "seed": args.seed,
        "relay_accuracy_prefix": {"7": args.relay_shots_d7, "15": args.relay_shots_d15},
        "relay_requested_time_shots_d15": args.relay_time_shots_d15,
        "d7_shots_p_at_least_008": min(args.shots_d7_high, args.shots_small),
        "telescope_batch_timing_prefix": args.telescope_batch_time_shots,
        "versions": {"stim": stim.__version__, "numpy": np.__version__, "pymatching": pymatching.__version__,
                     "relay_bp": importlib.metadata.version("relay-bp"), "python": sys.version,
                     "host": platform.platform()},
        "tensor": qecbench_implementation_identity(), "script_sha256": sha256(Path(__file__)),
        "profiles": {str(d): profile.identity() for d, profile in profiles.items()},
        "fpga_only_defer_policy": "explicit host reporting proxy of logical zero, with zero software decoding",
        "latency": "model core cycles plus measured host projection/completion; excludes UART, queue wait and sampling",
        "timing_sample": "random accepted/deferred strata weighted to their shares; Relay uses its accuracy prefix",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    manifest = args.output / "manifest.json"
    prior = args.output / "initial_manifest.json"
    intermediate = args.output / "capped_relay_manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_bytes())
        if prior.exists():
            settings["initial_manifest_sha256"] = sha256(prior)
        if intermediate.exists():
            settings["capped_relay_manifest_sha256"] = sha256(intermediate)
        for key in ("initial_completed_points", "capped_relay_completed_points"):
            if key in recorded:
                settings[key] = recorded[key]
        if args.continue_capped_relay:
            shared = set(recorded) - {"script_sha256", "timing_sample"}
            if prior.exists() or not args.resume or any(settings.get(k) != recorded[k] for k in shared):
                parser.error("a capped Relay continuation requires the original cohort settings and unused migration")
            manifest.rename(prior)
            settings["initial_manifest_sha256"] = sha256(prior)
            settings["initial_completed_points"] = sorted(path.stem for path in checkpoint_dir.glob("*.json"))
            atomic_json(manifest, settings)
        if args.continue_high_noise_budget:
            shared = set(recorded) - {"script_sha256"}
            if intermediate.exists() or not args.resume or any(settings.get(k) != recorded[k] for k in shared):
                parser.error("a runtime budget continuation requires unchanged completed points and unused migration")
            manifest.rename(intermediate)
            settings["capped_relay_manifest_sha256"] = sha256(intermediate)
            settings["capped_relay_completed_points"] = sorted(
                path.stem for path in checkpoint_dir.glob("*.json")
                if path.stem not in settings.get("initial_completed_points", ())
            )
            atomic_json(manifest, settings)
        if not args.resume or json.loads(manifest.read_bytes()) != settings:
            parser.error("choose a fresh output directory or resume with the original settings and implementation")
    else:
        atomic_json(manifest, settings)
    rows = []
    for distance, profile in profiles.items():
        frontend = profile.front()
        for p in args.rates:
            n = args.shots_d15 if distance == 15 else args.shots_small
            if distance == 7 and p >= .008:
                n = settings["d7_shots_p_at_least_008"]
            point_seed = args.seed + distance * 1_000_000 + round(p * 1_000_000)
            checkpoint = checkpoint_dir / f"d{distance}-p{round(p * 1_000_000):05d}.json"
            if checkpoint.exists() and args.resume:
                point = json.loads(checkpoint.read_bytes())
            else:
                context, indices = build_context(profile, p)
                events, truth = context.circuit.compile_detector_sampler(seed=point_seed).sample(
                    n, bit_packed=True, separate_observables=True,
                )
                result = frontend.decode(project(events, context, indices))
                point = compare_point(
                    profile, context, indices, events, truth, result, args.time_shots,
                    args.batch_size, seed=point_seed,
                    relay_shots={7: args.relay_shots_d7, 15: args.relay_shots_d15}.get(distance),
                    relay_time_shots=args.relay_time_shots_d15 if distance == 15 else args.time_shots,
                    telescope_batch_time_shots=args.telescope_batch_time_shots,
                )
                atomic_json(checkpoint, point)
            rows.extend(point)
            write_tables(args.output, rows, settings)
            print(f"d{distance} p={p:.4f}: " + ", ".join(
                f"{row['mode']} {row['errors']}/{row['shots']}" for row in point
            ), flush=True)


if __name__ == "__main__":
    main()

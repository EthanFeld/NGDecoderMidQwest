"""Generate paired UART test packets and software controls for the two decoders."""

import argparse
from base64 import b64encode
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
from pymatching import Matching
import pymatching
import stim


ROOT = Path(__file__).resolve().parents[2]
SURFACE_ROOT = ROOT / "past_experiments/surface_fpga"
TANG_ROOT = ROOT / "past_experiments/tang_nano_20k_d15"
SURFACE_PROFILE = SURFACE_ROOT / "profiles/d7z-dual-p007-cut0-81mhz.json"
NOISE_RATES = (0.005, 0.0075, 0.01)
DEFAULT_OUT = Path(__file__).resolve().parent / ".build"
BATCH_SIZE = 4096


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def timed_decode(cpu, rows, decode_batch):
    """Match the production CPU call with a single shot in flight."""
    ends = min(128, len(rows))
    for row in rows[:ends]:
        cpu(row)
    decisions = np.empty(len(rows), dtype=np.uint8)
    micros = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        start = perf_counter_ns()
        result = cpu(row)
        micros[i] = (perf_counter_ns() - start) / 1000
        decisions[i] = result
    if not np.array_equal(decisions, decode_batch(rows)):
        raise ValueError("Single-shot CPU predictions differ from the batched reference.")
    return decisions, micros


def counts(expected, cpu, status, front, accepted):
    hybrid = np.where(accepted, front, cpu)
    bad = hybrid != expected
    return {
        "shots": len(expected),
        "fpga_accepts": int(accepted.sum()),
        "fpga_wrong": int((accepted & bad).sum()),
        "cpu_defers": int((~accepted).sum()),
        "cpu_defer_wrong": int((~accepted & bad).sum()),
        "hybrid_wrong": int(bad.sum()),
        "cpu_only_wrong": int((cpu != expected).sum()),
        "status_counts": {int(code): int((status == code).sum()) for code in np.unique(status)},
    }


def merge_counts(total, additional):
    for key, value in additional.items():
        if key == "status_counts":
            target = total.setdefault(key, {})
            for code, count in value.items():
                target[code] = target.get(code, 0) + count
        else:
            total[key] = total.get(key, 0) + value


def packet_rows(requests, truth, cpu, timings, status, cycles=None):
    resulting = []
    for i, (request, true, pred, us, code) in enumerate(
        zip(requests, truth, cpu, timings, status, strict=True)
    ):
        resulting.append([
            b64encode(request).decode("ascii"),
            int(true),
            int(pred),
            round(float(us), 3),
            int(code),
            None if cycles is None else int(cycles[i]),
        ])
    return resulting


def build_surface(points, shot_count, make_packets, seed_base, profile=SURFACE_PROFILE):
    # Reuse the independent full Stim circuit that checks the FPGA graph.
    sys.path.insert(0, str(SURFACE_ROOT))
    import host
    from reference import Predecoder, matching
    from growth_reference import GrowthModel
    from surface_model import generate, seal

    stim_check = load_module("surface_stim_check", SURFACE_ROOT / "tests/stim_check.py")
    image = json.loads(profile.read_text())
    front = Predecoder(image)
    tag = seal(image)
    budget = image.get("growth_cycles", 0)
    growth = GrowthModel(profile.with_suffix(".growth.txt")) if budget else None
    metadata = {
        "distance": image["distance"],
        "basis": image["basis"],
        "rounds": image["rounds"],
        "sampling": "Full Stim circuit from Surface Telescope tests/stim_check.py",
        "fallback": "PyMatching on the original selected-basis syndrome with the profile-specific circuit graph weighted at the measured p",
        "baud_rate": 1_000_000,
        "clock_hz": image.get("decoder_clock_mhz", 135) * 1_000_000,
        "image_tag": tag,
        "width": image["width"],
        "rows": image["rounds"] + 1,
        "max_cycles": front.plan.max_clocks + (budget + 1 if growth else 0),
    }
    cohorts, summary = [], []
    offset = 0
    for k, p in enumerate(points):
        seed = seed_base + k
        args = (image["distance"], image["basis"], image["rounds"], p)
        circuit = stim_check.circuit(*args)
        if circuit.num_detectors != image["checks"]:
            raise ValueError("Stim detectors do not match the Surface Telescope image.")
        cpu = matching(generate(*args))
        sampler = circuit.compile_detector_sampler(seed=seed)
        cohort = {"p": p, "seed": seed}
        total = {}
        for start in range(0, shot_count, BATCH_SIZE):
            batch_size = min(BATCH_SIZE, shot_count - start)
            detector, observables = sampler.sample(batch_size, separate_observables=True)
            truth = observables[:, 0]
            local = front.decode(detector)
            if growth:
                growth.recover(local, detector, budget)
            status = np.where(local.accepted, 1, 3)
            if make_packets:
                decide = lambda row: cpu.decode(row)[0]
                decide_batch = lambda rows: cpu.decode_batch(rows)[:, 0]
                fallback, timings = timed_decode(decide, detector, decide_batch)
                width = image["width"]
                requests = (
                    host.frame(host.REQUEST.pack(host.COMMAND["tilt"], (offset + i) & 0xFFFF, tag)
                               + host.pack(row, width))
                    for i, row in enumerate(detector)
                )
                cohort.setdefault("shots", []).extend(
                    packet_rows(requests, truth, fallback, timings, status, local.clocks)
                )
                offset += batch_size
            else:
                fallback = cpu.decode_batch(detector)[:, 0]
            merge_counts(total, counts(truth, fallback, status, local.logical, local.accepted))
        if make_packets:
            cohorts.append(cohort)
        summary.append({"p": p, "seed": seed, **total})
        print(f"Surface p={p:g}, {shot_count:,} shots: {total['hybrid_wrong']:,} hybrid errors",
              flush=True)
    return {**metadata, "cohorts": cohorts}, {**metadata, "cohorts": summary}


def build_tang(points, shot_count, make_packets, seed_base, directory, *, baud=115_200):
    sys.path.insert(0, str(TANG_ROOT))
    from geometry import circuit
    from transport import ACCEPT_0, ACCEPT_1, project, request

    implementation = load_module("tang_reference", TANG_ROOT / "reference.py")
    library = implementation.build_reference(directory)
    metadata = {
        "distance": 15,
        "basis": "z",
        "rounds": 15,
        "sampling": "Full Stim rotated_memory_z circuit with all four circuit noise channels at p",
        "fallback": "Correlated PyMatching on the full original Stim detector record",
        "baud_rate": baud,
        "clock_hz": 27_000_000,
    }
    cohorts, summary = [], []
    offset = 0
    with implementation.Reference(library) as engine:
        for k, p in enumerate(points):
            seed = seed_base + k
            program = circuit(p)
            cpu = Matching.from_detector_error_model(
                program.detector_error_model(decompose_errors=True), enable_correlations=True,
            )
            sampler = program.compile_detector_sampler(seed=seed)
            cohort = {"p": p, "seed": seed}
            total = {}
            for start in range(0, shot_count, BATCH_SIZE):
                batch_size = min(BATCH_SIZE, shot_count - start)
                detector, observables = sampler.sample(
                    batch_size, bit_packed=True, separate_observables=True,
                )
                truth = observables[:, 0] & 1
                projected = project(detector)
                status, _ = engine.decode(projected)
                accepted = (status == ACCEPT_0) | (status == ACCEPT_1)

                def decide_batch(rows):
                    return cpu.decode_batch(
                        rows, bit_packed_shots=True, enable_correlations=True,
                    )[:, 0]

                if make_packets:
                    fallback, timings = timed_decode(
                        lambda row: decide_batch(row[None, :])[0], detector, decide_batch,
                    )
                    requests = (request(row, (offset + i) & 255) for i, row in enumerate(projected))
                    cohort.setdefault("shots", []).extend(
                        packet_rows(requests, truth, fallback, timings, status)
                    )
                    offset += batch_size
                else:
                    fallback = decide_batch(detector)
                merge_counts(total, counts(truth, fallback, status, status == ACCEPT_1, accepted))
            if make_packets:
                cohorts.append(cohort)
            summary.append({"p": p, "seed": seed, **total})
            print(f"Tang p={p:g}, {shot_count:,} shots: {total['hybrid_wrong']:,} hybrid errors",
                  flush=True)
    return {**metadata, "cohorts": cohorts}, {**metadata, "cohorts": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--surface-shots", type=int, default=8192)
    parser.add_argument("--surface-profile", type=Path, default=SURFACE_PROFILE)
    parser.add_argument("--tang-shots", type=int, default=2048)
    parser.add_argument("--tang-baud", type=int, choices=(115_200, 1_000_000), default=115_200)
    parser.add_argument("--model-shots", type=int, default=0,
                        help="Generate independent model-only cohorts of this size instead of UART packets.")
    parser.add_argument("--rates", type=float, nargs="+", default=NOISE_RATES)
    args = parser.parse_args()
    if min(args.surface_shots, args.tang_shots) < 1 or args.model_shots < 0:
        parser.error("shot counts must be positive or zero to disable model-only mode")
    if any(not 0 < p < .5 for p in args.rates):
        parser.error("rates must lie between zero and 0.5")
    args.out.mkdir(parents=True, exist_ok=True)
    make_packets = args.model_shots == 0
    seed_shift = 0 if make_packets else 1_000_000
    surface, surface_summary = build_surface(
        args.rates, args.model_shots or args.surface_shots, make_packets, 622701 + seed_shift, args.surface_profile,
    )
    tang, tang_summary = build_tang(
        args.rates, args.model_shots or args.tang_shots, make_packets, 623015 + seed_shift, args.out,
        baud=args.tang_baud,
    )
    versions = {"stim": stim.__version__, "pymatching": pymatching.__version__,
                "numpy": np.__version__, "batch_size": BATCH_SIZE}
    summary = {"versions": versions, "surface": surface_summary, "tang": tang_summary}
    target = args.out / ("reference.json" if make_packets else "model-reference.json")
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if make_packets:
        data = {"versions": versions, "surface": surface, "tang": tang}
        (args.out / "fixtures.json").write_text(
            json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()

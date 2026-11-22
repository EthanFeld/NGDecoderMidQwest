"""Prepare and verify a paired d15 FPGA replay with both detector bases retained."""

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pymatching
import stim
from vectors import DEFAULT_OUT, ROOT, SURFACE_ROOT, packet_rows, timed_decode

PROFILE = "d15z-gap3-81mhz"
PREFIX = "d15_telescope"


def prepare(output, rates, shots, seed):
    sys.path[:0] = [str(SURFACE_ROOT), str(ROOT / "decoders/telescope"), str(ROOT / "decoders/telescope/tensor_decoder/src")]
    import host
    from reference import Predecoder, matching
    from surface_model import generate, seal
    from telescope_circuit import build_context

    manifest = json.loads((SURFACE_ROOT / "bitstreams/manifest.json").read_text(encoding="utf-8"))
    entry = next(item for item in manifest if item["profile"] == PROFILE)
    graph = (SURFACE_ROOT / entry["graph"]).read_bytes()
    image = json.loads(graph)
    tag = seal(image)
    if sha256(graph).hexdigest() != entry["graph_sha256"] or tag != int(entry["image_tag"], 16):
        raise ValueError("The profile is not the manifest-matched d15 FPGA graph.")
    front = Predecoder(image)
    config = {
        "distance": image["distance"],
        "basis": image["basis"],
        "rounds": image["rounds"],
        "clock_hz": entry["clock_mhz"] * 1_000_000,
        "baud_rate": 1_000_000,
        "image_tag": tag,
        "width": image["width"],
        "rows": image["rounds"] + 1,
        "max_cycles": front.plan.max_clocks,
        "cohorts": [],
    }
    arrays = {"rates": np.asarray(rates), "seeds": np.arange(seed, seed + len(rates))}
    for index, noise in enumerate(rates):
        context, memory_indices = build_context(image, noise)
        if index and not np.array_equal(arrays["memory_indices"], memory_indices):
            raise ValueError("FPGA detector order changed between noise cohorts.")
        arrays["memory_indices"] = memory_indices
        syndrome, observable = context.circuit.compile_detector_sampler(seed=seed + index).sample(
            shots, separate_observables=True
        )
        memory = syndrome[:, memory_indices]
        truth = observable[:, 0].astype(np.uint8)
        expected = front.decode(memory)
        status = np.where(expected.accepted, 1, 3)
        cpu = matching(generate(image["distance"], image["basis"], image["rounds"], noise))
        control, times = timed_decode(
            lambda row, cpu=cpu: cpu.decode(row)[0], memory,
            lambda rows, cpu=cpu: cpu.decode_batch(rows)[:, 0],
        )
        frames = (
            host.frame(host.REQUEST.pack(host.COMMAND["tilt"], (index * shots + i) & 0xffff, tag)
                       + host.pack(row, image["width"]))
            for i, row in enumerate(memory)
        )
        config["cohorts"].append({
            "p": noise, "seed": seed + index,
            "shots": packet_rows(frames, truth, control, times, status, expected.clocks),
        })
        arrays[f"events_{index}"] = np.packbits(syndrome, axis=1, bitorder="little")
        arrays[f"truth_{index}"] = truth
        arrays[f"residual_{index}"] = np.packbits(expected.residual, axis=1, bitorder="little")
        arrays[f"frame_bits_{index}"] = expected.logical.astype(np.uint8)
        print(f"p={noise:g}, seed {seed + index}, {shots} shots, {np.count_nonzero(status == 3)} predicted defers",
              flush=True)

    versions = {"stim": stim.__version__, "pymatching": pymatching.__version__, "numpy": np.__version__}
    payload = json.dumps({"surface": config, "versions": versions}, separators=(",", ":"))
    page = Path(__file__).with_name("uart_benchmark.html").read_text(encoding="utf-8")
    page = page.replace("<!-- BENCHMARK_FIXTURES -->",
                        f'<script id="embedded-fixtures" type="application/json">{payload}</script>')
    page = page.replace("Surface Telescope d7</option>", "Surface Telescope d15 Z prior tilt</option>")
    page = page.replace("fpga-uart-measurements.json", f"{PREFIX}_board.json")
    page = page.replace("if (window.showSaveFilePicker) {", "if (false) {")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{PREFIX}.html").write_text(page, encoding="utf-8")
    (output / f"{PREFIX}_fixtures.json").write_text(payload, encoding="utf-8")
    np.savez_compressed(output / "telescope_board_full.npz", **arrays)


def verify(output, capture):
    raw = json.loads(capture.read_text(encoding="utf-8-sig"))
    fixtures = json.loads((output / f"{PREFIX}_fixtures.json").read_text(encoding="utf-8"))["surface"]
    columns = {name: index for index, name in enumerate(raw["columns"])}
    residual_field = columns["fpga_residual_hex"]
    frame_field = columns["fpga_frame_bit"]
    with np.load(output / "telescope_board_full.npz") as arrays:
        for index, (run, expected) in enumerate(zip(raw["runs"], fixtures["cohorts"], strict=True)):
            rows = run["rows"]
            if (not run["complete"] or run["fault"] or (run["p"], run["seed"]) != (expected["p"], expected["seed"])
                or run["clock_hz"] != fixtures["clock_hz"] or run["baud_rate"] != fixtures["baud_rate"]
                or run["mode"] != "surface" or len(rows) != len(expected["shots"])):
                raise ValueError(f"Unexpected or incomplete FPGA capture for p={expected['p']:g}.")
            for i, (row, shot) in enumerate(zip(rows, expected["shots"], strict=True)):
                if row[:3] != shot[1:4] or row[10:13] != [shot[4], shot[4], shot[5]] or row[4] != shot[5]:
                    raise ValueError(f"Shot metadata, status or cycle disagreement at p={run['p']:g}, shot {i}.")
                residual = arrays[f"residual_{index}"][i].tobytes().hex() if shot[4] == 3 else None
                frame = int(arrays[f"frame_bits_{index}"][i])
                if (row[residual_field] != residual or row[frame_field] != frame
                    or row[3] != (None if shot[4] == 3 else frame)):
                    raise ValueError(f"Residual or logical bit disagreement at p={run['p']:g}, shot {i}.")
            retries = sum(row[13] for row in rows)
            round_trip = np.median([row[6] for row in rows])
            print(f"p={run['p']:g}: {len(rows)} exact replies, {retries} retries, median UART {round_trip:.1f} us")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rates", type=float, nargs="+", default=[.003, .005, .0075])
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=51915031)
    parser.add_argument("--capture", type=Path, help="Verify an existing board capture instead of preparing fixtures.")
    args = parser.parse_args()
    if args.capture:
        verify(args.out, args.capture)
    else:
        if args.shots < 1:
            parser.error("--shots must be positive")
        prepare(args.out, args.rates, args.shots, args.seed)


if __name__ == "__main__":
    main()

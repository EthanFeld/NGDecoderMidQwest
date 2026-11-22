"""Build a fresh d7 Web Serial fixture and retain the paired joint Stim shots."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pymatching
import stim


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SURFACE = ROOT / "past_experiments/surface_fpga"
BENCHMARK = ROOT / "past_experiments/fpga_benchmark"
sys.path[:0] = [str(ROOT), str(SURFACE), str(ROOT / "decoders/telescope"), str(ROOT / "decoders/telescope/tensor_decoder/src"), str(ROOT / "decoders/pymatching_telescope/src"), str(ROOT / "past_experiments/decoder_testing/src")]

import host
from growth_reference import GrowthModel
from reference import Predecoder, matching
from surface_model import generate, seal
from telescope_circuit import build_context
from vectors import packet_rows, timed_decode


PROFILE = "d7z-dual-p007-cut0-81mhz"
RATES = (.001, .003, .005, .007)


def prepare(output: Path, shots: int, seed: int):
    manifest = json.loads((SURFACE / "bitstreams/manifest.json").read_bytes())
    entry = next(item for item in manifest if item["profile"] == PROFILE)
    graph_path = SURFACE / entry["graph"]
    graph = graph_path.read_bytes()
    bitstream = (SURFACE / entry["bitstream"]).read_bytes()
    image = json.loads(graph)
    tag = seal(image)
    if (sha256(graph).hexdigest() != entry["graph_sha256"]
            or sha256(bitstream).hexdigest() != entry["bitstream_sha256"]
            or tag != int(entry["image_tag"], 16)):
        raise ValueError("The d7 graph and FPGA bitstream must match the manifest.")

    front = Predecoder(image)
    growth = GrowthModel(graph_path.with_suffix(".growth.txt"))
    budget = image["growth_cycles"]
    config = {
        "distance": image["distance"], "basis": image["basis"], "rounds": image["rounds"],
        "clock_hz": entry["clock_mhz"] * 1_000_000, "baud_rate": 1_000_000,
        "image_tag": tag, "width": image["width"], "rows": image["rounds"] + 1,
        "max_cycles": front.plan.max_clocks + budget + 1, "cohorts": [],
        "profile": PROFILE, "graph_sha256": entry["graph_sha256"],
        "bitstream_sha256": entry["bitstream_sha256"],
        "sampling": "Joint X/Z Surface Telescope Stim circuit",
        "fallback": "Projected PyMatching control; separate joint tensor replay for Telescope",
    }
    arrays = {"rates": np.asarray(RATES), "seeds": np.arange(seed, seed + len(RATES))}
    for index, rate in enumerate(RATES):
        cohort_seed = seed + index
        context, mapping = build_context(image, rate)
        if index and not np.array_equal(arrays["memory_indices"], mapping):
            raise ValueError("Memory detector ordering changed between noise rates.")
        arrays["memory_indices"] = mapping
        events, observables = context.circuit.compile_detector_sampler(seed=cohort_seed).sample(
            shots, bit_packed=True, separate_observables=True,
        )
        memory = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")[:, mapping]
        truth = observables[:, 0]
        model = front.decode(memory)
        growth.recover(model, memory, budget)
        status = np.where(model.accepted, 1, 3)
        control = matching(generate(image["distance"], image["basis"], image["rounds"], rate))
        predictions, times = timed_decode(
            lambda row: control.decode(row)[0], memory,
            lambda rows: control.decode_batch(rows)[:, 0],
        )
        frames = (
            host.frame(host.REQUEST.pack(host.COMMAND["tilt"], (index * shots + n) & 0xffff, tag)
                       + host.pack(row, image["width"]))
            for n, row in enumerate(memory)
        )
        config["cohorts"].append({
            "p": rate, "seed": cohort_seed,
            "shots": packet_rows(frames, truth, predictions, times, status, model.clocks),
        })
        arrays[f"events_{index}"] = events
        arrays[f"truth_{index}"] = truth
        print(f"d7 p={rate:g}: prepared {shots:,} shot fixtures (seed {cohort_seed}).", flush=True)

    versions = {"stim": stim.__version__, "pymatching": pymatching.__version__, "numpy": np.__version__,
                "batch_size": shots}
    payload = json.dumps({"surface": config, "versions": versions}, separators=(",", ":"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "d7_fixtures.json").write_text(payload, encoding="utf-8")
    np.savez_compressed(output / "d7_joint.npz", **arrays)
    page = (BENCHMARK / "uart_benchmark.html").read_text(encoding="utf-8")
    page = page.replace("<!-- BENCHMARK_FIXTURES -->",
                        f'<script id="embedded-fixtures" type="application/json">{payload}</script>')
    page = page.replace("Surface Telescope d7</option>", "Surface Telescope d7 Z, 81 MHz</option>")
    page = page.replace("fpga-uart-measurements.json", "d7_board.json")
    page = page.replace("if (window.showSaveFilePicker) {", "if (false) {")
    page = page.replace("'fpga_residual_hex', 'fpga_frame_bit',",
                        "'fpga_residual_hex', 'fpga_frame_bit', 'fpga_wire_hex',")
    page = page.replace("bytes: wire.length, frameBit: bit, residual};",
                        "bytes: wire.length, frameBit: bit, residual,\n"
                        "        wireHex: wire.map(byte => byte.toString(16).padStart(2, '0')).join('')};")
    page = page.replace("residualHex, packet.frameBit ?? null]);",
                        "residualHex, packet.frameBit ?? null, packet.wireHex ?? null]);")
    (output / "d7_uart.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=83127101)
    args = parser.parse_args()
    prepare(args.output, args.shots, args.seed)

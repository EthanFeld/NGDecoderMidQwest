"""Generate paired joint-syndrome inputs and the d15 Web Serial capture page."""

from __future__ import annotations

import json
import os
import sys
from base64 import b64encode
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pymatching
import stim

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT / "decoders/pymatching_telescope/src"))

from pymatching_tensor_decoder.telescope import host
from pymatching_tensor_decoder.telescope.growth_reference import GrowthModel
from pymatching_tensor_decoder.telescope.reference import Predecoder, matching
from pymatching_tensor_decoder.telescope.surface_model import generate, seal
from pymatching_tensor_decoder.telescope.telescope_circuit import build_context


def replace_once(page, old, new):
    if page.count(old) != 1:
        raise ValueError(f"Expected exactly one template marker: {old!r}")
    return page.replace(old, new)


def main():
    rates = (.001, .003, .005, .007)
    seeds = list(range(94625701, 94625705))
    shots = 10_000
    outputs = ("d15_fixtures.json", "d15_joint.npz", "d15.html", "manifest.json")
    if any((OUT / name).exists() for name in outputs):
        raise FileExistsError("Choose a fresh capture directory before preparing new fixtures")
    surface = ROOT / "past_experiments/surface_fpga"
    build = surface / "build/d15-tilt-growth-full-ram"
    profile = surface / "experiments/d15-weighted-growth/d15z-growth-cap4000.json"
    bitstream = surface / "build/gowin-images/surface-d15z-tilt-growth-27mhz-debug-gowin.fs"
    template = ROOT / "past_experiments/fpga_benchmark/uart_benchmark.html"
    image = json.loads((build / "graph.json").read_bytes())
    tag, budget = seal(image), image["growth_cycles"]
    if tag != seal(json.loads(profile.read_bytes())) or tag != 0x592e6076 or budget != 4000:
        raise ValueError("The selected graph and profile do not match the d15 growth image")
    front = Predecoder(image)
    growth = GrowthModel(build / "growth_graph.txt", build / "sim/growth_model.dll")
    config = {
        "distance": 15, "basis": "z", "rounds": 15,
        "width": image["width"], "rows": image["rounds"] + 1,
        "max_cycles": front.plan.max_clocks + budget + 1,
        "baud_rate": 500_000, "clock_hz": 27_000_000, "image_tag": tag,
        "sampling": "Surface Telescope full joint Stim rotated memory Z circuit",
        "profile": profile.relative_to(ROOT).as_posix(),
        "profile_sha256": sha256(profile.read_bytes()).hexdigest(),
        "bitstream": bitstream.relative_to(ROOT).as_posix(),
        "bitstream_sha256": sha256(bitstream.read_bytes()).hexdigest(),
        "cohorts": [],
    }
    arrays = {"rates": np.asarray(rates), "seeds": np.asarray(seeds)}
    for index, (p, seed) in enumerate(zip(rates, seeds, strict=True)):
        context, mapping = build_context(image, p)
        if index and not np.array_equal(arrays["memory_indices"], mapping):
            raise ValueError("FPGA detector order changed between cohorts")
        arrays["memory_indices"] = mapping
        events, observables = context.circuit.compile_detector_sampler(seed=seed).sample(
            shots, bit_packed=True, separate_observables=True,
        )
        truth = observables[:, 0]
        memory = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")[:, mapping] != 0
        arrays[f"events_{index}"] = events
        arrays[f"truth_{index}"] = truth
        expected = front.decode(memory)
        growth.recover(expected, memory, budget)
        status = np.where(expected.accepted, 1, 3)
        arrays[f"expected_status_{index}"] = status.astype(np.uint8)
        arrays[f"expected_cycles_{index}"] = expected.clocks
        arrays[f"expected_frame_{index}"] = expected.logical.astype(np.uint8)
        arrays[f"expected_residual_{index}"] = np.packbits(expected.residual, axis=1, bitorder="little")
        cpu = matching(generate(15, "z", 15, p))
        for row in memory[:16]:
            cpu.decode(row)
        predictions = np.empty(shots, np.uint8)
        times = np.empty(shots, np.float64)
        for i, row in enumerate(memory):
            started = perf_counter_ns()
            predictions[i] = cpu.decode(row)[0]
            times[i] = (perf_counter_ns() - started) / 1000
        if not np.array_equal(predictions, cpu.decode_batch(memory)[:, 0]):
            raise ValueError("Projected matching single-shot/batch predictions differ")
        arrays[f"projected_matching_{index}"] = predictions
        arrays[f"projected_matching_us_{index}"] = times
        packets = []
        for i, row in enumerate(memory):
            request = host.frame(host.REQUEST.pack(host.COMMAND["tilt"], index * shots + i, tag)
                                 + host.pack(row, image["width"]))
            packets.append([
                b64encode(request).decode("ascii"), int(truth[i]), int(predictions[i]),
                round(float(times[i]), 3), int(status[i]), int(expected.clocks[i]),
            ])
        config["cohorts"].append({"p": p, "seed": seed, "shots": packets})
        print(f"d15 p={p:g}, seed={seed}: {shots:,} packets; model defers={int((status == 3).sum()):,}", flush=True)

    versions = {"stim": stim.__version__, "numpy": np.__version__, "pymatching": pymatching.__version__}
    payload = json.dumps({"surface": config, "versions": versions}, separators=(",", ":"))
    page = template.read_text(encoding="utf-8")
    page = replace_once(page, "<!-- BENCHMARK_FIXTURES -->",
                        f'<script id="embedded-fixtures" type="application/json">{payload}</script>')
    page = replace_once(page, "Surface Telescope d7</option>", "Surface Telescope d15 weighted growth 27 MHz</option>")
    page = replace_once(page, "current = {mode, p: cohort.p", "current = {mode, image_tag: config.image_tag, p: cohort.p")
    page = replace_once(page, "'fpga_frame_bit',", "'fpga_frame_bit', 'fpga_wire_hex',")
    page = replace_once(page, "frameBit: bit, residual};",
                        "frameBit: bit, residual, wireHex: wire.map(byte => byte.toString(16).padStart(2, '0')).join('')};")
    page = replace_once(page, "residualHex, packet.frameBit ?? null]);",
                        "residualHex, packet.frameBit ?? null, packet.wireHex ?? null]);")
    page = replace_once(page, "fpga-uart-measurements.json", "telescope-d15-deck-v2-2026-11-22.json")
    page = replace_once(page, "if (window.showSaveFilePicker) {", "if (false) {")
    (OUT / "d15_fixtures.json").write_text(payload, encoding="utf-8")
    np.savez_compressed(OUT / "d15_joint.npz", **arrays)
    (OUT / "d15.html").write_text(page, encoding="utf-8")
    manifest = {
        "distance": 15, "basis": "z", "rounds": 15, "shots_per_rate": shots,
        "rates": list(rates), "seeds": seeds, "image_tag": f"0x{tag:08x}",
        "clock_hz": config["clock_hz"], "baud_rate": config["baud_rate"],
        "profile": config["profile"], "profile_sha256": config["profile_sha256"],
        "bitstream": config["bitstream"], "bitstream_sha256": config["bitstream_sha256"],
        "files": {name: sha256((OUT / name).read_bytes()).hexdigest() for name in outputs[:3]},
        "versions": versions,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"page": (OUT / "d15.html").as_uri(), "image_tag": manifest["image_tag"], "bitstream": str(bitstream)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

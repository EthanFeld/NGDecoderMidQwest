"""Paired original-versus-residual inputs on a fresh Surface Telescope cohort."""

from __future__ import annotations

import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "decoders/pymatching_telescope/src"))

import numpy as np
from pymatching_tensor_decoder import TensorDecoder, qecbench_implementation_identity
from pymatching_tensor_decoder.telescope.reference import Predecoder
from pymatching_tensor_decoder.telescope.telescope_circuit import build_context


def main():
    output = OUT / "residual_comparison.json"
    if output.exists():
        raise FileExistsError(output)
    profile = ROOT / "decoders/telescope/profiles/d15z-gap3-81mhz.json"
    image = json.loads(profile.read_bytes())
    context, mapping = build_context(image, .005)
    seed, shots = 94629701, 4096
    events, observable = context.circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True, bit_packed=True,
    )
    truth = observable[:, 0]
    bits = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder="little")
    memory = np.ascontiguousarray(bits[:, mapping], dtype=bool)
    front = Predecoder(image).decode(memory)
    bits[:, mapping] = front.residual
    mixed = np.packbits(bits, axis=1, bitorder="little")
    events.flags.writeable = mixed.flags.writeable = False
    original = front.logical.astype(np.uint8, copy=True)
    residual = original.copy()
    record = {
        "complete": False, "shots": 0, "target_shots": shots, "seed": seed,
        "distance": 15, "rounds": 15, "noise": .005, "basis": "z",
        "circuit": "Surface Telescope full joint circuit",
        "frontend": "gap3 rescue cycle model, not a new physical board capture",
        "population": "4096 fresh full memory experiments; both telescopes preserve all FPGA accepts and decode only defers",
        "comparison": "Original full joint syndrome versus memory-residual-substituted joint syndrome with FPGA frame XOR on defers",
        "profile": profile.relative_to(ROOT).as_posix(),
        "profile_sha256": sha256(profile.read_bytes()).hexdigest(),
        "identity": qecbench_implementation_identity(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "fpga_accepts": int(front.accepted.sum()), "fpga_defers": int((~front.accepted).sum()),
        "batch_size": 256,
    }
    decoder = TensorDecoder(context)
    started = time.perf_counter()
    try:
        for start in range(0, shots, 256):
            while (OUT / "PAUSE_FOR_TIMING").exists():
                time.sleep(1)
            stop = min(start+256, shots)
            pending = np.flatnonzero(~front.accepted[start:stop]) + start
            if len(pending):
                original_input = np.ascontiguousarray(events[pending])
                residual_input = np.ascontiguousarray(mixed[pending])
                original_input.flags.writeable = residual_input.flags.writeable = False
                original[pending] = decoder.decode_batch(original_input)[:, 0]
                residual[pending] = decoder.decode_batch(residual_input)[:, 0] ^ front.logical[pending]
            record["shots"] = min(start+256, shots)
            record["seconds"] = time.perf_counter()-started
            output.write_text(json.dumps(record, indent=2)+"\n")
            print(f"residual comparison {record['shots']}/{shots}", flush=True)
    finally:
        decoder.close()
    plain = original
    residual_telescope = residual
    predictions = {
        "original_telescope": plain,
        "residual_plus_frame_telescope": residual_telescope,
    }
    record["results"] = {}
    for name, prediction in predictions.items():
        wrong = prediction != truth
        record["results"][name] = {
            "shots": shots, "errors": int(wrong.sum()), "ler": float(wrong.mean()),
            "errors_on_defers": int(np.count_nonzero(wrong & ~front.accepted)),
            "errors_on_accepts": int(np.count_nonzero(wrong & front.accepted)),
        }
    wrong_original, wrong_residual = plain != truth, residual_telescope != truth
    record["residual_vs_original_telescope"] = {
        "fixes": int(np.count_nonzero(wrong_original & ~wrong_residual)),
        "regressions": int(np.count_nonzero(~wrong_original & wrong_residual)),
        "prediction_differences": int(np.count_nonzero(plain != residual_telescope)),
    }
    fixture = OUT / "residual_inputs.npz"
    np.savez_compressed(
        fixture, events=events, truth=truth, memory_indices=mapping,
        accepted=front.accepted, frame=front.logical,
        residual=np.packbits(front.residual, axis=1, bitorder="little"),
        cycles=front.clocks, **predictions,
    )
    record["fixture"] = fixture.name
    record["fixture_sha256"] = sha256(fixture.read_bytes()).hexdigest()
    record["complete"] = True
    output.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()

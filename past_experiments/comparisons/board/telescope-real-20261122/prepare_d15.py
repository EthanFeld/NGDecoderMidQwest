"""Prepare a fresh d15 joint-syndrome capture with the existing Chrome UART harness."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pymatching
import stim

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BENCH = ROOT / 'past_experiments/fpga_benchmark'
SURFACE = ROOT / 'past_experiments/surface_fpga'
sys.path[:0] = [str(ROOT), str(SURFACE), str(ROOT / "decoders/telescope"), str(ROOT / "decoders/telescope/tensor_decoder/src"), str(ROOT / "decoders/pymatching_telescope/src"), str(ROOT / "past_experiments/decoder_testing/src")]

import host
from growth_reference import GrowthModel
from reference import Predecoder, matching
from surface_model import generate, seal
from telescope_circuit import build_context
from vectors import packet_rows, timed_decode


def main():
    rates = (.001, .003, .005, .007)
    seeds = list(range(22111501, 22111505))
    shots = 10_000
    build = SURFACE / 'build/d15-tilt-growth-full-ram'
    profile = SURFACE / 'experiments/d15-weighted-growth/d15z-growth-cap4000.json'
    bitstream = SURFACE / 'build/gowin-images/surface-d15z-tilt-growth-27mhz-debug-gowin.fs'
    image = json.loads((build / 'graph.json').read_bytes())
    tag, budget = seal(image), image['growth_cycles']
    assert tag == seal(json.loads(profile.read_bytes())) == 0x592e6076 and budget == 4000
    front = Predecoder(image)
    growth = GrowthModel(build / 'growth_graph.txt', build / 'sim/growth_model.dll')
    config = {
        'distance': 15, 'basis': 'z', 'rounds': 15,
        'width': image['width'], 'rows': image['rounds'] + 1,
        'max_cycles': front.plan.max_clocks + budget + 1,
        'baud_rate': 500_000, 'clock_hz': 27_000_000, 'image_tag': tag,
        'sampling': 'Surface Telescope full joint Stim rotated memory Z circuit',
        'profile': profile.relative_to(ROOT).as_posix(),
        'profile_sha256': sha256(profile.read_bytes()).hexdigest(),
        'bitstream': bitstream.relative_to(ROOT).as_posix(),
        'bitstream_sha256': sha256(bitstream.read_bytes()).hexdigest(),
        'cohorts': [],
    }
    arrays = {'rates': np.array(rates), 'seeds': np.array(seeds)}
    for index, (p, seed) in enumerate(zip(rates, seeds, strict=True)):
        context, mapping = build_context(image, p)
        if index and not np.array_equal(arrays['memory_indices'], mapping):
            raise ValueError('FPGA detector order changed between cohorts')
        arrays['memory_indices'] = mapping
        events, observables = context.circuit.compile_detector_sampler(seed=seed).sample(
            shots, bit_packed=True, separate_observables=True,
        )
        truth = observables[:, 0]
        memory = np.unpackbits(events, axis=1, count=context.num_detectors, bitorder='little')[:, mapping] != 0
        arrays[f'events_{index}'] = events
        arrays[f'truth_{index}'] = truth
        expected = front.decode(memory)
        growth.recover(expected, memory, budget)
        status = np.where(expected.accepted, 1, 3)
        cpu = matching(generate(15, 'z', 15, p))
        predictions, times = timed_decode(
            lambda row: cpu.decode(row)[0], memory, lambda rows: cpu.decode_batch(rows)[:, 0],
        )
        packets = (
            host.frame(host.REQUEST.pack(host.COMMAND['tilt'], index * shots + i, tag)
                       + host.pack(row, image['width']))
            for i, row in enumerate(memory)
        )
        config['cohorts'].append({
            'p': p, 'seed': seed,
            'shots': packet_rows(packets, truth, predictions, times, status, expected.clocks),
        })
        print(f'd15 p={p:g}: prepared {shots:,} packets', flush=True)

    versions = {'stim': stim.__version__, 'numpy': np.__version__, 'pymatching': pymatching.__version__}
    payload = json.dumps({'surface': config, 'versions': versions}, separators=(',', ':'))
    (OUT / 'd15_fixtures.json').write_text(payload, encoding='utf-8')
    np.savez_compressed(OUT / 'd15_joint.npz', **arrays)
    page = (BENCH / 'uart_benchmark.html').read_text(encoding='utf-8')
    page = page.replace('<!-- BENCHMARK_FIXTURES -->',
                        f'<script id="embedded-fixtures" type="application/json">{payload}</script>')
    page = page.replace('Surface Telescope d7</option>', 'Surface Telescope d15 weighted growth 27 MHz</option>')
    page = page.replace("current = {mode, p: cohort.p", "current = {mode, image_tag: config.image_tag, p: cohort.p")
    page = page.replace("'fpga_frame_bit',", "'fpga_frame_bit', 'fpga_wire_hex',")
    page = page.replace('frameBit: bit, residual};',
                        "frameBit: bit, residual, wireHex: wire.map(byte => byte.toString(16).padStart(2, '0')).join('')};")
    page = page.replace('residualHex, packet.frameBit ?? null]);',
                        'residualHex, packet.frameBit ?? null, packet.wireHex ?? null]);')
    page = page.replace('fpga-uart-measurements.json', 'telescope-d15-real-2026-11-22.json')
    page = page.replace('if (window.showSaveFilePicker) {', 'if (false) {')
    (OUT / 'd15.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()

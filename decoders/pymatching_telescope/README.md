# PyMatching telescope

A distance-15 FPGA-assisted tensor decoder using PyMatching 2.4.0. The FPGA front
end handles local syndromes and returns a logical prediction. Deferred shots
retain the full joint syndrome for correlated matching and a two-start
logical-sector tensor search.

## Performance

Fresh physical-board results use d15, 15 rounds, Surface memory-Z, physical
p = 0.005, weighted growth at 27 MHz and plain telescope routing.

| Metric | Result |
| --- | ---: |
| Median excluding UART | **53.1 µs** |
| p99 excluding UART | **5.480 ms** |
| Observed logical error rate | **0.19% — 19 / 10,000 shots** |
| Median including UART | **16.920 ms** |
| FPGA defers | **574 / 10,000 (5.74%)** |

The separate paired model cohort recorded 81.2 µs median and
4/2,048 telescope errors, versus 334.6 µs and
8/2,048 for ordinary PyMatching.
[All regenerated tables and raw sources](../../TENSOR_TELESCOPE_DECK_V2_NUMBERS.md).

## Install and run

Python 3.11 or newer is required. From the repository root:

```shell
python -m pip install -e ./decoders/pymatching_telescope
python decoders/pymatching_telescope/telescope.py --shots 2
```

The installation includes NumPy, SciPy, Stim and PyMatching 2.4.0. Verified native
binaries are bundled for Windows x64. Source builds require GCC with C++20.

The installed entry points are also available:

```shell
python -m pymatching_tensor_decoder.telescope --shots 2
pymatching-telescope --shots 2
```

## FPGA and input files

The bundled default profile is `d15z-gap3-81mhz`. Software front-end replay is the
default. For a connected FPGA, install serial support and select its port:

```shell
python -m pip install -e "./decoders/pymatching_telescope[fpga]"
python decoders/pymatching_telescope/telescope.py --port COM6 --events joint-events.npy
```

`--events` accepts packed full-basis detector rows in a NumPy `.npy` file.
`--policy plain` returns valid FPGA accepts directly and decodes original joint
syndromes on defers. `--policy guarded` adds the residual cut gate before the
tensor tail. `--defer-batch N` queues unresolved shots for batched decoding.

## GPU options

CPU is the default. `--device cuda` enables NVIDIA FP64 conditioning;
`--device cuda-batch` batches initial conditioning work. `--cuda-correlated`
selects the native CUDA correlated matcher. CUDA requires an NVIDIA GPU, its
driver and the CUDA runtime compiler. `--tensor-workers N` sets CPU workers for
batched conditioned solves.

## Python interface

The standalone tensor factory is `pymatching_tensor_decoder:build_decoder`.
It accepts a `DecoderContext` and exposes `decode_batch(events)` with read-only,
little-endian packed `uint8` detector inputs. The FPGA pipeline is exposed by
`pymatching_tensor_decoder.telescope.TelescopeDecoder`.

This package has its own Python namespace and native cache and can be installed
alongside the [independent telescope](../telescope/README.md).
[Third-party attribution](THIRD_PARTY.md) Ã‚· [Results](results/README.md) Ã‚·
[Implementation provenance](results/provenance/).

## Measured weighted-growth image

The performance configuration uses the [27 MHz image](hardware/surface-d15z-growth-27mhz.fs)
and its [profile](hardware/d15z-growth-cap4000.json). The quick-start default is
the gap3 profile. With the measured image in SRAM, run from the repository root:

```shell
python decoders/pymatching_telescope/telescope.py --profile decoders/pymatching_telescope/hardware/d15z-growth-cap4000.json --port COM6 --baud 500000 --events joint-events.npy
```

The [hardware manifest](hardware/manifest.json) pairs both images with their
profiles, clocks and hashes.

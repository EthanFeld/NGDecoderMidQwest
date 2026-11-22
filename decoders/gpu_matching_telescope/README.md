# GPU matching Telescope

This distance-15 Telescope uses the preserved native CUDA correlated matcher
from the PyMatching Telescope. FPGA accepts supply the logical bit. Deferred
shots retain the original packed syndrome from both check bases for the tensor
backend.

## Install and run

From the repository root, use Python 3.11 or newer with an NVIDIA GPU, a working
driver, and a CUDA toolkit with NVRTC. On Windows, set `CUDA_PATH` to the toolkit
directory if the installer did not set it. The package includes verified x64
Windows CPU support libraries. Source builds require GCC with C++20.

```sh
python -m pip install -e ./decoders/gpu_matching_telescope
gpu-matching-telescope --shots 4 --noise 0.005
```

This command uses software front-end replay for the bundled gap3 profile. It
loads the CUDA matcher during startup and reports one JSON object per shot.
`python -m gpu_matching_telescope.telescope` and the local `telescope.py` wrapper
accept the same arguments.

## Decode with the FPGA

Install serial support and load the matching bitstream into the Tang Nano 20K's
SRAM. The `hardware` directory includes the 81 MHz gap3 image (the default
profile at 1 Mbaud) and the 27 MHz weighted-growth image (500 kbaud). The
[manifest](hardware/manifest.json) records the paired profiles and SHA-256 hashes.
For the 27 MHz image, run:

```sh
python -m pip install -e "./decoders/gpu_matching_telescope[fpga]"
gpu-matching-telescope --profile decoders/gpu_matching_telescope/hardware/d15z-growth-cap4000.json --port COM6 --baud 500000 --events joint-events.npy
```

`--events` expects a NumPy array of little-endian packed `uint8` rows with 3,360
detector bits per shot. `--defer-batch N` streams accepted shots and batches up
to `N` deferred shots for the backend. The Python equivalent is
`gpu_matching_telescope.telescope.TelescopeDecoder`.

## Matching and tensor execution

Native CUDA runs the initial two-pass correlated Sparse Blossom match. The tensor
search and conditioned solves use the CPU. `--device cuda-batch` also enables
CUDA weight conditioning. The legacy per-shot CPU matching fallback handles any
row for which the CUDA matcher returns no correction. CUDA startup and driver
errors are reported as errors. Select CPU correlation explicitly with
`--no-cuda-correlated` and CPU weight conditioning with `--device cpu`.

[Third-party attribution](THIRD_PARTY.md)

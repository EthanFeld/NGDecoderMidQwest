# Telescope

Distance-15 FPGA-assisted tensor decoding with an independent C++ correlated
matching engine. The FPGA resolves local syndromes. Deferred shots retain the
original syndrome from both check bases, and FPGA residuals provide an initial
matching guess.

## Install and run

Python 3.11+, NumPy, SciPy, Stim and a C++20 compiler are required. From the
repository root:

```shell
python -m pip install -e ./decoders/telescope
python decoders/telescope/telescope.py --shots 2
```

The native engine builds on first use and caches its compiled library. GCC is
selected through `CXX`; the Windows MSYS2 installation is detected automatically.

## Input and FPGA connection

The default profile is `d15z-gap3-81mhz`, with 15 rounds and Z-memory readout.
Software replay supplies the FPGA responses by default. A connected board uses
the bundled [81 MHz image](bitstreams/surface-d15z-gap3-81mhz.fs):

```shell
python decoders/telescope/telescope.py --port COM6 --events joint-events.npy
```

`--events` loads a read-only NumPy array of packed full-basis syndromes. Inputs
contain 3,360 detector bits per shot; the FPGA receives 1,792 memory-basis bits.
Valid FPGA accepts supply the final logical bit. Defers use the original joint
syndrome and the tensor search. `--policy guarded` adds a residual cut gate.

## Parallel execution

The matcher runs both check bases concurrently and uses four persistent path
search workers per basis. `--basis-workers` and `--path-workers` set these counts.
`--tensor-workers` controls independent conditioned solves in a batch.
`--defer-batch N` queues deferred shots for batched processing. NVIDIA FP64 weight
conditioning is available through `--device cuda` and `--device cuda-batch`.

## Source and results

| Component | Location |
| --- | --- |
| Telescope routing and CLI | [telescope.py](telescope.py) |
| Matching integration and FPGA hints | [owned_backend.py](owned_backend.py) |
| Independent C++ matching | [native/](native/) |
| Tensor search and fault model | [tensor_decoder/src/tensor_decoder/](tensor_decoder/src/tensor_decoder/) |
| Circuit and detector mapping | [telescope_circuit.py](telescope_circuit.py) |
| FPGA profile and image | [profiles/](profiles/), [bitstreams/](bitstreams/) |

The [architecture](docs/ARCHITECTURE.md) describes the shared telescope pipeline.
The [UART protocol](docs/UART.md) defines board communication. Profiling and replay
records are under [tensor_decoder/results/](tensor_decoder/results/README.md).
The project headline measurements belong to the
[PyMatching telescope](../pymatching_telescope/results/README.md).

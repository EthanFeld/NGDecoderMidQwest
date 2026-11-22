# Distance-15 weighted growth

Archived weighted-growth FPGA experiment for distance-15 Z memory, including build patches, the compiled profile and the 27 MHz FPGA image. A recorded 4,096-shot board cohort at p=0.005 accepted 3,875 shots, deferred 221 and produced six logical errors on FPGA accepts. Board status and cycle telemetry differ from the corresponding cycle model; the raw capture records preserve those differences.

`build.ps1` generates the design, `retime27.ps1` creates the 27 MHz route and `compare_telescope.py` scores captured replies with the PyMatching tensor fallback. `benchmark_tensor_tail.py` and `benchmark_correlated_cuda.py` measure the software stages. These scripts retain their original experiment settings; choose shot counts explicitly when running a new comparison.

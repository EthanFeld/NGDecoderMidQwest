# Tang Nano 20K distance-15 decoder

Distance-15 Z-memory FPGA decoder using isolated-pair pruning, bounded proximity components, boundary and repair rules. Deferred shots use correlated PyMatching on the CPU. `reference.cpp` implements the graph-rule software reference; `rtl/` contains the FPGA source. UART, fast-UART, low-defer and GPIO images are retained in `bitstreams/`, with model data in `data/` and measurements in `results/`.

From this folder, install dependencies with `python -m pip install -r requirements.txt` and inspect `python run.py --help`. The `cpu`, `reference`, `decode` and `score` commands process packed input files; the included sample is `examples/shots.npy`. The software reference requires a GCC-compatible C++17 compiler. UART board operation also requires PySerial; hardware image loading uses an appropriate Tang Nano 20K programmer.

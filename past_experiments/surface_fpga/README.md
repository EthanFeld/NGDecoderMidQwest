# Surface FPGA experiments

Archived Tang Nano 20K implementations include distance-3 exact logical-MAP lookup, distance-7/distance-15 prior-tilt pairing, local residual rescue and distance-7 weighted cluster growth. `profiles/` and `bitstreams/manifest.json` identify the shipped models and FPGA images. The distance-15 weighted-growth implementation is retained under `experiments/d15-weighted-growth/`.

The software reference entry point is `reference.py`; `surface_model.py` generates models and `Makefile` builds the RTL with OSS CAD Suite. Run scripts from this folder. The maintained distance-15 gap3 telescope and its default FPGA profile live in [decoders/telescope](../../decoders/telescope/README.md). Raw experiment results are retained as JSON and CSV files here.

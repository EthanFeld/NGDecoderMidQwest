# MORPH-31

Rotated surface-code decoder using sparse collision clustering, union by size and forest peeling. The optional 4 KiB motif table removes selected local pairs before clustering. Source, trained tables, circuit samples and recorded comparisons are retained here; `hardware/` contains interface documentation, without RTL or board integration.

Python 3.11+, NumPy, Stim and PyMatching are required. From the repository root, install with `python -m pip install -e ./past_experiments/morph31`. Run `python -m morph31 --help` for graph export, training and benchmark options. The qecbench factory is `morph31.integration:build_decoder`; model tables are in `models/` and measurements in `results/`.

# Decoder records

Raw measurements, implementation fingerprints and saved replay inputs are
retained in this directory.

- `runs/owned_profile_20261122T171631759966Z/` contains the independent matcher's
  fixed inputs, source snapshots, profiling traces and successive replay results.
- `runs/owned_matching_initial.json` records the initial independent-engine comparison.
- `validation/` and `latency.json` describe the earlier standalone PyMatching
  tensor backend on the standard Stim circuit.
- `runs/d15_telescope_*` contains the earlier PyMatching telescope measurements.

The maintained [PyMatching telescope results](../../../pymatching_telescope/results/README.md)
present the project headline figures and their raw sources. Historical source
paths inside records retain their original values.

# PyMatching telescope results

The current project measurements are in the [regenerated deck tables](../../../TENSOR_TELESCOPE_DECK_V2_NUMBERS.md) and [raw run directory](../../../measurements/2026-11-22-deck-v2/).
The records below describe the earlier gap3 configuration.

## Telescope configuration

Distance 15, 15 measurement rounds, Surface memory-Z joint circuit, physical
noise p = 0.005, `d15z-gap3-81mhz` FPGA profile and `plain` policy. Every valid
FPGA accept supplies the logical prediction; defers use the original 3,360-bit
joint syndrome. The FPGA sees 1,792 memory-basis detector bits.

## Accuracy

| Cohort | Front end | Seed | Errors / shots | Observed LER |
| --- | --- | ---: | ---: | ---: |
| [Confirmation](telescope/d15_telescope_confirm_p005.json) | FPGA cycle model | 51928032 | **7 / 12,000** | **0.0583%** |
| [Holdout](telescope/d15_telescope_holdout_p005.json) | FPGA cycle model | 51927032 | 0 / 12,000 | 0% |
| [Captured-board replay](telescope/d15_telescope_board_cuda_batch_confirm.json) | Physical FPGA replies | 51915032 | 1 / 1,024 | 0.0977% |

The confirmation cohort records the gap3 accuracy comparison. Cohorts are reported
separately with their implementation fingerprints and confidence intervals in
the linked records. Zero observed errors in the holdout gives a 95% Wilson upper
bound of 0.0320%.

## Latency

The [captured-board replay](telescope/d15_telescope_board_cuda_batch_confirm.json)
times 256 paired single-shot inputs at p = 0.005:

| Backend | Median excluding UART | p99 excluding UART | Median including UART |
| --- | ---: | ---: | ---: |
| CPU | **0.994 ms** | **6.270 ms** | 17.942 ms |
| CUDA-batch | 1.050 ms | 6.086 ms | 17.971 ms |

Completed single-shot time includes detector projection, FPGA core time and
host decoding. The UART-inclusive column includes the captured serial
request/reply time. Construction, sampling and warmup are excluded. The captured
FPGA deferred 944 / 1,024 shots (92.1875%); median core time was 0.963 Ã‚µs.

These timing and accuracy figures come from separate cohorts. The measurement
host ran Windows 11 with Stim 1.16.0 and NumPy 2.5.5. Raw records include source
and native-library hashes.

## Standalone tensor measurements

The [validation records](validation/validation.json) and [CPU latency record](latency.json)
cover the standalone tensor backend on the standard Stim rotated memory-Z
circuit at d15, 15 rounds and p = 0.005:

| Metric | Result |
| --- | ---: |
| Logical errors | 89 / 400,000 |
| Observed LER | 0.02225% |
| CPU median / p99 | 1.235 / 8.863 ms |

This is a separate circuit and tensor-only workload from the telescope results
above. [Provenance records](provenance/) retain native identities, source hashes
and the package restoration checks.

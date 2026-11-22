# IN PROGRESS — Tensor Telescope — Deck v2 numbers

Generated: 2026-11-22T19:16:41.137634+00:00

M = newly recorded cycle-model/software cohorts. B = newly captured physical-board replies with paired CPU replay. LER is per complete memory experiment. All latency values below are microseconds; single-shot medians exclude UART unless stated.

Zero-error measurements retain LER = 0 in the count tables. Log plots use the 95% Wilson upper bound and an upper-limit marker.

IN PROGRESS: `Pending` marks a measurement that has not completed. This draft is not the final numerical appendix.

## Slide 2 — LER versus speed

d15, 15 rounds, Z memory, physical p = 0.005; source M.

| Distance | Physical p | Decoder | Errors | Shots | LER | Median µs | Timed shots |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d15 | 0.5% | FPGA only (zero on defer) | 8 | 2,048 | 0.39062% | Pending | 0 |
| d15 | 0.5% | Tensor only | 0 | 2,048 | 0% | Pending | 0 |
| d15 | 0.5% | Telescope | 4 | 2,048 | 0.19531% | Pending | 0 |
| d15 | 0.5% | PyMatching | 8 | 2,048 | 0.39062% | Pending | 0 |
| d15 | 0.5% | Correlated PyMatching | 0 | 2,048 | 0% | Pending | 0 |
| d15 | 0.5% | Relay-BP | 15 | 64 | 23.4375% | Pending | 0 |

Tensor-only median / telescope median: **Pending**. PyMatching median / telescope median: **Pending**.

## Slide 3 — Pipeline

A valid FPGA accept supplies the logical bit. Each defer sends the original joint syndrome to the tensor backend.

| M, d15 at p = 0.005 | Value |
| --- | --- |
| FPGA defers / shots | 13 / 2,048 |
| FPGA defer rate | 0.63477% |
| Tensor-only → telescope median µs | Pending → Pending |

Original-versus-residual probe: d15 gap3 rescue, p = 0.005, 4,096 newly sampled Surface joint-circuit shots; cycle-model frontend.

| Input / routing | Errors | Shots | LER | Errors on defers | Errors on accepts |
| --- | --- | --- | --- | --- | --- |
| Original joint syndrome, tensor only | Pending | Pending | Pending | Pending | Pending |
| Residual memory checks + conjugate checks + FPGA frame, all shots | Pending | Pending | Pending | Pending | Pending |
| FPGA accepts + original-syndrome fallback | 1 | 4,096 | 0.02441% | 1 | 0 |
| FPGA accepts + residual-syndrome fallback | 152 | 4,096 | 3.71094% | 152 | 0 |

Probe seed: 94629701; FPGA accepts/defers: 255/3,841. Source: [residual_comparison.json](measurements/2026-11-22-deck-v2/standard_stim/residual_comparison.json).

## Slide 4 — Initial FPGA decoding

| Distance | Physical edge families | Virtual bridge families | First pass clocks | Two-pass clocks |
| --- | --- | --- | --- | --- |
| d7 | 6 | 5 | 31 | 62 |
| d15 | 6 | 5 | 39 | 78 |

The retry swaps the first two shape priorities. A completed first pass can answer before the retry.

Source: [stencils.py](decoders/pymatching_telescope/src/pymatching_tensor_decoder/telescope/stencils.py) and [reference.py](decoders/pymatching_telescope/src/pymatching_tensor_decoder/telescope/reference.py).

## Slide 5 — Growth clusters

$w = \log((1-q)/q)$

$t = 2\max(2,\lfloor Rw/w_{\min}+b+0.5\rfloor)$

Here b is the boundary bias, applied only to boundary edges; q uses the compiled growth prior.

| Profile | Resolution R | Boundary bias b | Growth watchdog clocks | Initial clocks + handoff + watchdog | Clock MHz | Core deadline µs |
| --- | --- | --- | --- | --- | --- | --- |
| d7 dual growth | 2.75 | 0 | 368 | 431 | 81 | 5.321 |
| d15 weighted growth | 2 | 0 | 4000 | 4079 | 27 | 151.074 |

Sources: [growth.py](past_experiments/surface_fpga/growth.py); [d7z-dual-p007-cut0-81mhz.json](past_experiments/surface_fpga/profiles/d7z-dual-p007-cut0-81mhz.json); [d15z-growth-cap4000.json](past_experiments/surface_fpga/experiments/d15-weighted-growth/d15z-growth-cap4000.json).

## Slide 6 — Matching

Tensor routing uses **path score < 3d**; at d15 the threshold is **45**.

| Measurement | Board defers | Mean µs / defer | Scope |
| --- | --- | --- | --- |
| Historical CPU correlated matching | 221 | 459.624 | 221-shot batch, three repeats; fresh measurement pending |

Historical source: [d15_correlated_cuda_resident_72621005.json](decoders/telescope/tensor_decoder/results/runs/d15_correlated_cuda_resident_72621005.json).

Routing source: [decoder.py](decoders/pymatching_telescope/src/pymatching_tensor_decoder/decoder.py). Matching uses the PyMatching backend in these measurements.

## Slide 7 — Tensor and refinement

| Rule | Current setting |
| --- | --- |
| d7 bounded frontier beam | 128 |
| Conditioned correction accepted when gain ≥ | 8.0 |
| Sector competition when \|gap\| ≤ | 8.0 |
| Also compete when | conditioned and initial matching predictions disagree |
| Sector choice accepted when \|log-sum-exp margin\| ≥ | 3.0 |
| Alternating iterations per seed/sector | 2 |
| Reverse seed | true |

$m=\operatorname{logsumexp}_{c\in C_1}(-\mathrm{cost}(c))-\operatorname{logsumexp}_{c\in C_0}(-\mathrm{cost}(c))$.

The score sums retained candidate witnesses. Sources: [config.py](decoders/pymatching_telescope/src/pymatching_tensor_decoder/config.py); [policy.py](decoders/pymatching_telescope/src/pymatching_tensor_decoder/_small/policy.py); [decoder.py](decoders/pymatching_telescope/src/pymatching_tensor_decoder/decoder.py).

Separate backend validation: standard Stim rotated-memory Z circuit, d15, 15 rounds, p = 0.005.

| Decoder | Errors | Shots | LER | Status |
| --- | --- | --- | --- | --- |
| Tensor only | 92 | 400,000 | 0.023% | Complete |
| Correlated PyMatching | 140 | 400,000 | 0.035% | Complete |

Paired fixes/regressions versus correlated matching: **61/13**. Target shots: 400,000. Seeds: 94628701, 94628702, 94628703, 94628704.

Source: [comparison.json](measurements/2026-11-22-deck-v2/standard_stim/comparison.json).

## Slide 8 — LER versus distance

Source M; d7 and d15 at physical p = 0.003 and 0.0075. Relay-BP uses the shot count shown in its own row.

| Distance | Physical p | Decoder | Errors | Shots | LER | Log-plot value / limit |
| --- | --- | --- | --- | --- | --- | --- |
| d7 | 0.3% | FPGA only (zero on defer) | Pending | Pending | Pending | Pending |
| d7 | 0.3% | Tensor only | Pending | Pending | Pending | Pending |
| d7 | 0.3% | Telescope | Pending | Pending | Pending | Pending |
| d7 | 0.3% | PyMatching | Pending | Pending | Pending | Pending |
| d7 | 0.3% | Correlated PyMatching | Pending | Pending | Pending | Pending |
| d7 | 0.3% | Relay-BP | Pending | Pending | Pending | Pending |
| d15 | 0.3% | FPGA only (zero on defer) | 0 | 2,048 | 0% | < 0.18722% (95% upper) |
| d15 | 0.3% | Tensor only | 0 | 2,048 | 0% | < 0.18722% (95% upper) |
| d15 | 0.3% | Telescope | 0 | 2,048 | 0% | < 0.18722% (95% upper) |
| d15 | 0.3% | PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) |
| d15 | 0.3% | Correlated PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) |
| d15 | 0.3% | Relay-BP | 4 | 64 | 6.25% | 6.25% |
| d7 | 0.75% | FPGA only (zero on defer) | Pending | Pending | Pending | Pending |
| d7 | 0.75% | Tensor only | Pending | Pending | Pending | Pending |
| d7 | 0.75% | Telescope | Pending | Pending | Pending | Pending |
| d7 | 0.75% | PyMatching | Pending | Pending | Pending | Pending |
| d7 | 0.75% | Correlated PyMatching | Pending | Pending | Pending | Pending |
| d7 | 0.75% | Relay-BP | Pending | Pending | Pending | Pending |
| d15 | 0.75% | FPGA only (zero on defer) | 136 | 2,048 | 6.64062% | 6.64062% |
| d15 | 0.75% | Tensor only | 18 | 2,048 | 0.87891% | 0.87891% |
| d15 | 0.75% | Telescope | 83 | 2,048 | 4.05273% | 4.05273% |
| d15 | 0.75% | PyMatching | 75 | 2,048 | 3.66211% | 3.66211% |
| d15 | 0.75% | Correlated PyMatching | 26 | 2,048 | 1.26953% | 1.26953% |
| d15 | 0.75% | Relay-BP | 22 | 64 | 34.375% | 34.375% |

## Slide 9 — LER versus physical error rate

Source M; full d15 sweep. FPGA-only predictions use logical zero on a defer. Defer is an FPGA frontend property; software-only baselines decode every input.

| Physical p | Decoder | Errors | Shots | LER | Log-plot value / limit | FPGA defers |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1% | FPGA only (zero on defer) | 0 | 2,048 | 0% | < 0.18722% (95% upper) | 0/2,048 (0%) |
| 0.1% | Tensor only | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.1% | Telescope | 0 | 2,048 | 0% | < 0.18722% (95% upper) | 0/2,048 (0%) |
| 0.1% | PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.1% | Correlated PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.1% | Relay-BP | 1 | 64 | 1.5625% | 1.5625% | — |
| 0.2% | FPGA only (zero on defer) | 0 | 2,048 | 0% | < 0.18722% (95% upper) | 0/2,048 (0%) |
| 0.2% | Tensor only | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.2% | Telescope | 0 | 2,048 | 0% | < 0.18722% (95% upper) | 0/2,048 (0%) |
| 0.2% | PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.2% | Correlated PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.2% | Relay-BP | 1 | 64 | 1.5625% | 1.5625% | — |
| 0.3% | FPGA only (zero on defer) | 0 | 2,048 | 0% | < 0.18722% (95% upper) | 0/2,048 (0%) |
| 0.3% | Tensor only | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.3% | Telescope | 0 | 2,048 | 0% | < 0.18722% (95% upper) | 0/2,048 (0%) |
| 0.3% | PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.3% | Correlated PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.3% | Relay-BP | 4 | 64 | 6.25% | 6.25% | — |
| 0.4% | FPGA only (zero on defer) | 1 | 2,048 | 0.04883% | 0.04883% | 0/2,048 (0%) |
| 0.4% | Tensor only | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.4% | Telescope | 1 | 2,048 | 0.04883% | 0.04883% | 0/2,048 (0%) |
| 0.4% | PyMatching | 1 | 2,048 | 0.04883% | 0.04883% | — |
| 0.4% | Correlated PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.4% | Relay-BP | 7 | 64 | 10.9375% | 10.9375% | — |
| 0.5% | FPGA only (zero on defer) | 8 | 2,048 | 0.39062% | 0.39062% | 13/2,048 (0.63477%) |
| 0.5% | Tensor only | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.5% | Telescope | 4 | 2,048 | 0.19531% | 0.19531% | 13/2,048 (0.63477%) |
| 0.5% | PyMatching | 8 | 2,048 | 0.39062% | 0.39062% | — |
| 0.5% | Correlated PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) | — |
| 0.5% | Relay-BP | 15 | 64 | 23.4375% | 23.4375% | — |
| 0.6% | FPGA only (zero on defer) | 27 | 2,048 | 1.31836% | 1.31836% | 38/2,048 (1.85547%) |
| 0.6% | Tensor only | 2 | 2,048 | 0.09766% | 0.09766% | — |
| 0.6% | Telescope | 14 | 2,048 | 0.68359% | 0.68359% | 38/2,048 (1.85547%) |
| 0.6% | PyMatching | 16 | 2,048 | 0.78125% | 0.78125% | — |
| 0.6% | Correlated PyMatching | 6 | 2,048 | 0.29297% | 0.29297% | — |
| 0.6% | Relay-BP | 22 | 64 | 34.375% | 34.375% | — |
| 0.7% | FPGA only (zero on defer) | 87 | 2,048 | 4.24805% | 4.24805% | 87/2,048 (4.24805%) |
| 0.7% | Tensor only | 7 | 2,048 | 0.3418% | 0.3418% | — |
| 0.7% | Telescope | 46 | 2,048 | 2.24609% | 2.24609% | 87/2,048 (4.24805%) |
| 0.7% | PyMatching | 33 | 2,048 | 1.61133% | 1.61133% | — |
| 0.7% | Correlated PyMatching | 13 | 2,048 | 0.63477% | 0.63477% | — |
| 0.7% | Relay-BP | 14 | 64 | 21.875% | 21.875% | — |
| 0.75% | FPGA only (zero on defer) | 136 | 2,048 | 6.64062% | 6.64062% | 111/2,048 (5.41992%) |
| 0.75% | Tensor only | 18 | 2,048 | 0.87891% | 0.87891% | — |
| 0.75% | Telescope | 83 | 2,048 | 4.05273% | 4.05273% | 111/2,048 (5.41992%) |
| 0.75% | PyMatching | 75 | 2,048 | 3.66211% | 3.66211% | — |
| 0.75% | Correlated PyMatching | 26 | 2,048 | 1.26953% | 1.26953% | — |
| 0.75% | Relay-BP | 22 | 64 | 34.375% | 34.375% | — |
| 0.8% | FPGA only (zero on defer) | 223 | 2,048 | 10.88867% | 10.88867% | 195/2,048 (9.52148%) |
| 0.8% | Tensor only | 24 | 2,048 | 1.17188% | 1.17188% | — |
| 0.8% | Telescope | 127 | 2,048 | 6.20117% | 6.20117% | 195/2,048 (9.52148%) |
| 0.8% | PyMatching | 133 | 2,048 | 6.49414% | 6.49414% | — |
| 0.8% | Correlated PyMatching | 35 | 2,048 | 1.70898% | 1.70898% | — |
| 0.8% | Relay-BP | 25 | 64 | 39.0625% | 39.0625% | — |
| 0.9% | FPGA only (zero on defer) | 369 | 2,048 | 18.01758% | 18.01758% | 360/2,048 (17.57812%) |
| 0.9% | Tensor only | 69 | 2,048 | 3.36914% | 3.36914% | — |
| 0.9% | Telescope | 205 | 2,048 | 10.00977% | 10.00977% | 360/2,048 (17.57812%) |
| 0.9% | PyMatching | 226 | 2,048 | 11.03516% | 11.03516% | — |
| 0.9% | Correlated PyMatching | 101 | 2,048 | 4.93164% | 4.93164% | — |
| 0.9% | Relay-BP | 29 | 64 | 45.3125% | 45.3125% | — |
| 1% | FPGA only (zero on defer) | Pending | Pending | Pending | Pending | Pending/Pending (Pending) |
| 1% | Tensor only | Pending | Pending | Pending | Pending | — |
| 1% | Telescope | Pending | Pending | Pending | Pending | Pending/Pending (Pending) |
| 1% | PyMatching | Pending | Pending | Pending | Pending | — |
| 1% | Correlated PyMatching | Pending | Pending | Pending | Pending | — |
| 1% | Relay-BP | Pending | Pending | Pending | Pending | — |

## Slide 10 — Defer rate

M: d15 cycle-model cohorts.

| Physical p | Defers | Shots | Defer rate |
| --- | --- | --- | --- |
| 0.1% | 0 | 2,048 | 0% |
| 0.2% | 0 | 2,048 | 0% |
| 0.3% | 0 | 2,048 | 0% |
| 0.4% | 0 | 2,048 | 0% |
| 0.5% | 13 | 2,048 | 0.63477% |
| 0.6% | 38 | 2,048 | 1.85547% |
| 0.7% | 87 | 2,048 | 4.24805% |
| 0.75% | 111 | 2,048 | 5.41992% |
| 0.8% | 195 | 2,048 | 9.52148% |
| 0.9% | 360 | 2,048 | 17.57812% |
| 1% | Pending | Pending | Pending |

B: newly captured physical d15 replies. Status comes from the hardware response.

| Physical p | Defers | Shots | Defer rate | Telescope errors | Telescope LER | Median µs excl. UART | Median µs incl. UART |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1% | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| 0.3% | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| 0.5% | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| 0.7% | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

B: newly captured physical d7 replies.

| Physical p | Defers | Shots | Defer rate | Telescope errors | Telescope LER |
| --- | --- | --- | --- | --- | --- |
| 0.1% | 2 | 10,000 | 0.02% | 0 | 0% |
| 0.3% | 35 | 10,000 | 0.35% | 14 | 0.14% |
| 0.5% | 163 | 10,000 | 1.63% | 87 | 0.87% |
| 0.7% | 482 | 10,000 | 4.82% | 315 | 3.15% |

## Slide 11 — Decoding time versus distance

Source M; completed single-shot calls at p = 0.003 and 0.0075. FPGA/telescope values add cycle-model core time to measured CPU work. UART and queue fill are excluded.

| Distance | Physical p | Decoder | Median µs | p99 µs | Timed shots | PyMatching / telescope median |
| --- | --- | --- | --- | --- | --- | --- |
| d7 | 0.3% | FPGA only (zero on defer) | Pending | Pending | Pending | — |
| d7 | 0.3% | Tensor only | Pending | Pending | Pending | — |
| d7 | 0.3% | Telescope | Pending | Pending | Pending | Pending |
| d7 | 0.3% | PyMatching | Pending | Pending | Pending | — |
| d7 | 0.3% | Correlated PyMatching | Pending | Pending | Pending | — |
| d7 | 0.3% | Relay-BP | Pending | Pending | Pending | — |
| d15 | 0.3% | FPGA only (zero on defer) | Pending | Pending | 0 | — |
| d15 | 0.3% | Tensor only | Pending | Pending | 0 | — |
| d15 | 0.3% | Telescope | Pending | Pending | 0 | Pending |
| d15 | 0.3% | PyMatching | Pending | Pending | 0 | — |
| d15 | 0.3% | Correlated PyMatching | Pending | Pending | 0 | — |
| d15 | 0.3% | Relay-BP | Pending | Pending | 0 | — |
| d7 | 0.75% | FPGA only (zero on defer) | Pending | Pending | Pending | — |
| d7 | 0.75% | Tensor only | Pending | Pending | Pending | — |
| d7 | 0.75% | Telescope | Pending | Pending | Pending | Pending |
| d7 | 0.75% | PyMatching | Pending | Pending | Pending | — |
| d7 | 0.75% | Correlated PyMatching | Pending | Pending | Pending | — |
| d7 | 0.75% | Relay-BP | Pending | Pending | Pending | — |
| d15 | 0.75% | FPGA only (zero on defer) | Pending | Pending | 0 | — |
| d15 | 0.75% | Tensor only | Pending | Pending | 0 | — |
| d15 | 0.75% | Telescope | Pending | Pending | 0 | Pending |
| d15 | 0.75% | PyMatching | Pending | Pending | 0 | — |
| d15 | 0.75% | Correlated PyMatching | Pending | Pending | 0 | — |
| d15 | 0.75% | Relay-BP | Pending | Pending | 0 | — |

## Sources

- [results.json](measurements/2026-11-22-deck-v2/model/results.json)
- [results.csv](measurements/2026-11-22-deck-v2/model/results.csv)
- [score.json](measurements/2026-11-22-deck-v2/board/d7/score.json)
- [comparison.json](measurements/2026-11-22-deck-v2/standard_stim/comparison.json)
- [residual_comparison.json](measurements/2026-11-22-deck-v2/standard_stim/residual_comparison.json)
- [board_summary.csv](measurements/2026-11-22-deck-v2/board_summary.csv)

The model/software sweep uses the Surface joint circuit and the compiled d7/d15 growth profiles. The 400,000-shot backend study uses the separate standard Stim circuit. Accuracy, single-shot latency, amortized work and UART transport are separate measurements.

The board CSV exports five measured modes; Relay-BP is measured in the separate model/software sweep. FPGA-only LER uses logical zero on defers. Core and UART columns are actual hardware measurements. Host/telescope timing columns remain blank until measured; the FPGA-only row does not substitute core time for an unmeasured host-completion time.

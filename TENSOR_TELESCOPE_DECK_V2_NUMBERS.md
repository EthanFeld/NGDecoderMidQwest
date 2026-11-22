# Tensor Telescope — Deck v2 numbers

Generated: 2026-11-22T19:34:54.224958+00:00

M = newly recorded cycle-model/software cohorts. B = newly captured physical-board replies with paired CPU replay. LER is per complete memory experiment. All latency values below are microseconds; single-shot medians exclude UART unless stated.

Zero-error measurements retain LER = 0 in the count tables. Log plots use the 95% Wilson upper bound and an upper-limit marker.

## Slide 2 — LER versus speed

d15, 15 rounds, Z memory, physical p = 0.005; source M.

| Distance | Physical p | Decoder | Errors | Shots | LER | Median µs | Timed shots |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d15 | 0.5% | FPGA only (zero on defer) | 8 | 2,048 | 0.39062% | 89.089 | 204 |
| d15 | 0.5% | Tensor only | 0 | 2,048 | 0% | 1,304.1 | 204 |
| d15 | 0.5% | Telescope | 4 | 2,048 | 0.19531% | 81.189 | 204 |
| d15 | 0.5% | PyMatching | 8 | 2,048 | 0.39062% | 334.6 | 204 |
| d15 | 0.5% | Correlated PyMatching | 0 | 2,048 | 0% | 873 | 204 |
| d15 | 0.5% | Relay-BP | 15 | 64 | 23.4375% | 627,200.4 | 24 |

Tensor-only median / telescope median: **16.06×**. PyMatching median / telescope median: **4.12×**.

## Slide 3 — Pipeline

A valid FPGA accept supplies the logical bit. Each defer sends the original joint syndrome to the tensor backend.

| M, d15 at p = 0.005 | Value |
| --- | --- |
| FPGA defers / shots | 13 / 2,048 |
| FPGA defer rate | 0.63477% |
| Tensor-only → telescope median µs | 1,304.1 → 81.189 |

Original-versus-residual probe: d15 gap3 rescue, p = 0.005, 4,096 newly sampled Surface joint-circuit shots; cycle-model frontend.

| Input / routing | Errors | Shots | LER | Errors on defers | Errors on accepts |
| --- | --- | --- | --- | --- | --- |
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

| Measurement | Actual FPGA defers | Mean µs | Median µs | p99 µs |
| --- | --- | --- | --- | --- |
| B, d15 p = 0.005, completed correlated edge extraction | 574 | 812.243 | 805.2 | 1,366.528 |

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
| d7 | 0.3% | FPGA only (zero on defer) | 8 | 8,192 | 0.09766% | 0.09766% |
| d7 | 0.3% | Tensor only | 5 | 8,192 | 0.06104% | 0.06104% |
| d7 | 0.3% | Telescope | 6 | 8,192 | 0.07324% | 0.07324% |
| d7 | 0.3% | PyMatching | 6 | 8,192 | 0.07324% | 0.07324% |
| d7 | 0.3% | Correlated PyMatching | 5 | 8,192 | 0.06104% | 0.06104% |
| d7 | 0.3% | Relay-BP | 27 | 8,192 | 0.32959% | 0.32959% |
| d15 | 0.3% | FPGA only (zero on defer) | 0 | 2,048 | 0% | < 0.18722% (95% upper) |
| d15 | 0.3% | Tensor only | 0 | 2,048 | 0% | < 0.18722% (95% upper) |
| d15 | 0.3% | Telescope | 0 | 2,048 | 0% | < 0.18722% (95% upper) |
| d15 | 0.3% | PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) |
| d15 | 0.3% | Correlated PyMatching | 0 | 2,048 | 0% | < 0.18722% (95% upper) |
| d15 | 0.3% | Relay-BP | 4 | 64 | 6.25% | 6.25% |
| d7 | 0.75% | FPGA only (zero on defer) | 459 | 8,192 | 5.60303% | 5.60303% |
| d7 | 0.75% | Tensor only | 195 | 8,192 | 2.38037% | 2.38037% |
| d7 | 0.75% | Telescope | 297 | 8,192 | 3.62549% | 3.62549% |
| d7 | 0.75% | PyMatching | 310 | 8,192 | 3.78418% | 3.78418% |
| d7 | 0.75% | Correlated PyMatching | 224 | 8,192 | 2.73438% | 2.73438% |
| d7 | 0.75% | Relay-BP | 57 | 1,024 | 5.56641% | 5.56641% |
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
| 1% | FPGA only (zero on defer) | 557 | 2,048 | 27.19727% | 27.19727% | 547/2,048 (26.70898%) |
| 1% | Tensor only | 158 | 2,048 | 7.71484% | 7.71484% | — |
| 1% | Telescope | 362 | 2,048 | 17.67578% | 17.67578% | 547/2,048 (26.70898%) |
| 1% | PyMatching | 397 | 2,048 | 19.38477% | 19.38477% | — |
| 1% | Correlated PyMatching | 214 | 2,048 | 10.44922% | 10.44922% | — |
| 1% | Relay-BP | 32 | 64 | 50% | 50% | — |

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
| 1% | 547 | 2,048 | 26.70898% |

B: newly captured physical d15 replies. Status comes from the hardware response.

| Physical p | Defers | Shots | Defer rate | Telescope errors | Telescope LER | Median µs excl. UART | Median µs incl. UART |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1% | 2 | 10,000 | 0.02% | 0 | 0% | 26.089 | 16,843.9 |
| 0.3% | 80 | 10,000 | 0.8% | 0 | 0% | 32.848 | 16,922.5 |
| 0.5% | 574 | 10,000 | 5.74% | 19 | 0.19% | 53.063 | 16,920 |
| 0.7% | 1,514 | 10,000 | 15.14% | 206 | 2.06% | 97.319 | 16,956.1 |

Board source: [score.json](measurements/2026-11-22-deck-v2/board/d15/score.json).

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
| d7 | 0.3% | FPGA only (zero on defer) | 17.536 | 57.065 | 207 | — |
| d7 | 0.3% | Tensor only | 63.8 | 222 | 207 | — |
| d7 | 0.3% | Telescope | 16.859 | 65.465 | 207 | 1.09× |
| d7 | 0.3% | PyMatching | 18.4 | 51.4 | 207 | — |
| d7 | 0.3% | Correlated PyMatching | 36.2 | 111.8 | 207 | — |
| d7 | 0.3% | Relay-BP | 1,149.4 | 49,731.8 | 207 | — |
| d15 | 0.3% | FPGA only (zero on defer) | 47.885 | 106.289 | 192 | — |
| d15 | 0.3% | Tensor only | 571.3 | 1,108 | 192 | — |
| d15 | 0.3% | Telescope | 46.585 | 142.789 | 192 | 3.01× |
| d15 | 0.3% | PyMatching | 140 | 451.2 | 192 | — |
| d15 | 0.3% | Correlated PyMatching | 356.7 | 639.1 | 192 | — |
| d15 | 0.3% | Relay-BP | 192,654.2 | 624,447.9 | 24 | — |
| d7 | 0.75% | FPGA only (zero on defer) | 13.233 | 94.584 | 213 | — |
| d7 | 0.75% | Tensor only | 126.3 | 458,078.9 | 213 | — |
| d7 | 0.75% | Telescope | 15.209 | 538.621 | 213 | 2.1× |
| d7 | 0.75% | PyMatching | 32 | 134.8 | 213 | — |
| d7 | 0.75% | Correlated PyMatching | 78 | 265.8 | 213 | — |
| d7 | 0.75% | Relay-BP | 7,002.1 | 66,836.2 | 216 | — |
| d15 | 0.75% | FPGA only (zero on defer) | 132.085 | 233.559 | 214 | — |
| d15 | 0.75% | Tensor only | 7,616.7 | 56,769.6 | 214 | — |
| d15 | 0.75% | Telescope | 115.793 | 40,813.211 | 214 | 5.42× |
| d15 | 0.75% | PyMatching | 627.4 | 1,557.6 | 214 | — |
| d15 | 0.75% | Correlated PyMatching | 1,471.9 | 3,176.4 | 214 | — |
| d15 | 0.75% | Relay-BP | 602,365.9 | 1,095,997.1 | 26 | — |

## Sources

- [results.json](measurements/2026-11-22-deck-v2/model/results.json)
- [results.csv](measurements/2026-11-22-deck-v2/model/results.csv)
- [timing_scope.json](measurements/2026-11-22-deck-v2/model/timing_scope.json)
- [score.json](measurements/2026-11-22-deck-v2/board/d7/score.json)
- [score.json](measurements/2026-11-22-deck-v2/board/d15/score.json)
- [comparison.json](measurements/2026-11-22-deck-v2/standard_stim/comparison.json)
- [residual_comparison.json](measurements/2026-11-22-deck-v2/standard_stim/residual_comparison.json)
- [board_summary.csv](measurements/2026-11-22-deck-v2/board_summary.csv)

The model/software sweep uses the Surface joint circuit and the compiled d7/d15 growth profiles. The 400,000-shot backend study uses the separate standard Stim circuit. Accuracy, single-shot latency, amortized work and UART transport are separate measurements.

The board CSV exports five measured modes; Relay-BP is measured in the separate model/software sweep. FPGA-only LER uses logical zero on defers. Core and UART columns are actual hardware measurements. Host/telescope timings are measured separately; FPGA-only host-completion time is unmeasured, and its row reports core timing separately.

The requested model single-shot timing points are d15 at p = 0.003, 0.005 and 0.0075, and d7 at p = 0.003 and 0.0075. The complete eleven-rate d15 LER sweep requires accuracy at every rate.

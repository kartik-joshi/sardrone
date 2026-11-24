# Tiled detection (issue #1)

Generated from `tiled.csv` (predictions at commit 5e564cd) and `tiled_gain.csv`
(paired bootstrap, 1,000 resamples, commit 8d64240). Method, fixed before scoring:
1280 px tiles at full resolution overlapping by 256 px, plus the whole image at
1280 as before; boxes on inner cut lines dropped; NMS at IoU 0.5; top 300 boxes.
See `sardrone/train/tiled.py`.

## HERIDAL (101 images, 337 people, never trained on)

| Model | mAP@0.5 full | mAP@0.5 tiled | Gain (95 % paired interval) | Tiled better in | Max recall full / tiled |
| --- | --- | --- | --- | --- | --- |
| v0 | 0.184 | 0.310 | +0.126 (+0.075 to +0.179) | 100 % | 0.38 / 0.58 |
| v1 | 0.030 | 0.080 | +0.051 (+0.025 to +0.085) | 100 % | 0.16 / 0.20 |
| v4 | 0.231 | 0.352 | +0.120 (+0.067 to +0.175) | 100 % | 0.57 / 0.74 |
| v6 | 0.262 | 0.364 | +0.102 (+0.047 to +0.161) | 100 % | 0.60 / 0.74 |
| v7 | 0.213 | 0.356 | +0.141 (+0.083 to +0.205) | 100 % | 0.55 / 0.77 |
| v8 | 0.261 | 0.345 | +0.086 (+0.036 to +0.136) | 100 % | 0.59 / 0.72 |

## SARD test (1,387 images)

| Model | mAP@0.5 full | mAP@0.5 tiled | Gain (95 % paired interval) | Tiled better in | Max recall full / tiled |
| --- | --- | --- | --- | --- | --- |
| v0 | 0.591 | 0.666 | +0.076 (+0.069 to +0.083) | 100 % | 0.69 / 0.78 |
| v1 | 0.014 | 0.016 | +0.001 (-0.002 to +0.007) | 59 % | 0.09 / 0.10 |
| v4 | 0.547 | 0.572 | +0.024 (+0.013 to +0.034) | 100 % | 0.80 / 0.86 |
| v6 | 0.597 | 0.591 | -0.006 (-0.016 to +0.005) | 13 % | 0.82 / 0.86 |
| v7 | 0.498 | 0.541 | +0.043 (+0.033 to +0.053) | 100 % | 0.81 / 0.87 |
| v8 | 0.483 | 0.514 | +0.030 (+0.021 to +0.038) | 100 % | 0.75 / 0.81 |

Precision and recall at each model's scorecard cut-off are in `tiled.csv`.
Those cut-offs were chosen for untiled predictions on simulated validation
frames; at them, tiling raises more confident false alarms (v6 on HERIDAL:
precision 0.31 untiled, 0.10 tiled).

## Cut-offs chosen on real photos (`tiled_cutoffs.csv`, commit 08bf6e7)

Each model's best-F1 cut-off, chosen separately for untiled and tiled
detection on 294 SARD pool photos that no model trained on (the pool minus
the 300 drawn by `export.py --add-real 300`, seed 51, re-drawn with the same
call; the original export lists were deleted, so the exact match is assumed
from the call being deterministic). The test sets only apply the cut-off.

| Model | Method | Cut-off | Validation F1 | HERIDAL precision / recall | SARD test precision / recall |
| --- | --- | --- | --- | --- | --- |
| v0 | full | 0.029 | 0.47 | 0.30 / 0.23 | 0.65 / 0.59 |
| v0 | tiled | 0.036 | 0.54 | 0.20 / 0.43 | 0.64 / 0.67 |
| v4 | full | 0.295 | 0.82 | 0.40 / 0.33 | 0.67 / 0.62 |
| v4 | tiled | 0.471 | 0.81 | 0.20 / 0.57 | 0.68 / 0.63 |
| v6 | full | 0.368 | 0.83 | 0.49 / 0.29 | 0.68 / 0.62 |
| v6 | tiled | 0.473 | 0.82 | 0.30 / 0.55 | 0.64 / 0.62 |
| v7 | full | 0.179 | 0.81 | 0.32 / 0.35 | 0.52 / 0.66 |
| v7 | tiled | 0.369 | 0.78 | 0.20 / 0.59 | 0.54 / 0.66 |
| v8 | full | 0.052 | 0.74 | 0.19 / 0.45 | 0.49 / 0.62 |
| v8 | tiled | 0.132 | 0.74 | 0.14 / 0.57 | 0.54 / 0.63 |

On SARD, at these cut-offs, tiled and untiled detection perform about the
same. On HERIDAL, where people are tiny, tiling finds about twice as many
people (v6 recall 0.29 to 0.55, v4 0.33 to 0.57) at lower precision (v6 0.49
to 0.30). A cut-off chosen on SARD-like photos is not ideal for HERIDAL
either way: the photos differ (a linear classifier tells SARD from HERIDAL
98 % of the time).


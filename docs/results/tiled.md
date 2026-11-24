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

Precision and recall at each model's scorecard cut-off are in `tiled.csv`. The
cut-offs were chosen for untiled predictions; tiling produces more confident
false alarms at those cut-offs (for example v6 on HERIDAL: precision 0.31
untiled, 0.10 tiled). A cut-off for tiled use has not been chosen.

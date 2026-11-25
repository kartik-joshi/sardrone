# Measuring a sim-to-real gap for aerial person detection

Technical report for the `sardrone` project. Kartik Joshi, November 2026.

Every number here comes from a run recorded in this repository. Each table
names its source file; the source files name the git commit and, for
models, the hash of the training data.

## 1. Problem

A drone searching for a missing person needs a detector that works on
real aerial images, where a person is a few dozen pixels, often lying
down or partly under trees. Real labelled footage of that is scarce.
Simulation can produce unlimited footage with exact labels. The question
this project asks is the practical one: **if a detector is trained on
simulated flights, how much of its performance survives on real images,
and what moves that number?**

The answer is reported as a gap: the detector's score on unseen simulated
worlds minus its score on real photographs it never saw, with a
confidence interval, for each of several training recipes that differ in
one thing.

A second question follows from the first: once a detector exists, does a
drone that relies on it, and no longer on ground truth, still find
people?

## 2. Setup

### Simulated flights

- **Simulator:** PX4 SITL v1.18 with Gazebo Harmonic on an Apple Silicon
  Mac, headless. A quadcopter carries a camera (1280 x 960, 99.7 degree
  horizontal field of view) pitched 45 degrees down, to match the oblique
  views of the real dataset.
- **Worlds** are generated from a seeded row of a variation matrix: world
  type (meadow, forest edge, park), 1 to 5 people standing, sitting or
  lying, optional placement under tree canopies, time of day (sun
  elevation, colour and strength), and altitude (15, 25 or 40 m). The same
  row and seed always give byte-identical world files.
- **Mission:** a lawnmower search with lane spacing computed from the
  camera's ground footprint. On a sighting the drone descends, circles the
  spot with the camera on it, and reports a position.
- **Recording:** each flight is one MCAP file with camera frames at 10 Hz,
  pose, GPS, mission events, and labels. A batch runner records a whole
  matrix unattended and writes a manifest line per flight with its seed,
  configuration and git commit.

### Labels

Each camera frame carries exact person boxes from Gazebo's bounding-box
sensor, mounted beside the colour camera with the same lens. It boxes only
visible pixels, so occlusion is handled by the renderer: a person hidden
by a canopy has no label. An earlier approach, projecting 3D boxes with
the autopilot's estimated pose, was measured and rejected because the
boxes landed 20 to 40 pixels off.

### Data

| Set | Source | Frames | Person boxes | Use |
| --- | --- | --- | --- | --- |
| `train-v1` | 54 simulated flights | 24,667 | 34,909 | training (a 6,000-frame sample, see 3) |
| `val-v1` | 6 simulated flights | 2,861 | 4,735 | choosing the confidence cut-off |
| `sim-test` | 15 simulated flights | 7,453 | 10,541 | simulated test score |
| `sard-train-pool` | SARD, frames 1 to 621 | 594 | 1,561 | the "+ real" experiment only |
| `sard-test` | SARD, later frames | 1,387 | 4,971 | real test score; never trained on |

Frames are taken at 2 Hz. Near-duplicate frames within a flight are
removed from training and validation by cosine distance on CLIP
embeddings (threshold 0.004, chosen so that under 1 % of removed frames
change the number of people shown). Splits are by whole flight, and no
world seed is shared between splits. Source: `docs/results/v1.manifest.csv`,
`docs/DEVLOG.md`.

SARD (Sambolek and Ivasic-Kos, CC BY 4.0) is 1,981 hand-labelled frames
from real drone video; its six posture classes are mapped to `person`. It
is split by frame number rather than at random, because neighbouring video
frames are near-copies of each other.

Simulated people are a median 28 px on the longer side in 1280 px wide
frames; SARD people are a median 56 px in 1920 px wide frames, or about
37 px at the 1280 px training width.

## 3. Method

### Models

All models are Ultralytics YOLO11n fine-tuned from COCO weights for 10
epochs at 1280 px, batch 8, seed 51, on an Apple GPU (about 150 minutes
each). At 1280 px the hardware trains about 8 images per second, so each
model sees a seeded sample of 5,999 `train-v1` frames, taken within each
flight in proportion to its length; the sample is identical for every
version. Each version changes one thing:

| Version | Training data | Data hash |
| --- | --- | --- |
| v0 | none: COCO weights as downloaded | |
| v1 | the 5,999 simulated frames | `d741bdbb19fbe45d` |
| v2 | 6,000 frames from 30 separate flights over domain-randomized worlds: ground palette, sun direction and strength, clothing, body size, 15 to 45 unlabelled distractors | `61b514bcb1580625` |
| v3 | v1's frames, each degraded like a real camera: exposure, blur, sensor noise, JPEG compression | `86479461bb4f3bb7` |
| v4 | v3's frames plus 300 real images from `sard-train-pool` (5 %) | `7918bf71d200e0da` |
| v5 | v1's frames, with the backbone (modules 0 to 10) frozen at its COCO weights | `d741bdbb19fbe45d` |
| v6 | v4 with the simulated frames taken from the randomized flights instead | `1acc17da393128c2` |
| v7 | v6 recipe on randomized worlds with 1 to 3 box-built parked cars | `e411d1520339f4fb` |
| v8 | v4 on all 24,667 simulated training frames instead of a 6,000 sample | `c09b4aca41099b26` |
| v9 | v1's frames, everything but the detection layer (module 23) frozen | `d741bdbb19fbe45d` |
| v10 | v1's frames, AdamW at a learning rate of 0.0002 (v1: 0.002, chosen automatically) | `d741bdbb19fbe45d` |
| v11 | v8 with all 594 SARD pool images instead of 300 (2.4 % real) | `0bed5c53134d98f3` |
| v12 | v8 with its 300 real images repeated 4x (4.6 % real); resumed after epoch 6 | `a31d75ef2142dfc0` |

v0 was not in the original plan. It was added as a reference and turned
out to matter most. v5 and v6 were added after v1 to v4 to answer two
questions those results raised.

### Metrics

Predictions are matched to ground truth greedily by confidence at an IoU
threshold; average precision is the COCO 101-point interpolated area under
the precision-recall curve. mAP@0.5 is the headline; mAP@0.5:0.95 is also
recorded. The implementation is this project's own and was checked against
FiftyOne's on 4,105 frames (0.5411 against 0.541).

Precision and recall are reported at the confidence cut-off with the best
F1 on `val-v1`, fixed before either test set is touched. Confidence
intervals are 95 % percentile intervals from 1,000 bootstrap resamples of
whole images. The gap's interval comes from differencing the two sets'
bootstrap draws.

### An image-level check

Independently of any detector, a logistic regression on CLIP embeddings is
trained to tell `sim-test` frames from `sard-test` frames (5-fold
cross-validation). 50 % would mean the domains look alike; 100 % that they
are trivially separable.

## 4. Results

### The scorecard

Source: `docs/results/scorecard.csv`.

| Version | Sim-test mAP@0.5 | SARD-test mAP@0.5 | Gap | SARD precision / recall |
| --- | --- | --- | --- | --- |
| v0 pretrained | 0.453 (0.440 to 0.465) | 0.591 (0.574 to 0.607) | -0.138 (-0.159 to -0.117) | 0.83 / 0.51 |
| v1 simulation only | 0.947 (0.944 to 0.953) | 0.014 (0.009 to 0.019) | 0.933 (0.927 to 0.941) | 0.19 / 0.01 |
| v2 domain randomization | 0.935 (0.931 to 0.941) | 0.032 (0.029 to 0.038) | 0.902 (0.896 to 0.910) | 0.27 / 0.04 |
| v3 + camera realism | 0.941 (0.935 to 0.945) | 0.020 (0.017 to 0.023) | 0.921 (0.913 to 0.925) | 0.13 / 0.02 |
| v4 + 5 % real | 0.934 (0.931 to 0.941) | 0.547 (0.529 to 0.570) | 0.387 (0.364 to 0.408) | 0.70 / 0.58 |
| v5 v1, backbone frozen | 0.943 (0.938 to 0.946) | 0.127 (0.119 to 0.138) | 0.816 (0.804 to 0.825) | 0.35 / 0.16 |
| v6 v4 on randomized worlds | 0.926 (0.922 to 0.931) | 0.597 (0.580 to 0.617) | 0.329 (0.307 to 0.346) | 0.51 / 0.70 |
| v7 v6 + box-built cars | 0.933 (0.928 to 0.938) | 0.498 (0.479 to 0.519) | 0.435 (0.414 to 0.454) | 0.59 / 0.61 |
| v8 v4 on all sim frames | 0.946 (0.942 to 0.948) | 0.483 (0.465 to 0.507) | 0.462 (0.440 to 0.481) | 0.73 / 0.40 |
| v9 v1, detection layer only | 0.912 (0.906 to 0.919) | 0.435 (0.418 to 0.458) | 0.477 (0.453 to 0.497) | 0.59 / 0.53 |
| v10 v1, 10x lower learning rate | 0.945 (0.942 to 0.951) | 0.025 (0.022 to 0.029) | 0.920 (0.915 to 0.927) | 0.18 / 0.04 |
| v11 v8, all 594 pool images | 0.945 (0.939 to 0.947) | 0.506 (0.485 to 0.528) | 0.439 (0.416 to 0.459) | 0.67 / 0.55 |
| v12 v8, 300 real images x4 | 0.946 (0.943 to 0.952) | 0.544 (0.526 to 0.566) | 0.402 (0.380 to 0.421) | 0.67 / 0.64 |

mAP@0.5:0.95 on SARD: v0 0.301, v1 0.007, v2 0.013, v3 0.005, v4 0.235,
v5 0.041, v6 0.255.

**Finding 1. Simulation-only fine-tuning destroyed real-world
performance.** The pretrained network scored 0.591 on real photographs.
After ten epochs on simulated frames it scored 0.014, while reaching 0.947
on simulated worlds it had never seen. This is catastrophic forgetting:
every training example of `person` was a rendered figure on rendered
ground, and the network redefined the class around those.

**Finding 2a. Domain randomization helped measurably and uselessly.**
v2's real score is 0.032 against v1's 0.014, and the intervals do not
overlap, so the effect is real. Recall on real images is still 4 %. In
simulation it cost almost nothing: v2 never saw a plain world and scores
0.935 on the plain test worlds.

**Finding 2b. Making simulated images look like camera output did not
help.** v3's real score is 0.020 against v1's 0.014. The two intervals overlap
(0.009 to 0.019 and 0.017 to 0.023), so even that small difference is not
established, and it would be of no practical use if it were.

**Finding 3. A small amount of real data restored real performance at no
cost in simulation.** Three hundred real images took the real score from
0.020 to 0.547 while the simulated score moved from 0.941 to 0.934.

**Finding 4. No trained version beats the untrained network on real
images.** v4's 0.547 is below v0's 0.591. v6 reaches 0.597, level with v0
(the intervals overlap). On this evidence the simulated data has not been
shown to improve real-image detection. What v4 and v6 offer over v0 is
that they also work in the simulated worlds (0.934 and 0.926 against
0.453).

**Finding 5. Freezing the backbone prevents most of the damage to the
features and little of the damage to the result.** v5 keeps nine times
more real-image performance than v1 (0.127 against 0.014) at no cost in
simulation, and is still far below v0. With an identical backbone, the
difference from v0 lies in the neck and head, which learned to call only
rendered figures a person.

**Finding 6. Randomization does not measurably help, even with real data.**
The first v6 run scored 0.597 against v4's 0.547. Trained three times
each, v4 averages 0.555 (0.532 to 0.586) and v6 0.570 (0.526 to 0.597):
the difference is well inside either recipe's own spread. See
"Run-to-run variation" below.

**Finding 7. On a second real dataset the pattern holds.** HERIDAL is
different footage, never trained on (101 test images, 337 people; source
and licence in DETAILS.md). Scores from `docs/results/extra_real.csv`:

| Model | HERIDAL mAP@0.5 (95 % interval) | Max recall |
| --- | --- | --- |
| v0, untrained | 0.184 (0.118 to 0.258) | 0.38 |
| v1, simulation only | 0.030 (0.013 to 0.067) | 0.16 |
| v4, + 300 SARD images | 0.231 (0.154 to 0.324) | 0.57 |
| v6, v4 + randomized worlds | 0.262 (0.179 to 0.358) | 0.60 |
| v7, v6 + box-built cars | 0.213 (0.142 to 0.302) | 0.55 |
| v8, v4 on all sim frames | 0.261 (0.174 to 0.355) | 0.59 |
| v11, v8 + all 594 pool images | 0.150 (0.094 to 0.221) | 0.52 |
| v12, v8 + 300 images x4 | 0.215 (0.150 to 0.287) | 0.52 |

Simulation-only training collapses again (0.030). The models that saw 300
SARD images score above the untrained v0, though with only 101 images the
mAP intervals overlap; their max recall (0.55 to 0.60 against 0.38) is the
clearer gap. All scores are much lower than on SARD: the median person is
56 x 53 px in a 4000 px photo, about 18 x 17 px at the model's 1280 input.

**Finding 8. Four times the simulated data did not help on real photos.**
v8 is v4 on all 24,667 simulated training frames. It scored 0.946 on
simulation (v4 0.934) and 0.483 (0.465 to 0.507) on SARD, below all three
v4 runs (0.532 to 0.586), with lower max recall (0.75 against 0.80). On
HERIDAL it scored 0.261, inside v4's interval. One run, and confounded:
the same 300 real images were 4.8 % of v4's training set and 1.2 % of
v8's, so a smaller real share explains the result as well as more
simulation does.

*Follow-up (issue #2).* Two more runs keep all 24,667 simulated frames and
raise the real share: v11 (all 594 SARD pool images, 2.4 %) and v12 (the
same 300 images repeated 4x, 4.6 %). On SARD the score rises with the
share, 0.483 (v8) → 0.506 (v11) → 0.544 (v12), and v12 is back inside v4's
range. **The smaller real share explains v8's drop, not the extra
simulation.** Weight mattered more than variety here: 300 repeated images
beat 594 different ones at half the share, though their intervals only just
touch. HERIDAL does not follow the same order (v11 0.150, v12 0.215, v8
0.261, with wide, overlapping intervals). One training seed per run; v12
resumed after epoch 6 from its checkpoint.

**Finding 9. Full-resolution tiles help every useful model on HERIDAL.**
Source: `docs/results/tiled.md`. The method was fixed before scoring: 1280
px tiles overlapping by 256 px, plus the whole image as before, boxes on
inner cut lines dropped, NMS at IoU 0.5. Gains in mAP@0.5 with 95 % paired
bootstrap intervals (the same images resampled for both methods):

| Model | HERIDAL full / tiled | Gain | SARD full / tiled | Gain |
| --- | --- | --- | --- | --- |
| v0 untrained | 0.184 / 0.310 | +0.126 (+0.075 to +0.179) | 0.591 / 0.666 | +0.076 (+0.069 to +0.083) |
| v4 | 0.231 / 0.352 | +0.120 (+0.067 to +0.175) | 0.547 / 0.572 | +0.024 (+0.013 to +0.034) |
| v6 | 0.262 / 0.364 | +0.102 (+0.047 to +0.161) | 0.597 / 0.591 | -0.006 (-0.016 to +0.005) |
| v8 | 0.261 / 0.345 | +0.086 (+0.036 to +0.136) | 0.483 / 0.514 | +0.030 (+0.021 to +0.038) |

The gain is largest where people are smallest (HERIDAL, 4000 px photos) and
smaller on SARD (1920 px). It does not change the training story: on SARD
the untrained v0, tiled, is still the best model (0.666). Tiling also
raises more confident false alarms: at the scorecard cut-offs, chosen for
untiled use, v6's precision on HERIDAL falls from 0.31 to 0.10. With
cut-offs chosen on 294 real SARD photos that no model trained on, tiled and
untiled detection perform alike on SARD, while on HERIDAL tiling finds
about twice as many people (v6 recall 0.29 to 0.55) at lower precision
(0.49 to 0.30). Source: `docs/results/tiled_cutoffs.csv`.

**Finding 10. Where simulation-only training does its damage.** All on v1's
6,000 simulated frames, no real images, one run each:

| Version | Trains | Real SARD mAP@0.5 | HERIDAL mAP@0.5 | HERIDAL max recall |
| --- | --- | --- | --- | --- |
| v1 | every layer | 0.014 | 0.030 | 0.16 |
| v10 | every layer, 10x lower learning rate | 0.025 | 0.059 | 0.29 |
| v5 | neck and head (backbone frozen) | 0.127 | not scored | |
| v9 | the detection layer only | 0.435 (0.418 to 0.458) | 0.264 (0.184 to 0.360) | 0.67 |

Learning more slowly barely helps; freezing more does. Most of the loss is
in the neck, the layers between the backbone's features and the detection
layer. v9 is still below the untrained v0 on SARD (0.591), and pays for it
on simulation (0.912 against 0.947). On HERIDAL, with no real images at
all, it scores level with v6, which trained on 300 real SARD photos
(0.262), and has the highest max recall of any untiled model (0.67). One
run, overlapping intervals: a direction to test with seeds and with real
images added, not yet a result to rely on.

**Finding 11. A third real dataset, and thermal.** WiSARD's one-flight sample
(University of Washington; source and licence in the README): 264 colour
images at 3840 x 2160 with 1,022 people, and 264 thermal images at 640 x 512
with 1,006, in flight order. Test only. Sources: `extra_real.csv`,
`tiled.csv`, `tiled_gain.csv` (commit 820e17e; gains with a paired bootstrap).

| Model | Colour, untiled | Colour, tiled | Tiling gain | Thermal, untiled |
| --- | --- | --- | --- | --- |
| v0 untrained | 0.125 | 0.314 | +0.190 (+0.153 to +0.225) | 0.150 |
| v4 (300 SARD) | 0.100 | 0.138 | +0.037 (+0.015 to +0.058) | 0.136 |
| v6 (300 SARD) | 0.097 | 0.148 | +0.054 (+0.030 to +0.075) | 0.044 |
| v9 (sim only, frozen) | 0.076 | 0.232 | +0.156 (+0.126 to +0.189) | 0.045 |

Across three real datasets the 300 SARD images help on SARD's own test
photos, roughly break even on HERIDAL, and hurt here: what they teach is
SARD's footage. Tiling matters most for the models that kept the pretrained
network's general knowledge (v0, v9). Colour-only models find some people in
thermal images; the untrained one does about as well as on colour. One
flight, one site: a single sample of real footage, not a benchmark.

### Precision-recall curves

Source: `docs/images/pr_curves.png`, from the stored predictions. On real
photos the untrained v0 is the most precise model up to about 0.55 recall
and cannot go beyond 0.69 recall at any cut-off; v4 and v6 reach 0.80 and
0.82. Their most confident boxes are less reliable than v0's (precision
about 0.77 for v4 and 0.87 for v6 at the start of the curve). For search
and rescue, where a missed person costs more than a false alarm, the
high-recall end is the part that matters, and there the trained models
lead. mAP, which averages the whole curve, does not show this.

### What the detector mistakes for people

The 24 most confident false alarms on the real test images were reviewed
by eye. For v4 all 24 are one red car on grass, seen in different frames.
For v6, trained on worlds with 21 clothing colours and coloured
distractor boxes, the red car disappears from the top, and silver and
navy cars take its place. The simulated worlds contain no vehicles, so
nothing taught the detector that a compact coloured object on grass can
be something other than a person. v7 tested exactly that: randomized
worlds with one to three box-built cars as background. Its top mistakes
were still vehicles (20 of 24), at lower confidence (0.77 to 0.82, against
0.89 to 0.91 for v4), and its real-image score, 0.498 from one run, was
below all six v4 and v6 runs. A box with a dark cabin is not enough of a
car to teach the difference.

### Run-to-run variation

Source: `docs/results/seed_runs.csv`. The same recipe trained again with
only the training seed changed:

| Recipe | Real-image mAP@0.5, by seed | Spread |
| --- | --- | --- |
| v1 | 0.014, 0.011 | 0.003 |
| v4 | 0.547, 0.586, 0.532 | 0.054 |
| v6 | 0.597, 0.526, 0.587 | 0.072 |

The bootstrap intervals in the scorecard cover the choice of test images,
about plus or minus 0.02. Training variation is larger. Findings 1, 3, 4
and 5 hold across seeds (v1's collapse is reproduced; five of six v4 and
v6 runs are below v0). The first version of finding 6, that randomization
helps once real data is present, rested on one lucky v6 run and was
withdrawn.

### The image-level gap

Source: `docs/results/domain_gap.json`.

| Comparison | Classifier accuracy |
| --- | --- |
| simulated vs real | 100 % |
| simulated after camera degradation vs real | 100 % |
| domain-randomized simulated vs real | 100 % |
| domain-randomized vs plain simulated | 93.2 % |
| random halves of simulated | 50.4 % |
| random halves of real | 49.7 % |

The two domains are trivially separable, and neither the camera-realism
treatment nor domain randomization changes that. Randomized frames are
also no nearer to real ones: mean cosine distance to the nearest real
frame is 0.303 for randomized and 0.291 for plain simulated frames,
against 0.023 between real frames. Each check was run before the
corresponding model finished training and predicted its result.

### Detection in the loop

Source: `docs/results/closed_loop_v4.md`. Fifteen `sim-test` worlds were
flown with v4 as the only source of sightings. A tracker (constant-velocity
Kalman filter, Hungarian matching on centre distance) keeps identities
through missed frames, and the mission acts only on tracks seen three
times. A report counts as a find if a true person is within 3 m.

| | Closed loop, v4 | Ground-truth oracle |
| --- | --- | --- |
| People found | 39 of 45 | 45 of 45 |
| Flights with everyone found | 10 of 15 | 15 of 15 |
| False reports | 0 | 0 |
| False alarms rejected by the confirmation orbit | 9 | 0 |
| Missed rescues (in view for 10+ frames, never reported) | 6 | 0 |
| Mean position error of reports | 0.45 m | |
| Mean flight time | 344 s | 260 s |

| Condition | Found |
| --- | --- |
| In the open / under trees | 33 of 34 / 6 of 11 |
| Altitude 15 m / 25 m / 40 m | 14 of 14 / 14 of 17 / 11 of 14 |
| Meadow / park / forest edge | 15 of 15 / 15 of 15 / 9 of 15 |

### A second pass over tree cover

Source: `docs/results/closed_loop_v4_second_pass.md`. The five forest-edge
worlds flown again with the same detector, adding a second lawnmower at
15 m over the tree-covered rectangle, at right angles to the first.

| | One pass | With second pass | Strict second pass |
| --- | --- | --- | --- |
| People found | 9 of 15 | 12 of 15 | 10 of 15 |
| False reports | 0 | 2 (one phantom, reported twice) | 0 |
| False alarms rejected by the orbit | 3 | 19 | 8 |
| Total flight time | 1,440 s | 3,521 s | 2,548 s |

The low pass finds people the high pass could not, and sees trees large
enough for the detector to mistake far more often. One flight per world.
The double report exposed a bug, since fixed: a confirmation that ends on
an already-reported spot is no longer reported again.

### The mission agent

Source: `docs/results/agent_eval.md` and `agent_eval.json`, commit facb659.
Twenty plain-English missions, each scored by checks fixed before any model
ran (right areas in order, right altitude, exactly the right people
reported, battery limit obeyed, every search finished, home at the end).

| Model | Missions passed | Valid tool calls | Mean steps |
| --- | --- | --- | --- |
| Rule-based keywords | 9 of 20 | 100 % | 10.5 |
| Cohere `command-a-03-2025`, temperature 0 | 17 of 20 | 100 % | 10.5 |

Of the eleven missions the keyword rules fail, the model passed nine,
among them altitude in words, two areas in order, "do not report adults",
"except people in black", and "do not launch the drone" (one status call,
no take-off). Of the nine the rules pass, it passed eight. Its three failures (m02, m11, m17) are one behaviour: each asks
for specific people, and having confirmed and reported exactly them, the
model went home with the search unfinished. Whether that is a fault
depends on the operator; the scorer was written to require full coverage,
and that rule was not changed after seeing the results.

212 model calls, 587,320 input and 12,351 output tokens, 38 s per mission
(mostly the 3.5 s spacing the free trial key's rate limit requires). One
call was rate-limited and retried with the identical request. No cost is
computed: the trial key is free and no price was assumed.

### The auto-labeller

Source: `docs/results/auto_labeler.md`. OWLv2 (base, patch 16, ensemble)
prompted for "a person" on 4,105 frames of an earlier 10-flight pilot set
with 6,078 true boxes:

| Confidence at least | Precision | Recall |
| --- | --- | --- |
| 0.05 | 0.695 | 0.829 |
| 0.10 | 0.902 | 0.770 |
| 0.20 | 0.991 | 0.531 |

Recall was 0.927 for people in the open and 0.370 under trees; 0.493 for
boxes under 16 px. Of the least confident 5 % of its boxes, 10 % were
correct.

## 5. Failure analysis

**Why simulation-only training failed.** Three body meshes, two tree
models and flat procedurally textured ground are a small world. The real
images show people in tall dry grass, on quarry stone and on dirt roads.
Fine-tuning all layers on the small world for ten epochs was enough to
overwrite the pretrained notion of a person. Nothing in the recipe
protected it: no real examples, no frozen layers, no reduced learning
rate.

**Why neither randomization nor camera realism helped.** The embedding
check shows the domains differ in content, not in colour, lighting or
sensor character. Randomization varies parameters of the same content;
noise and compression change pixels. Neither adds long grass that partly
covers a person, or a fourth body shape.

**Where the detector fails in flight.** All six closed-loop misses were in
forest-edge worlds and five were people placed under canopies. Every one
was a missed rescue: the label sensor saw the person in at least ten
frames, so coverage was adequate and detection was not. The auto-labeller
failed in the same place. Under a canopy a person is a few visible pixels,
and the labels, correctly, are small or absent there.

**What the confirmation orbit bought.** The detector held nine false
tracks long enough to trigger a detour. The orbit rejected all nine and no
false report was made. The price was flight time: a third longer than
with ground truth.

**Where the tracker is weak.** Matching by box overlap broke three people
into 70 tracks in the first closed-loop flight, because a 30 px box often
does not overlap its own position one frame earlier. Matching by centre
distance reduced that to 47. It still fragments: people leave and
re-enter the frame, and the drone's own motion is not compensated. The
mission is unaffected because identity is decided from ground position.

## 6. Limits

- **Simulation only.** No real aircraft flew. The closed-loop results are
  for worlds from the same generator as the training worlds.
- **The real score of v4 is flattered.** Its real training images and the
  real test images are from the same footage: the same shoots, places and
  people. It measures "more of the same", not a new location. No second
  real dataset was used.
- **Few runs per version.** v1 was trained twice and v4 three times; the
  others once. v4's three runs spread over 0.054 on real images, so
  single-run differences smaller than that are not meaningful.
- **Small training budget.** A quarter of the simulated training frames,
  for ten epochs. A longer run would raise the simulated score; nothing
  here suggests it would raise the real one.
- **The simulated test interval is too narrow.** `sim-test` keeps
  near-duplicate frames from 15 flights, so its images are far from
  independent.
- **Cut-offs were chosen on simulated validation data,** which is the
  wrong domain for tuning a real-image operating point. It affects the
  precision and recall columns, not mAP.
- **v2 is not a clean ablation of v1.** Its 6,000 frames come from 30
  flights, v1's from 54. (v6 later combined randomization with real data;
  see Finding 6.)
- **Planned and not done at the time of writing:** weather (this Gazebo
  setup does not render fog; see the README limitations).
- **The mission agent was evaluated once per mission, with one model, on
  20 missions written for this project, against a simulator of the
  mission, not the drone.
- **Small counts in the closed-loop breakdowns:** 9 to 34 people per cell.

## 7. What I would do next

1. ~~Separate the two causes in v8.~~ Done (Finding 8, follow-up): the real
   share explains it. More seeds of v11 and v12 would firm up the ranking
   between them.
2. Choose cut-offs on validation photos that resemble the target footage:
   a cut-off chosen on SARD-like photos trades recall for precision on
   HERIDAL (Finding 9).
3. Combine Finding 10 with real images: v9's freezing plus the 300 real
   images of v4, over several seeds.
4. Widen the simulated content (more bodies, vegetation, terrain) before
   spending more on rendering fidelity, since content is where the domains
   differ.
5. Find a confirmation rule for the low pass that keeps more of its
   extra finds: the strict version (6 sightings, 20 frames) removed the
   false reports but also two of the three extra people.

# sardrone

A search-and-rescue drone in simulation, built to practise the robotics data
loop end to end: **see → curate → embed → label → train → evaluate**, and
then measure how much of what was learned in simulation survives on real
images.

The drone flies PX4 in Gazebo, records every flight as one `.mcap` episode,
and the episodes go through FiftyOne to become a labelled dataset, a trained
detector, a scorecard, and finally a detector flying in the loop.

![A closed-loop flight: the detector finds a person, the drone circles to confirm](docs/images/demo.gif)

*A closed-loop flight. Yellow boxes are the tracker's, from the trained
detector looking at the camera image; no ground truth reaches the mission.
The drone has seen a person, descends and circles to confirm, then reports
the position.*

## What it found

**1. Training on simulation alone made the detector worse on real images
than not training it at all.**

| Version | Change | Simulated test mAP@0.5 | Real test mAP@0.5 | Gap |
| --- | --- | --- | --- | --- |
| v0 | pretrained YOLO11n, never trained on our data | 0.453 | 0.591 | -0.138 |
| v1 | fine-tuned on simulation only | 0.947 | 0.014 | 0.933 |
| v2 | v1's recipe on domain-randomized worlds | 0.935 | 0.032 | 0.902 |
| v3 | v1 + camera realism (noise, blur, JPEG, exposure) | 0.941 | 0.020 | 0.921 |
| v4 | v3 + 300 real images (5 %) | 0.934 | 0.547 | 0.387 |
| v5 | v1 with the backbone frozen | 0.943 | 0.127 | 0.816 |
| v6 | v4 with domain-randomized simulated frames | 0.926 | 0.597 | 0.329 |
| v7 | v6 with box-built cars in the worlds | 0.933 | 0.498 | 0.435 |
| v8 | v4 on all 24,667 simulated frames (4x) | 0.946 | 0.483 | 0.462 |

The pretrained network already found half the real people. Ten epochs on
simulated frames taught it to find 93 % of simulated people and left it
nearly blind to real ones. Domain randomization (ground, light, clothing,
body size, clutter) and camera realism each left the real score near zero.
Three hundred real images restored real performance without costing
anything on simulation. Freezing the backbone kept nine times more
real-world ability than training every layer, and was still far short.
Randomization did not measurably help, alone or with real images: three
training runs each of v4 and v6 average 0.555 and 0.570 on real photos,
and each recipe's runs spread over 0.05 to 0.07. Full table with 95 % bootstrap intervals:
[docs/results/scorecard.md](docs/results/scorecard.md).

**2. No trained version beats the untrained network on real photos.** Across
three training runs each, v4 averages 0.555 and v6 0.570, against the
untrained network's 0.591; five of the six runs are below it.
On this evidence the simulated data has not been shown to help real-world
detection; what it buys is performance in the simulated worlds the drone
actually flies in (0.926 against 0.453) without losing the real-world
ability.

The precision-recall curves add a nuance the single numbers hide: on real
photos the untrained network is more precise at high confidence, but it
never finds more than 69 % of the people; v4 and v6 reach 80 % and 82 %.

![Precision-recall curves](docs/images/pr_curves.png)

Reviewing the real images that fool it: all 24 of v4's most confident
mistakes are one red car on grass, because the simulated worlds had no
vehicles. Adding box-built cars to the worlds (v7) made the model less sure
about real cars but did not stop the mistake, and scored lower overall.

**3. With the detector in the loop, the drone found 39 of 45 people** across
15 unseen simulated worlds, with no false reports. All six misses were in
forest worlds, five of them people under trees.
[docs/results/closed_loop_v4.md](docs/results/closed_loop_v4.md)
A second, lower pass over the tree cover raised that to 12 of 15 in the
five forest worlds (from 9), at the cost of one phantom reported twice,
six times as many false alarms and 2.4 times the flight time
([table](docs/results/closed_loop_v4_second_pass.md)). A stricter version
(6 sightings before a detour, 20 frames to confirm) found 10 of 15 with no
false report and 8 false alarms
([table](docs/results/closed_loop_v4_strict_second_pass.md)).

**4. An off-the-shelf auto-labeller is a first draft, not a labeller.**
Scored against exact simulator labels, OWLv2 reached 90 % precision at 77 %
recall, and found 93 % of people in the open but 37 % of those under trees.
[docs/results/auto_labeler.md](docs/results/auto_labeler.md)

**5. A language model turns plain-English missions into correct tool calls
far more often than keyword rules.** On 20 written missions against a
mission simulator, Cohere's `command-a-03-2025` passed 17 against 9 for a
rule-based baseline, with every tool call valid. It handled the missions
keyword rules cannot ("fly low, at 15 metres", "do not report adults",
"except people in black", "do not launch the drone"). All three failures
are the same habit: having reported exactly the right people, it flew home
before finishing the search, which the scorer counts as a failure.
[docs/results/agent_eval.md](docs/results/agent_eval.md)

**6. Looking at photos at full resolution finds far more people.** The
detector shrinks every photo to 1280 px; in a 4000 px HERIDAL photo a person
becomes about 18 px. Running it on overlapping 1280 px tiles as well
(method fixed before scoring) raised every model on HERIDAL, each gain with
a paired interval above zero: +0.09 to +0.14 mAP@0.5 for all but the
simulation-only v1 (+0.05). v6 went from 0.262 to
0.364, the untrained v0 from 0.184 to 0.310. On SARD's smaller photos the
gains are smaller (v0 0.591 to 0.666, v4 +0.024) and v6 does not change.
The catch: at each model's existing cut-off, tiling also raises more false
alarms; a cut-off for tiled use has not been chosen.
[docs/results/tiled.md](docs/results/tiled.md)

## How it fits together

```
 ┌─ SEE ───────────────────────────────────────────────────────────────────┐
 │ variation matrix ─> world generator ─> Gazebo + PX4 ─> search mission    │
 │  (YAML, seeded)     (trees, people,     (lawnmower, confirm, report)     │
 │                      sun, clutter)            │                          │
 │                                               ▼                          │
 │                              one .mcap per flight: camera, pose, GPS,    │
 │                              events, exact label boxes, ground truth     │
 └───────────────────────────────────────────────┬──────────────────────────┘
                                                 ▼
 ┌─ CURATE / EMBED / LABEL (FiftyOne) ─────────────────────────────────────┐
 │ frames at 2 Hz ─> CLIP embeddings ─> near-duplicates, train/val/test     │
 │                   by episode, hard-case views; OWLv2 auto-labels scored  │
 │                   against ground truth                                   │
 └───────────────────────────────────────────────┬──────────────────────────┘
                                                 ▼
 ┌─ TRAIN / EVALUATE ──────────────────────────────────────────────────────┐
 │ YOLO export + data hash ─> YOLO11n ─> sim-test and real SARD ─> scorecard│
 └───────────────────────────────────────────────┬──────────────────────────┘
                                                 ▼
            detector + tracker back in the mission loop ─> mission metrics
```

The same loop at three sizes. "Tonight's loop" is what one evening buys:
the whole loop, each stage done once and by hand.

| Stage | Tonight's loop (one evening) | This project | Production |
| --- | --- | --- | --- |
| See | One recorded flight, or a folder of drone images | 135 simulated flights as MCAP episodes, each with a manifest line (seed, config, commit) | A fleet's logs in object storage, indexed and searchable |
| Curate | Scroll through the images in FiftyOne | Embeddings, near-duplicate tagging, hard-case views, splits by whole flight | Mining across millions of scenes; scenario taxonomies |
| Label | A pretrained or open-vocabulary detector's boxes, taken as they come | Exact labels from the simulator; an auto-labeller measured against them | Auto-labelling with human review queues and label QA |
| Train | One fine-tuning run in a notebook | One command per version, with a data hash, a run record and repeat seeds | Scheduled distributed training with experiment tracking |
| Evaluate | One mAP number on whatever was held out | A scorecard: simulated and real test sets, bootstrap intervals, run-to-run spread, the gap | Many frozen test sets, regression gates, slice metrics |
| Close the loop | Look at a few predictions | The detector flies 15 unseen worlds; mission metrics; false-alarm review feeds the next world design | Shadow mode and on-road validation feeding the next data collection |

## Run it

No simulator needed for the first three:

```bash
make test            # create .venv, install, run the test suite
```

```bash
make search-fake     # the whole search mission with a fake drone, about 2 s
```

```bash
make agent-eval      # the mission agent's 20 missions with the rule-based baseline
```

With PX4 and Gazebo installed (see [docs/DETAILS.md](docs/DETAILS.md#px4-setup)):

```bash
make assets          # people and tree models from Gazebo Fuel (45 MB, once)
```

```bash
make batch MATRIX=configs/v1.yaml     # record 75 episodes unattended (about 6 hours)
```

```bash
make frames && make curate            # labelled frames into FiftyOne, embedded and split
```

```bash
make export EXPORT=v1 && make train EXPORT=v1 VERSION=v1 && make evaluate VERSION=v1
```

```bash
make view            # open the episodes in FiftyOne's multimodal viewer
```

Every phase, with its commands and pictures, is in
[docs/DETAILS.md](docs/DETAILS.md). What broke along the way, and what each
failure taught, is in [docs/DEVLOG.md](docs/DEVLOG.md).

## The pieces

| Piece | What it does | Where |
| --- | --- | --- |
| Episodes | One `.mcap` per flight, protobuf Foxglove schemas, one clock, a JSON sidecar | `sardrone/mcap/` |
| Worlds | Seeded Gazebo worlds from a YAML row: terrain, trees, people, sun, optional randomization | `sardrone/worlds/` |
| Flying | PX4 through MAVSDK behind a small `Drone` interface, with a fake drone for tests | `sardrone/sim/` |
| Search | Camera footprint, lawnmower coverage, a state machine (search, confirm, report, return) | `sardrone/mission/` |
| Perception in flight | Trained detector, Kalman tracker with memory, mission metrics | `sardrone/mission/` |
| Batches | Whole matrices unattended, with a manifest and resume | `sardrone/batch.py` |
| Dataset | Frames, exact labels, embeddings, de-duplication, splits, auto-labels | `sardrone/fo/` |
| Training | YOLO export with a data hash, training, prediction, metrics, scorecard | `sardrone/train/` |
| Workflow | The loop as one command with a data gate for retraining | `sardrone/pipeline.py`, [docs/WORKFLOW.md](docs/WORKFLOW.md) |
| Agent | Plain-English missions to tool calls, with a 20-mission evaluation | `sardrone/agent/` |

## What is verified, and what is not

| Part | Status |
| --- | --- |
| Time and frame conversions, `EpisodeWriter`, synthetic episode | Tested (`make test`), files read back with an independent MCAP reader |
| Mission logic in `record_flight.py` | Tested end to end with `FakeDrone` on simulated time |
| FiftyOne dataset + App | Tested; viewer checked by hand: 3D, camera, map and log tiles replay with no warnings |
| `MavsdkDrone` against a running PX4 | Flown on PX4 SITL v1.18 + Gazebo Harmonic (`gz_x500`, headless), from a cold start. Not part of `make test`, which must run without a simulator |
| Gazebo camera | Byte protocol and `GazeboCamera` tested without a simulator (stand-in bridge). Flown once on `gz_x500_mono_cam`, headless: 591 frames at 1280x960, checked in FiftyOne |
| World generator | Tested without Gazebo (54 tests). Three pilot worlds rendered and flown on PX4; pictures above are from those runs |
| Ground-truth boxes | In the file and the sidecar. Projected into frames with PX4's estimated pose they land 20 to 40 px off the people; exact image labels are Phase 2D |
| Coverage planner | Tested: every sample point of rectangles, triangles, tilted and L-shaped areas is covered, at four lane headings |
| Search state machine | Tested with `FakeDrone`: find-all, nobody seen, false alarm, low battery, illegal transition. One real PX4 flight found 3 of 3 with the oracle. Low battery was not tried on PX4 |
| Batch runner | Bookkeeping tested with a stand-in runner (resume, retry, timeout, disk limit). One real batch of 10 completed with no failures, so retry and timeout have not been exercised against the simulator |
| Image labels | From Gazebo's bounding-box sensor. Reader, dataset and overlay code tested; 20 random overlays from the real batch checked by eye; box counts cross-checked against geometry on all 4,678 frames |
| Curation | De-duplication, split and saved views tested on a hand-made dataset; threshold chosen from a sweep on the pilot data |
| Auto-labeller | Scoring code tested against numbers worked out by hand; one full run on 4,105 frames |
| Metrics | AP, precision, recall and bootstrap code tested against hand-worked numbers, and it reproduces FiftyOne's mAP on 4,105 frames |
| Detector v1 | Trained once (10 epochs, 6,000-frame sample). Scores come from `make evaluate`; the real-image pipeline was cross-checked with the untrained network |
| Detection in flight | 15 closed-loop PX4 flights with detector v4; tracker and identity logic tested with the fake drone, including a persistent false positive |
| Gazebo window (GUI) | Not tried: it needs XQuartz, which needs an admin password to install |

Numbers from the recorded real flight (`flight-0001`, 61.6 s): cruise height
9.6 to 10.1 m, top ground speed 3.1 m/s, square corners within the 1 m
tolerance, touchdown 0.5 m from the takeoff point, samples every 0.10 s on
PX4's own clock, battery reported on a 0 to 100 scale.

## Limitations

- **Simulation only.** No real drone has flown this. The closed-loop
  numbers are for simulated worlds built by the same generator as the
  training worlds.
- **Narrow simulated content.** Three body meshes, two tree models, flat
  ground. Randomizing colours, light and clutter did not change that. The real test images (tall dry grass, quarry stone, roads) look
  nothing like it, and a linear classifier on image embeddings separates
  the two domains 100 % of the time.
- **v4's real SARD score is flattered.** Its 300 real training images and
  the real test images come from the same SARD footage. On HERIDAL, other
  footage never trained on, every model scores far lower (v4 0.231, v6
  0.262, untrained v0 0.184, simulation-only v1 0.030; 101 images, so the
  intervals overlap). Simulation-only training collapses there too, and the
  mixed models again find more of the people than v0 (max recall 0.55 to
  0.60 against 0.38).
  ([docs/results/extra_real.csv](docs/results/extra_real.csv))
- **Small-scale training.** Each model saw a seeded 6,000-frame sample for
  10 epochs (v8 saw all 24,667 frames and scored lower on real photos,
  but its real share was a quarter of v4's, so the two causes are mixed). v4 and v6 were each trained three times; their real-image
  scores spread over 0.053 and 0.072, so differences between versions
  smaller than that are not meaningful.
  ([docs/results/seed_runs.md](docs/results/seed_runs.md)) The intervals cover the choice of test
  images, not the randomness of training.
- **No fog or rain.** This Gazebo setup does not render fog, so `weather`
  only accepts `clear`.
- **No obstacle avoidance.** Missions rely on flying above the tallest tree.
- **The tracker fragments.** People leave and re-enter the picture and the
  drone's own motion is not compensated; identity is decided by ground
  position instead.
- **The mission agent flies a mission simulator, not the drone,** and was
  run once per mission with one model. Its 20 missions were written for
  this project, so 17 of 20 is a check, not a benchmark.
- **The retraining gate's threshold (200 hard frames) is not tuned.**

## What's next

v1.0 is complete; work continues. Each item is an issue in the
[v2 milestone](https://github.com/kartik-joshi/sardrone/milestone/1):

1. ~~**Tiled detection** on full-resolution images.~~ Done: up to +0.14
   mAP@0.5 on HERIDAL (finding 6). Next: a cut-off chosen for tiled use.
   ([#1](https://github.com/kartik-joshi/sardrone/issues/1))
2. **Untangle v8:** more simulation, or a smaller real share? ([#2](https://github.com/kartik-joshi/sardrone/issues/2))
3. **Protect more of the network** in simulation-only training. ([#3](https://github.com/kartik-joshi/sardrone/issues/3))
4. **Richer simulated content:** real vehicle models, more body types,
   vegetation and uneven terrain. ([#4](https://github.com/kartik-joshi/sardrone/issues/4))
5. **A better confirmation rule** for the low second pass. ([#5](https://github.com/kartik-joshi/sardrone/issues/5))
6. **The agent:** repeat runs and a newer model ([#6](https://github.com/kartik-joshi/sardrone/issues/6)), a larger
   mission set ([#7](https://github.com/kartik-joshi/sardrone/issues/7)), and flying the simulated drone from a
   sentence ([#8](https://github.com/kartik-joshi/sardrone/issues/8)).
7. **Real drone video,** not only still photos. ([#9](https://github.com/kartik-joshi/sardrone/issues/9))
8. **The whole pipeline** end to end with one command. ([#10](https://github.com/kartik-joshi/sardrone/issues/10))
9. **A FiftyOne plugin** that measures the sim-to-real gap on any pair of
   datasets ([#11](https://github.com/kartik-joshi/sardrone/issues/11)), and **the simulated dataset on Hugging Face**
   ([#12](https://github.com/kartik-joshi/sardrone/issues/12)).
10. **Photo triage for search-and-rescue teams:** a prototype that ranks
    drone photos by how likely they hold a person, built with the teams
    who would use it. ([#13](https://github.com/kartik-joshi/sardrone/issues/13))

Details in [docs/report.md](docs/report.md), section 7.

## Licences and credits

- **This repository's code:** MIT ([LICENSE](LICENSE)). Use, copy and change
  it freely.
- **What MIT does not cover.** The training and detection modules
  (`sardrone/train/`, `sardrone/mission/detector.py`) import Ultralytics,
  which is AGPL-3.0. Distributing a product that combines them with
  Ultralytics is subject to AGPL-3.0 (or an Ultralytics licence). The data
  and models listed below keep their own licences, and none of them is
  included in this repository.
- **Ultralytics YOLO11:** AGPL-3.0. Fine for a portfolio project; it would
  matter for a closed-source product.
- **SARD** (the real test set): Sasa Sambolek and Marina Ivasic-Kos,
  "Search and Rescue Image Dataset for Person Detection", CC BY 4.0,
  doi:10.21227/ahxm-k331. Not redistributed here; one annotated picture in
  [docs/images](docs/images/v1_on_sard.jpg) shows frames from it.
- **3D models** from Gazebo Fuel (CC0 and CC BY 4.0): see
  [docs/ASSETS.md](docs/ASSETS.md). Not redistributed here.
- **PX4-Autopilot:** BSD 3-Clause; its x500 model is referenced, not copied.
- **OWLv2:** Apache-2.0. **CLIP weights** via the FiftyOne model zoo.
- **FiftyOne:** Apache-2.0. **Foxglove SDK / MCAP:** MIT.

## More

- [docs/DETAILS.md](docs/DETAILS.md): every phase, with commands and pictures
- [docs/DEVLOG.md](docs/DEVLOG.md): what broke and what was learned
- [docs/results/](docs/results/): scorecard, closed-loop table, auto-labeller report, manifests
- [docs/WORKFLOW.md](docs/WORKFLOW.md): the loop as a pipeline, and why it is not an Airflow DAG

# Development log

One entry per phase: what was done, what broke, what was learned.

## Phase 2A - A world worth searching (2026-11-14)

### Done

- `sardrone/worlds/`: a variation row (YAML) becomes a `WorldSpec` (every
  tree and person with an exact pose), which is written as an SDF world plus
  `spec.json`. Same row and seed give byte-identical files.
- Three world types as placement rules: meadow (a few lone trees),
  forest edge (a dense band along the north side), park (planted clumps and
  footpaths).
- People in three poses with nine clothing colours; "partial" occlusion puts
  them under a tree canopy.
- Sun elevation, azimuth, colour and strength follow the time of day.
- `x500_sar_cam`: PX4's x500 with the camera pitched 45 degrees down. PX4
  attaches to it; no PX4 file is edited.
- `/ground_truth/people` (foxglove.SceneUpdate, 1 Hz) and a `people` list in
  the sidecar, in the episode's `map` frame.
- `make preview` renders a world from fixed cameras in about 15 s without
  flying.

### What broke, and what was learned

- **Fuel models cannot be used as downloaded.** Their `model.sdf` files name
  other models, use Ogre material scripts (unsupported in Harmonic, so trees
  rendered white), and one pulls a further model from the internet. Fix:
  reference mesh and texture files directly and set the texture with
  `<pbr><metal><albedo_map>`.
- **"Rescue Randy" is not lying down.** The mesh stands with arms out. The
  lying pose is that mesh rolled 90 degrees and lifted by half its depth.
- **Mesh units differ.** One file is in centimetres, one in tenths of a
  millimetre, the trees are in inches with a 0.0254 node scale. Sizes are
  measured from the files by `mesh_bounds()` and a test compares them with
  the table in `assets.py`.
- **Fog is not rendered.** A world with `<scene><fog>` gave pixel-identical
  images to one without (mean difference 0.00005 of 255). So `weather` only
  accepts `clear`, and the matrix rejects anything else.
- **Golden hour rendered almost black.** A sun 12 degrees up lights flat
  ground at only sin(12 deg) = 21 % strength. The low-sun presets now use a
  stronger light and more ambient.
- **The camera mount recorded in Week 1 was wrong by 0.24 m.** PX4's
  `x500_mono_cam` gives the camera pose as `.12 .03 .242`, which I took to be
  from the drone body. It is from the MODEL origin, and the body
  (`base_link`) sits 0.24 m above that. The camera is 2 mm above the body,
  not 24 cm. Found when the first pitched camera ended up at ground level.
  Fixed in `GazeboCamera.mount_m`; episode `flight-0002` carries the old
  value.
- **Drone body height.** `base_link` rests 0.227 m above the ground (0.24
  minus a measured 0.013 m settling). `map` z = world z - 0.227.
- **Ground-truth boxes projected with PX4's pose are 20 to 40 pixels off.**
  Projecting the boxes into frames using the recorded `/tf` puts them next
  to the people, not on them, in directions that change from frame to frame.
  PX4's pose is an ESTIMATE (GPS and compass noise); about 1 m or 2 degrees
  of error is enough. Phase 2D must take the camera's TRUE pose from Gazebo,
  or use Gazebo's bounding-box sensor.
  Example crops with the projected box drawn in magenta:
  `docs/images/phase2a/*_zoom.jpg`.
- **A PX4 console without a terminal floods the log.** With stdin closed it
  prints its prompt forever (430 MB in a minute). The launcher gives it a
  pipe that never closes.

### Numbers

- Preview render: 14 s for four cameras.
- Flights: `pilot-meadow` 82 s, `pilot-forest` 95 s, `pilot-park` 105 s;
  the episode files are 60 to 70 MB each (real images compress far less
  than the grey plane did), which matters for the 75-episode dataset.
- `pilot-meadow` (15 m, 30 m square), 808 frames. People
  fully in view: none in 313 frames, one in 312, two in 163, three in 20
  (by projection with PX4's pose, so approximate).
- A standing person at 15 m altitude is roughly 35 to 50 pixels tall.

### Not done

- Terrain is flat. Hills are left for the randomization phase.
- Three body meshes only. That lack of variety is part of the sim-to-real
  gap this project sets out to measure.
- `success` is `false` for these flights: nothing detects people yet
  (`detector: none`).

## Phase 2B - Search behaviour (2026-11-15)

### Done

- `sardrone/mission/footprint.py`: what the tilted camera usefully sees
  (centre distance, slant range, swath width, person size in pixels).
- `sardrone/mission/coverage.py`: lawnmower lanes over any polygon, with
  overlap and a lead-in because the camera looks ahead.
- `sardrone/mission/projection.py`: map point to pixel, and pixel to a
  point on flat ground.
- `sardrone/mission/oracle.py`: a stand-in detector that reads ground truth.
- `sardrone/mission/search.py`: the state machine
  `TAKEOFF -> SEARCH -> CONFIRM -> REPORT -> RESUME -> ... -> RTL -> LAND -> DONE`,
  every transition written to `/events`.
- `make search` (PX4) and `make search-fake` (no simulator, about 2 s).

### What broke, and what was learned

- **Lanes left corners uncovered.** The first planner cut each lane where
  its CENTRE LINE left the polygon. On a triangle or a tilted area the edge
  is slanted, so ground beside the centre line reaches farther. The coverage
  test caught it on 9 of 16 polygon and heading combinations. Fix: a lane
  runs from the first to the last point of the area anywhere in its strip.
- **A camera that looks ahead needs a lead-in.** At 45 degrees the image
  centre is `height / tan(pitch)` ahead of the drone (15 m at 15 m). Each
  lane starts that far before the area's edge.
- **Altitude is a trade.** Higher means a wider swath and fewer lanes, but
  smaller people. With this camera a 1.7 m person in the image centre is
  43 px tall at 15 m, 26 px at 25 m and 16 px at 40 m. At 40 m the swath is
  limited by resolution, not by the field of view, and a 20 px requirement
  cannot be met at all.
- **Where is the person, given a box?** Casting the box CENTRE onto the
  ground lands behind the person (the centre is waist-high), by metres at a
  shallow angle. That put the first confirmation orbits up to 3 m off. The
  middle of the box's BOTTOM edge is where the body meets the ground and is
  much closer.
- **The orbit radius follows from the camera.** A fixed camera pitched 45
  degrees keeps a target centred only when circling at
  `height / tan(pitch)` with the nose towards the centre. No gimbal needed.

### Numbers (flight `pilot-meadow-search`, PX4 SITL, oracle detector)

- 216 s flight, 2 lanes 30 m apart, swath half-width 25.5 m, planned path
  232 m, search altitude 15 m, confirmation altitude 10 m.
- 3 of 3 people sighted, confirmed (108 frames in view each) and reported.
- Reported position error against ground truth: 0.22, 0.87 and 0.47 m.
- 17 state transitions, all in `/events`.
- Episode file: 177 MB for 2121 frames.

### How far to trust those numbers

- **The detector is an oracle.** It projects true positions with the
  drone's pose and declares a sighting. It shows that the flying logic
  works. It says nothing about finding people in images, and it ignores
  occlusion.
- **The position errors are optimistic.** The oracle computes the pixel
  with PX4's estimated pose and the mission casts it back to the ground
  with the same pose, so the pose error cancels out. What remains is only
  the bias of using the box's bottom edge. With a real detector the pose
  error (roughly 1 m, see Phase 2A) will be added.
- **The pictures do agree.** Frames taken during the three orbits show each
  person within about 60 px of the image centre (`docs/images/phase2b_confirm.jpg`).
- All three people were sighted while flying TO the first lane, before the
  lanes proper. The lanes were still flown in full.

### Not done

- No obstacle avoidance: the mission relies on flying above the tallest tree.
- A person never confirmed is retried after 30 s, forever; there is no cap.
- Battery low was tested with the fake drone only.

## Phase 2C - Batch runner (2026-11-15)

### Done

- `sardrone/batch.py`: for each matrix row, generate the world, start
  Gazebo and PX4, fly the search in its own process with a 15-minute limit,
  stop the simulator, append one manifest line.
- Manifest (CSV): episode id, attempt, outcome, success, people placed and
  found, detector, world, seed, flight and wall time, size, commit, dirty
  flag, start time, file, error, and the full matrix row as JSON.
- Resume: an episode counts as done when the manifest says `completed` AND
  its files exist. A failed episode is retried once.
- Matrix files can hold a seeded `sample` block; values are dealt like
  cards so each option appears about equally often.
- The FiftyOne loader copies the new sidecar fields.

### Result: `configs/pilot10.yaml`, PX4 SITL, oracle detector, commit ddc06b4

| | |
| --- | --- |
| Episodes | 10 of 10 completed, first attempt, no failures or timeouts |
| Unattended wall time | 43.8 min |
| Flight time recorded | 2308 s (38.5 min), 22,881 camera frames |
| People | 30 placed, 30 sighted, confirmed and reported |
| Mean / max reported position error | 0.43 m / 0.87 m (optimistic, see Phase 2B) |
| Disk | 1.89 GB (115 to 370 MB per episode) |
| Mix | 4 forest edge, 3 meadow, 3 park; 4 noon, 3 golden hour, 3 dusk; 15 m and 25 m; 4 with occlusion |

The manifest is saved as `docs/results/pilot10.manifest.csv`. Running the
same command again skipped all ten. In FiftyOne the ten samples filter by
world, time of day, altitude and detector.

### What broke, and what was learned

- **Do not edit code while a batch runs.** Each episode is a new process
  that imports the code from disk, but the commit hash is read once at the
  start. An edit mid-batch would give later episodes different code under
  the same recorded commit. I committed the runner first, then ran, so every
  line says `ddc06b4`, `dirty: no`.
- **Logs were empty while episodes ran.** Output written to a file is
  buffered. The batch now sets `PYTHONUNBUFFERED` for episodes and flushes
  its own progress lines. (Fixed after this run.)
- **A leftover simulator would be reused with the wrong world.** PX4
  attaches to any running Gazebo. If a batch is killed hard, Gazebo
  survives. The launcher now refuses to start when a simulator is already
  running; checked by starting a second one on purpose.
- **Sampling needed balance.** Independent random picks over 10 rows can
  give 7 meadows. Dealing values from a shuffled deck gives 3 or 4 of each.
- **JPEG quality is not the lever for size.** Re-encoding at quality 75
  saved under 10 %; the fine grain of the ground texture dominates. Kept at
  85 and full resolution.

### How far to trust this

- 30 of 30 found is the ORACLE's result. It ignores occlusion, so the four
  occluded episodes say nothing about seeing people under trees.
- Retry, timeout and disk-full handling are covered by tests with a
  stand-in runner. None of them happened in the real run, so none is proven
  against the simulator. Killing the batch mid-episode was not tried.
- Episodes ran 144 to 371 s. Three exceed the plan's 2 to 4 minutes; five
  people at 25 s per confirmation add up.

### Projection for the v1 dataset (75 episodes)

At this run's averages (231 s, 189 MB, 263 s wall per episode): about
14 GB and 5.5 hours. 30 GB were free before this batch.

## Phase 2D - Frames and ground-truth labels (2026-11-15)

### Done

- **Labels come from Gazebo's bounding-box camera**, the plan's first
  choice. It works headless on this Mac. The drone's camera link carries
  two sensors with the same lens and rate: the colour camera, and the
  bounding-box camera, which reports a pixel rectangle around the VISIBLE
  part of every labelled object. Each person model has its own label number.
- The image bridge forwards a picture only together with the boxes Gazebo
  rendered at the same simulation time.
- `/ground_truth/boxes` (foxglove.ImageAnnotations) is written with every
  camera frame. An empty message means "checked, nobody in view".
- `sardrone/mcap/reader.py` reads labelled frames back at 2 Hz.
- `make frames` builds the FiftyOne image dataset `sar-frames`; each sample
  has `ground_truth` Detections and links to `episode_id` and `timestamp_ns`.
- `make overlay` saves 20 seeded random frames with boxes and 4x crops.

### Result (pilot batch re-recorded at commit 56c0763, plus one earlier search)

| | |
| --- | --- |
| Batch | 10 of 10 completed, 44.5 min, 1.87 GB, 30 of 30 reported (oracle) |
| `sar-frames` | 4,678 frames from 11 episodes, 6,355 boxes, 403 MB of JPEGs |
| People per frame | 0: 1603, 1: 1144, 2: 1073, 3: 417, 4: 391, 5: 50 |
| Box size (longer side) | median 30 px, 5th percentile 11 px, 95th 62 px, range 4 to 165 |
| Truncated at the image border | 245 of 6,355 |
| By pose | standing 2898, sitting 2252, lying 1205 |
| By world | forest edge 1714, park 1623, meadow 1341 frames |
| By time of day | noon 1820, dusk 1461, golden hour 1397 frames |

### The count check ("do box counts match the people in view?")

Each frame also gets `expected_by_geometry`: how many people's 3D centres
project inside the image, from PX4's pose, ignoring trees. That is a
second, independent way to count.

| Episodes | Frames | Same count | Labels fewer | Labels more |
| --- | --- | --- | --- | --- |
| occlusion `none` | 3146 | 92.6 % | 3.2 % | 4.2 % |
| occlusion `partial` | 1532 | 63.3 % | 34.7 % | 2.0 % |
| all | 4678 | 83.0 % | 13.5 % | 3.5 % |

They are not meant to be equal. Where people stand under trees, a third of
the frames have FEWER labels than geometry expects: the canopy hides them,
and the sensor only boxes what is visible. That is the behaviour wanted
from labels. In the open the two agree in 92.6 % of frames; the rest are
people at the image border, where a visible arm gets a box while the body
centre is outside (or the reverse), plus PX4's pose error. No frame has more
boxes than people placed.

The 20 overlays (`docs/images/phase2d_overlays.jpg`) were looked at by me:
every box is on a person, in all three world types and all three times of
day, and frames without people have no boxes. Kartik should look too; that
is what the overlays are for.

### What broke, and what was learned

- **The first idea would not have worked.** Phase 2A showed that
  projecting 3D boxes with PX4's pose misses by 20 to 40 px. The sensor
  renders the labels from the same view as the picture, so pose error does
  not enter at all, and occlusion comes for free.
- **Gazebo gives the first and last pixel of a box, both inside it.**
  Adding 1 to the last pixel gives edges, so width = x1 - x0. Off-by-one
  errors here would shift every label by a pixel.
- **Empty slivers.** At the image border the sensor sometimes reports a box
  of zero width. Boxes under 1 px are dropped when recording, and boxes
  under 4 px on the longer side when extracting (39 dropped in total).
- **I overwrote a test file.** My new `tests/test_frames.py` replaced the
  Week 1 file of the same name (the NED/ENU conversion tests). The suite
  still passed, with 20 fewer tests, and only the count gave it away.
  Restored in the next commit; the new tests are in
  `tests/test_labelled_frames.py`.
- **The whole pilot had to be recorded again** (45 min) because labels are
  captured at flight time. The old unlabelled episodes were deleted.

### How far to trust this

- The labels are exact for THIS renderer. They say nothing about realism.
- "Visible" is decided per pixel by Gazebo; a person with only a hand
  showing still gets a (small) box. No minimum visible fraction is applied.
- 31 % of the frames are from confirmation orbits around one person, so
  they are near-duplicates. Phase 2E deals with that.
- All frames come from flights where the oracle steered the drone towards
  people, so people are over-represented compared with a blind search.

## Phase 2E - The FiftyOne loop: curate, embed, label (2026-11-15)

### Done

- `make curate`: CLIP ViT-B/32 embeddings (stored as unit vectors), a
  similarity index and a UMAP map through `fiftyone.brain`, near-duplicate
  tagging, a train/val split by whole episode, and saved views.
- `make autolabel`: OWLv2 (`google/owlv2-base-patch16-ensemble`) prompted
  with "a person", "a backpack", "a tent", "a jacket" writes `auto_labels`;
  those are scored against ground truth and the least confident 5 % become
  a review queue.
- New `ml` extra: torch 2.15, torchvision, transformers 5.23, umap-learn.
  Model weights are cached in `.cache/` inside the project (about 950 MB).

### Results on the pilot data (4,678 frames, 11 episodes)

| | |
| --- | --- |
| Embedding time | 65 s for 4,678 frames (CPU) |
| Near-duplicates tagged | 573 (12.2 %) at cosine distance 0.006; 5 of them show a different number of people than the frame kept |
| Split | train 3,937 frames (9 episodes), val 741 (pilot-009, pilot-010) |
| `train-v1` | 3,472 frames (train, duplicates left out) |
| `val-v1` | 633 frames |
| `sar-hard-cases` | 2,480 frames (lying, occluded, under 16 px, or anyone at dusk) |
| `sar-false-alarms` | 0 (filled when the first trained model is evaluated) |

Threshold sweep (within-episode cosine distance):

| Threshold | Tagged | Share | Different people count |
| --- | --- | --- | --- |
| 0.002 | 86 | 1.8 % | 0 |
| 0.003 | 205 | 4.4 % | 0 |
| 0.004 | 326 | 7.0 % | 1 |
| 0.006 | 573 | 12.2 % | 5 |
| 0.010 | 1241 | 26.5 % | 41 |
| 0.015 | 2132 | 45.6 % | 127 |

0.006 is the largest value at which under 1 % of the removed frames differ
in people count from the frame kept (5 of 573 = 0.9 %).

### Auto-labeller against ground truth (commit 71571aa, 4,105 unique frames, 6,078 people)

Full tables: `docs/results/auto_labeler.md`.

- mAP@0.5:0.95 for `person`: 0.541.
- Best F1 0.831 at confidence 0.1: precision 0.902, recall 0.770.
- At confidence 0.2: precision 0.991, recall 0.531. At 0.05: precision
  0.695, recall 0.829.
- Recall by occlusion: 0.927 in the open, 0.370 under trees.
- Recall by size: 0.493 under 16 px, 0.829 at 16 to 32 px, 0.925 at 32 to 64 px.
- Recall by pose: standing 0.904, lying 0.799, sitting 0.751.
- It also drew 3,970 "backpack", 2,924 "tent" and 1,723 "jacket" boxes.
  The worlds contain no backpacks or tents, so those 6,894 are all wrong.
- Review queue: 362 boxes at confidence 0.055 or less, 36 of them correct.
- Speed: 0.52 s per frame on the Mac's GPU (MPS), 35 min for the run.

### What this means

- **An auto-labeller is a starting point, not a labeller.** At a cut-off
  that keeps 9 in 10 boxes correct it misses nearly a quarter of the
  people, and it misses exactly the hard ones: under trees and small.
  Training only on its labels would teach a model to ignore those.
- **Its confidence is not a probability.** No box scores above about 0.45,
  and the useful cut-offs are between 0.05 and 0.2. A cut-off must be
  chosen from data like this, not assumed.
- **The review queue works as intended.** Only 10 % of the least confident
  boxes are right, so that is where a human's time removes the most errors.
- **The map shows how narrow the data is.** Frames group by world and
  lighting, which here means by episode, into about ten islands. Whether a
  person is in the frame barely moves a picture on the map: at 30 px a
  person is too small to change a whole-image embedding. So this map can
  show scene variety, but not label variety.

### What broke, and what was learned

- **FiftyOne's duplicate finder did not behave like a cosine threshold.**
  At a threshold of 0.005 it found no duplicates although 1,678 pairs are
  closer than that, and the distances it reported did not match cosine or
  Euclidean distance on the same vectors, before or after I normalised
  them. I did not find out why. Rather than tune a number I could not
  explain, de-duplication is 25 lines of explicit code on the same
  embeddings, with hand-worked tests. The first commit of this phase
  never contained the FiftyOne-based version; an earlier run of mine
  (504 tagged at "0.05") is superseded.
- **My guess about confirmation orbits was wrong.** In Phase 2D I wrote
  that the 31 % of frames from orbits are near-duplicates. Measured, only
  82 of the 573 duplicates are orbit frames; 298 are from the search lanes,
  where the drone flies over plain grass. Circling changes the picture more
  than flying straight over an empty field.
- **Worker processes crashed** when embeddings were computed from a
  script fed through a pipe; images are now loaded in the main process.
- **I broke three modules for a minute** by removing `import os` while its
  uses remained. The test suite caught it before the commit.

### How far to trust this

- Every number is on SIMULATED images. OWLv2 was trained on real photos,
  so its scores here are not its scores on real drone footage, in either
  direction.
- The time-of-day figures are confounded. Dusk has the highest recall
  (0.920) only because all three dusk episodes happen to have no occlusion.
- IoU 0.5 is strict for 30 px boxes, but loosening it to 0.3 changes
  recall by about 1 point, so localisation is not the problem; missing
  people is.
- The val split is two pilot episodes. It is enough to debug training, not
  to report results. The plan's 60 + 15 episode dataset is not recorded yet.

## Phase 3 groundwork, written while the v1 batch records (2026-11-15)

The 75-episode v1 batch takes about six hours. Training would compete with
the simulator for CPU and GPU and could push episodes past their time
limit, so only code the batch does not use was written meanwhile. No file
under `sardrone/` that an episode imports was touched after the batch
started at commit 33d89df.

### Done

- **v1 matrix** (`configs/v1.yaml`): 60 training and 15 sim-test episodes
  from two blocks with different seeds. Splits now come from the episode
  id: `test-*` is sim-test, every 10th `train-*` is validation.
- **SARD** (`make sard`): 1,981 real images, 6,532 people, CC BY 4.0, from
  the authors' open Zenodo record (IEEE DataPort wants a subscription).
  MD5 checked against the published value. `sard-test` (1,387 images,
  frozen) and `sard-train-pool` (594), cut by frame number.
- **YOLO export** (`make export`): symlinks, labels, `dataset.yaml`, and a
  hash over every image and label.
- **Metrics** (`sardrone/train/metrics.py`): matching, COCO-style AP,
  precision and recall at a cut-off, bootstrap intervals over images.
- **Scorecard**, **training**, **prediction** and **evaluation** commands
  (`make smoke-train`, `make train`, `make evaluate`).

### Checks

- My AP code against FiftyOne's, on the 4,105 pilot frames with OWLv2
  auto-labels: mAP@0.5:0.95 0.5411 against 0.541; precision / recall at
  cut-offs 0.05, 0.1, 0.2 agree to three decimals. 1,000 bootstrap draws
  take 0.9 s.
- Smoke test on the CPU: 48 training and 16 validation frames, 2 epochs at
  640 px, 0.4 min. Export, train, predict and score all ran. The scores
  from it mean nothing.

### What broke, and what was learned

- **The dry run caught a broken third of the dataset.** Before the real
  batch I ran all 75 rows with the fake drone. Every 40 m mission was
  refused: the stand-in camera was 640x480, where a person at 40 m is
  11 px, under the 12 px limit; the real 1280x960 camera gives 16 px. The
  mission was right and the dry run was wrong, so the stand-in now has the
  real camera's size and lens. Six minutes of dry run against hours lost.
- **Two OpenCV packages cannot share an environment.** Ultralytics depends
  on `opencv-python`, FiftyOne on `opencv-python-headless`; both install
  the same `cv2` module. Ultralytics is installed without its dependencies
  and uses the OpenCV that is already there.
- **Real people are twice as big.** Median box in SARD is 56 px on the
  longer side in 1920x1080 images, against 30 px in our 1280x960 frames.
  Scaled to the same 1280 px training width that is 37 against 30 px, so
  the training image size (1280, not Ultralytics' default 640) matters for
  both.
- **SARD looks nothing like the simulation.** Tall dry grass, quarry
  stone, dirt roads, forest shade. A large gap is to be expected.

### Not done, waiting for the batch

- `make evaluate` has not run end to end: it needs the `sim-test` view,
  which does not exist until the v1 frames are extracted.
- No real training. The smoke test proves the plumbing only.
- Disk: 18 GB were freed (tool caches, an unrelated 14 GB language model,
  the 11 pilot recordings) with Kartik's approval.

## v1 dataset, and Phases 3A and 3B: the first model and the first gap (2026-11-16)

### The v1 dataset (commit 33d89df)

| | |
| --- | --- |
| Batch | 75 of 75 episodes completed on the first attempt, 6.2 h unattended, no failure, timeout or retry |
| People | 225 placed, 225 reported (oracle) |
| Recorded | 5.4 h of flight, 15.2 GB |
| Frames at 2 Hz | 37,014 from v1 (41,692 with the pilot frames) |
| Near-duplicates | 2,832 tagged at cosine distance 0.004 (the 1 % rule; 0.006 gave 1.8 % on this data) |
| `train-v1` | 24,667 frames, 54 episodes, 34,909 boxes, 7,270 empty frames, median box 28 px |
| `val-v1` | 2,861 frames, 6 episodes, 4,735 boxes |
| `sim-test` | 7,453 frames, 15 episodes, 10,541 boxes (duplicates kept) |
| Shared world seeds between splits | 0 |

The validation episodes happen to contain no 15 m flights (four at 40 m,
two at 25 m), so validation people are smaller than average.

### Phase 3A: training v1

- Export `v1-6k`, hash `d741bdbb19fbe45d`: a seeded sample of 5,999 of the
  24,667 `train-v1` frames, taken within each episode in proportion to its
  size. Reason: at 1280 px this Mac trains about 8 images per second, so a
  full epoch would take 52 minutes and 10 epochs most of a day.
- YOLO11n from COCO weights, 10 epochs, 1280 px, batch 8, MPS, seed 51,
  Ultralytics 8.4.214, torch 2.15.0, commit 11045a0. 150 minutes.
- Ultralytics' own validation on `val-v1`: mAP@0.5 0.953, mAP@0.5:0.95 0.720,
  still rising at the last epoch.

### Phase 3B: evaluation

Confidence cut-off chosen on `val-v1` (best F1), then fixed. 1,000 bootstrap
draws over images. Full rows: `docs/results/scorecard.md`.

| Version | Sim-test mAP@0.5 | SARD mAP@0.5 | Gap |
| --- | --- | --- | --- |
| v0: COCO-pretrained YOLO11n, never trained on our data | 0.453 (0.440 to 0.465) | 0.591 (0.574 to 0.607) | -0.138 |
| v1: fine-tuned on simulation only | 0.947 (0.944 to 0.953) | 0.014 (0.009 to 0.019) | 0.933 (0.927 to 0.941) |

- v1 on SARD at its cut-off: precision 0.19, recall 0.01. It draws 402
  boxes above confidence 0.25 across 1,387 images holding 4,971 people.
- v0 on SARD at its cut-off: precision 0.83, recall 0.51.

### What this means

**Training on simulation alone did not transfer. It made the detector
worse on real images than not training it at all.** The pretrained network
could already find half the real people. Ten epochs on simulated frames
taught it to find 93 % of simulated people and left it nearly blind to real
ones. This is "catastrophic forgetting": with every example of `person`
now a rendered mannequin on rendered grass, the network redefined the
class around those and let go of what it knew.

So the gap is not only "the simulation is unrealistic". The training
recipe matters as much as the data: how much the network is allowed to
change, and whether it ever sees a real person while it does. That shapes
the next versions.

Picture: `docs/images/v1_on_sard.jpg` (truth magenta, v0 cyan, v1 yellow;
v1 draws nothing on these four).

### Checks made before believing 0.014

- The real-image pipeline is sound: the same code gives the untrained v0
  0.591 on the same images.
- SARD labels were drawn on random images and sit on the people.
- The metric code matches FiftyOne's on the pilot auto-labels.
- v1 is not broken: it scores 0.947 on simulated worlds it never saw.

### The image-level gap, without any detector

A logistic regression on CLIP embeddings tells `sim-test` frames from
`sard-test` frames 100 % of the time (5-fold), and still 100 % after the
camera-realism treatment; random halves of one domain score 50 %. On the
map the two form separate islands (`docs/images/gap_map.png`). Noise, blur
and JPEG artefacts do not move simulated pictures towards real ones as far
as CLIP is concerned, which is a warning that v3 alone may not help.

### How far to trust this

- v1 saw a quarter of the training frames, for 10 epochs. A longer run on
  all of them would score higher on simulation. Nothing here suggests it
  would do better on real images.
- The sample, epochs and settings are held fixed for v3 and v4, so that
  each version changes one thing.
- `sim-test` contains near-duplicates on purpose, so its interval is
  narrower than the number of independent scenes (15 episodes) justifies.
- SARD's `difficult` boxes are counted as ordinary people.

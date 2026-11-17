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

## Phase 3C: closing the gap, one change per version (2026-11-16)

The plan orders the versions v2 (domain randomization), v3 (camera
realism), v4 (+5 % real). v2 needs new episodes recorded, which takes
hours of simulator time, so tonight's order is v3, then v4. Each row says
what it was built on.

### v3 = v1 + camera realism

- Export `v3-6k`, hash `86479461bb4f3bb7`: the same 5,999 frames as v1, each
  degraded once (exposure 0.8 to 1.2, gamma 0.85 to 1.15, blur on half of
  them up to 1.2 px, noise sigma 1 to 6, JPEG quality 35 to 85), seeded by
  file name. Validation images are untouched.
- Same network, epochs and settings as v1. 149.8 minutes, commit 06d5f53.

| Version | Sim-test mAP@0.5 | SARD mAP@0.5 | Gap |
| --- | --- | --- | --- |
| v1 | 0.947 (0.944 to 0.953) | 0.014 (0.009 to 0.019) | 0.933 |
| v3 | 0.941 (0.935 to 0.945) | 0.020 (0.017 to 0.023) | 0.921 (0.913 to 0.925) |

**Camera realism did almost nothing.** The real score went from 0.014 to
0.020. The two intervals overlap (0.009 to 0.019 and 0.017 to 0.023), so
even that small change is not established; either way it is useless in
practice: recall on SARD is still 1.6 %. (An earlier version of this entry
said the intervals do not overlap. That was wrong.) The
simulated score dropped slightly, as expected when training images are
made harder.

This matches what the embedding check said beforehand: degraded simulated
frames were still 100 % separable from real ones. The difference between
these worlds and SARD is not sensor noise. It is content: three body
meshes on flat textured ground against real people in long grass, on
stone and on roads.

### v4 = v3 + 5 % real images

- Export `v4-6k`, hash `7918bf71d200e0da`: v3's 5,999 degraded simulated
  frames plus 300 real images drawn with
  seed 51 from `sard-train-pool` (594 images, frames 1 to 621 of the SARD
  sequence). `sard-test` (frames after 621) is untouched.
- Same network, epochs and settings. 161.2 minutes. The run record says
  `dirty: True`: when training started, the scorecard files written by
  the v3 evaluation were not committed yet. No code differed from commit
  e316ec6.

| Version | Sim-test mAP@0.5 | SARD mAP@0.5 | Gap | SARD precision / recall |
| --- | --- | --- | --- | --- |
| v0 pretrained | 0.453 | 0.591 (0.574 to 0.607) | -0.138 | 0.83 / 0.51 |
| v1 simulation only | 0.947 | 0.014 (0.009 to 0.019) | 0.933 | 0.19 / 0.01 |
| v3 + camera realism | 0.941 | 0.020 (0.017 to 0.023) | 0.921 | 0.13 / 0.02 |
| v4 + 5 % real | 0.934 (0.931 to 0.941) | 0.547 (0.529 to 0.570) | 0.387 (0.364 to 0.408) | 0.70 / 0.59 |

**Three hundred real images did what six thousand simulated ones could
not.** The real score went from 0.020 to 0.547, and the simulated score
barely moved (0.941 to 0.934). The gap fell from 0.92 to 0.39.

What v4 is, and is not:

- It is the first version that works in BOTH domains: 0.93 on simulated
  worlds and 0.55 on real photos. v0 manages 0.45 and 0.59.
- It does not beat the untrained network on real images (0.547 against
  0.591; the intervals do not overlap, so v0 is still slightly better
  there). Its recall on SARD is higher (0.59 against 0.51) and its
  precision lower (0.70 against 0.83), at cut-offs chosen on simulated
  validation data, which is not the right place to tune for real images.
- So on this evidence the simulated data adds nothing to REAL performance.
  What it adds is performance on the simulated worlds the drone actually
  flies in, without losing the real-world ability.

### How far to trust v4's real score

- **The real training images and the real test images are from the same
  footage.** SARD is one set of video shoots; the pool is its first 621
  frames and the test set the rest. Scenes, people and clothing recur. So
  0.547 measures "more of the same shoot", not "a new place". A second
  real dataset (WiSARD, HERIDAL) would be the honest test of that.
- One training run per version, one seed. The bootstrap interval covers
  the choice of test images, not the randomness of training. A v4 trained
  again might land a few points away.
- 5 % is of the 6,000-frame sample, which is 300 images, or half of the
  pool. The plan's "+5 %" of the full training set would be 1,233 images,
  more than the pool holds.

## Phase 3E: workflow and retraining gate (2026-11-16)

- `sardrone/pipeline.py`, `make pipeline`: ingest and curate always run;
  a gate then allows export, train and evaluate only if `sar-hard-cases`
  has grown by at least 200 frames since the last training. Dry run by
  default. The state is `runs/pipeline_state.json`.
- Recorded today's state for v4: 21,701 hard frames. A dry run then
  reports 0 new hard frames and skips the three expensive stages.
- No Airflow. `docs/WORKFLOW.md` gives the reasons (one machine, fixed
  order, dependency conflicts) and a DAG sketch that has not been run.
- Not done: the plan's optional S3 storage and container evaluation job.
- Not tried: `make pipeline RUN=1` end to end. Its stages have each been
  run by hand; the gate logic is covered by tests with a stand-in runner.

## Phase 4A: mission agent (2026-11-16)

### Done

- Six tools as JSON schemas; a deterministic mission simulator; the agent
  loop with a full transcript; a Cohere back-end, a rule-based baseline
  and a scripted model; 20 written missions with checks; `make agent-eval`.
- The Cohere adapter was written against the installed SDK (cohere 7.3.1):
  `ClientV2.chat(model, messages, tools, temperature, seed)`, tool calls at
  `response.message.tool_calls[i].function.{name, arguments}`, token counts
  at `response.usage.tokens`.

### Result so far (no language model)

| Model | Missions succeeded | Valid tool calls | Mean steps |
| --- | --- | --- | --- |
| rule-based keywords | 9 of 20 (45 %) | 100 % | 10.5 |
| reference policy, told the answers | 20 of 20 | 100 % | (test only) |

The baseline fails m04 (altitude written as "15 metres"), m05 and m14
(two areas in order), m06, m08, m11, m15, m17 (who to report needs
understanding: a child not an adult, lying not standing, two colours,
"except people in black", white shirt AND sitting), m07 (confirm but do
not report), m12 ("the field to the north") and m20 ("do not launch").

### What this means

- The evaluation discriminates: nine missions are solvable by keyword
  matching, eleven are not. A language model has to earn its place on
  those eleven.
- The reference policy passing all 20 shows that every mission can be
  solved within the battery budget and that the scorer accepts a correct
  run. Without that check, a mission the simulator made impossible would
  look like a model failure.

### Not done

- **No Cohere call has been made.** It needs `COHERE_API_KEY` and
  `COHERE_MODEL` in `.env`, and it costs money. The adapter is therefore
  untested against the live API.
- Cost is only computed from prices given on the command line. None is
  assumed.
- The agent acts on the simulator, not the drone. Its decisions are passed
  to an `on_decision` hook, which is where writing to `/events` would go;
  that is not wired to a real flight.
- One run per mission. With a real model at temperature 0 and a fixed seed
  the plan is still to repeat runs, since providers do not guarantee
  identical output.

## Phase 3D: detection in the loop and object memory (2026-11-16)

### Done

- The search mission runs its detector once per camera frame and feeds the
  boxes to a tracker (a constant-velocity Kalman filter per box, Hungarian
  matching, coasting for up to 15 frames). It acts only on tracks seen
  three times.
- `YoloDetector` puts a trained model behind the oracle's interface. It
  gives no identities: targets are told apart by ground position (4 m),
  and a report counts as a find only if a true person is within 3 m.
- `/detections` (ImageAnnotations with track labels) on every frame;
  events `target_acquired` and `target_lost`.
- `mission_metrics` in every sidecar; `sardrone.mission.report` tabulates
  a folder of episodes.
- `make search --detector v4`, `sardrone.batch --detector v4 --prefix test-`.

### Result: 15 sim-test worlds, closed loop, detector v4 (cut-off 0.378)

PX4 SITL, 93.5 minutes for 14 flights (the first was flown beforehand).
Full table: `docs/results/closed_loop_v4.md`. For comparison, the oracle
found 45 of 45 on the same worlds.

| | Closed loop (v4) | Oracle |
| --- | --- | --- |
| People found | 39 of 45 (86.7 %) | 45 of 45 |
| Episodes with everyone found | 10 of 15 | 15 of 15 |
| False reports (nobody there) | 0 | 0 |
| False alarms rejected by the orbit | 9 | 0 |
| Missed rescues (visible in images, never reported) | 6 | 0 |
| Searches flown to the end | 15 of 15 | 15 of 15 |
| Mean time to first find | 56 s | |
| Mean position error of reports | 0.45 m | |
| Mean cross-track error | 0.03 m | |
| Tracker ADE, all frames / while coasting | 5.6 px / 24.8 px | |
| Mean flight time | 344 s | 260 s |

Where the six misses are:

| | Found |
| --- | --- |
| Occlusion none / partial | 33 of 34 / 6 of 11 |
| Altitude 15 m / 25 m / 40 m | 14 of 14 / 14 of 17 / 11 of 14 |
| Meadow / park / forest edge | 15 of 15 / 15 of 15 / 9 of 15 |
| Noon / golden hour / dusk | 7 of 9 / 16 of 17 / 16 of 19 |

### What this means

- **All six misses are in forest-edge worlds, five of them people under
  trees.** In the open the closed loop found 33 of 34. That is the same
  weakness the auto-labeller had, and the same one the label statistics
  predicted: under a canopy a person is a few visible pixels.
- **All six were "missed rescues":** the simulator's label sensor did see
  them in at least 10 frames, so they were in view, but the detector never
  held them long enough to trigger a confirmation. More coverage would not
  have helped; a better detector, a lower altitude or a second pass would.
- **Nothing was reported that was not there.** Nine times the detector
  held a false track for three frames and the drone flew a confirmation
  orbit; nine times the orbit rejected it. That is the orbit doing its
  job, and it is not free: closed-loop flights were a third longer than
  the oracle's.
- **Lower is better for finding, and costs time.** 14 of 14 at 15 m
  against 11 of 14 at 40 m, on few episodes.

### What broke, and what was learned

- **Overlap-based matching fragments small objects.** The first
  closed-loop flight found 3 of 3 people but as 70 separate tracks. A
  30 px box often does not overlap its own position one frame earlier when
  the drone turns. Matching by centre distance, gated at 1.5 body-lengths,
  cut that to 47, with the same mission result. It is still fragmented:
  people leave and re-enter the picture, and nothing compensates for the
  drone's own motion. The mission is not affected because identity is
  decided by ground position, not by track id.
- **A one-frame glimpse should not move a drone.** With the tracker's
  three-hit rule a single false box no longer causes a detour; a test
  changed from "false alarm" to "nothing happens" to reflect that.
- **The first closed-loop flight worked first time.** The fake-drone tests
  with a detector stripped of identities, plus one with a persistent false
  positive, had already exercised every path.

### How far to trust this

- 15 flights, one per world, one run each. The breakdowns above have
  between 9 and 34 people per cell; they point in a direction and prove
  little.
- The detector was trained on worlds built by the same generator as the
  test worlds, with the same three body meshes. This is closed-loop
  performance in simulation, not a claim about real flights.
- "Found" needs a report within 3 m of a true person. Position error
  includes PX4's pose error here, unlike with the oracle.
- The tracker ADE compares the tracker's one-frame-ahead prediction with
  the simulator's label box centre, matched by nearest box.

## Phase 3C, v2: domain randomization (2026-11-16)

### Done

- `randomize: true` in a matrix row also varies: ground palette (7
  families from lush green to bare earth) and patch scale, footpaths in
  any world, sun azimuth all round the compass, elevation, brightness,
  ambient light and colour temperature, clothing (21 colours), body size
  (0.85 to 1.1), and 15 to 45 unlabelled distractors (rocks, logs, bushes,
  coloured boxes). Rows without the switch are unchanged: all 75 recorded
  v1 worlds were regenerated and compared, with no difference.
- `configs/v2.yaml`: 30 randomized training episodes, recorded at commit
  c9ab4b8 in 2.5 h. 30 of 30 completed; one (dr-023) needed its retry
  because PX4 was not ready within 60 s. 90 of 90 people reported (oracle).
  14,241 frames, of which 13,760 after de-duplication (`train-v2-dr`).
- Export `v2-6k`, hash `61b514bcb1580625`: 6,000 of those frames by
  episode. Same network, epochs and settings as v1. 177.8 minutes, commit
  af178c1. Validation and test sets are the v1 ones (plain worlds).

### Predicted before training

The image-level check on the randomized frames, run while v2 trained:

| | |
| --- | --- |
| Randomized simulated vs real: linear classifier accuracy | 100 % |
| Randomized vs plain simulated | 93.2 % |
| Mean cosine distance to the nearest real frame, plain simulated | 0.291 |
| ... randomized simulated | 0.303 |
| ... between real frames | 0.023 |

Randomization made simulated frames different from each other and no
closer to the real ones. Prediction: v2 will not close the gap.

### Result

| Version | Sim-test mAP@0.5 | SARD mAP@0.5 | Gap | SARD precision / recall |
| --- | --- | --- | --- | --- |
| v0 pretrained | 0.453 | 0.591 (0.574 to 0.607) | -0.138 | 0.83 / 0.51 |
| v1 simulation only | 0.947 | 0.014 (0.009 to 0.019) | 0.933 | 0.19 / 0.01 |
| v2 domain randomization | 0.935 (0.931 to 0.941) | 0.032 (0.029 to 0.038) | 0.902 (0.896 to 0.910) | 0.27 / 0.04 |
| v3 camera realism | 0.941 | 0.020 (0.017 to 0.023) | 0.921 | 0.13 / 0.02 |
| v4 + 5 % real | 0.934 | 0.547 (0.529 to 0.570) | 0.387 | 0.70 / 0.58 |

- **The prediction held.** The real score roughly doubled, from 0.014 to
  0.032, and the intervals do not overlap, so this is a real effect. It is
  also useless: 4 % recall on real images.
- **Randomization cost almost nothing in simulation.** v2 never saw a
  plain world and still scores 0.935 on the plain test worlds (v1: 0.947),
  from 30 flights instead of 54.
- **Of the three levers the plan named, only real data moved the real
  score in a way that matters.** Rendering more variety of the same
  content (v2) and degrading the pixels (v3) both left it near zero.

### What this means for the simulator

Randomizing colours, light and clutter is what this generator can do
cheaply, and it is not what separates it from SARD. The differences are
structural: body shapes and poses from three meshes, no long grass or
undergrowth that partly covers a person, flat ground, and no real texture
at the scale of a 30 px figure. Closing the gap from the simulation side
would need new content, not new parameters.

### How far to trust this

- v2 and v1 differ in more than randomization: v2's 6,000 frames come from
  30 flights, v1's from 54. The comparison is "same budget of frames", not
  "same worlds plus randomization".
- v2 was not combined with real images. Whether randomization helps ON TOP
  of real data (a "v5") is untested, and is the more interesting question
  given these results.
- One run, one seed, as for every version.

## Follow-up experiments after the plan's versions (2026-11-16)

Two questions the plan's versions left open. Version numbers continue from
v4; each row states what it is built on.

### v5 = v1 with the backbone frozen

Question: v1 lost the pretrained network's ability on real images. Is that
because the feature layers were overwritten? If so, keeping them fixed
should keep the real score.

- Same export as v1 (`d741bdbb19fbe45d`), same settings, plus `freeze=11`:
  YOLO11's backbone is modules 0 to 10, so those keep their COCO weights
  and only the neck and detection head train. The training log lists the
  frozen layers. 113.5 minutes (faster, since fewer gradients are computed),
  commit 433ae47.

| Version | Sim-test mAP@0.5 | SARD mAP@0.5 | SARD precision / recall |
| --- | --- | --- | --- |
| v0 pretrained | 0.453 | 0.591 (0.574 to 0.607) | 0.83 / 0.51 |
| v1 all layers trained | 0.947 | 0.014 (0.009 to 0.019) | 0.19 / 0.01 |
| v5 backbone frozen | 0.943 (0.938 to 0.946) | 0.127 (0.119 to 0.138) | 0.35 / 0.16 |

- **Freezing the backbone kept nine times more real-world ability (0.127
  against 0.014) at no cost in simulation (0.943 against 0.947).** If only
  simulated data is available, freeze the backbone.
- **It is still a long way below the untrained network (0.591).** So the
  backbone was not the main thing lost. With the backbone identical to
  v0's, the difference is entirely in the neck and head: trained only on
  rendered figures, they learn to call only rendered figures a person.
  The features can still see real people; the head has stopped listening.
- This also explains why 300 real images were enough for v4: the head
  needed examples of what counts, not new features.

Limits: one run. "Frozen" covers the backbone only; the neck trained
freely. Lower learning rates or freezing the neck too were not tried.

### v6 = v4 with domain-randomized simulated frames

Question: randomization alone did nothing useful for real images (v2).
Does it add anything once real examples are in the training set?

- Export `v6-6k`, hash `1acc17da393128c2`: 6,000 frames from the 30
  randomized flights, degraded like v3's, plus the same 300 real images as
  v4 (same seed, same draw from `sard-train-pool`). Same settings. 165.4
  minutes. The run record says `dirty: True` because the v5 dev-log entry
  was not committed yet when it started; the code is commit 433ae47.

| Version | Simulated frames | Sim-test mAP@0.5 | SARD mAP@0.5 | Gap | SARD precision / recall |
| --- | --- | --- | --- | --- | --- |
| v0 pretrained | none | 0.453 | 0.591 (0.574 to 0.607) | -0.138 | 0.83 / 0.51 |
| v4 | plain + realism + 300 real | 0.934 | 0.547 (0.529 to 0.570) | 0.387 | 0.70 / 0.58 |
| v6 | randomized + realism + 300 real | 0.926 (0.922 to 0.931) | 0.597 (0.580 to 0.617) | 0.329 (0.307 to 0.346) | 0.51 / 0.70 |

- **With real images present, randomization helped: 0.597 against 0.547,
  and the intervals do not overlap.** Alone it was worth 0.018 (v2 against
  v1); on top of real data it is worth 0.050.
- **v6 is the first trained version level with the untrained network on
  real photos** (0.597 against 0.591; the intervals overlap, so "level",
  not "better"), while scoring 0.926 on simulated worlds, where the
  untrained network scores 0.453.
- It costs a little in simulation (0.926 against v4's 0.934) and its
  operating point is different: more recall and less precision on SARD
  (0.70 / 0.51 against 0.58 / 0.70), at a cut-off of 0.14 chosen on
  simulated validation frames.

A reading of v2 and v6 together: randomization does not teach the network
what a real person looks like, so alone it cannot help. It may stop the
network from leaning on the simulator's fixed look (one grass colour, one
sun direction, nothing else on the ground), which leaves the 300 real
examples more room to count. That is an interpretation; nothing here
tests it directly.

### How far to trust v5 and v6

- One run each, one seed. v4 and v6 differ by 0.050 on real images, and
  how much two trainings of the SAME recipe differ is not known yet. Repeat
  runs of v4 and v1 with other seeds are training now
  (`docs/results/seed_runs.csv`); until they are in, treat "v6 beats v4"
  as likely, not established.
- v6 also differs from v4 in the number of flights its simulated frames
  come from (30 against 54).
- The real training and test images share footage, as for v4.

### Repeat runs with other training seeds (overnight, 2026-11-17)

Question: every version was trained once. How much does a score move when
the SAME recipe is trained again? Three repeat runs, same export, same
settings, only the training seed changed (51 was the original). Commit
30c8c68. Results in `docs/results/seed_runs.csv`.

| Run | Sim-test mAP@0.5 | SARD mAP@0.5 (95 % interval over images) |
| --- | --- | --- |
| v1, seed 51 | 0.947 | 0.014 (0.009 to 0.019) |
| v1, seed 52 | 0.946 | 0.011 (0.009 to 0.017) |
| v4, seed 51 | 0.934 | 0.547 (0.529 to 0.570) |
| v4, seed 52 | 0.937 | 0.586 (0.567 to 0.606) |
| v4, seed 53 | 0.934 | 0.532 (0.513 to 0.554) |
| v6, seed 51 (one run) | 0.926 | 0.597 (0.580 to 0.617) |
| v0, untrained | 0.453 | 0.591 (0.574 to 0.607) |

**The training seed alone moves v4's real score across 0.054** (0.532 to
0.586). That is larger than the bootstrap intervals suggested: those
cover the choice of test images, and say nothing about training.

What survives, and what does not:

- **Survives: simulation-only training collapses on real images.** Both
  v1 runs are at 0.01. Not a fluke of one run.
- **Survives: the simulated score is stable.** 0.934 to 0.937 across v4's
  three runs.
- **Survives: v4 does not beat the untrained network on real images.**
  All three runs (0.532, 0.547, 0.586) are below v0's 0.591.
- **Does NOT survive: "v6 beats v4".** v6's 0.597 is above every v4 run,
  but only by 0.011 over the best one, and the v4 runs alone spread over
  0.054. With one v6 run, the difference from v4 cannot be told apart from
  training noise. The earlier entry for v6 said "likely, not
  established"; on this evidence it is not established at all.
- **Does not survive: "v6 is level with v0" as a statement about v6.** It
  is true of that one run; a second v6 run could land anywhere in a range
  like v4's.

Practical rule from this: on this setup, a real-image difference smaller
than about 0.06 between two single training runs means nothing. The
headline differences (0.01 against 0.55, or 0.13 against 0.55) are far
outside that.

## A second, lower pass over tree cover (2026-11-17)

Question: every closed-loop miss was in a forest-edge world, five of six
under trees. Does a second pass, lower and at right angles to the first,
over the tree-covered part of the area, find them?

- `search --second-pass`: after the main lanes, the drone flies a second
  lawnmower over a rectangle around all tree canopies (the simplest
  land-cover map), at the lowest safe altitude (15 m here), with lanes at
  90 degrees to the first. Commit 30c8c68 plus the batch flag in 72e37a3.
- Flown on PX4 SITL on the five forest-edge sim-test worlds with the same
  detector (v4) and settings as the first closed-loop batch. 65.7 minutes.
  Table: `docs/results/closed_loop_v4_second_pass.md`.

| World | Altitude | Under trees | Found, one pass | Found, with second pass | False reports | False alarms | Flight time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test-003 | 40 m | yes | 0 of 2 | 1 of 2 | 0 -> 0 | 0 -> 2 | 155 -> 447 s |
| test-004 | 25 m | yes | 4 of 5 | 4 of 5 | 0 -> 0 | 0 -> 4 | 362 -> 798 s |
| test-007 | 25 m | no | 4 of 5 | 5 of 5 | 0 -> 2 | 0 -> 7 | 333 -> 1222 s |
| test-012 | 40 m | yes | 0 of 1 | 0 of 1 | 0 -> 0 | 0 -> 1 | 154 -> 265 s |
| test-014 | 25 m | yes | 1 of 2 | 2 of 2 | 0 -> 0 | 3 -> 5 | 436 -> 789 s |
| **Total** | | | **9 of 15** | **12 of 15** | **0 -> 2** | **3 -> 19** | **1440 -> 3521 s** |

- **It found three more people (9 to 12 of 15).**
- **It reported two people who were not there**, both in test-007. Before,
  the closed loop had made no false report in 15 worlds. Looking at the
  log, both reports are the SAME spot (0.4 m apart), 76 s apart: one
  phantom, reported twice. See the bug below.
- **False alarms rose from 3 to 19**, each one a descent and an orbit, and
  total flight time grew 2.4 times.

Why: from 15 m the camera sees trees, trunks and canopy shadows large and
close, and the detector, trained on people seen from 15 to 40 m over open
ground, fires on them far more often. Most of those tracks are rejected by
the confirmation orbit, but not all.

So the second pass trades precision and time for recall. For search and
rescue a missed person is usually worse than a wasted detour, so the trade
can be right, but two false reports in five worlds is not acceptable as
is. The obvious next step is a stricter confirmation (more frames, or a
higher confidence cut-off) during the low pass only. Not done.

How far to trust this: five worlds, one flight each with and without, so
15 people in all. The closed-loop flights themselves vary from run to run
(detector timing, PX4 noise); the first batch was not repeated. Read the
totals as a direction, not a measurement.

### Bug found by the second pass: one spot reported twice

In test-007 a phantom was reported as target_7 and again as target_8 at
the same position. Targets without identity are told apart by where they
were first sighted. The second sighting, from far away, was placed more
than 4 m from the first report, so it counted as a new target; its orbit
then converged on the same spot, and it was reported again.

Fix: a confirmation whose final position is within 4 m of an existing
report is not reported again; the target is remembered where it was
sighted, so it is not re-confirmed either. Counted as
`duplicates_avoided` in the mission metrics.

The test for it first passed without ever reaching the duplicate case
(the scenario never produced a second sighting far enough away), and then
a second time it switched modes during takeoff. It now forces the case and
was checked to fail with the fix switched off.

Re-flown on PX4 (test-007, second pass, detector v4, commit 3263c54):

| test-007 | Found | False reports | False alarms | Duplicates avoided | Flight time |
| --- | --- | --- | --- | --- | --- |
| second pass, before the fix | 5 of 5 | 2 (one spot twice) | 7 | (not counted) | 1222 s |
| second pass, with the fix | 4 of 5 | 0 | 5 | 2 | 914 s |

The duplicate reports are gone; twice the mission recognised a spot it
had already reported. It found one person fewer this time; with one
flight each, that is run-to-run variation in a closed loop (the missed
person, person_3, was in view and not held long enough), not an effect of
the fix, which only acts after a confirmation.

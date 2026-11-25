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

### Result: `configs/pilot10.yaml`, PX4 SITL, oracle detector, commit 1fd6d6a

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
  line says `1fd6d6a`, `dirty: no`.
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

### Result (pilot batch re-recorded at commit 987224d, plus one earlier search)

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

### Auto-labeller against ground truth (commit e200c3c, 4,105 unique frames, 6,078 people)

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
started at commit 241a85d.

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

### The v1 dataset (commit 241a85d)

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
  Ultralytics 8.4.214, torch 2.15.0, commit f874f72. 150 minutes.
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
- Same network, epochs and settings as v1. 149.8 minutes, commit 590d470.

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
  d3c4e34.

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

- (Superseded 2026-11-22: the live run is below.)
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
  5a4a877 in 2.5 h. 30 of 30 completed; one (dr-023) needed its retry
  because PX4 was not ready within 60 s. 90 of 90 people reported (oracle).
  14,241 frames, of which 13,760 after de-duplication (`train-v2-dr`).
- Export `v2-6k`, hash `61b514bcb1580625`: 6,000 of those frames by
  episode. Same network, epochs and settings as v1. 177.8 minutes, commit
  99e8568. Validation and test sets are the v1 ones (plain worlds).

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
  commit 5e48845.

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
  was not committed yet when it started; the code is commit 5e48845.

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
57e5bc4. Results in `docs/results/seed_runs.csv`.

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
  90 degrees to the first. Commit 57e5bc4 plus the batch flag in bf5e936.
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

Re-flown on PX4 (test-007, second pass, detector v4, commit a239ba0):

| test-007 | Found | False reports | False alarms | Duplicates avoided | Flight time |
| --- | --- | --- | --- | --- | --- |
| second pass, before the fix | 5 of 5 | 2 (one spot twice) | 7 | (not counted) | 1222 s |
| second pass, with the fix | 4 of 5 | 0 | 5 | 2 | 914 s |

The duplicate reports are gone; twice the mission recognised a spot it
had already reported. It found one person fewer this time; with one
flight each, that is run-to-run variation in a closed loop (the missed
person, person_3, was in view and not held long enough), not an effect of
the fix, which only acts after a confirmation.

## Housekeeping: training recordings deleted (2026-11-17)

With Kartik's approval, the 90 training recordings were deleted to free
19 GB: the 60 `train-*` and 30 `dr-*` `.mcap` files. Their JSON sidecars
(people, reports, mission metrics) are kept, and their frames and labels
remain in `data/frames` and the `sar-frames` dataset, which is all that
training and evaluation use. Kept: the 15 `test-*` recordings, all
closed-loop flights, and the Week 1 and agent episodes.

Consequences:

- Those flights no longer appear in the `sar-episodes` viewer (`make view`
  drops samples whose file is gone).
- `make batch` treats a missing file as not done, so running it again on
  `configs/v1.yaml` or `configs/v2.yaml` would re-record them (about 6 and
  2.5 hours). They would be flown from the same seeds, but would not be
  bit-identical to the originals.
- `make frames` would not re-extract them, since their frames are already
  in the dataset.

## Precision-recall curves (2026-11-17)

Phase 3B asked for PR curves; until now only the summary numbers existed.
`python -m sardrone.train.curves` draws them from the predictions stored
by `make evaluate` (`docs/images/pr_curves.png`). The area under each curve
reproduces the scorecard's mAP@0.5 to four decimals for all 14 curves,
which checks the plotting against the scoring.

What the curves show that the numbers do not:

| On real photos | Highest recall reached (cut-off 0.001) | Precision of its most confident boxes |
| --- | --- | --- |
| v0, untrained | 0.69 | about 0.99 |
| v4 | 0.80 | about 0.77 |
| v6 | 0.82 | about 0.87 |

- **The untrained network is more precise; the trained ones find more
  people.** The curves cross at about 0.55 recall. Below that, v0 is ahead;
  above it, v4 and v6 are, and v0 cannot reach past 0.69 at any cut-off.
  mAP averages over the whole curve and so hides that the trained models
  win where search and rescue cares most.
- **v4's most confident boxes on real images are wrong about a quarter of
  the time.** Its curve starts at precision 0.77, not near 1. Something in
  real scenes looks more like a simulated person to it than real people do.
  Not investigated.
- **On simulated worlds every trained version looks alike** (the curves
  overlap up to recall 0.9); the scorecard's differences there are small.

One training run per version; given the seed spread (0.054 mAP for v4),
the exact crossing point and maximum recalls will move between runs.

## False alarms: what fools the detector (2026-11-17)

`python -m sardrone.fo.false_alarms --version v4` marks every box at or
above the model's cut-off that matches no true person (IoU below 0.5),
using the scorecard's matching.

**Simulated frames.** The Phase 2E view `sar-false-alarms`, empty until
now, holds the frames where v4 made one: 114 of 2,861 validation frames
(159 boxes) and 225 of 7,453 sim-test frames (241 boxes).

**Real photos.** The 24 most confident false alarms of each model, cut out
(`docs/images/v4_false_alarms_real.jpg`, `v6_false_alarms_real.jpg`):

- **v4: all 24 are one red car parked on grass**, in different frames,
  at confidence 0.89 to 0.91. That is why v4's precision-recall curve
  starts at 0.77: its most confident boxes on real images are that car.
- **v6: the red car is gone from the top; silver and navy cars took its
  place**, at 0.82 to 0.91.

Reading: v4 trained on worlds where the only bright, compact thing on
grass is a person, so "red blob on grass" became a person. v6's worlds
had 21 clothing colours (greys and navy among them) and coloured boxes
lying around as distractors, which broke the link to red but not to
"car-sized coloured blob on grass". Neither the simulated worlds nor the
300 real images showed a car. (The real training pool is SARD frames 1 to
621; these car scenes are frames 924 to 1,091, in the test half. Whether
the pool shows any car was not checked.)

This is the data loop's "review the real images that still fail" step,
and it names the next piece of data: vehicles and other large man-made
objects as unlabelled distractors in the worlds, or real images that
contain them.

Counts are not compared across models here: v4 and v6 have different
cut-offs (0.378 and 0.140), so v6 draws far more boxes in total.

## Vehicles as distractors (2026-11-17)

Following the false-alarm review: a matrix row's `vehicles: N` places N
parked cars in the world, built from boxes (body, dark cabin, roof, four
wheels), 3.8 to 4.6 m long, in common car colours, at least 4 m from any
person and clear of trees and the takeoff point. No label plugin, so the
label sensor never boxes them: in training they are background. Vehicles
use their own random stream, so a row without them is unchanged; all 105
recorded worlds were regenerated and compared (identical apart from the
`scale` field added to people after the v1 batch, always 1.0 there).
Rendered once to check they read as cars from the drone's height.

`configs/v7.yaml`: 30 randomized episodes with 1 to 3 vehicles each. Plan
for v7: v6's recipe (randomized worlds, camera realism, the same 300 real
images) on these worlds, then check whether cars still top the list of
confident mistakes on real photos.

## The Cohere adapter, checked offline (2026-11-17)

Before spending trial-key calls, `tests/test_agent_cohere.py` checks the
adapter against the installed SDK (cohere 7.3.1) without any network call:

- A whole mission (seven tool calls) is run through the adapter with a
  stand-in client. Every request is checked against the SDK's own types:
  each message as `SystemChatMessageV2`, `UserChatMessageV2`,
  `AssistantChatMessageV2` or `ToolChatMessageV2`, each tool call as
  `ToolCallV2` with JSON-string arguments, each tool as `ToolV2`.
- Responses are built with `V2ChatResponse` (what `ClientV2.chat` returns,
  per its return annotation) and parsed back into the right tool name,
  arguments, text and token counts.
- The text-answer path (the loop's one reminder) gives a valid
  conversation too.
- The SDK's models do reject malformed messages (missing `tool_call_id`,
  wrong `type`, arguments as a dict instead of a string, wrong role), so
  the validation means something.

What this still cannot show: that the live API accepts the request, that
a trial key allows tool use, or how well a Cohere model does the missions.

## v6 repeated with two more seeds (2026-11-17)

The v6 export was rebuilt from scratch first (its export had been deleted
to free disk); the hash matched the original, `1acc17da393128c2`, so the
same images and labels were used. Commit afd87ca and earlier; results in
`docs/results/seed_runs.csv`.

| Recipe | Seed 51 | Seed 52 | Seed 53 | Mean | Spread |
| --- | --- | --- | --- | --- | --- |
| v4 real (SARD) mAP@0.5 | 0.547 | 0.586 | 0.532 | 0.555 | 0.053 |
| v6 real (SARD) mAP@0.5 | 0.597 | 0.526 | 0.587 | 0.570 | 0.072 |
| v4 sim-test mAP@0.5 | 0.934 | 0.937 | 0.934 | 0.935 | 0.003 |
| v6 sim-test mAP@0.5 | 0.926 | 0.925 | 0.930 | 0.927 | 0.005 |
| v0, untrained, real | 0.591 | | | | |

- **No measurable effect of randomization on real images, with or without
  real data.** v6's mean is 0.015 above v4's; each recipe alone spreads
  over 0.05 to 0.07. Three runs each cannot tell them apart. The first v6
  run (0.597) was simply a lucky seed. The headline "randomization helped
  once real data was present" is withdrawn, not just weakened.
- **Randomization costs a little in simulation, consistently.** v6's three
  simulated scores (0.925 to 0.930) are all below v4's (0.934 to 0.937).
  Small, but the ranges do not overlap.
- **Neither recipe beats the untrained network on real images.** Five of
  the six runs are below v0's 0.591; one v6 run is 0.006 above it.
- **v6 trades precision for recall**, at the cut-offs chosen on simulated
  validation data: SARD recall 0.64 to 0.70 against v4's 0.57 to 0.64,
  precision 0.51 to 0.58 against 0.70.

What this leaves as the project's findings, all of which hold across
seeds: simulation-only training collapses on real images; a few hundred
real images restore most of it; nothing tried on the simulation side
(randomization, camera realism) measurably helps real images; and no
recipe here beats the untrained network on real images by mAP.

## Strict second pass flown (2026-11-17)

The five forest-edge worlds again, detector v4, with `--strict-second-pass`
(during the low pass a track needs 6 sightings before a detour, and the
orbit must hold the target 20 frames) and the duplicate-report fix. 47.8
minutes. Table: `docs/results/closed_loop_v4_strict_second_pass.md`.

| Five forest worlds, v4 | People found | False reports | False alarms | Total flight time |
| --- | --- | --- | --- | --- |
| One pass | 9 of 15 | 0 | 3 | 1,440 s |
| Second pass | 12 of 15 | 2 (one spot twice) | 19 | 3,521 s |
| Strict second pass | 10 of 15 | 0 | 8 | 2,548 s |

Per world (found): test-003 0 / 1 / 1, test-004 4 / 4 / 4, test-007
4 / 5 / 4, test-012 0 / 0 / 0, test-014 1 / 2 / 1 (of 2, 5, 5, 1, 2).

- **Strictness worked on what it targets:** no false report, false alarms
  down from 19 to 8, flight time down by a quarter.
- **It also gave back two of the three extra finds.** In test-007 and
  test-014 the normal second pass found a person the strict one did not.
  A real person under trees is itself seen only intermittently, so "6
  sightings in a row" filters some of them out along with the ghosts.
- **The trade-off is the usual one, now measured three ways.** No setting
  here is free: more finds come with more detours.

The second pass's two false reports were one phantom reported twice,
before the duplicate fix. With the fix, re-flying that world gave none, so
the normal second pass would probably have shown 0 or 1, not 2.

How far to trust this: one flight per world per setting, 15 people. In
test-007 the normal second pass found 5 of 5 once and 4 of 5 on a re-fly,
so a single person's difference between settings is within run-to-run
variation. The direction (strict fewer false alarms, fewer finds) is
consistent across worlds; the size is not established.

## v7: randomized worlds with parked vehicles (2026-11-17)

- `configs/v7.yaml`: 30 randomized episodes, 1 to 3 box-built cars each,
  recorded at commit ed1ef5e; 30 of 30 completed (veh-020 failed once,
  PX4 not ready within 60 s, and passed on its retry); the oracle reported
  88 of the 90 people placed. (Corrected 2026-11-18: this entry first said
  commit 0cceff3, which is v7's training commit, and 90 of 90.)
- Export `v7-6k`, hash `e411d1520339f4fb`: 6,000 frames from those flights,
  camera realism, plus the same 300 real images as v4 and v6. Same
  settings. 164.6 minutes. Recorded as `dirty: True` (documentation and
  tests uncommitted at the start; training code as in 0cceff3).

| Recipe | Real (SARD) mAP@0.5 | Sim-test mAP@0.5 |
| --- | --- | --- |
| v4, three runs | 0.532 / 0.547 / 0.586 | 0.934 / 0.937 / 0.934 |
| v6, three runs | 0.597 / 0.526 / 0.587 | 0.926 / 0.925 / 0.930 |
| v7, one run | 0.498 (0.480 to 0.519) | 0.933 (0.928 to 0.938) |

**v7 is below all six earlier v4 and v6 runs on real images.** One run,
so the size is not established; but vehicles in the simulated worlds did
not help, and may have hurt.

What it did to the mistakes it was built to fix (24 most confident false
alarms on the real test images, counted by eye from the contact sheets):

| | Vehicles among the top 24 | Confidence range of the top 24 |
| --- | --- | --- |
| v4 | 24 (one red car) | 0.89 to 0.91 |
| v6 | 24 (silver and navy cars) | 0.82 to 0.91 |
| v7 | 20 (silver cars, white off-road vehicles) | 0.77 to 0.82 |

- **Cars are still its most confident mistakes.** Seeing box-built cars
  as background lowered how sure it is about real cars, by about 0.1, but
  did not stop it calling them people. A box with a dark cabin is not
  enough of a car to teach that.
- **Two of the 24 are real people it localised badly.** A person lying with
  legs outstretched: v7's box covers the upper body (102 px wide), the true
  box the whole body (182 px); IoU 0.44, just under the 0.5 needed. So
  that is a box-size error on lying people, not a hallucination, and the
  person also counts as missed.
- The other two are a birch log and a signpost.

(A first, automatic count of "car frames" gave 6 of 24 for v7. It used
a hand-made list of frame numbers that turned out to be incomplete; the
count above is by eye from `docs/images/v7_false_alarms_real.jpg`.)

So the false-alarm review named the right problem, and the first fix for
it was too crude. Real vehicle images, or proper vehicle meshes, would be
the next step; neither was tried.

## HERIDAL: a second real test set nobody trained on (2026-11-17)

SARD's 300 training images and its test images come from the same footage,
so v4's real score is flattered. HERIDAL (Aalto's CC BY 3.0 copy, see
DETAILS.md) is different footage: 101 test images, 4000 x 3000, 337
people. Nothing from it was trained on. Same models as the scorecard
(seed 51 runs), same inference size (1280), each scored at its own
scorecard cut-off. `docs/results/extra_real.csv`.

| Model | HERIDAL mAP@0.5 (95 % interval) | Max recall |
| --- | --- | --- |
| v0, untrained | 0.184 (0.118 to 0.258) | 0.38 |
| v1, simulation only | 0.030 (0.013 to 0.067) | 0.16 |
| v4, + 300 SARD images | 0.231 (0.154 to 0.324) | 0.57 |
| v6, v4 + randomized worlds | 0.262 (0.179 to 0.358) | 0.60 |
| v7, v6 + box-built cars | 0.213 (0.142 to 0.302) | 0.55 |

- **Simulation-only training collapses here too** (v1: 0.030). That
  result now holds on two real datasets.
- **Mixing in 300 SARD images helps beyond SARD.** v4, v6 and v7 all score
  above the untrained v0, but the intervals overlap (101 images is few), so
  by mAP alone that is not established. Max recall is clearer: v0 cannot
  find more than 38 % of the people at any cut-off; v4 and v6 reach 57 %
  and 60 %. Same pattern as on SARD.
- **Everything scores far lower than on SARD.** People are small here:
  median box 56 x 53 px in a 4000 px image, about 18 x 17 px once the image
  is shrunk to 1280 for the model. Inference on tiles at full resolution
  would probably help every model; not tried.
- v7 again sits below v4 and v6, inside their intervals.

## v8: v4's recipe on all the simulated frames (2026-11-18)

Question from the report's "next" list: does more simulated data move the
real score at all? v8 is v4 exactly (camera realism, the same 300 real
SARD images, 10 epochs, 1280, seed 51) but on all 24,667 simulated
training frames instead of a 6,000-frame sample. Export `v8-full`, hash
`c09b4aca41099b26`, 24,967 images. Commit af1a931, clean tree. 576.6
minutes of training on the laptop's GPU (MPS).

| | Sim-test mAP@0.5 | Real SARD mAP@0.5 | SARD max recall | HERIDAL mAP@0.5 |
| --- | --- | --- | --- | --- |
| v4 (6,000 sim + 300 real) | 0.934 (0.931 to 0.941) | 0.547; three runs 0.532 to 0.586 | 0.80 | 0.231 (0.154 to 0.324) |
| v8 (24,667 sim + 300 real) | 0.946 (0.942 to 0.948) | 0.483 (0.465 to 0.507) | 0.75 | 0.261 (0.174 to 0.355) |

- **Four times the simulated data made it better at simulation and worse
  at real SARD photos.** 0.483 is below all three v4 runs. One run; but
  v4's lowest was 0.532, and v8's upper bound is 0.507.
- **It is more cautious on real photos:** precision 0.73 and recall 0.40
  at its cut-off (v4: 0.70 and 0.58), and it never finds more than 75 % of
  the SARD people at any cut-off (v4: 80 %).
- **This is confounded, and I cannot separate the two causes.** The 300
  real images were 4.8 % of v4's training set and are 1.2 % of v8's. So
  "more simulation hurts" and "a smaller real share hurts" predict the
  same thing. Keeping the share at 5 % would need about 1,250 real images,
  and the SARD training pool has only 594. Using all 594 (2.4 %), or
  repeating the 300 to restore 5 %, would test it; neither was run.
- On HERIDAL v8 scores 0.261, level with v6 and above v4, with intervals
  far too wide (101 images) to call that a difference.

So, more simulated frames of the same narrow worlds did not help the real
score. It is the same lesson as v2 and v3, from a different direction.

## Phase 4A: the mission agent with a real model (2026-11-22)

- Trial key in `.env` (git-ignored). The key can use 17 chat models
  (`ClientV2.models.list(endpoint="chat")`); `command-a-03-2025` was used,
  as configured. A newer `command-a-plus-05-2026` is available and was not
  tried.
- First, one mission (m01) as a live check of the adapter: passed, 10
  valid tool calls, so the offline-tested request and response handling
  holds against the real API.
- The first full run stopped with a 429: the trial key allows 20 calls a
  minute, and 3.2 s spacing left no margin. Fixed at commit facb659:
  3.5 s spacing, and a 429 is retried with the identical request after a
  wait (tested; the count is recorded). Nothing from the stopped run is
  used.
- Full run at facb659, clean tree: `python -m sardrone.agent.evaluate
  --model rule-based --model cohere`.

| Model | Passed | Valid tool calls | Mean steps | Tokens in / out | Mean latency |
| --- | --- | --- | --- | --- | --- |
| rule-based | 9 of 20 | 100 % | 10.5 | 0 / 0 | 0 s |
| cohere:command-a-03-2025 | 17 of 20 | 100 % | 10.5 | 587,320 / 12,351 | 38.3 s |

212 model calls, one 429 retried.

- Of the eleven missions the keyword rules fail (m04 to m08, m11, m12,
  m14, m15, m17, m20), it passed nine; m11 and m17 are two of its three
  failures. Of the nine the rules pass, it failed one (m02). m20 ("do not
  launch") took one status call and stopped.
- It failed m02, m11 and m17, all on `finished_every_search` only. Each
  asks for specific people; it reported exactly them and went home with
  the area unfinished. Arguably sensible, but the scorer was fixed before
  the run and requires full coverage, so these stay failures.
- One run per mission. Temperature 0 and a fixed seed, but Cohere does not
  promise identical output, so repeat runs would show how stable 17 is.

## History rewritten before going public (2026-11-22)

Two committed manifests carried a home-folder path inside a PX4 error
message. Before making the repository public, the history was rewritten
to remove it (only those two files changed; the final tree is identical).
That gave 38 commits new IDs. Every commit ID cited in the docs and
results was updated to its new ID, and `docs/results/commit_map.csv` maps
each old ID to its new one, for commit messages and older files that
still quote an old ID.

A second rewrite the same day removed personal notes and drafts that were
never meant as project documents.
`commit_map.csv` maps IDs from both earlier histories to the current one.

## Tiled detection, issue #1 (2026-11-23)

People in HERIDAL's 4000 px photos shrink to about 18 px at the model's
1280 px input. `sardrone/train/tiled.py` also runs the model on 1280 px
tiles at full resolution.

- Method fixed before any full run: 1280 px tiles overlapping by 256 px
  (the median person is 56 px, so each is whole in some tile), plus the
  whole image at 1280 as before; boxes touching an inner cut line dropped;
  NMS at IoU 0.5; top 300. Person boxes only. Each model keeps its
  scorecard cut-off; nothing tuned on the test sets.
- Two bugs caught before the runs, both now covered by tests: the
  untrained COCO model's cars and dogs would have been labelled "person",
  and I had imported a commit helper that did not exist under that name.
- A smoke test on 3 HERIDAL photos looked worse for tiling (0.55 against
  0.64). Three images say nothing; the full runs decide.
- Full runs: six models, HERIDAL and SARD, predictions at 5e564cd. The SARD
  rows record `dirty: true` only because the HERIDAL half had just written
  `tiled.csv`; the code was unchanged.
- Overlapping intervals cannot settle a comparison on the same images, so
  `metrics.paired_difference` resamples the same images for both methods.

| | HERIDAL gain (paired 95 %) | SARD gain (paired 95 %) |
| --- | --- | --- |
| v0 | +0.126 (+0.075 to +0.179) | +0.076 (+0.069 to +0.083) |
| v1 | +0.051 (+0.025 to +0.085) | +0.001 (-0.002 to +0.007) |
| v4 | +0.120 (+0.067 to +0.175) | +0.024 (+0.013 to +0.034) |
| v6 | +0.102 (+0.047 to +0.161) | -0.006 (-0.016 to +0.005) |
| v7 | +0.141 (+0.083 to +0.205) | +0.043 (+0.033 to +0.053) |
| v8 | +0.086 (+0.036 to +0.136) | +0.030 (+0.021 to +0.038) |

- Best real-photo scores in the project so far: 0.364 on HERIDAL (v6,
  tiled) and 0.666 on SARD (untrained v0, tiled).
- What it does not change: on SARD the untrained model is still the best,
  tiled or not.
- The cost: more confident false alarms at the old cut-offs (v6 on
  HERIDAL, precision 0.31 to 0.10). For a photo-triage tool that means
  more photos to look at; a tiled cut-off is the next step.

## Cut-offs for tiled detection, chosen on real photos (2026-11-23)

The scorecard cut-offs were chosen on simulated validation frames, for
untiled detection. For a fair tiled-versus-untiled comparison both need a
cut-off chosen on the same real photos.

- Validation: the 294 SARD pool photos (767 people) that no model trained
  on: the pool minus the 300 drawn by `export.py --add-real 300` with seed
  51 (v4, v6, v7, v8 all used that draw). Re-drawn with the same call; the
  original export lists were deleted in the disk cleanup, so the match is
  assumed, not checked. No overlap with the SARD test set (checked).
- Best-F1 cut-off per model and method; the test sets only apply it.
  `docs/results/tiled_cutoffs.csv`, commit 08bf6e7, clean tree.

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

- SARD: tiled and untiled perform alike at their own cut-offs.
- HERIDAL: tiling roughly doubles recall (v6 0.29 to 0.55, v4 0.33 to 0.57)
  and lowers precision (v6 0.49 to 0.30). For search and rescue, where a
  miss costs more than a false alarm, usually the right trade; but it is
  a trade.
- Side finding from the sim2real-gap plugin (issue #11): SARD and HERIDAL,
  both real drone photos, are 98 % separable by a linear classifier on CLIP
  embeddings, nearly as much as simulated against real (100 %). So a
  cut-off chosen on one real dataset need not suit another, and
  separability measures "looks different", not "is simulated".

Also: rebuilding v1's training export from its recorded settings reproduced
its hash, d741bdbb19fbe45d, exactly.

## Agent: twelve harder missions; the Cohere trial limit (2026-11-23)

- `sardrone/agent/missions_hard.yaml`: 12 new missions, written after the
  first 20 were scored and fixed before any model ran: a correction
  ("actually, make that the west woods"), two conditionals (one where the
  person is found in the first area, so no second search is needed),
  typos, numbers in words, "as low as you're allowed", an exclusion
  ("not the person in the orange vest"), standing by without launching,
  and a battery limit that cuts a two-area search short. Same checks; the
  reference policy passes all 12, so each is solvable.
- Keyword rules: 1 of 12 (`docs/results/agent_eval_hard.md`), against 9 of
  20 on the first set. The new set does what it was written for.
- No model scores yet. The planned run (command-a-03 on all 32 missions,
  command-a-plus-05-2026 on the first 20, about 550 calls) stopped on its
  first call: the trial key's monthly limit of 1,000 calls was already
  used. My estimate that about 400 had been used was wrong. The scores
  wait for the limit to reset.
- The retry logic treated that 429 like the per-minute limit and retried
  for about 7 minutes. Now a monthly-limit 429 stops at once with a clear
  error (tested).

## The agent flies the simulated drone (issue #8, 2026-11-23)

`sardrone/agent/flight.py` gives the agent's six tools a real (simulated)
flight behind the mission simulator's interface, so the same agent loop
flies PX4 in Gazebo and the flight is recorded as an MCAP episode with every
decision on /events.

- Areas are halves of the world square. The search runs while the model
  thinks; get_status waits for about 20 % more coverage, a new sighting,
  the end of the area or low battery. A pilot task always streams setpoints,
  because PX4 leaves offboard mode without them. If the agent never comes
  home, a safety return lands the drone.
- Confirmed descriptions (clothing, pose) come from the world file entry of
  the nearest person to the orbit's estimate. The detector finds people but
  recognises neither; that is a stand-in for an attribute model.
- Bugs found on the way: with the fake drone, the search task never yielded
  to the event loop, so the agent's first call never returned (fixed: every
  wait yields); and a dry run reported a south-field person on a
  north-field mission, because the pitched camera sees beyond the searched
  area (fixed: only sightings inside the area, 2 m margin, are queued).

Three flights in PX4 + Gazebo, rule-based agent (keyword rules), oracle
detector (the GPU was busy training), commit 3fc77af. Recordings in
`data/agent_flights/` (not committed).

| World | Mission | Right answer | Reported | |
| --- | --- | --- | --- | --- |
| test-002 | "Search the north field and report anyone you find." | person_1, 2, 3 | person_1, 2, 3 | right |
| test-010 | "Search the east meadow and report anyone." | person_2 | person_2 | right |
| test-006 | "Search the west woods. Report anyone sitting." | person_4 | person_1, 3, 4 | wrong |

No false reports, no rejected tool calls, every flight home and landed, 2 to
4 minutes each. The third is the expected failure: keyword rules ignore
"sitting" and report everyone in the area. It is the mission to fly with a
language model once the Cohere key's monthly limit resets.

Limit found: PX4's simulated battery stops draining at 50 % by default
(`SIM_BAT_MIN_PCT`), and every flight ended at exactly 50.0 %. Battery
limits below 50 % cannot trigger in these flights without changing it.

## Second pass with a confidence floor: no better (issue #5, 2026-11-24)

Rule: during the low pass only, act on a sighting only at confidence 0.41
or above. 0.41 was chosen on validation frames, not the test worlds: it is
where v4's precision reaches 90 % on the 787 validation frames under tree
cover (at the normal cut-off, 0.378, it is 88 %). Five forest_edge test
worlds, v4, commit 2f9b9a7, one flight each (test-004 needed its retry: PX4
not ready). Table: `docs/results/closed_loop_v4_floor_second_pass.md`.

| Low-pass rule | Found | False reports | False alarms | Flight time |
| --- | --- | --- | --- | --- |
| no second pass | 9 / 15 | 0 | 3 | 1,440 s |
| second pass, normal rules | 12 / 15 | 2 | 19 | 3,521 s |
| strict: 6 sightings, 20 frames | 10 / 15 | 0 | 8 | 2,548 s |
| confidence floor 0.41 | 9 / 15 | 0 | 14 | 2,961 s |

- The floor found no more people than a single pass, at twice the flight
  time, and lost a person the other rules found (test-003, 0 of 2).
- My reading: people under trees are also seen at low confidence, so a
  confidence bar removes them along with the canopy ghosts. Persistence
  (the strict rule) separates them better than confidence does.
- One flight per world: a difference of one person may be noise. But the
  issue's bar (more than 10 of 15 with no false report) is not met; #5
  stays open. A next try would combine persistence with a wider orbit.

## Protecting the network in simulation-only training (issue #3, 2026-11-24)

Two variants of v1 (simulated frames only, export d741bdbb19fbe45d, which
rebuilt exactly from its recorded settings), one run each:

- v9: `--freeze 23`, everything but the detection layer frozen. Commit
  e21958c, clean, 96 minutes.
- v10: `--optimizer AdamW --lr0 0.0002`, 10x below the 0.002 Ultralytics
  chose automatically for v1 (the log confirms AdamW at 0.0002). 177
  minutes; recorded `dirty: true` only because v9's evaluation had just
  added its scorecard row.

| | Sim-test | Real SARD | HERIDAL | HERIDAL max recall |
| --- | --- | --- | --- | --- |
| v1, all layers | 0.947 | 0.014 | 0.030 | 0.16 |
| v10, 10x lower lr | 0.945 | 0.025 | 0.059 | 0.29 |
| v5, backbone frozen | 0.943 | 0.127 | | |
| v9, detection layer only | 0.912 | 0.435 (0.418 to 0.458) | 0.264 (0.184 to 0.360) | 0.67 |

- A lower learning rate does almost nothing. Freezing more of the network
  does a lot: most of the damage is in the neck.
- v9 is still below the untrained v0 on SARD (0.591) but, with no real
  images, scores level with v6 on HERIDAL and finds the most people there
  of any untiled model. One run, overlapping intervals.
- Next: v9's freezing with v4's 300 real images, over seeds.

## WiSARD: a third real dataset, colour and thermal (issue #9, 2026-11-24)

The one-flight sample (0.97 GB) downloaded after Google Drive refused it for
about a day ("too many downloads"). 264 colour images (4K) with 1,022
people, 264 thermal (640 x 512) with 1,006, labels YOLO-style, frames in
flight order but sampled (frame numbers skip), so it is a sequence of
stills, not video. Counts match the archive's own count.txt.

Scores (mAP@0.5, scorecard cut-offs, nothing tuned on WiSARD):

| | Colour | Colour tiled | Thermal |
| --- | --- | --- | --- |
| v0 | 0.125 | 0.314 | 0.150 |
| v4 | 0.100 | 0.138 | 0.136 |
| v6 | 0.097 | 0.148 | 0.044 |
| v9 | 0.076 | 0.232 | 0.045 |

- v4 and v6 (300 SARD photos) are below the untrained model here. Over
  three real datasets the SARD photos help on SARD, break even on HERIDAL
  and hurt on WiSARD.
- Tiling: +0.19 for v0, +0.16 for v9, small for v4 and v6.
- Thermal: the untrained colour model finds people about as well as on
  colour; the fine-tuned ones much worse. Relevant to the triage idea (#13).
- Bug caught: the tiled-gains step rewrote `tiled_gain.csv` with only the
  WiSARD rows, dropping HERIDAL's and SARD's. Restored from git; the file
  is now merged row by row (tested).
- Still open in #9: continuous real video, to test the tracker.

## Untangling v8: more simulation, or a smaller real share? (issue #2, 2026-11-25)

v8 trained v4's recipe on all 24,667 simulated frames. It scored 0.483 on
real SARD photos, below every v4 run (0.532 to 0.586 over three seeds). But
its 300 real images fell from 4.8 % to 1.2 % of training, so two causes were
mixed. Two follow-ups keep all the simulated frames and raise the real share
back:

| Version | Simulated frames | Real images | Real share | SARD mAP@0.5 (95 % interval) | HERIDAL mAP@0.5 (95 % interval) |
|---|---|---|---|---|---|
| v4 (3 seeds) | 6,000 | 300 | 4.8 % | 0.532 to 0.586 | 0.231 (0.154 to 0.324), seed 51 |
| v8 | 24,667 | 300 | 1.2 % | 0.483 (0.465 to 0.507) | 0.261 (0.174 to 0.355) |
| v11 | 24,667 | 594 (all SARD pool images) | 2.4 % | 0.506 (0.485 to 0.528) | 0.150 (0.094 to 0.221) |
| v12 | 24,667 | 300, each repeated 4x | 4.6 % | **0.544 (0.526 to 0.566)** | 0.215 (0.150 to 0.287) |

### What it says
- **On SARD, the real share explains v8's drop, not the extra simulation.**
  With all simulated frames and the real share back to 4.6 % (v12), the score
  returns to v4's range (0.532 to 0.586). The real share rises from v8 to v11
  to v12, and so does the score: 0.483 → 0.506 → 0.544.
- **Weight mattered more than variety here.** v12 (300 images, repeated) beat
  v11 (594 different images, at half the share). Their intervals only just
  touch (v11 up to 0.528, v12 from 0.526).
- **HERIDAL, which no model trained on, does not follow.** v8 (0.261) is no
  worse than v4 (0.231), and v11 is lowest (0.150). With 101 images the
  intervals overlap widely, so there is no clear order on HERIDAL. More
  weight on SARD photos tunes the model to SARD; it does not obviously help
  on other real data.

### How far to trust this
- **One training seed each for v11 and v12.** The issue planned three. We
  chose one, because v11 alone trained for 10.6 h, so two more seeds per
  recipe would be about 42 more GPU hours. v4's three seeds span 0.054, so a
  single run can land anywhere in a range that wide. "v12 is back in v4's
  range" is solid; "v12 beats v11" is not.
- **v12 was interrupted after epoch 6**, when the app session running the
  queue ended. It resumed from `last.pt` with its optimizer state
  (`run.json`: `"resumed": true`; 257 min after the resume). The queue now
  runs detached, so this cannot happen again.
- v12 repeats 300 images rather than adding new ones: more weight, not more
  variety.

## v13: rich worlds, and the first end-to-end pipeline run (issues #4 and #10, 2026-11-25)

v13 is v6's recipe (camera realism + 300 real SARD images) on the **rich
worlds** of `configs/v13.yaml`: real vehicle meshes instead of box-built
cars, 3 to 8 rock meshes, and two more bodies (walking adult, child). 5,999
frames from 30 episodes, sampled by episode, 10 epochs. It came out of one
command, from new episodes to the scorecard
([results/pipeline_v13.md](results/pipeline_v13.md)).

| Version | Simulated worlds | SARD mAP@0.5 (95 % interval) | HERIDAL mAP@0.5 |
|---|---|---|---|
| v4 (3 seeds) | plain | 0.532 to 0.586 | 0.231 (seed 51) |
| v6 (3 seeds) | randomized | 0.597 (seed 51); mean 0.570 | 0.262 |
| v7 | randomized + box-built cars | 0.498 (0.479 to 0.519) | 0.213 |
| **v13** | **rich: vehicle and rock meshes, more bodies** | **0.614 (0.599 to 0.632)** | 0.236 (0.165 to 0.315) |

- **The best single run on real SARD photos so far**, and the smallest
  sim-to-real gap (0.295). One seed: v4's and v6's seeds spread 0.05 to
  0.07, so "rich worlds beat randomized worlds" needs more seeds; "rich
  worlds are at least as good as anything before" is solid.
- **HERIDAL** does not separate the versions (101 images; the intervals
  overlap widely).
- **False alarms changed character.** Of the 24 most confident false alarms
  on real photos, counted by eye from `docs/images/v13_false_alarms_real.jpg`:
  - **4 vehicles** (v7: 20 of 24): a white Land Rover, a white car, and one
    dark blue shape beside a car, which looks like a car roof (2 frames);
  - **5 frames of one birch log**;
  - **15 real people**, whose predicted box does not match the labelled box
    closely enough (IoU < 0.5): mostly people lying down and groups.

  The vehicle meshes did what box-built cars could not. Most of the
  remaining top "false alarms" are box-size disagreements, not detections
  of nothing.
- **The image-level gap did not close.** The rich-world frames are still
  100 % separable from real photos in CLIP space, and slightly farther from
  their nearest real photo than the plain worlds (mean cosine distance
  0.313 vs 0.292; `docs/images/gap_map_v13_rich.png`,
  `docs/results/domain_gap_v13_rich.json`). Content that a detector can use
  is not the same as content that makes the image look real.

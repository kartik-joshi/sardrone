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

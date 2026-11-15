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

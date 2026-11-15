# sardrone

A search-and-rescue drone in simulation. Each flight is recorded as **one
`.mcap` file** (an "episode") plus a small JSON sidecar, and the episodes are
loaded into **FiftyOne's multimodal viewer**, where camera, GPS, pose and
events replay on a single timeline.

Week 1 scope: the recording and viewing pipeline. Detection comes later.

## Quick start

```bash
make test          # create .venv, install, run the test suite (no simulator needed)
make synthetic     # write a 10 s fake episode  -> data/episodes/synthetic-0001.mcap
make record-fake   # fly the real mission with a fake drone -> data/episodes/fake-0001.mcap
make view          # load every episode into FiftyOne and open http://localhost:5151
```

`make view` installs FiftyOne on first use (about 900 MB). To record from
PX4, start the simulator in one terminal and record from another:

```bash
make sim           # PX4 SITL + Gazebo with a forward camera, headless; leave it running
```

```bash
make record        # arm, take off to 10 m, fly a 20 m square, land -> flight-NNNN.mcap
```

`make record` takes its images from the simulated camera. For the plain
drone without a camera use `make sim SIM_MODEL=gz_x500` and
`make record CAMERA=synthetic`.

`make sim` expects PX4 in `../PX4-Autopilot`; override with
`make sim PX4_DIR=/path/to/PX4-Autopilot`. See [PX4 setup](#px4-setup).

## What is verified, and what is not

| Part | Status |
| --- | --- |
| Time and frame conversions, `EpisodeWriter`, synthetic episode | Tested (`make test`), files read back with an independent MCAP reader |
| Mission logic in `record_flight.py` | Tested end to end with `FakeDrone` on simulated time |
| FiftyOne dataset + App | Tested; viewer checked by hand: 3D, camera, map and log tiles replay with no warnings |
| `MavsdkDrone` against a running PX4 | Flown on PX4 SITL v1.18 + Gazebo Harmonic (`gz_x500`, headless), from a cold start. Not part of `make test`, which must run without a simulator |
| Gazebo camera | Byte protocol and `GazeboCamera` tested without a simulator (stand-in bridge). Flown once on `gz_x500_mono_cam`, headless: 591 frames at 1280x960, checked in FiftyOne |
| Person detection | Not implemented. The `person_seen` event is still scripted from the drone's position; the default Gazebo world has no person in it |
| Gazebo window (GUI) | Not tried: it needs XQuartz, which needs an admin password to install |

Numbers from the recorded real flight (`flight-0001`, 61.6 s): cruise height
9.6 to 10.1 m, top ground speed 3.1 m/s, square corners within the 1 m
tolerance, touchdown 0.5 m from the takeoff point, samples every 0.10 s on
PX4's own clock, battery reported on a 0 to 100 scale.

## Layout

| Path | Purpose |
| --- | --- |
| `sardrone/mcap/timeutil.py` | Integer-nanosecond time, `ns_to_ts`, `safe_ts`, `EpisodeClock` |
| `sardrone/mcap/frames.py` | NED/FRD (PX4) to ENU/FLU (Z-up) for positions and quaternions |
| `sardrone/mcap/geo.py` | Metres from home to latitude/longitude and back |
| `sardrone/mcap/episode_writer.py` | `EpisodeWriter`: channels, schemas, sidecar |
| `sardrone/sim/camera.py` | `CameraSource` interface, `SyntheticCamera`, `GazeboCamera` |
| `sardrone/sim/gz_image_bridge.py` | Helper run by Gazebo's own Python; pipes camera frames to `GazeboCamera` |
| `sardrone/sim/drone.py` | `Drone` interface, `DroneState`, `FakeDrone` |
| `sardrone/sim/mavsdk_drone.py` | `MavsdkDrone`: the PX4 implementation |
| `sardrone/sim/record_flight.py` | The mission and the 10 Hz recorder |
| `sardrone/sim/synthetic.py` | Fake 10 s circle episode |
| `sardrone/fo/load.py` | Builds the `sar-episodes` dataset and launches the App |
| `data/episodes/` | `.mcap` + `.json` output (git-ignored) |

```
 PX4 SITL ──MAVSDK──> MavsdkDrone ┐
                                  ├─ Drone ──> FlightRecorder ──> EpisodeWriter ──> flight-0001.mcap
             (tests)   FakeDrone ─┘               │   ▲                │             flight-0001.json
                                     CameraSource ┘   │                │
                                  EpisodeClock (one clock)   NED→ENU happens here only
                                                                       │
                                                         fo/load.py ──> FiftyOne "sar-episodes"
```

## Topics in an episode

| Topic | Schema | Encoding | Notes |
| --- | --- | --- | --- |
| `/camera/front` | `foxglove.CompressedImage` | protobuf | JPEG, 10 Hz |
| `/camera/front/calibration` | `foxglove.CameraCalibration` | protobuf | once; pinhole, no distortion |
| `/gps` | `foxglove.LocationFix` | protobuf | latitude, longitude, altitude |
| `/pose` | `foxglove.PoseInFrame` | protobuf | drone in `map`, Z up |
| `/tf` | `foxglove.FrameTransform` | protobuf | `map → base_link` and `base_link → camera_front`, every sample |
| `/events` | `foxglove.Log` | protobuf | `takeoff`, `waypoint_reached`, `person_seen`, `land` |
| `/telemetry` | `sardrone.Telemetry` | JSON + JSON Schema | battery, flight mode, velocity (Z up) |

Every `foxglove.*` schema is protobuf, taken from the SDK's own
`msgs.X.get_schema()`. Only the custom `/telemetry` type is JSON, which
viewers accept because it is not in the `foxglove.` namespace. FiftyOne shows
it in the raw Message inspector rather than as a plot.

The sidecar (`flight-0001.json`) holds `world`, `weather`, `time_of_day`,
`mission_prompt`, `people_placed`, `people_found`, `success`, plus
`episode_id`, `start_time_ns`, `duration_s` and per-topic message counts. The
same data is embedded in the `.mcap` as a metadata record named `episode`.

## Conventions

**Time.** Everything is integer nanoseconds. `ns_to_ts()` splits into
`(sec, nsec)` only when a message is filled in. Each episode has one clock:
the recorder takes PX4's boot-relative timestamp from the first sample as the
anchor and maps every later sample with `start + (reading − anchor)`.
`safe_ts()` rejects anything before 2020, so a boot-relative or sim-relative
time can never be written as wall-clock time.

**Frames.** PX4 speaks NED (z down) with an FRD body. The episode stores ENU
(z up) with an FLU body. Callers pass PX4-native values to `EpisodeWriter`;
the conversion happens in exactly one place.

| Frame id | Meaning |
| --- | --- |
| `map` | ENU world, origin at PX4's local origin (roughly the takeoff point) |
| `base_link` | drone body, x forward, y left, z up |
| `camera_front` | optical frame: x right, y down, z into the image |

## PX4 setup

What was done on this Mac (Apple Silicon, macOS 26), following the PX4 macOS
guide:

```bash
git clone https://github.com/PX4/PX4-Autopilot.git ../PX4-Autopilot
```

```bash
cd ../PX4-Autopilot && git submodule update --init --recursive --force
```

```bash
./Tools/setup/macos.sh --sim-tools
```

```bash
brew install pkgconf
```

- The setup script installs build tools with Homebrew, a Python venv in
  `PX4-Autopilot/.venv`, and Gazebo Harmonic in
  `PX4-Autopilot/Tools/setup/macos/.pixi`.
- `pkgconf` is not installed by the script, but the build fails without it
  ("Could NOT find PkgConfig").
- The script's last step installs XQuartz and asks for your password. It is
  only needed for the Gazebo window; `make sim` runs headless without it.
  For the window: `brew install --cask xquartz`, log out and in, then
  `make sim HEADLESS=0`.

## What the first real flight taught

The first recording against PX4 completed, but the data showed three
problems that the fake drone could not reveal. Each is now handled in code:

| Seen in the data | Cause | Fix |
| --- | --- | --- |
| Cruise at 12 m, not 10 m | PX4's height estimate was still converging after boot (1.9 m on the ground) | `wait_until_ready()` also waits for the height to hold within 10 cm for 5 s |
| 10 m/s legs, 1 m overshoot | One far setpoint makes PX4 fly at its maximum speed | The setpoint is moved along each leg at 3 m/s (`_fly_leg`) |
| Landed 6 m from takeoff | Landing started while still moving | A corner counts as reached only below 0.5 m/s |

Known limits: if PX4 does not supply attitude timestamps the recorder falls
back to this computer's monotonic clock for the whole episode (still one
clock); and if a waypoint is not reached in 90 s the mission stops with a
timeout, commands a landing and saves the partial episode.

## The Gazebo camera

`make sim` starts `gz_x500_mono_cam`, the x500 with a forward camera
(1280x960, 99.7 degree horizontal field of view, 30 Hz). Other camera models
in the checkout are `x500_mono_cam_down`, `x500_depth` and `x500_gimbal`.

```
 Gazebo ──gz-transport──> gz_image_bridge.py ──pipe──> GazeboCamera ──JPEG──> EpisodeWriter
          (Gazebo's Python 3.13)                (project venv, reader thread)
```

- **Two Pythons.** Gazebo's Python bindings exist only in the Python that
  ships with Gazebo and cannot be installed into this venv. So a small helper
  runs there and pipes raw frames over stdout: one JSON line with the lens,
  then a fixed header plus pixels per frame.
- **Newest frame only.** The bridge forwards at most 10 frames per second and
  the reader keeps one, so a slow recorder never logs a queue of old images.
  An image can be up to one render interval (1/30 s) older than its pose.
- **Fail loudly.** The camera is opened before the drone arms. No image within
  15 s, a bridge that exits, or no new frame for 3 s each raise an error
  instead of silently repeating a picture.
- **Calibration from the simulator.** Field of view comes from Gazebo's
  `camera_info`, and the mount position (12 cm forward, 3 cm left, 24 cm up)
  from the model's SDF; it is written into the `base_link → camera_front`
  transform.

Known limits:

- The default world is a flat grey plane, so the pictures show a horizon and
  the propellers. There is no person to see; `person_seen` is still scripted.
- Gazebo prints "Unable to load Ogre Plugin ... Rendering will not be
  possible" at startup when headless on this Mac. Images are rendered anyway.
  The cause was not found.
- `gz topic` and other Gazebo tools need `GZ_IP=127.0.0.1` on this setup,
  because PX4 runs Gazebo on the loopback interface.
- Start `make sim` from a real terminal. Without one, PX4's console prints
  its prompt endlessly.

`SyntheticCamera` remains for tests and for `make record-fake`.

## Decisions that differ from the original brief

- **`/tf` is a separate topic.** The brief put `PoseInFrame` and
  `FrameTransform` both on `/pose`; one topic with two schemas confuses
  viewers, so the transform goes on `/tf`.
- **The camera mount transform repeats with every pose.** Written once,
  FiftyOne flagged it as "Pose data is stale" a few seconds in.
- **MAVSDK 4 native API.** `pip install mavsdk` now gives v4, whose API is
  different from the gRPC package most tutorials show. The code targets v4.
- **FiftyOne's database lives in `.fiftyone/` inside this project**, and
  usage analytics are off. Export `FIFTYONE_DATABASE_DIR` yourself to use
  your normal `~/.fiftyone` instead. Launching the App still creates two tiny
  files outside the project, `~/.fiftyone/var/uid` and `welcome.json`;
  FiftyOne hard-codes that location.
- **`safe_ts()` also rejects dates after 2100**, which catches nanoseconds
  multiplied by 1000 by mistake.

## Interview talking points

- **Time:** "Each clock domain is explicit and anchored once per episode, and
  the writer validates at the boundary, so a boot-relative timestamp raises
  immediately instead of producing a 50-year timeline."
- **Frames:** "An attitude quaternion relates two frames, so converting NED to
  ENU means changing both the world and the body frame:
  `q_enu = Q_ned→enu · q · Q_frd→flu`. I test it by checking that rotating a
  vector in either convention lands on the same physical direction."
- **Schemas:** "I record with typed protobuf schemas from the SDK, so any
  Foxglove-compatible viewer can decode the file with no custom code, and my
  tests read the files back with a different MCAP library than the one that
  wrote them."
- **Sim-to-real gap:** "My fake drone passed every test, and the first real
  flight still cruised 2 m too high and overshot corners at 10 m/s. I found
  that by reading the recorded episode, which is the point of recording
  everything on one timeline."
- **Process boundary:** "The simulator's Python bindings could not be
  installed in my environment, so I put a 100-line bridge in the simulator's
  Python and defined a tiny binary protocol between them. The protocol is
  pure byte parsing, so it is unit-tested with in-memory streams and a
  stand-in bridge, with no simulator."
- **Testability:** "The mission is written against a five-method `Drone`
  interface. A fake drone with its own simulated clock lets the whole
  arm-takeoff-square-land sequence, including the timeout-and-land failure
  path, run in under a second in CI."
- **Data model:** "One sample per episode, with mission facts as filterable
  fields. That makes questions like 'show me failed foggy flights' a sidebar
  filter instead of a script."

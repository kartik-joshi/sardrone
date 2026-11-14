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
PX4, start the simulator in another terminal first:

```bash
cd /path/to/PX4-Autopilot && make px4_sitl gz_x500
```

```bash
make record        # arm, take off to 10 m, fly a 20 m square, land -> flight-0001.mcap
```

## What is verified, and what is not

| Part | Status |
| --- | --- |
| Time and frame conversions, `EpisodeWriter`, synthetic episode | Tested (`make test`), files read back with an independent MCAP reader |
| Mission logic in `record_flight.py` | Tested end to end with `FakeDrone` on simulated time |
| FiftyOne dataset + App | Tested; viewer checked by hand: 3D, camera, map and log tiles replay with no warnings |
| `MavsdkDrone` against a **running PX4** | **Not verified.** PX4 was not installed on the machine this was built on. The API calls were checked against the installed `mavsdk` 4.2.0 source, and the "no autopilot" error path was run, but no real flight has been recorded yet |
| Gazebo camera | Not implemented (`GazeboCamera` is a documented TODO); frames are synthetic |

First real flight checklist: see [Verifying against PX4](#verifying-against-px4).

## Layout

| Path | Purpose |
| --- | --- |
| `sardrone/mcap/timeutil.py` | Integer-nanosecond time, `ns_to_ts`, `safe_ts`, `EpisodeClock` |
| `sardrone/mcap/frames.py` | NED/FRD (PX4) to ENU/FLU (Z-up) for positions and quaternions |
| `sardrone/mcap/geo.py` | Metres from home to latitude/longitude and back |
| `sardrone/mcap/episode_writer.py` | `EpisodeWriter`: channels, schemas, sidecar |
| `sardrone/sim/camera.py` | `CameraSource` interface, `SyntheticCamera`, `GazeboCamera` TODO |
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

## Verifying against PX4

Nothing below has been run yet. On the first real flight, check:

1. `make record` connects. The default URL is `udpin://0.0.0.0:14540`; MAVSDK
   4.x prints a deprecation warning for the older `udp://:14540` spelling,
   which means the same thing. Override with
   `.venv/bin/python -m sardrone.sim.record_flight --url ...`.
2. The console prints `takeoff`, four `waypoint_reached`, and two `land` lines.
3. In `flight-0001.json`, `duration_s` is plausible (roughly 60 to 90 s) and
   the topic counts are about 10 per second.
4. In the viewer, `/telemetry` shows `battery_percent` near 100 at the start.
   If it reads near 1.0, MAVSDK is reporting a 0..1 fraction and
   `mavsdk_drone.py` needs a `* 100`.
5. In the 3D tile the drone climbs **up**, and the first leg goes **north**
   (toward +Y).

Known limits of the real path: if PX4 does not supply attitude timestamps the
recorder falls back to this computer's monotonic clock for the whole episode
(still one clock); and the mission stops with a timeout error, commands a
landing and saves the partial episode if a waypoint is not reached in 90 s.

## The camera TODO

`make px4_sitl gz_x500` has no camera. Per the PX4 Gazebo vehicle docs, the
x500 variants with one are `gz_x500_mono_cam` (forward), `gz_x500_mono_cam_down`
(downward), `gz_x500_depth` (forward depth) and `gz_x500_gimbal`. Confirm in
your checkout with:

```bash
ls PX4-Autopilot/Tools/simulation/gz/models | grep -i -E "cam|depth|gimbal"
```

`GazeboCamera` in `sardrone/sim/camera.py` lists the steps to implement it.
Until then, `SyntheticCamera` draws a scrolling ground grid and an orange
marker when the drone is within 3 m of a scripted "person" at the far corner
of the square. That marker, and the `person_seen` event it triggers, are
placeholders for a real detector.

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
- **Testability:** "The mission is written against a five-method `Drone`
  interface. A fake drone with its own simulated clock lets the whole
  arm-takeoff-square-land sequence, including the timeout-and-land failure
  path, run in under a second in CI."
- **Data model:** "One sample per episode, with mission facts as filterable
  fields. That makes questions like 'show me failed foggy flights' a sidebar
  filter instead of a script."

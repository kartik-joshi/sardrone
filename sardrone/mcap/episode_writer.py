"""EpisodeWriter: one simulated flight -> one .mcap file + one .json sidecar.

Checked against the INSTALLED foxglove-sdk (0.31.0) with help()/__doc__:

* message classes live in `foxglove.messages`; each has `.get_schema()`
  (returns a protobuf Schema) and `.encode()` (returns protobuf bytes)
* `foxglove.Channel(topic, schema=..., message_encoding=..., context=...)`
* `foxglove.open_mcap(path, allow_overwrite=..., context=...)` returns an
  MCAPWriter with `write_metadata()` and `close()`
* `CameraCalibration` takes uppercase D, K, R, P
* `Quaternion` is constructed as (x, y, z, w) keywords - note w is LAST there,
  while our own Quat tuple is (w, x, y, z). Always use keywords.

Design rules enforced here:

* every foxglove.* schema is written with PROTOBUF encoding (never JSON)
* callers pass PX4-native values (NED position, NED/FRD quaternion); the
  conversion to Z-up ENU happens in exactly one place, inside this class
* every timestamp is integer wall-clock nanoseconds and is range-checked
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import foxglove
from foxglove import messages as msgs

from sardrone.mcap.frames import Quat, ned_quat_to_enu, ned_to_enu, quat_from_rpy, quat_multiply
from sardrone.mcap.timeutil import NS_PER_SEC, safe_ts

# ---- topics -----------------------------------------------------------------

TOPIC_IMAGE = "/camera/front"
TOPIC_CALIBRATION = "/camera/front/calibration"
TOPIC_GPS = "/gps"
TOPIC_POSE = "/pose"
TOPIC_TF = "/tf"
TOPIC_TELEMETRY = "/telemetry"
TOPIC_EVENTS = "/events"
TOPIC_GROUND_TRUTH_PEOPLE = "/ground_truth/people"

# ---- coordinate frame names -------------------------------------------------

FRAME_WORLD = "map"  # ENU, origin at the takeoff point
FRAME_BODY = "base_link"  # FLU, origin at the drone
FRAME_CAMERA = "camera_front"  # optical: x right, y down, z into the image

# Rotation of the camera's optical frame inside the body frame, for a camera
# looking straight ahead: optical z -> body forward, optical x -> body right,
# optical y -> body down.
CAMERA_IN_BODY = Quat(w=0.5, x=-0.5, y=0.5, z=-0.5)


def camera_in_body(pitch_deg: float) -> Quat:
    """Rotation of the optical frame inside the body frame for a camera
    tilted `pitch_deg` DOWN from straight ahead (0 = ahead, 90 = down).

    The body's y axis points left, so by the right-hand rule a positive
    turn about it tips the forward axis towards the ground. The tilt is
    applied after the forward-looking mount: q = q_tilt * CAMERA_IN_BODY.
    """
    tilt = quat_from_rpy(0.0, math.radians(pitch_deg), 0.0)
    return quat_multiply(tilt, CAMERA_IN_BODY)

# ---- events -----------------------------------------------------------------

EVENT_TAKEOFF = "takeoff"
EVENT_WAYPOINT_REACHED = "waypoint_reached"
EVENT_PERSON_SEEN = "person_seen"
EVENT_LAND = "land"
KNOWN_EVENTS = (EVENT_TAKEOFF, EVENT_WAYPOINT_REACHED, EVENT_PERSON_SEEN, EVENT_LAND)

# /telemetry has no matching Foxglove schema, so it is our own message type.
# Custom types MAY use JSON (with a JSON Schema); only foxglove.* types must
# not. The name is deliberately outside the "foxglove." namespace.
TELEMETRY_SCHEMA_NAME = "sardrone.Telemetry"
TELEMETRY_JSON_SCHEMA = {
    "type": "object",
    "title": TELEMETRY_SCHEMA_NAME,
    "properties": {
        "timestamp": {
            "type": "object",
            "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
        },
        "battery_percent": {"type": "number", "description": "0..100"},
        "battery_voltage_v": {"type": "number"},
        "flight_mode": {"type": "string"},
        "velocity_enu_m_s": {
            "type": "object",
            "description": "World-frame velocity, Z up",
            "properties": {
                "x": {"type": "number", "description": "east"},
                "y": {"type": "number", "description": "north"},
                "z": {"type": "number", "description": "up"},
            },
        },
        "ground_speed_m_s": {"type": "number"},
    },
}


@dataclass
class EpisodeMeta:
    """Episode-level facts. Written to the JSON sidecar and into the MCAP."""

    world: str
    weather: str
    time_of_day: str
    mission_prompt: str
    people_placed: int = 0
    people_found: int = 0
    success: bool = False


def _ts(wall_ns: int) -> msgs.Timestamp:
    """Wall-clock nanoseconds -> Foxglove Timestamp, range-checked."""
    ts = safe_ts(wall_ns)
    return msgs.Timestamp(sec=ts.sec, nsec=ts.nsec)


def _quat_msg(q: Quat) -> msgs.Quaternion:
    return msgs.Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)


class EpisodeWriter:
    """Writes one episode. Use as a context manager:

        with EpisodeWriter("data/episodes/flight-0001.mcap", meta) as ep:
            ep.log_event(t, EVENT_TAKEOFF)
            ep.log_pose_ned(t, north, east, down, q_ned)
            ...
            ep.meta.success = True
    """

    def __init__(
        self,
        path,
        meta: EpisodeMeta,
        *,
        overwrite: bool = False,
        camera_mount_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        camera_pitch_deg: float = 0.0,
        extra_sidecar: dict | None = None,
    ):
        """`camera_mount_m` is where the camera sits on the drone, in metres,
        in the body frame (x forward, y left, z up). `camera_pitch_deg` is
        how far it looks down: 0 = straight ahead, 90 = straight down.
        `extra_sidecar` is merged into the JSON sidecar (world id, seed, ...).
        """
        self.path = Path(path)
        self.camera_mount_m = tuple(float(v) for v in camera_mount_m)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.camera_in_body = camera_in_body(self.camera_pitch_deg)
        self.extra_sidecar = dict(extra_sidecar or {})
        if self.path.suffix != ".mcap":
            raise ValueError(f"episode path must end in .mcap, got {self.path}")
        self.sidecar_path = self.path.with_suffix(".json")
        self.meta = meta
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # A private Context keeps this episode's channels separate from any
        # other writer in the same process (e.g. several tests, or two flights).
        self._context = foxglove.Context()
        self._writer = foxglove.open_mcap(
            self.path, allow_overwrite=overwrite, context=self._context
        )

        self._image = self._protobuf_channel(TOPIC_IMAGE, msgs.CompressedImage)
        self._calibration = self._protobuf_channel(TOPIC_CALIBRATION, msgs.CameraCalibration)
        self._gps = self._protobuf_channel(TOPIC_GPS, msgs.LocationFix)
        self._pose = self._protobuf_channel(TOPIC_POSE, msgs.PoseInFrame)
        self._tf = self._protobuf_channel(TOPIC_TF, msgs.FrameTransform)
        self._events = self._protobuf_channel(TOPIC_EVENTS, msgs.Log)
        self._people = self._protobuf_channel(TOPIC_GROUND_TRUTH_PEOPLE, msgs.SceneUpdate)
        self._telemetry = foxglove.Channel(
            TOPIC_TELEMETRY,
            schema=foxglove.Schema(
                name=TELEMETRY_SCHEMA_NAME,
                encoding="jsonschema",
                data=json.dumps(TELEMETRY_JSON_SCHEMA).encode(),
            ),
            message_encoding="json",
            context=self._context,
        )

        self._calibration_written = False
        self._closed = False
        self._first_ns: int | None = None
        self._last_ns: int | None = None
        self.message_counts: dict[str, int] = {}
        self.event_counts: dict[str, int] = {}

    def _protobuf_channel(self, topic: str, message_class) -> foxglove.Channel:
        schema = message_class.get_schema()
        # Guard against the "Encoding unsupported" trap in viewers.
        assert schema.encoding == "protobuf", f"{schema.name} is not protobuf"
        return foxglove.Channel(
            topic, schema=schema, message_encoding="protobuf", context=self._context
        )

    def _log(self, channel: foxglove.Channel, topic: str, payload, wall_ns: int) -> None:
        if self._closed:
            raise RuntimeError("episode is already closed")
        channel.log(payload, log_time=wall_ns)
        self.message_counts[topic] = self.message_counts.get(topic, 0) + 1
        if self._first_ns is None or wall_ns < self._first_ns:
            self._first_ns = wall_ns
        if self._last_ns is None or wall_ns > self._last_ns:
            self._last_ns = wall_ns

    # ---- camera -------------------------------------------------------------

    def log_calibration(
        self, wall_ns: int, width: int, height: int, fx: float, fy: float, cx: float, cy: float
    ) -> None:
        """Pinhole intrinsics, no distortion. Call exactly once per episode."""
        if self._calibration_written:
            raise RuntimeError("calibration is written once per episode")
        calibration = msgs.CameraCalibration(
            timestamp=_ts(wall_ns),
            frame_id=FRAME_CAMERA,
            width=width,
            height=height,
            distortion_model="plumb_bob",
            D=[0.0, 0.0, 0.0, 0.0, 0.0],
            K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
            R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self._log(self._calibration, TOPIC_CALIBRATION, calibration.encode(), wall_ns)
        self._calibration_written = True

    def log_image(self, wall_ns: int, jpeg_bytes: bytes) -> None:
        if not jpeg_bytes.startswith(b"\xff\xd8"):
            raise ValueError("log_image expects JPEG bytes (missing FF D8 header)")
        image = msgs.CompressedImage(
            timestamp=_ts(wall_ns), frame_id=FRAME_CAMERA, data=jpeg_bytes, format="jpeg"
        )
        self._log(self._image, TOPIC_IMAGE, image.encode(), wall_ns)

    # ---- navigation ---------------------------------------------------------

    def log_gps(
        self, wall_ns: int, latitude_deg: float, longitude_deg: float, altitude_m: float
    ) -> None:
        if not -90.0 <= latitude_deg <= 90.0:
            raise ValueError(f"latitude out of range: {latitude_deg}")
        if not -180.0 <= longitude_deg <= 180.0:
            raise ValueError(f"longitude out of range: {longitude_deg}")
        fix = msgs.LocationFix(
            timestamp=_ts(wall_ns),
            frame_id=FRAME_BODY,
            latitude=latitude_deg,
            longitude=longitude_deg,
            altitude=altitude_m,
        )
        self._log(self._gps, TOPIC_GPS, fix.encode(), wall_ns)

    def log_pose_ned(
        self, wall_ns: int, north_m: float, east_m: float, down_m: float, q_ned_frd: Quat
    ) -> None:
        """Drone pose as PX4 reports it. Converted to ENU here, then written
        as a PoseInFrame on /pose and as two transforms on /tf:

            map -> base_link          (the moving drone)
            base_link -> camera_front (the fixed camera mount)

        The mount never changes, but it is repeated with every pose on
        purpose: viewers look transforms up by time and flag (or drop) a
        link in the chain that is many seconds old. FiftyOne showed "Pose
        data is stale" when the mount was written only once.
        """
        position = ned_to_enu(north_m, east_m, down_m)
        orientation = _quat_msg(ned_quat_to_enu(q_ned_frd))
        timestamp = _ts(wall_ns)

        pose = msgs.PoseInFrame(
            timestamp=timestamp,
            frame_id=FRAME_WORLD,
            pose=msgs.Pose(
                position=msgs.Vector3(x=position.x, y=position.y, z=position.z),
                orientation=orientation,
            ),
        )
        self._log(self._pose, TOPIC_POSE, pose.encode(), wall_ns)

        transform = msgs.FrameTransform(
            timestamp=timestamp,
            parent_frame_id=FRAME_WORLD,
            child_frame_id=FRAME_BODY,
            translation=msgs.Vector3(x=position.x, y=position.y, z=position.z),
            rotation=orientation,
        )
        self._log(self._tf, TOPIC_TF, transform.encode(), wall_ns)

        mount = msgs.FrameTransform(
            timestamp=timestamp,
            parent_frame_id=FRAME_BODY,
            child_frame_id=FRAME_CAMERA,
            translation=msgs.Vector3(
                x=self.camera_mount_m[0], y=self.camera_mount_m[1], z=self.camera_mount_m[2]
            ),
            rotation=_quat_msg(self.camera_in_body),
        )
        self._log(self._tf, TOPIC_TF, mount.encode(), wall_ns)

    # ---- telemetry & events -------------------------------------------------

    def log_telemetry(
        self,
        wall_ns: int,
        *,
        battery_percent: float,
        battery_voltage_v: float,
        flight_mode: str,
        velocity_ned_m_s: tuple[float, float, float],
    ) -> None:
        """Velocity is passed as PX4 gives it (north, east, down) and stored Z-up."""
        ts = safe_ts(wall_ns)
        velocity = ned_to_enu(*velocity_ned_m_s)
        message = {
            "timestamp": {"sec": ts.sec, "nsec": ts.nsec},
            "battery_percent": float(battery_percent),
            "battery_voltage_v": float(battery_voltage_v),
            "flight_mode": str(flight_mode),
            "velocity_enu_m_s": {"x": velocity.x, "y": velocity.y, "z": velocity.z},
            "ground_speed_m_s": (velocity.x**2 + velocity.y**2) ** 0.5,
        }
        self._log(self._telemetry, TOPIC_TELEMETRY, message, wall_ns)

    def log_event(self, wall_ns: int, event: str, detail: str = "") -> None:
        """Mission event. The Log message text is "<event>" or "<event>: <detail>"."""
        if event not in KNOWN_EVENTS:
            raise ValueError(f"unknown event {event!r}; expected one of {KNOWN_EVENTS}")
        log = msgs.Log(
            timestamp=_ts(wall_ns),
            level=msgs.LogLevel.Info,
            message=f"{event}: {detail}" if detail else event,
            name="sardrone",
        )
        self._log(self._events, TOPIC_EVENTS, log.encode(), wall_ns)
        self.event_counts[event] = self.event_counts.get(event, 0) + 1

    def log_ground_truth_people(self, wall_ns: int, people: list[dict]) -> None:
        """The TRUE position of every person, as 3D boxes in the `map` frame.

        This is the answer key the simulator gives us for free. Each person
        is a dict with `id`, `box_centre` (x, y, z in map), `box_size`,
        `box_quat_wxyz`, plus `pose`, `clothing` and `occlusion` labels.
        Written as a foxglove.SceneUpdate, so 3D viewers draw the boxes.
        """
        entities = []
        for person in people:
            cx, cy, cz = person["box_centre"]
            sx, sy, sz = person["box_size"]
            w, x, y, z = person["box_quat_wxyz"]
            entities.append(
                msgs.SceneEntity(
                    timestamp=_ts(wall_ns),
                    frame_id=FRAME_WORLD,
                    id=person["id"],
                    frame_locked=True,
                    metadata=[
                        msgs.KeyValuePair(key=key, value=str(person[key]))
                        for key in ("pose", "clothing", "occlusion")
                        if key in person
                    ],
                    cubes=[
                        msgs.CubePrimitive(
                            pose=msgs.Pose(
                                position=msgs.Vector3(x=cx, y=cy, z=cz),
                                orientation=msgs.Quaternion(x=x, y=y, z=z, w=w),
                            ),
                            size=msgs.Vector3(x=sx, y=sy, z=sz),
                            color=msgs.Color(r=0.1, g=0.9, b=0.2, a=0.45),
                        )
                    ],
                )
            )
        update = msgs.SceneUpdate(entities=entities)
        self._log(self._people, TOPIC_GROUND_TRUTH_PEOPLE, update.encode(), wall_ns)

    # ---- finishing ----------------------------------------------------------

    def sidecar(self) -> dict:
        """Everything that goes into the JSON sidecar."""
        has_data = self._first_ns is not None
        return {
            "episode_id": self.path.stem,
            "mcap_file": self.path.name,
            **asdict(self.meta),
            "camera_pitch_deg": self.camera_pitch_deg,
            "camera_mount_m": list(self.camera_mount_m),
            **self.extra_sidecar,
            "start_time_ns": self._first_ns,
            "duration_s": (self._last_ns - self._first_ns) / NS_PER_SEC if has_data else 0.0,
            "message_counts": dict(sorted(self.message_counts.items())),
            "event_counts": dict(sorted(self.event_counts.items())),
        }

    def close(self) -> None:
        """Finish the MCAP and write the sidecar. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        sidecar = self.sidecar()
        # MCAP metadata values must be strings; this copy makes the .mcap
        # self-describing even if the sidecar gets separated from it.
        self._writer.write_metadata(
            "episode", {key: json.dumps(value) for key, value in sidecar.items()}
        )
        self._writer.close()
        self.sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

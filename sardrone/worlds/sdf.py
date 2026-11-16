"""Write a WorldSpec as SDF, the XML format Gazebo loads.

An SDF world is a list of things: physics settings, lights, and models.
Each model has a <pose> of six numbers: x y z (metres) and roll pitch yaw
(radians). This module only turns numbers into text; every decision about
what goes where was already made in layout.py.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from sardrone.worlds.assets import (
    CLOTHING,
    GROUND_EXTENT_M,
    PEOPLE,
    TREE_TEXTURES,
    TREES,
)
from sardrone.worlds.layout import (
    BASE_LINK_IN_MODEL_Z_M,
    ClutterSpec,
    DRONE_MODEL,
    PersonSpec,
    TreeSpec,
    WorldSpec,
)

# PX4's simulated home position (the same as PX4's own default world).
HOME_LATITUDE_DEG = 47.397971057728974
HOME_LONGITUDE_DEG = 8.546163739800146
HOME_ELEVATION_M = 488.0

# Resolution and lens of the camera, copied from PX4's mono_cam model.
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_HFOV_RAD = 1280, 960, 1.74


@dataclass(frozen=True)
class PreviewCamera:
    """A fixed camera in the sky, for looking at a world without flying."""

    name: str
    x: float
    y: float
    z: float
    pitch_deg: float  # 0 = level, 90 = straight down
    yaw_deg: float  # 0 = looking east, 90 = looking north


def _n(value: float) -> str:
    """A number as short, stable text (so the same world gives the same file)."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0) -> str:
    return f"<pose>{' '.join(_n(v) for v in (x, y, z, roll, pitch, yaw))}</pose>"


def _rgb(colour, alpha: float = 1.0) -> str:
    return " ".join(_n(c) for c in (*colour, alpha))


def _uri(path: Path) -> str:
    return f"file://{path}"


def _mesh_visual(name: str, mesh: Path, scale: float = 1.0, submesh: str = "", material: str = "") -> str:
    sub = f"<submesh><name>{submesh}</name></submesh>" if submesh else ""
    return (
        f'      <visual name="{name}"><geometry><mesh><uri>{_uri(mesh)}</uri>'
        f"<scale>{_n(scale)} {_n(scale)} {_n(scale)}</scale>{sub}</mesh></geometry>{material}</visual>\n"
    )


def _albedo(texture: Path, double_sided: bool = False) -> str:
    two = "<double_sided>true</double_sided>" if double_sided else ""
    return (
        f"<material>{two}<diffuse>1 1 1 1</diffuse><specular>0.05 0.05 0.05 1</specular>"
        f"<pbr><metal><albedo_map>{texture}</albedo_map><metalness>0</metalness>"
        f"<roughness>0.9</roughness></metal></pbr></material>"
    )


def tree_sdf(index: int, tree: TreeSpec, fuel_dir: Path) -> str:
    asset = TREES[tree.kind]
    folder = fuel_dir / asset.key
    visuals = "".join(
        _mesh_visual(part.lower(), folder / asset.mesh, tree.scale, part,
                     _albedo(folder / texture, double_sided=(part == "Branch")))
        for part, texture in TREE_TEXTURES[tree.kind].items()
    )  # fmt: skip
    return (
        f'    <model name="tree_{index:03d}_{tree.kind}"><static>true</static>'
        f"{_pose(tree.x, tree.y, 0, yaw=tree.yaw)}\n"
        f'      <link name="link">\n{visuals}      </link></model>\n'
    )


def person_sdf(person: PersonSpec, variant_dir: Path) -> str:
    """One person. The Label plugin gives the model a number; Gazebo's
    bounding-box sensor reports that number with every box, which tells us
    WHICH person a box belongs to."""
    asset = PEOPLE[person.pose]
    material = "" if asset.textured_by_mesh else _albedo(variant_dir / asset.clothing_texture)
    visual = _mesh_visual("body", variant_dir / asset.mesh, scale=person.scale, material=material)
    return (
        f'    <model name="{person.id}"><static>true</static>'
        f"{_pose(person.x, person.y, person.z, person.roll, person.pitch, person.yaw)}\n"
        f'      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">'
        f"<label>{person.label}</label></plugin>\n"
        f'      <link name="link">\n{visual}      </link></model>\n'
    )


def clutter_sdf(index: int, item: ClutterSpec) -> str:
    """A distractor made of a plain shape. It carries NO label plugin, so
    the bounding-box sensor never draws a box around it."""
    a, b, c = item.size
    if item.kind == "log":
        geometry = f"<cylinder><radius>{_n(a)}</radius><length>{_n(c)}</length></cylinder>"
    elif item.kind == "box":
        geometry = f"<box><size>{_n(a)} {_n(b)} {_n(c)}</size></box>"
    else:  # rock, bush: a squashed sphere
        geometry = f"<ellipsoid><radii>{_n(a)} {_n(b)} {_n(c)}</radii></ellipsoid>"
    colour = _rgb(item.rgb)
    return (
        f'    <model name="clutter_{index:03d}_{item.kind}"><static>true</static>'
        f"{_pose(item.x, item.y, item.z, item.roll, item.pitch, item.yaw)}\n"
        f'      <link name="link"><visual name="v"><geometry>{geometry}</geometry>'
        f"<material><ambient>{colour}</ambient><diffuse>{colour}</diffuse>"
        f"<specular>0.05 0.05 0.05 1</specular></material></visual></link></model>\n"
    )


def ground_sdf(texture: Path, far_texture: Path) -> str:
    """A textured square under the search area, on top of a huge plain plane
    that reaches the horizon (an oblique camera sees very far). The far
    plane uses a one-colour texture, so both are lit the same way and the
    join between them is not a visible line."""
    return f"""    <model name="ground_plane"><static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>1 1</size></plane></geometry></collision>
        <visual name="far"><geometry><plane><normal>0 0 1</normal><size>6000 6000</size></plane></geometry>
          {_albedo(far_texture)}</visual>
        <visual name="near"><pose>0 0 0.01 0 0 0</pose>
          <geometry><plane><normal>0 0 1</normal><size>{_n(GROUND_EXTENT_M)} {_n(GROUND_EXTENT_M)}</size></plane></geometry>
          {_albedo(texture)}</visual>
      </link>
    </model>
"""


def preview_camera_sdf(camera: PreviewCamera) -> str:
    # Named like the drone's camera (camera_link / camera), so the same
    # GazeboCamera class can read it.
    return f"""    <model name="{camera.name}"><static>true</static>
      {_pose(camera.x, camera.y, camera.z, 0, math.radians(camera.pitch_deg), math.radians(camera.yaw_deg))}
      <link name="camera_link">
        <sensor name="camera" type="camera">
          <camera><horizontal_fov>{CAMERA_HFOV_RAD}</horizontal_fov>
            <image><width>{CAMERA_WIDTH}</width><height>{CAMERA_HEIGHT}</height></image>
            <clip><near>0.1</near><far>3000</far></clip></camera>
          <always_on>1</always_on><update_rate>5</update_rate>
        </sensor>
      </link>
    </model>
"""


def world_sdf(
    spec: WorldSpec,
    *,
    fuel_dir: Path,
    variant_dirs: dict[str, Path],
    ground_texture: Path,
    ground_far_texture: Path,
    include_drone: bool = True,
    preview_cameras: tuple[PreviewCamera, ...] = (),
) -> str:
    """The complete world file. `variant_dirs` maps person id -> model folder."""
    sun = spec.sun
    parts = [f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by sardrone.worlds from row {spec.id!r} (seed {spec.seed}). Do not edit. -->
<sdf version="1.9">
  <world name="{spec.gz_world}">
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <grid>false</grid>
      <ambient>{_rgb(sun.ambient)}</ambient>
      <background>{_rgb(sun.sky)}</background>
      <shadows>true</shadows>
    </scene>
    <light name="sun" type="directional">
      <pose>0 0 500 0 0 0</pose>
      <cast_shadows>true</cast_shadows>
      <intensity>{_n(sun.intensity)}</intensity>
      <direction>{' '.join(_n(v) for v in sun.direction)}</direction>
      <diffuse>{_rgb(sun.diffuse)}</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
    </light>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{HOME_LATITUDE_DEG}</latitude_deg>
      <longitude_deg>{HOME_LONGITUDE_DEG}</longitude_deg>
      <elevation>{HOME_ELEVATION_M}</elevation>
    </spherical_coordinates>
"""]
    parts.append(ground_sdf(ground_texture, ground_far_texture))
    parts += [tree_sdf(i, tree, fuel_dir) for i, tree in enumerate(spec.trees, start=1)]
    parts += [person_sdf(person, variant_dirs[person.id]) for person in spec.people]
    parts += [clutter_sdf(i, item) for i, item in enumerate(spec.clutter, start=1)]
    parts += [preview_camera_sdf(camera) for camera in preview_cameras]
    if include_drone:
        parts.append(
            f"    <include><uri>model://{DRONE_MODEL}</uri><name>{spec.gz_model}</name>"
            f"{_pose(0, 0, 0)}</include>\n"
        )
    parts.append("  </world>\n</sdf>\n")
    return "".join(parts)


BOXES_TOPIC = "sar/boxes"  # Gazebo publishes the label boxes on /sar/boxes
CAMERA_RATE_HZ = 30


def _camera_block() -> str:
    return (
        f"<camera><horizontal_fov>{CAMERA_HFOV_RAD}</horizontal_fov>"
        f"<image><width>{CAMERA_WIDTH}</width><height>{CAMERA_HEIGHT}</height></image>"
        f"<clip><near>0.1</near><far>3000</far></clip>"
    )


def drone_model_sdf(pitch_deg: float, mount_m) -> str:
    """PX4's x500 quadcopter plus our search camera, pitched down.

    The camera link carries TWO sensors in the same place with the same
    lens: the ordinary colour camera, and Gazebo's "bounding box camera",
    which does not produce a picture but a list of rectangles around every
    labelled object it can see. Because both render the same view at the
    same instants, each picture gets exact labels for free.

    `mount_m` is measured from the body origin (`base_link`). Poses in this
    file are measured from the MODEL origin, which is 0.24 m lower, so that
    is added. Positive pitch about the body's y axis (which points left)
    tips the camera's forward axis towards the ground.
    """
    x, y, z = mount_m
    pose = _pose(x, y, z + BASE_LINK_IN_MODEL_Z_M, 0, math.radians(pitch_deg), 0)
    relative = _pose(x, y, z, 0, math.radians(pitch_deg), 0).replace(
        "<pose>", '<pose relative_to="base_link">'
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by sardrone.worlds. The x500 airframe comes from PX4-Autopilot. -->
<sdf version="1.9">
  <model name="{DRONE_MODEL}">
    <include merge="true"><uri>model://x500</uri></include>
    <link name="camera_link">
      {pose}
      <inertial>
        <mass>0.05</mass>
        <inertia><ixx>0.00004</ixx><iyy>0.00004</iyy><izz>0.00004</izz></inertia>
      </inertial>
      <sensor name="camera" type="camera">
        {_camera_block()}</camera>
        <always_on>1</always_on><update_rate>{CAMERA_RATE_HZ}</update_rate>
      </sensor>
      <sensor name="boxes" type="boundingbox_camera">
        <topic>{BOXES_TOPIC}</topic>
        {_camera_block()}<box_type>visible_2d</box_type></camera>
        <always_on>1</always_on><update_rate>{CAMERA_RATE_HZ}</update_rate>
      </sensor>
    </link>
    <joint name="CameraJoint" type="fixed">
      <parent>base_link</parent>
      <child>camera_link</child>
      {relative}
    </joint>
  </model>
</sdf>
"""


DRONE_MODEL_CONFIG = f"""<?xml version="1.0"?>
<model><name>{DRONE_MODEL}</name><version>1.0</version><sdf version="1.9">model.sdf</sdf>
<description>PX4 x500 with a forward-down search camera (generated by sardrone).</description></model>
"""

__all__ = ["CLOTHING", "PreviewCamera", "world_sdf", "drone_model_sdf", "DRONE_MODEL_CONFIG"]

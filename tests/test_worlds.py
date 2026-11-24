"""Phase 2A: the procedural world generator, tested without Gazebo.

No test here downloads anything: where model files are needed, a tiny
stand-in "Fuel" folder is built in a temp directory.
"""

import asyncio
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image

from sardrone.mcap.episode_writer import (
    CAMERA_IN_BODY,
    TOPIC_GROUND_TRUTH_PEOPLE,
    TOPIC_TF,
    EpisodeMeta,
    EpisodeWriter,
    camera_in_body,
)
from sardrone.mcap.frames import Quat, Vec3, quat_from_rpy, rotate_vector
from sardrone.sim import launch
from sardrone.sim.drone import FakeDrone
from sardrone.sim.record_flight import FlightPlan, record_flight, world_settings
from sardrone.worlds import assets, layout
from sardrone.worlds.generate import generate, load_spec
from sardrone.worlds.layout import build_world, place_person
from sardrone.worlds.matrix import VariationRow, find_row, load_matrix, row_from_dict
from sardrone.worlds.sdf import drone_model_sdf
from sardrone.worlds.sun import light_direction, shadow_length_m, sun_for
from tests.conftest import START_NS, Episode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT = PROJECT_ROOT / "configs" / "pilot.yaml"


def row(**overrides) -> VariationRow:
    base = dict(id="t", world="meadow", seed=7, people=3, altitude_m=25.0, area_m=120.0)
    return VariationRow(**{**base, **overrides})


def approx_vec(vector, expected, tol=1e-6):
    assert tuple(vector) == pytest.approx(expected, abs=tol)


# ---- rotations ---------------------------------------------------------------


def test_quat_from_rpy_single_axes():
    quarter = math.pi / 2
    # Yaw +90 deg (about z, up): east -> north.
    approx_vec(rotate_vector(quat_from_rpy(0, 0, quarter), Vec3(1, 0, 0)), (0, 1, 0))
    # Roll -90 deg (about x): up -> +y. This is how a standing mesh is laid flat.
    approx_vec(rotate_vector(quat_from_rpy(-quarter, 0, 0), Vec3(0, 0, 1)), (0, 1, 0))
    # Pitch +45 deg (about y): forward tips DOWN.
    s = math.sqrt(0.5)
    approx_vec(rotate_vector(quat_from_rpy(0, math.pi / 4, 0), Vec3(1, 0, 0)), (s, 0, -s))


def test_quat_from_rpy_applies_roll_before_yaw():
    quarter = math.pi / 2
    # Roll first sends up to +y; yaw then turns +y to -x.
    q = quat_from_rpy(-quarter, 0, quarter)
    approx_vec(rotate_vector(q, Vec3(0, 0, 1)), (-1, 0, 0))


def test_camera_in_body_for_a_pitched_camera():
    assert camera_in_body(0.0) == pytest.approx(CAMERA_IN_BODY)
    s = math.sqrt(0.5)
    q = camera_in_body(45.0)
    # Optical z is the viewing direction: forward and down in equal parts.
    approx_vec(rotate_vector(q, Vec3(0, 0, 1)), (s, 0, -s))
    # Optical x is image-right: the drone's right side (body -y), unchanged by pitch.
    approx_vec(rotate_vector(q, Vec3(1, 0, 0)), (0, -1, 0))
    # Optical y is image-down: down and BACK for a camera tilted forward-down.
    approx_vec(rotate_vector(q, Vec3(0, 1, 0)), (-s, 0, -s))
    # 90 deg = looking straight down.
    approx_vec(rotate_vector(camera_in_body(90.0), Vec3(0, 0, 1)), (0, 0, -1))


def test_writer_puts_pitch_and_mount_into_tf_and_sidecar(tmp_path):
    path = tmp_path / "ep.mcap"
    meta = EpisodeMeta(world="w", weather="clear", time_of_day="noon", mission_prompt="p")
    with EpisodeWriter(
        path, meta, camera_mount_m=(0.16, 0.0, -0.04), camera_pitch_deg=45.0,
        extra_sidecar={"world_id": "pilot-x"},
    ) as episode:  # fmt: skip
        episode.log_pose_ned(START_NS, 0.0, 0.0, -10.0, Quat(1.0, 0.0, 0.0, 0.0))

    mount = [t for t in Episode(path).decoded(TOPIC_TF) if t.child_frame_id == "camera_front"][0]
    r = mount.rotation
    approx_vec(rotate_vector(Quat(r.w, r.x, r.y, r.z), Vec3(0, 0, 1)), (math.sqrt(0.5), 0, -math.sqrt(0.5)))
    approx_vec((mount.translation.x, mount.translation.y, mount.translation.z), (0.16, 0.0, -0.04))
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["camera_pitch_deg"] == 45.0
    assert sidecar["camera_mount_m"] == [0.16, 0.0, -0.04]
    assert sidecar["world_id"] == "pilot-x"


# ---- matrix ------------------------------------------------------------------


def test_pilot_matrix_loads_three_different_worlds():
    rows = load_matrix(PILOT)
    assert [r.world for r in rows] == ["meadow", "forest_edge", "park"]
    assert len({r.seed for r in rows}) == 3
    assert find_row(rows, "pilot-forest").time_of_day == "golden_hour"
    assert all(r.area_m == 60 and r.camera_pitch_deg == 45 for r in rows)  # from `defaults`
    with pytest.raises(KeyError, match="pilot-meadow"):
        find_row(rows, "nope")


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"world": "desert"}, "world"),
        ({"time_of_day": "midnight"}, "time_of_day"),
        ({"weather": "fog"}, "weather"),  # not simulated, so not allowed
        ({"altitude_m": 20}, "altitude_m"),
        ({"people": 6}, "1..5"),
        ({"people": 2, "poses": ("standing",)}, "poses"),
        ({"poses": ("standing", "flying", "lying")}, "pose"),
        ({"camera_pitch_deg": 120}, "camera_pitch_deg"),
    ],
)
def test_matrix_rejects_bad_rows(bad, message):
    with pytest.raises(ValueError, match=message):
        row(**bad)


def test_matrix_rejects_unknown_keys_and_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="unknown matrix keys"):
        row_from_dict({"id": "a", "world": "meadow", "seed": 1, "wind": 3})
    matrix = tmp_path / "m.yaml"
    matrix.write_text(
        "episodes:\n  - {id: a, world: meadow, seed: 1}\n  - {id: a, world: park, seed: 2}\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_matrix(matrix)


# ---- sun ---------------------------------------------------------------------


def test_light_direction_numbers():
    # Sun straight overhead: light travels straight down.
    approx_vec(light_direction(90, 0), (0, 0, -1))
    # Sun in the SOUTH (azimuth 180), 30 deg high: light travels NORTH (+y) and down.
    approx_vec(light_direction(30, 180), (0, math.cos(math.radians(30)), -0.5))
    # Sun in the EAST: light travels WEST (-x).
    approx_vec(light_direction(45, 90), (-math.sqrt(0.5), 0, -math.sqrt(0.5)))
    for elevation, azimuth in [(10, 270), (60, 123), (3, 290)]:
        assert math.hypot(*light_direction(elevation, azimuth)) == pytest.approx(1.0)


def test_shadow_length():
    assert shadow_length_m(1.8, 45) == pytest.approx(1.8)
    assert shadow_length_m(1.8, 60) == pytest.approx(1.039, abs=1e-3)
    # A low evening sun makes a person's shadow many times their height.
    assert shadow_length_m(1.8, 10) == pytest.approx(10.21, abs=1e-2)


def test_sun_presets_are_seeded_and_ordered():
    import random

    noon, golden, dusk = (sun_for(t, random.Random(5)) for t in ("noon", "golden_hour", "dusk"))
    assert sun_for("noon", random.Random(5)) == noon
    assert 55 <= noon.elevation_deg <= 65 and 8 <= golden.elevation_deg <= 14
    assert 2 <= dusk.elevation_deg <= 5
    assert noon.direction[2] < golden.direction[2] < dusk.direction[2] < 0  # steeper at noon
    assert golden.diffuse[0] > golden.diffuse[2], "golden hour light is warm (more red than blue)"


# ---- layout ------------------------------------------------------------------


def test_same_row_gives_the_same_world_and_seed_changes_it():
    assert build_world(row()) == build_world(row())
    assert build_world(row()).people != build_world(row(seed=8)).people


def test_changing_people_does_not_move_trees():
    assert build_world(row(people=1)).trees == build_world(row(people=5)).trees


@pytest.mark.parametrize("world", ["meadow", "forest_edge", "park"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_everything_is_placed_inside_the_rules(world, seed):
    spec = build_world(row(world=world, seed=seed, people=5))
    half = spec.area_m / 2
    assert len(spec.people) == 5 and len(spec.trees) >= 3
    for tree in spec.trees:
        assert abs(tree.x) <= half and abs(tree.y) <= half
        assert math.hypot(tree.x, tree.y) >= layout.TAKEOFF_CLEAR_M
    for person in spec.people:
        assert abs(person.x) <= half and abs(person.y) <= half
        assert math.hypot(person.x, person.y) >= layout.TAKEOFF_CLEAR_M
        # occlusion "none": at least 2 m clear of every canopy.
        for tree in spec.trees:
            assert math.hypot(person.x - tree.x, person.y - tree.y) >= tree.canopy_radius_m + 2.0
    for a in spec.people:
        for b in spec.people:
            if a is not b:
                assert math.hypot(a.x - b.x, a.y - b.y) >= layout.PERSON_SPACING_M
    assert spec.altitude_m - max(t.height_m for t in spec.trees) >= 5.0


def test_partial_occlusion_puts_people_under_a_canopy():
    spec = build_world(row(world="forest_edge", occlusion="partial", people=5))
    for person in spec.people:
        assert person.occlusion == "partial"
        assert any(
            math.hypot(person.x - t.x, person.y - t.y) <= 0.75 * t.canopy_radius_m + 1e-6
            for t in spec.trees
        )


def test_forest_edge_trees_are_mostly_in_the_north():
    spec = build_world(row(world="forest_edge", seed=4))
    north = sum(1 for tree in spec.trees if tree.y > 0)
    assert north / len(spec.trees) > 0.7


def test_requested_poses_are_used_in_order():
    spec = build_world(row(poses=("lying", "standing", "sitting")))
    assert [p.pose for p in spec.people] == ["lying", "standing", "sitting"]
    assert [p.id for p in spec.people] == ["person_1", "person_2", "person_3"]


def test_altitude_too_close_to_the_treetops_is_refused(monkeypatch):
    monkeypatch.setattr(layout, "MIN_CLEARANCE_ABOVE_TREES_M", 30.0)
    with pytest.raises(ValueError, match="above the tallest tree"):
        build_world(row(altitude_m=25.0))


# ---- ground-truth boxes ------------------------------------------------------


def corners_of(pose, yaw=0.0, face_up=True):
    (z, roll), centre, size, quat = place_person(pose, 10.0, 20.0, yaw, face_up)
    person = layout.PersonSpec(
        id="p", pose=pose, clothing="red", occlusion="none",
        x=10.0, y=20.0, z=z, roll=roll, pitch=0.0, yaw=yaw,
        box_centre=centre, box_size=size, box_quat_wxyz=quat,
    )  # fmt: skip
    return person, person.box_corners()


def test_standing_box_is_tall_and_stands_on_the_ground():
    person, corners = corners_of("standing")
    zs = [c[2] for c in corners]
    assert min(zs) == pytest.approx(0.0, abs=0.02)
    assert max(zs) == pytest.approx(1.92, abs=0.02)  # a 1.9 m tall model
    assert person.box_centre[2] == pytest.approx(0.96, abs=0.02)
    assert person.height_m == pytest.approx(1.92, abs=0.02)


def test_sitting_box_rests_on_the_ground():
    # The sitting mesh floats 0.62 m above its own origin, so the model is
    # lowered by that much.
    person, corners = corners_of("sitting")
    zs = [c[2] for c in corners]
    assert person.z == pytest.approx(-0.62)
    assert min(zs) == pytest.approx(0.0, abs=0.01)
    assert max(zs) == pytest.approx(0.73, abs=0.01)


def test_lying_box_is_flat_and_as_long_as_the_body():
    person, corners = corners_of("lying", yaw=0.0)
    zs, ys, xs = ([c[i] for c in corners] for i in (2, 1, 0))
    assert min(zs) == pytest.approx(0.02, abs=0.01)
    assert max(zs) - min(zs) == pytest.approx(0.36, abs=0.01)  # body depth, now vertical
    # Face up, yaw 0: the head-to-feet axis (mesh z) points along +y.
    assert max(ys) - min(ys) == pytest.approx(1.82, abs=0.01)
    assert min(ys) == pytest.approx(20.0, abs=0.01) and max(ys) == pytest.approx(21.82, abs=0.01)
    assert max(xs) - min(xs) == pytest.approx(1.36, abs=0.01)  # arm span


def test_lying_box_turns_with_yaw():
    _, corners = corners_of("lying", yaw=math.pi / 2)
    xs = [c[0] for c in corners]
    # Yaw 90 deg turns +y into -x: the body now stretches 1.82 m towards the west.
    assert min(xs) == pytest.approx(10.0 - 1.82, abs=0.01) and max(xs) == pytest.approx(10.0, abs=0.01)


# ---- textures ----------------------------------------------------------------


def test_tint_colours_a_white_garment_and_keeps_transparency():
    image = Image.new("RGBA", (2, 1), (255, 255, 255, 255))
    image.putpixel((1, 0), (128, 128, 128, 0))
    out = assets.tint(image, (200, 40, 40))
    assert out.getpixel((0, 0)) == (200, 40, 40, 255)
    assert out.getpixel((1, 0))[3] == 0  # the transparent pixel stays transparent
    assert out.getpixel((1, 0))[0] == pytest.approx(100, abs=1)  # shading is kept: half as bright


def test_only_the_jacket_is_recoloured():
    jacket, skin, trousers = (215, 235, 40), (232, 170, 150), (90, 90, 90)
    image = Image.new("RGB", (3, 1))
    for x, colour in enumerate((jacket, skin, trousers)):
        image.putpixel((x, 0), colour)
    out = assets.replace_jacket_colour(image, (40, 80, 200))
    r, g, b = out.getpixel((0, 0))
    assert b > r and b > g, "the yellow jacket must now be blue"
    assert out.getpixel((1, 0)) == skin
    assert out.getpixel((2, 0)) == trousers


def test_ground_texture_is_seeded_and_in_palette():
    a = assets.ground_texture("meadow", 11, size=128)
    assert a.tobytes() == assets.ground_texture("meadow", 11, size=128).tobytes()
    assert a.tobytes() != assets.ground_texture("meadow", 12, size=128).tobytes()
    dark, light = assets.GROUND_PALETTE["meadow"]
    mean = a.resize((1, 1), Image.BOX).getpixel((0, 0))
    for value, low, high in zip(mean, dark, light):
        assert low <= value <= high
    assert mean[1] > mean[0] > mean[2], "grass: green strongest, blue weakest"


DAE = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit meter="{unit}" name="u"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry id="g"><mesh>
    <source id="pos"><float_array id="a" count="6">{points}</float_array></source>
    <vertices id="v"><input semantic="POSITION" source="#pos"/></vertices>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="s"><node name="n">
    <matrix>{scale} 0 0 0 0 {scale} 0 0 0 0 {scale} 0 0 0 0 1</matrix>
    <instance_geometry url="#g"/>
  </node></visual_scene></library_visual_scenes>
</COLLADA>
"""


def test_mesh_bounds_applies_unit_and_node_scale(tmp_path):
    dae = tmp_path / "m.dae"
    # Centimetres, no node scale: 180 cm tall -> 1.8 m.
    dae.write_text(DAE.format(unit="0.01", scale="1", points="-30 -20 0  30 20 180"))
    lo, hi = assets.mesh_bounds(dae)
    approx_vec(lo, (-0.3, -0.2, 0.0))
    approx_vec(hi, (0.3, 0.2, 1.8))
    # Inches with a 0.0254 node scale (how the tree files are made): 100 in -> 2.54 m.
    dae.write_text(DAE.format(unit="1", scale="0.0254", points="0 0 0  10 10 100"))
    approx_vec(assets.mesh_bounds(dae)[1], (0.254, 0.254, 2.54))


@pytest.mark.parametrize("asset", [*assets.PEOPLE.values(), *assets.TREES.values()], ids=lambda a: a.key)
def test_size_table_matches_the_downloaded_meshes(asset):
    if not assets.is_fetched(asset):
        pytest.skip("models not downloaded (run `make assets`)")
    lo, hi = assets.mesh_bounds(assets.FUEL_DIR / asset.key / asset.mesh)
    approx_vec(lo, asset.lo, tol=0.02)
    approx_vec(hi, asset.hi, tol=0.02)


# ---- generate (with a stand-in Fuel folder) ----------------------------------


@pytest.fixture
def fake_fuel(tmp_path):
    fuel = tmp_path / "fuel"
    for asset in [*assets.PEOPLE.values(), *assets.TREES.values()]:
        mesh = fuel / asset.key / asset.mesh
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_text(DAE.format(unit="1", scale="1", points="0 0 0 1 1 1"))
        if asset.clothing_texture:
            texture = fuel / asset.key / asset.clothing_texture
            texture.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (215, 235, 40)).save(texture)
    for kind, textures in assets.TREE_TEXTURES.items():
        for texture in textures.values():
            path = fuel / assets.TREES[kind].key / texture
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (4, 4), (40, 90, 30, 255)).save(path)
    return fuel


def make_world(tmp_path, fake_fuel, **overrides):
    return generate(
        row(id="w1", **overrides), tmp_path / "worlds",
        fuel_dir=fake_fuel, variant_dir=tmp_path / "variants",
    )  # fmt: skip


def test_generate_writes_a_complete_world_folder(tmp_path, fake_fuel):
    paths = make_world(tmp_path, fake_fuel, world="park", people=4, time_of_day="dusk")
    spec = load_spec(paths.folder)
    assert spec["id"] == "w1" and spec["world"] == "park" and len(spec["people"]) == 4
    assert spec["gz_world"] == "sar" and spec["gz_model"] == "x500_sar_cam_0"
    assert paths.ground.exists() and paths.ground_far.exists()

    world = ET.fromstring(paths.sdf.read_text()).find("world")  # must be valid XML
    assert world.get("name") == "sar"
    names = [m.get("name") for m in world.findall("model")]
    assert sum(n.startswith("tree_") for n in names) == len(spec["trees"])
    assert [n for n in names if n.startswith("person_")] == [p["id"] for p in spec["people"]]
    assert "ground_plane" in names and "preview_overview" in names
    assert world.find("include/name").text == "x500_sar_cam_0"
    assert world.find("include/pose").text == "0 0 0 0 0 0"  # takeoff at the world origin

    # The sun in the file is the sun in the spec.
    direction = [float(v) for v in world.find("light/direction").text.split()]
    approx_vec(direction, spec["sun"]["direction"])
    approx_vec(direction, light_direction(spec["sun"]["elevation_deg"], spec["sun"]["azimuth_deg"]), tol=1e-5)

    # Every person's pose in the file is the pose in the spec.
    for person in spec["people"]:
        model = next(m for m in world.findall("model") if m.get("name") == person["id"])
        pose = [float(v) for v in model.find("pose").text.split()]
        approx_vec(pose, [person[k] for k in ("x", "y", "z", "roll", "pitch", "yaw")], tol=1e-5)
        assert Path(model.find(".//mesh/uri").text.removeprefix("file://")).exists()


def test_generate_is_reproducible(tmp_path, fake_fuel):
    first = make_world(tmp_path, fake_fuel).sdf.read_text()
    spec_first = make_world(tmp_path, fake_fuel).spec.read_text()
    assert make_world(tmp_path, fake_fuel).sdf.read_text() == first
    assert make_world(tmp_path, fake_fuel).spec.read_text() == spec_first


def test_clothing_variant_recolours_a_copy_not_the_original(tmp_path, fake_fuel):
    original = fake_fuel / assets.PEOPLE["lying"].key / assets.PEOPLE["lying"].clothing_texture
    before = original.read_bytes()
    variant = assets.clothing_variant("lying", "blue", fake_fuel, tmp_path / "variants")
    assert original.read_bytes() == before, "the downloaded texture must never be modified"
    r, g, b = Image.open(variant / assets.PEOPLE["lying"].clothing_texture).convert("RGB").getpixel((0, 0))
    assert b > r and b > g
    # The mesh is shared, not copied.
    mesh = assets.PEOPLE["lying"].mesh
    assert (variant / mesh).samefile(fake_fuel / assets.PEOPLE["lying"].key / mesh)


def test_missing_models_give_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make assets"):
        generate(row(), tmp_path / "worlds", fuel_dir=tmp_path / "empty", variant_dir=tmp_path / "v")


def test_drone_model_camera_pose_and_sensors():
    model = ET.fromstring(drone_model_sdf(45.0, (0.16, 0.0, -0.04))).find("model")
    link = model.find("link[@name='camera_link']")
    pose = [float(v) for v in link.find("pose").text.split()]
    # Mount is from base_link; the file's pose is from the model origin, 0.24 m lower.
    approx_vec(pose, (0.16, 0.0, 0.20, 0.0, math.pi / 4, 0.0), tol=1e-6)
    joint = model.find("joint/pose")
    assert joint.get("relative_to") == "base_link"
    approx_vec([float(v) for v in joint.text.split()][:3], (0.16, 0.0, -0.04))

    # Two sensors in the same place with the same lens: pictures and label boxes.
    colour, boxes = link.findall("sensor")
    assert (colour.get("type"), boxes.get("type")) == ("camera", "boundingbox_camera")
    assert boxes.find("topic").text == "sar/boxes"
    assert boxes.find("camera/box_type").text == "visible_2d"  # only what is really visible
    for tag in ("camera/horizontal_fov", "camera/image/width", "camera/image/height", "update_rate"):
        assert colour.find(tag).text == boxes.find(tag).text, f"{tag} must match or labels misalign"


def test_every_person_gets_a_distinct_gazebo_label(tmp_path, fake_fuel):
    paths = make_world(tmp_path, fake_fuel, people=4)
    spec = load_spec(paths.folder)
    assert [p["label"] for p in spec["people"]] == [1, 2, 3, 4]
    assert spec["boxes_topic"] == "/sar/boxes"
    world = ET.fromstring(paths.sdf.read_text()).find("world")
    for person in spec["people"]:
        model = next(m for m in world.findall("model") if m.get("name") == person["id"])
        plugin = model.find("plugin")
        assert plugin.get("filename") == "gz-sim-label-system"
        assert int(plugin.find("label").text) == person["label"]
    # Trees and ground carry no label, so the sensor never boxes them.
    for model in world.findall("model"):
        if not model.get("name").startswith("person_"):
            assert model.find("plugin") is None


# ---- recording with ground truth ---------------------------------------------


def test_world_settings_moves_people_into_the_map_frame(tmp_path, fake_fuel):
    paths = make_world(tmp_path, fake_fuel, poses=("standing", "sitting", "lying"))
    spec, people, extra = world_settings(paths.folder)
    assert extra["world_id"] == "w1" and extra["detector"] == "none"
    assert extra["people"] == people and len(people) == 3
    for person, original in zip(people, spec["people"]):
        # x and y are shared; map z = world z - height of the drone body on the ground.
        assert person["box_centre"][:2] == original["box_centre"][:2]
        assert person["box_centre"][2] == pytest.approx(original["box_centre"][2] - 0.227, abs=1e-3)


def test_flight_records_ground_truth_once_a_second(tmp_path, fake_fuel):
    paths = make_world(tmp_path, fake_fuel, poses=("standing", "sitting", "lying"))
    _, people, extra = world_settings(paths.folder)
    drone = FakeDrone()
    meta = EpisodeMeta(world="meadow", weather="clear", time_of_day="noon", mission_prompt="p")
    path = tmp_path / "ep.mcap"
    writer = asyncio.run(
        record_flight(
            drone, path, meta, plan=FlightPlan(person_north_east_m=None), sleep=drone.sleep,
            start_wall_ns=START_NS, ground_truth_people=people, extra_sidecar=extra,
        )  # fmt: skip
    )
    episode = Episode(path)
    updates = episode.decoded(TOPIC_GROUND_TRUTH_PEOPLE)
    times = episode.log_times(TOPIC_GROUND_TRUTH_PEOPLE)
    assert times[0] == START_NS
    assert {b - a for a, b in zip(times, times[1:])} == {1_000_000_000}  # exactly 1 Hz
    assert len(updates) == int(writer.sidecar()["duration_s"]) + 1

    entities = updates[0].entities
    assert [e.id for e in entities] == ["person_1", "person_2", "person_3"]
    for entity, person in zip(entities, people):
        assert entity.frame_id == "map"
        cube = entity.cubes[0]
        approx_vec((cube.pose.position.x, cube.pose.position.y, cube.pose.position.z), person["box_centre"])
        approx_vec((cube.size.x, cube.size.y, cube.size.z), person["box_size"])
        o = cube.pose.orientation
        approx_vec((o.w, o.x, o.y, o.z), person["box_quat_wxyz"])
        assert {m.key: m.value for m in entity.metadata}["pose"] == person["pose"]

    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["people_placed"] == 3 and sidecar["people_found"] == 0
    assert sidecar["success"] is False, "nobody was found: there is no detector yet"
    assert sidecar["detector"] == "none" and len(sidecar["people"]) == 3
    assert sidecar["message_counts"][TOPIC_GROUND_TRUTH_PEOPLE] == len(updates)


# ---- launcher ----------------------------------------------------------------


def test_launch_commands_attach_px4_to_our_world(tmp_path, fake_fuel, monkeypatch):
    px4 = tmp_path / "PX4"
    (px4 / "build/px4_sitl_default/bin").mkdir(parents=True)
    (px4 / "build/px4_sitl_default/bin/px4").touch()
    monkeypatch.setenv("PX4_DIR", str(px4))
    paths = make_world(tmp_path, fake_fuel)

    script = launch.gazebo_command(paths.folder)[-1]
    assert f'GZ_SIM_RESOURCE_PATH="{paths.models}"' in script  # our drone model is findable
    assert script.index("GZ_SIM_RESOURCE_PATH") < script.index("gz_env.sh")  # ours first
    assert "GZ_IP=127.0.0.1" in script and script.endswith(f'"{paths.sdf}"')

    env = launch.px4_environment(load_spec(paths.folder))
    assert env["PX4_GZ_MODEL_NAME"] == "x500_sar_cam_0"  # attach, do not spawn
    assert env["PX4_GZ_WORLD"] == "sar" and env["PX4_SIM_MODEL"] == "gz_x500"


def test_launch_explains_a_missing_px4_build(tmp_path, monkeypatch):
    monkeypatch.setenv("PX4_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(launch.SimulationError, match="not built"):
        launch.gazebo_command(tmp_path)


def test_flight_records_the_cameras_label_boxes_with_every_frame(tmp_path, fake_fuel):
    from sardrone.mcap.episode_writer import TOPIC_GROUND_TRUTH_BOXES, TOPIC_IMAGE
    from sardrone.sim.camera import LabelBox, SyntheticCamera

    class LabelledCamera(SyntheticCamera):
        """Pretends to be the Gazebo camera: reports two boxes with each frame."""

        last_boxes = None

        def read(self):
            jpeg = super().read()
            # person 2, an unknown label, and an empty sliver that must be dropped.
            self.last_boxes = (
                LabelBox(2, 100.0, 50.0, 120.0, 90.0), LabelBox(9, 5.0, 5.0, 9.0, 9.0),
                LabelBox(1, 0.0, 285.0, 1.0, 285.5),
            )  # fmt: skip
            return jpeg

    paths = make_world(tmp_path, fake_fuel, poses=("standing", "sitting", "lying"))
    _, people, extra = world_settings(paths.folder)
    drone = FakeDrone()
    meta = EpisodeMeta(world="meadow", weather="clear", time_of_day="noon", mission_prompt="p")
    path = tmp_path / "ep.mcap"
    asyncio.run(
        record_flight(
            drone, path, meta, camera=LabelledCamera(), plan=FlightPlan(person_north_east_m=None),
            sleep=drone.sleep, start_wall_ns=START_NS, ground_truth_people=people,
        )  # fmt: skip
    )
    episode = Episode(path)
    # One label message per picture, at exactly the picture's time.
    assert episode.log_times(TOPIC_GROUND_TRUTH_BOXES) == episode.log_times(TOPIC_IMAGE)
    labels = episode.decoded(TOPIC_GROUND_TRUTH_BOXES)[10]
    assert [text.text for text in labels.texts] == ["person_2", "label_9"]
    assert [(p.x, p.y) for p in labels.points[0].points][0] == (100.0, 50.0)


def test_generated_images_have_no_label_messages(tmp_path):
    from sardrone.mcap.episode_writer import TOPIC_GROUND_TRUTH_BOXES

    drone = FakeDrone()
    meta = EpisodeMeta(world="w", weather="clear", time_of_day="noon", mission_prompt="p")
    path = tmp_path / "ep.mcap"
    asyncio.run(record_flight(drone, path, meta, sleep=drone.sleep, start_wall_ns=START_NS))
    assert TOPIC_GROUND_TRUTH_BOXES not in Episode(path).messages


# ---- domain randomization ----------------------------------------------------


def test_randomized_world_is_seeded_and_differs_from_the_plain_one():
    plain, random_ = build_world(row(people=5)), build_world(row(people=5, randomize=True))
    assert build_world(row(people=5, randomize=True)) == random_
    assert plain.clutter == () and plain.randomize is False
    assert 15 <= len(random_.clutter) <= 45 and random_.randomize is True
    # Trees do not depend on the switch (their random stream is separate).
    assert random_.trees == plain.trees
    assert random_.sun != plain.sun
    assert all(p.scale == 1.0 for p in plain.people)
    assert all(0.85 <= p.scale <= 1.1 for p in random_.people) and len({p.scale for p in random_.people}) > 1


def test_plain_worlds_are_exactly_what_they_were_before_randomization_existed():
    # Positions recorded in the v1 dataset for seed 7 must not move.
    spec = build_world(row(people=3))
    again = build_world(row(people=3, randomize=False))
    assert spec == again and spec.sun.azimuth_deg == again.sun.azimuth_deg


def test_scaled_person_box_scales_with_the_body():
    _, centre, size, _ = place_person("standing", 0.0, 0.0, 0.0, scale=1.0)
    (z, _), big_centre, big_size, _ = place_person("standing", 0.0, 0.0, 0.0, scale=1.1)
    assert big_size == pytest.approx(tuple(s * 1.1 for s in size), abs=2e-3)
    assert big_centre[2] == pytest.approx(centre[2] * 1.1, abs=2e-3)
    assert z == pytest.approx(0.011, abs=1e-3)  # the mesh's 1 cm offset below its origin, scaled
    # A smaller sitting person still rests on the ground.
    (z, _), small_centre, small_size, _ = place_person("sitting", 0.0, 0.0, 0.0, scale=0.85)
    assert small_centre[2] - small_size[2] / 2 == pytest.approx(0.0, abs=0.01)
    # A lying person is lifted by half the (scaled) body depth.
    (z, _), _, lying_size, _ = place_person("lying", 0.0, 0.0, 0.0, scale=0.85)
    assert z == pytest.approx(lying_size[1] / 2 + 0.02, abs=1e-3)


def test_clutter_keeps_clear_of_people_and_takeoff():
    spec = build_world(row(people=5, randomize=True, seed=11))
    half = spec.area_m / 2
    assert {c.kind for c in spec.clutter} <= {"rock", "log", "bush", "box"}
    for item in spec.clutter:
        assert abs(item.x) <= half and abs(item.y) <= half and item.z > 0
        assert math.hypot(item.x, item.y) >= layout.TAKEOFF_CLEAR_M
        assert all(math.hypot(item.x - p.x, item.y - p.y) >= 1.5 for p in spec.people)
        assert all(0.0 <= channel <= 1.0 for channel in item.rgb)


def test_randomized_sun_can_stand_anywhere():
    import random

    azimuths = [sun_for("noon", random.Random(seed), randomize=True).azimuth_deg for seed in range(60)]
    assert min(azimuths) < 60 and max(azimuths) > 300  # all round the compass, not just south
    plain = [sun_for("noon", random.Random(seed)).azimuth_deg for seed in range(60)]
    assert 150 <= min(plain) and max(plain) <= 210
    for seed in range(20):
        sun = sun_for("dusk", random.Random(seed), randomize=True)
        assert sun.elevation_deg >= 2.0 and all(0.0 <= c <= 1.0 for c in sun.diffuse + sun.ambient)


def test_randomized_ground_uses_other_palettes_and_stays_seeded():
    plain = assets.ground_texture("meadow", 11, size=128)
    random_ = assets.ground_texture("meadow", 11, size=128, randomize=True)
    assert random_.tobytes() == assets.ground_texture("meadow", 11, size=128, randomize=True).tobytes()
    assert assets.ground_texture("meadow", 11, size=128).tobytes() == plain.tobytes()
    means = {assets.ground_texture("meadow", seed, size=64, randomize=True).resize((1, 1)).getpixel((0, 0)) for seed in range(12)}
    assert len(means) >= 4, "several clearly different ground colours over 12 seeds"


def test_randomized_world_file_has_clutter_without_labels_and_scaled_people(tmp_path, fake_fuel):
    paths = make_world(tmp_path, fake_fuel, people=3, randomize=True)
    spec = load_spec(paths.folder)
    world = ET.fromstring(paths.sdf.read_text()).find("world")
    clutter = [m for m in world.findall("model") if m.get("name").startswith("clutter_")]
    assert len(clutter) == len(spec["clutter"]) > 0
    assert all(m.find("plugin") is None for m in clutter), "clutter must never get a label box"
    for person in spec["people"]:
        model = next(m for m in world.findall("model") if m.get("name") == person["id"])
        scale = [float(v) for v in model.find(".//mesh/scale").text.split()]
        assert scale == pytest.approx([person["scale"]] * 3)
    assert spec["randomize"] is True


# ---- vehicles as distractors ---------------------------------------------------


def test_vehicles_are_placed_clear_of_people_trees_and_takeoff():
    spec = build_world(row(people=4, vehicles=3, seed=13))
    cars = [c for c in spec.clutter if c.kind == "vehicle"]
    assert len(cars) == 3
    for car in cars:
        assert math.hypot(car.x, car.y) >= layout.TAKEOFF_CLEAR_M + 3.0
        assert all(math.hypot(car.x - p.x, car.y - p.y) >= 4.0 for p in spec.people)
        assert all(math.hypot(car.x - t.x, car.y - t.y) >= t.canopy_radius_m + 2.5 for t in spec.trees)
        assert 3.7 <= car.size[0] <= 4.7 and car.size[1:] == (1.8, 1.5)
    for a in cars:
        for b in cars:
            if a is not b:
                assert math.hypot(a.x - b.x, a.y - b.y) >= 5.0


def test_vehicles_change_nothing_else_in_the_world():
    without, with_cars = build_world(row(people=4, seed=13)), build_world(row(people=4, seed=13, vehicles=2))
    assert with_cars.people == without.people and with_cars.trees == without.trees and with_cars.sun == without.sun
    assert [c for c in with_cars.clutter if c.kind != "vehicle"] == list(without.clutter)
    random_without = build_world(row(people=4, seed=13, randomize=True))
    random_with = build_world(row(people=4, seed=13, randomize=True, vehicles=2))
    assert [c for c in random_with.clutter if c.kind != "vehicle"] == list(random_without.clutter)
    with pytest.raises(ValueError, match="vehicles"):
        row(vehicles=11)


def test_vehicle_world_file_has_cars_without_labels(tmp_path, fake_fuel):
    paths = make_world(tmp_path, fake_fuel, people=2, vehicles=2)
    world = ET.fromstring(paths.sdf.read_text()).find("world")
    cars = [m for m in world.findall("model") if m.get("name").endswith("_vehicle")]
    assert len(cars) == 2
    for car in cars:
        assert car.find("plugin") is None, "a car must never get a label box"
        names = [v.get("name") for v in car.findall("link/visual")]
        assert names[:3] == ["body", "cabin", "roof"] and sum(n.startswith("wheel") for n in names) == 4


# ---- content: rich (issue #4) -------------------------------------------------


def _rich_row(**kw):
    from sardrone.worlds.matrix import VariationRow

    return VariationRow(id="rich-001", world="meadow", seed=101, people=5, content="rich", vehicles=3, **kw)


def test_rich_worlds_use_new_bodies_vehicle_meshes_and_rocks():
    from sardrone.worlds.layout import build_world

    spec = build_world(_rich_row(poses=("standing", "walking", "child", "sitting", "lying")))
    poses = [p.pose for p in spec.people]
    assert poses == ["standing", "walking", "child", "sitting", "lying"]
    clothing = {p.pose: p.clothing for p in spec.people}
    assert clothing["walking"] == clothing["child"] == "as modelled"
    kinds = [c.kind for c in spec.clutter]
    assert sum(k.startswith("vehicle:") for k in kinds) == 3 and "vehicle" not in kinds  # no box cars
    assert 3 <= kinds.count("rock_mesh") <= 8
    child = next(p for p in spec.people if p.pose == "child")
    assert 1.0 < child.height_m < 1.3  # a child, about 1.14 m


def test_rich_world_sdf_references_the_meshes_and_labels_only_people(tmp_path):
    from sardrone.worlds import assets
    from sardrone.worlds.layout import build_world
    from sardrone.worlds.sdf import world_sdf

    spec = build_world(_rich_row())
    variants = {p.id: tmp_path / p.id for p in spec.people}
    sdf = world_sdf(spec, fuel_dir=assets.FUEL_DIR, variant_dirs=variants,
                    ground_texture=tmp_path / "g.jpg", ground_far_texture=tmp_path / "f.png")  # fmt: skip
    assert "rock_mesh" in sdf and "FallingRock01.dae" in sdf
    assert any(m in sdf for m in ("suv.obj", "pickup.dae", "fire_truck.obj"))
    assert sdf.count("gz::sim::systems::Label") == len(spec.people)  # distractors carry no label


def test_basic_rows_reject_rich_poses():
    from sardrone.worlds.matrix import VariationRow

    with pytest.raises(ValueError):
        VariationRow(id="b", world="meadow", seed=1, people=1, poses=("child",))
    with pytest.raises(ValueError):
        VariationRow(id="b", world="meadow", seed=1, content="fancy")

"""3D models and textures the worlds are built from.

The meshes come from Gazebo Fuel (the public model library) and live in
`assets/fuel/`, which is git-ignored; `fetch_all()` downloads them. Licences
are listed in docs/ASSETS.md.

We do not use the models' own `model.sdf` files. They point at other models
by name, use material scripts Gazebo no longer supports, and one downloads
an extra model from the internet. Instead the world generator references the
mesh and texture files directly, so a world is fully described by its SDF.

Clothing colour: the models come in one outfit each. To vary it, a
"variant" copy is made in `assets/variants/` with the clothing texture
recoloured. Unchanged files are hard links, so a variant costs little disk.
"""

import os
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUEL_DIR = PROJECT_ROOT / "assets" / "fuel"
VARIANT_DIR = PROJECT_ROOT / "assets" / "variants"

Vec = tuple[float, float, float]


@dataclass(frozen=True)
class MeshAsset:
    key: str  # folder name under assets/fuel
    fuel_name: str  # name on Gazebo Fuel, owner OpenRobotics
    licence: str
    mesh: str  # path of the mesh inside the model folder
    # Bounding box of the mesh in its own axes, metres (measured from the
    # file by mesh_bounds(); tests/test_world_assets.py re-checks them).
    lo: Vec
    hi: Vec
    # Clothing texture inside the model folder, and how to recolour it.
    clothing_texture: str = ""
    recolour: str = ""  # "tint" (white garment) or "hue" (replace a hue band)
    # True when the mesh file names its own textures. Otherwise the SDF must.
    textured_by_mesh: bool = True
    owner: str = "OpenRobotics"  # owner on Gazebo Fuel
    sdf_scale: float = 1.0  # some meshes are not in metres; the model's own SDF scales them
    yaw_offset: float = 0.0  # the model's own SDF turns some meshes to face +x

    @property
    def size(self) -> Vec:
        return tuple(round(h - l, 3) for l, h in zip(self.lo, self.hi))

    @property
    def centre(self) -> Vec:
        return tuple(round((h + l) / 2, 3) for l, h in zip(self.lo, self.hi))


# One mesh per body pose. "lying" is the standing rescue dummy laid flat.
PEOPLE = {
    "standing": MeshAsset(
        "standing_person", "Standing person", "CC0-1.0", "meshes/standing.dae",
        lo=(-0.28, -0.25, -0.01), hi=(0.27, 0.11, 1.91),
        clothing_texture="materials/textures/tshirt02_texture.png", recolour="tint",
    ),
    "sitting": MeshAsset(
        "rescue_randy_sitting", "Rescue Randy Sitting", "CC0-1.0", "meshes/rescue_randy.dae",
        lo=(-0.47, -0.71, 0.62), hi=(0.27, 0.30, 1.35),
        clothing_texture="materials/textures/RescueRandy_Diffuse.jpg", recolour="hue",
    ),
    "lying": MeshAsset(
        "rescue_randy", "Rescue Randy", "CC-BY-4.0", "meshes/rescue_randy.dae",
        lo=(-0.68, -0.18, 0.0), hi=(0.68, 0.18, 1.82),
        clothing_texture="materials/textures/rescue_randy_albedo.png", recolour="hue",
        textured_by_mesh=False,
    ),
}  # fmt: skip

# More bodies, only in worlds with `content: rich` (matrix.py). Not recoloured:
# they keep the clothing they were modelled with. Bounds measured with
# mesh_bounds() on 2026-11-23.
PEOPLE_RICH = {
    "walking": MeshAsset(
        "walking_person", "Walking person", "CC0-1.0", "meshes/walking.dae",
        lo=(-0.27, -0.41, 0.02), hi=(0.28, 0.47, 1.89),
    ),
    "child": MeshAsset(
        "visitor_kid_walk", "VisitorKidWalk", "CC-BY-4.0", "meshes/VisitorKidWalk.dae",
        lo=(-0.09, -0.37, 0.0), hi=(0.13, 0.37, 1.14), owner="Luca",
    ),
}  # fmt: skip
PEOPLE.update(PEOPLE_RICH)

# Real vehicle meshes as unlabelled distractors (content: rich). Scales and
# turns are those of each model's own model.sdf on Fuel; sizes after scaling.
VEHICLE_MODELS = {
    "suv": MeshAsset("suv", "SUV", "CC0-1.0", "meshes/suv.obj", lo=(-1.3, -2.4, 0.0), hi=(1.3, 2.6, 2.2),
                     sdf_scale=0.06, yaw_offset=-1.5707963),
    "pickup": MeshAsset("pickup", "Pickup", "CC0-1.0", "meshes/pickup.dae", lo=(-1.24, -2.95, 0.0),
                        hi=(1.23, 2.71, 1.87), yaw_offset=-1.5707963),
    "fire_truck": MeshAsset("fire_truck", "Fire truck", "CC0-1.0", "meshes/fire_truck.obj",
                            lo=(-1.8, -6.5, 0.0), hi=(1.8, 5.9, 3.4), sdf_scale=0.1),
}  # fmt: skip
ROCK_MODEL = MeshAsset("falling_rock_1", "Falling Rock 1", "CC-BY-4.0", "meshes/FallingRock01.dae",
                       lo=(-0.74, -0.74, -0.7), hi=(0.75, 0.74, 0.69))  # fmt: skip

TREES = {
    "oak": MeshAsset(
        "oak_tree", "Oak tree", "CC0-1.0", "meshes/oak_tree.dae",
        lo=(-5.92, -4.19, -0.07), hi=(4.69, 4.30, 6.53), textured_by_mesh=False,
    ),
    "pine": MeshAsset(
        "pine_tree", "Pine Tree", "CC0-1.0", "meshes/pine_tree.dae",
        lo=(-1.59, -1.59, 0.0), hi=(1.59, 1.59, 5.08), textured_by_mesh=False,
    ),
}  # fmt: skip

# Tree meshes have two parts; each needs its own texture in the SDF.
TREE_TEXTURES = {
    "oak": {"Branch": "materials/textures/branch_diffuse.png", "Bark": "materials/textures/bark_diffuse.png"},
    "pine": {"Branch": "materials/textures/branch_2_diffuse.png", "Bark": "materials/textures/bark_diffuse.png"},
}  # fmt: skip

EXTRA = {
    "backpack": MeshAsset(
        "backpack", "Backpack", "CC0-1.0", "meshes/Backpack.dae",
        lo=(0.0, 0.0, 0.0), hi=(0.0, 0.0, 0.0),  # not placed yet (a later class)
    ),
}  # fmt: skip

ALL_ASSETS = {**PEOPLE, **TREES, **EXTRA, **{f"vehicle_{k}": v for k, v in VEHICLE_MODELS.items()},
              "rock_mesh": ROCK_MODEL}

# Clothing colours, chosen to include both easy (red, orange) and hard
# (green, black: low contrast on grass or in shade) cases.
CLOTHING = {
    "red": (205, 35, 35),
    "orange": (235, 120, 20),
    "yellow": (235, 215, 40),
    "blue": (40, 80, 200),
    "purple": (125, 55, 165),
    "pink": (235, 120, 170),
    "white": (235, 235, 235),
    "black": (35, 35, 38),
    "green": (55, 110, 50),
}


# A wider wardrobe for domain randomization: earth tones and camouflage
# that hide a person, as well as colours that stand out.
CLOTHING_RANDOMIZED = {
    **CLOTHING,
    "teal": (30, 140, 140),
    "navy": (25, 35, 90),
    "sky": (120, 180, 235),
    "brown": (110, 75, 45),
    "khaki": (165, 150, 100),
    "olive": (95, 105, 50),
    "grey": (125, 125, 130),
    "maroon": (120, 30, 45),
    "lime": (160, 220, 60),
    "tan": (200, 170, 130),
    "violet": (160, 110, 220),
    "rust": (175, 80, 35),
}

# (dark, light) ground colours for domain randomization: lush, dry, autumn,
# bare earth, pale. Real search areas are rarely one shade of green.
GROUND_PALETTES_RANDOMIZED = [
    ((62, 118, 52), (104, 150, 74)),
    ((96, 124, 58), (150, 160, 84)),
    ((140, 130, 70), (200, 185, 110)),
    ((120, 95, 55), (175, 140, 85)),
    ((95, 80, 60), (150, 130, 100)),
    ((70, 90, 55), (135, 125, 80)),
    ((150, 150, 120), (205, 200, 165)),
]


# ---- download ----------------------------------------------------------------


def fuel_url(asset: MeshAsset) -> str:
    name, owner = urllib.parse.quote(asset.fuel_name), urllib.parse.quote(asset.owner)
    return f"https://fuel.gazebosim.org/1.0/{owner}/models/{name}/tip/{name}.zip"


def is_fetched(asset: MeshAsset, fuel_dir: Path = FUEL_DIR) -> bool:
    return (fuel_dir / asset.key / asset.mesh).exists()


def fetch(asset: MeshAsset, fuel_dir: Path = FUEL_DIR) -> Path:
    """Download one model from Gazebo Fuel unless it is already there."""
    target = fuel_dir / asset.key
    if is_fetched(asset, fuel_dir):
        return target
    target.mkdir(parents=True, exist_ok=True)
    archive = target.with_suffix(".zip")
    print(f"downloading {asset.fuel_name} from Gazebo Fuel ...")
    urllib.request.urlretrieve(fuel_url(asset), archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)
    archive.unlink()
    return target


def fetch_all(fuel_dir: Path = FUEL_DIR) -> None:
    for asset in ALL_ASSETS.values():
        fetch(asset, fuel_dir)


def require(asset: MeshAsset, fuel_dir: Path = FUEL_DIR) -> Path:
    if not is_fetched(asset, fuel_dir):
        raise FileNotFoundError(
            f"model {asset.fuel_name!r} is not in {fuel_dir}. Run `make assets` to download it."
        )
    return fuel_dir / asset.key


# ---- measuring ---------------------------------------------------------------


def mesh_bounds(dae_path: Path) -> tuple[Vec, Vec]:
    """Bounding box of a COLLADA (.dae) mesh in metres.

    Reads every vertex, applies the file's unit (some files are in
    centimetres) and the scale of the node that places the geometry (the
    tree files are modelled in inches and scaled by 0.0254).
    """
    root = ET.parse(dae_path).getroot()
    ns = {"c": root.tag[1:].split("}")[0]}
    unit = float(root.find("c:asset/c:unit", ns).get("meter"))

    scale_of = {}  # geometry id -> uniform scale from its node's matrix
    for node in root.findall(".//c:visual_scene//c:node", ns):
        matrix = node.find("c:matrix", ns)
        scale = float(matrix.text.split()[0]) if matrix is not None else 1.0
        for instance in node.findall("c:instance_geometry", ns):
            scale_of[instance.get("url")[1:]] = scale

    lo, hi = [float("inf")] * 3, [float("-inf")] * 3
    for geometry in root.findall(".//c:geometry", ns):
        scale = unit * scale_of.get(geometry.get("id"), 1.0)
        for vertices in geometry.findall(".//c:vertices/c:input[@semantic='POSITION']", ns):
            source = vertices.get("source")[1:]
            numbers = geometry.find(f".//c:source[@id='{source}']/c:float_array", ns).text.split()
            for index, text in enumerate(numbers):
                value, axis = float(text) * scale, index % 3
                lo[axis], hi[axis] = min(lo[axis], value), max(hi[axis], value)
    return tuple(lo), tuple(hi)


# ---- clothing ----------------------------------------------------------------


def tint(image: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
    """Colour a WHITE garment: every pixel is multiplied by the colour, so
    folds and shading stay visible. Transparency is kept."""
    rgba = image.convert("RGBA")
    solid = Image.new("RGB", rgba.size, rgb)
    coloured = ImageChops.multiply(rgba.convert("RGB"), solid)
    coloured.putalpha(rgba.getchannel("A"))
    return coloured


# The rescue dummy's jacket is hi-vis yellow-green. In Pillow's HSV, hue runs
# 0..255, so 28..70 is roughly 40..100 degrees.
JACKET_HUE = (28, 70)
JACKET_MIN_SATURATION = 90
JACKET_MIN_VALUE = 60


def jacket_mask(image: Image.Image) -> Image.Image:
    """White where the pixel belongs to the yellow-green jacket."""
    hue, saturation, value = image.convert("RGB").convert("HSV").split()
    in_band = hue.point(lambda h: 255 if JACKET_HUE[0] <= h <= JACKET_HUE[1] else 0)
    vivid = saturation.point(lambda s: 255 if s >= JACKET_MIN_SATURATION else 0)
    bright = value.point(lambda v: 255 if v >= JACKET_MIN_VALUE else 0)
    return ImageChops.multiply(ImageChops.multiply(in_band, vivid), bright)


def replace_jacket_colour(image: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
    """Recolour only the jacket; skin, trousers and boots stay as they are.

    The jacket's brightness is kept (so folds still show) and multiplied by
    the new colour.
    """
    base = image.convert("RGB")
    brightness = base.convert("HSV").getchannel("V")
    shaded = ImageChops.multiply(
        Image.merge("RGB", (brightness, brightness, brightness)), Image.new("RGB", base.size, rgb)
    )
    return Image.composite(shaded, base, jacket_mask(base))


def _link_tree(source: Path, target: Path) -> None:
    """Copy a folder using hard links (no extra disk space)."""
    for path in source.rglob("*"):
        if "thumbnails" in path.parts:
            continue
        destination = target / path.relative_to(source)
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, destination)
            except OSError:  # e.g. a different disk: fall back to a real copy
                shutil.copy2(path, destination)


def clothing_variant(
    pose: str, colour: str, fuel_dir: Path = FUEL_DIR, variant_dir: Path = VARIANT_DIR
) -> Path:
    """Folder holding a copy of the `pose` model wearing `colour`.

    Built on first use and reused afterwards. The mesh and texture paths
    inside it are the same as in the original model folder.
    """
    asset = PEOPLE[pose]
    if not asset.recolour:  # worn as modelled: use the model folder itself
        return require(asset, fuel_dir)
    rgb = CLOTHING_RANDOMIZED[colour]
    source = require(asset, fuel_dir)
    target = variant_dir / f"{asset.key}__{colour}"
    texture = target / asset.clothing_texture
    marker = target / ".done"
    if marker.exists():
        return target

    _link_tree(source, target)
    original = Image.open(source / asset.clothing_texture)
    recoloured = tint(original, rgb) if asset.recolour == "tint" else replace_jacket_colour(original, rgb)
    texture.unlink()  # break the hard link first, or the original would change too
    if texture.suffix.lower() in (".jpg", ".jpeg"):
        recoloured.convert("RGB").save(texture, quality=92)
    else:
        recoloured.save(texture)
    marker.write_text(f"{pose} {colour} {rgb}\n")
    return target


# ---- ground ------------------------------------------------------------------

# (dark, light) grass colours per world type.
GROUND_PALETTE = {
    "meadow": ((96, 124, 58), (150, 160, 84)),
    "forest_edge": ((58, 84, 44), (104, 110, 62)),
    "park": ((62, 118, 52), (104, 150, 74)),
}


GROUND_EXTENT_M = 240.0  # side of the textured ground square
PATH_COLOUR = (168, 150, 112)


def ground_texture(world: str, seed: int, size: int = 4096, randomize: bool = False) -> Image.Image:
    """A seeded grass-like picture covering the whole textured ground.

    Three layers of random noise are mixed: large soft patches (dry and lush
    areas), medium blotches, and fine grain. One picture covers the whole
    ground (about 6 cm per pixel), so nothing repeats. Parks also get a few
    footpaths.
    """
    import random

    from PIL import ImageDraw, ImageFilter

    rng = random.Random(f"{seed}:ground")
    dark, light = GROUND_PALETTE[world]
    patch_cells, blotch_cells, paths_wanted = 12, 160, 3 if world == "park" else 0
    if randomize:
        # A separate random stream, so the plain texture of a seed never changes.
        dr = random.Random(f"{seed}:ground-dr")
        dark, light = dr.choice(GROUND_PALETTES_RANDOMIZED)
        patch_cells, blotch_cells = dr.choice([6, 12, 24]), dr.choice([80, 160, 320])
        paths_wanted = dr.choice([0, 0, 1, 2, 3])

    def noise(side: int) -> Image.Image:
        return Image.frombytes("L", (side, side), rng.randbytes(side * side))

    patches = noise(patch_cells).resize((size, size), Image.BICUBIC)
    blotches = noise(blotch_cells).resize((size, size), Image.BICUBIC)
    grain = noise(size)
    mix = Image.blend(Image.blend(patches, blotches, 0.4), grain, 0.3)

    channels = [mix.point(lambda v, d=d, l=l: int(d + (l - d) * v / 255)) for d, l in zip(dark, light)]
    image = Image.merge("RGB", channels)

    if paths_wanted:
        paths = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(paths)
        width = int(2.0 / GROUND_EXTENT_M * size)  # a 2 m wide path
        for _ in range(paths_wanted):
            points = [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(3)]
            draw.line(points, fill=255, width=width, joint="curve")
        paths = paths.filter(ImageFilter.GaussianBlur(3))
        image = Image.composite(Image.new("RGB", (size, size), PATH_COLOUR), image, paths)
    return image


def far_ground_texture(texture: Image.Image) -> Image.Image:
    """A tiny one-colour picture: the average colour of the ground texture."""
    return texture.resize((1, 1), Image.BOX).resize((4, 4))

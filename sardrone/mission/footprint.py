"""How much ground the tilted camera usefully sees.

Picture the camera at height h, tilted down by `pitch`. The middle of the
image looks at the ground a distance h / tan(pitch) AHEAD of the drone, and
that point is h / sin(pitch) away along the line of sight (the "slant
range"). Across the image, the view is that slant range times
tan(hfov / 2) wide on each side.

That width is the SWATH: the strip of ground the drone sweeps as it flies
forward. Lanes are spaced by the swath, so it decides how long a search
takes. There is a catch: far away, a person is only a few pixels tall and
cannot be detected. So the swath is also limited to where a person is still
at least `min_person_px` pixels tall.
"""

import math
from dataclasses import dataclass

from sardrone.mission.projection import CameraModel

PERSON_HEIGHT_M = 1.7


@dataclass(frozen=True)
class Footprint:
    altitude_m: float
    pitch_deg: float
    centre_ground_m: float  # how far ahead the image centre looks
    centre_slant_m: float  # line-of-sight distance to that point
    near_ground_m: float  # ground distance seen at the BOTTOM of the image
    far_ground_m: float  # ... at the TOP (inf if the top is above the horizon)
    half_width_m: float  # usable swath half-width at the centre row
    limited_by: str  # "field_of_view" or "resolution"
    person_px_centre: float  # height in pixels of a person at the image centre
    person_px_edge: float  # ... at the edge of the usable swath

    @property
    def swath_m(self) -> float:
        return 2 * self.half_width_m


def person_pixels(model: CameraModel, slant_m: float, person_height_m: float = PERSON_HEIGHT_M) -> float:
    """Roughly how many pixels tall a person is at this distance.

    Size in pixels = focal length (in pixels) * real size / distance. This
    is the upper bound (a person seen side-on); from steeply above they look
    shorter.
    """
    return model.fy * person_height_m / slant_m


def footprint(
    model: CameraModel,
    altitude_m: float,
    min_person_px: float = 12.0,
    person_height_m: float = PERSON_HEIGHT_M,
) -> Footprint:
    pitch = math.radians(model.pitch_deg)
    if altitude_m <= 0:
        raise ValueError("altitude must be positive")
    if model.pitch_deg < 10.0:
        raise ValueError("a camera pitched less than 10 deg down mostly sees the horizon")

    half_v = math.radians(model.vfov_deg) / 2
    half_h = math.radians(model.hfov_deg) / 2
    centre_ground = altitude_m / math.tan(pitch)
    centre_slant = altitude_m / math.sin(pitch)

    low = pitch + half_v  # angle below horizontal of the image's bottom edge
    near = altitude_m / math.tan(low) if low < math.pi / 2 else -altitude_m * math.tan(low - math.pi / 2)
    high = pitch - half_v  # ... of the top edge; <= 0 means at or above the horizon
    far = altitude_m / math.tan(high) if high > 1e-6 else math.inf

    max_slant = model.fy * person_height_m / min_person_px
    if centre_slant > max_slant:
        raise ValueError(
            f"at {altitude_m:g} m a person in the middle of the image is only "
            f"{person_pixels(model, centre_slant, person_height_m):.1f} px tall, below the "
            f"required {min_person_px:g} px. Fly lower or accept smaller people."
        )
    by_view = centre_slant * math.tan(half_h)
    by_resolution = math.sqrt(max_slant**2 - centre_slant**2)
    half_width = min(by_view, by_resolution)
    edge_slant = math.hypot(centre_slant, half_width)
    return Footprint(
        altitude_m=altitude_m,
        pitch_deg=model.pitch_deg,
        centre_ground_m=centre_ground,
        centre_slant_m=centre_slant,
        near_ground_m=near,
        far_ground_m=far,
        half_width_m=half_width,
        limited_by="field_of_view" if by_view <= by_resolution else "resolution",
        person_px_centre=person_pixels(model, centre_slant, person_height_m),
        person_px_edge=person_pixels(model, edge_slant, person_height_m),
    )

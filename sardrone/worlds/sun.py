"""Time of day -> where the sun is and what colour the light has.

Two angles place the sun. ELEVATION is how high it is above the horizon
(90 deg = straight overhead). AZIMUTH is its compass bearing (0 = north,
90 = east). A low sun makes long shadows and warm light, which is what makes
"golden hour" and "dusk" images look different from "noon".
"""

import math
import random
from dataclasses import dataclass

Rgb = tuple[float, float, float]


@dataclass(frozen=True)
class SunSpec:
    time_of_day: str
    elevation_deg: float
    azimuth_deg: float
    direction: tuple[float, float, float]  # the way the light TRAVELS, in ENU
    diffuse: Rgb
    intensity: float
    ambient: Rgb
    sky: Rgb


# (elevation range, azimuth range, light colour, intensity, ambient, sky)
# A low sun lights flat ground only weakly (by the sine of its elevation),
# while real evening scenes stay bright from the sky. So the low-sun presets
# use a stronger light and more ambient; the first try without that rendered
# golden hour almost black.
_PRESETS = {
    "noon": ((55.0, 65.0), (150.0, 210.0), (1.0, 0.98, 0.94), 1.0, (0.45, 0.45, 0.45), (0.53, 0.70, 0.92)),
    "golden_hour": ((8.0, 14.0), (250.0, 280.0), (1.0, 0.76, 0.52), 2.6, (0.62, 0.52, 0.42), (0.86, 0.68, 0.52)),
    "dusk": ((2.0, 5.0), (280.0, 300.0), (0.62, 0.55, 0.70), 2.2, (0.34, 0.33, 0.42), (0.30, 0.31, 0.45)),
}  # fmt: skip


def light_direction(elevation_deg: float, azimuth_deg: float) -> tuple[float, float, float]:
    """Unit vector along which sunlight travels (from the sun to the ground).

    In ENU, the sun SITS at (cos(el)*sin(az), cos(el)*cos(az), sin(el));
    light travels the opposite way.
    """
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    return (-math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), -math.sin(el))


def shadow_length_m(height_m: float, elevation_deg: float) -> float:
    """Length of the shadow of a vertical object on flat ground."""
    return height_m / math.tan(math.radians(elevation_deg))


def sun_for(time_of_day: str, rng: random.Random) -> SunSpec:
    (el_lo, el_hi), (az_lo, az_hi), diffuse, intensity, ambient, sky = _PRESETS[time_of_day]
    elevation = round(rng.uniform(el_lo, el_hi), 2)
    azimuth = round(rng.uniform(az_lo, az_hi), 2)
    direction = tuple(round(v, 6) for v in light_direction(elevation, azimuth))
    return SunSpec(time_of_day, elevation, azimuth, direction, diffuse, intensity, ambient, sky)

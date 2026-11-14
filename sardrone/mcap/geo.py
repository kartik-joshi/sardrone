"""Small-area geodesy: metres from a home point <-> latitude/longitude.

For a flight that stays within a few hundred metres of home, the Earth is
flat enough that we can treat one degree of latitude as a fixed number of
metres, and scale longitude by cos(latitude). The error over 1 km is a few
millimetres, far below GPS noise.
"""

import math

EARTH_RADIUS_M = 6_378_137.0  # WGS-84 equatorial radius
METERS_PER_DEG_LAT = math.pi * EARTH_RADIUS_M / 180.0  # ~111_319.49 m


def enu_offset_to_latlon(
    home_lat_deg: float, home_lon_deg: float, east_m: float, north_m: float
) -> tuple[float, float]:
    """Latitude/longitude of a point `east_m`, `north_m` away from home."""
    lat = home_lat_deg + north_m / METERS_PER_DEG_LAT
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(home_lat_deg))
    lon = home_lon_deg + east_m / meters_per_deg_lon
    return lat, lon


def latlon_to_enu_offset(
    home_lat_deg: float, home_lon_deg: float, lat_deg: float, lon_deg: float
) -> tuple[float, float]:
    """Inverse of enu_offset_to_latlon. Returns (east_m, north_m)."""
    north_m = (lat_deg - home_lat_deg) * METERS_PER_DEG_LAT
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(home_lat_deg))
    east_m = (lon_deg - home_lon_deg) * meters_per_deg_lon
    return east_m, north_m

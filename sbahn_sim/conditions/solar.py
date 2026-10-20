"""Solar position (NOAA simplified algorithm) and clear-sky sun model.

World convention of the scene: +Y = north, +X = east, +Z = up. Azimuth is
measured clockwise from north (0 = N, 90 = E), which is also what the Blender
Sky Texture's `sun_rotation` expects (checked empirically on Blender 5.0).
"""
from __future__ import annotations

import datetime as _dt
import math

import numpy as np

# default site: Munich S-Bahn trunk (48.14 N, 11.56 E), CET/CEST
DEFAULT_LAT = 48.14
DEFAULT_LON = 11.56


def parse_time_of_day(t) -> float:
    """'HH:MM' / 'HH:MM:SS' / float hours -> float hours."""
    if isinstance(t, str):
        parts = [float(p) for p in t.split(":")]
        return parts[0] + (parts[1] if len(parts) > 1 else 0) / 60 + (parts[2] if len(parts) > 2 else 0) / 3600
    return float(t)


def utc_offset_hours(date: _dt.date) -> float:
    """Central European time with EU summer time (last Sunday Mar .. last Sunday Oct)."""
    def last_sunday(y, m):
        d = _dt.date(y, m + 1, 1) - _dt.timedelta(days=1) if m < 12 else _dt.date(y, 12, 31)
        return d - _dt.timedelta(days=(d.weekday() + 1) % 7)
    return 2.0 if last_sunday(date.year, 3) <= date < last_sunday(date.year, 10) else 1.0


def sun_position(date, local_hours: float, lat=DEFAULT_LAT, lon=DEFAULT_LON, utc_offset=None):
    """Return (elevation_deg, azimuth_deg) of the sun for local clock time."""
    if isinstance(date, str):
        date = _dt.date.fromisoformat(date)
    if utc_offset is None:
        utc_offset = utc_offset_hours(date)
    doy = date.timetuple().tm_yday
    g = 2 * math.pi / 365.0 * (doy - 1 + (local_hours - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = local_hours * 60 + eqtime + 4 * lon - 60 * utc_offset          # true solar time, minutes
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cos_zen = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    el = 90 - math.degrees(zen)
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(la) - math.tan(decl) * math.cos(la))) + 180
    return el, az % 360


def airmass(el_deg: float) -> float:
    """Kasten-Young air mass; large but finite at the horizon."""
    el = max(el_deg, -0.5)
    z = 90.0 - el
    return 1.0 / (math.cos(math.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)


def sun_direct(el_deg: float):
    """Relative direct-normal irradiance (1.0 at 38 deg elevation) and RGB tint."""
    if el_deg <= -0.8:
        return 0.0, (1.0, 0.45, 0.2)
    am = airmass(el_deg)
    dni = 0.7 ** (am ** 0.678)
    ref = 0.7 ** (airmass(38.0) ** 0.678)
    tau = np.array([0.045, 0.085, 0.19])          # Rayleigh + aerosol optical depth per RGB channel
    tint = np.exp(-tau * am)
    tint = tint / tint.max()
    fade = min(1.0, (el_deg + 0.8) / 1.6)          # disc sinking below the horizon
    return float(dni / ref * fade), tuple(float(c) for c in tint)


# Mean radiance of Blender's MULTIPLE_SCATTERING sky (strength 1, sun disc on) over the sphere,
# measured in Blender 5.0.1; used to keep overcast / twilight brightness consistent with the sky.
_SKY_MEAN = np.array([[-18, 0.00002], [-12, 0.0012], [-10, 0.003], [-6, 0.0108], [-4, 0.0264], [-2, 0.0911],
                      [0, 0.284], [2, 0.587], [10, 1.70], [20, 2.25], [38, 2.45], [90, 2.5]])


def sky_mean(el_deg: float) -> float:
    return float(np.exp(np.interp(el_deg, _SKY_MEAN[:, 0], np.log(_SKY_MEAN[:, 1]))))


def sun_vector(el_deg: float, az_deg: float):
    el, az = math.radians(el_deg), math.radians(az_deg)
    return (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))

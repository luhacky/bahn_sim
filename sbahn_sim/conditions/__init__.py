"""Weather, lighting and day/night for the S-Bahn cab-view scene.

One entry point, used by the snapshot renderer:

    from sbahn_sim.conditions import apply_conditions, PRESETS
    eff = apply_conditions(bpy.context.scene, {"preset": "fog_dense", "time_of_day": "07:10"})
    json.dump(eff, log)        # effective values, JSON-serialisable

Call it after the .blend is loaded; it can be called again at any time with
new conditions (it rebuilds the world, fog material and rain setup, and
reuses the objects it created). Everything it creates lives in the
'Lighting' collection or is parented to CabCamera / Car_0, so place_train(s)
moves it along. Between calls, only frame-dependent things change on their
own: rain streak positions re-seed with scene.frame_current, and the wiper
follows cond["time_s"] (or frame / fps).

Condition keys (all optional; a preset fills defaults, explicit keys win):

  preset              name from PRESETS
  time_of_day         "HH:MM" or hours (local CET/CEST); with date/latitude/longitude -> sun position
  date                "YYYY-MM-DD" (default 2026-06-21)
  latitude, longitude default Munich (48.14, 11.56)
  sun_elevation_deg   overrides the computed elevation (negative = below horizon)
  sun_azimuth_deg     overrides the computed azimuth (0 = N = world +Y, 90 = E = world +X)
  sun_azimuth_rel_deg azimuth relative to the train heading (0 = sun straight ahead,
                      i.e. glare) - needs the train placed first
  cloud_cover         0..1 (1 = overcast, no direct sun)
  fog_visibility_m    meteorological visibility in the ground fog (None = no fog)
  visibility_m        alias of fog_visibility_m
  fog_density         alternative to fog_visibility_m: extinction coefficient in 1/m (NOT 0..1;
                      0.026 = 150 m visibility, 0.0065 = 600 m)

Unknown keys are ignored and listed under "ignored_keys" in the result.
Objects that must not block line-of-sight checks (fog domain, rain streaks,
windscreen) carry obj["sbahn_non_occluding"] = True. The wiper arm does occlude.
  fog_top_m           fog layer thickness above the rail (default 80 m)
  fog_patchiness      0..1, fog banks fixed in the world (default 0.35 when fog)
  haze_visibility_m   background visibility (default 14000 clear, 8000 overcast/rain)
  rain_rate_mm_h      0 = dry; 0.5 drizzle, 2.5 light, 8 moderate, 25 heavy, 50 cloudburst
  wetness             0..1 surface wetness (default from rain_rate, 0.35 after fog)
  wind_ms             crosswind for the rain streaks (m/s, + = from the left)
  train_speed_kmh     only used for rain streak direction/length (default 100)
  shutter_s           exposure time for streak length (default 1/120)
  windscreen          "auto" | True | False   droplets / streaks on the glass (auto: when rain > 0)
  wiper               "auto" | "on" | "off" | "parked"  (auto: on when rain >= 1 mm/h)
  wiper_period_s      full sweep period (default 1.6 s heavy, 3 s light rain)
  time_s              clock for the wiper phase (default frame / fps)
  headlights          True | False (default True: DB runs with the Spitzensignal always lit)
  headlight_mode      "auto" | "dipped" | "high"  (auto: high at night without fog)
  headlight_power     scale factor (1.0)
  exposure_ev         "auto" or float EV added to the view transform
  seed                int, varies clouds / fog banks / droplet layout
"""
from __future__ import annotations

import copy
import datetime as _dt
import math

import bpy

from . import fog, headlights, rain, sky, solar

__all__ = ["apply_conditions", "PRESETS", "resolve"]

PRESETS = {
    "clear_noon":     dict(time_of_day="12:30", cloud_cover=0.05),
    "clear_morning":  dict(time_of_day="07:30", cloud_cover=0.15),
    "partly_cloudy":  dict(time_of_day="14:00", cloud_cover=0.45),
    "overcast":       dict(time_of_day="13:00", cloud_cover=1.0),
    "low_sun_glare":  dict(time_of_day="19:45", cloud_cover=0.1, sun_azimuth_rel_deg=0.0, sun_elevation_deg=4.0),
    "sunset":         dict(time_of_day="21:10", cloud_cover=0.25),
    "twilight":       dict(time_of_day="21:45", cloud_cover=0.2),
    "night_clear":    dict(time_of_day="23:30", cloud_cover=0.1),
    "night_overcast": dict(time_of_day="01:00", cloud_cover=1.0),
    "fog_light":      dict(time_of_day="09:00", cloud_cover=0.9, fog_visibility_m=600.0, fog_top_m=60.0),
    "fog_dense":      dict(time_of_day="07:30", cloud_cover=1.0, fog_visibility_m=150.0, fog_top_m=80.0),
    "night_fog":      dict(time_of_day="23:00", cloud_cover=1.0, fog_visibility_m=250.0, fog_top_m=60.0),
    "rain_light":     dict(time_of_day="15:00", cloud_cover=1.0, rain_rate_mm_h=2.5),
    "rain_heavy":     dict(time_of_day="16:00", cloud_cover=1.0, rain_rate_mm_h=25.0),
    "night_rain":     dict(time_of_day="22:30", cloud_cover=1.0, rain_rate_mm_h=8.0),
}

DEFAULTS = dict(date="2026-06-21", latitude=solar.DEFAULT_LAT, longitude=solar.DEFAULT_LON,
                time_of_day="12:30", cloud_cover=0.1, fog_visibility_m=None, fog_top_m=80.0,
                rain_rate_mm_h=0.0, wind_ms=0.0, train_speed_kmh=100.0, shutter_s=1 / 120,
                windscreen="auto", wiper="auto", headlights=True, headlight_mode="auto",
                headlight_power=1.0, exposure_ev="auto", seed=0)

_KNOWN = set(DEFAULTS) | {"preset", "sun_elevation_deg", "sun_azimuth_deg", "sun_azimuth_rel_deg", "fog_density",
                          "fog_patchiness", "haze_visibility_m", "wetness", "wiper_period_s", "time_s"}

# Exposure: a camera with auto exposure, but only partly compensating, so night stays night.
EV_REF_ILLUMINANCE = 2.45      # horizontal illuminance (Blender units) of the 38 deg clear-sky reference
EV_COMPENSATION = 0.62
EV_MAX = 4.0
EV_BIAS = 0.5          # lighting overhaul: real cab cameras meter for the darker ground, not the sky


def resolve(cond: dict) -> dict:
    """Merge defaults, preset and explicit keys. Pure Python (no bpy), usable to plan a run."""
    cond = dict(cond or {})
    if "visibility_m" in cond and "fog_visibility_m" not in cond:      # alias used by the dataset renderer
        cond["fog_visibility_m"] = cond.pop("visibility_m")
    unknown = sorted(set(cond) - _KNOWN)
    for k in unknown:                                                  # tolerated, reported in the result
        cond.pop(k)
    p = copy.deepcopy(DEFAULTS)
    if cond.get("preset"):
        if cond["preset"] not in PRESETS:
            raise KeyError(f"unknown preset {cond['preset']!r}; have {sorted(PRESETS)}")
        p.update(PRESETS[cond["preset"]])
    p.update({k: v for k, v in cond.items() if v is not None or k in ("fog_visibility_m",)})
    p["_ignored_keys"] = unknown
    return p


def _train_heading_deg(scene):
    """Azimuth (0 = N, clockwise) the cab camera looks along."""
    cam = scene.camera
    if cam is None:
        return 0.0
    fwd = cam.matrix_world.to_3x3() @ __import__("mathutils").Vector((0, 0, -1))
    return math.degrees(math.atan2(fwd.x, fwd.y)) % 360


def apply_conditions(scene, cond: dict) -> dict:
    """Apply time of day, sky, fog, rain, wetness and headlights. Returns effective parameters."""
    scene = scene or bpy.context.scene
    p = resolve(cond)
    eff = dict(preset=p.get("preset"), seed=int(p["seed"]))
    if p["_ignored_keys"]:
        eff["ignored_keys"] = p["_ignored_keys"]

    # ---- sun --------------------------------------------------------------------
    tod = solar.parse_time_of_day(p["time_of_day"])
    el, az = solar.sun_position(_dt.date.fromisoformat(p["date"]), tod, p["latitude"], p["longitude"])
    if p.get("sun_elevation_deg") is not None:
        el = float(p["sun_elevation_deg"])
    if p.get("sun_azimuth_rel_deg") is not None:
        az = (_train_heading_deg(scene) + float(p["sun_azimuth_rel_deg"])) % 360
    elif p.get("sun_azimuth_deg") is not None:
        az = float(p["sun_azimuth_deg"]) % 360
    heading = _train_heading_deg(scene)
    rel = (az - heading + 180) % 360 - 180
    eff.update(date=p["date"], time_of_day=f"{int(tod):02d}:{int(round((tod % 1) * 60)) % 60:02d}",
               sun_elevation_deg=round(el, 3), sun_azimuth_deg=round(az, 3),
               sun_azimuth_rel_train_deg=round(rel, 2), train_heading_deg=round(heading, 2))
    if el > 6:
        light = "day"
    elif el > -6:
        light = "twilight" if el <= 0 else "low_sun"
    else:
        light = "night"
    eff["daylight"] = light

    # ---- rain (decides cloud / haze defaults) --------------------------------------
    rr = max(0.0, float(p["rain_rate_mm_h"]))
    cc = float(p["cloud_cover"])
    if rr > 0 and "cloud_cover" not in (cond or {}):
        cc = max(cc, 0.95)
    cc = min(max(cc, 0.0), 1.0)
    eff["cloud_cover"] = cc
    skyv = sky.build_world(scene, dict(sun_elevation_deg=el, sun_azimuth_deg=az, cloud_cover=cc, seed=p["seed"]))
    eff.update({k: v for k, v in skyv.items() if k != "horizontal_illuminance_rel"})

    # ---- fog / haze / rain extinction ----------------------------------------------
    if p.get("fog_density") is not None:
        sig_fog = float(p["fog_density"])
    elif p.get("fog_visibility_m"):
        sig_fog = fog.KOSCHMIEDER / float(p["fog_visibility_m"])
    else:
        sig_fog = 0.0
    haze_v = p.get("haze_visibility_m") or (8000.0 if (cc > 0.8 or rr > 0) else CLEAR_HAZE_M)
    rain_ext = 0.25e-3 * rr ** 0.63 if rr > 0 else 0.0           # 1/m, empirical rain extinction
    patch = p.get("fog_patchiness")
    patch = (0.35 if sig_fog > 0 else 0.0) if patch is None else float(patch)
    fogv = fog.build_fog(scene, dict(fog_density=sig_fog, haze_density=fog.KOSCHMIEDER / float(haze_v),
                                     rain_extinction=rain_ext, fog_top_m=float(p["fog_top_m"]),
                                     fog_patchiness=patch, seed=p["seed"], sky_radiance=skyv["sky_radiance_mean"],
                                     eye_height_m=float(scene.get("sbahn_cab_eye_h", 2.45))))
    eff.update(fogv)
    eff["haze_visibility_m"] = float(haze_v)
    eff["haze_density_per_m"] = round(fog.KOSCHMIEDER / float(haze_v), 6)
    eff["rain_extinction_per_m"] = round(rain_ext, 6)

    # ---- rain, windscreen, wiper, wet surfaces ---------------------------------------
    rainv = rain.build_rain(scene, dict(p, rain_rate_mm_h=rr, fog_density=sig_fog))
    eff.update(rainv)

    # ---- headlights ----------------------------------------------------------------------
    mode = p["headlight_mode"]
    if mode == "auto":
        mode = "high" if (el < -4 and sig_fog < fog.KOSCHMIEDER / 1000.0 and rr < 10) else "dipped"
    eff.update(headlights.build_headlights(scene, dict(headlights=p["headlights"], headlight_mode=mode,
                                                       headlight_power=p["headlight_power"])))

    # ---- exposure --------------------------------------------------------------------------
    illum = skyv["horizontal_illuminance_rel"]
    if p["exposure_ev"] == "auto":
        ev = EV_COMPENSATION * math.log2(EV_REF_ILLUMINANCE / max(illum, 1e-6))
        ev = min(max(ev + EV_BIAS, -1.0), EV_MAX)
    else:
        ev = float(p["exposure_ev"])
    scene.view_settings.exposure = ev
    eff["exposure_ev"] = round(ev, 3)
    eff["ambient_illuminance_rel"] = round(illum, 6)
    _engine_settings(scene, eff)
    scene["sbahn_conditions"] = __import__("json").dumps(eff)
    return eff


CLEAR_HAZE_M = 14000.0         # summer visibility in a German lowland; 25 km gave no aerial perspective
VIEW_LOOK = "AgX - Medium High Contrast"   # closer to a real camera's tone curve than plain AgX


def _engine_settings(scene, eff):
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = VIEW_LOOK
    except TypeError:
        pass
    eff["view_look"] = scene.view_settings.look
    if scene.render.engine == "CYCLES":
        c = scene.cycles
        for attr, val in (("volume_bounces", 0), ("volume_step_rate", 4.0 if eff["volume"] == "heterogeneous" else 1.0),
                          ("volume_max_steps", 256), ("transparent_max_bounces", 16)):
            if hasattr(c, attr):
                setattr(c, attr, val)
    else:
        ee = getattr(scene, "eevee", None)
        if ee is not None:
            for attr, val in (("volumetric_start", 0.5), ("volumetric_end", 2000.0), ("volumetric_tile_size", "4"),
                              ("volumetric_samples", 128), ("use_volumetric_shadows", True)):
                if hasattr(ee, attr):
                    try:
                        setattr(ee, attr, val)
                    except Exception:
                        pass

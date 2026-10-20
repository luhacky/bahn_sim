"""Sky, clouds and sun lamp (time of day + cloud cover).

World shader (rebuilt on every apply):

    clear   = Sky Texture (MULTIPLE_SCATTERING) for the sun position; the sun
              disc is visible to camera rays only, lighting uses the Sun lamp
    night   = dark blue zenith with a faint warm light-pollution horizon glow
    clouds  = fBm noise projected on a cloud deck (dir.xy / dir.z), thresholded
              by cloud_cover; fully overcast -> CIE overcast gradient
    colour  = clouds over (clear + night), times one strength

The Sun lamp carries the direct light (energy from air mass, tint from
per-channel optical depth, dimmed and softened by cloud cover).
"""
from __future__ import annotations

import math

import bpy
from mathutils import Vector

from . import solar
from .nodes import NodeBuilder

SKY_SCALE = 0.04          # same scale as the build's placeholder sky (keeps material brightness calibrated)
SUN_ENERGY_REF = 3.5      # Sun lamp energy at 38 deg, clear sky (build calibration)
# Lighting overhaul (2026-10-02): the clear sky was about half as bright as a real summer sky relative to
# sunlit ground (sky near the horizon should be roughly as bright as sunlit ballast), which made the sky
# look like dusk and left shaded areas and forest interiors nearly black. SKY_GAIN brightens the dome and
# its fill light together; diffuse / direct on the ground is then ~0.25, typical of a hazy central-European day.
SKY_GAIN = 2.4
# Blender < 5 has no MULTIPLE_SCATTERING sky and falls back to NISHITA, measured in Blender 4.4.3 at
# 0.75 to 0.8 of the MULTIPLE_SCATTERING mean radiance by day (solar._SKY_MEAN); compensate so both match.
NISHITA_COMP = 1.3
DUST = 2.6               # more aerosol: paler, brighter horizon like a typical summer day (was 1.6)


def sky_scale() -> float:
    types = [e.identifier for e in bpy.types.ShaderNodeTexSky.bl_rna.properties["sky_type"].enum_items]
    return SKY_SCALE * SKY_GAIN * (1.0 if "MULTIPLE_SCATTERING" in types else NISHITA_COMP)


NIGHT_ZENITH = (0.012, 0.016, 0.028)
NIGHT_HORIZON = (0.05, 0.035, 0.022)   # sodium / LED light pollution of a German suburb
NIGHT_LEVEL = 4.0e-4       # radiance of the night sky before exposure


def _lighting_collection():
    col = bpy.data.collections.get("Lighting")
    if col is None:
        col = bpy.data.collections.new("Lighting")
        bpy.context.scene.collection.children.link(col)
    return col


def _sky_node(g: NodeBuilder, el, az, disc: bool):
    sky = g.node("ShaderNodeTexSky")
    types = [e.identifier for e in sky.bl_rna.properties["sky_type"].enum_items]
    sky.sky_type = "MULTIPLE_SCATTERING" if "MULTIPLE_SCATTERING" in types else ("NISHITA" if "NISHITA" in types else types[0])
    sky.sun_elevation = math.radians(el)
    sky.sun_rotation = math.radians(az)
    for attr, val in (("sun_disc", disc), ("sun_intensity", 0.6), ("altitude", 300.0),
                      ("air_density", 1.0), ("aerosol_density", 1.6), ("dust_density", DUST), ("ozone_density", 1.0)):
        if hasattr(sky, attr):
            setattr(sky, attr, val)
    return sky.outputs["Color"]


def build_world(scene, p: dict) -> dict:
    """p: sun_elevation_deg, sun_azimuth_deg, cloud_cover, seed, haze_tint. Returns effective values."""
    el, az, cc = p["sun_elevation_deg"], p["sun_azimuth_deg"], p["cloud_cover"]
    world = bpy.data.worlds.get("SBahnWorld") or bpy.data.worlds.new("SBahnWorld")
    scene.world = world
    try:
        world.use_nodes = True
    except Exception:
        pass
    g = NodeBuilder(world.node_tree, clear=True)
    out = g.node("ShaderNodeOutputWorld")
    tc = g.node("ShaderNodeTexCoord")
    direction = g.vmath("NORMALIZE", tc.outputs["Generated"])
    dx, dy, dz = g.separate(direction)
    up = g.math("MAXIMUM", dz, 0.0)

    # --- clear sky: disc only for camera rays --------------------------------------
    lp = g.node("ShaderNodeLightPath")
    sky_light = _sky_node(g, el, az, disc=False)
    sky_cam = _sky_node(g, el, az, disc=True)
    clear = g.mix_color(lp.outputs["Is Camera Ray"], sky_light, sky_cam)
    clear = g.mix_color(1.0, clear, g.rgb((sky_scale(),) * 3), blend="MULTIPLY")

    # --- night sky (always added; negligible at day) ----------------------------------
    horizon_w = g.math("POWER", g.sub(1.0, up), 6.0)
    night = g.mix_color(horizon_w, g.rgb(NIGHT_ZENITH), g.rgb(NIGHT_HORIZON))
    night = g.mix_color(1.0, night, g.rgb((NIGHT_LEVEL / 0.02,) * 3), blend="MULTIPLY")
    clear = g.mix_color(1.0, clear, night, blend="ADD")

    # --- clouds -------------------------------------------------------------------
    sky_mean = solar.sky_mean(el) * sky_scale()
    # overcast radiance: ~30 % of the clear-sky global light by day, darker than clear at twilight
    ov = max(0.30 * max(math.sin(math.radians(el)), 0.0) / math.pi * 2.2, 0.45 * sky_mean) + NIGHT_LEVEL * 0.6
    _, sun_tint = solar.sun_direct(max(el, 0.5))
    warm = max(0.0, min(1.0, (12.0 - el) / 12.0)) * 0.5
    # broken clouds are sun-lit and brighter than the blue sky; a closed deck is the overcast level
    w_ov = cc ** 2
    lit = max(ov, 2.6 * sky_mean)
    cloud_level = w_ov * ov + (1 - w_ov) * lit
    cloud_rgb = tuple(cloud_level * ((1 - warm) + warm * t) for t in sun_tint)
    cie = g.maprange(up, 0.0, 1.0, 1.0 / 1.4, 3.0 / 1.4)          # (1 + 2 sin h) / 3, normalised
    overcast = g.mix_color(1.0, g.rgb(cloud_rgb), g.combine(cie, cie, cie), blend="MULTIPLY")
    if cc > 0.0:
        seed = float(p.get("seed", 0))
        deck = g.add(up, 0.06)
        uv = g.combine(g.div(dx, deck), g.div(dy, deck), seed * 7.31)
        nz = g.node("ShaderNodeTexNoise", noise_dimensions="3D")
        g.set(nz.inputs["Vector"], uv)
        nz.inputs["Scale"].default_value = 1.1
        nz.inputs["Detail"].default_value = 8.0
        nz.inputs["Roughness"].default_value = 0.6
        n01 = g.maprange(nz.outputs["Fac"], 0.32, 0.68, 0.0, 1.0)
        thr = 1.0 - cc
        mask = g.smoothstep(thr - 0.12, thr + 0.12, n01) if cc < 0.97 else g.value(1.0)
        # towards the horizon clouds merge into a sheet: blend the mask to the mean coverage
        hz = g.smoothstep(0.0, 0.25, up)
        mask = g.add(g.mul(mask, hz), g.mul(g.sub(1.0, hz), min(1.0, cc * 1.15)))
        # below the horizon keep the clear sky (hidden by terrain anyway)
        mask = g.mul(mask, g.smoothstep(-0.02, 0.01, dz))
        shade = g.maprange(nz.outputs["Fac"], 0.3, 0.7, 1.25, 0.8)        # thin edges brighter, cores darker
        cloud_col = g.mix_color(1.0, overcast, g.combine(shade, shade, shade), blend="MULTIPLY")
        color = g.mix_color(mask, clear, cloud_col)
    else:
        color = clear
    bg = g.node("ShaderNodeBackground")
    g.set(bg.inputs["Color"], color)
    bg.inputs["Strength"].default_value = 1.0
    g.link(bg.outputs[0], out.inputs["Surface"])

    # --- sun lamp -------------------------------------------------------------------
    col = _lighting_collection()
    so = bpy.data.objects.get("Sun")
    if so is None or so.type != "LIGHT":
        so = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
        col.objects.link(so)
    rel, tint = solar.sun_direct(el)
    sun_factor = max(0.0, 1.0 - cc ** 2.2) if cc < 0.95 else 0.0
    energy = SUN_ENERGY_REF * rel * sun_factor
    so.data.energy = energy
    so.data.color = tint
    so.data.angle = math.radians(0.53 + 12.0 * cc)
    so.hide_render = energy <= 1e-4
    so.rotation_euler = Vector(solar.sun_vector(el, az)).to_track_quat("Z", "Y").to_euler()
    ambient = math.pi * (sky_mean * (1 - cc) + ov * cc) + math.pi * NIGHT_LEVEL
    horiz = energy * max(math.sin(math.radians(el)), 0.0) + ambient
    return dict(sun_energy=round(energy, 4), sun_color=[round(c, 3) for c in tint],
                sky_radiance_mean=round(sky_mean * (1 - cc) + ov * cc, 6),
                horizontal_illuminance_rel=horiz)

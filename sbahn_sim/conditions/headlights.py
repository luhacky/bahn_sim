"""Dreilicht-Spitzensignal: the three white head lights of the leading car.

Positions match the lamp discs modelled by train_rig on Car_0 (car-local:
x = +-1.0 m at 1.35 m above rail, centre lamp at 3.75 m; +Y forward).
Each lamp is a spot light parented to Car_0, so place_train() moves them.

    mode "dipped" (Abblendlicht): 44 x 15 deg elliptical cone, aimed 2.0 deg down
    mode "high"   (Fernlicht):    26 x 8 deg elliptical cone, aimed 0.2 deg up, ~4x the intensity

The cones are flattened (object scale on the spot light) like real head
lamps, so the beam lights the track far ahead instead of a hot spot on the
ballast right in front of the train.
    signal lamps glow (emission of the lamp_white faces) whenever the lights are on

Spot energy is calibrated so that, at night with high beam, the rails are
lit to roughly 150 to 250 m (German EBO / TSI LOC&PAS: >= 1 lx at 100 m
in high-beam; the renderer's exposure makes that visible).
"""
from __future__ import annotations

import math

import bpy
from mathutils import Matrix

LAMPS = (("HeadLight_L", -1.0, 1.35), ("HeadLight_R", 1.0, 1.35), ("HeadLight_Top", 0.0, 3.75))
MODES = {
    #          cone_deg (horizontal), vertical/horizontal, blend, tilt_down_deg, energy_W lower, energy_W top
    "dipped": (44.0, 0.34, 0.9, 2.0, 1500.0, 500.0),
    "high":   (26.0, 0.30, 0.8, -0.2, 6000.0, 2000.0),
}
LAMP_EMISSION = 40.0


def _car0():
    return bpy.data.objects.get("Car_0")


def build_headlights(scene, p: dict) -> dict:
    on = bool(p.get("headlights", True))
    mode = p.get("headlight_mode", "dipped")
    if mode not in MODES:
        raise ValueError(f"headlight_mode must be one of {list(MODES)}")
    car = _car0()
    cone, flat, blend, tilt, e_low, e_top = MODES[mode]
    scale = float(p.get("headlight_power", 1.0))
    col = bpy.data.collections.get("Lighting") or scene.collection
    car_len = float(scene.get("sbahn_car_len_m", 16.85))
    for name, x, z in LAMPS:
        ob = bpy.data.objects.get(name)
        if ob is None:
            ld = bpy.data.lights.new(name, "SPOT")
            ob = bpy.data.objects.new(name, ld)
            col.objects.link(ob)
        ld = ob.data
        ld.spot_size = math.radians(cone)
        ld.spot_blend = blend
        ld.energy = (e_top if "Top" in name else e_low) * scale
        ld.color = (1.0, 0.96, 0.90)             # LED / halogen mix, ~5000 K
        ld.shadow_soft_size = 0.08
        if hasattr(ld, "use_soft_falloff"):
            ld.use_soft_falloff = False
        ob.hide_render = not on
        if car is not None:
            ob.parent = car
            ob.matrix_parent_inverse = Matrix.Identity(4)
            # spot shines along local -Z: point it along car +Y, tilted down
            ob.matrix_basis = (Matrix.Translation((x, car_len / 2 + 0.08, z))
                               @ Matrix.Rotation(math.radians(90.0 - tilt), 4, "X")
                               @ Matrix.Diagonal((1.0, flat, 1.0, 1.0)))
    mat = bpy.data.materials.get("SB_lamp_white")          # the lamp faces on Car_0 (and the rear car)
    if mat is not None and mat.node_tree:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Emission Strength"].default_value = LAMP_EMISSION if on else 0.0
    return dict(headlights_on=on, headlight_mode=mode if on else "off", headlight_power=scale,
                headlight_cone_deg=cone if on else None)

"""Volumetric fog / haze / rain extinction in one camera-following volume.

Extinction coefficient (1/m) follows Koschmieder: sigma = 3.912 / V for a
meteorological visibility V (5 % contrast threshold). Blender's volume
density is in 1/scene-unit, so it takes sigma directly.

    sigma(p) = sigma_bg                                  haze + rain, homogeneous
             + sigma_fog * layer(h) * patches(p)         ground fog

    h        = height above the rail top under the camera (fog settles on the
               track and pools in lower terrain: layer = 1 below the rail level)
    layer    = 1 - smoothstep(0.6 * top, 1.4 * top, h)
    patches  = 1 + patchiness * (2 * fBm(p / 150 m) - 1)   fixed in the world, so
               the train drives into and out of banks

The domain is a large box parented to the cab camera; everything in the
density uses world coordinates, so the fog does not move with the train.
Headlights and the sun scatter in it (single scattering, Henyey-Greenstein
g = 0.75 for fog droplets), which gives the volumetric beams.
"""
from __future__ import annotations

import bpy
from mathutils import Matrix

from .nodes import NodeBuilder

KOSCHMIEDER = 3.912
DOMAIN_NAME = "FogDomain"
DOMAIN_HALF_M = 4500.0
MS_TOPUP = 0.55


def _domain(cam, half_h, shadow=False):
    ob = bpy.data.objects.get(DOMAIN_NAME)
    if ob is None:
        me = bpy.data.meshes.new(DOMAIN_NAME)
        v = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        me.from_pydata(v, [], f)
        ob = bpy.data.objects.new(DOMAIN_NAME, me)
        col = bpy.data.collections.get("Lighting") or bpy.context.scene.collection
        col.objects.link(ob)
    if cam is not None:
        ob.parent = cam
        ob.matrix_parent_inverse = Matrix.Identity(4)
    # camera local: X right, Y up, -Z forward
    ob.matrix_basis = Matrix.Diagonal((DOMAIN_HALF_M, half_h, DOMAIN_HALF_M, 1.0))
    # Shadow rays ignore the fog: with a single scattering bounce the inside of a dense fog layer
    # (and the ground under it) would go dark, while real fog (albedo ~1) passes most of the
    # diffuse daylight on by multiple scattering. Camera rays still get the full extinction.
    ob.visible_shadow = bool(shadow)
    ob["sbahn_non_occluding"] = True
    return ob


def build_fog(scene, p: dict) -> dict:
    """p keys: fog_density (1/m, ground fog at track level), haze_density (1/m), rain_extinction (1/m),
    fog_top_m, fog_patchiness, eye_height_m, seed, sky_radiance (mean world radiance). Returns effective values."""
    sig_fog = float(p.get("fog_density", 0.0))
    sig_bg = float(p.get("haze_density", 0.0)) + float(p.get("rain_extinction", 0.0))
    top = float(p.get("fog_top_m", 80.0))
    patch = float(p.get("fog_patchiness", 0.0)) if sig_fog > 0 else 0.0
    total = sig_bg + sig_fog
    cam = scene.camera
    ob = bpy.data.objects.get(DOMAIN_NAME)
    if total <= 1e-7:
        if ob:
            ob.hide_render = True
        return dict(fog_density_per_m=0.0, extinction_per_m=0.0,
                    visibility_m=None, fog_top_m=None, fog_patchiness=0.0, volume="none")
    ob = _domain(cam, max(250.0, top * 2.5 + 60.0))
    ob.hide_render = False
    mat = bpy.data.materials.get("SB_Fog") or bpy.data.materials.new("SB_Fog")
    try:
        mat.use_nodes = True
    except Exception:
        pass
    g = NodeBuilder(mat.node_tree, clear=True)
    out = g.node("ShaderNodeOutputMaterial")
    heterogeneous = sig_fog > 0 and (top < 400.0 or patch > 0)
    if heterogeneous:
        geo = g.node("ShaderNodeNewGeometry")
        _, _, z = g.separate(geo.outputs["Position"])
        oi = g.node("ShaderNodeObjectInfo")
        _, _, zc = g.separate(oi.outputs["Location"])
        h = g.sub(z, g.sub(zc, float(p.get("eye_height_m", 2.45))))
        layer = g.sub(1.0, g.smoothstep(0.6 * top, 1.4 * top, h))
        dens = layer
        if patch > 0:
            nz = g.node("ShaderNodeTexNoise", noise_dimensions="4D")
            mp = g.vmath("SCALE", geo.outputs["Position"], scale=1.0 / 150.0)
            g.set(nz.inputs["Vector"], mp)
            nz.inputs["W"].default_value = float(p.get("seed", 0)) * 1.618
            nz.inputs["Scale"].default_value = 1.0
            nz.inputs["Detail"].default_value = 3.0
            pat = g.math("MAXIMUM", g.add(1.0, g.mul(patch, g.sub(g.mul(nz.outputs["Fac"], 2.0), 1.0))), 0.0)
            # stretch the noise contrast (Perlin fBm sits mostly within 0.3..0.7)
            pat = g.math("MAXIMUM", g.add(1.0, g.mul(g.sub(pat, 1.0), 2.5)), 0.0)
            dens = g.mul(dens, pat)
        density = g.add(sig_bg, g.mul(dens, sig_fog))
    else:
        density = g.value(total)
    # fog droplets scatter white and forward; pure haze gets a faint blue (aerial perspective)
    fog_share = sig_fog / total if total > 0 else 0.0
    albedo = 0.97
    blue = (0.80 + 0.20 * fog_share, 0.88 + 0.12 * fog_share, 1.0)
    scat = g.node("ShaderNodeVolumeScatter")
    scat.inputs["Color"].default_value = (*blue, 1.0)
    g.set(scat.inputs["Density"], g.mul(density, albedo))
    scat.inputs["Anisotropy"].default_value = 0.75 if fog_share > 0.3 else 0.55
    absn = g.node("ShaderNodeVolumeAbsorption")
    absn.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    g.set(absn.inputs["Density"], g.mul(density, 1.0 - albedo))
    add = g.node("ShaderNodeAddShader")
    g.link(scat.outputs[0], add.inputs[0])
    g.link(absn.outputs[0], add.inputs[1])
    # Multiple-scattering top-up: single scattering only sees the upper-hemisphere sky, real
    # fog radiance tends to the sky radiance. Emitted radiance per metre = sigma * L_extra, so a
    # thick fog saturates at (single-scatter + L_extra). Zero at night (headlights do the work).
    l_extra = MS_TOPUP * float(p.get("sky_radiance", 0.0))
    if l_extra > 0:
        emi = g.node("ShaderNodeEmission")
        emi.inputs["Color"].default_value = (*blue, 1.0)
        g.set(emi.inputs["Strength"], g.mul(density, l_extra))
        add2 = g.node("ShaderNodeAddShader")
        g.link(add.outputs[0], add2.inputs[0])
        g.link(emi.outputs[0], add2.inputs[1])
        add = add2
    g.link(add.outputs[0], out.inputs["Volume"])
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    if hasattr(mat, "cycles"):
        try:
            mat.cycles.homogeneous_volume = not heterogeneous
        except Exception:
            pass
    return dict(fog_density_per_m=round(sig_fog, 6),
                extinction_per_m=round(total, 6), visibility_m=round(KOSCHMIEDER / total, 1),
                fog_top_m=top if sig_fog > 0 else None, fog_patchiness=patch,
                volume="heterogeneous" if heterogeneous else "homogeneous")

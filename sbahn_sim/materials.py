"""Procedural PBR materials, with optional texture sets from assets/textures.

Every material is created once and cached by name. When a texture set for the
role exists in <asset_dir>/textures (see assets.py), the material uses image
textures with box projection in object space (so no UV unwrap is needed on the
generated meshes); otherwise a fully procedural node setup is used.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

import bpy

_TEX_PATTERNS = {   # ambientCG and Poly Haven naming
    "color": ("*Color*", "*_diff_*", "*_diffuse_*", "*albedo*", "*Albedo*", "*BaseColor*"),
    "rough": ("*Roughness*", "*_rough_*"),
    "normal": ("*NormalGL*", "*_nor_gl_*", "*Normal*"),
    "disp": ("*Displacement*", "*_disp_*", "*Height*"),
}


def find_texture_set(asset_dir: str, role: str) -> Optional[Dict[str, str]]:
    """Return {'color': path, 'rough': ..., 'normal': ...} for assets/textures/<role>/ or None."""
    folder = os.path.join(asset_dir, "textures", role)
    if not os.path.isdir(folder):
        return None
    found = {}
    for key, pats in _TEX_PATTERNS.items():
        for pat in pats:
            hits = sorted(glob.glob(os.path.join(folder, "**", pat), recursive=True))
            hits = [h for h in hits if h.lower().endswith((".jpg", ".jpeg", ".png", ".exr", ".tif", ".tiff"))]
            if hits:
                found[key] = hits[0]
                break
    return found if "color" in found else None


def _new(name: str) -> Tuple[bpy.types.Material, bpy.types.ShaderNodeTree, bpy.types.Node]:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    return mat, nt, bsdf


def _n(nt, idname, loc=(0, 0), **props):
    n = nt.nodes.new(idname)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n


def _objcoords(nt, scale_m: float, loc=(-900, 0)):
    """Texture Coordinate(Object) -> Mapping(scale 1/scale_m)."""
    tc = _n(nt, "ShaderNodeTexCoord", loc)
    mp = _n(nt, "ShaderNodeMapping", (loc[0] + 200, loc[1]))
    mp.inputs["Scale"].default_value = (1.0 / scale_m,) * 3
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    return mp


class MaterialLibrary:
    """Lazily created, cached materials. `mats.get("ballast")`."""

    def __init__(self, asset_dir: str, use_textures: bool = True):
        self.asset_dir = asset_dir
        self.use_textures = use_textures
        self._cache: Dict[str, bpy.types.Material] = {}

    def get(self, role: str) -> bpy.types.Material:
        if role in self._cache:
            return self._cache[role]
        existing = bpy.data.materials.get(f"SB_{role}")
        if existing:
            self._cache[role] = existing
            return existing
        spec = _SPECS[role]
        texset = find_texture_set(self.asset_dir, role) if self.use_textures else None
        if "build" in spec:          # weathered builders use a texture set as their base when present
            mat = spec["build"](f"SB_{role}", texset)
        else:
            mat = textured_material(f"SB_{role}", texset, **spec["tex"]) if texset else spec["proc"](f"SB_{role}")
        mat["sbahn_role"] = role
        self._cache[role] = mat
        return mat


# --------------------------------------------------------------------------
# Textured
# --------------------------------------------------------------------------
def textured_material(name, texset, tile_m=2.0, roughness=0.8, metallic=0.0, normal_strength=1.0,
                      tint=(1, 1, 1, 1)):
    mat, nt, bsdf = _new(name)
    mp = _objcoords(nt, tile_m)
    bsdf.inputs["Metallic"].default_value = metallic

    def img(path, loc, non_color=False):
        node = _n(nt, "ShaderNodeTexImage", loc, projection="BOX", projection_blend=0.2)
        node.image = bpy.data.images.load(path, check_existing=True)
        if non_color:
            node.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mp.outputs["Vector"], node.inputs["Vector"])
        return node

    col = img(texset["color"], (-500, 300))
    mix = _n(nt, "ShaderNodeMix", (-200, 300), data_type="RGBA", blend_type="MULTIPLY")
    mix.inputs["Factor"].default_value = 1.0
    mix.inputs[7].default_value = tint  # B color
    nt.links.new(col.outputs["Color"], mix.inputs[6])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    if "rough" in texset:
        r = img(texset["rough"], (-500, 0), True)
        nt.links.new(r.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = roughness
    if "normal" in texset:
        nrm = img(texset["normal"], (-500, -300), True)
        nm = _n(nt, "ShaderNodeNormalMap", (-200, -300))
        nm.inputs["Strength"].default_value = normal_strength
        nt.links.new(nrm.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# --------------------------------------------------------------------------
# Procedural fallbacks
# --------------------------------------------------------------------------
def _noise_bump(nt, bsdf, mp, scale=8.0, detail=6.0, strength=0.3, distortion=0.0):
    nz = _n(nt, "ShaderNodeTexNoise", (-500, -300))
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = detail
    nz.inputs["Distortion"].default_value = distortion
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    bump = _n(nt, "ShaderNodeBump", (-200, -300))
    bump.inputs["Strength"].default_value = strength
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return nz


def _ramp(nt, loc, stops):
    cr = _n(nt, "ShaderNodeValToRGB", loc)
    els = cr.color_ramp.elements
    while len(els) > 1:
        els.remove(els[-1])
    els[0].position, els[0].color = stops[0]
    for pos, col in stops[1:]:
        e = els.new(pos); e.color = col
    return cr


def proc_ballast(name):
    mat, nt, bsdf = _new(name)
    mp = _objcoords(nt, 1.0)
    vor = _n(nt, "ShaderNodeTexVoronoi", (-600, 300), feature="F1")
    vor.inputs["Scale"].default_value = 14.0          # ~7 cm stones
    vor.inputs["Randomness"].default_value = 1.0
    nt.links.new(mp.outputs["Vector"], vor.inputs["Vector"])
    ramp = _ramp(nt, (-300, 300), [(0.0, (0.10, 0.10, 0.10, 1)), (0.5, (0.22, 0.21, 0.20, 1)), (1.0, (0.36, 0.34, 0.31, 1))])
    nt.links.new(vor.outputs["Color"], ramp.inputs["Fac"])
    # brownish dirt between stones near the surface
    dirt = _n(nt, "ShaderNodeTexNoise", (-600, 0))
    dirt.inputs["Scale"].default_value = 0.6
    nt.links.new(mp.outputs["Vector"], dirt.inputs["Vector"])
    mix = _n(nt, "ShaderNodeMix", (-50, 300), data_type="RGBA", blend_type="MIX")
    mix.inputs[7].default_value = (0.20, 0.16, 0.12, 1)
    nt.links.new(dirt.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(ramp.outputs["Color"], mix.inputs[6])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    bump = _n(nt, "ShaderNodeBump", (-200, -300))
    bump.inputs["Strength"].default_value = 0.6
    nt.links.new(vor.outputs["Distance"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def proc_concrete(name, base=(0.62, 0.61, 0.58, 1), rough=0.75, scale=6.0):
    mat, nt, bsdf = _new(name)
    layer_concrete(nt, bsdf, _objcoords(nt, 1.0), base=base, rough=rough, scale=scale)
    return mat


def proc_rail_body(name):
    mat, nt, bsdf = _new(name)
    mp = _objcoords(nt, 1.0)
    nz = _n(nt, "ShaderNodeTexNoise", (-600, 300))
    nz.inputs["Scale"].default_value = 3.0
    nz.inputs["Detail"].default_value = 6.0
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    ramp = _ramp(nt, (-300, 300), [(0.35, (0.16, 0.08, 0.04, 1)), (0.65, (0.30, 0.16, 0.09, 1))])
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    bsdf.inputs["Metallic"].default_value = 0.2
    _noise_bump(nt, bsdf, mp, scale=60.0, detail=3.0, strength=0.2)
    return mat


def proc_rail_head(name):
    mat, nt, bsdf = _new(name)
    bsdf.inputs["Base Color"].default_value = (0.75, 0.75, 0.76, 1)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.22
    mp = _objcoords(nt, 1.0)
    nz = _n(nt, "ShaderNodeTexNoise", (-600, 0))
    nz.inputs["Scale"].default_value = 200.0
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    mr = _n(nt, "ShaderNodeMapRange", (-300, 0))
    mr.inputs["To Min"].default_value = 0.15
    mr.inputs["To Max"].default_value = 0.35
    nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    return mat


def _shift(nt, loc, dx, dy):
    return (loc[0] + dx, loc[1] + dy)


def layer_grass(nt, bsdf, mp, loc=(0, 0), base=(0.11, 0.16, 0.05, 1)):
    """Lineside turf seen from a few metres: large dry / lush patches, clumps and blade striation
    (was a smooth khaki-green ramp that read as bare sand between the grass tufts)."""
    nz = _n(nt, "ShaderNodeTexNoise", _shift(nt, loc, -800, 300))
    nz.inputs["Scale"].default_value = 0.15
    nz.inputs["Detail"].default_value = 10.0
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    ramp = _ramp(nt, _shift(nt, loc, -550, 300), [(0.28, (0.20, 0.18, 0.085, 1)), (0.42, tuple(c * 0.9 for c in base[:3]) + (1,)),
                                                 (0.75, (0.055, 0.10, 0.032, 1))])
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    clump = _n(nt, "ShaderNodeTexNoise", _shift(nt, loc, -800, 0))
    clump.inputs["Scale"].default_value = 3.0
    clump.inputs["Detail"].default_value = 4.0
    nt.links.new(mp.outputs["Vector"], clump.inputs["Vector"])
    mix = _n(nt, "ShaderNodeMix", _shift(nt, loc, -300, 300), data_type="RGBA", blend_type="OVERLAY")
    mix.inputs["Factor"].default_value = 0.45
    nt.links.new(ramp.outputs["Color"], mix.inputs[6])
    nt.links.new(clump.outputs["Fac"], mix.inputs[7])
    nz2 = _n(nt, "ShaderNodeTexNoise", _shift(nt, loc, -800, -300))
    nz2.inputs["Scale"].default_value = 70.0
    nz2.inputs["Detail"].default_value = 6.0
    nt.links.new(mp.outputs["Vector"], nz2.inputs["Vector"])
    mix2 = _n(nt, "ShaderNodeMix", _shift(nt, loc, -150, 300), data_type="RGBA", blend_type="OVERLAY")
    mix2.inputs["Factor"].default_value = 0.30
    nt.links.new(mix.outputs[2], mix2.inputs[6])
    nt.links.new(nz2.outputs["Fac"], mix2.inputs[7])
    nt.links.new(mix2.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.92
    bump = _n(nt, "ShaderNodeBump", _shift(nt, loc, -200, -300))
    bump.inputs["Strength"].default_value = 0.6
    nt.links.new(nz2.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return bsdf


def layer_textured(nt, bsdf, mp, texset, tile_m=2.0, loc=(0, 0), patch_scale=0.15, patch_amount=0.45,
                   tint=(1.0, 1.0, 1.0, 1.0)):
    """Photo texture layer for large surfaces (terrain): box projection at tile_m metres, multiplied by a
    large-scale brightness / hue variation so the repeat of a 2 m texture does not show over 100 m."""
    sc = _n(nt, "ShaderNodeMapping", _shift(nt, loc, -1100, 300))
    sc.inputs["Scale"].default_value = (1.0 / tile_m,) * 3
    nt.links.new(mp.outputs["Vector"], sc.inputs["Vector"])

    def img(path, l, non_color=False):
        node = _n(nt, "ShaderNodeTexImage", l, projection="BOX", projection_blend=0.3)
        node.image = bpy.data.images.load(path, check_existing=True)
        if non_color:
            node.image.colorspace_settings.name = "Non-Color"
        nt.links.new(sc.outputs["Vector"], node.inputs["Vector"])
        return node.outputs["Color"]

    col = img(texset["color"], _shift(nt, loc, -850, 400))
    if tuple(tint[:3]) != (1.0, 1.0, 1.0):
        tm = _n(nt, "ShaderNodeMix", _shift(nt, loc, -600, 450), data_type="RGBA", blend_type="MULTIPLY")
        tm.inputs["Factor"].default_value = 1.0
        tm.inputs[7].default_value = tint
        nt.links.new(col, tm.inputs[6])
        col = tm.outputs[2]
    nz = _n(nt, "ShaderNodeTexNoise", _shift(nt, loc, -850, 100))
    nz.inputs["Scale"].default_value = patch_scale
    nz.inputs["Detail"].default_value = 6.0
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    var = _ramp(nt, _shift(nt, loc, -600, 100), [(0.3, (0.75, 0.72, 0.60, 1)), (0.7, (1.15, 1.15, 1.05, 1))])
    nt.links.new(nz.outputs["Fac"], var.inputs["Fac"])
    mix = _n(nt, "ShaderNodeMix", _shift(nt, loc, -350, 300), data_type="RGBA", blend_type="MULTIPLY")
    mix.inputs["Factor"].default_value = patch_amount
    nt.links.new(col, mix.inputs[6])
    nt.links.new(var.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    if "rough" in texset:
        nt.links.new(img(texset["rough"], _shift(nt, loc, -850, -200), True), bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.9
    if "normal" in texset:
        nm = _n(nt, "ShaderNodeNormalMap", _shift(nt, loc, -350, -300))
        nt.links.new(img(texset["normal"], _shift(nt, loc, -850, -500), True), nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return bsdf


def layer_concrete(nt, bsdf, mp, loc=(0, 0), base=(0.62, 0.61, 0.58, 1), rough=0.75, scale=6.0):
    nz = _n(nt, "ShaderNodeTexNoise", _shift(nt, loc, -600, 300))
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = 8.0
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    ramp = _ramp(nt, _shift(nt, loc, -300, 300), [(0.3, tuple(c * 0.75 for c in base[:3]) + (1,)), (0.7, base)])
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    nz2 = _n(nt, "ShaderNodeTexNoise", _shift(nt, loc, -600, -300))
    nz2.inputs["Scale"].default_value = 40.0
    nz2.inputs["Detail"].default_value = 4.0
    nt.links.new(mp.outputs["Vector"], nz2.inputs["Vector"])
    bump = _n(nt, "ShaderNodeBump", _shift(nt, loc, -200, -300))
    bump.inputs["Strength"].default_value = 0.15
    nt.links.new(nz2.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return bsdf


def layer_field(nt, bsdf, mp, loc=(0, 0)):
    wave = _n(nt, "ShaderNodeTexWave", _shift(nt, loc, -600, 300), wave_type="BANDS", bands_direction="X")
    wave.inputs["Scale"].default_value = 1.4       # ~0.7 m furrow spacing
    wave.inputs["Distortion"].default_value = 1.5
    nt.links.new(mp.outputs["Vector"], wave.inputs["Vector"])
    nz = _n(nt, "ShaderNodeTexNoise", _shift(nt, loc, -600, 0))
    nz.inputs["Scale"].default_value = 0.05
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    ramp = _ramp(nt, _shift(nt, loc, -300, 0), [(0.4, (0.22, 0.16, 0.08, 1)), (0.6, (0.30, 0.34, 0.10, 1))])
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    mix = _n(nt, "ShaderNodeMix", _shift(nt, loc, -50, 300), data_type="RGBA", blend_type="MULTIPLY")
    mix.inputs["Factor"].default_value = 0.35
    nt.links.new(ramp.outputs["Color"], mix.inputs[6])
    nt.links.new(wave.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.95
    bump = _n(nt, "ShaderNodeBump", _shift(nt, loc, -200, -300))
    bump.inputs["Strength"].default_value = 0.4
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return bsdf


def proc_grass(name, base=(0.11, 0.16, 0.05, 1)):
    mat, nt, bsdf = _new(name)
    layer_grass(nt, bsdf, _objcoords(nt, 1.0), base=base)
    return mat


def proc_forest_floor(name):
    return proc_grass(name, base=(0.12, 0.10, 0.05, 1))


def proc_field(name):
    """Ploughed / cropped field with furrows (anisotropic stripes)."""
    mat, nt, bsdf = _new(name)
    layer_field(nt, bsdf, _objcoords(nt, 1.0))
    return mat


def proc_asphalt(name):
    return proc_concrete(name, base=(0.12, 0.12, 0.12, 1), rough=0.85, scale=30.0)


def proc_galvanised(name):
    mat, nt, bsdf = _new(name)
    bsdf.inputs["Base Color"].default_value = (0.55, 0.56, 0.57, 1)
    bsdf.inputs["Metallic"].default_value = 0.9
    bsdf.inputs["Roughness"].default_value = 0.5
    return mat


def proc_flat(name, color, rough=0.6, metallic=0.0):
    mat, nt, bsdf = _new(name)
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def proc_emission(name, color, strength=8.0):
    mat, nt, bsdf = _new(name)
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Emission Color"].default_value = color
    bsdf.inputs["Emission Strength"].default_value = strength
    return mat


def proc_bark(name):
    return proc_concrete(name, base=(0.25, 0.18, 0.12, 1), rough=0.9, scale=12.0)


def proc_leaves(name, color=(0.06, 0.18, 0.04, 1)):
    mat, nt, bsdf = _new(name)
    mp = _objcoords(nt, 1.0)
    nz = _n(nt, "ShaderNodeTexNoise", (-600, 300))
    nz.inputs["Scale"].default_value = 3.0
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    ramp = _ramp(nt, (-300, 300), [(0.35, tuple(c * 0.6 for c in color[:3]) + (1,)), (0.65, color)])
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.7
    bsdf.inputs["Subsurface Weight"].default_value = 0.2 if "Subsurface Weight" in bsdf.inputs else 0.0
    return mat


def proc_noise_wall(name):
    """Concrete sound-barrier panels with horizontal ribs."""
    mat, nt, bsdf = _new(name)
    mp = _objcoords(nt, 1.0)
    wave = _n(nt, "ShaderNodeTexWave", (-600, 300), wave_type="BANDS", bands_direction="Z")
    wave.inputs["Scale"].default_value = 6.0
    nt.links.new(mp.outputs["Vector"], wave.inputs["Vector"])
    nz = _n(nt, "ShaderNodeTexNoise", (-600, 0))
    nz.inputs["Scale"].default_value = 2.0
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    ramp = _ramp(nt, (-300, 0), [(0.3, (0.30, 0.30, 0.285, 1)), (0.7, (0.42, 0.415, 0.40, 1))])
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    # weathering: rain streaks running down the panels, splash dirt and moss along the foot
    tc = _n(nt, "ShaderNodeTexCoord", (-900, -600))
    smp = _n(nt, "ShaderNodeMapping", (-750, -600))
    smp.inputs["Scale"].default_value = (18.0, 18.0, 0.7)
    nt.links.new(tc.outputs["Object"], smp.inputs["Vector"])
    st = _noise(nt, smp.outputs["Vector"], 1.0, 4.0, loc=(-600, -600))
    streak = _m(nt, "MULTIPLY", _smooth(nt, st.outputs["Fac"], 0.45, 0.75, (-450, -600)), 0.5, loc=(-300, -600))
    col = _mixc(nt, streak, ramp.outputs["Color"], (0.16, 0.155, 0.14, 1), (-100, 0))
    sep = _n(nt, "ShaderNodeSeparateXYZ", (-600, -850))
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    foot = _m(nt, "SUBTRACT", 1.0, _smooth(nt, sep.outputs["Z"], 0.1, 0.9, (-450, -850)), loc=(-300, -850))
    fn = _noise(nt, tc.outputs["Object"], 4.0, 4.0, loc=(-450, -1050))
    foot = _m(nt, "MULTIPLY", foot, _smooth(nt, fn.outputs["Fac"], 0.3, 0.6, (-300, -1050)), loc=(-150, -900))
    col = _mixc(nt, foot, col, (0.10, 0.11, 0.06, 1), (50, 0))
    nt.links.new(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    bump = _n(nt, "ShaderNodeBump", (-200, -300))
    bump.inputs["Strength"].default_value = 0.5
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# --------------------------------------------------------------------------
# Weathered track materials (realism pass, 2026-10-02)
# --------------------------------------------------------------------------
# Real track is never clean: ballast next to the rails turns rust-brown from rail and brake dust,
# sleepers carry rust halos round the fastenings and grime at the ends, rail heads are polished only
# on the running band, masts collect streaks and splash dirt. These builders take an optional photo
# texture set (assets/textures/<role>) for the base and add the weathering on top, so the look holds
# with and without downloaded textures.
def _m(nt, op, a, b=None, loc=(0, 0), clamp=False):
    n = _n(nt, "ShaderNodeMath", loc, operation=op)
    n.use_clamp = clamp
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[i].default_value = x
        else:
            nt.links.new(x, n.inputs[i])
    return n.outputs[0]


def _smooth(nt, x, lo, hi, loc=(0, 0)):
    mr = _n(nt, "ShaderNodeMapRange", loc, interpolation_type="SMOOTHSTEP")
    mr.inputs["From Min"].default_value = lo
    mr.inputs["From Max"].default_value = hi
    nt.links.new(x, mr.inputs["Value"])
    return mr.outputs["Result"]


def _mixc(nt, fac, a, b, loc=(0, 0), blend="MIX"):
    m = _n(nt, "ShaderNodeMix", loc, data_type="RGBA", blend_type=blend)
    m.clamp_result = True
    for sock, x in ((m.inputs[0], fac), (m.inputs[6], a), (m.inputs[7], b)):
        if isinstance(x, (int, float, tuple)):
            sock.default_value = x
        else:
            nt.links.new(x, sock)
    return m.outputs[2]


def _noise(nt, vec, scale, detail=4.0, rough=0.55, loc=(0, 0), dims="3D"):
    nz = _n(nt, "ShaderNodeTexNoise", loc, noise_dimensions=dims)
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = detail
    nz.inputs["Roughness"].default_value = rough
    nt.links.new(vec, nz.inputs["Vector"])
    return nz


def _tex_base(nt, texset, mp, loc=(-900, 600), normal_strength=1.0):
    """Box-projected photo texture: returns (color, roughness, height) sockets (height may be None)."""
    def img(path, l, non_color=False):
        node = _n(nt, "ShaderNodeTexImage", l, projection="BOX", projection_blend=0.25)
        node.image = bpy.data.images.load(path, check_existing=True)
        if non_color:
            node.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mp.outputs["Vector"], node.inputs["Vector"])
        return node.outputs["Color"]
    col = img(texset["color"], loc)
    rough = img(texset["rough"], (loc[0], loc[1] - 300), True) if "rough" in texset else None
    disp = img(texset["disp"], (loc[0], loc[1] - 600), True) if "disp" in texset else None
    nrm = None
    if "normal" in texset:
        nm = _n(nt, "ShaderNodeNormalMap", (loc[0] + 300, loc[1] - 900))
        nm.inputs["Strength"].default_value = normal_strength
        nt.links.new(img(texset["normal"], (loc[0], loc[1] - 900), True), nm.inputs["Color"])
        nrm = nm.outputs["Normal"]
    return col, rough, disp, nrm


def ballast_material(name, texset=None, track_spacing=4.0, rail_c=0.7535):
    """Crushed-stone ballast. Uses the point attribute 'lat' (metres right of the double-track centre
    line, written by track_geometry.build_ballast) for rust near the rails and in the four-foot."""
    mat, nt, bsdf = _new(name)
    tc = _n(nt, "ShaderNodeTexCoord", (-2200, 0))
    if texset:
        # the 2 m gravel_stones scan has ~1.5 cm grit: stretched to 6 m its stones match the 31.5-63 mm
        # ballast grading (at 2 m it read as a smooth gravel road)
        mp = _objcoords(nt, 6.0, (-2200, 600))
        col, rough, disp, nrm = _tex_base(nt, texset, mp, normal_strength=2.5)
        stone = _mixc(nt, 1.0, col, (1.45, 1.45, 1.41, 1), (-600, 800), "MULTIPLY")   # scan mean 0.08 -> 0.11
        if nrm is not None:
            nt.links.new(nrm, bsdf.inputs["Normal"])
        if rough is not None:
            nt.links.new(rough, bsdf.inputs["Roughness"])
        crev = _smooth(nt, disp, 0.05, 0.45) if disp is not None else 1.0
    else:
        # angular stones: Voronoi cells with a random planar facet each, dark crevices between them
        warp = _noise(nt, tc.outputs["Object"], 4.0, 2.0, loc=(-2000, 300))
        wv = _n(nt, "ShaderNodeVectorMath", (-1800, 300), operation="MULTIPLY_ADD")
        nt.links.new(warp.outputs["Color"], wv.inputs[0])
        wv.inputs[1].default_value = (0.035, 0.035, 0.035)
        nt.links.new(tc.outputs["Object"], wv.inputs[2])
        vor = _n(nt, "ShaderNodeTexVoronoi", (-1600, 400), feature="F1", distance="EUCLIDEAN")
        vor.inputs["Scale"].default_value = 21.0          # ~4.5 cm stones (31.5-63 mm grading)
        nt.links.new(wv.outputs[0], vor.inputs["Vector"])
        edge = _n(nt, "ShaderNodeTexVoronoi", (-1600, 0), feature="DISTANCE_TO_EDGE")
        edge.inputs["Scale"].default_value = 21.0
        nt.links.new(wv.outputs[0], edge.inputs["Vector"])
        crev = _smooth(nt, edge.outputs["Distance"], 0.0, 0.16, (-1350, 0))
        sep = _n(nt, "ShaderNodeSeparateColor", (-1350, 400))
        nt.links.new(vor.outputs["Color"], sep.inputs["Color"])
        rnd_r = sep.outputs["Red"]
        grey = _ramp(nt, (-1150, 500), [(0.0, (0.085, 0.085, 0.083, 1)), (0.55, (0.20, 0.195, 0.185, 1)),
                                        (1.0, (0.36, 0.35, 0.33, 1))])
        nt.links.new(rnd_r, grey.inputs["Fac"])
        warm = _mixc(nt, _m(nt, "MULTIPLY", sep.outputs["Green"], 0.45, loc=(-1150, 250)), grey.outputs["Color"],
                     (0.30, 0.24, 0.18, 1), (-950, 450))
        grit = _noise(nt, tc.outputs["Object"], 260.0, 3.0, loc=(-1150, 150))
        stone = _mixc(nt, 0.25, warm, grit.outputs["Fac"], (-800, 450), "OVERLAY")
        # facet: height = dot(p * scale, random_vector - 0.5) gives every stone its own flat face
        fv = _n(nt, "ShaderNodeVectorMath", (-1150, -250), operation="SUBTRACT")
        nt.links.new(vor.outputs["Color"], fv.inputs[0])
        fv.inputs[1].default_value = (0.5, 0.5, 0.5)
        dp = _n(nt, "ShaderNodeVectorMath", (-950, -250), operation="DOT_PRODUCT")
        sc = _n(nt, "ShaderNodeVectorMath", (-1150, -450), operation="SCALE")
        nt.links.new(wv.outputs[0], sc.inputs[0])
        sc.inputs["Scale"].default_value = 21.0
        nt.links.new(sc.outputs[0], dp.inputs[0])
        nt.links.new(fv.outputs[0], dp.inputs[1])
        height = _m(nt, "ADD", _m(nt, "MULTIPLY", dp.outputs["Value"], 0.35, loc=(-800, -250)),
                    _m(nt, "MULTIPLY", crev, 0.8, loc=(-800, -400)), loc=(-650, -300))
        height = _m(nt, "ADD", height, _m(nt, "MULTIPLY", grit.outputs["Fac"], 0.15, loc=(-650, -450)), loc=(-500, -300))
        bump = _n(nt, "ShaderNodeBump", (-300, -300))
        bump.inputs["Distance"].default_value = 0.03
        bump.inputs["Strength"].default_value = 1.0
        nt.links.new(height, bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        nt.links.new(_m(nt, "SUBTRACT", 0.95, _m(nt, "MULTIPLY", crev, 0.12), loc=(-300, -100)), bsdf.inputs["Roughness"])
    # crevices are dark (shadowed fines, dirt)
    stone = _mixc(nt, _m(nt, "SUBTRACT", 1.0, crev, loc=(-700, 700)), stone, (0.03, 0.028, 0.026, 1), (-500, 650))
    # rust / brake dust: strongest along the rails, a lighter tint over the whole four-foot
    lat = _n(nt, "ShaderNodeAttribute", (-2200, -700))
    lat.attribute_name = "lat"
    lat.attribute_type = "GEOMETRY"
    h = track_spacing / 2.0
    d = _m(nt, "MINIMUM", _m(nt, "ABSOLUTE", _m(nt, "SUBTRACT", lat.outputs["Fac"], h, loc=(-2000, -650)), loc=(-1850, -650)),
           _m(nt, "ABSOLUTE", _m(nt, "ADD", lat.outputs["Fac"], h, loc=(-2000, -800)), loc=(-1850, -800)), loc=(-1700, -700))
    r = _m(nt, "ABSOLUTE", _m(nt, "SUBTRACT", d, rail_c, loc=(-1550, -650)), loc=(-1400, -650))
    near_rail = _m(nt, "SUBTRACT", 1.0, _smooth(nt, r, 0.03, 0.42, (-1250, -650)), loc=(-1100, -650))
    fourfoot = _m(nt, "SUBTRACT", 1.0, _smooth(nt, d, 0.55, 0.95, (-1250, -850)), loc=(-1100, -850))
    var = _noise(nt, tc.outputs["Object"], 0.7, 3.0, loc=(-1250, -1050))
    var = _smooth(nt, var.outputs["Fac"], 0.3, 0.7, (-1100, -1050))
    rust = _m(nt, "MULTIPLY", _m(nt, "ADD", _m(nt, "MULTIPLY", near_rail, 0.75), _m(nt, "MULTIPLY", fourfoot, 0.35),
                                 loc=(-900, -750)), _m(nt, "ADD", _m(nt, "MULTIPLY", var, 0.6), 0.5, loc=(-900, -1000)),
              loc=(-750, -800), clamp=True)
    rusty = _mixc(nt, 1.0, stone, (0.42, 0.25, 0.15, 1), (-500, 400), "MULTIPLY")
    rusty = _mixc(nt, 0.5, rusty, (0.13, 0.075, 0.045, 1), (-350, 400))
    stone = _mixc(nt, rust, stone, rusty, (-200, 500))
    # the shoulders and slopes collect soil and weeds: slightly duller and browner away from the track
    shoulder = _smooth(nt, d, 1.7, 3.2, (-700, -1200))
    stone = _mixc(nt, _m(nt, "MULTIPLY", shoulder, 0.35, loc=(-500, -1200)), stone, (0.16, 0.14, 0.11, 1), (-50, 500))
    nt.links.new(stone, bsdf.inputs["Base Color"])
    _set_spec(bsdf, 0.3)
    return mat


def _set_spec(bsdf, v):
    for key in ("Specular IOR Level", "Specular"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = v
            return


def sleeper_material(name, texset=None, rail_c=0.7535):
    """B70 prestressed-concrete sleeper (object space: X across the track, Z up, top at 0): per-sleeper
    tone, rust halos round the fastenings, grime towards the ends and on the sides."""
    mat, nt, bsdf = _new(name)
    tc = _n(nt, "ShaderNodeTexCoord", (-2000, 0))
    oi = _n(nt, "ShaderNodeObjectInfo", (-2000, -400))
    if texset:
        mp = _objcoords(nt, 1.0, (-2000, 600))
        col, rough, disp, nrm = _tex_base(nt, texset, mp)
        base = _mixc(nt, 1.0, col, (1.7, 1.7, 1.67, 1), (-600, 600), "MULTIPLY")
        if nrm is not None:
            nt.links.new(nrm, bsdf.inputs["Normal"])
    else:
        nz = _noise(nt, tc.outputs["Object"], 7.0, 8.0, loc=(-1700, 500))
        base = _ramp(nt, (-1450, 500), [(0.3, (0.19, 0.185, 0.175, 1)), (0.7, (0.29, 0.28, 0.265, 1))])
        nt.links.new(nz.outputs["Fac"], base.inputs["Fac"])
        base = base.outputs["Color"]
        pores = _noise(nt, tc.outputs["Object"], 180.0, 2.0, loc=(-1450, 200))
        base = _mixc(nt, 0.18, base, pores.outputs["Fac"], (-1200, 450), "OVERLAY")
        bump = _n(nt, "ShaderNodeBump", (-300, -500))
        bump.inputs["Strength"].default_value = 0.25
        bump.inputs["Distance"].default_value = 0.004
        nt.links.new(pores.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # per-sleeper tone (instances get their own random value)
    tone = _m(nt, "ADD", _m(nt, "MULTIPLY", oi.outputs["Random"], 0.35, loc=(-1700, -400)), 0.8, loc=(-1550, -400))
    base = _mixc(nt, 1.0, base, _n(nt, "ShaderNodeCombineColor", (-1400, -400)).outputs[0], (-1000, 400), "MULTIPLY")
    cc = base.node.inputs[7].links[0].from_node
    for i in range(3):
        nt.links.new(tone, cc.inputs[i])
    sep = _n(nt, "ShaderNodeSeparateXYZ", (-1800, -800))
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    ax = _m(nt, "ABSOLUTE", sep.outputs["X"], loc=(-1600, -800))
    # rust halos round the rail seats (fastenings at |x| = rail_c +- 0.14)
    q = _m(nt, "ABSOLUTE", _m(nt, "SUBTRACT", ax, rail_c, loc=(-1450, -800)), loc=(-1300, -800))
    halo = _m(nt, "SUBTRACT", 1.0, _smooth(nt, q, 0.10, 0.32, (-1150, -800)), loc=(-1000, -800))
    hv = _noise(nt, tc.outputs["Object"], 14.0, 3.0, loc=(-1150, -1000))
    halo = _m(nt, "MULTIPLY", halo, _smooth(nt, hv.outputs["Fac"], 0.35, 0.65, (-1000, -1000)), loc=(-850, -850))
    halo = _m(nt, "MULTIPLY", halo, _m(nt, "ADD", _m(nt, "MULTIPLY", oi.outputs["Random"], 0.5), 0.35), loc=(-700, -850))
    base = _mixc(nt, halo, base, (0.20, 0.10, 0.05, 1), (-600, 300))
    # grime: sleeper ends and the sides that sit in the ballast
    ends = _smooth(nt, ax, 1.0, 1.3, (-1150, -1250))
    geo = _n(nt, "ShaderNodeNewGeometry", (-1800, -1450))
    gz = _n(nt, "ShaderNodeSeparateXYZ", (-1600, -1450))
    nt.links.new(geo.outputs["Normal"], gz.inputs[0])
    side = _m(nt, "SUBTRACT", 1.0, _smooth(nt, gz.outputs["Z"], 0.5, 0.9, (-1400, -1450)), loc=(-1200, -1450))
    gn = _noise(nt, tc.outputs["Object"], 3.0, 5.0, loc=(-1150, -1650))
    grime = _m(nt, "ADD", _m(nt, "MULTIPLY", ends, 0.35), _m(nt, "MULTIPLY", side, 0.55), loc=(-900, -1350))
    grime = _m(nt, "MULTIPLY", _m(nt, "ADD", grime, _m(nt, "MULTIPLY", gn.outputs["Fac"], 0.25)), 1.0,
               loc=(-750, -1350), clamp=True)
    base = _mixc(nt, grime, base, (0.12, 0.10, 0.08, 1), (-400, 300))
    # brownish ballast dust settles on the whole top surface
    dn = _noise(nt, tc.outputs["Object"], 9.0, 4.0, loc=(-700, -1700))
    dust = _m(nt, "ADD", _m(nt, "MULTIPLY", dn.outputs["Fac"], 0.3), 0.12, loc=(-550, -1700))
    base = _mixc(nt, dust, base, (0.22, 0.17, 0.12, 1), (-250, 300))
    nt.links.new(base, bsdf.inputs["Base Color"])
    if texset and rough is not None:
        nt.links.new(rough, bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.88
    _set_spec(bsdf, 0.35)
    return mat


def rail_head_material(name, texset=None):
    """Polished running band on top of the head, oxidised steel on the head sides (by surface normal)."""
    mat, nt, bsdf = _new(name)
    geo = _n(nt, "ShaderNodeNewGeometry", (-1200, 0))
    sep = _n(nt, "ShaderNodeSeparateXYZ", (-1000, 0))
    nt.links.new(geo.outputs["Normal"], sep.inputs[0])
    top = _smooth(nt, sep.outputs["Z"], 0.88, 0.975, (-800, 0))
    tc = _n(nt, "ShaderNodeTexCoord", (-1200, -400))
    nz = _noise(nt, tc.outputs["Object"], 2.5, 6.0, loc=(-1000, -400))
    side_col = _ramp(nt, (-800, -400), [(0.35, (0.13, 0.075, 0.045, 1)), (0.65, (0.24, 0.14, 0.08, 1))])
    nt.links.new(nz.outputs["Fac"], side_col.inputs["Fac"])
    streak = _noise(nt, tc.outputs["Object"], 120.0, 2.0, loc=(-1000, 300))
    band = _mixc(nt, 0.15, (0.60, 0.60, 0.61, 1), streak.outputs["Fac"], (-600, 300), "OVERLAY")
    nt.links.new(_mixc(nt, top, side_col.outputs["Color"], band, (-300, 200)), bsdf.inputs["Base Color"])
    rough = _n(nt, "ShaderNodeMapRange", (-500, -100))
    nt.links.new(top, rough.inputs["Value"])
    rough.inputs["To Min"].default_value = 0.62
    rough.inputs["To Max"].default_value = 0.2
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    met = _n(nt, "ShaderNodeMapRange", (-500, -300))
    nt.links.new(top, met.inputs["Value"])
    met.inputs["To Min"].default_value = 0.25
    met.inputs["To Max"].default_value = 1.0
    nt.links.new(met.outputs["Result"], bsdf.inputs["Metallic"])
    return mat


def weathered_metal(name, base, rough=0.55, metallic=0.0, rust=(0.20, 0.12, 0.07, 1)):
    """Painted / galvanised steel with vertical rain streaks, splash dirt near the ground (object Z)
    and roughness variation."""
    def build(name, texset=None):
        mat, nt, bsdf = _new(name)
        tc = _n(nt, "ShaderNodeTexCoord", (-1400, 0))
        mp = _n(nt, "ShaderNodeMapping", (-1200, 0))
        mp.inputs["Scale"].default_value = (25.0, 25.0, 0.6)      # streaks run down the mast
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        st = _noise(nt, mp.outputs["Vector"], 1.0, 4.0, loc=(-1000, 200))
        streak = _smooth(nt, st.outputs["Fac"], 0.45, 0.75, (-800, 200))
        col = _mixc(nt, _m(nt, "MULTIPLY", streak, 0.45, loc=(-650, 200)), base, (base[0] * 0.55, base[1] * 0.55, base[2] * 0.52, 1), (-500, 300))
        sep = _n(nt, "ShaderNodeSeparateXYZ", (-1000, -300))
        nt.links.new(tc.outputs["Object"], sep.inputs[0])
        splash = _m(nt, "SUBTRACT", 1.0, _smooth(nt, sep.outputs["Z"], 0.2, 1.4, (-800, -300)), loc=(-650, -300))
        sn = _noise(nt, tc.outputs["Object"], 6.0, 4.0, loc=(-800, -500))
        splash = _m(nt, "MULTIPLY", splash, _smooth(nt, sn.outputs["Fac"], 0.3, 0.6, (-650, -500)), loc=(-500, -400))
        col = _mixc(nt, splash, col, rust, (-300, 200))
        nt.links.new(col, bsdf.inputs["Base Color"])
        rn = _noise(nt, tc.outputs["Object"], 3.0, 3.0, loc=(-800, -800))
        rr = _n(nt, "ShaderNodeMapRange", (-500, -800))
        nt.links.new(rn.outputs["Fac"], rr.inputs["Value"])
        rr.inputs["To Min"].default_value = rough - 0.15
        rr.inputs["To Max"].default_value = min(1.0, rough + 0.2)
        nt.links.new(rr.outputs["Result"], bsdf.inputs["Roughness"])
        bsdf.inputs["Metallic"].default_value = metallic
        return mat
    return build


def fastening_material(name, color, rough, metallic=0.0, rusty=0.0):
    def build(name, texset=None):
        mat, nt, bsdf = _new(name)
        tc = _n(nt, "ShaderNodeTexCoord", (-800, 0))
        nz = _noise(nt, tc.outputs["Object"], 40.0, 4.0, loc=(-600, 0))
        col = _mixc(nt, _m(nt, "MULTIPLY", nz.outputs["Fac"], rusty, loc=(-400, 100)), color, (0.22, 0.11, 0.05, 1), (-200, 100))
        nt.links.new(col, bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = metallic
        return mat
    return build


_SPECS = {
    "ballast":      dict(build=ballast_material),
    "sleeper":      dict(build=sleeper_material),
    "fastening_plastic": dict(build=fastening_material("fp", (0.030, 0.030, 0.030, 1), 0.55, 0.0, 0.15)),
    "fastening_steel":   dict(build=fastening_material("fs", (0.07, 0.065, 0.06, 1), 0.6, 0.6, 0.8)),
    "rail_body":    dict(proc=proc_rail_body, tex=dict(tile_m=0.8, metallic=0.3, tint=(0.62, 0.60, 0.58, 1))),
    "rail_head":    dict(build=rail_head_material),
    "grass":        dict(proc=proc_grass, tex=dict(tile_m=3.0)),
    "forest_floor": dict(proc=proc_forest_floor, tex=dict(tile_m=3.0)),
    "field":        dict(proc=proc_field, tex=dict(tile_m=4.0)),
    "soil":         dict(proc=lambda n: proc_concrete(n, (0.30, 0.24, 0.17, 1), 0.9, 10.0), tex=dict(tile_m=2.0)),
    "asphalt":      dict(proc=proc_asphalt, tex=dict(tile_m=3.0)),
    "concrete":     dict(proc=lambda n: proc_concrete(n, (0.40, 0.39, 0.37, 1), 0.85, 6.0),
                         tex=dict(tile_m=2.0, tint=(2.1, 2.1, 2.05, 1))),
    "galvanised":   dict(build=weathered_metal("g", (0.50, 0.51, 0.52, 1), 0.5, 0.85)),
    "mast_paint":   dict(build=weathered_metal("m", (0.33, 0.345, 0.33, 1), 0.6, 0.0)),
    "signal_black": dict(proc=lambda n: proc_flat(n, (0.02, 0.02, 0.02, 1), 0.4), tex=dict(tile_m=1.0)),
    "signal_white": dict(proc=lambda n: proc_flat(n, (0.85, 0.85, 0.85, 1), 0.5), tex=dict(tile_m=1.0)),
    "signal_red":   dict(proc=lambda n: proc_flat(n, (0.65, 0.05, 0.03, 1), 0.5), tex=dict(tile_m=1.0)),
    "lamp_red":     dict(proc=lambda n: proc_emission(n, (1.0, 0.03, 0.02, 1), 0.0), tex=dict()),
    "lamp_green":   dict(proc=lambda n: proc_emission(n, (0.05, 1.0, 0.25, 1), 0.0), tex=dict()),
    "lamp_yellow":  dict(proc=lambda n: proc_emission(n, (1.0, 0.65, 0.05, 1), 0.0), tex=dict()),
    "lamp_white":   dict(proc=lambda n: proc_emission(n, (1.0, 0.95, 0.85, 1), 0.0), tex=dict()),
    "insulator":    dict(proc=lambda n: proc_flat(n, (0.55, 0.40, 0.25, 1), 0.3), tex=dict()),
    "copper_wire":  dict(proc=lambda n: proc_flat(n, (0.45, 0.30, 0.22, 1), 0.45, 0.8), tex=dict()),
    "bark":         dict(proc=proc_bark, tex=dict(tile_m=1.0)),
    "leaves_spruce": dict(proc=lambda n: proc_leaves(n, (0.04, 0.12, 0.04, 1)), tex=dict(tile_m=1.0)),
    "leaves_deciduous": dict(proc=lambda n: proc_leaves(n, (0.08, 0.22, 0.05, 1)), tex=dict(tile_m=1.0)),
    "leaves_birch": dict(proc=lambda n: proc_leaves(n, (0.18, 0.30, 0.08, 1)), tex=dict(tile_m=1.0)),
    "noise_wall":   dict(proc=proc_noise_wall, tex=dict(tile_m=2.0)),
    "house_wall":   dict(proc=lambda n: proc_concrete(n, (0.80, 0.76, 0.68, 1), 0.8, 3.0), tex=dict(tile_m=2.0)),
    "roof_tile":    dict(proc=lambda n: proc_concrete(n, (0.45, 0.16, 0.10, 1), 0.7, 20.0), tex=dict(tile_m=1.0)),
    "platform":     dict(proc=lambda n: proc_concrete(n, (0.55, 0.54, 0.52, 1), 0.8, 12.0), tex=dict(tile_m=2.0)),
    "train_body":   dict(proc=lambda n: proc_flat(n, (0.80, 0.80, 0.82, 1), 0.35), tex=dict()),
    "train_red":    dict(proc=lambda n: proc_flat(n, (0.72, 0.04, 0.03, 1), 0.35), tex=dict()),
    "glass_dark":   dict(proc=lambda n: proc_flat(n, (0.02, 0.03, 0.04, 1), 0.1), tex=dict()),
    "andreaskreuz_white": dict(proc=lambda n: proc_flat(n, (0.9, 0.9, 0.9, 1), 0.5), tex=dict()),
}

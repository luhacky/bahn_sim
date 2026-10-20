"""Shader library for the procedural model library (bark, foliage, houses,
vehicles). Materials are cached by name in bpy.data, prefixed "hd_".

Per-instance variation uses Object Info > Random, which Cycles and Eevee
evaluate per Geometry-Nodes instance, so one mesh scattered 10 000 times does
not look cloned. Foliage also reads the mesh attributes written by trees.py
(leaf_rnd per leaf, crown_ao = inner leaves darker).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import os

import bpy

Color = Tuple[float, float, float, float]


def _new(name):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat, None, None
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, bsdf


def _n(nt, idname, loc=(0, 0), **props):
    n = nt.nodes.new(idname)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n


def _ramp(nt, loc, stops, interp="LINEAR"):
    r = _n(nt, "ShaderNodeValToRGB", loc)
    cr = r.color_ramp
    cr.interpolation = interp
    while len(cr.elements) > len(stops):
        cr.elements.remove(cr.elements[-1])
    while len(cr.elements) < len(stops):
        cr.elements.new(0.5)
    for el, (pos, col) in zip(cr.elements, stops):
        el.position = pos
        el.color = col if len(col) == 4 else (*col, 1.0)
    return r


def _math(nt, op, a=None, b=None, loc=(0, 0), clamp=False):
    m = _n(nt, "ShaderNodeMath", loc, operation=op)
    m.use_clamp = clamp
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            m.inputs[i].default_value = x
        else:
            nt.links.new(x, m.inputs[i])
    return m.outputs[0]


def _mix_rgb(nt, fac, a, b, loc=(0, 0), blend="MIX"):
    m = _n(nt, "ShaderNodeMix", loc, data_type="RGBA", blend_type=blend)
    for sock, x in ((m.inputs[0], fac), (m.inputs[6], a), (m.inputs[7], b)):
        if isinstance(x, (int, float)):
            sock.default_value = x
        elif isinstance(x, tuple):
            sock.default_value = x
        else:
            nt.links.new(x, sock)
    return m.outputs[2]


def _set(bsdf, key, val):
    if key in bsdf.inputs:
        bsdf.inputs[key].default_value = val


def _bump(nt, bsdf, height, strength=0.4, dist=0.02, loc=(0, -400)):
    b = _n(nt, "ShaderNodeBump", loc)
    b.inputs["Strength"].default_value = strength
    b.inputs["Distance"].default_value = dist
    nt.links.new(height, b.inputs["Height"])
    nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])


def _obj_random(nt, loc=(-1200, 600)):
    oi = _n(nt, "ShaderNodeObjectInfo", loc)
    return oi.outputs["Random"]


def _attr(nt, name, loc):
    a = _n(nt, "ShaderNodeAttribute", loc, attribute_name=name, attribute_type="GEOMETRY")
    return a.outputs["Fac"]


# ----------------------------------------------------------------------
# bark
# ----------------------------------------------------------------------
BARK = {
    # base, dark, fissure depth, uv stretch (around, along)
    "spruce": ((0.26, 0.19, 0.14, 1), (0.10, 0.07, 0.05, 1), 0.5, (10.0, 3.0)),
    "pine":   ((0.30, 0.22, 0.17, 1), (0.10, 0.07, 0.05, 1), 0.8, (6.0, 1.2)),
    "beech":  ((0.30, 0.30, 0.28, 1), (0.18, 0.19, 0.17, 1), 0.08, (6.0, 2.0)),
    "oak":    ((0.25, 0.22, 0.19, 1), (0.07, 0.06, 0.05, 1), 0.9, (8.0, 1.5)),
    "birch":  ((0.80, 0.79, 0.74, 1), (0.04, 0.04, 0.04, 1), 0.2, (4.0, 1.0)),
    "maple":  ((0.32, 0.29, 0.25, 1), (0.14, 0.12, 0.10, 1), 0.5, (8.0, 2.0)),
    "shrub":  ((0.30, 0.26, 0.21, 1), (0.15, 0.12, 0.10, 1), 0.2, (6.0, 2.0)),
    "sapling": ((0.33, 0.30, 0.25, 1), (0.16, 0.13, 0.10, 1), 0.2, (6.0, 2.0)),
}


# photo-scanned bark (assets/textures/bark_<species>, Poly Haven CC0): texture height in metres
BARK_SCANS = {"pine": 2.0, "spruce": 1.0, "oak": 2.0}
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")


def _bark_scan(nt, bsdf, uv, species):
    """Image bark on the tube UVs (u = once around, v = metres up). Returns the colour socket or None."""
    from ..materials import find_texture_set
    if species not in BARK_SCANS:
        return None
    ts = find_texture_set(ASSET_DIR, f"bark_{species}")
    if not ts:
        return None
    mp = _n(nt, "ShaderNodeMapping", (-1200, -800))
    mp.inputs["Scale"].default_value = (1.0, 1.0 / BARK_SCANS[species], 1.0)
    nt.links.new(uv, mp.inputs["Vector"])

    def img(path, loc, non_color=False):
        node = _n(nt, "ShaderNodeTexImage", loc)
        node.image = bpy.data.images.load(path, check_existing=True)
        if non_color:
            node.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mp.outputs["Vector"], node.inputs["Vector"])
        return node.outputs["Color"]

    if "rough" in ts:
        nt.links.new(img(ts["rough"], (-950, -1000), True), bsdf.inputs["Roughness"])
    if "normal" in ts:
        nm = _n(nt, "ShaderNodeNormalMap", (-600, -1200))
        nm.inputs["Strength"].default_value = 1.2
        nt.links.new(img(ts["normal"], (-950, -1200), True), nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return img(ts["color"], (-950, -800))


def bark(species: str):
    name = f"hd_bark_{species}"
    mat, nt, bsdf = _new(name)
    if nt is None:
        return mat
    base, dark, depth, (su, sv) = BARK.get(species, BARK["oak"])
    uv = _n(nt, "ShaderNodeTexCoord", (-1400, 0)).outputs["UV"]
    scan = _bark_scan(nt, bsdf, uv, species)
    mp = _n(nt, "ShaderNodeMapping", (-1200, 0))
    mp.inputs["Scale"].default_value = (su, sv, 1.0)
    nt.links.new(uv, mp.inputs["Vector"])
    vor = _n(nt, "ShaderNodeTexVoronoi", (-950, 100), feature="DISTANCE_TO_EDGE")
    vor.inputs["Scale"].default_value = 2.0
    nt.links.new(mp.outputs["Vector"], vor.inputs["Vector"])
    nz = _n(nt, "ShaderNodeTexNoise", (-950, -150))
    nz.inputs["Scale"].default_value = 6.0
    nz.inputs["Detail"].default_value = 8.0
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    fiss = _ramp(nt, (-700, 100), [(0.0, (0, 0, 0, 1)), (0.12, (1, 1, 1, 1))])
    nt.links.new(vor.outputs["Distance"], fiss.inputs["Fac"])
    col = _mix_rgb(nt, fiss.outputs["Color"], dark, base, (-450, 100))
    col = _mix_rgb(nt, nz.outputs["Fac"], col, (base[0] * 0.8, base[1] * 0.85, base[2] * 0.8, 1), (-250, 100), "OVERLAY")
    if scan is not None:
        col = scan
    if species == "birch":
        # black horizontal lenticels + dark, furrowed foot of the trunk
        mp2 = _n(nt, "ShaderNodeMapping", (-1200, 400))
        mp2.inputs["Scale"].default_value = (3.0, 40.0, 1.0)
        nt.links.new(uv, mp2.inputs["Vector"])
        nz2 = _n(nt, "ShaderNodeTexNoise", (-950, 400))
        nz2.inputs["Scale"].default_value = 1.5
        nt.links.new(mp2.outputs["Vector"], nz2.inputs["Vector"])
        dash = _ramp(nt, (-700, 400), [(0.62, (1, 1, 1, 1)), (0.66, (0, 0, 0, 1))], "CONSTANT")
        nt.links.new(nz2.outputs["Fac"], dash.inputs["Fac"])
        col = _mix_rgb(nt, dash.outputs["Color"], (0.05, 0.05, 0.05, 1), col, (-100, 300))
        oc = _n(nt, "ShaderNodeTexCoord", (-1400, 700)).outputs["Object"]
        sep = _n(nt, "ShaderNodeSeparateXYZ", (-1200, 700))
        nt.links.new(oc, sep.inputs[0])
        foot = _ramp(nt, (-900, 700), [(0.0, (1, 1, 1, 1)), (1.0, (0, 0, 0, 1))])
        mr = _math(nt, "MAP_RANGE" if False else "DIVIDE", sep.outputs["Z"], 2.2, (-1050, 700), clamp=True)
        nt.links.new(mr, foot.inputs["Fac"])
        col = _mix_rgb(nt, foot.outputs["Color"], col, (0.12, 0.11, 0.10, 1), (50, 400))
    if species == "pine":
        # orange "mirror" bark in the upper trunk and branches
        oc = _n(nt, "ShaderNodeTexCoord", (-1400, 700)).outputs["Object"]
        sep = _n(nt, "ShaderNodeSeparateXYZ", (-1200, 700))
        nt.links.new(oc, sep.inputs[0])
        up = _ramp(nt, (-900, 700), [(0.45, (0, 0, 0, 1)), (0.62, (1, 1, 1, 1))])
        nt.links.new(_math(nt, "DIVIDE", sep.outputs["Z"], 24.0, (-1050, 700), clamp=True), up.inputs["Fac"])
        col = _mix_rgb(nt, up.outputs["Color"], col, (0.62, 0.33, 0.15, 1), (50, 400))
    nt.links.new(col, bsdf.inputs["Base Color"])
    if scan is not None:              # roughness and normal come from the scan
        return mat
    _set(bsdf, "Roughness", 0.85 if species != "beech" else 0.6)
    h = _math(nt, "ADD", _math(nt, "MULTIPLY", fiss.outputs["Color"], depth, (-450, -300)),
              _math(nt, "MULTIPLY", nz.outputs["Fac"], 0.2, (-450, -450)), (-250, -350))
    _bump(nt, bsdf, h, 0.6, 0.02, (0, -350))
    return mat


# ----------------------------------------------------------------------
# foliage
# ----------------------------------------------------------------------
FOLIAGE = {
    # dark, light, translucent tint, instance hue jitter
    "spruce": ((0.020, 0.050, 0.022, 1), (0.050, 0.110, 0.045, 1), (0.10, 0.25, 0.05, 1), 0.015),
    "pine":   ((0.035, 0.070, 0.035, 1), (0.090, 0.150, 0.070, 1), (0.15, 0.30, 0.08, 1), 0.015),
    "beech":  ((0.040, 0.095, 0.020, 1), (0.130, 0.240, 0.045, 1), (0.30, 0.50, 0.05, 1), 0.02),
    "oak":    ((0.035, 0.080, 0.020, 1), (0.110, 0.200, 0.045, 1), (0.25, 0.45, 0.05, 1), 0.02),
    "birch":  ((0.060, 0.120, 0.025, 1), (0.180, 0.300, 0.060, 1), (0.35, 0.55, 0.08, 1), 0.02),
    "maple":  ((0.045, 0.100, 0.022, 1), (0.150, 0.260, 0.050, 1), (0.30, 0.50, 0.05, 1), 0.03),
    "shrub":  ((0.040, 0.090, 0.022, 1), (0.130, 0.220, 0.050, 1), (0.25, 0.45, 0.06, 1), 0.03),
    "sapling": ((0.050, 0.110, 0.025, 1), (0.160, 0.280, 0.055, 1), (0.30, 0.50, 0.06, 1), 0.03),
}


LEAF_ALBEDO = 1.25     # lighting overhaul: leaves were on the dark side of measured reflectance
LEAF_TRANSMIT = 0.55


def foliage(species: str):
    name = f"hd_foliage_{species}"
    mat, nt, bsdf = _new(name)
    if nt is None:
        return mat
    dark, light, trans, jitter = FOLIAGE.get(species, FOLIAGE["beech"])
    dark, light = (tuple(min(1.0, c * LEAF_ALBEDO) for c in v[:3]) + (1,) for v in (dark, light))
    rnd = _attr(nt, "leaf_rnd", (-1400, 300))
    ao = _attr(nt, "crown_ao", (-1400, 100))
    orand = _obj_random(nt, (-1400, 600))
    ramp = _ramp(nt, (-1100, 300), [(0.0, dark), (1.0, light)])
    fac = _math(nt, "ADD", _math(nt, "MULTIPLY", rnd, 0.55, (-1250, 350)),
                _math(nt, "MULTIPLY", orand, 0.45, (-1250, 500)), (-1180, 400))
    nt.links.new(fac, ramp.inputs["Fac"])
    hsv = _n(nt, "ShaderNodeHueSaturation", (-850, 300))
    nt.links.new(ramp.outputs["Color"], hsv.inputs["Color"])
    hue = _math(nt, "ADD", _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", orand, 0.5, (-1100, 650)), jitter * 2, (-1000, 650)),
                0.5, (-900, 650))
    nt.links.new(hue, hsv.inputs["Hue"])
    shade = _math(nt, "ADD", _math(nt, "MULTIPLY", ao, 0.75, (-1100, 50)), 0.25, (-950, 50))
    col = _mix_rgb(nt, 1.0, hsv.outputs["Color"], (1, 1, 1, 1), (-600, 250), "MULTIPLY")
    m = col.node
    nt.links.new(_n(nt, "ShaderNodeCombineColor", (-800, 50)).outputs[0], m.inputs[7])
    cc = m.inputs[7].links[0].from_node
    for i in range(3):
        nt.links.new(shade, cc.inputs[i])
    nt.links.new(col, bsdf.inputs["Base Color"])
    _set(bsdf, "Roughness", 0.55 if species not in ("spruce", "pine") else 0.7)
    _set(bsdf, "Specular IOR Level", 0.35)
    # translucency: mix in a Translucent BSDF (cheap, reads well against the sun)
    out = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"][0]
    tr = _n(nt, "ShaderNodeBsdfTranslucent", (250, -250))
    # transmitted light is the translucent tint itself (a leaf passes ~10 to 25 % of green light),
    # dimmed only by the crown shading; multiplying by the reflected colour as before left it near zero
    tcol = _mix_rgb(nt, 1.0, tuple(c * LEAF_TRANSMIT for c in trans[:3]) + (1,), (1, 1, 1, 1), (0, -250), "MULTIPLY")
    nt.links.new(cc.outputs[0], tcol.node.inputs[7])
    nt.links.new(tcol, tr.inputs["Color"])
    mix = _n(nt, "ShaderNodeMixShader", (450, 0))
    mix.inputs[0].default_value = 0.32 if species not in ("spruce", "pine") else 0.12
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def grass_blades():
    name = "hd_grass_blades"
    mat, nt, bsdf = _new(name)
    if nt is None:
        return mat
    uv = _n(nt, "ShaderNodeTexCoord", (-1400, 0)).outputs["UV"]
    sep = _n(nt, "ShaderNodeSeparateXYZ", (-1200, 0))
    nt.links.new(uv, sep.inputs[0])
    rnd = _attr(nt, "leaf_rnd", (-1400, 300))
    orand = _obj_random(nt, (-1400, 500))
    ramp = _ramp(nt, (-900, 300), [(0.0, (0.05, 0.09, 0.02, 1)), (0.55, (0.13, 0.20, 0.045, 1)),
                                   (0.85, (0.24, 0.26, 0.09, 1)), (1.0, (0.40, 0.36, 0.20, 1))])
    f = _math(nt, "ADD", _math(nt, "MULTIPLY", rnd, 0.7, (-1150, 300)), _math(nt, "MULTIPLY", orand, 0.3, (-1150, 450)), (-1050, 350))
    nt.links.new(f, ramp.inputs["Fac"])
    # darker at the base (self-shadowing in the sward)
    base = _ramp(nt, (-900, 0), [(0.0, (0.35, 0.35, 0.35, 1)), (0.6, (1, 1, 1, 1))])
    nt.links.new(sep.outputs["Y"], base.inputs["Fac"])
    col = _mix_rgb(nt, 1.0, ramp.outputs["Color"], base.outputs["Color"], (-600, 200), "MULTIPLY")
    nt.links.new(col, bsdf.inputs["Base Color"])
    _set(bsdf, "Roughness", 0.6)
    _set(bsdf, "Specular IOR Level", 0.3)
    out = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"][0]
    tr = _n(nt, "ShaderNodeBsdfTranslucent", (250, -250))
    nt.links.new(col, tr.inputs["Color"])
    mix = _n(nt, "ShaderNodeMixShader", (450, 0))
    mix.inputs[0].default_value = 0.2
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def flat(name, color, rough=0.5, metallic=0.0, emission=0.0, transmission=0.0, coat=0.0, ior=1.45):
    mat, nt, bsdf = _new(f"hd_{name}")
    if nt is None:
        return mat
    bsdf.inputs["Base Color"].default_value = color
    _set(bsdf, "Roughness", rough)
    _set(bsdf, "Metallic", metallic)
    if coat:
        _set(bsdf, "Coat Weight", coat)
        _set(bsdf, "Coat Roughness", 0.05)
    if transmission:
        _set(bsdf, "Transmission Weight", transmission)
        _set(bsdf, "IOR", ior)
    if emission:
        _set(bsdf, "Emission Color", color)
        _set(bsdf, "Emission Strength", emission)
    return mat


def palette_paint(name: str, colors: Sequence[Color], rough=0.3, metallic=0.5, coat=1.0):
    """Car paint picking one colour per instance (Object Info > Random, constant ramp)."""
    mat, nt, bsdf = _new(f"hd_{name}")
    if nt is None:
        return mat
    ramp = _ramp(nt, (-500, 0), [(i / len(colors), c) for i, c in enumerate(colors)], "CONSTANT")
    nt.links.new(_obj_random(nt, (-800, 0)), ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    _set(bsdf, "Roughness", rough)
    _set(bsdf, "Metallic", metallic)
    _set(bsdf, "Coat Weight", coat)
    _set(bsdf, "Coat Roughness", 0.03)
    return mat


def uv_pattern(name, kind, colors: Sequence[Color], rough=0.8, scale=(1.0, 1.0), bump=0.3, per_instance=True):
    """Box/UV-mapped architectural patterns (UVs are in metres):
    plaster, roof_tiles, brick, clinker, boards, corrugated, paving, gravel."""
    mat, nt, bsdf = _new(f"hd_{name}")
    if nt is None:
        return mat
    uv = _n(nt, "ShaderNodeTexCoord", (-1600, 0)).outputs["UV"]
    mp = _n(nt, "ShaderNodeMapping", (-1400, 0))
    mp.inputs["Scale"].default_value = (scale[0], scale[1], 1.0)
    nt.links.new(uv, mp.inputs["Vector"])
    vec = mp.outputs["Vector"]
    nz = _n(nt, "ShaderNodeTexNoise", (-1100, -300))
    nz.inputs["Scale"].default_value = 3.0
    nz.inputs["Detail"].default_value = 6.0
    nt.links.new(vec, nz.inputs["Vector"])
    # base colour: pick per instance from colors (constant ramp)
    pick = _ramp(nt, (-900, 400), [(i / len(colors), c) for i, c in enumerate(colors)], "CONSTANT")
    if per_instance and len(colors) > 1:
        nt.links.new(_obj_random(nt, (-1100, 400)), pick.inputs["Fac"])
    else:
        pick.inputs["Fac"].default_value = 0.0
    base = pick.outputs["Color"]
    height = nz.outputs["Fac"]
    if kind == "plaster":
        col = _mix_rgb(nt, _math(nt, "MULTIPLY", nz.outputs["Fac"], 0.25), base, (0.3, 0.3, 0.3, 1), (-500, 200), "MULTIPLY")
        col = _mix_rgb(nt, 0.3, base, col, (-300, 200))
    elif kind in ("roof_tiles", "brick", "clinker", "paving"):
        br = _n(nt, "ShaderNodeTexBrick", (-1100, 100))
        if kind == "roof_tiles":      # rows along the slope (v), 0.30 x 0.36 m pantiles
            br.inputs["Scale"].default_value = 1.0
            br.inputs["Brick Width"].default_value = 0.30
            br.inputs["Row Height"].default_value = 0.34
            br.inputs["Mortar Size"].default_value = 0.012
            br.offset = 0.5
        elif kind == "paving":
            br.inputs["Scale"].default_value = 1.0
            br.inputs["Brick Width"].default_value = 0.20
            br.inputs["Row Height"].default_value = 0.10
            br.inputs["Mortar Size"].default_value = 0.004
        else:
            br.inputs["Scale"].default_value = 1.0
            br.inputs["Brick Width"].default_value = 0.25
            br.inputs["Row Height"].default_value = 0.0725
            br.inputs["Mortar Size"].default_value = 0.008
        br.inputs["Color1"].default_value = (1, 1, 1, 1)
        br.inputs["Color2"].default_value = (0.75, 0.75, 0.75, 1)
        br.inputs["Mortar"].default_value = (0.35, 0.35, 0.35, 1)
        nt.links.new(vec, br.inputs["Vector"])
        col = _mix_rgb(nt, 1.0, base, br.outputs["Color"], (-500, 200), "MULTIPLY")
        col = _mix_rgb(nt, _math(nt, "MULTIPLY", nz.outputs["Fac"], 0.3), col, (0.2, 0.2, 0.2, 1), (-300, 200), "MULTIPLY")
        height = br.outputs["Fac"]
        if kind == "roof_tiles":
            # curved pantile profile across each tile
            wave = _n(nt, "ShaderNodeTexWave", (-1100, -600), wave_type="BANDS", bands_direction="X")
            wave.inputs["Scale"].default_value = 1.0 / 0.30 / 1.0
            wave.inputs["Scale"].default_value = 3.33
            nt.links.new(vec, wave.inputs["Vector"])
            height = _math(nt, "ADD", _math(nt, "MULTIPLY", wave.outputs["Fac"], 0.5), _math(nt, "MULTIPLY", br.outputs["Fac"], 1.0))
    elif kind == "boards":             # vertical timber cladding
        wave = _n(nt, "ShaderNodeTexWave", (-1100, 100), wave_type="BANDS", bands_direction="X", wave_profile="SAW")
        wave.inputs["Scale"].default_value = 1.0 / 0.16 / 6.28
        wave.inputs["Scale"].default_value = 1.0
        nt.links.new(vec, wave.inputs["Vector"])
        grain = _n(nt, "ShaderNodeTexNoise", (-1100, -100))
        grain.inputs["Scale"].default_value = 8.0
        mp2 = _n(nt, "ShaderNodeMapping", (-1300, -100))
        mp2.inputs["Scale"].default_value = (20.0, 1.0, 1.0)
        nt.links.new(vec, mp2.inputs["Vector"])
        nt.links.new(mp2.outputs["Vector"], grain.inputs["Vector"])
        col = _mix_rgb(nt, _math(nt, "MULTIPLY", grain.outputs["Fac"], 0.5), base, (0.3, 0.3, 0.3, 1), (-500, 200), "MULTIPLY")
        height = wave.outputs["Fac"]
    elif kind == "corrugated":
        wave = _n(nt, "ShaderNodeTexWave", (-1100, 100), wave_type="BANDS", bands_direction="X", wave_profile="SIN")
        wave.inputs["Scale"].default_value = 1.0
        nt.links.new(vec, wave.inputs["Vector"])
        col = _mix_rgb(nt, _math(nt, "MULTIPLY", nz.outputs["Fac"], 0.15), base, (0.4, 0.3, 0.25, 1), (-500, 200), "MULTIPLY")
        height = wave.outputs["Fac"]
    else:
        col = _mix_rgb(nt, _math(nt, "MULTIPLY", nz.outputs["Fac"], 0.4), base, (0.4, 0.4, 0.4, 1), (-500, 200), "MULTIPLY")
    nt.links.new(col, bsdf.inputs["Base Color"])
    _set(bsdf, "Roughness", rough)
    if bump:
        _bump(nt, bsdf, height, bump, 0.02, (0, -400))
    return mat


def window_glass():
    """Dark reflective window glass with a random lit / curtain interior per instance."""
    mat, nt, bsdf = _new("hd_window_glass")
    if nt is None:
        return mat
    uv = _n(nt, "ShaderNodeTexCoord", (-1200, 0)).outputs["Object"]
    vor = _n(nt, "ShaderNodeTexVoronoi", (-900, 0))
    vor.inputs["Scale"].default_value = 0.6
    nt.links.new(uv, vor.inputs["Vector"])
    ramp = _ramp(nt, (-600, 0), [(0.0, (0.015, 0.018, 0.02, 1)), (0.6, (0.03, 0.03, 0.03, 1)),
                                 (0.85, (0.12, 0.11, 0.09, 1))], "CONSTANT")
    nt.links.new(vor.outputs["Color"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    _set(bsdf, "Roughness", 0.03)
    _set(bsdf, "Specular IOR Level", 0.8)
    _set(bsdf, "Coat Weight", 1.0)
    _set(bsdf, "Coat Roughness", 0.0)
    return mat


def flower_heads():
    mat, nt, bsdf = _new("hd_flower_heads")
    if nt is None:
        return mat
    ramp = _ramp(nt, (-600, 0), [(0.0, (0.85, 0.84, 0.78, 1)), (0.25, (0.85, 0.65, 0.02, 1)),
                                 (0.5, (0.35, 0.10, 0.45, 1)), (0.75, (0.35, 0.50, 0.85, 1))], "CONSTANT")
    nt.links.new(_attr(nt, "leaf_rnd", (-900, 0)), ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    _set(bsdf, "Roughness", 0.5)
    return mat


PLASTER = [(0.86, 0.85, 0.80, 1), (0.84, 0.78, 0.64, 1), (0.88, 0.82, 0.60, 1), (0.72, 0.72, 0.70, 1),
           (0.80, 0.66, 0.50, 1), (0.90, 0.88, 0.84, 1), (0.78, 0.60, 0.52, 1), (0.70, 0.74, 0.70, 1)]
ROOF = [(0.36, 0.11, 0.06, 1), (0.28, 0.08, 0.05, 1), (0.07, 0.07, 0.075, 1), (0.22, 0.12, 0.08, 1),
        (0.42, 0.16, 0.08, 1), (0.12, 0.12, 0.13, 1)]


def house_materials():
    """Materials in houses.SLOTS order."""
    return [
        uv_pattern("plaster", "plaster", PLASTER, rough=0.9, scale=(1.0, 1.0), bump=0.15),
        uv_pattern("roof_tiles", "roof_tiles", ROOF, rough=0.65, bump=0.5),
        window_glass(),
        flat("window_frame", (0.85, 0.85, 0.83, 1), 0.4),
        flat("trim_anthracite", (0.09, 0.09, 0.10, 1), 0.6),
        uv_pattern("timber_boards", "boards", [(0.28, 0.17, 0.09, 1), (0.20, 0.13, 0.08, 1), (0.30, 0.24, 0.18, 1)],
                   rough=0.85, scale=(6.0, 1.0), bump=0.4),
        uv_pattern("clinker", "brick", [(0.40, 0.14, 0.08, 1), (0.30, 0.12, 0.08, 1), (0.50, 0.30, 0.18, 1)],
                   rough=0.8, bump=0.4),
        uv_pattern("metal_profile", "corrugated", [(0.55, 0.57, 0.58, 1), (0.40, 0.45, 0.50, 1), (0.60, 0.58, 0.52, 1),
                                                    (0.25, 0.30, 0.35, 1)], rough=0.45, scale=(5.0, 1.0), bump=0.3),
        uv_pattern("paving_grey", "paving", [(0.45, 0.44, 0.42, 1)], rough=0.85, bump=0.2, per_instance=False),
        flat("roller_shutter", (0.45, 0.45, 0.45, 1), 0.5),
    ]


def number_plate():
    """EU number plate: white field, blue band on the left, dark glyph blocks."""
    mat, nt, bsdf = _new("hd_number_plate")
    if nt is None:
        return mat
    uv = _n(nt, "ShaderNodeTexCoord", (-1200, 0)).outputs["UV"]
    sep = _n(nt, "ShaderNodeSeparateXYZ", (-1000, 0))
    nt.links.new(uv, sep.inputs[0])
    band = _math(nt, "LESS_THAN", sep.outputs["X"], 0.09, (-800, 100))
    mp = _n(nt, "ShaderNodeMapping", (-1000, -300))
    mp.inputs["Scale"].default_value = (9.0, 1.0, 1.0)
    nt.links.new(uv, mp.inputs["Vector"])
    chk = _n(nt, "ShaderNodeTexChecker", (-800, -300))
    chk.inputs["Scale"].default_value = 1.0
    nt.links.new(mp.outputs["Vector"], chk.inputs["Vector"])
    inner = _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", sep.outputs["Y"], 0.22), _math(nt, "LESS_THAN", sep.outputs["Y"], 0.78))
    inner = _math(nt, "MULTIPLY", inner, _math(nt, "GREATER_THAN", sep.outputs["X"], 0.14))
    inner = _math(nt, "MULTIPLY", inner, _math(nt, "LESS_THAN", sep.outputs["X"], 0.95))
    glyph = _math(nt, "MULTIPLY", inner, chk.outputs["Fac"])
    col = _mix_rgb(nt, glyph, (0.85, 0.85, 0.83, 1), (0.02, 0.02, 0.02, 1), (-500, -100))
    col = _mix_rgb(nt, band, col, (0.02, 0.08, 0.40, 1), (-300, 0))
    nt.links.new(col, bsdf.inputs["Base Color"])
    _set(bsdf, "Roughness", 0.35)
    return mat

"""Rain: falling streaks, windscreen droplets / streaks, wiper and wet surfaces.

* RainStreaks: a Geometry Nodes particle field parented to the cab camera.
  N points are scattered in a box 1.6 to 30 m ahead of the eye, each
  instancing a thin water cylinder aligned with the drop velocity *relative
  to the train* (fall speed + head wind from the train speed + crosswind);
  its length is |v_rel| x shutter time, so at speed the streaks fan out from
  the vanishing point like in real cab footage. Positions are re-seeded
  from seed + scene frame, so every snapshot gets a new drop field.
  Drops further away than ~30 m are not resolvable; their effect is the rain
  extinction added to the fog volume (conditions.__init__).
* Windscreen: a glass pane just in front of the eye (camera-parented, visible
  to camera rays only). Its shader is transparent except where water sits:
  round drops (Voronoi cells, random radius, hemispherical height -> bump
  normal) and elongated streaks pushed upward by the airflow, rendered with
  a water Glass BSDF, so each drop refracts the scene behind it.
* Wiper: an arm pivoting at the right end of the parked wiper modelled by
  train_rig; it sweeps with a triangle wave in time. Water re-accumulates
  behind the blade: in the swept sector the drop density is
  min(1, time since the blade last passed / refill time). The phase comes
  from cond["time_s"] or scene frame / fps; a frame-change handler keeps it
  in sync when the renderer steps frames.
* Wetness: every scene material gets a tiny node insert (once) that darkens
  base colour and lowers roughness by the View Layer attribute
  "sbahn_wetness" (scene custom property), so wet ballast, rails and
  platforms turn dark and glossy.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Matrix

from .nodes import NodeBuilder

RAIN_OBJ = "RainStreaks"
GLASS_OBJ = "Windscreen"
WIPER_OBJ = "WiperArm"
GLASS_DIST = 1.10                 # m in front of the eye (A-pillars at 1.15)
WIPER_PIVOT = (0.65, -0.45)       # camera-local x, y of the parked wiper's pivot end
WIPER_LEN = 1.0
WIPER_PARK_DEG = 180.0            # arm points left when parked
WIPER_SWEEP_DEG = -105.0          # sweeps up to 75 deg
DROP_FALL_MS = 7.0
BOX = dict(x=(-9.0, 9.0), y=(-2.6, 6.0), z=(-30.0, -1.6))   # camera-local, -Z forward


def _camera_only(ob):
    for attr in ("visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter", "visible_shadow"):
        if hasattr(ob, attr):
            setattr(ob, attr, False)
    ob.visible_camera = True
    ob["sbahn_non_occluding"] = True


def _collection():
    return bpy.data.collections.get("Lighting") or bpy.context.scene.collection


def _parent_to_cam(ob, cam, basis=Matrix.Identity(4)):
    if cam is not None:
        ob.parent = cam
        ob.matrix_parent_inverse = Matrix.Identity(4)
    ob.matrix_basis = basis


# ---------------------------------------------------------------- rain streaks
def _streak_material():
    mat = bpy.data.materials.get("SB_RainStreak") or bpy.data.materials.new("SB_RainStreak")
    try:
        mat.use_nodes = True
    except Exception:
        pass
    g = NodeBuilder(mat.node_tree, clear=True)
    out = g.node("ShaderNodeOutputMaterial")
    glass = g.node("ShaderNodeBsdfGlass")
    glass.inputs["IOR"].default_value = 1.33
    glass.inputs["Roughness"].default_value = 0.05
    tr = g.node("ShaderNodeBsdfTransparent")
    mix = g.node("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.45          # streaks are faint: motion-blurred drops
    g.link(tr.outputs[0], mix.inputs[1])
    g.link(glass.outputs[0], mix.inputs[2])
    g.link(mix.outputs[0], out.inputs["Surface"])
    return mat


def _streak_tree():
    tree = bpy.data.node_groups.get("SB_RainStreaks")
    if tree is not None:
        return tree
    tree = bpy.data.node_groups.new("SB_RainStreaks", "GeometryNodeTree")
    tree.is_modifier = True
    iface = tree.interface
    iface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    iface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    for name, typ in (("Count", "NodeSocketInt"), ("Seed", "NodeSocketInt"), ("Box Min", "NodeSocketVector"),
                      ("Box Max", "NodeSocketVector"), ("Velocity", "NodeSocketVector"), ("Length", "NodeSocketFloat"),
                      ("Radius", "NodeSocketFloat"), ("Material", "NodeSocketMaterial")):
        iface.new_socket(name, in_out="INPUT", socket_type=typ)
    n = tree.nodes
    L = tree.links.new
    gi = n.new("NodeGroupInput")
    go = n.new("NodeGroupOutput")
    st = n.new("GeometryNodeInputSceneTime")
    seed = n.new("ShaderNodeMath"); seed.operation = "ADD"
    L(gi.outputs["Seed"], seed.inputs[0]); L(st.outputs["Frame"], seed.inputs[1])
    rpos = n.new("FunctionNodeRandomValue"); rpos.data_type = "FLOAT_VECTOR"
    L(gi.outputs["Box Min"], rpos.inputs[0]); L(gi.outputs["Box Max"], rpos.inputs[1]); L(seed.outputs[0], rpos.inputs["Seed"])
    pts = n.new("GeometryNodePoints")
    L(gi.outputs["Count"], pts.inputs["Count"]); L(rpos.outputs[0], pts.inputs["Position"])
    cyl = n.new("GeometryNodeMeshCylinder")
    cyl.inputs["Vertices"].default_value = 4
    cyl.fill_type = "NONE"
    cyl.inputs["Radius"].default_value = 1.0
    cyl.inputs["Depth"].default_value = 1.0
    align = n.new("FunctionNodeAlignRotationToVector"); align.axis = "Z"
    L(gi.outputs["Velocity"], align.inputs["Vector"])
    # per-drop scale: radius x radius x length, each jittered
    rs = n.new("FunctionNodeRandomValue"); rs.data_type = "FLOAT"
    rs.inputs[2].default_value, rs.inputs[3].default_value = 0.6, 1.4
    seed2 = n.new("ShaderNodeMath"); seed2.operation = "ADD"; seed2.inputs[1].default_value = 7919
    L(seed.outputs[0], seed2.inputs[0]); L(seed2.outputs[0], rs.inputs["Seed"])
    r = n.new("ShaderNodeMath"); r.operation = "MULTIPLY"; L(gi.outputs["Radius"], r.inputs[0]); L(rs.outputs[1], r.inputs[1])
    ln = n.new("ShaderNodeMath"); ln.operation = "MULTIPLY"; L(gi.outputs["Length"], ln.inputs[0]); L(rs.outputs[1], ln.inputs[1])
    sc = n.new("ShaderNodeCombineXYZ"); L(r.outputs[0], sc.inputs[0]); L(r.outputs[0], sc.inputs[1]); L(ln.outputs[0], sc.inputs[2])
    iop = n.new("GeometryNodeInstanceOnPoints")
    L(pts.outputs[0], iop.inputs["Points"]); L(cyl.outputs["Mesh"], iop.inputs["Instance"])
    L(align.outputs[0], iop.inputs["Rotation"]); L(sc.outputs[0], iop.inputs["Scale"])
    sm = n.new("GeometryNodeSetMaterial")
    L(iop.outputs[0], sm.inputs["Geometry"]); L(gi.outputs["Material"], sm.inputs["Material"])
    L(sm.outputs[0], go.inputs[0])
    return tree


def _rain_streaks(cam, p, rr):
    ob = bpy.data.objects.get(RAIN_OBJ)
    if rr <= 0:
        if ob:
            ob.hide_render = True
        return dict(rain_streaks=0)
    if ob is None:
        ob = bpy.data.objects.new(RAIN_OBJ, bpy.data.meshes.new(RAIN_OBJ))
        _collection().objects.link(ob)
        ob.modifiers.new("Rain", "NODES")
    _parent_to_cam(ob, cam)
    _camera_only(ob)
    ob.hide_render = False
    mod = ob.modifiers["Rain"]
    mod.node_group = _streak_tree()
    v_train = float(p["train_speed_kmh"]) / 3.6
    wind = float(p.get("wind_ms", 0.0))
    vel = (wind, -DROP_FALL_MS, v_train)                          # camera local: +Z = towards the camera
    speed = math.sqrt(sum(c * c for c in vel))
    length = max(0.05, speed * float(p["shutter_s"]))
    vol = ((BOX["x"][1] - BOX["x"][0]) * (BOX["y"][1] - BOX["y"][0]) * (BOX["z"][1] - BOX["z"][0]))
    # visible drops (D > ~1 mm) per m^3; Marshall-Palmer gives ~ R^0.21 for all drops, the
    # large-drop share grows faster, 0.9 / m^3 at 8 mm/h matches cab footage reasonably
    n_m3 = 0.9 * (rr / 8.0) ** 0.6
    count = int(min(60000, n_m3 * vol))
    radius = 0.0009 + 0.00025 * math.log1p(rr)
    vals = {"Count": count, "Seed": int(p["seed"]) * 1009, "Box Min": (BOX["x"][0], BOX["y"][0], BOX["z"][0]),
            "Box Max": (BOX["x"][1], BOX["y"][1], BOX["z"][1]), "Velocity": vel, "Length": length,
            "Radius": radius, "Material": _streak_material()}
    for item in mod.node_group.interface.items_tree:
        if getattr(item, "in_out", None) == "INPUT" and item.name in vals:
            mod[item.identifier] = vals[item.name]
    ob.update_tag()
    return dict(rain_streaks=count, rain_streak_len_m=round(length, 3), rain_drops_per_m3=round(n_m3, 3))


# ---------------------------------------------------------------- windscreen
def _voronoi_drops(g, coord, scale, density, rmin, rmax, seed_w, stretch=1.0):
    """Returns (mask 0/1, height 0..1) for Voronoi droplets. stretch > 1 elongates vertically (camera Y)."""
    x, y, _ = g.separate(coord)
    c = g.combine(g.mul(x, scale), g.mul(y, scale / stretch), seed_w)
    vor = g.node("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="F1", distance="EUCLIDEAN")
    vor.inputs["Scale"].default_value = 1.0
    vor.inputs["Randomness"].default_value = 1.0
    g.set(vor.inputs["Vector"], c)
    rnd_r, rnd_g, rnd_b = g.separate(vor.outputs["Color"])
    present = g.math("LESS_THAN", rnd_r, density)
    radius = g.add(rmin, g.mul(rnd_g, rmax - rmin))
    q = g.div(vor.outputs["Distance"], radius)
    inside = g.math("LESS_THAN", q, 1.0)
    h = g.math("SQRT", g.math("MAXIMUM", g.sub(1.0, g.mul(q, q)), 0.0))
    mask = g.mul(present, inside)
    return mask, g.mul(h, mask)


def _glass_material(p, dens):
    mat = bpy.data.materials.get("SB_Windscreen") or bpy.data.materials.new("SB_Windscreen")
    try:
        mat.use_nodes = True
    except Exception:
        pass
    g = NodeBuilder(mat.node_tree, clear=True)
    out = g.node("ShaderNodeOutputMaterial")
    tc = g.node("ShaderNodeTexCoord")
    co = tc.outputs["Object"]                                      # metres on the pane, x right / y up
    seed_w = float(p["seed"]) * 3.7 + float(p.get("_frame", 0)) * 0.61
    # accumulation factor from the wiper: named value nodes, updated per frame by update_frame()
    acc = _wiper_accumulation(g, co)
    d_small = g.mul(acc, dens["drops"])
    d_streak = g.mul(acc, dens["streaks"])
    m1, h1 = _voronoi_drops(g, co, 55.0, d_small, 0.10, 0.34, seed_w)            # round drops, 2 - 6 mm
    m2, h2 = _voronoi_drops(g, co, 22.0, d_streak, 0.06, 0.16, seed_w + 11.0, stretch=5.0)  # airflow streaks
    height = g.math("MAXIMUM", h1, h2)
    mask = g.math("MAXIMUM", m1, m2)
    bump = g.node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = 0.004
    g.set(bump.inputs["Height"], height)
    glass = g.node("ShaderNodeBsdfGlass")
    glass.inputs["IOR"].default_value = 1.33
    glass.inputs["Roughness"].default_value = 0.03
    g.link(bump.outputs["Normal"], glass.inputs["Normal"])
    tr = g.node("ShaderNodeBsdfTransparent")
    # a wet screen is never perfectly clear: a thin film adds a little haze
    film = g.node("ShaderNodeBsdfTransparent")
    film.inputs["Color"].default_value = (0.96, 0.97, 0.98, 1.0)
    base = g.node("ShaderNodeMixShader")
    base.inputs[0].default_value = float(dens["film"])
    g.link(tr.outputs[0], base.inputs[1])
    g.link(film.outputs[0], base.inputs[2])
    mix = g.node("ShaderNodeMixShader")
    g.set(mix.inputs[0], mask)
    g.link(base.outputs[0], mix.inputs[1])
    g.link(glass.outputs[0], mix.inputs[2])
    g.link(mix.outputs[0], out.inputs["Surface"])
    return mat


def _wiper_accumulation(g, co):
    """Shader expression: 1 outside the swept sector, min(1, since / refill) inside.
    Wiper state comes from named Value nodes WiperTau (cycle time / period), WiperActive, WiperRefill."""
    tau = g.value(0.0, label="WiperTau")          # current phase 0..1 of the cycle
    active = g.value(0.0, label="WiperActive")
    refill = g.value(1.0, label="WiperRefill")    # refill time in units of the period
    x, y, _ = g.separate(co)
    dx = g.sub(x, WIPER_PIVOT[0])
    dy = g.sub(y, WIPER_PIVOT[1])
    r = g.math("SQRT", g.add(g.mul(dx, dx), g.mul(dy, dy)))
    ang = g.math("DEGREES", g.math("ARCTAN2", dy, dx))              # -180..180, park at 180
    ang = g.math("WRAP", ang, -90.0, 270.0)                         # 180 stays 180, 75 stays 75
    frac = g.div(g.sub(ang, WIPER_PARK_DEG), WIPER_SWEEP_DEG)       # 0 at park, 1 at far end
    in_sector = g.mul(g.mul(g.math("GREATER_THAN", frac, 0.0), g.math("LESS_THAN", frac, 1.0)),
                      g.mul(g.math("GREATER_THAN", r, 0.18), g.math("LESS_THAN", r, WIPER_LEN)))
    t_out = g.mul(frac, 0.5)                                        # blade passes going out
    t_back = g.sub(1.0, t_out)                                      # and coming back
    # time since last pass (cycle units)
    since_a = g.add(tau, t_out)                                     # tau < t_out: last pass was previous return
    since_b = g.sub(tau, t_out)                                     # t_out <= tau < t_back
    since_c = g.sub(tau, t_back)                                    # tau >= t_back
    lt_out = g.math("LESS_THAN", tau, t_out)
    lt_back = g.math("LESS_THAN", tau, t_back)
    since = g.add(g.mul(lt_out, since_a),
                  g.add(g.mul(g.mul(g.sub(1.0, lt_out), lt_back), since_b),
                        g.mul(g.sub(1.0, lt_back), since_c)))
    a = g.math("MINIMUM", g.div(since, refill), 1.0)
    cleaned = g.mul(g.mul(in_sector, active), g.sub(1.0, a))
    return g.sub(1.0, cleaned)


def _windscreen(cam, p, on, rr, fog_wet):
    ob = bpy.data.objects.get(GLASS_OBJ)
    if not on:
        if ob:
            ob.hide_render = True
        return dict(windscreen_drops=False)
    if ob is None:
        me = bpy.data.meshes.new(GLASS_OBJ)
        hw, hh = 1.15, 0.85
        me.from_pydata([(-hw, -hh, 0), (hw, -hh, 0), (hw, hh, 0), (-hw, hh, 0)], [], [(0, 1, 2, 3)])
        ob = bpy.data.objects.new(GLASS_OBJ, me)
        _collection().objects.link(ob)
    _parent_to_cam(ob, cam, Matrix.Translation((0, 0.05, -GLASS_DIST)))
    _camera_only(ob)
    ob.hide_render = False
    lr = math.log1p(rr)
    dens = dict(drops=min(0.85, 0.12 + 0.2 * lr) if rr > 0 else 0.10 * fog_wet,
                streaks=min(0.6, 0.05 + 0.12 * lr) if rr > 0 else 0.0,
                film=min(0.12, 0.02 + 0.03 * lr))
    mat = _glass_material(p, dens)
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    return dict(windscreen_drops=True, windscreen_drop_density=round(dens["drops"], 3),
                windscreen_streak_density=round(dens["streaks"], 3))


def _wiper_obj(cam):
    ob = bpy.data.objects.get(WIPER_OBJ)
    if ob is None:
        me = bpy.data.meshes.new(WIPER_OBJ)
        # arm along local -X from the pivot, 2.2 cm wide, blade thicker on the outer 80 %
        v, f = [], []
        def box(x0, x1, w, t, z):
            i = len(v)
            for xx in (x0, x1):
                for yy in (-w / 2, w / 2):
                    for zz in (z - t / 2, z + t / 2):
                        v.append((xx, yy, zz))
            f.extend([(i, i + 1, i + 3, i + 2), (i + 4, i + 6, i + 7, i + 5), (i, i + 4, i + 5, i + 1),
                      (i + 2, i + 3, i + 7, i + 6), (i, i + 2, i + 6, i + 4), (i + 1, i + 5, i + 7, i + 3)])
        box(-WIPER_LEN, 0.0, 0.018, 0.015, 0.0)
        box(-WIPER_LEN, -0.18, 0.022, 0.018, 0.012)
        me.from_pydata(v, [], f)
        ob = bpy.data.objects.new(WIPER_OBJ, me)
        _collection().objects.link(ob)
        m = bpy.data.materials.get("SB_signal_black")
        if m:
            me.materials.append(m)
    if cam is not None:
        ob.parent = cam
        ob.matrix_parent_inverse = Matrix.Identity(4)
    return ob


def wiper_angle(time_s: float, period_s: float) -> tuple:
    """(angle_deg camera-plane, cycle phase 0..1). Triangle wave starting at the park position."""
    tau = (time_s / period_s) % 1.0
    tri = 1.0 - abs(2.0 * tau - 1.0)
    return WIPER_PARK_DEG + WIPER_SWEEP_DEG * tri, tau


def update_frame(scene=None, time_s: float = None):
    """Advance frame-dependent state (wiper position + cleaned sector, rain seed via Scene Time).
    Called by the frame-change handler and by apply_conditions; the renderer may call it with an
    explicit time_s for each snapshot."""
    scene = scene or bpy.context.scene
    st = scene.get("sbahn_wiper")
    if not st:
        return None
    st = dict(st)
    if time_s is None:
        time_s = st.get("time_s_override")
        if time_s is None:
            time_s = scene.frame_current / max(scene.render.fps / scene.render.fps_base, 1e-6)
    ob = bpy.data.objects.get(WIPER_OBJ)
    mat = bpy.data.materials.get("SB_Windscreen")
    period = float(st["period_s"])
    mode = st["mode"]
    if mode == "on":
        ang, tau = wiper_angle(time_s, period)
    else:
        ang, tau = WIPER_PARK_DEG, 0.0
    if ob:
        ob.hide_render = not (mode == "on" and st.get("visible", True))
        ob.matrix_basis = (Matrix.Translation((WIPER_PIVOT[0], WIPER_PIVOT[1], -GLASS_DIST - 0.03))
                           @ Matrix.Rotation(math.radians(ang - 180.0), 4, "Z"))
    if mat and mat.node_tree:
        nodes = mat.node_tree.nodes
        if "WiperTau" in nodes:
            nodes["WiperTau"].outputs[0].default_value = tau
            nodes["WiperActive"].outputs[0].default_value = 1.0 if mode == "on" else 0.0
            nodes["WiperRefill"].outputs[0].default_value = float(st["refill_cycles"])
    scene["sbahn_wiper_angle_deg"] = ang
    return dict(wiper_angle_deg=round(ang, 2), wiper_phase=round(tau, 4), time_s=round(time_s, 3))


def _frame_handler(scene, *args):
    try:
        update_frame(scene)
    except Exception as e:          # never break a render because of the wiper
        print("[conditions] wiper update failed:", e)


def _install_handler():
    hs = bpy.app.handlers.frame_change_pre
    for h in list(hs):
        if getattr(h, "__name__", "") == "_frame_handler":
            hs.remove(h)
    hs.append(_frame_handler)


# ---------------------------------------------------------------- wet surfaces
_SKIP_PREFIX = ("SB_Fog", "SB_Windscreen", "SB_RainStreak", "SB_lamp", "SB_glass")


def _inject_wetness():
    """Insert (once per material) roughness / albedo modulation by the 'sbahn_wetness' attribute."""
    n = 0
    for mat in bpy.data.materials:
        if not mat.node_tree or mat.get("sbahn_wet") or mat.name.startswith(_SKIP_PREFIX):
            continue
        nt = mat.node_tree
        bsdfs = [nd for nd in nt.nodes if nd.bl_idname == "ShaderNodeBsdfPrincipled"]
        if not bsdfs:
            continue
        g = NodeBuilder(nt)
        wet = g.attr("sbahn_wetness", "VIEW_LAYER", "Fac")
        dark = g.sub(1.0, g.mul(wet, 0.45))                  # wet albedo ~ 55 %
        rough_k = g.sub(1.0, g.mul(wet, 0.75))
        for b in bsdfs:
            # base colour
            sock = b.inputs["Base Color"]
            src = sock.links[0].from_socket if sock.is_linked else g.rgb(sock.default_value)
            mixn = g.node("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
            mixn.inputs["Factor"].default_value = 1.0
            g.set(mixn.inputs["A"], src)
            g.set(mixn.inputs["B"], g.combine(dark, dark, dark))
            nt.links.new(mixn.outputs["Result"], sock)
            # roughness
            rs = b.inputs["Roughness"]
            rsrc = rs.links[0].from_socket if rs.is_linked else rs.default_value
            r = g.math("MAXIMUM", g.mul(rsrc, rough_k), 0.06)
            nt.links.new(r, rs)
        mat["sbahn_wet"] = 1
        n += 1
    return n


# ---------------------------------------------------------------- entry
def build_rain(scene, p: dict) -> dict:
    rr = float(p["rain_rate_mm_h"])
    cam = scene.camera
    fog_wet = 1.0 if float(p.get("fog_density", 0.0)) > 0 else 0.0
    eff = dict(rain_rate_mm_h=rr)
    eff.update(_rain_streaks(cam, p, rr))

    # wetness
    w = p.get("wetness")
    if w is None:
        w = min(1.0, 0.45 + 0.18 * math.log2(1.0 + rr)) if rr > 0 else (0.3 if fog_wet else 0.0)
    w = float(min(max(w, 0.0), 1.0))
    injected = _inject_wetness()
    scene["sbahn_wetness"] = w
    eff["wetness"] = round(w, 3)
    if injected:
        eff["wetness_materials_patched"] = injected

    # windscreen + wiper
    ws = p.get("windscreen", "auto")
    on = (rr > 0) if ws == "auto" else bool(ws)
    eff.update(_windscreen(cam, p, on, rr, fog_wet))
    wmode = p.get("wiper", "auto")
    if wmode == "auto":
        wmode = "on" if rr >= 1.0 else "off"
    period = p.get("wiper_period_s") or (1.6 if rr >= 8 else 3.0)
    refill = max(0.25, min(8.0, 12.0 / max(rr, 0.5) / period))        # cycles until the glass is fully wet again
    _wiper_obj(cam)
    scene["sbahn_wiper"] = dict(mode=wmode, period_s=float(period), refill_cycles=float(refill),
                                time_s_override=p.get("time_s"), visible=True)
    _install_handler()
    wf = update_frame(scene, p.get("time_s"))
    eff.update(wiper=wmode, wiper_period_s=float(period))
    if wf:
        eff.update(wiper_angle_deg=wf["wiper_angle_deg"], wiper_phase=wf["wiper_phase"], clock_s=wf["time_s"])
    return eff

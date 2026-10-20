"""Signalling and lineside furniture: Ks main signals at block distances,
Hektometertafeln on catenary masts, and level crossings (Bahnübergänge) with
Andreaskreuze, half barriers and warning lights.

Every lamp is its own small mesh object with a custom property "on" (0/1)
read by the shared emission material, so aspects can be switched per signal:
    signals.set_aspect(bpy.data.objects["KsSignal_T0_03"], "Hp0")
"""
from __future__ import annotations

import math
from typing import Dict, List

import bpy
import numpy as np
from mathutils import Matrix

from . import util
from .config import Config
from .materials import MaterialLibrary
from .track_path import TrackPath

ASPECTS = {           # Ks aspects -> which lamps are lit
    "Ks1": ("green",),            # Fahrt (proceed)
    "Ks2": ("yellow",),           # Halt erwarten (expect stop)
    "Hp0": ("red",),              # Halt
    "off": (),
}


def _lamp_material(mats: MaterialLibrary, color: str):
    """Emission material driven by the object property 'on'."""
    name = f"SB_lamp_{color}_switchable"
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    base = mats.get(f"lamp_{color}")
    mat = base.copy()
    mat.name = name
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "OBJECT"
    attr.attribute_name = "on"
    attr.location = (-500, -400)
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 12.0
    mul.location = (-250, -400)
    nt.links.new(attr.outputs["Fac"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], bsdf.inputs["Emission Strength"])
    return mat


def _disc(radius, z, y, segments=16):
    ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    v = np.stack([radius * np.cos(ang), np.full(segments, y), z + radius * np.sin(ang)], axis=1)
    v = np.concatenate([v, [[0, y, z]]])
    f = [[segments, i, (i + 1) % segments] for i in range(segments)]
    return v, f


def set_aspect(root: bpy.types.Object, aspect: str) -> None:
    lit = ASPECTS[aspect]
    for ch in root.children_recursive:
        col = ch.get("sbahn_lamp")
        if col:
            ch["on"] = 1.0 if col in lit else 0.0
    root["aspect"] = aspect


def build_ks_signal(name: str, mats: MaterialLibrary, col, aspect="Ks1") -> bpy.types.Object:
    """Ks signal facing -Y (towards an approaching train travelling +Y)."""
    parts = [
        (*util.cylinder((0, 0, -0.6), (0, 0, 4.4), 0.075, 10), 0),                       # mast
        (*util.box((0.55, 0.16, 1.05), (0, 0, 4.85)), 1),                                # screen
        (*util.box((0.61, 0.02, 1.11), (0, 0.075, 4.85)), 2),                            # white rim (behind)
        (*util.box((0.20, 0.03, 0.60), (0, -0.09, 2.9)), 3),                             # Mastschild (red/white)
        (*util.box((0.30, 0.12, 0.22), (0, 0, 4.15)), 1),                                # Zusatzanzeiger box
        (*util.box((0.60, 0.20, 0.03), (0, -0.06, 5.39)), 1),                            # hood
    ]
    V, F, M = util.merge(parts)
    root = util.new_mesh_object(name, V, F, col, [mats.get("mast_paint"), mats.get("signal_black"),
                                                  mats.get("signal_white"), mats.get("signal_red")], M)
    root["sbahn_kind"] = "ks_signal"
    for color, z in (("yellow", 5.12), ("green", 4.85), ("red", 4.58)):
        v, f = _disc(0.085, z, -0.083)
        lamp = util.new_mesh_object(f"{name}_lamp_{color}", v, f, col, [_lamp_material(mats, color)])
        lamp["sbahn_lamp"] = color
        lamp["on"] = 0.0
        lamp.parent = root
    set_aspect(root, aspect)
    return root


def plan_signals(cfg: Config, tp: TrackPath, mast_s: List[float], track_offsets) -> List[dict]:
    """Ks main signals per track at block spacing, on the right of each track in
    its running direction, avoiding catenary mast positions."""
    sg = cfg.signals
    mast_s = np.asarray(mast_s)
    out = []
    n_blocks = max(2, int(round(tp.length_m / sg.block_length_m)))
    block = tp.length_m / n_blocks
    for ti, off in enumerate(track_offsets):
        direction = 1.0 if ti == 0 else -1.0
        for b in range(n_blocks):
            s = (b + (0.37 if ti == 0 else 0.62)) * block
            # keep >= 9 m from any mast
            for _ in range(10):
                if np.min(np.abs(tp.wrap(mast_s - s + tp.length_m / 2) - tp.length_m / 2)) > 9.0:
                    break
                s += 6.0
            lateral = off + direction * sg.signal_offset_m
            out.append(dict(name=f"KsSignal_T{ti}_{b:02d}", track=ti, s=float(tp.wrap(s)), lateral_m=float(lateral),
                            direction=int(direction), aspect="Ks1"))
    return out


def build_signals(cfg: Config, tp: TrackPath, mats: MaterialLibrary, plan: List[dict]):
    col = util.ensure_collection("Signals")
    objs = []
    for sp in plan:
        ob = build_ks_signal(sp["name"], mats, col, sp["aspect"])
        yaw = 0.0 if sp["direction"] > 0 else math.pi
        ob.matrix_world = tp.matrix(sp["s"], lateral=sp["lateral_m"], height=cfg.pway.formation_h + 0.05, yaw=yaw, upright=True)
        ob["sbahn_s"] = sp["s"]; ob["sbahn_track"] = sp["track"]
        objs.append(ob)
    return objs


def build_hektometer(cfg: Config, tp: TrackPath, mats: MaterialLibrary, mast_s: List[float], mast_lateral: float):
    """White km boards on the right-hand catenary masts closest to every 200 m."""
    col = util.ensure_collection("Hektometer")
    mast_s = np.asarray(mast_s)
    objs = []
    step = cfg.signals.hektometer_spacing_m
    for k in range(int(tp.length_m // step)):
        target = k * step
        i = int(np.argmin(np.abs(mast_s - target)))
        s = mast_s[i]
        km = s / 1000.0
        text = f"{int(km)}\n{int(round((km - int(km)) * 10))}"
        v, f = util.box((0.30, 0.02, 0.42), (0, 0, 0))
        board = util.new_mesh_object(f"Hektometer_{k:03d}", v, f, col, [mats.get("signal_white")])
        board.matrix_world = tp.matrix(s, lateral=mast_lateral - 0.20, height=3.2, upright=True) @ Matrix.Rotation(math.pi / 2, 4, "Z")
        txt = util.text_object(f"HektometerText_{k:03d}", text, 0.15, mats.get("signal_black"), col, extrude=0.002)
        txt.parent = board
        txt.matrix_parent_inverse = Matrix.Identity(4)
        txt.matrix_basis = Matrix.Translation((0, -0.012, 0)) @ Matrix.Rotation(math.pi / 2, 4, "X")
        board["sbahn_s"] = float(s)
        objs.append(board)
    return objs


# --------------------------------------------------------------------------
# Level crossings
# --------------------------------------------------------------------------
def plan_crossings(cfg: Config, tp: TrackPath, station: dict = None) -> List[dict]:
    """Put crossings in the middle of the longest straights (perpendicular roads)."""
    straights = [sg for sg in tp.meta["segments"] if sg["kind"] == "straight" and sg["s1"] - sg["s0"] > 150]
    straights.sort(key=lambda sg: -(sg["s1"] - sg["s0"]))
    out = []
    for sg in straights[: cfg.signals.crossings]:
        s = 0.5 * (sg["s0"] + sg["s1"])
        if station and station["s0"] - 80 < s < station["s1"] + 80:
            s = sg["s0"] + 80 if sg["s0"] + 80 < station["s0"] - 80 else sg["s1"] - 80
        fr = tp.sample(s)
        out.append(dict(name=f"Crossing_{len(out):02d}", s=float(s), heading=float(fr["heading"][0]),
                        road_half_len_m=70.0, road_w_m=cfg.signals.crossing_road_w_m))
    return out


def crossing_flatten_lines(cfg: Config, tp: TrackPath, terrain, crossings: List[dict]):
    """Road profiles for the terrain: level with rail top over the tracks, then
    down to natural ground within road_half_len (<= ~6 % where possible)."""
    from .terrain import FlattenLine
    lines = []
    for cr in crossings:
        f = tp.sample(cr["s"])
        right = f["right"][0]; pos = f["pos"][0]
        u = np.linspace(-cr["road_half_len_m"], cr["road_half_len_m"], 29)
        xy = pos[None, :2] + u[:, None] * right[None, :2]
        z_nat = terrain.natural_height(xy[:, 0], xy[:, 1])
        z_c = pos[2] - 0.02
        w = np.clip((np.abs(u) - 6.0) / (cr["road_half_len_m"] - 6.0), 0, 1)
        w = w * w * (3 - 2 * w)
        z = z_c * (1 - w) + z_nat * w
        pts = np.stack([xy[:, 0], xy[:, 1], z], axis=1)
        lines.append(FlattenLine(pts, half_w=cr["road_w_m"] / 2 + 1.0, blend_w=5.0))
        cr["road_profile"] = pts.tolist()
    return lines


def _andreaskreuz(mats, col, name):
    parts = [(*util.cylinder((0, 0, 0), (0, 0, 2.6), 0.05, 8), 0)]
    for ang in (math.radians(35), math.radians(-35)):
        v, f = util.box((1.3, 0.04, 0.14), (0, 0, 0))
        R = np.array([[math.cos(ang), 0, -math.sin(ang)], [0, 1, 0], [math.sin(ang), 0, math.cos(ang)]])
        v = v @ R.T + np.array([0, -0.05, 2.2])
        parts.append((v, f, 1))
        # red tips
        for sgn in (-1, 1):
            tv, tf = util.box((0.22, 0.045, 0.145), (sgn * 0.54, 0, 0))
            tv = tv @ R.T + np.array([0, -0.05, 2.2])
            parts.append((tv, tf, 2))
    V, F, M = util.merge(parts)
    return util.new_mesh_object(name, V, F, col, [mats.get("galvanised"), mats.get("andreaskreuz_white"), mats.get("signal_red")], M)


def _barrier(mats, col, name, boom_len):
    hv, hf = util.box((0.5, 0.5, 1.1), (0, 0, 0.55))
    housing = util.new_mesh_object(name, hv, hf, col, [mats.get("galvanised")])
    parts = [(*util.cylinder((0, 0, 0), (boom_len, 0, 0), 0.045, 8), 0)]
    for i in range(int(boom_len // 0.5)):
        if i % 2 == 1:
            v, f = util.cylinder((i * 0.5, 0, 0), ((i + 1) * 0.5, 0, 0), 0.047, 8)
            parts.append((v, f, 1))
    V, F, M = util.merge(parts)
    boom = util.new_mesh_object(name + "_boom", V, F, col, [mats.get("andreaskreuz_white"), mats.get("signal_red")], M)
    boom.parent = housing
    boom.matrix_parent_inverse = Matrix.Identity(4)
    boom.location = (0.3, 0, 1.0)
    boom["barrier_angle_deg"] = 0.0          # 0 = closed (horizontal), 90 = open
    return housing, boom


def build_crossings(cfg: Config, tp: TrackPath, mats: MaterialLibrary, terrain, crossings: List[dict], track_offsets):
    col = util.ensure_collection("Crossings")
    objs = []
    half_deck = max(abs(o) for o in track_offsets) + 2.2
    for cr in crossings:
        s = cr["s"]
        w = cr["road_w_m"]
        # road strip on the terrain
        prof = np.asarray(cr["road_profile"])
        f = tp.sample(s)
        right = f["right"][0]; tang = f["tangent"][0]
        n = len(prof)
        V = np.concatenate([prof + (-w / 2) * tang, prof + (w / 2) * tang])
        V[:, 2] += 0.06
        F = np.array([[i, i + 1, n + i + 1, n + i] for i in range(n - 1)])
        road = util.new_mesh_object(f"{cr['name']}_road", V, F, col, [mats.get("asphalt")], smooth=True)
        objs.append(road)
        # crossing deck over both tracks, level with the rail head (rails poke through by 1 cm)
        ss = np.linspace(s - w / 2 - 0.6, s + w / 2 + 0.6, 6)
        fd = tp.sample(ss)
        dprof = [(-half_deck, -0.012), (half_deck, -0.012)]
        dv, df = util.extrude_profile(dprof, fd["pos"], fd["right"], fd["up"], closed_path=False, close_profile=False)
        deck = util.new_mesh_object(f"{cr['name']}_deck", dv, df, col, [mats.get("asphalt")])
        objs.append(deck)
        # furniture on each side, right-hand side of the road for traffic approaching the track
        for side in (1.0, -1.0):
            lat = side * (half_deck + 2.4)
            road_off = -side * (w / 2 + 1.0)          # right of the road when driving towards the track
            base = tp.matrix(s + road_off, lateral=lat, height=0.0, upright=True)
            zg = float(terrain.height(np.array([base[0][3]]), np.array([base[1][3]]))[0])
            base[2][3] = zg
            ak = _andreaskreuz(mats, col, f"{cr['name']}_Andreaskreuz_{'R' if side > 0 else 'L'}")
            ak.matrix_world = base @ Matrix.Rotation(math.pi / 2 * (1 if side > 0 else -1), 4, "Z")
            objs.append(ak)
            housing, boom = _barrier(mats, col, f"{cr['name']}_Barrier_{'R' if side > 0 else 'L'}", w / 2 + 0.6)
            hb = tp.matrix(s + road_off, lateral=lat + side * 1.2, height=0.0, upright=True)
            hb[2][3] = float(terrain.height(np.array([hb[0][3]]), np.array([hb[1][3]]))[0])
            # boom points across the road: local +X of the housing must be along the track towards the road axis
            housing.matrix_world = hb @ Matrix.Rotation(math.pi / 2 * (1 if side > 0 else -1) + math.pi / 2 * (-side), 4, "Z")
            objs += [housing, boom]
            lamp_v, lamp_f = _disc(0.1, 2.9, -0.06)
            lamp = util.new_mesh_object(f"{cr['name']}_Light_{'R' if side > 0 else 'L'}", lamp_v, lamp_f, col, [_lamp_material(mats, "red")])
            lamp["sbahn_lamp"] = "red"; lamp["on"] = 0.0
            lamp.parent = ak
            lamp.matrix_parent_inverse = Matrix.Identity(4)
            lamp.matrix_basis = Matrix.Identity(4)
            objs.append(lamp)
        cr["deck_half_w_m"] = float(half_deck)
    return objs

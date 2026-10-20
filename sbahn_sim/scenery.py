"""Lineside scenery: procedural tree / grass / house proxies (instance sources),
noise barriers (Lärmschutzwände) in suburban sectors, and an S-Bahn station.

Drop better tree models (.glb/.gltf/.blend/.obj) into assets/models/trees and
they replace the procedural proxies in the SRC_Trees collection.
"""
from __future__ import annotations

import glob
import math
import os
from typing import Dict, List

import bpy
import numpy as np

from . import util
from .config import Config
from .materials import MaterialLibrary
from .track_path import TrackPath


# --------------------------------------------------------------------------
# Procedural tree / grass / house proxies
# --------------------------------------------------------------------------
def _icosphere(radius, subdiv=2):
    """Icosphere via bmesh (verts, tri faces)."""
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=radius)
    v = np.array([vv.co[:] for vv in bm.verts])
    f = np.array([[vv.index for vv in ff.verts] for ff in bm.faces])
    bm.free()
    return v, f


def spruce(rng, height=18.0, name="Spruce"):
    trunk_r = 0.012 * height
    parts = [(*util.cylinder((0, 0, 0), (0, 0, height * 0.55), trunk_r, 8, trunk_r * 0.35), 0)]
    tiers = 6
    for i in range(tiers):
        z0 = height * (0.18 + 0.72 * i / tiers)
        z1 = z0 + height * 0.22
        r = 0.14 * height * (1.0 - 0.75 * i / tiers) * rng.uniform(0.85, 1.15)
        v, f = util.cylinder((0, 0, z0), (0, 0, z1), r, 10, 0.02)
        v[:, :2] += rng.normal(0, r * 0.06, (len(v), 2))
        parts.append((v, f, 1))
    return util.merge(parts)


def deciduous(rng, height=13.0, name="Oak"):
    trunk_r = 0.022 * height
    parts = [(*util.cylinder((0, 0, 0), (0, 0, height * 0.45), trunk_r, 8, trunk_r * 0.6), 0)]
    for i in range(4):
        r = height * rng.uniform(0.20, 0.30)
        c = np.array([rng.uniform(-0.15, 0.15) * height, rng.uniform(-0.15, 0.15) * height,
                      height * rng.uniform(0.5, 0.75)])
        v, f = _icosphere(r, 1)
        v += rng.normal(0, r * 0.08, v.shape)
        v[:, 2] *= 0.85
        parts.append((v + c, f, 1))
    return util.merge(parts)


def birch(rng, height=14.0):
    trunk_r = 0.012 * height
    parts = [(*util.cylinder((0, 0, 0), (0, 0, height * 0.6), trunk_r, 8, trunk_r * 0.4), 2)]
    for i in range(3):
        r = height * rng.uniform(0.12, 0.2)
        c = np.array([rng.uniform(-0.08, 0.08) * height, rng.uniform(-0.08, 0.08) * height,
                      height * rng.uniform(0.55, 0.85)])
        v, f = _icosphere(r, 1)
        v += rng.normal(0, r * 0.1, v.shape)
        parts.append((v + c, f, 1))
    return util.merge(parts)


def bush(rng, size=2.2):
    parts = []
    for i in range(3):
        r = size * rng.uniform(0.35, 0.55)
        c = np.array([rng.uniform(-0.3, 0.3) * size, rng.uniform(-0.3, 0.3) * size, r * 0.8])
        v, f = _icosphere(r, 1)
        v += rng.normal(0, r * 0.1, v.shape)
        parts.append((v + c, f, 1))
    return util.merge(parts)


def grass_tuft(rng, blades=7, h=0.35):
    parts = []
    for i in range(blades):
        a = rng.uniform(0, 2 * math.pi)
        lean = rng.uniform(0.1, 0.5)
        hh = h * rng.uniform(0.6, 1.3)
        base = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), 0])
        tip = base + np.array([math.cos(a) * lean * hh, math.sin(a) * lean * hh, hh])
        w = 0.02
        side = np.array([-math.sin(a), math.cos(a), 0]) * w
        v = np.array([base - side, base + side, tip])
        parts.append((v, np.array([[0, 1, 2]]), 0))
    return util.merge(parts)


def house(rng, w=10.0, d=8.0, h=6.2, roof_h=3.0):
    v, f = util.box((w, d, h), (0, 0, h / 2))
    parts = [(v, f, 0)]
    # gable roof (prism along X)
    rv = np.array([[-w / 2 - 0.4, -d / 2 - 0.4, h], [w / 2 + 0.4, -d / 2 - 0.4, h], [w / 2 + 0.4, d / 2 + 0.4, h],
                   [-w / 2 - 0.4, d / 2 + 0.4, h], [-w / 2 - 0.4, 0, h + roof_h], [w / 2 + 0.4, 0, h + roof_h]])
    rf = [[0, 1, 5, 4], [2, 3, 4, 5], [0, 4, 3], [1, 2, 5]]
    parts.append((rv, rf, 1))
    cv, cf = util.box((0.6, 0.6, 1.5), (w * 0.25, 0.6, h + roof_h * 0.6))
    parts.append((cv, cf, 1))
    return util.merge(parts)


def _import_external_trees(cfg: Config, col) -> int:
    folder = os.path.join(cfg.asset_dir, "models", "trees")
    count = 0
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        ext = path.lower().rsplit(".", 1)[-1]
        before = set(bpy.data.objects)
        try:
            if ext in ("glb", "gltf"):
                bpy.ops.import_scene.gltf(filepath=path)
            elif ext == "obj":
                bpy.ops.wm.obj_import(filepath=path)
            elif ext == "fbx":
                bpy.ops.import_scene.fbx(filepath=path)
            elif ext == "blend":
                with bpy.data.libraries.load(path, link=False) as (src, dst):
                    dst.objects = [n for n in src.objects]
                for ob in dst.objects:
                    if ob is not None:
                        bpy.context.scene.collection.objects.link(ob)
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"[scenery] could not import {path}: {exc}")
            continue
        new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        if not new:
            continue
        ob = util.join_meshes(f"Tree_{os.path.basename(path).rsplit('.', 1)[0]}", new, col) if len(new) > 1 else new[0]
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        col.objects.link(ob)
        ob.location = (0, 0, 0)
        count += 1
    return count


def build_instance_sources(cfg: Config, mats: MaterialLibrary) -> Dict[str, bpy.types.Collection]:
    if cfg.scatter.hd_models:
        # detailed procedural trees / grass / house plots with parked cars (sbahn_sim.models)
        from .models import library
        return library.build_instance_sources(cfg, mats)
    rng = np.random.default_rng(cfg.seed + 20)
    trees = util.ensure_collection("SRC_Trees")
    n_ext = _import_external_trees(cfg, trees) if cfg.use_downloaded_assets else 0
    if n_ext == 0:
        bark, spr, dec, bir = mats.get("bark"), mats.get("leaves_spruce"), mats.get("leaves_deciduous"), mats.get("leaves_birch")
        white = mats.get("signal_white")
        for i in range(3):
            V, F, M = spruce(rng, rng.uniform(15, 24))
            util.new_mesh_object(f"Tree_Spruce_{i}", V, F, trees, [bark, spr], M, smooth=True)
        for i in range(2):
            V, F, M = deciduous(rng, rng.uniform(11, 16))
            util.new_mesh_object(f"Tree_Deciduous_{i}", V, F, trees, [bark, dec], M, smooth=True)
        V, F, M = birch(rng, 14)
        util.new_mesh_object("Tree_Birch", V, F, trees, [bark, bir, white], M, smooth=True)
    bushes = util.ensure_collection("SRC_Bushes")
    for i in range(2):
        V, F, M = bush(rng)
        util.new_mesh_object(f"Bush_{i}", V, F, bushes, [mats.get("bark"), mats.get("leaves_deciduous")], M, smooth=True)
    grass = util.ensure_collection("SRC_Grass")
    for i in range(3):
        V, F, M = grass_tuft(rng)
        util.new_mesh_object(f"GrassTuft_{i}", V, F, grass, [mats.get("grass")], M)
    houses = util.ensure_collection("SRC_Houses")
    for i in range(3):
        V, F, M = house(rng, rng.uniform(8, 13), rng.uniform(7, 10), rng.uniform(5.5, 7.5))
        util.new_mesh_object(f"House_{i}", V, F, houses, [mats.get("house_wall"), mats.get("roof_tile")], M)
    for c in (trees, bushes, grass, houses):
        util.hide_collection_instances_source(c)
    return dict(trees=trees, bushes=bushes, grass=grass, houses=houses)


# --------------------------------------------------------------------------
# Noise barriers
# --------------------------------------------------------------------------
def build_noise_barriers(cfg: Config, tp: TrackPath, mats: MaterialLibrary, sectors: List[dict], mast_s: List[float],
                         keep_clear: List[tuple] = ()):
    """keep_clear: [(s0, s1), ...] ranges (station, crossings) that get no wall."""
    sc = cfg.scatter
    col = util.ensure_collection("NoiseBarriers")
    src = util.ensure_collection("SRC_NoiseBarrier")
    rng = np.random.default_rng(cfg.seed + 30)
    v, f = util.box((sc.panel_len_m - 0.05, 0.16, sc.noise_barrier_h_m), (0, 0, sc.noise_barrier_h_m / 2))
    panel = util.new_mesh_object("NoisePanel", v, f, src, [mats.get("noise_wall")])
    pv, pf = util.box((0.20, 0.22, sc.noise_barrier_h_m + 0.1), (sc.panel_len_m / 2, 0, sc.noise_barrier_h_m / 2))
    util.new_mesh_object("NoisePost", pv, pf, src, [mats.get("galvanised")])
    util.hide_collection_instances_source(src)
    pts, rots, idx, runs = [], [], [], []
    base_h = cfg.pway.formation_h - 0.06
    for sec in sectors:
        if sec["type"] != "suburban":
            continue
        L = sec["s1"] - sec["s0"]
        run_len = sc.noise_barrier_share * L
        s0 = sec["s0"] + rng.uniform(0.05, 0.35) * (L - run_len)
        sides = [1.0, -1.0] if rng.random() < 0.6 else [rng.choice([1.0, -1.0])]
        n = int(run_len // sc.panel_len_m)
        s = s0 + (np.arange(n) + 0.5) * sc.panel_len_m
        keep = np.ones(n, dtype=bool)
        for a, b in keep_clear:
            keep &= ~((s > a) & (s < b))
        s = s[keep]
        n = len(s)
        if n == 0:
            continue
        fr = tp.sample(s, upright=True)
        for side in sides:
            P = fr["pos"] + side * sc.noise_barrier_offset_m * fr["right"] + base_h * fr["up"]
            r = np.zeros((n, 3)); r[:, 2] = fr["heading"]   # panel X along the track
            pts += [P, P]; rots += [r, r]; idx += [np.zeros(n, int), np.ones(n, int)]
            runs.append(dict(s0=float(s0), s1=float(s0 + n * sc.panel_len_m), side="right" if side > 0 else "left",
                             lateral_m=float(side * sc.noise_barrier_offset_m)))
    if not pts:
        return dict(runs=[])
    util.instancer("NoiseBarriers", np.concatenate(pts), np.concatenate(rots), src, indices=np.concatenate(idx), collection=col)
    return dict(runs=runs)


# --------------------------------------------------------------------------
# Station
# --------------------------------------------------------------------------
def plan_station(cfg: Config, tp: TrackPath, sectors: List[dict], length_m: float = 210.0):
    """Choose a station location in the first suburban sector, on a section that
    is as straight as possible."""
    sub = [s for s in sectors if s["type"] == "suburban"]
    if not sub:
        return None
    sec = sub[0]
    best, best_k = None, 1e9
    for s0 in np.arange(sec["s0"] + 50, sec["s1"] - length_m - 50, 25.0):
        k = float(np.max(np.abs(tp._interp("curvature", np.arange(s0, s0 + length_m, 10.0)))))
        if k < best_k:
            best, best_k = s0, k
    if best is None:
        return None
    return dict(name="Musterstadt", s0=float(best), s1=float(best + length_m), length_m=length_m,
                platform_h=0.96, edge_lateral_m=cfg.track.track_spacing_m / 2.0 + 1.65, width_m=4.6)


def build_station(cfg: Config, tp: TrackPath, mats: MaterialLibrary, st: dict):
    col = util.ensure_collection("Station")
    n = int(st["length_m"] / 2.5) + 1
    s = np.linspace(st["s0"], st["s1"], n)
    fr = tp.sample(s)
    objs = []
    h = st["platform_h"]
    zb = cfg.pway.formation_h - 0.1
    for side in (1.0, -1.0):
        e = st["edge_lateral_m"]; w = st["width_m"]
        prof = [(side * e, zb), (side * e, h), (side * (e + w), h), (side * (e + w), zb)]
        v, f = util.extrude_profile(prof, fr["pos"], fr["right"], fr["up"], closed_path=False, close_profile=True)
        # end caps
        k = 4
        caps = [[0, 1, 2, 3], [(n - 1) * k + 3, (n - 1) * k + 2, (n - 1) * k + 1, (n - 1) * k]]
        F = [list(x) for x in f] + caps
        ob = util.new_mesh_object(f"Platform_{'R' if side > 0 else 'L'}", v, F, col, [mats.get("platform")])
        objs.append(ob)
        # tactile / safety stripe 0.6 m from the edge
        prof2 = [(side * (e + 0.55), h + 0.004), (side * (e + 0.85), h + 0.004)]
        v2, f2 = util.extrude_profile(prof2, fr["pos"], fr["right"], fr["up"], closed_path=False, close_profile=False)
        objs.append(util.new_mesh_object(f"PlatformStripe_{'R' if side > 0 else 'L'}", v2, f2, col, [mats.get("signal_white")]))
        # shelter + sign in the middle of the platform
        mid = 0.5 * (st["s0"] + st["s1"])
        m = tp.matrix(mid, lateral=side * (e + w * 0.62), height=h, upright=True)
        sv, sf = util.box((6.0, 2.4, 0.15), (0, 0, 2.6))
        pv, pf = util.box((0.12, 0.12, 2.6), (0, 0, 1.3))
        parts = [(sv, sf, 0)]
        for dx, dy in ((-2.8, -1.0), (2.8, -1.0), (-2.8, 1.0), (2.8, 1.0)):
            parts.append((pv + np.array([dx, dy, 0]), pf, 1))
        bv, bf = util.box((5.5, 0.5, 0.45), (0, 0.8, 0.5))
        parts.append((bv, bf, 1))
        V, F, M = util.merge(parts)
        sh = util.new_mesh_object(f"Shelter_{'R' if side > 0 else 'L'}", V, F, col, [mats.get("concrete"), mats.get("galvanised")], M)
        sh.matrix_world = m
        objs.append(sh)
        sign = util.text_object(f"StationSign_{'R' if side > 0 else 'L'}", f"S  {st['name']}", 0.45, mats.get("signal_white"), col, extrude=0.01)
        sign.matrix_world = tp.matrix(mid + 8.0, lateral=side * (e + w * 0.62), height=h + 2.9, upright=True)
        from mathutils import Matrix
        sign.matrix_world = sign.matrix_world @ Matrix.Rotation(math.pi / 2, 4, "X") @ Matrix.Rotation(math.pi / 2 * (1 if side > 0 else -1), 4, "Y")
        objs.append(sign)
    return objs

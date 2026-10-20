"""Permanent way: 60E1 rails, B70 concrete sleepers and the ballast bed for
both tracks of the double-track loop.

Rails and ballast are swept meshes built from the TrackPath frames (so cant
and gradients are exact); sleepers are Geometry Nodes instances.
"""
from __future__ import annotations

import numpy as np

from . import util
from .config import Config
from .materials import MaterialLibrary
from .track_path import TrackPath

# 60E1 (UIC 60) half profile in mm: (lateral, height above rail foot)
_HALF = [(0, 0), (75, 0), (75, 11), (55, 15), (30, 28), (11, 38), (8.25, 46), (8.25, 118),
         (12, 122), (36, 126), (36, 156), (33, 165), (22, 171), (0, 172)]


def rail_profile(rail_height: float):
    """Closed 60E1 outline (metres), rail top at height 0, ordered CCW."""
    right = [(x / 1000.0, z / 1000.0 - rail_height) for x, z in _HALF]
    left = [(-x, z) for x, z in reversed(right[1:-1])]
    return right + left


def build_rails(cfg: Config, tp: TrackPath, mats: MaterialLibrary, col, track_offsets):
    """One mesh per track containing both rails; head faces use the 'rail_head' material."""
    step = cfg.track.sample_step_m
    n = int(round(tp.length_m / step))
    s = np.arange(n) * step
    fr = tp.sample(s)
    prof = rail_profile(cfg.pway.rail_height_m)
    prof_np = np.asarray(prof)
    head = [(prof_np[j, 1] > -0.02) and (prof_np[(j + 1) % len(prof), 1] > -0.02) for j in range(len(prof))]
    objs = []
    for ti, off in enumerate(track_offsets):
        parts = []
        for side in (-1, 1):
            lat = off + side * cfg.pway.rail_centre_offset_m
            P = fr["pos"] + lat * fr["right"]
            v, f = util.extrude_profile(prof, P, fr["right"], fr["up"], closed_path=True)
            mi = np.tile(np.array([1 if h else 0 for h in head], dtype=np.int32), n)
            parts.append((v, f, mi))
        V = np.concatenate([p[0] for p in parts])
        F = np.concatenate([parts[0][1], parts[1][1] + len(parts[0][0])])
        M = np.concatenate([parts[0][2], parts[1][2]])
        ob = util.new_mesh_object(f"Rails_T{ti}", V, F, col, [mats.get("rail_body"), mats.get("rail_head")], M, smooth=True)
        ob["sbahn_track_index"] = ti
        ob["sbahn_lateral_m"] = float(off)
        objs.append(ob)
    return objs


def sleeper_with_fastenings(pw):
    """B70 sleeper (trapezoid section, chamfered top edges) with W14-style fastenings on both rails:
    angled guide plates either side of the rail foot, a tension clamp loop and the sleeper screw.
    Returns merged (verts, faces, material index) with 0 = concrete, 1 = plastic, 2 = steel."""
    L, hgt = pw.sleeper_len_m / 2.0, pw.sleeper_h_m
    wb, wt, ch = pw.sleeper_w_m / 2.0, pw.sleeper_w_m * 0.37, 0.02
    sec = [(-wb, -hgt), (wb, -hgt), (wt + ch, -ch), (wt, 0.0), (-wt, 0.0), (-wt - ch, -ch)]   # (y, z)
    k = len(sec)
    v = np.array([[x, y, z] for x in (-L, L) for (y, z) in sec])
    f = [[i, (i + 1) % k, k + (i + 1) % k, k + i] for i in range(k)]
    f.append(list(range(k))[::-1])
    f.append(list(range(k, 2 * k)))
    parts = [(v, f, 0)]
    rc = pw.rail_centre_offset_m                  # rail centre line from the track axis
    foot = 0.075
    for side in (-1.0, 1.0):
        for out in (-1.0, 1.0):                   # inner / outer side of the rail
            xc = side * rc + out * (foot + 0.055)
            gv, gf = util.box((0.11, 0.17, 0.034), (xc, 0.0, 0.017))
            parts.append((gv, gf, 1))
            # tension clamp: two legs pressing on the rail foot and the loop round the screw
            x0, x1 = xc + out * 0.045, side * rc + out * (foot - 0.012)
            for yy in (-0.04, 0.04):
                cv, cf = util.cylinder((x0, yy, 0.046), (x1, yy * 0.6, 0.040), 0.0065, 6)
                parts.append((cv, cf, 2))
            cv, cf = util.cylinder((x0, -0.042, 0.046), (x0, 0.042, 0.046), 0.0065, 6)
            parts.append((cv, cf, 2))
            sv, sf = util.cylinder((xc + out * 0.01, 0.0, 0.030), (xc + out * 0.01, 0.0, 0.058), 0.017, 6)
            parts.append((sv, sf, 2))
    V, F, M = util.merge(parts)
    return V, F, np.array(M)


def build_sleepers(cfg: Config, tp: TrackPath, mats: MaterialLibrary, col, track_offsets, seed: int):
    pw = cfg.pway
    rng = np.random.default_rng(seed + 11)
    # sleeper source mesh: origin at top centre, X along the sleeper (lateral), Y along track
    src_col = util.ensure_collection("SRC_Sleeper")
    V, F, M = sleeper_with_fastenings(pw)
    src = util.new_mesh_object("Sleeper_B70", V, F, src_col,
                               [mats.get("sleeper"), mats.get("fastening_plastic"), mats.get("fastening_steel")], M)
    util.hide_collection_instances_source(src_col)
    objs = []
    n = int(round(tp.length_m / pw.sleeper_spacing_m))
    spacing = tp.length_m / n
    for ti, off in enumerate(track_offsets):
        s = np.arange(n) * spacing + rng.uniform(-0.02, 0.02, n)
        fr = tp.sample(s)
        lat = off + rng.uniform(-0.015, 0.015, n)
        P = fr["pos"] + lat[:, None] * fr["right"] - pw.rail_height_m * fr["up"]
        rot = util.frame_euler(fr["right"], fr["tangent"], fr["up"])
        rot[:, 2] += rng.uniform(-0.006, 0.006, n)          # tiny yaw jitter
        ob = util.instancer(f"Sleepers_T{ti}", P, rot, src_col, collection=col)
        ob["sbahn_track_index"] = ti
        objs.append(ob)
    return objs


def ballast_profile(cfg: Config, track_offsets):
    pw = cfg.pway
    half_sleeper = pw.sleeper_len_m / 2.0
    top = pw.ballast_top_h
    dz = top - pw.formation_h
    dd = dz * pw.ballast_slope
    outer = max(abs(o) for o in track_offsets) + half_sleeper + pw.ballast_shoulder_m
    toe = outer + dd
    # open profile from left toe to right toe with a small drainage crown
    return [(-toe, pw.formation_h), (-outer, top), (-0.5, top + 0.0), (0.0, top - 0.06),
            (0.5, top), (outer, top), (toe, pw.formation_h)], toe


def build_ballast(cfg: Config, tp: TrackPath, mats: MaterialLibrary, col, track_offsets, seed: int):
    prof, toe = ballast_profile(cfg, track_offsets)
    step = 2.0
    n = int(round(tp.length_m / step))
    s = np.arange(n) * step
    fr = tp.sample(s)
    v, f = util.extrude_profile(prof, fr["pos"], fr["right"], fr["up"], closed_path=True, close_profile=False)
    rng = np.random.default_rng(seed + 12)
    v[:, 2] += rng.uniform(-0.015, 0.015, len(v))            # break up the perfectly flat surface
    ob = util.new_mesh_object("Ballast", v, f, col, [mats.get("ballast")], smooth=True)
    # lateral position of every vertex (metres right of the centre line): the ballast shader uses it
    # for the rust-brown band along the rails and the four-foot
    util.set_point_attr(ob.data, "lat", "FLOAT", np.tile([p[0] for p in prof], n))
    ob["sbahn_toe_m"] = float(toe)
    return ob


def build(cfg: Config, tp: TrackPath, mats: MaterialLibrary):
    col = util.ensure_collection("Track")
    half = cfg.track.track_spacing_m / 2.0
    offsets = (+half, -half)      # index 0 = running track (right, +s direction), 1 = opposite track
    rails = build_rails(cfg, tp, mats, col, offsets)
    sleepers = build_sleepers(cfg, tp, mats, col, offsets, cfg.seed)
    ballast = build_ballast(cfg, tp, mats, col, offsets, cfg.seed)
    return dict(rails=rails, sleepers=sleepers, ballast=ballast, track_offsets=offsets)

"""Overhead line equipment (Oberleitung, Re 200 style): masts with cantilevers
on both sides of the double track, contact wire with stagger, messenger wire
with sag, droppers.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from . import util
from .config import Config
from .materials import MaterialLibrary
from .track_path import TrackPath


def mast_positions(cfg: Config, tp: TrackPath) -> np.ndarray:
    """Arc lengths of mast locations; shorter spans in curves; closes exactly."""
    cc = cfg.catenary
    s_list = [0.0]
    while True:
        s = s_list[-1]
        k = abs(float(tp.sample(s)["curvature"][0]))
        span = cc.span_curve_m if k > 1.0 / 1500.0 else cc.span_straight_m
        if s + span * 1.5 >= tp.length_m:
            break
        s_list.append(s + span)
    s_arr = np.array(s_list)
    # rescale so the loop closes with a whole number of spans
    n = len(s_arr)
    last_span = tp.length_m - s_arr[-1]
    s_arr = s_arr * (tp.length_m / (s_arr[-1] + last_span))
    return s_arr


def _mast_mesh(cfg: Config, mats: MaterialLibrary, src_col):
    """H-profile steel mast with concrete foundation; origin at rail-top level on the mast axis."""
    cc, pw = cfg.catenary, cfg.pway
    z0 = pw.formation_h - 0.05
    ztop = cc.mast_height_m
    parts = [
        (*util.box((0.30, 0.02, ztop - z0), (0, 0.14, (z0 + ztop) / 2)), 0),   # flange (track side)
        (*util.box((0.30, 0.02, ztop - z0), (0, -0.14, (z0 + ztop) / 2)), 0),  # flange
        (*util.box((0.014, 0.26, ztop - z0), (0, 0, (z0 + ztop) / 2)), 0),     # web
        (*util.box((0.30, 0.30, 0.02), (0, 0, ztop)), 0),                       # cap
        (*util.box((0.9, 0.9, 0.5), (0, 0, z0 + 0.15)), 1),                     # foundation
    ]
    V, F, M = util.merge(parts)
    return util.new_mesh_object("CatenaryMast", V, F, src_col, [mats.get("mast_paint"), mats.get("concrete")], M)


def _cantilever_mesh(cfg: Config, mats: MaterialLibrary, src_col, stagger: float, name: str):
    """Cantilever for a mast on the RIGHT of its track (track axis at local -X)."""
    cc = cfg.catenary
    dx = -cc.mast_offset_m - stagger         # registration point (contact wire) x
    zc = cc.contact_wire_h
    zm = zc + cc.system_height               # messenger support
    r = 0.028
    top_att = (-0.16, 0, zm + 0.25)
    low_att = (-0.16, 0, zc + 0.55)
    M = (dx + 0.15, 0, zm)                   # messenger support point (slightly mast-side of the wire)
    # steady-arm drop point on the diagonal tube
    t = 0.62
    D = (low_att[0] + t * (M[0] - low_att[0]), 0, low_att[2] + t * (M[2] - low_att[2]))
    Dlow = (D[0], 0, zc + 0.28)
    parts = [
        (*util.cylinder(top_att, M, r), 0),                       # horizontal tube
        (*util.cylinder(low_att, M, r), 0),                       # diagonal tube
        (*util.cylinder(D, Dlow, 0.02), 0),                       # steady-arm support
        (*util.cylinder(Dlow, (dx, 0, zc + 0.02), 0.018), 0),     # steady arm to the contact wire
        (*util.cylinder((0.0, 0, top_att[2]), (top_att[0] - 0.35, 0, top_att[2]), 0.05), 1),  # insulator
        (*util.cylinder((0.0, 0, low_att[2]), (low_att[0] - 0.35, 0, low_att[2]), 0.05), 1),  # insulator
        (*util.cylinder((M[0], 0, M[2] - 0.05), (M[0], 0, M[2] + 0.12), 0.03), 0),           # messenger clamp
    ]
    V, F, Mi = util.merge(parts)
    return util.new_mesh_object(name, V, F, src_col, [mats.get("galvanised"), mats.get("insulator")], Mi, smooth=True)


def build(cfg: Config, tp: TrackPath, mats: MaterialLibrary, track_offsets) -> Dict:
    cc = cfg.catenary
    col = util.ensure_collection("Catenary")
    src_masts = util.ensure_collection("SRC_CatenaryMast")
    src_cant = util.ensure_collection("SRC_Cantilever")
    src_drop = util.ensure_collection("SRC_Dropper")
    _mast_mesh(cfg, mats, src_masts)
    _cantilever_mesh(cfg, mats, src_cant, +cc.stagger_m, "Cantilever_PullIn")   # idx 0: wire pulled toward mast
    _cantilever_mesh(cfg, mats, src_cant, -cc.stagger_m, "Cantilever_PushOut")  # idx 1: wire pushed away
    dv, df = util.cylinder((0, 0, 0), (0, 0, 1.0), 0.004, 6)
    util.new_mesh_object("Dropper", dv, df, src_drop, [mats.get("galvanised")])
    for c in (src_masts, src_cant, src_drop):
        util.hide_collection_instances_source(c)

    s_m = mast_positions(cfg, tp)
    n = len(s_m)
    fr = tp.sample(s_m, upright=True)      # masts stand vertical
    kappa = fr["curvature"]
    half = cfg.track.track_spacing_m / 2.0
    mast_lat = half + cc.mast_offset_m

    # stagger sign per mast (in track-right terms): alternate on straights,
    # towards the outside of the curve in curves (outside of a left curve = right).
    stag_sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    curved = np.abs(kappa) > 1.0 / 2500.0
    stag_sign = np.where(curved, np.sign(kappa) + (kappa == 0), stag_sign)

    mast_pts, mast_rot, cant_pts, cant_rot, cant_idx = [], [], [], [], []
    for side, sgn in (("R", 1.0), ("L", -1.0)):
        lat = sgn * mast_lat
        P = fr["pos"] + lat * fr["right"]
        rot = util.frame_euler(fr["right"], fr["tangent"], fr["up"])
        if sgn < 0:
            rot[:, 2] += math.pi          # left mast faces the track (local -X toward track)
        mast_pts.append(P); mast_rot.append(rot)
        cant_pts.append(P); cant_rot.append(rot)
        # the cantilever's local -X points to the track; 'pull in' means the wire moves
        # toward the mast, i.e. track-right for the right mast, track-left for the left mast
        wire_lat = stag_sign * cc.stagger_m          # desired wire offset from the track axis (track-right +)
        pull_in = (wire_lat * sgn) > 0
        cant_idx.append(np.where(pull_in, 0, 1))
    mast_ob = util.instancer("CatenaryMasts", np.concatenate(mast_pts), np.concatenate(mast_rot), src_masts, collection=col)
    cant_ob = util.instancer("Cantilevers", np.concatenate(cant_pts), np.concatenate(cant_rot), src_cant,
                             indices=np.concatenate(cant_idx), collection=col)

    # ---- wires per track ---------------------------------------------------
    wire_mat = mats.get("copper_wire")
    contact_lines, messenger_lines, drop_pts, drop_scale = [], [], [], []
    spans = np.append(np.diff(s_m), tp.length_m - s_m[-1])
    for ti, off in enumerate(track_offsets):
        contact = []
        messenger = []
        for i in range(n):
            s0, span = s_m[i], spans[i]
            i1 = (i + 1) % n
            lat0 = off + stag_sign[i] * cc.stagger_m
            lat1 = off + stag_sign[i1] * cc.stagger_m
            m = max(2, int(math.ceil(span / 5.0)))
            t = np.linspace(0.0, 1.0, m, endpoint=False)
            ss = s0 + t * span
            lat = lat0 + (lat1 - lat0) * t
            sag = cc.midspan_sag_m * (span / cc.span_straight_m) ** 2
            hm = cc.contact_wire_h + cc.system_height - 4.0 * sag * t * (1 - t)
            f = tp.sample(ss)
            contact.append(f["pos"][:1] + lat[:1, None] * f["right"][:1] + cc.contact_wire_h * f["up"][:1])
            messenger.append(f["pos"] + lat[:, None] * f["right"] + hm[:, None] * f["up"])
            # droppers
            nd = max(1, int(round(span / cc.dropper_spacing_m)))
            td = (np.arange(nd) + 0.5) / nd
            sd = s0 + td * span
            latd = lat0 + (lat1 - lat0) * td
            hmd = cc.contact_wire_h + cc.system_height - 4.0 * sag * td * (1 - td)
            fd = tp.sample(sd)
            base = fd["pos"] + latd[:, None] * fd["right"] + cc.contact_wire_h * fd["up"]
            drop_pts.append(base)
            drop_scale.append(np.stack([np.ones(nd), np.ones(nd), hmd - cc.contact_wire_h], axis=1))
            drop_rot = util.frame_euler(fd["right"], fd["tangent"], fd["up"])
            drop_scale[-1] = (drop_scale[-1], drop_rot)
        contact_lines.append(np.concatenate(contact))
        messenger_lines.append(np.concatenate(messenger))
    for ti in range(len(track_offsets)):
        cw = util.polyline_curve(f"ContactWire_T{ti}", contact_lines[ti], cc.contact_wire_radius, wire_mat, cyclic=True, collection=col)
        mw = util.polyline_curve(f"MessengerWire_T{ti}", messenger_lines[ti], cc.messenger_radius, wire_mat, cyclic=True, collection=col)
        cw["sbahn_track_index"] = ti; mw["sbahn_track_index"] = ti
    P = np.concatenate(drop_pts)
    S = np.concatenate([d[0] for d in drop_scale])
    R = np.concatenate([d[1] for d in drop_scale])
    util.instancer("Droppers", P, R, src_drop, scales=S, collection=col)
    return dict(mast_s=s_m.tolist(), mast_lateral_m=float(mast_lat), masts=mast_ob, cantilevers=cant_ob,
                stagger_sign=stag_sign.tolist())

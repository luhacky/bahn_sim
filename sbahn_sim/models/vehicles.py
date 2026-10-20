"""Procedural road vehicles typical for German traffic.

    small      Kleinwagen (Polo / up! class)          3.9 x 1.70 x 1.47 m
    hatchback  Kompaktklasse (Golf class)             4.3 x 1.79 x 1.46 m
    estate     Kombi (Passat Variant class)           4.8 x 1.83 x 1.50 m
    sedan      Limousine (3er / A4 class)             4.7 x 1.83 x 1.44 m
    suv        Kompakt-SUV (Tiguan class)             4.5 x 1.84 x 1.66 m
    van        Kleinbus (Transporter class)           4.9 x 1.90 x 1.97 m
    sprinter   Kastenwagen (Sprinter class)           5.9 x 2.00 x 2.60 m
    truck      7.5 t Koffer-LKW                       7.8 x 2.45 x 3.50 m

The body is lofted from cross-sections along the car (plan width, belt line,
roof line, wheel arches), then wheels with tyres and rims, lamps, EU number
plates, mirrors and a grille are added. Local frame: front towards +Y, x to
the right, ground at z = 0 (same convention as obstacles.build_car).

make_vehicle(kind, rng) -> MeshBuilder with material slots VEHICLE_SLOTS.
obstacle_parts(kind, rng, colour) -> (parts, mats) for obstacles.BUILDERS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .geom import MeshBuilder, box, tube

VEHICLE_SLOTS = ["paint", "glass", "tyre", "trim", "chrome", "headlamp", "taillamp", "plate", "rim"]
V_ = {k: i for i, k in enumerate(VEHICLE_SLOTS)}

# common German car colours (silver, greys, black, white, dark blue, red, VW blue)
CAR_COLOURS = {
    "silver": (0.55, 0.56, 0.57, 1), "grey": (0.22, 0.23, 0.24, 1), "black": (0.012, 0.012, 0.014, 1),
    "white": (0.80, 0.80, 0.78, 1), "darkblue": (0.02, 0.04, 0.10, 1), "red": (0.45, 0.03, 0.02, 1),
    "blue": (0.05, 0.14, 0.35, 1), "graphite": (0.10, 0.10, 0.11, 1), "beige": (0.55, 0.50, 0.40, 1),
    "green": (0.05, 0.14, 0.08, 1),
}


@dataclass
class Body:
    L: float
    W: float
    top: List[Tuple[float, float]]      # (t along 0 = rear .. 1 = front, z) roof / hood / boot line
    belt: List[Tuple[float, float]]     # side window sill line
    glass: Tuple[float, float]          # t range of the greenhouse
    pillars: List[Tuple[float, float]]  # (t, half width in t) body-coloured pillars between side windows
    wheels: Tuple[float, float]         # t of rear and front axle
    wheel_r: float = 0.32
    clearance: float = 0.17
    tumble: float = 0.80                # roof half width / belt half width
    nose_round: float = 0.10            # plan rounding at the ends (fraction of L)
    windscreen: Tuple[float, float] = (0.0, 0.0)   # t range where the top surface is glass (front)
    rear_glass: Tuple[float, float] = (0.0, 0.0)   # t range, rear window


BODIES = {
    "hatchback": Body(4.28, 1.79,
                      top=[(0, 0.98), (0.02, 1.10), (0.07, 1.38), (0.12, 1.45), (0.55, 1.46), (0.60, 1.42), (0.72, 1.03), (0.95, 0.88), (1.0, 0.70)],
                      belt=[(0, 0.95), (0.3, 0.98), (0.72, 0.97), (0.95, 0.86), (1.0, 0.70)],
                      glass=(0.06, 0.72), pillars=[(0.10, 0.03), (0.42, 0.012)], wheels=(0.19, 0.80),
                      windscreen=(0.58, 0.72), rear_glass=(0.02, 0.10)),
    "small": Body(3.90, 1.70,
                  top=[(0, 1.00), (0.02, 1.12), (0.06, 1.40), (0.10, 1.47), (0.56, 1.47), (0.61, 1.42), (0.73, 1.02), (0.95, 0.88), (1.0, 0.70)],
                  belt=[(0, 0.97), (0.3, 1.0), (0.73, 0.98), (0.95, 0.86), (1.0, 0.70)],
                  glass=(0.05, 0.73), pillars=[(0.09, 0.03), (0.42, 0.012)], wheels=(0.17, 0.82),
                  windscreen=(0.59, 0.73), rear_glass=(0.02, 0.09), wheel_r=0.30),
    "estate": Body(4.77, 1.83,
                   top=[(0, 1.00), (0.015, 1.18), (0.04, 1.42), (0.07, 1.49), (0.58, 1.49), (0.63, 1.43), (0.74, 1.02), (0.96, 0.86), (1.0, 0.68)],
                   belt=[(0, 0.97), (0.3, 0.98), (0.74, 0.97), (0.96, 0.85), (1.0, 0.68)],
                   glass=(0.04, 0.74), pillars=[(0.07, 0.025), (0.25, 0.015), (0.46, 0.012)], wheels=(0.21, 0.80),
                   windscreen=(0.60, 0.74), rear_glass=(0.015, 0.06)),
    "sedan": Body(4.70, 1.83,
                  top=[(0, 0.98), (0.02, 1.02), (0.13, 1.05), (0.22, 1.36), (0.28, 1.44), (0.56, 1.44), (0.61, 1.40), (0.73, 1.01), (0.96, 0.86), (1.0, 0.68)],
                  belt=[(0, 0.95), (0.13, 1.0), (0.73, 0.97), (0.96, 0.85), (1.0, 0.68)],
                  glass=(0.20, 0.73), pillars=[(0.25, 0.03), (0.46, 0.012)], wheels=(0.20, 0.80),
                  windscreen=(0.59, 0.73), rear_glass=(0.14, 0.25)),
    "suv": Body(4.50, 1.84,
                top=[(0, 1.10), (0.02, 1.30), (0.06, 1.60), (0.10, 1.66), (0.56, 1.66), (0.61, 1.60), (0.72, 1.18), (0.95, 1.05), (1.0, 0.85)],
                belt=[(0, 1.08), (0.3, 1.10), (0.72, 1.10), (0.95, 1.02), (1.0, 0.85)],
                glass=(0.05, 0.72), pillars=[(0.09, 0.03), (0.42, 0.012)], wheels=(0.19, 0.81),
                windscreen=(0.58, 0.72), rear_glass=(0.02, 0.09), wheel_r=0.36, clearance=0.20),
    "van": Body(4.90, 1.90,
                top=[(0, 1.10), (0.01, 1.90), (0.03, 1.97), (0.80, 1.97), (0.84, 1.90), (0.90, 1.25), (0.97, 1.10), (1.0, 0.85)],
                belt=[(0, 1.05), (0.84, 1.05), (0.90, 1.05), (0.97, 1.02), (1.0, 0.85)],
                glass=(0.02, 0.90), pillars=[(0.30, 0.02), (0.55, 0.02), (0.77, 0.015)], wheels=(0.14, 0.82),
                windscreen=(0.83, 0.90), rear_glass=(0.0, 0.02), wheel_r=0.34, clearance=0.18, tumble=0.93, nose_round=0.05),
    "sprinter": Body(5.93, 2.02,
                     top=[(0, 2.60), (0.02, 2.62), (0.76, 2.62), (0.80, 2.45), (0.86, 1.30), (0.96, 1.12), (1.0, 0.85)],
                     belt=[(0, 2.55), (0.72, 2.55), (0.74, 1.15), (0.86, 1.15), (0.96, 1.08), (1.0, 0.85)],
                     glass=(0.73, 0.86), pillars=[], wheels=(0.20, 0.80),
                     windscreen=(0.79, 0.86), wheel_r=0.35, clearance=0.20, tumble=0.97, nose_round=0.04),
}


def _interp(pts, t):
    a = np.array(pts)
    return np.interp(t, a[:, 0], a[:, 1])


def _body(mb: MeshBuilder, b: Body, rng):
    L, W = b.L, b.W
    ns = 56
    t = np.concatenate([[0.0], np.linspace(0.004, 0.996, ns - 2), [1.0]])
    y = (t - 0.5) * L
    ztop = _interp(b.top, t)
    zbelt = np.minimum(_interp(b.belt, t), ztop)
    # plan: rounded ends
    e = np.clip(np.minimum(t, 1 - t) / b.nose_round, 0, 1)
    plan = 0.5 * W * (0.82 + 0.18 * np.sqrt(1 - (1 - e) ** 2))
    plan[[0, -1]] *= 0.97
    # bottom line with wheel arches
    zb = np.full(ns, b.clearance)
    for tw in b.wheels:
        d = np.abs(y - (tw - 0.5) * L)
        ar = b.wheel_r + 0.05
        zb = np.maximum(zb, np.where(d < ar, np.sqrt(np.maximum(ar ** 2 - d ** 2, 0)) + b.wheel_r * 0.9, 0))
    zb = np.minimum(zb, zbelt - 0.1)
    # cross-section ring (left side going up, across the roof, down the right side, across the floor)
    hw_low = plan
    hw_belt = plan * 0.99
    greenhouse = (ztop - zbelt) > 0.15
    hw_roof = np.where(greenhouse, plan * b.tumble, plan * 0.93)
    zsill = zb + 0.12
    zsh = np.minimum(zbelt - 0.02, zsill + 0.6 * (zbelt - zsill) + 0.08)  # shoulder crease
    ring = [(-hw_low * 0.93, zb), (-hw_low, zsill), (-hw_belt * 1.0, zsh), (-hw_belt, zbelt),
            (-hw_roof, ztop - 0.04 * greenhouse), (-hw_roof * 0.75, ztop), (hw_roof * 0.75, ztop),
            (hw_roof, ztop - 0.04 * greenhouse), (hw_belt, zbelt), (hw_belt * 1.0, zsh), (hw_low, zsill), (hw_low * 0.93, zb)]
    k = len(ring)
    V = np.zeros((ns, k, 3))
    for j, (xx, zz) in enumerate(ring):
        V[:, j, 0] = xx
        V[:, j, 1] = y
        V[:, j, 2] = zz
    Vf = V.reshape(-1, 3)
    faces, mats = [], []
    in_glass = lambda tt, a: a[0] <= tt <= a[1]
    for i in range(ns - 1):
        tm = 0.5 * (t[i] + t[i + 1])
        gh = greenhouse[i] and greenhouse[i + 1]
        side_glass = gh and in_glass(tm, b.glass) and not any(abs(tm - p) < w for p, w in b.pillars)
        top_glass = in_glass(tm, b.windscreen) or in_glass(tm, b.rear_glass)
        for j in range(k):
            jn = (j + 1) % k
            q = [i * k + j, i * k + jn, (i + 1) * k + jn, (i + 1) * k + j]
            m = V_["paint"]
            if j in (3, 7) and side_glass:
                m = V_["glass"]
            elif j in (4, 5, 6) and top_glass:
                m = V_["glass"]
            elif j == 11:
                m = V_["trim"]
            elif j in (0, 10):
                m = V_["trim"] if (i % 1 == 0 and False) else V_["paint"]
            faces.append(q)
            mats.append(m)
    off = mb.add(Vf, np.array(faces), 0, smooth=True)
    # re-assign material per face
    Q = mb.Q[-1]
    mb.QM[-1] = np.array(mats)
    # end caps (fan)
    for i in (0, ns - 1):
        c = Vf[i * k:(i + 1) * k].mean(axis=0)
        cv = np.vstack([Vf[i * k:(i + 1) * k], c])
        idx = np.arange(k)
        F = np.stack([idx, (idx + 1) % k, np.full(k, k)], 1)
        if i == 0:
            F = F[:, ::-1]
        mb.add(cv, F, V_["paint"], smooth=False)
    # underbody block between the wheels (hides the arch cut-outs from the front)
    wb = b.wheels
    v, f, uv = box((W - 0.55, (wb[1] - wb[0]) * L + 2 * b.wheel_r, 0.45), (0, (0.5 * (wb[0] + wb[1]) - 0.5) * L, b.clearance + 0.3))
    mb.add(v, f, V_["trim"], uv=uv)
    return dict(ztop=ztop, zbelt=zbelt, plan=plan, y=y, t=t)


def _wheel(mb, x, y, r, width, rim_slot="rim"):
    seg = 20
    side = 1 if x > 0 else -1
    # tyre: short tube with a rounded profile (3 rings)
    pts = np.array([[x - width / 2, y, r], [x - width * 0.35, y, r], [x + width * 0.35, y, r], [x + width / 2, y, r]])
    rad = np.array([r * 0.80, r, r, r * 0.80])
    v, f, uv = tube(pts, rad, seg)
    mb.add(v, f, V_["tyre"], uv=uv, smooth=True)
    # outer sidewall + rim disc
    xo = x + side * width / 2
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = lambda rr, xx: np.stack([np.full(seg, xx), y + rr * np.cos(ang), r + rr * np.sin(ang)], 1)
    rim_r = r * 0.62
    Vr = np.vstack([ring(r * 0.80, xo), ring(rim_r, xo + side * 0.005), ring(rim_r * 0.92, xo - side * 0.01), [[xo - side * 0.03, y, r]]])
    i = np.arange(seg); j = (i + 1) % seg
    F = np.concatenate([np.stack([i, j, j + seg, i + seg], 1)])
    mb.add(Vr, F, V_["tyre"])
    off = mb.n - len(Vr)
    Fr = np.stack([i + seg, j + seg, j + 2 * seg, i + 2 * seg], 1)
    Fc = np.stack([i + 2 * seg, j + 2 * seg, np.full(seg, 3 * seg)], 1)
    mb.add(Vr, Fr, V_[rim_slot])
    mb.add(Vr, Fc, V_[rim_slot])
    # spokes suggestion: 5 dark slots
    for s in range(5):
        a = s * 2 * np.pi / 5
        c = np.array([xo + side * 0.008, y + 0.35 * rim_r * math.cos(a), r + 0.35 * rim_r * math.sin(a)])
        v, f, uv = box((0.01, 0.22 * rim_r, 0.22 * rim_r), c)
        mb.add(v, f, V_["trim"])


def _quad_on(mb, c, u, v, w, h, slot, uv=None):
    c, u, v = (np.asarray(a, float) for a in (c, u, v))
    V = np.array([c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2, c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2])
    mb.add(V, [[0, 1, 2, 3]], V_[slot], uv=uv if uv is not None else np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))


def _details(mb, b: Body, prof, rng):
    L, W = b.L, b.W
    yf, yr = L / 2, -L / 2
    zf = _interp(b.belt, 0.985)
    zr = _interp(b.belt, 0.015)
    X = np.array([1.0, 0, 0]); Z = np.array([0, 0, 1.0])
    # headlamps, grille, plate (front)
    for sx in (-1, 1):
        _quad_on(mb, (sx * W * 0.31, yf + 0.012, zf - 0.10), X, Z, W * 0.22, 0.10, "headlamp")
        _quad_on(mb, (sx * W * 0.33, yr - 0.012, zr - 0.10), X, Z, W * 0.20, 0.10, "taillamp")
    _quad_on(mb, (0, yf + 0.01, zf - 0.20), X, Z, W * 0.36, 0.14, "trim")
    _quad_on(mb, (0, yf + 0.02, b.clearance + 0.22), X, Z, 0.52, 0.11, "plate")
    _quad_on(mb, (0, yr - 0.02, zr - 0.30), X, Z, 0.52, 0.11, "plate")
    # bumpers (dark lower trim)
    v, f, uv = box((W * 0.94, 0.06, 0.16), (0, yf - 0.05, b.clearance + 0.10)); mb.add(v, f, V_["trim"], uv=uv)
    v, f, uv = box((W * 0.94, 0.06, 0.16), (0, yr + 0.05, b.clearance + 0.10)); mb.add(v, f, V_["trim"], uv=uv)
    # mirrors at the base of the A pillar
    tA = b.windscreen[0] + 0.7 * (b.windscreen[1] - b.windscreen[0]) if b.windscreen[1] else 0.7
    yA = (tA - 0.5) * L
    zA = _interp(b.belt, tA) + 0.12
    for sx in (-1, 1):
        v, f, uv = box((0.20, 0.10, 0.14), (sx * (W / 2 + 0.08), yA, zA)); mb.add(v, f, V_["paint"], uv=uv)
    # door handles / shut lines are too small to matter at > 20 m


def _wheels(mb, b: Body):
    tw = 0.21 if b.W < 2.1 else 0.26
    for ta in b.wheels:
        for sx in (-1, 1):
            _wheel(mb, sx * (b.W / 2 - tw / 2 - 0.03), (ta - 0.5) * b.L, b.wheel_r, tw)


def _truck(mb, rng):
    """7.5 t box truck: cab-over cab + box body on a ladder frame."""
    L, W = 7.8, 2.45
    cab = Body(1.9, 2.30,
               top=[(0, 2.85), (0.85, 2.85), (0.93, 2.70), (1.0, 1.60)],
               belt=[(0, 1.75), (0.85, 1.75), (1.0, 1.55)],
               glass=(0.55, 1.0), pillars=[], wheels=(0.5, 0.5), wheel_r=0.45, clearance=0.55,
               tumble=0.96, nose_round=0.05, windscreen=(0.90, 1.0))
    sub = MeshBuilder()
    prof = _body(sub, cab, rng)
    sub.transform(np.array([[1, 0, 0, 0], [0, 1, 0, L / 2 - 0.95], [0, 0, 1, 0], [0, 0, 0, 1.0]]))
    mb.extend(sub)
    # box body
    v, f, uv = box((W, 5.9, 2.6), (0, -L / 2 + 2.95, 0.95 + 1.3 + 0.35)); mb.add(v, f, V_["chrome"], uv=uv)
    v, f, uv = box((0.9, 7.0, 0.3), (0, -0.2, 0.85)); mb.add(v, f, V_["trim"], uv=uv)
    for ta, y in ((0, L / 2 - 1.2), (1, -L / 2 + 2.0)):
        for sx in (-1, 1):
            _wheel(mb, sx * (W / 2 - 0.3), y, 0.45, 0.30)
    for sx in (-1, 1):
        _quad_on(mb, (sx * 0.8, L / 2 + 0.01, 0.9), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 0.3, 0.15, "headlamp")
        _quad_on(mb, (sx * 1.0, -L / 2 - 0.01, 0.9), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 0.25, 0.12, "taillamp")
    _quad_on(mb, (0, L / 2 + 0.02, 0.62), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 0.52, 0.11, "plate")
    return (L, W)


def make_vehicle(kind: str, rng: np.random.Generator) -> MeshBuilder:
    mb = MeshBuilder()
    if kind == "truck":
        _truck(mb, rng)
        return mb
    b = BODIES[kind]
    prof = _body(mb, b, rng)
    _wheels(mb, b)
    _details(mb, b, prof, rng)
    return mb


def dims(kind):
    if kind == "truck":
        return 7.8, 2.45
    return BODIES[kind].L, BODIES[kind].W


def vehicle_materials(paint=None):
    """Materials in VEHICLE_SLOTS order. paint=None -> per-instance palette of CAR_COLOURS."""
    from . import materials as M
    if paint is None:
        p = M.palette_paint("car_paint_palette", list(CAR_COLOURS.values()))
    elif isinstance(paint, str):
        p = M.flat(f"car_paint_{paint}", CAR_COLOURS[paint], 0.3, 0.5, coat=1.0)
    else:
        p = M.flat(f"car_paint_{int(paint[0]*100)}_{int(paint[1]*100)}_{int(paint[2]*100)}", tuple(paint), 0.3, 0.5, coat=1.0)
    return [p,
            M.flat("car_glass", (0.015, 0.02, 0.025, 1), 0.02, coat=1.0),
            M.flat("tyre", (0.018, 0.018, 0.018, 1), 0.85),
            M.flat("car_trim", (0.03, 0.03, 0.03, 1), 0.55),
            M.flat("truck_box", (0.78, 0.78, 0.76, 1), 0.5),
            M.flat("headlamp", (0.85, 0.85, 0.80, 1), 0.05, coat=1.0),
            M.flat("taillamp", (0.45, 0.01, 0.01, 1), 0.1, coat=1.0),
            M.number_plate(),
            M.flat("rim_alloy", (0.55, 0.55, 0.56, 1), 0.3, metallic=1.0)]


def obstacle_parts(kind: str, rng: np.random.Generator, paint=None):
    """(parts, mats) in the format obstacles.BUILDERS returns, so obstacles.py can use
    these bodies directly: parts = [(verts, faces, mat_index), ...]."""
    mb = make_vehicle(kind, rng)
    V = np.concatenate(mb.V)
    groups = list(zip(mb.T, mb.TM)) + list(zip(mb.Q, mb.QM))
    parts = []
    for m in range(len(VEHICLE_SLOTS)):
        faces = [F[M == m] for F, M in groups if (M == m).any()]
        if not faces:
            continue
        used = np.unique(np.concatenate([f.ravel() for f in faces]))
        remap = np.full(len(V), -1)
        remap[used] = np.arange(len(used))
        parts.append((V[used], [list(remap[f]) for fs in faces for f in fs], m))
    return parts, vehicle_materials(paint)

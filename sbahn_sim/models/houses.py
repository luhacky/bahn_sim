"""Procedural German buildings for the lineside scenery.

    efh     Einfamilienhaus: 1.5 storeys, steep gable roof, optional dormer and garage
    dhh     Doppelhaus (semi-detached pair), mirrored entrances
    mfh     Mehrfamilienhaus: 3-4 storeys, window grid, balconies, stairwell strip
    barn    Scheune / Bauernhof: brick base, timber cladding, big doors
    hall    Gewerbehalle: flat roof, profiled metal cladding, ribbon windows, roller doors
    garage  Fertiggarage (double), flat roof

make_building(kind, rng) -> (MeshBuilder, footprint (L, D)) in local coordinates:
x along the ridge / long side, y depth, front (entrance) facing -y, ground at z=0.

Material slots (see SLOTS): walls pick a plaster colour per instance, roofs a
tile colour per instance, so a few meshes scattered many times still vary.
"""
from __future__ import annotations

import math

import numpy as np

from .geom import MeshBuilder, box, quad

SLOTS = ["wall", "roof", "glass", "frame", "trim", "wood", "brick", "metal", "paving", "shutter"]
S = {k: i for i, k in enumerate(SLOTS)}


def _box(mb, x0, x1, y0, y1, z0, z1, slot):
    v, f, uv = box((x1 - x0, y1 - y0, z1 - z0), ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    mb.add(v, f, S[slot], uv=uv)


def _walls(mb, L, D, h, slot="wall", plinth=0.45):
    _box(mb, -L / 2, L / 2, -D / 2, D / 2, plinth, h, slot)
    _box(mb, -L / 2 - 0.02, L / 2 + 0.02, -D / 2 - 0.02, D / 2 + 0.02, -0.3, plinth, "trim")


def _gable_triangles(mb, L, D, z0, ridge, slot="wall"):
    for x in (-L / 2, L / 2):
        V = np.array([[x, -D / 2, z0], [x, D / 2, z0], [x, 0, ridge]])
        uv = V[:, 1:]
        mb.add(V, [[0, 1, 2]] if x > 0 else [[0, 2, 1]], S[slot], uv=uv)


def _roof_gable(mb, L, D, z0, pitch_deg, eave=0.55, gable=0.35, t=0.22):
    """Two roof slabs with fascia; ridge along x. Returns ridge height."""
    rise = math.tan(math.radians(pitch_deg)) * D / 2
    ridge = z0 + rise
    run = D / 2 + eave
    drop = math.tan(math.radians(pitch_deg)) * eave
    x0, x1 = -L / 2 - gable, L / 2 + gable
    for sgn in (-1, 1):
        top = np.array([[x0, 0, ridge + t], [x1, 0, ridge + t]])
        low = np.array([[x0, sgn * run, z0 - drop + t], [x1, sgn * run, z0 - drop + t]])
        slope = math.hypot(run, rise + drop)
        # upper tiled surface with UV (u along ridge, v down the slope), metres
        V = np.array([top[0], top[1], low[1], low[0]])
        uv = np.array([[x0, 0], [x1, 0], [x1, slope], [x0, slope]])
        f = [[0, 3, 2, 1]] if sgn < 0 else [[0, 1, 2, 3]]
        mb.add(V, f, S["roof"], uv=uv)
        # underside + fascia board (trim)
        Vb = V - np.array([0, 0, t])
        mb.add(Vb, [[0, 1, 2, 3]] if sgn < 0 else [[0, 3, 2, 1]], S["trim"], uv=uv)
        e = np.array([low[0], low[1], Vb[2], Vb[3]])
        mb.add(e, [[0, 1, 2, 3]] if sgn > 0 else [[0, 3, 2, 1]], S["trim"], uv=e[:, [0, 2]])
        for xe, (a, b) in ((x0, (0, 3)), (x1, (1, 2))):
            g = np.array([V[a], V[b], Vb[b], Vb[a]])
            mb.add(g, [[0, 1, 2, 3]], S["trim"], uv=g[:, 1:])
        # gutter along the eave
        gy = sgn * (run + 0.06)
        _box(mb, x0, x1, gy - 0.06, gy + 0.06, z0 - drop - 0.05, z0 - drop + 0.06, "metal")
    # ridge cap
    _box(mb, x0, x1, -0.12, 0.12, ridge + t - 0.04, ridge + t + 0.08, "roof")
    return ridge


def _roof_hip(mb, L, D, z0, pitch_deg, eave=0.6, t=0.2):
    rise = math.tan(math.radians(pitch_deg)) * D / 2
    ridge = z0 + rise
    drop = math.tan(math.radians(pitch_deg)) * eave
    zl = z0 - drop + t
    a, b = L / 2 + eave, D / 2 + eave
    r = max(0.5, L / 2 - D / 2)
    c = [(-a, -b), (a, -b), (a, b), (-a, b)]
    R0, R1 = np.array([-r, 0, ridge + t]), np.array([r, 0, ridge + t])
    faces = [
        (np.array([[*c[0], zl], [*c[1], zl], R1, R0]), 0),
        (np.array([[*c[2], zl], [*c[3], zl], R0, R1]), 0),
        (np.array([[*c[1], zl], [*c[2], zl], R1]), 1),
        (np.array([[*c[3], zl], [*c[0], zl], R0]), 1),
    ]
    for V, tri in faces:
        e = V[1] - V[0]; e /= np.linalg.norm(e)
        n = np.cross(e, V[2] - V[0]); n /= np.linalg.norm(n)
        w = np.cross(n, e)
        uv = np.stack([(V - V[0]) @ e, -(V - V[0]) @ w], 1)
        mb.add(V, [list(range(len(V)))], S["roof"], uv=uv)
    _box(mb, -a, a, -b, b, zl - t, zl, "trim")
    for sy in (-1, 1):
        _box(mb, -a, a, sy * (b + 0.06) - 0.06, sy * (b + 0.06) + 0.06, zl - 0.12, zl, "metal")
    return ridge


def _flat_roof(mb, L, D, h, parapet=0.4):
    _box(mb, -L / 2 - 0.1, L / 2 + 0.1, -D / 2 - 0.1, D / 2 + 0.1, h, h + parapet, "trim")


def _window(mb, rng, c, u, n, w, h, shutter=None, mullion=True, frame=0.07, sill=True):
    """Window centred at c on a wall with horizontal direction u and outward normal n."""
    c = np.asarray(c, float); u = np.asarray(u, float); n = np.asarray(n, float)
    z = np.array([0, 0, 1.0])
    def rect(cx, cz, ww, hh, off, slot, depth=None):
        p = c + u * cx + z * cz + n * off
        V = np.array([p - u * ww / 2 - z * hh / 2, p + u * ww / 2 - z * hh / 2,
                      p + u * ww / 2 + z * hh / 2, p - u * ww / 2 + z * hh / 2])
        mb.add(V, [[0, 1, 2, 3]], S[slot], uv=np.array([[0, 0], [ww, 0], [ww, hh], [0, hh]]))
    # dark reveal behind the frame, then glass, then the frame bars
    rect(0, 0, w + 0.12, h + 0.12, 0.012, "trim")
    rect(0, 0, w, h, 0.02, "glass")
    fz = 0.035
    rect(0, h / 2 - frame / 2, w, frame, fz, "frame")
    rect(0, -h / 2 + frame / 2, w, frame, fz, "frame")
    rect(-w / 2 + frame / 2, 0, frame, h, fz, "frame")
    rect(w / 2 - frame / 2, 0, frame, h, fz, "frame")
    if mullion and w > 0.9:
        rect(0, 0, frame * 0.8, h, fz, "frame")
    if shutter is not None and shutter > 0.02:        # roller shutter partly down
        sh = h * shutter
        rect(0, h / 2 - sh / 2, w - 0.02, sh, 0.028, "shutter")
    if sill:
        p = c - z * (h / 2 + 0.04) + n * 0.06
        V, F, uv = box((w + 0.12, 0.14, 0.04), (0, 0, 0))
        R = np.stack([u, np.cross(z, u), z], 1)
        mb.add(V @ R.T + p, F, S["frame"], uv=uv)


def _door(mb, c, u, n, w=1.1, h=2.15, canopy=True):
    c = np.asarray(c, float); u = np.asarray(u, float); n = np.asarray(n, float)
    z = np.array([0, 0, 1.0])
    p = c + n * 0.02
    V = np.array([p - u * w / 2, p + u * w / 2, p + u * w / 2 + z * h, p - u * w / 2 + z * h])
    mb.add(V, [[0, 1, 2, 3]], S["wood"], uv=np.array([[0, 0], [w, 0], [w, h], [0, h]]))
    p2 = c + n * 0.03 + z * (h * 0.55)
    V = np.array([p2 - u * 0.15, p2 + u * 0.15, p2 + u * 0.15 + z * 0.7, p2 - u * 0.15 + z * 0.7])
    mb.add(V, [[0, 1, 2, 3]], S["glass"])
    if canopy:
        R = np.stack([u, n, z], 1)
        V2, F2, uv2 = box((w + 0.8, 1.0, 0.08), (0, 0.5, h + 0.35))
        mb.add(V2 @ R.T + c, F2, S["metal"], uv=uv2)
    # step
    R = np.stack([u, n, z], 1)
    V3, F3, uv3 = box((w + 0.6, 0.6, 0.18), (0, 0.3, -0.09))
    mb.add(V3 @ R.T + c + z * 0.18, F3, S["paving"], uv=uv3)


def _row(mb, rng, L, x_positions, y_face, n_sign, z, w, h, shutter_p=0.3, mullion=True):
    u = np.array([1.0, 0, 0]) if n_sign < 0 else np.array([-1.0, 0, 0])
    n = np.array([0, float(n_sign), 0])
    for x in x_positions:
        sh = rng.uniform(0.2, 1.0) if rng.random() < shutter_p else None
        _window(mb, rng, (x, y_face, z), u, n, w, h, shutter=sh, mullion=mullion)


def _row_side(mb, rng, D, y_positions, x_face, n_sign, z, w, h, shutter_p=0.3):
    u = np.array([0, 1.0, 0]) if n_sign > 0 else np.array([0, -1.0, 0])
    n = np.array([float(n_sign), 0, 0])
    for y in y_positions:
        sh = rng.uniform(0.2, 1.0) if rng.random() < shutter_p else None
        _window(mb, rng, (x_face, y, z), u, n, w, h, shutter=sh)


def _chimney(mb, x, y, z_base, z_top):
    _box(mb, x - 0.3, x + 0.3, y - 0.3, y + 0.3, z_base, z_top, "brick")
    _box(mb, x - 0.36, x + 0.36, y - 0.36, y + 0.36, z_top, z_top + 0.08, "trim")


def _spread(n, L, margin):
    if n <= 0:
        return []
    if n == 1:
        return [0.0]
    return list(np.linspace(-L / 2 + margin, L / 2 - margin, n))


# ----------------------------------------------------------------------
def efh(rng):
    mb = MeshBuilder()
    L, D = rng.uniform(9.5, 12.5), rng.uniform(8.0, 10.0)
    storeys = rng.choice([1, 2], p=[0.55, 0.45])
    knee = rng.uniform(0.6, 1.1) if storeys == 1 else 0.0
    h = 0.45 + 2.75 * storeys + knee
    pitch = rng.uniform(38, 48) if storeys == 1 else rng.uniform(28, 40)
    _walls(mb, L, D, h)
    ridge = _roof_gable(mb, L, D, h, pitch)
    _gable_triangles(mb, L, D, h, ridge)
    nwin = int(L // 2.6)
    for st in range(storeys):
        z = 0.45 + 1.55 + 2.75 * st
        xs = _spread(nwin, L, 1.3)
        door_x = xs[len(xs) // 2] if st == 0 else None
        _row(mb, rng, L, [x for x in xs if x != door_x], -D / 2, -1, z, 1.2, 1.35)
        _row(mb, rng, L, _spread(nwin - 1, L, 1.8), D / 2, 1, z, 1.4 if st == 0 else 1.2, 1.4 if st == 0 else 1.35)
        for sx in (-1, 1):
            _row_side(mb, rng, D, _spread(2, D, 2.2), sx * L / 2, sx, z, 1.1, 1.3)
        if st == 0:
            _door(mb, (door_x, -D / 2, 0.45), (1, 0, 0), (0, -1, 0))
    # gable attic windows
    za = h + 1.3
    if ridge - h > 2.4:
        for sx in (-1, 1):
            _row_side(mb, rng, D, [-0.8, 0.8] if D > 9 else [0.0], sx * L / 2, sx, za, 0.9, 1.1, shutter_p=0.15)
    if storeys == 1 and rng.random() < 0.6:          # dormer on the front slope
        _dormer(mb, rng, rng.uniform(-L / 4, L / 4), D, h, pitch)
    _chimney(mb, rng.uniform(-L / 3, L / 3), rng.uniform(0.4, 1.4), ridge - 1.2, ridge + 0.9)
    if rng.random() < 0.5:                           # solar panels on one slope
        _solar(mb, rng, L, D, h, pitch, side=rng.choice([-1, 1]))
    return mb, (L, D)


def _dormer(mb, rng, x, D, h, pitch):
    w = rng.uniform(2.2, 3.4)
    y_front = -D / 2 + 1.0
    dh = 1.7
    z0 = h + math.tan(math.radians(pitch)) * (D / 2 - abs(y_front))
    _box(mb, x - w / 2, x + w / 2, y_front, y_front + 2.2, z0 - 0.4, z0 + dh, "wall")
    _box(mb, x - w / 2 - 0.2, x + w / 2 + 0.2, y_front - 0.3, y_front + 2.6, z0 + dh, z0 + dh + 0.18, "roof")
    _window(mb, rng, (x, y_front, z0 + 0.8), (1, 0, 0), (0, -1, 0), w - 0.8, 1.1, shutter=None)


def _solar(mb, rng, L, D, h, pitch, side):
    a = math.radians(pitch)
    n_rows, n_cols = int(rng.integers(2, 4)), int(rng.integers(4, 8))
    pw, ph = 1.0, 1.7
    for r in range(n_rows):
        for c in range(n_cols):
            x = (c - (n_cols - 1) / 2) * (pw + 0.03)
            d = 1.2 + r * (ph + 0.03) + ph / 2      # distance down the slope from the ridge
            y = side * math.cos(a) * d
            z = h + math.tan(a) * D / 2 - math.sin(a) * d + 0.32
            V = np.array([[x - pw / 2, y - side * math.cos(a) * ph / 2, z + math.sin(a) * ph / 2],
                          [x + pw / 2, y - side * math.cos(a) * ph / 2, z + math.sin(a) * ph / 2],
                          [x + pw / 2, y + side * math.cos(a) * ph / 2, z - math.sin(a) * ph / 2],
                          [x - pw / 2, y + side * math.cos(a) * ph / 2, z - math.sin(a) * ph / 2]])
            mb.add(V, [[0, 1, 2, 3]] if side > 0 else [[0, 3, 2, 1]], S["glass"])


def dhh(rng):
    """Semi-detached pair: two EFH halves sharing the long wall axis (ridge along x)."""
    mb = MeshBuilder()
    Lh, D = rng.uniform(6.5, 8.0), rng.uniform(9.0, 11.0)
    L = 2 * Lh
    h = 0.45 + 2.75 * 2
    pitch = rng.uniform(35, 45)
    _walls(mb, L, D, h)
    ridge = _roof_gable(mb, L, D, h, pitch, gable=0.3)
    _gable_triangles(mb, L, D, h, ridge)
    for side in (-1, 1):
        x0 = side * Lh / 2
        for st in range(2):
            z = 0.45 + 1.55 + 2.75 * st
            xs = [x0 - side * 1.6, x0 + side * 1.9] if st == 0 else [x0 - side * 1.8, x0 + side * 0.2, x0 + side * 2.2]
            _row(mb, rng, L, xs, -D / 2, -1, z, 1.0 if st == 0 else 1.1, 1.3)
            _row(mb, rng, L, [x0 - side * 1.4, x0 + side * 1.6], D / 2, 1, z, 1.6, 1.4)
            _row_side(mb, rng, D, _spread(2, D, 2.4), side * L / 2, side, z, 1.1, 1.3)
        _door(mb, (x0 + side * 0.2 * 0, -D / 2, 0.45), (1, 0, 0), (0, -1, 0))
        _chimney(mb, x0 + side * 1.5, 1.0, ridge - 1.0, ridge + 0.8)
    return mb, (L, D)


def mfh(rng):
    mb = MeshBuilder()
    storeys = int(rng.integers(3, 5))
    bays = int(rng.integers(6, 11))
    bay = rng.uniform(2.6, 3.1)
    L, D = bays * bay, rng.uniform(10.5, 12.5)
    sh = 2.85
    h = 0.6 + storeys * sh
    _walls(mb, L, D, h, plinth=0.6)
    roof = rng.choice(["gable", "hip", "flat"], p=[0.55, 0.3, 0.15])
    if roof == "gable":
        ridge = _roof_gable(mb, L, D, h, rng.uniform(25, 35), eave=0.5, gable=0.3)
        _gable_triangles(mb, L, D, h, ridge)
    elif roof == "hip":
        ridge = _roof_hip(mb, L, D, h, rng.uniform(22, 30))
    else:
        _flat_roof(mb, L, D, h)
        ridge = h
    xs = [(-L / 2 + bay * (i + 0.5)) for i in range(bays)]
    stair = [xs[len(xs) // 3], xs[2 * len(xs) // 3]] if bays >= 8 else [xs[len(xs) // 2]]
    balcony_side = rng.choice([-1, 1])
    for st in range(storeys):
        z = 0.6 + 1.5 + sh * st
        _row(mb, rng, L, [x for x in xs if x not in stair], -D / 2, -1, z, 1.25, 1.4, shutter_p=0.25)
        _row(mb, rng, L, xs, D / 2, 1, z, 1.25, 1.4, shutter_p=0.25)
        for x in stair:                               # stairwell: offset half-landing windows, entrance
            if st == 0:
                _door(mb, (x, -D / 2, 0.6), (1, 0, 0), (0, -1, 0), w=1.6, h=2.3)
            else:
                _window(mb, rng, (x, -D / 2, z - 1.3), (1, 0, 0), (0, -1, 0), 0.9, 1.6, shutter=None, mullion=False)
        for sx in (-1, 1):
            _row_side(mb, rng, D, _spread(3, D, 2.5), sx * L / 2, sx, z, 1.0, 1.3, shutter_p=0.2)
    # balconies on every other bay of the garden side
    y = balcony_side * D / 2
    for i, x in enumerate(xs):
        if i % 2:
            continue
        for st in range(1 if rng.random() < 0.5 else 0, storeys):
            z = 0.6 + sh * st + 0.05
            y0, y1 = (y, y + balcony_side * 1.5)
            _box(mb, x - 1.6, x + 1.6, min(y0, y1), max(y0, y1), z, z + 0.18, "paving")
            yr = y1 - balcony_side * 0.04
            _box(mb, x - 1.6, x + 1.6, yr - 0.03, yr + 0.03, z + 0.18, z + 1.1, "metal")
            for xx in (x - 1.6, x + 1.6):
                _box(mb, xx - 0.03, xx + 0.03, min(y0, y1), max(y0, y1), z + 0.18, z + 1.1, "metal")
    if roof != "flat":
        for x in xs[1::3]:
            _chimney(mb, x, 0.8, ridge - 1.5, ridge + 0.7)
    return mb, (L, D + 3.0)


def barn(rng):
    mb = MeshBuilder()
    L, D = rng.uniform(18, 30), rng.uniform(11, 15)
    h = rng.uniform(4.5, 6.0)
    base = rng.uniform(1.2, 2.5)
    _box(mb, -L / 2, L / 2, -D / 2, D / 2, 0, base, "brick")
    _box(mb, -L / 2 + 0.05, L / 2 - 0.05, -D / 2 + 0.05, D / 2 - 0.05, base, h, "wood")
    ridge = _roof_gable(mb, L, D, h, rng.uniform(35, 48), eave=0.6, gable=0.3)
    _gable_triangles(mb, L - 0.1, D - 0.1, h, ridge, slot="wood")
    for x in _spread(int(rng.integers(1, 3)), L, 5.0):   # barn doors on the eave side
        V, F, uv = box((4.0, 0.12, 4.0), (x, -D / 2 - 0.05, 2.0))
        mb.add(V, F, S["wood"], uv=uv)
        _box(mb, x - 2.1, x + 2.1, -D / 2 - 0.12, -D / 2 - 0.06, 3.95, 4.1, "metal")
    for x in _spread(int(L // 3.5), L, 1.5):              # small stable windows
        _window(mb, rng, (x, D / 2, base * 0.6 + 0.3), (-1, 0, 0), (0, 1, 0), 0.8, 0.6, mullion=False, sill=False)
    V, F, uv = box((3.2, 0.12, 3.4), (0.0, 0, 0))
    for sx in (-1, 1):
        mb.add(V @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]]).T + np.array([sx * (L / 2 + 0.05), 0, 1.7]), F, S["wood"], uv=uv)
    return mb, (L, D)


def hall(rng):
    mb = MeshBuilder()
    L, D = rng.uniform(25, 45), rng.uniform(15, 25)
    h = rng.uniform(6, 9)
    _box(mb, -L / 2, L / 2, -D / 2, D / 2, 0, 1.0, "trim")
    _box(mb, -L / 2 + 0.03, L / 2 - 0.03, -D / 2 + 0.03, D / 2 - 0.03, 1.0, h, "metal")
    _flat_roof(mb, L, D, h, parapet=0.6)
    for y, sgn in ((-D / 2, -1), (D / 2, 1)):             # ribbon windows
        V = np.array([[-L / 2 + 2, y + sgn * 0.02, h - 2.2], [L / 2 - 2, y + sgn * 0.02, h - 2.2],
                      [L / 2 - 2, y + sgn * 0.02, h - 1.2], [-L / 2 + 2, y + sgn * 0.02, h - 1.2]])
        mb.add(V, [[0, 1, 2, 3]] if sgn < 0 else [[0, 3, 2, 1]], S["glass"])
    for x in _spread(int(rng.integers(1, 4)), L, 6.0):     # roller doors
        _box(mb, x - 2.3, x + 2.3, -D / 2 - 0.08, -D / 2 + 0.02, 0.0, 4.5, "shutter")
        _box(mb, x - 2.5, x + 2.5, -D / 2 - 0.12, -D / 2, 4.5, 4.9, "trim")
    _door(mb, (L / 2 - 3.0, -D / 2, 0.0), (1, 0, 0), (0, -1, 0), canopy=False)
    return mb, (L, D)


def garage(rng):
    mb = MeshBuilder()
    n = int(rng.integers(1, 3))
    L, D, h = 3.1 * n, 6.0, 2.7
    _box(mb, -L / 2, L / 2, -D / 2, D / 2, 0, h, "wall")
    _flat_roof(mb, L, D, h, parapet=0.2)
    for i in range(n):
        x = -L / 2 + 3.1 * (i + 0.5)
        _box(mb, x - 1.25, x + 1.25, -D / 2 - 0.03, -D / 2 + 0.01, 0.0, 2.1, "shutter")
    return mb, (L, D)


BUILDERS = dict(efh=efh, dhh=dhh, mfh=mfh, barn=barn, hall=hall, garage=garage)


def make_building(kind: str, rng: np.random.Generator):
    return BUILDERS[kind](rng)

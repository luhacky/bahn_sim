"""Closed-loop railway alignment: straights + clothoid transitions + circular
arcs in plan, smoothed gradients in profile, cant (superelevation) in curves.

Pure numpy for the maths so it can be unit-tested outside Blender; the
Blender bridge (curve object + LUT stored in a Text datablock) is at the end.

Coordinate conventions
----------------------
* Right-handed, Z up, metres. `s` is the arc length along the *centreline*
  between the two tracks (0 <= s < length_m), increasing in the running
  direction of the train (Germany drives on the right, so the train's track is
  at lateral = +track_spacing/2, the opposing track at -track_spacing/2).
* `lateral` is positive to the RIGHT of the direction of travel.
* `height` is measured from rail top (Schienenoberkante, SOK) along the canted
  "up" vector, so rails sit at height 0 and the driver's eye at ~2.45 m.
* `heading` is the plan angle of the tangent (atan2(dy, dx)); positive
  curvature = left-hand curve (counter-clockwise).
* `cant` is the roll angle (radians) of the track plane; positive raises the
  right rail (left-hand curve), as on the real railway.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, TrackConfig

LUT_TEXT_NAME = "sbahn_track_lut.json"
CENTRELINE_NAME = "TrackCentreline"

# Plan templates: polygon vertices (CCW). Corners become clothoid-arc-clothoid
# curves, edges become straights. Concave corners (dents) become reverse
# curves so the loop contains S-curves like a real DB route.
TEMPLATES: Dict[str, List[Tuple[float, float]]] = {
    # one gentle dent on the south side -> one S-curve; ~395 deg total turning
    "bean": [(0.00, 0.00), (1.40, 0.20), (2.80, 0.00), (3.45, 0.70), (3.00, 1.55),
             (1.60, 1.80), (0.15, 1.50), (-0.35, 0.75)],
    # elongated hexagon, no reverse curves, longest straights (~360 deg turning)
    "stadium": [(0.0, 0.0), (3.0, 0.0), (3.65, 0.8), (3.0, 1.6), (0.0, 1.6), (-0.65, 0.8)],
    # two dents -> two S-curves, tighter radii, more "S-Bahn in the city"
    "twist": [(0.0, 0.0), (1.2, 0.25), (2.4, 0.0), (3.2, 0.5), (3.2, 1.3), (2.2, 1.7),
              (1.3, 1.45), (0.3, 1.7), (-0.3, 0.9)],
}


# --------------------------------------------------------------------------
# Plan design
# --------------------------------------------------------------------------
@dataclass
class Corner:
    turn: float          # signed deflection (rad), + = left
    radius: float        # circular radius (m)
    lt: float            # clothoid length on each side (m)

    @property
    def arc_angle(self) -> float:
        return abs(self.turn) - self.lt / self.radius

    @property
    def tangent_length(self) -> float:
        """Distance from the polygon vertex to the start of the transition."""
        r, lt, th = self.radius, self.lt, abs(self.turn)
        shift = lt * lt / (24.0 * r)
        xm = lt / 2.0 - lt ** 3 / (240.0 * r * r)
        return (r + shift) * math.tan(th / 2.0) + xm

    @property
    def length(self) -> float:
        return 2.0 * self.lt + self.radius * self.arc_angle


def _signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def _design_corners(poly: np.ndarray, tc: TrackConfig) -> Tuple[List[Corner], np.ndarray]:
    """Pick radii / transition lengths so that every corner fits on its edges."""
    n = len(poly)
    edges = np.roll(poly, -1, axis=0) - poly
    elen = np.linalg.norm(edges, axis=1)
    turns = [_signed_angle(edges[i - 1], edges[i]) for i in range(n)]
    corners = [Corner(turn=t, radius=tc.max_radius_m, lt=tc.transition_length_m) for t in turns]

    for _ in range(60):
        ok = True
        for i in range(n):            # edge i joins corner i and corner i+1
            a, b = corners[i], corners[(i + 1) % n]
            budget = elen[i] - tc.min_straight_m
            over = a.tangent_length + b.tangent_length - budget
            if over <= 0:
                continue
            ok = False
            # shrink the larger corner first (radius, then transition length)
            for c in sorted((a, b), key=lambda c: -c.tangent_length):
                if c.radius > tc.min_radius_m:
                    need = c.tangent_length - over * 0.6
                    tan_half = math.tan(abs(c.turn) / 2.0)
                    c.radius = max(tc.min_radius_m, (need - c.lt / 2.0) / tan_half)
                    break
                elif c.lt > tc.min_transition_m:
                    c.lt = max(tc.min_transition_m, c.lt - over * 0.6)
                    break
        if ok:
            break
    for c in corners:                 # keep a circular part in every corner
        if c.arc_angle <= 0.02:
            c.lt = max(20.0, 0.45 * c.radius * abs(c.turn))
    return corners, elen


# --------------------------------------------------------------------------
# Curvature profile -> plan coordinates
# --------------------------------------------------------------------------
def _curvature_profile(straights: np.ndarray, corners: List[Corner], ds: float):
    """Return fine s grid, curvature at each sample, and segment table."""
    pieces = []           # (length, kappa_start, kappa_end, kind)
    segs = []             # human-readable segment table
    for i, c in enumerate(corners):
        pieces.append((straights[i], 0.0, 0.0, "straight"))
        k = math.copysign(1.0 / c.radius, c.turn)
        pieces.append((c.lt, 0.0, k, "spiral"))
        pieces.append((c.radius * c.arc_angle, k, k, "arc"))
        pieces.append((c.lt, k, 0.0, "spiral"))
    total = sum(p[0] for p in pieces)
    n = int(math.ceil(total / ds)) + 1
    s = np.arange(n) * ds
    kappa = np.zeros(n)
    s0 = 0.0
    for length, k0, k1, kind in pieces:
        if length <= 0:
            continue
        m = (s >= s0) & (s < s0 + length)
        t = (s[m] - s0) / length
        kappa[m] = k0 + (k1 - k0) * t
        segs.append({"kind": kind, "s0": round(s0, 2), "s1": round(s0 + length, 2),
                     "radius": (None if kind == "straight" else round(1.0 / max(abs(k0), abs(k1)), 1))})
        s0 += length
    return s, kappa, total, segs


def _integrate(s: np.ndarray, kappa: np.ndarray, heading0: float, start: np.ndarray):
    ds = np.diff(s, prepend=s[0])
    ds[0] = 0.0
    heading = heading0 + np.cumsum(0.5 * (kappa + np.roll(kappa, 1)) * ds)
    heading[0] = heading0
    cx = np.cos(heading)
    sy = np.sin(heading)
    x = start[0] + np.cumsum(0.5 * (cx + np.roll(cx, 1)) * ds)
    y = start[1] + np.cumsum(0.5 * (sy + np.roll(sy, 1)) * ds)
    x[0], y[0] = start
    return x, y, heading


def design_plan(tc: TrackConfig, seed: int, ds: float = 0.25):
    """Design the closed loop in plan. Returns dict of numpy arrays + segments."""
    rng = np.random.default_rng(seed)
    poly = np.array(TEMPLATES[tc.template], dtype=np.float64)
    poly += rng.uniform(-tc.jitter, tc.jitter, poly.shape)
    # initial scale so that the polygon perimeter ~ target length
    per = np.sum(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1))
    scale = tc.target_length_m / per
    result = None
    for _ in range(8):                                   # converge scale
        p = poly * scale
        corners, elen = _design_corners(p, tc)
        n = len(p)
        straights = np.array([elen[i] - corners[i].tangent_length -
                              corners[(i + 1) % n].tangent_length for i in range(n)])
        straights = np.maximum(straights, tc.min_straight_m)
        edges = np.roll(p, -1, axis=0) - p
        dirs = edges / elen[:, None]
        # closure: adjust straight lengths (weighted least-norm) until the
        # integrated path returns to its start
        for _it in range(12):
            s, kappa, total, segs = _curvature_profile(straights, corners, ds)
            heading0 = math.atan2(edges[0][1], edges[0][0])
            x, y, heading = _integrate(s, kappa, heading0, p[0] + dirs[0] * corners[0].tangent_length)
            err = np.array([x[-1] - x[0], y[-1] - y[0]])
            if np.linalg.norm(err) < 0.05:
                break
            A = dirs.T                                   # 2 x n
            W = np.diag(straights)
            dL = W @ A.T @ np.linalg.solve(A @ W @ A.T, -err)
            straights = np.maximum(straights + dL, tc.min_straight_m)
        result = dict(s=s, kappa=kappa, x=x, y=y, heading=heading, total=total,
                      segs=segs, corners=corners, straights=straights, closure_err=float(np.linalg.norm(err)))
        scale *= tc.target_length_m / total
        if abs(total - tc.target_length_m) / tc.target_length_m < 0.01:
            break
    # remove residual drift linearly (sub-centimetre after the solver)
    err = np.array([result["x"][-1] - result["x"][0], result["y"][-1] - result["y"][0]])
    t = result["s"] / result["s"][-1]
    result["x"] = result["x"] - err[0] * t
    result["y"] = result["y"] - err[1] * t
    return result


# --------------------------------------------------------------------------
# TrackPath: LUT and sampling
# --------------------------------------------------------------------------
class TrackPath:
    """Arc-length lookup table of the loop centreline with sampling helpers."""

    FIELDS = ("s", "x", "y", "z", "heading", "curvature", "cant", "gradient")

    def __init__(self, data: Dict[str, np.ndarray], meta: Optional[dict] = None):
        for f in self.FIELDS:
            setattr(self, f, np.asarray(data[f], dtype=np.float64))
        self.meta = meta or {}
        self.length_m = float(self.meta.get("length_m", self.s[-1] + (self.s[1] - self.s[0])))
        self._build_wrapped()

    # ---- construction ------------------------------------------------------
    @classmethod
    def build(cls, cfg: Config, ground_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None) -> "TrackPath":
        tc = cfg.track
        plan = design_plan(tc, cfg.seed)
        L = plan["total"]
        step = tc.sample_step_m
        n = int(round(L / step))
        L = n * step                        # make the loop an exact multiple of step
        s = np.arange(n) * step
        fine_s = plan["s"] * (L / plan["total"])          # rescale fine grid
        x = np.interp(s, fine_s, plan["x"])
        y = np.interp(s, fine_s, plan["y"])
        heading = np.interp(s, fine_s, plan["heading"])
        kappa = np.interp(s, fine_s, plan["kappa"])

        # ---- profile: follow smoothed terrain, limited gradient --------------
        if ground_fn is not None:
            zraw = ground_fn(x, y)
        else:
            zraw = np.zeros(n)
        z = _circular_gaussian(zraw, sigma=tc.elevation_smooth_m / step)
        z = np.mean(z) + tc.elevation_scale * (z - np.mean(z))
        grad = np.gradient(np.concatenate([z[-1:], z, z[:1]]), step)[1:-1]
        gmax = np.max(np.abs(grad)) if n else 0.0
        if gmax > tc.max_gradient:
            z = np.mean(z) + (z - np.mean(z)) * (tc.max_gradient / gmax)
        gradient = np.gradient(np.concatenate([z[-1:], z, z[:1]]), step)[1:-1]

        # ---- cant ------------------------------------------------------------
        v = tc.design_speed_kmh
        u_mm = np.minimum(tc.max_cant_mm, 0.75 * 11.8 * v * v * np.abs(kappa))
        cant = np.sign(kappa) * np.arcsin(u_mm / 1500.0)

        meta = dict(length_m=L, step_m=step, seed=cfg.seed, template=tc.template,
                    track_spacing_m=tc.track_spacing_m,
                    running_track_offset_m=tc.running_track_offset_m,
                    gauge_m=tc.gauge_m, design_speed_kmh=v,
                    closure_err_m=plan["closure_err"],
                    segments=plan["segs"],
                    min_radius_m=float(min(c.radius for c in plan["corners"])),
                    max_gradient=float(np.max(np.abs(gradient))),
                    bbox=[float(x.min()), float(y.min()), float(x.max()), float(y.max())])
        return cls(dict(s=s, x=x, y=y, z=z, heading=heading, curvature=kappa, cant=cant, gradient=gradient), meta)

    def _build_wrapped(self):
        # append the first sample at s = L so np.interp wraps cleanly
        L = self.length_m
        self._ws = np.concatenate([self.s, [L]])
        self._wrapped = {}
        for f in self.FIELDS[1:]:
            a = getattr(self, f)
            first = a[0] + (2 * math.pi if f == "heading" and a[-1] > a[0] + math.pi else 0.0)
            self._wrapped[f] = np.concatenate([a, [first]])

    # ---- queries -----------------------------------------------------------
    def wrap(self, s):
        return np.mod(s, self.length_m)

    def distance_ahead(self, s_from: float, s_to: float) -> float:
        """Along-track distance from s_from forward to s_to (0 <= d < L)."""
        return float(self.wrap(s_to - s_from))

    def _interp(self, field: str, s):
        return np.interp(self.wrap(s), self._ws, self._wrapped[field])

    def sample(self, s, upright: bool = False) -> Dict[str, np.ndarray]:
        """Vectorised: pos (N,3), tangent, right, up (canted), heading, curvature, cant, gradient.

        upright=True returns a frame with a horizontal tangent, horizontal right
        vector and world-Z up (for masts, signals, walls and other vertical things)."""
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        x, y, z = self._interp("x", s), self._interp("y", s), self._interp("z", s)
        h = self._interp("heading", s)
        g = self._interp("gradient", s)
        c = self._interp("cant", s)
        k = self._interp("curvature", s)
        # tangent includes the slope; right/up are rolled by the cant
        if upright:
            th = np.zeros_like(g)
            c = np.zeros_like(c)
        else:
            th = np.arctan(g)
        tangent = np.stack([np.cos(h) * np.cos(th), np.sin(h) * np.cos(th), np.sin(th)], axis=1)
        right0 = np.stack([np.sin(h), -np.cos(h), np.zeros_like(h)], axis=1)
        up0 = np.cross(right0, tangent)
        right = np.cos(c)[:, None] * right0 + np.sin(c)[:, None] * up0
        up = -np.sin(c)[:, None] * right0 + np.cos(c)[:, None] * up0
        return dict(pos=np.stack([x, y, z], axis=1), tangent=tangent, right=right, up=up,
                    heading=h, curvature=k, cant=c, gradient=g)

    def point(self, s, lateral=0.0, height=0.0, upright: bool = False) -> np.ndarray:
        """World position(s) at arc length s, lateral offset (+right) and height above rail top."""
        f = self.sample(s, upright)
        lateral = np.asarray(lateral, dtype=np.float64).reshape(-1, 1)
        height = np.asarray(height, dtype=np.float64).reshape(-1, 1)
        return f["pos"] + lateral * f["right"] + height * f["up"]

    def matrix(self, s: float, lateral: float = 0.0, height: float = 0.0, yaw: float = 0.0, upright: bool = False):
        """4x4 mathutils.Matrix: local X = right, local Y = forward, local Z = up.

        Objects modelled 'facing +Y' can be placed with obj.matrix_world = tp.matrix(s).
        upright=True keeps local Z vertical (no cant / gradient roll)."""
        from mathutils import Matrix, Vector
        f = self.sample(s, upright)
        t, r, u = (Vector(f[k][0]) for k in ("tangent", "right", "up"))
        pos = Vector(f["pos"][0]) + lateral * r + height * u
        m = Matrix.Identity(4)
        m.col[0][:3] = r
        m.col[1][:3] = t
        m.col[2][:3] = u
        m.col[3][:3] = pos
        if yaw:
            m = m @ Matrix.Rotation(yaw, 4, "Z")
        return m

    def nearest_s(self, x, y) -> np.ndarray:
        """Arc length of the closest centreline sample to each (x, y). O(N*M) numpy; fine for <1e4 queries."""
        x = np.atleast_1d(x); y = np.atleast_1d(y)
        out = np.empty(len(x))
        for i in range(0, len(x), 256):
            dx = x[i:i + 256, None] - self.x[None, :]
            dy = y[i:i + 256, None] - self.y[None, :]
            out[i:i + 256] = self.s[np.argmin(dx * dx + dy * dy, axis=1)]
        return out

    def kdtree(self):
        """mathutils.kdtree over centreline samples (index == LUT index)."""
        from mathutils import kdtree
        kd = kdtree.KDTree(len(self.s))
        for i in range(len(self.s)):
            kd.insert((self.x[i], self.y[i], 0.0), i)
        kd.balance()
        return kd

    # ---- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        d = {f: np.round(getattr(self, f), 6).tolist() for f in self.FIELDS}
        d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrackPath":
        return cls({f: np.asarray(d[f]) for f in cls.FIELDS}, d.get("meta", {}))

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_json(cls, path: str) -> "TrackPath":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def store_in_scene(self) -> None:
        import bpy
        txt = bpy.data.texts.get(LUT_TEXT_NAME) or bpy.data.texts.new(LUT_TEXT_NAME)
        txt.clear()
        txt.write(json.dumps(self.to_dict()))
        txt.use_fake_user = True
        scene = bpy.context.scene
        scene["sbahn_track_length_m"] = self.length_m
        scene["sbahn_track_spacing_m"] = self.meta.get("track_spacing_m", 4.0)
        scene["sbahn_running_track_offset_m"] = self.meta.get("running_track_offset_m", 2.0)

    @classmethod
    def from_scene(cls) -> "TrackPath":
        """Rebuild the TrackPath from the LUT stored in the open .blend."""
        import bpy
        txt = bpy.data.texts.get(LUT_TEXT_NAME)
        if txt is None:
            raise RuntimeError(f"No '{LUT_TEXT_NAME}' text datablock: build the scene first.")
        return cls.from_dict(json.loads(txt.as_string()))

    # ---- Blender curve -----------------------------------------------------
    def to_curve_object(self, name: str = CENTRELINE_NAME, lateral: float = 0.0, height: float = 0.0,
                        step_m: float = 2.0, bevel_depth: float = 0.0, collection=None):
        """Create a closed POLY curve following the loop (tilt = cant)."""
        import bpy
        n = max(8, int(round(self.length_m / step_m)))
        s = np.arange(n) * (self.length_m / n)
        pts = self.point(s, lateral, height)
        cu = bpy.data.curves.new(name, "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = bevel_depth
        cu.resolution_u = 1
        sp = cu.splines.new("POLY")
        sp.points.add(n - 1)
        co = np.concatenate([pts, np.ones((n, 1))], axis=1).ravel()
        sp.points.foreach_set("co", co.tolist())
        sp.points.foreach_set("tilt", self._interp("cant", s).tolist())
        sp.use_cyclic_u = True
        ob = bpy.data.objects.new(name, cu)
        (collection or bpy.context.scene.collection).objects.link(ob)
        ob["sbahn_lateral_m"] = lateral
        return ob


def _circular_gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(a) < 4:
        return a.copy()
    n = len(a)
    k = np.arange(n)
    k = np.minimum(k, n - k).astype(np.float64)
    kern = np.exp(-0.5 * (k / sigma) ** 2)
    kern /= kern.sum()
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(kern)))

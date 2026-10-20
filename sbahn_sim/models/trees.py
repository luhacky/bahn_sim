"""Procedural Central-European trees and shrubs.

A small recursive branching grower (in the spirit of Weber & Penn 1995) with
species presets for the trees that dominate German rail corridors:

    spruce  (Fichte, Picea abies)      conical, drooping side branches, needle brushes
    pine    (Kiefer, Pinus sylvestris) tall clear bole, orange upper bark, flat crown of needle tufts
    beech   (Buche, Fagus sylvatica)   smooth grey bark, dense ovoid crown
    oak     (Eiche, Quercus robur)     gnarled limbs, broad irregular crown
    birch   (Birke, Betula pendula)    white bark, open crown, hanging twigs
    maple   (Ahorn, Acer)              round crown, big leaves (street / garden tree)
    shrub   (Hasel / Weißdorn)         multi-stem bush, 2 to 4 m
    sapling (young broadleaf)          2 to 5 m, forest edges

Every tree is real geometry (bark tubes + individual leaves / needle tufts, no
alpha cards), so it stays opaque and cheap for Cycles. Two detail levels:
lod=0 for the ribbon near the track, lod=1 (fewer, larger leaves, thinner
branch structure) for the far scatter.

make_tree(species, rng, lod) -> MeshBuilder with material slots
    0 bark, 1 foliage     (per-vertex attributes: leaf_rnd, crown_ao)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .geom import MeshBuilder, rotate_about, tube

GOLDEN = math.radians(137.5)


# ----------------------------------------------------------------------
# crown envelopes: rel = 0 at crown base .. 1 at the top -> length factor
# ----------------------------------------------------------------------
def env_conical(rel):
    return 0.08 + 0.92 * (1.0 - rel) ** 0.9


def env_ovoid(rel):
    return 0.25 + 0.75 * math.sin(math.pi * min(1.0, 0.12 + 0.88 * rel) ** 0.8)


def env_broad(rel):
    return 0.45 + 0.55 * math.sin(math.pi * min(1.0, 0.25 + 0.75 * rel))


def env_flat_top(rel):
    return 0.3 + 0.7 * math.sin(math.pi * (0.2 + 0.6 * rel))


@dataclass
class Level:
    n: int                      # children per parent (per metre if per_m)
    t: tuple = (0.3, 1.0)       # range along the parent where children start
    angle: tuple = (50, 10)     # deg away from the parent direction (mean, sd)
    length: tuple = (0.5, 0.1)  # fraction of parent length (mean, sd)
    radius: float = 0.6         # child base radius / parent radius at the start
    segs: int = 6               # polyline segments
    sides: int = 6              # tube sides
    tropism: float = 0.0        # + bends up, - droops, per segment
    wobble: float = 0.1         # random direction change per segment
    per_m: bool = False         # n is per metre of parent
    leaves: float = 0.0         # leaves / needle tufts per metre on this level (0 = none)
    leaf_t0: float = 0.25       # leaves only beyond this fraction
    min_len: float = 0.05


@dataclass
class Species:
    name: str
    height: tuple               # (min, max) m
    crown_base: tuple           # fraction of height where branches start
    trunk_r: float              # radius / height
    trunk_sides: int
    trunk_top: float            # trunk polyline reaches this fraction of height
    envelope: Callable
    levels: List[Level]
    leaf: str = "broad"         # broad / needle / needle_tuft
    leaf_size: tuple = (0.10, 0.06)   # length, width (m) at lod 0
    leaf_up: float = 0.5        # how much leaves turn their face to the sky
    trunk_wobble: float = 0.02
    level1_len: float = 0.35    # level-1 max length / height
    whorl: int = 0              # conifers: branches per whorl (0 = spiral)
    whorl_step: float = 0.45    # m between whorls
    lean: float = 0.0           # random trunk lean (rad)
    stems: int = 1              # shrubs: several stems from the ground
    lod1_leaf_scale: float = 1.9
    lod1_keep: float = 0.35
    leaf_count: int = 40000     # leaf elements at lod 0 (sites are resampled to this, giving clumps)
    lod1_leaf_frac: float = 0.28
    leaf_jitter: float = 0.15   # m, spreads resampled leaves into clumps


class Grower:
    def __init__(self, sp: Species, rng: np.random.Generator, lod: int = 0):
        self.sp, self.rng, self.lod = sp, rng, lod
        self.bark = MeshBuilder(attrs=("leaf_rnd", "crown_ao"))
        self.leaf_pos, self.leaf_dir, self.leaf_twig = [], [], []
        self.H = float(rng.uniform(*sp.height))
        self.tubes = 0

    # ------------------------------------------------------------------
    def _polyline(self, p0, d0, length, segs, tropism, wobble, droop_curve=0.0):
        rng = self.rng
        pts = [np.asarray(p0, dtype=np.float64)]
        d = np.asarray(d0, dtype=np.float64) / np.linalg.norm(d0)
        step = length / segs
        for i in range(segs):
            d = d + rng.normal(0, wobble, 3)
            d[2] += tropism * (1.0 + droop_curve * i / segs)
            d /= np.linalg.norm(d)
            pts.append(pts[-1] + d * step)
        return np.array(pts)

    def _emit_tube(self, pts, r0, r1, sides):
        if self.lod == 1:
            sides = max(3, sides - 2)
        n = len(pts)
        radii = r0 + (r1 - r0) * np.linspace(0, 1, n) ** 0.9
        v, f, uv = tube(pts, radii, sides, uv_scale=(1.0, 1.0))
        self.bark.add(v, f, 0, uv=uv, smooth=True)
        self.tubes += 1

    @staticmethod
    def _interp(pts, t):
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        L = cum[-1]
        x = t * L
        i = int(np.clip(np.searchsorted(cum, x) - 1, 0, len(seg) - 1))
        a = (x - cum[i]) / max(seg[i], 1e-9)
        p = pts[i] * (1 - a) + pts[i + 1] * a
        d = pts[i + 1] - pts[i]
        return p, d / max(np.linalg.norm(d), 1e-9), L

    # ------------------------------------------------------------------
    def grow(self):
        sp, rng, H = self.sp, self.rng, self.H
        r_trunk = sp.trunk_r * H
        for stem in range(sp.stems):
            if sp.stems > 1:
                a = rng.uniform(0, 2 * math.pi)
                base = np.array([math.cos(a), math.sin(a), 0]) * rng.uniform(0.05, 0.35)
                d0 = np.array([math.cos(a) * 0.35, math.sin(a) * 0.35, 1.0]) + rng.normal(0, 0.1, 3)
                h = H * rng.uniform(0.7, 1.0)
            else:
                base = np.zeros(3)
                lean = rng.normal(0, sp.lean) if sp.lean else 0.0
                la = rng.uniform(0, 2 * math.pi)
                d0 = np.array([math.sin(lean) * math.cos(la), math.sin(lean) * math.sin(la), 1.0])
                h = H
            trunk_len = h * sp.trunk_top
            segs = 14 if self.lod == 0 else 8
            pts = self._polyline(base - np.array([0, 0, 0.3]), d0, trunk_len + 0.3, segs, 0.0, sp.trunk_wobble)
            rr = r_trunk * (1 if sp.stems == 1 else 0.45)
            radii = rr * (1.0 - 0.85 * np.linspace(0, 1, len(pts)) ** 1.1)
            radii[:2] *= np.array([1.45, 1.15])                 # root flare
            v, f, uv = tube(pts, radii, sp.trunk_sides if self.lod == 0 else max(5, sp.trunk_sides - 3))
            self.bark.add(v, f, 0, uv=uv, smooth=True)
            if sp.whorl:
                self._conifer_branches(pts, radii, h)
            else:
                self._children(pts, radii[0], 0, trunk_len, crown=(sp.crown_base[0], h))
        self._make_leaves()
        return self

    def _level(self, i) -> Optional[Level]:
        lv = self.sp.levels
        return lv[i] if i < len(lv) else None

    def _children(self, ppts, pr0, li, plen, crown=None):
        """Spawn children of level li along parent polyline ppts."""
        sp, rng = self.sp, self.rng
        lv = self._level(li)
        if lv is None:
            return
        keep = 1.0 if self.lod == 0 else (self.sp.lod1_keep if li >= len(sp.levels) - 1 else 0.7)
        n = lv.n * plen if lv.per_m else lv.n
        n = int(max(0, rng.poisson(max(n * keep, 0.0)))) if lv.per_m else int(round(n * (keep if li > 0 else 1.0)))
        phase = rng.uniform(0, 2 * math.pi)
        last = li == len(sp.levels) - 1
        for k in range(n):
            if li == 0 and crown is not None:
                t = lv.t[0] + (lv.t[1] - lv.t[0]) * (k + rng.uniform(0.15, 0.85)) / max(n, 1)
            else:
                t = rng.uniform(*lv.t)
            p, d, L = self._interp(ppts, t)
            if li == 0:
                rel = np.clip((p[2] - sp.crown_base[0] * self.H) / max(self.H * (1 - sp.crown_base[0]), 1e-3), 0, 1)
                length = self.H * sp.level1_len * sp.envelope(rel) * rng.uniform(0.8, 1.15)
            else:
                length = plen * max(0.05, rng.normal(*lv.length)) * (1.0 - 0.5 * t)
            if length < lv.min_len:
                continue
            ang = math.radians(max(5.0, rng.normal(*lv.angle)))
            rot = phase + k * GOLDEN + rng.normal(0, 0.3)
            side = np.cross(d, [0, 0, 1.0]) if abs(d[2]) < 0.95 else np.cross(d, [1.0, 0, 0])
            side /= np.linalg.norm(side)
            side = rotate_about(side[None], d, rot)[0]
            cd = rotate_about(d[None], side, ang)[0]
            segs = lv.segs if self.lod == 0 else max(2, lv.segs - 2)
            pts = self._polyline(p, cd, length, segs, lv.tropism, lv.wobble)
            pr_at = pr0 * (1 - 0.85 * t)
            r0 = max(0.004, pr_at * lv.radius)
            r1 = max(0.002, r0 * 0.25)
            if r0 > 0.006 or self.lod == 0:
                self._emit_tube(pts, r0, r1, lv.sides)
            if lv.leaves > 0:
                self._add_leaf_sites(pts, length, lv)
            if not last:
                self._children(pts, r0, li + 1, length)

    def _conifer_branches(self, tpts, tradii, h):
        """Whorls of level-0 branches along the trunk (spruce / pine)."""
        sp, rng = self.sp, self.rng
        lv = sp.levels[0]
        z0 = sp.crown_base[0] * h * rng.uniform(0.9, 1.1)
        step = sp.whorl_step * (1.0 if self.lod == 0 else 1.6)
        seg = np.linalg.norm(np.diff(tpts, axis=0), axis=1)
        L = seg.sum()
        z = z0
        while z < L - 0.3:
            t = z / L
            p, d, _ = self._interp(tpts, t)
            rel = (z - z0) / max(L - z0, 1e-3)
            n = max(2, int(round(sp.whorl * rng.uniform(0.8, 1.2))))
            phase = rng.uniform(0, 2 * math.pi)
            for k in range(n):
                a = phase + 2 * math.pi * k / n + rng.normal(0, 0.25)
                horiz = np.array([math.cos(a), math.sin(a), 0.0])
                ang = math.radians(rng.normal(*lv.angle))
                cd = horiz * math.sin(ang) + np.array([0, 0, 1.0]) * math.cos(ang)
                length = self.H * sp.level1_len * sp.envelope(rel) * rng.uniform(0.8, 1.15)
                if length < 0.3:
                    continue
                # droop grows with branch length and position low in the crown
                trop = lv.tropism * (0.5 + 1.0 * (1 - rel))
                segs = lv.segs if self.lod == 0 else max(3, lv.segs - 2)
                pts = self._polyline(p, cd, length, segs, trop, lv.wobble, droop_curve=1.5)
                r0 = max(0.008, float(np.interp(t, np.linspace(0, 1, len(tradii)), tradii)) * lv.radius)
                self._emit_tube(pts, r0, 0.004, lv.sides)
                if lv.leaves:
                    self._add_leaf_sites(pts, length, lv)
                self._children(pts, r0, 1, length)
            z += step * rng.uniform(0.8, 1.25) * (0.7 + 0.6 * (1 - rel))
        # leader tip tuft
        self._add_leaf_sites(tpts[-3:], np.linalg.norm(tpts[-1] - tpts[-3]), Level(0, leaves=40, leaf_t0=0.0))

    # ------------------------------------------------------------------
    def _add_leaf_sites(self, pts, length, lv: Level):
        rng = self.rng
        dens = lv.leaves * (1.0 if self.lod == 0 else 1.0 / (self.sp.lod1_leaf_scale ** 2) * 1.3)
        n = rng.poisson(max(0.0, dens * length * (1 - lv.leaf_t0)))
        if n == 0:
            return
        t = rng.uniform(lv.leaf_t0, 1.0, n) ** 0.7
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        x = t * cum[-1]
        i = np.clip(np.searchsorted(cum, x) - 1, 0, len(seg) - 1)
        a = ((x - cum[i]) / np.maximum(seg[i], 1e-9))[:, None]
        P = pts[i] * (1 - a) + pts[i + 1] * a
        D = pts[i + 1] - pts[i]
        D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
        self.leaf_pos.append(P)
        self.leaf_twig.append(D)

    def _make_leaves(self):
        if not self.leaf_pos:
            return
        sp, rng = self.sp, self.rng
        P = np.concatenate(self.leaf_pos)
        D = np.concatenate(self.leaf_twig)
        target = int(sp.leaf_count * (1.0 if self.lod == 0 else sp.lod1_leaf_frac) * rng.uniform(0.9, 1.1))
        pick = rng.choice(len(P), target, replace=target > len(P))
        P = P[pick] + rng.normal(0, sp.leaf_jitter, (target, 3)) * (1.0 if self.lod == 0 else 1.6)
        D = D[pick]
        n = len(P)
        scale = 1.0 if self.lod == 0 else sp.lod1_leaf_scale
        # crown ambient-occlusion proxy: distance from the crown's centroid axis, normalised
        c = P.mean(axis=0)
        rad = np.linalg.norm((P - c)[:, :2], axis=1)
        zrel = (P[:, 2] - P[:, 2].min()) / max(np.ptp(P[:, 2]), 1e-3)
        ao = np.clip(0.35 + 0.65 * (rad / max(np.percentile(rad, 95), 1e-3)) ** 1.2, 0, 1)
        ao = np.clip(ao * (0.75 + 0.35 * zrel), 0, 1)
        rnd = rng.random(n)
        if sp.leaf == "broad":
            self._broad_leaves(P, D, scale, ao, rnd)
        elif sp.leaf == "needle":
            self._spruce_sprays(P, D, scale, ao, rnd)
        else:
            self._needle_tufts(P, D, scale, ao, rnd)

    def _broad_leaves(self, P, D, scale, ao, rnd):
        sp, rng = self.sp, self.rng
        n = len(P)
        L = sp.leaf_size[0] * scale * rng.uniform(0.75, 1.25, n)
        W = sp.leaf_size[1] * scale * rng.uniform(0.8, 1.2, n)
        R = rng.normal(0, 1, (n, 3))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        A = D * 0.6 + R * 0.8 + np.array([0, 0, -0.15 if sp.name == "birch" else 0.1])
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        up = np.array([0, 0, 1.0])
        # leaf normal: turn towards the sky by leaf_up
        N0 = np.cross(A, rng.normal(0, 1, (n, 3)))
        N0 /= np.maximum(np.linalg.norm(N0, axis=1, keepdims=True), 1e-9)
        N0 *= np.sign(N0[:, 2:3] + 1e-6)
        N = N0 * (1 - sp.leaf_up) + up * sp.leaf_up
        N -= (N * A).sum(1, keepdims=True) * A
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        S = np.cross(N, A)
        base = P + A * (0.02 * scale)
        fold = (0.18 * W)[:, None] * N
        v0 = base
        v1 = base + A * (L * 0.42)[:, None] + S * (W * 0.5)[:, None] - fold
        v2 = base + A * L[:, None] - N * (0.12 * L)[:, None]
        v3 = base + A * (L * 0.42)[:, None] - S * (W * 0.5)[:, None] - fold
        V = np.stack([v0, v1, v2, v3], axis=1).reshape(-1, 3)
        k = np.arange(n)[:, None] * 4
        F = np.concatenate([k + np.array([0, 1, 2]), k + np.array([0, 2, 3])])
        uv = np.tile(np.array([[0.5, 0], [0, 0.42], [0.5, 1], [1, 0.42]]), (n, 1))
        self.bark.add(V, F, 1, uv=uv, smooth=False, leaf_rnd=np.repeat(rnd, 4), crown_ao=np.repeat(ao, 4))

    def _spruce_sprays(self, P, D, scale, ao, rnd):
        """Spruce: each element is a flattened, serrated shoot spray hanging from the twig
        (reads as the dense 'curtain' of a Fichte from > 10 m)."""
        sp, rng = self.sp, self.rng
        n = len(P)
        L = sp.leaf_size[0] * scale * rng.uniform(0.7, 1.3, n)
        W = sp.leaf_size[1] * scale * rng.uniform(0.8, 1.2, n)
        A = D + rng.normal(0, 0.35, (n, 3)) + np.array([0, 0, -0.35])
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        up = np.array([0, 0, 1.0])
        N = up - (A @ up)[:, None] * A
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        N = N + rng.normal(0, 0.25, (n, 3))
        N -= (N * A).sum(1, keepdims=True) * A
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        S = np.cross(N, A)
        fold = (0.35 * W)[:, None] * N          # sides hang down: a shallow roof over the twig
        a = lambda f: A * (L * f)[:, None]
        s_ = lambda f: S * (W * f)[:, None]
        v0 = P
        v1 = P + a(0.3) + s_(0.5) - fold
        v2 = P + a(0.3) - s_(0.5) - fold
        v3 = P + a(0.75) + s_(0.38) - fold * 0.8
        v4 = P + a(0.75) - s_(0.38) - fold * 0.8
        v5 = P + a(1.0)
        V = np.stack([v0, v1, v2, v3, v4, v5], axis=1).reshape(-1, 3)
        k = np.arange(n)[:, None] * 6
        F = np.concatenate([k + np.array([0, 1, 2]), k + np.array([1, 3, 2]), k + np.array([2, 3, 4]), k + np.array([3, 5, 4])])
        uv = np.tile(np.array([[0.5, 0], [0, 0.3], [1, 0.3], [0.1, 0.75], [0.9, 0.75], [0.5, 1]]), (n, 1))
        self.bark.add(V, F, 1, uv=uv, leaf_rnd=np.repeat(rnd, 6), crown_ao=np.repeat(ao, 6))

    def _needle_brush(self, P, D, scale, ao, rnd):
        """Spruce: each site is a short flattened brush of needles (8 blades) around the twig."""
        sp, rng = self.sp, self.rng
        n = len(P)
        nb = 7 if self.lod == 0 else 5
        L = sp.leaf_size[0] * scale * rng.uniform(0.8, 1.2, (n, nb))
        Wd = sp.leaf_size[1] * scale
        ref = np.where(np.abs(D[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
        U = np.cross(D, ref); U /= np.linalg.norm(U, axis=1, keepdims=True)
        Wv = np.cross(D, U)
        ang = np.linspace(0, 2 * np.pi, nb, endpoint=False)[None, :] + rng.uniform(0, 6.3, (n, 1))
        radial = np.cos(ang)[..., None] * U[:, None, :] + np.sin(ang)[..., None] * Wv[:, None, :]
        # flatten the brush (spruce shoots are denser on the upper side) and sweep needles forward
        radial[..., 2] *= 0.55
        radial /= np.linalg.norm(radial, axis=2, keepdims=True)
        dirv = radial * 0.75 + D[:, None, :] * 0.65
        dirv /= np.linalg.norm(dirv, axis=2, keepdims=True)
        side = np.cross(dirv, D[:, None, :])
        side /= np.maximum(np.linalg.norm(side, axis=2, keepdims=True), 1e-9)
        base = P[:, None, :] + rng.normal(0, 0.01, (n, nb, 3))
        tip = base + dirv * L[..., None]
        b0 = base + side * (Wd * 0.5)
        b1 = base - side * (Wd * 0.5)
        V = np.stack([b0, b1, tip], axis=2).reshape(-1, 3)
        m = n * nb
        F = np.arange(m)[:, None] * 3 + np.array([0, 1, 2])
        uv = np.tile(np.array([[0, 0], [1, 0], [0.5, 1]]), (m, 1))
        self.bark.add(V, F, 1, uv=uv, leaf_rnd=np.repeat(rnd, nb * 3), crown_ao=np.repeat(ao, nb * 3))

    def _needle_tufts(self, P, D, scale, ao, rnd):
        """Pine: tufts of long needles (pompons) at the shoot ends."""
        sp, rng = self.sp, self.rng
        n = len(P)
        nb = 8 if self.lod == 0 else 6
        L = sp.leaf_size[0] * scale * rng.uniform(0.8, 1.2, (n, nb))
        Wd = sp.leaf_size[1] * scale
        R = rng.normal(0, 1, (n, nb, 3))
        R /= np.linalg.norm(R, axis=2, keepdims=True)
        dirv = R + D[:, None, :] * 0.9 + np.array([0, 0, 0.35])
        dirv /= np.linalg.norm(dirv, axis=2, keepdims=True)
        side = np.cross(dirv, R[:, ::-1])
        side /= np.maximum(np.linalg.norm(side, axis=2, keepdims=True), 1e-9)
        base = P[:, None, :]
        tip = base + dirv * L[..., None]
        V = np.stack([base + side * Wd * 0.5, base - side * Wd * 0.5, tip], axis=2).reshape(-1, 3)
        m = n * nb
        F = np.arange(m)[:, None] * 3 + np.array([0, 1, 2])
        uv = np.tile(np.array([[0, 0], [1, 0], [0.5, 1]]), (m, 1))
        self.bark.add(V, F, 1, uv=uv, leaf_rnd=np.repeat(rnd, nb * 3), crown_ao=np.repeat(ao, nb * 3))


# ----------------------------------------------------------------------
# species presets
# ----------------------------------------------------------------------
SPECIES = {
    "spruce": Species(
        "spruce", height=(22, 30), crown_base=(0.12, 0), trunk_r=0.014, trunk_sides=10, trunk_top=0.99,
        envelope=env_conical, level1_len=0.21, whorl=5, whorl_step=0.42, leaf="needle",
        leaf_size=(0.40, 0.17), lod1_leaf_scale=1.8, leaf_count=28000, leaf_jitter=0.05,
        levels=[
            Level(0, angle=(80, 8), radius=0.18, segs=7, sides=5, tropism=-0.07, wobble=0.04,
                  leaves=14, leaf_t0=0.35),
            Level(2.2, per_m=True, t=(0.15, 0.95), angle=(55, 12), length=(0.32, 0.08), radius=0.5, segs=3,
                  sides=3, tropism=-0.12, wobble=0.08, leaves=46, leaf_t0=0.0),
            Level(2.0, per_m=True, t=(0.2, 0.9), angle=(50, 10), length=(0.45, 0.1), radius=0.5, segs=2,
                  sides=3, tropism=-0.25, wobble=0.1, leaves=46, leaf_t0=0.0, min_len=0.08),
        ]),
    "pine": Species(
        "pine", height=(20, 27), crown_base=(0.62, 0), trunk_r=0.016, trunk_sides=10, trunk_top=0.94,
        envelope=env_flat_top, level1_len=0.24, whorl=4, whorl_step=0.9, leaf="needle_tuft",
        leaf_size=(0.26, 0.035), leaf_count=9000, leaf_jitter=0.12, lean=0.06, trunk_wobble=0.05, lod1_leaf_scale=1.7,
        levels=[
            Level(0, angle=(62, 12), radius=0.35, segs=6, sides=5, tropism=0.05, wobble=0.14),
            Level(3.5, per_m=True, t=(0.25, 1.0), angle=(45, 12), length=(0.4, 0.1), radius=0.5, segs=3,
                  sides=4, tropism=0.08, wobble=0.18),
            Level(2.5, per_m=True, t=(0.4, 1.0), angle=(40, 12), length=(0.4, 0.1), radius=0.55, segs=2,
                  sides=3, tropism=0.12, wobble=0.15, leaves=34, leaf_t0=0.35),
        ]),
    "beech": Species(
        "beech", height=(22, 30), crown_base=(0.35, 0), trunk_r=0.017, trunk_sides=12, trunk_top=0.72,
        envelope=env_ovoid, level1_len=0.40, leaf="broad", leaf_size=(0.22, 0.14), leaf_up=0.55, leaf_count=50000,
        leaf_jitter=0.25,
        trunk_wobble=0.03, lod1_leaf_scale=1.9,
        levels=[
            Level(9, t=(0.40, 1.0), angle=(44, 10), radius=0.55, segs=8, sides=7, tropism=0.02, wobble=0.09),
            Level(2.0, per_m=True, t=(0.15, 1.0), angle=(52, 12), length=(0.45, 0.1), radius=0.5, segs=5,
                  sides=5, tropism=0.0, wobble=0.12, leaves=30, leaf_t0=0.4),
            Level(4.0, per_m=True, t=(0.15, 1.0), angle=(50, 14), length=(0.45, 0.12), radius=0.5, segs=3,
                  sides=3, tropism=-0.02, wobble=0.15, leaves=85, leaf_t0=0.1),
        ]),
    "oak": Species(
        "oak", height=(17, 24), crown_base=(0.25, 0), trunk_r=0.026, trunk_sides=12, trunk_top=0.55,
        envelope=env_broad, level1_len=0.44, leaf="broad", leaf_size=(0.22, 0.14), leaf_up=0.45, leaf_count=50000, leaf_jitter=0.25,
        trunk_wobble=0.06, lod1_leaf_scale=1.9, lean=0.05,
        levels=[
            Level(6, t=(0.5, 1.0), angle=(60, 10), radius=0.55, segs=9, sides=7, tropism=-0.01, wobble=0.16),
            Level(1.8, per_m=True, t=(0.15, 1.0), angle=(55, 15), length=(0.45, 0.12), radius=0.5, segs=5,
                  sides=5, tropism=0.01, wobble=0.25, leaves=30, leaf_t0=0.4),
            Level(3.0, per_m=True, t=(0.15, 1.0), angle=(55, 15), length=(0.45, 0.12), radius=0.5, segs=3,
                  sides=3, tropism=0.0, wobble=0.25, leaves=85, leaf_t0=0.1),
        ]),
    "birch": Species(
        "birch", height=(15, 21), crown_base=(0.35, 0), trunk_r=0.012, trunk_sides=9, trunk_top=0.95,
        envelope=env_ovoid, level1_len=0.25, leaf="broad", leaf_size=(0.14, 0.10), leaf_up=0.3, leaf_count=32000,
        trunk_wobble=0.04, lod1_leaf_scale=1.9, lean=0.07,
        levels=[
            Level(1.8, per_m=True, t=(0.35, 0.98), angle=(45, 10), radius=0.35, segs=6, sides=5, tropism=-0.01, wobble=0.08),
            Level(2.4, per_m=True, t=(0.2, 1.0), angle=(55, 12), length=(0.5, 0.12), radius=0.5, segs=4,
                  sides=3, tropism=-0.15, wobble=0.1, leaves=70, leaf_t0=0.15),
        ]),
    "maple": Species(
        "maple", height=(13, 19), crown_base=(0.3, 0), trunk_r=0.022, trunk_sides=10, trunk_top=0.62,
        envelope=env_ovoid, level1_len=0.38, leaf="broad", leaf_size=(0.26, 0.22), leaf_up=0.6, leaf_count=34000, leaf_jitter=0.25,
        trunk_wobble=0.03, lod1_leaf_scale=1.8,
        levels=[
            Level(6, t=(0.45, 1.0), angle=(38, 10), radius=0.55, segs=7, sides=6, tropism=0.03, wobble=0.1),
            Level(2.0, per_m=True, t=(0.15, 1.0), angle=(50, 12), length=(0.45, 0.1), radius=0.5, segs=4,
                  sides=4, tropism=0.02, wobble=0.12, leaves=30, leaf_t0=0.4),
            Level(3.0, per_m=True, t=(0.15, 1.0), angle=(50, 14), length=(0.45, 0.12), radius=0.5, segs=3,
                  sides=3, tropism=-0.01, wobble=0.15, leaves=60, leaf_t0=0.1),
        ]),
    "shrub": Species(
        "shrub", height=(2.2, 4.0), crown_base=(0.12, 0), trunk_r=0.012, trunk_sides=5, trunk_top=0.9,
        envelope=env_ovoid, level1_len=0.4, leaf="broad", leaf_size=(0.14, 0.10), leaf_up=0.4, leaf_count=9000,
        stems=7, trunk_wobble=0.12, lod1_leaf_scale=1.6, lod1_keep=0.5,
        levels=[
            Level(6, t=(0.3, 1.0), angle=(40, 12), radius=0.6, segs=4, sides=3, tropism=0.02, wobble=0.18),
            Level(4.0, per_m=True, t=(0.1, 1.0), angle=(45, 12), length=(0.5, 0.1), radius=0.5, segs=2,
                  sides=3, tropism=0.0, wobble=0.2, leaves=120, leaf_t0=0.0),
        ]),
    "sapling": Species(
        "sapling", height=(3.0, 6.0), crown_base=(0.3, 0), trunk_r=0.012, trunk_sides=6, trunk_top=0.9,
        envelope=env_ovoid, level1_len=0.3, leaf="broad", leaf_size=(0.14, 0.10), leaf_up=0.5, leaf_count=2500,
        trunk_wobble=0.05, lod1_leaf_scale=1.6, lod1_keep=0.5,
        levels=[
            Level(3.5, per_m=True, t=(0.3, 1.0), angle=(45, 10), radius=0.45, segs=4, sides=4, tropism=0.02, wobble=0.12),
            Level(4.0, per_m=True, t=(0.1, 1.0), angle=(45, 12), length=(0.5, 0.1), radius=0.5, segs=2,
                  sides=3, tropism=0.0, wobble=0.15, leaves=110, leaf_t0=0.0),
        ]),
}


def make_tree(species: str, rng: np.random.Generator, lod: int = 0) -> MeshBuilder:
    g = Grower(SPECIES[species], rng, lod).grow()
    return g.bark

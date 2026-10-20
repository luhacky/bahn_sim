"""Ground vegetation: meadow grass tufts, tall grass with seed heads, wildflower
clumps (yarrow, buttercup, knapweed, chicory), bracken fern.

All return a MeshBuilder with material slots
    0 grass blades (attribute leaf_rnd: 0 fresh .. 1 dry straw), 1 flower heads (leaf_rnd picks the colour)
"""
from __future__ import annotations

import math

import numpy as np

from .geom import MeshBuilder


def _blades(mb, rng, n, h_range, spread, width, lean_range, dry_share=0.12, segs=3, mat=0, rnd_bias=0.0):
    """n curved, tapered blades from a clump of radius `spread`."""
    ang = rng.uniform(0, 2 * np.pi, n)
    r = spread * np.sqrt(rng.random(n))
    base = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros(n)], axis=1)
    h = rng.uniform(*h_range, n)
    lean = rng.uniform(*lean_range, n)
    face = ang + rng.normal(0, 0.6, n)                       # lean outwards, roughly
    d = np.stack([np.cos(face), np.sin(face), np.zeros(n)], axis=1)
    side = np.stack([-np.sin(face), np.cos(face), np.zeros(n)], axis=1)
    w = width * rng.uniform(0.7, 1.3, n)
    V, F = [], []
    t = np.linspace(0, 1, segs + 1)
    for k in range(segs + 1):
        z = h * t[k]
        off = lean * h * t[k] ** 2                          # parabolic bend
        c = base + d * off[:, None] + np.array([0, 0, 1.0]) * (z - 0.25 * off * t[k])[:, None]
        if k < segs:
            ww = w * (1 - 0.75 * t[k])
            V.append(c - side * ww[:, None] * 0.5)
            V.append(c + side * ww[:, None] * 0.5)
        else:
            V.append(c)
    V = np.stack(V, axis=1)                                  # (n, 2*segs+1, 3)
    per = 2 * segs + 1
    faces = []
    for k in range(segs - 1):
        a = 2 * k
        faces.append([a, a + 1, a + 3, a + 2])
    a = 2 * (segs - 1)
    tri = [a, a + 1, 2 * segs]
    base_idx = np.arange(n)[:, None] * per
    Q = (base_idx[:, None, :] + np.array(faces)[None]).reshape(-1, 4) if faces else np.zeros((0, 4), int)
    T = base_idx + np.array(tri)
    uv = np.tile(np.concatenate([np.stack([np.tile([0, 1], segs), np.repeat(t[:-1], 2)], 1), [[0.5, 1.0]]]), (n, 1))
    rnd = np.clip(rng.random(n) * 0.8 + rnd_bias, 0, 0.84)
    dry = rng.random(n) < dry_share
    rnd[dry] = rng.uniform(0.85, 1.0, int(dry.sum()))
    mb.add(V.reshape(-1, 3), Q.tolist() + T.tolist(), mat, uv=uv, smooth=True, leaf_rnd=np.repeat(rnd, per))
    tips = base + d * lean[:, None] * h[:, None] + np.array([0, 0, 1.0]) * (h - 0.25 * lean * h)[:, None]
    return tips


def grass_tuft(rng, lod=0, tall=False):
    mb = MeshBuilder(attrs=("leaf_rnd",))
    n = int(rng.integers(34, 48)) if lod == 0 else int(rng.integers(14, 20))
    if tall:
        tips = _blades(mb, rng, n, (0.35, 0.75), 0.22, 0.012, (0.15, 0.5), dry_share=0.2, rnd_bias=0.15)
        # a few seed-heading stems (Rispen) with small spikelets
        m = int(rng.integers(4, 9))
        stips = _blades(mb, rng, m, (0.7, 1.05), 0.15, 0.004, (0.05, 0.2), dry_share=0.5, segs=3, rnd_bias=0.4)
        for tp in stips:
            k = 6
            c = tp + rng.normal(0, 0.03, (k, 3)) - np.array([0, 0, 0.06]) * rng.random((k, 1))
            V = np.repeat(c, 3, axis=0) + rng.normal(0, 0.012, (k * 3, 3))
            mb.add(V, np.arange(k * 3).reshape(-1, 3), 0, leaf_rnd=0.95)
    else:
        _blades(mb, rng, n, (0.12, 0.42), 0.25, 0.014, (0.2, 0.7), dry_share=0.1)
    return mb


FLOWER_KINDS = {  # leaf_rnd value -> colour picked in the flower material
    "yarrow": 0.1,      # Schafgarbe, white umbels
    "buttercup": 0.35,  # Hahnenfuß, yellow
    "knapweed": 0.6,    # Flockenblume, purple
    "chicory": 0.85,    # Wegwarte, light blue
}


def flower_clump(rng, kind=None, lod=0):
    kind = kind or list(FLOWER_KINDS)[rng.integers(len(FLOWER_KINDS))]
    mb = MeshBuilder(attrs=("leaf_rnd",))
    _blades(mb, rng, 22 if lod == 0 else 10, (0.10, 0.35), 0.2, 0.014, (0.2, 0.6))
    m = int(rng.integers(5, 12))
    tips = _blades(mb, rng, m, (0.35, 0.75), 0.15, 0.006, (0.0, 0.15), dry_share=0.0)
    col = FLOWER_KINDS[kind]
    for tp in tips:
        if kind == "yarrow":           # flat umbel of small florets
            k = 9
            a = rng.uniform(0, 2 * np.pi, k); r = 0.035 * np.sqrt(rng.random(k))
            c = tp + np.stack([r * np.cos(a), r * np.sin(a), rng.normal(0, 0.004, k)], 1)
            s = 0.012
        else:                          # single flower head: small star of petals
            k = 5
            a = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(0, 1)
            r = 0.018 if kind != "knapweed" else 0.012
            c = tp + np.stack([r * np.cos(a), r * np.sin(a), np.full(k, 0.004)], 1)
            s = 0.016 if kind != "knapweed" else 0.012
        V = []
        for p in c:
            V += [tp if kind != "yarrow" else p + [0, 0, -0.002], p + [s, 0, 0.003], p + [0, s, 0.003]]
        V = np.array(V)
        mb.add(V, np.arange(len(V)).reshape(-1, 3), 1, leaf_rnd=col)
    return mb


def fern(rng, lod=0):
    """Bracken (Adlerfarn): 5-9 arching fronds with pinnae, ~0.8 m."""
    mb = MeshBuilder(attrs=("leaf_rnd",))
    nf = int(rng.integers(5, 10)) if lod == 0 else 4
    for i in range(nf):
        a = rng.uniform(0, 2 * np.pi)
        L = rng.uniform(0.6, 1.1)
        d = np.array([math.cos(a), math.sin(a), 0.0])
        side = np.array([-math.sin(a), math.cos(a), 0.0])
        n = 10
        t = np.linspace(0, 1, n)
        spine = d[None] * (L * 0.8 * t)[:, None] + np.array([0, 0, 1.0])[None] * (L * (0.9 * t - 0.75 * t ** 2))[:, None]
        rnd = rng.uniform(0.0, 0.35)
        for k in range(1, n):
            w = 0.22 * L * math.sin(math.pi * min(1, t[k] * 1.1)) * (1 - 0.5 * t[k])
            p = spine[k]
            for sgn in (1, -1):
                q = p + side * sgn * w + np.array([0, 0, -0.04 * w])
                mid = p + (q - p) * 0.5
                V = np.array([p, mid + (spine[k] - spine[k - 1]) * 0.35, q, mid - (spine[k] - spine[k - 1]) * 0.35])
                mb.add(V, [[0, 1, 2], [0, 2, 3]], 0, leaf_rnd=rnd)
    return mb

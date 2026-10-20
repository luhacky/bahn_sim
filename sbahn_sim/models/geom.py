"""Mesh building helpers for the procedural model library.

MeshBuilder collects triangles and quads (numpy arrays) with a material slot,
per-vertex UVs and optional per-vertex float attributes, then writes one
Blender mesh in a single foreach_set pass. Everything is numpy; no bpy.ops.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import bpy
import numpy as np


class MeshBuilder:
    def __init__(self, attrs: Sequence[str] = ()):
        self.V, self.UV, self.T, self.Q, self.TM, self.QM, self.TS, self.QS = [], [], [], [], [], [], [], []
        self.A: Dict[str, list] = {a: [] for a in attrs}
        self.n = 0

    # ------------------------------------------------------------------
    def add(self, v, faces, mat: int = 0, uv=None, smooth: bool = False, **attrs) -> int:
        """Add vertices v (N,3) and faces (M,3) or (M,4) (indices local to v), or a
        list mixing both. Returns the vertex offset."""
        v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        off = self.n
        self.V.append(v)
        self.UV.append(np.zeros((len(v), 2)) if uv is None else np.asarray(uv, dtype=np.float64).reshape(-1, 2))
        for a, store in self.A.items():
            val = attrs.get(a, 0.0)
            store.append(np.broadcast_to(np.asarray(val, dtype=np.float64), (len(v),)).copy())
        if isinstance(faces, np.ndarray) and faces.ndim == 2:
            groups = [faces]
        else:
            faces = list(faces)
            groups = [np.array([f for f in faces if len(f) == 3], dtype=np.int64).reshape(-1, 3),
                      np.array([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)]
        for g in groups:
            if len(g) == 0:
                continue
            g = np.asarray(g, dtype=np.int64) + off
            if g.shape[1] == 3:
                self.T.append(g); self.TM.append(np.full(len(g), mat)); self.TS.append(np.full(len(g), smooth))
            elif g.shape[1] == 4:
                self.Q.append(g); self.QM.append(np.full(len(g), mat)); self.QS.append(np.full(len(g), smooth))
            else:
                raise ValueError("only tris and quads")
        self.n += len(v)
        return off

    def add_parts(self, parts, smooth=False):
        """Accept the [(verts, faces, mat), ...] lists used by util.merge / obstacles."""
        for p in parts:
            self.add(p[0], p[1], p[2] if len(p) > 2 else 0, smooth=smooth)

    def extend(self, other: "MeshBuilder", mat_offset: int = 0):
        off = self.n
        for v, uv in zip(other.V, other.UV):
            self.V.append(v); self.UV.append(uv)
        for a in self.A:
            self.A[a] += other.A.get(a, [np.zeros(len(v)) for v in other.V])
        for T, M, S in zip(other.T, other.TM, other.TS):
            self.T.append(T + off); self.TM.append(M + mat_offset); self.TS.append(S)
        for Q, M, S in zip(other.Q, other.QM, other.QS):
            self.Q.append(Q + off); self.QM.append(M + mat_offset); self.QS.append(S)
        self.n += other.n

    def transform(self, m4):
        m4 = np.asarray(m4, dtype=np.float64)
        self.V = [(v @ m4[:3, :3].T) + m4[:3, 3] for v in self.V]
        return self

    @property
    def n_faces(self):
        return sum(len(t) for t in self.T) + sum(len(q) for q in self.Q)

    # ------------------------------------------------------------------
    def to_mesh(self, name: str, materials: Sequence = ()) -> bpy.types.Mesh:
        V = np.concatenate(self.V) if self.V else np.zeros((0, 3))
        T = np.concatenate(self.T) if self.T else np.zeros((0, 3), np.int64)
        Q = np.concatenate(self.Q) if self.Q else np.zeros((0, 4), np.int64)
        TM = np.concatenate(self.TM) if self.TM else np.zeros(0, np.int64)
        QM = np.concatenate(self.QM) if self.QM else np.zeros(0, np.int64)
        TS = np.concatenate(self.TS) if self.TS else np.zeros(0, bool)
        QS = np.concatenate(self.QS) if self.QS else np.zeros(0, bool)
        me = bpy.data.meshes.new(name)
        me.vertices.add(len(V))
        me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        loops = np.concatenate([T.ravel(), Q.ravel()]).astype(np.int32)
        starts = np.concatenate([np.arange(len(T)) * 3, len(T) * 3 + np.arange(len(Q)) * 4]).astype(np.int32)
        me.loops.add(len(loops))
        me.polygons.add(len(T) + len(Q))
        me.loops.foreach_set("vertex_index", loops)
        me.polygons.foreach_set("loop_start", starts)
        try:
            me.polygons.foreach_set("loop_total", np.concatenate([np.full(len(T), 3), np.full(len(Q), 4)]).astype(np.int32))
        except Exception:
            pass
        me.update(calc_edges=True)
        for m in materials:
            me.materials.append(m)
        me.polygons.foreach_set("material_index", np.concatenate([TM, QM]).astype(np.int32))
        me.polygons.foreach_set("use_smooth", np.concatenate([TS, QS]).astype(bool))
        UV = np.concatenate(self.UV) if self.UV else np.zeros((0, 2))
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", UV[loops].astype(np.float32).ravel())
        for a, store in self.A.items():
            if not store:
                continue
            at = me.attributes.new(a, "FLOAT", "POINT")
            at.data.foreach_set("value", np.concatenate(store).astype(np.float32))
        me.validate(clean_customdata=False)
        me.update()
        return me

    def to_object(self, name: str, collection, materials: Sequence = ()) -> bpy.types.Object:
        ob = bpy.data.objects.new(name, self.to_mesh(name, materials))
        collection.objects.link(ob)
        return ob


# ----------------------------------------------------------------------
# primitives: return (verts, faces, uv)
# ----------------------------------------------------------------------
def _perp(d):
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, ref)
    return u / np.linalg.norm(u)


def tube(points, radii, segments=8, cap_start=False, uv_scale=(1.0, 1.0), twist=0.0):
    """Tapered tube along a polyline with parallel-transport frames.
    UV: u = angle (0..uv_scale[0] around), v = arc length * uv_scale[1]."""
    P = np.asarray(points, dtype=np.float64)
    r = np.broadcast_to(np.asarray(radii, dtype=np.float64), (len(P),))
    n = len(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    U = np.zeros_like(P)
    U[0] = _perp(T[0])
    for i in range(1, n):
        u = U[i - 1] - np.dot(U[i - 1], T[i]) * T[i]
        ln = np.linalg.norm(u)
        U[i] = u / ln if ln > 1e-9 else _perp(T[i])
    W = np.cross(T, U)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)[None, :] + twist
    ca, sa = np.cos(ang), np.sin(ang)
    V = P[:, None, :] + r[:, None, None] * (ca[..., None] * U[:, None, :] + sa[..., None] * W[:, None, :])
    V = V.reshape(-1, 3)
    i = np.arange(n - 1)[:, None] * segments
    j = np.arange(segments)[None, :]
    jn = (j + 1) % segments
    F = np.stack([i + j, i + jn, i + jn + segments, i + j + segments], axis=-1).reshape(-1, 4)
    seglen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    UVs = np.stack([np.broadcast_to(np.arange(segments)[None, :] / segments * uv_scale[0], (n, segments)),
                    np.broadcast_to(seglen[:, None] * uv_scale[1], (n, segments))], axis=-1).reshape(-1, 2)
    if cap_start:
        c = len(V)
        V = np.vstack([V, P[0]])
        UVs = np.vstack([UVs, [0, 0]])
        capf = np.stack([np.full(segments, c), (np.arange(segments) + 1) % segments, np.arange(segments)], axis=1)
        return V, [*F.tolist(), *capf.tolist()], UVs
    return V, F, UVs


def box(size, center=(0, 0, 0)):
    """Box with per-face vertices (24 verts) so UVs are in metres on every face."""
    sx, sy, sz = (s / 2.0 for s in size)
    c = np.asarray(center, dtype=np.float64)
    faces = [  # (normal axis, sign) -> 4 corners
        [(-sx, -sy, -sz), (-sx, sy, -sz), (sx, sy, -sz), (sx, -sy, -sz)],
        [(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)],
        [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz)],
        [(sx, sy, -sz), (-sx, sy, -sz), (-sx, sy, sz), (sx, sy, sz)],
        [(-sx, sy, -sz), (-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz)],
        [(sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz), (sx, -sy, sz)],
    ]
    V = np.array(faces, dtype=np.float64).reshape(-1, 3)
    UV = []
    for k, f in enumerate(faces):
        f = np.array(f)
        if k < 2:
            UV += list(f[:, [0, 1]])
        elif k < 4:
            UV += list(f[:, [0, 2]])
        else:
            UV += list(f[:, [1, 2]])
    F = np.arange(24).reshape(6, 4)
    return V + c, F, np.array(UV)


def quad(p0, p1, p2, p3):
    """Single quad with planar metre UVs."""
    P = np.array([p0, p1, p2, p3], dtype=np.float64)
    e1 = P[1] - P[0]
    e2 = P[3] - P[0]
    l1 = np.linalg.norm(e1)
    e1n = e1 / max(l1, 1e-9)
    e2p = e2 - np.dot(e2, e1n) * e1n
    e2n = e2p / max(np.linalg.norm(e2p), 1e-9)
    uv = np.stack([(P - P[0]) @ e1n, (P - P[0]) @ e2n], axis=1)
    return P, np.array([[0, 1, 2, 3]]), uv


def prism_x(profile_yz, x0, x1):
    """Extrude a closed convex-ish (y, z) polygon along X between x0 and x1 (caps triangulated as a fan)."""
    prof = np.asarray(profile_yz, dtype=np.float64)
    k = len(prof)
    a = np.column_stack([np.full(k, x0), prof])
    b = np.column_stack([np.full(k, x1), prof])
    V = np.vstack([a, b])
    i = np.arange(k); j = (i + 1) % k
    F = np.stack([i, j, j + k, i + k], axis=1).tolist()
    for t in range(1, k - 1):
        F.append([0, t + 1, t])
        F.append([k, k + t, k + t + 1])
    uv = np.column_stack([V[:, 0], V[:, 2] + V[:, 1]])
    return V, F, uv


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def mat4(R=None, t=(0, 0, 0)):
    m = np.eye(4)
    if R is not None:
        m[:3, :3] = R
    m[:3, 3] = t
    return m


def rotate_about(v, axis, angle):
    """Rodrigues rotation of vectors v (N,3) about unit axis (3,)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    v = np.asarray(v, dtype=np.float64)
    return v * c + np.cross(axis, v) * s + np.outer(v @ axis, axis) * (1 - c)

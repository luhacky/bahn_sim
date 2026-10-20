"""Blender helpers shared by the builder modules (bpy required)."""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT_COLLECTION = "SBahnLoop"


# --------------------------------------------------------------------------
# Collections / objects
# --------------------------------------------------------------------------
def ensure_collection(name: str, parent: Optional[bpy.types.Collection] = None) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    parent = parent or root_collection() if name != ROOT_COLLECTION else bpy.context.scene.collection
    if col.name not in parent.children:
        try:
            parent.children.link(col)
        except RuntimeError:
            pass
    return col


def root_collection() -> bpy.types.Collection:
    col = bpy.data.collections.get(ROOT_COLLECTION)
    if col is None:
        col = bpy.data.collections.new(ROOT_COLLECTION)
        bpy.context.scene.collection.children.link(col)
    return col


def hide_collection_instances_source(col: bpy.types.Collection) -> None:
    """Instance-source collections are excluded from render/viewport."""
    for vl in bpy.context.scene.view_layers:
        lc = _find_layer_collection(vl.layer_collection, col.name)
        if lc:
            lc.exclude = True


def _find_layer_collection(lc, name):
    if lc.name == name:
        return lc
    for c in lc.children:
        r = _find_layer_collection(c, name)
        if r:
            return r
    return None


def link(ob: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    if ob.name not in col.objects:
        col.objects.link(ob)
    return ob


def new_mesh_object(name: str, verts, faces=None, collection=None, materials: Sequence = (),
                    material_index=None, smooth: bool = False) -> bpy.types.Object:
    """Fast mesh creation from numpy arrays.

    verts: (N,3) float; faces: (M,k) int array (all same size) or list of lists.
    material_index: optional (M,) int array.
    """
    me = bpy.data.meshes.new(name)
    verts = np.ascontiguousarray(np.asarray(verts, dtype=np.float32)).reshape(-1, 3)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    if faces is not None and len(faces):
        if isinstance(faces, np.ndarray) and faces.ndim == 2:
            k = faces.shape[1]
            m = len(faces)
            me.loops.add(m * k)
            me.polygons.add(m)
            me.loops.foreach_set("vertex_index", np.ascontiguousarray(faces, dtype=np.int32).ravel())
            me.polygons.foreach_set("loop_start", (np.arange(m) * k).astype(np.int32))
            try:
                me.polygons.foreach_set("loop_total", np.full(m, k, dtype=np.int32))
            except Exception:
                pass
        else:
            loops = [int(i) for f in faces for i in f]
            starts, tot, acc = [], [], 0
            for f in faces:
                starts.append(acc); tot.append(len(f)); acc += len(f)
            me.loops.add(len(loops))
            me.polygons.add(len(faces))
            me.loops.foreach_set("vertex_index", loops)
            me.polygons.foreach_set("loop_start", starts)
            try:
                me.polygons.foreach_set("loop_total", tot)
            except Exception:
                pass
    me.update(calc_edges=True)
    me.validate()
    for mat in materials:
        me.materials.append(mat)
    if material_index is not None and len(me.polygons):
        me.polygons.foreach_set("material_index", np.asarray(material_index, dtype=np.int32))
    if smooth and len(me.polygons):
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    ob = bpy.data.objects.new(name, me)
    (collection or root_collection()).objects.link(ob)
    return ob


def set_point_attr(me: bpy.types.Mesh, name: str, data_type: str, values) -> None:
    attr = me.attributes.get(name) or me.attributes.new(name, data_type, "POINT")
    values = np.asarray(values)
    if data_type == "FLOAT_VECTOR":
        attr.data.foreach_set("vector", np.ascontiguousarray(values, dtype=np.float32).ravel())
    elif data_type == "FLOAT":
        attr.data.foreach_set("value", np.ascontiguousarray(values, dtype=np.float32).ravel())
    elif data_type == "INT":
        attr.data.foreach_set("value", np.ascontiguousarray(values, dtype=np.int32).ravel())
    elif data_type == "BOOLEAN":
        attr.data.foreach_set("value", np.ascontiguousarray(values, dtype=bool).ravel())
    else:
        raise ValueError(data_type)


def join_meshes(name: str, objects: List[bpy.types.Object], collection=None) -> bpy.types.Object:
    """Join mesh objects (world-space) into one object, removing the sources."""
    if not objects:
        return None
    bm_verts, bm_faces, mats, midx = [], [], [], []
    off = 0
    for ob in objects:
        me = ob.data
        mw = np.array(ob.matrix_world)
        v = np.ones((len(me.vertices), 4), dtype=np.float64)
        me.vertices.foreach_get("co", v[:, :3].ravel())
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        v[:, :3] = co.reshape(-1, 3)
        v = (mw @ v.T).T[:, :3]
        bm_verts.append(v)
        for p in me.polygons:
            bm_faces.append([off + i for i in p.vertices])
            m = me.materials[p.material_index] if me.materials else None
            if m not in mats:
                mats.append(m)
            midx.append(mats.index(m))
        off += len(v)
    ob = new_mesh_object(name, np.concatenate(bm_verts), bm_faces, collection, [m for m in mats if m], midx)
    for o in objects:
        bpy.data.objects.remove(o, do_unlink=True)
    return ob


# --------------------------------------------------------------------------
# Primitive builders (numpy, no bpy.ops) -- all return (verts, faces) arrays
# --------------------------------------------------------------------------
def box(size, center=(0, 0, 0)):
    sx, sy, sz = (s / 2.0 for s in size)
    cx, cy, cz = center
    v = np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                  [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]]) + np.array(center)
    f = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    return v, f


def cylinder(p0, p1, radius, segments=8, radius1=None):
    """Cylinder (or cone) between two points."""
    p0 = np.asarray(p0, dtype=np.float64); p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    L = np.linalg.norm(axis)
    if L < 1e-9:
        axis = np.array([0, 0, 1.0]); L = 1e-6
    axis = axis / L
    ref = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    r1 = radius if radius1 is None else radius1
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring0 = p0 + radius * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w))
    ring1 = p1 + r1 * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w))
    v = np.concatenate([ring0, ring1])
    i = np.arange(segments); j = (i + 1) % segments
    f = np.stack([i, j, j + segments, i + segments], axis=1)
    return v, f


def merge(parts):
    """Merge [(verts, faces, mat_index), ...] into arrays (faces padded to quads/tris lists)."""
    V, F, M = [], [], []
    off = 0
    for part in parts:
        v, f = part[0], part[1]
        mi = part[2] if len(part) > 2 else 0
        V.append(np.asarray(v, dtype=np.float64))
        for face in f:
            F.append([int(i) + off for i in face])
            M.append(mi)
        off += len(v)
    return np.concatenate(V), F, M


def extrude_profile(profile_xy, path_pts, right, up, closed_path=False, close_profile=True):
    """Sweep a 2-D profile (list of (lateral, height)) along path points with
    per-point right/up frames. Returns (verts, quad faces ndarray)."""
    prof = np.asarray(profile_xy, dtype=np.float64)
    P = np.asarray(path_pts); R = np.asarray(right); U = np.asarray(up)
    n, k = len(P), len(prof)
    verts = P[:, None, :] + prof[None, :, 0, None] * R[:, None, :] + prof[None, :, 1, None] * U[:, None, :]
    verts = verts.reshape(-1, 3)
    ii = np.arange(n if closed_path else n - 1)
    jj = np.arange(k if close_profile else k - 1)
    i0 = (ii[:, None] * k + jj[None, :])
    i1 = (((ii + 1) % n)[:, None] * k + jj[None, :])
    j1 = (jj + 1) % k
    f = np.stack([i0, i1, ((ii + 1) % n)[:, None] * k + j1[None, :], ii[:, None] * k + j1[None, :]], axis=-1)
    return verts, f.reshape(-1, 4)


# --------------------------------------------------------------------------
# Geometry Nodes helpers
# --------------------------------------------------------------------------
def gn_tree(name: str):
    tree = bpy.data.node_groups.get(name)
    if tree:
        return tree, tree.nodes["Group Input"], tree.nodes["Group Output"]
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = tree.nodes.new("NodeGroupInput"); gi.name = "Group Input"; gi.location = (-600, 0)
    go = tree.nodes.new("NodeGroupOutput"); go.name = "Group Output"; go.location = (600, 0)
    return tree, gi, go


def node(tree, bl_idname, location=(0, 0), **props):
    n = tree.nodes.new(bl_idname)
    n.location = location
    for k, v in props.items():
        setattr(n, k, v)
    return n


def named_attr(tree, name, data_type="FLOAT", location=(0, 0)):
    n = node(tree, "GeometryNodeInputNamedAttribute", location, data_type=data_type)
    n.inputs["Name"].default_value = name
    return n


def gn_link(tree, a, out_name, b, in_name):
    o = a.outputs[out_name] if isinstance(out_name, str) else a.outputs[out_name]
    i = b.inputs[in_name] if isinstance(in_name, str) else b.inputs[in_name]
    tree.links.new(o, i)


def instancer(name: str, points: np.ndarray, rotations: np.ndarray, source_col: bpy.types.Collection,
              scales: Optional[np.ndarray] = None, indices: Optional[np.ndarray] = None,
              collection=None) -> bpy.types.Object:
    """Point-cloud mesh + Geometry Nodes 'Instance on Points' from a collection.

    rotations: (N,3) XYZ Euler in radians. indices pick the child object of
    source_col (modulo child count) when given, else child 0.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    ob = new_mesh_object(name, points, None, collection)
    me = ob.data
    set_point_attr(me, "rot", "FLOAT_VECTOR", np.asarray(rotations, dtype=np.float64).reshape(-1, 3))
    set_point_attr(me, "scale", "FLOAT_VECTOR", np.ones((n, 3)) if scales is None else np.asarray(scales).reshape(-1, 3))
    set_point_attr(me, "idx", "INT", np.zeros(n, dtype=np.int32) if indices is None else np.asarray(indices, dtype=np.int32))
    tree, gi, go = gn_tree(f"GN_{name}")
    iop = node(tree, "GeometryNodeInstanceOnPoints", (200, 0))
    ci = node(tree, "GeometryNodeCollectionInfo", (-200, -200), transform_space="ORIGINAL")
    ci.inputs["Collection"].default_value = source_col
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    rot = named_attr(tree, "rot", "FLOAT_VECTOR", (-200, 100))
    scl = named_attr(tree, "scale", "FLOAT_VECTOR", (-200, -50))
    idx = named_attr(tree, "idx", "INT", (-200, -350))
    iop.inputs["Pick Instance"].default_value = True
    gn_link(tree, gi, "Geometry", iop, "Points")
    gn_link(tree, ci, "Instances", iop, "Instance")
    gn_link(tree, rot, "Attribute", iop, "Rotation")
    gn_link(tree, scl, "Attribute", iop, "Scale")
    gn_link(tree, idx, "Attribute", iop, "Instance Index")
    gn_link(tree, iop, "Instances", go, "Geometry")
    mod = ob.modifiers.new("Instancer", "NODES")
    mod.node_group = tree
    ob["sbahn_instancer_source"] = source_col.name
    return ob


def frame_euler(right: np.ndarray, tangent: np.ndarray, up: np.ndarray) -> np.ndarray:
    """XYZ Euler angles for frames with local X=right, Y=tangent, Z=up. Vectorised."""
    R = np.stack([right, tangent, up], axis=-1)   # columns are local axes -> rotation matrix
    # ZYX decomposition equivalent to Blender's XYZ euler (apply X, then Y, then Z)
    sy = -R[:, 2, 0]
    sy = np.clip(sy, -1, 1)
    y = np.arcsin(sy)
    x = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    z = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.stack([x, y, z], axis=1)


# --------------------------------------------------------------------------
# Curves (wires) and text
# --------------------------------------------------------------------------
def polyline_curve(name: str, points: np.ndarray, radius: float, material=None, cyclic=False,
                   collection=None, resolution=1) -> bpy.types.Object:
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    cu.resolution_u = resolution
    cu.use_fill_caps = True
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    sp.points.foreach_set("co", np.concatenate([pts, np.ones((len(pts), 1))], axis=1).ravel().tolist())
    sp.use_cyclic_u = cyclic
    if material:
        cu.materials.append(material)
    ob = bpy.data.objects.new(name, cu)
    (collection or root_collection()).objects.link(ob)
    return ob


def multi_polyline_curve(name: str, polylines: Iterable[np.ndarray], radius: float, material=None,
                         collection=None) -> bpy.types.Object:
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    cu.resolution_u = 1
    cu.use_fill_caps = True
    for pts in polylines:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if len(pts) < 2:
            continue
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        sp.points.foreach_set("co", np.concatenate([pts, np.ones((len(pts), 1))], axis=1).ravel().tolist())
    if material:
        cu.materials.append(material)
    ob = bpy.data.objects.new(name, cu)
    (collection or root_collection()).objects.link(ob)
    return ob


def text_object(name: str, text: str, size: float, material=None, collection=None, extrude=0.0,
                align="CENTER") -> bpy.types.Object:
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = text
    cu.size = size
    cu.extrude = extrude
    cu.align_x = align
    cu.align_y = "CENTER"
    if material:
        cu.materials.append(material)
    ob = bpy.data.objects.new(name, cu)
    (collection or root_collection()).objects.link(ob)
    return ob


def set_matrix(ob: bpy.types.Object, m) -> None:
    ob.matrix_world = Matrix(np.asarray(m).tolist()) if not isinstance(m, Matrix) else m


def deg(x):
    return math.radians(x)

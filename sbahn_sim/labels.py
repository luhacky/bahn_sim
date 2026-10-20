"""Ground truth for one cab-camera frame: 2-D boxes, visibility, distances.

    from sbahn_sim import labels
    lab = labels.Labeller(scene, tp)
    objs = lab.annotate()                  # list of dicts, one per active obstacle
    labels.yolo_lines(objs, W, H)          # "cls cx cy w h" rows for the visible ones

Boxes are computed geometrically, independent of render engine and weather:

* bbox_amodal: projection of every mesh vertex, clipped to the image
  (the full object extent, as if nothing were in front of it).
* bbox_visible: from a grid of camera rays over the amodal box (up to
  `grid` x `grid`, i.e. pixel-exact for objects smaller than that). A ray
  counts as an object pixel when it hits the obstacle's own BVH, and as a
  visible pixel when the first opaque thing the scene ray hits is the
  obstacle. Objects flagged obj["sbahn_non_occluding"] (fog domains, rain,
  windscreen glass) or inside a "Weather" collection are skipped by the
  scene ray, so fog never removes a box: fog density is logged as a
  condition and the benchmark decides what should be detectable.
* visible_fraction = visible pixels / object pixels (occlusion by the cab
  frame, masts, the car in front, etc.), truncation = share of the projected
  box outside the image.

Parked cars baked into the house plots of the detailed scenery (objects with
the property sbahn_background_vehicles) are split out of the house meshes and
labelled as class car with background=True, so a detector that finds them is
not counted as a false positive.

Distances: along-track from the train front and from the camera eye (the
arc-length difference, what a braking curve needs), straight-line from the
camera, and camera-space depth.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import obstacles

NON_OCCLUDING_COLLECTIONS = ("Weather",)
RUNNING_TRACK_HALF_WIDTH = 1.6      # |lateral - running_track| below this = obstacle fouls the running track


def _non_occluding_names() -> set:
    names = set()
    for cname in NON_OCCLUDING_COLLECTIONS:
        col = bpy.data.collections.get(cname)
        if col:
            names.update(o.name for o in col.all_objects)
    for o in bpy.data.objects:
        if o.get("sbahn_non_occluding"):
            names.add(o.name)
    return names


DYNAMIC_COLLECTIONS = ("Train", "Obstacles")
DYNAMIC_PARENTS = ("CabCamera", "Car_")      # anything parented to these moves with the train (wiper, ...)


def _dynamic_objects() -> list:
    """Objects that move per frame: train, obstacles, and everything parented to the camera or a car."""
    out = {}
    for cname in DYNAMIC_COLLECTIONS:
        col = bpy.data.collections.get(cname)
        if col:
            out.update((o.name, o) for o in col.all_objects)
    for o in bpy.data.objects:
        p = o.parent
        while p is not None:
            if p.name.startswith(DYNAMIC_PARENTS):
                out[o.name] = o
                break
            p = p.parent
    return list(out.values())


def _mesh_tris(me) -> np.ndarray:
    """(T, 3, 3) float32 local-space triangles of a mesh."""
    try:
        me.calc_loop_triangles()
    except Exception:
        pass
    nt = len(me.loop_triangles)
    if nt == 0:
        return np.zeros((0, 3, 3), np.float32)
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    tri = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    return co.reshape(-1, 3)[tri.reshape(-1, 3)]


VEHICLE_MATERIAL_TAGS = ("car_paint", "car_glass", "tyre", "car_trim", "truck_box", "headlamp", "taillamp",
                         "number_plate", "rim_alloy")


def _tri_materials(me) -> np.ndarray:
    nt = len(me.loop_triangles)
    mi = np.zeros(nt, np.int32)
    if nt:
        me.loop_triangles.foreach_get("material_index", mi)
    return mi


def _split_vehicles(me, tris: np.ndarray):
    """House plots with parked cars (object property sbahn_background_vehicles): return
    (tris without the cars, [per-car (T,3,3) triangles]). Cars are the faces with vehicle
    materials, split into separate vehicles by 0.4 m proximity in plan."""
    names = [m.name if m else "" for m in me.materials]
    veh_slots = [i for i, n in enumerate(names) if any(t in n for t in VEHICLE_MATERIAL_TAGS)]
    if not veh_slots:
        return tris, []
    is_veh = np.isin(_tri_materials(me), veh_slots)
    vt = tris[is_veh]
    if not len(vt):
        return tris, []
    # union-find over triangles whose plan bounding boxes overlap within 0.4 m (cars stand >= 0.8 m apart)
    lo, hi = vt[:, :, :2].min(1) - 0.2, vt[:, :, :2].max(1) + 0.2
    parent = list(range(len(vt)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    order = np.argsort(lo[:, 0])
    active = []
    for i in order:
        active = [j for j in active if hi[j, 0] >= lo[i, 0]]
        for j in active:
            if lo[i, 1] <= hi[j, 1] and lo[j, 1] <= hi[i, 1]:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
        active.append(i)
    roots = np.array([find(i) for i in range(len(vt))])
    cars = [vt[roots == r] for r in np.unique(roots)]
    cars = [c for c in cars if np.ptp(c.reshape(-1, 3)[:, :2], axis=0).max() > 2.5]   # drop stray fragments
    return tris[~is_veh], cars


def _xf(tris: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (tris.reshape(-1, 3) @ M[:3, :3].T.astype(np.float32) + M[:3, 3].astype(np.float32)).reshape(-1, 3, 3)


class OccluderIndex:
    """Static scene geometry (terrain, track, catenary, scatter instances) as
    numpy triangle soups, culled per query to the pixels of one box and turned
    into a small BVHTree. scene.ray_cast costs ~0.1 s per ray on the 400 000
    instance scene; this costs ~1 s per frame for a few thousand rays."""

    def __init__(self, skip: set):
        dg = bpy.context.evaluated_depsgraph_get()
        dyn = {o.name for o in _dynamic_objects()}
        static, src_tris, src_sphere, inst = [], {}, {}, {}
        self.bg_cars: Dict[tuple, list] = {}        # instance source -> [local (T,3,3) per parked vehicle]
        for di in dg.object_instances:
            ob = di.object
            name = (di.parent.original.name if di.is_instance and di.parent else ob.original.name)
            if ob.original.name in dyn or name in dyn or ob.original.name in skip or name in skip:
                continue
            if ob.type not in ("MESH", "CURVE"):
                continue
            if ob.original.get("sbahn_obstacle"):
                continue
            M = np.array(di.matrix_world, dtype=np.float64)
            if di.is_instance:
                key = (ob.original.name, ob.data.name if ob.data else "")
                if key not in src_tris:
                    try:
                        t = _mesh_tris(ob.data) if ob.type == "MESH" else _mesh_tris(ob.to_mesh())
                        if ob.type == "MESH" and ob.original.get("sbahn_background_vehicles"):
                            t, cars = _split_vehicles(ob.data, t)
                            if cars:
                                self.bg_cars[key] = cars
                    except Exception:
                        t = np.zeros((0, 3, 3), np.float32)
                    src_tris[key] = t
                    if len(t):
                        c = t.reshape(-1, 3).mean(0)
                        src_sphere[key] = (c, float(np.linalg.norm(t.reshape(-1, 3) - c, axis=1).max()))
                if len(src_tris[key]):
                    inst.setdefault(key, []).append(M)
            else:
                try:
                    me = ob.data if ob.type == "MESH" else ob.to_mesh()
                    t = _mesh_tris(me)
                except Exception:
                    continue
                if len(t):
                    static.append(_xf(t, M))
        self.static = np.concatenate(static) if static else np.zeros((0, 3, 3), np.float32)
        self.static_c = self.static.mean(1)
        self.static_r = np.linalg.norm(self.static - self.static_c[:, None], axis=2).max(1)
        self.src_tris = src_tris
        self.inst = {}
        for key, Ms in inst.items():
            Ms = np.array(Ms)
            if key in self.bg_cars:
                self.bg_matrices = getattr(self, "bg_matrices", {})
                self.bg_matrices[key] = Ms
            c, r = src_sphere[key]
            centres = np.einsum("nij,j->ni", Ms[:, :3, :3], c) + Ms[:, :3, 3]
            scale = np.linalg.norm(Ms[:, :3, :3], axis=1).max(1)
            self.inst[key] = (Ms, centres.astype(np.float32), (r * scale).astype(np.float32))
        self.n_instances = sum(len(v[0]) for v in self.inst.values())

    @staticmethod
    def _cull(centres, radii, project, box, dmax):
        u, v, depth = project(centres.astype(np.float64))
        near = depth <= radii + 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            rp = project.__self__.intrinsics()["fx"] * radii / np.maximum(depth, 1e-3)
        x0, y0, x1, y1 = box
        ok = (depth > -radii) & (depth < dmax + radii) & (u > x0 - rp - 1) & (u < x1 + rp + 1) & (v > y0 - rp - 1) & (v < y1 + rp + 1)
        return ok | (near & (depth > -radii))

    def _local_bvh(self, key) -> BVHTree:
        cache = self.__dict__.setdefault("_src_bvh", {})
        if key not in cache:
            t = self.src_tris[key]
            cache[key] = BVHTree.FromPolygons(t.reshape(-1, 3).tolist(), np.arange(len(t) * 3).reshape(-1, 3).tolist(),
                                              all_triangles=True)
        return cache[key]

    def query(self, project, box, dmax, extra: list = ()):
        """Occluder for the rays through one box: static triangles (culled, one BVH) + dynamic
        triangles + scatter instances tested in their own local space (one cached BVH per source),
        so dense HD vegetation never gets flattened into a huge world-space triangle soup."""
        parts = []
        if len(self.static):
            parts.append(self.static[self._cull(self.static_c, self.static_r, project, box, dmax)])
        parts += [t for t in extra if len(t)]
        tris = np.concatenate(parts) if parts else np.zeros((0, 3, 3), np.float32)
        world = None
        if len(tris):
            world = BVHTree.FromPolygons(tris.reshape(-1, 3).tolist(), np.arange(len(tris) * 3).reshape(-1, 3).tolist(),
                                         all_triangles=True)
        cand = []
        for key, (Ms, centres, radii) in self.inst.items():
            sel = np.nonzero(self._cull(centres, radii, project, box, dmax))[0]
            if len(sel):
                cand.append((key, Ms[sel], centres[sel].astype(np.float64), radii[sel].astype(np.float64)))
        n_inst = sum(len(c[1]) for c in cand)
        return _Occluder(self, world, cand), len(tris) + n_inst


class _Occluder:
    def __init__(self, index, world, cand):
        self.world = world
        self.items = []
        for key, Ms, c, r in cand:
            bvh = index._local_bvh(key)
            inv = np.linalg.inv(Ms)
            self.items.append((bvh, inv, c, r))

    def ray_cast(self, o: Vector, d: Vector, max_dist: float):
        """(location,) of the first hit closer than max_dist, or (None,)."""
        if self.world is not None:
            hit = self.world.ray_cast(o, d, max_dist)
            if hit[0] is not None:
                return hit
        on = np.array(o); dn = np.array(d)
        for bvh, inv, c, r in self.items:
            rel = c - on
            t = rel @ dn
            perp2 = np.einsum("ij,ij->i", rel, rel) - t * t
            idx = np.nonzero((perp2 <= r * r) & (t + r > 0) & (t - r < max_dist))[0]
            for i in idx:
                lo = inv[i, :3, :3] @ on + inv[i, :3, 3]
                ld = inv[i, :3, :3] @ dn
                k = float(np.linalg.norm(ld))
                hit = bvh.ray_cast(Vector(lo), Vector(ld / k), max_dist * k)
                if hit[0] is not None:
                    return hit
        return (None,)


class Labeller:
    def __init__(self, scene=None, tp=None, grid: int = 48, min_visible_px: int = 1, min_visible_fraction: float = 0.0,
                 background_vehicles: bool = True, bg_max_depth_m: float = 400.0):
        self.background_vehicles = background_vehicles
        self.bg_max_depth_m = bg_max_depth_m
        self.scene = scene or bpy.context.scene
        self.tp = tp
        self.grid = grid
        self.min_visible_px = min_visible_px
        self.min_visible_fraction = min_visible_fraction
        self._bvh: Dict[str, tuple] = {}       # name -> (bvh in world space, matrix used)
        self._skip = None
        self._occ: Optional[OccluderIndex] = None

    def invalidate(self):
        """Call after static scene geometry changed (e.g. weather objects added)."""
        self._occ = None
        self._occ_sig = None

    def invalidate_if_changed(self):
        """Drop the occluder index only if the set of static occluding objects changed. Building it walks
        every scatter instance (about a minute on the full scene), and most condition changes only touch
        sky, fog and rain objects, which never occlude."""
        skip = _non_occluding_names()
        dyn = {o.name for o in _dynamic_objects()}
        sig = frozenset(o.name for o in bpy.data.objects
                        if o.type in ("MESH", "CURVE") and not o.hide_render and o.name not in skip
                        and o.name not in dyn and not o.get("sbahn_obstacle"))
        if sig != getattr(self, "_occ_sig", None):
            self._occ = None
            self._occ_sig = sig

    # ------------------------------------------------------------------ camera
    def resolution(self):
        r = self.scene.render
        k = r.resolution_percentage / 100.0
        return int(r.resolution_x * k), int(r.resolution_y * k)

    def intrinsics(self) -> dict:
        """Pinhole intrinsics in pixels for the rendered resolution (sensor fit AUTO/HORIZONTAL)."""
        cam = self.scene.camera
        W, H = self.resolution()
        cd = cam.data
        r = self.scene.render
        aspect = (W * r.pixel_aspect_x) / (H * r.pixel_aspect_y)
        fit = cd.sensor_fit
        if fit == "AUTO":
            fit = "HORIZONTAL" if aspect >= 1 else "VERTICAL"
        if fit == "HORIZONTAL":
            fx = cd.lens / cd.sensor_width * W
        else:
            fx = cd.lens / cd.sensor_height * H * r.pixel_aspect_y / r.pixel_aspect_x
        fy = fx * r.pixel_aspect_x / r.pixel_aspect_y
        cx = W * (0.5 - cd.shift_x * (W if fit == "HORIZONTAL" else H) / W)
        cy = H * 0.5 + cd.shift_y * (W if fit == "HORIZONTAL" else H)
        return dict(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy, lens_mm=cd.lens, sensor_mm=cd.sensor_width,
                    hfov_deg=math.degrees(2 * math.atan(W / (2 * fx))), vfov_deg=math.degrees(2 * math.atan(H / (2 * fy))),
                    clip_start=cd.clip_start, clip_end=cd.clip_end)

    def _camera_arrays(self):
        cam = self.scene.camera
        M = np.array(cam.matrix_world)
        Minv = np.linalg.inv(M)
        K = self.intrinsics()
        return M, Minv, K

    def project(self, pts_world: np.ndarray):
        """World (N,3) -> pixel u (right), v (down), depth (m along the view axis)."""
        M, Minv, K = self._camera_arrays()
        P = np.c_[pts_world, np.ones(len(pts_world))] @ Minv.T
        depth = -P[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K["cx"] + K["fx"] * P[:, 0] / depth
            v = K["cy"] - K["fy"] * P[:, 1] / depth
        return u, v, depth

    def pixel_ray(self, u, v):
        """World-space origin and unit direction through pixel (u, v)."""
        M, Minv, K = self._camera_arrays()
        d_cam = np.stack([(u - K["cx"]) / K["fx"], -(v - K["cy"]) / K["fy"], -np.ones_like(u)], axis=-1)
        d = d_cam @ M[:3, :3].T
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return M[:3, 3], d

    # ------------------------------------------------------------------ geometry
    def _world_verts(self, ob) -> np.ndarray:
        me = ob.data
        co = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        Mw = np.array(ob.matrix_world)
        return co @ Mw[:3, :3].T + Mw[:3, 3]

    def _bvh_for(self, ob) -> BVHTree:
        key = tuple(round(x, 5) for row in ob.matrix_world for x in row)
        cached = self._bvh.get(ob.name)
        if cached and cached[1] == key:
            return cached[0]
        V = self._world_verts(ob)
        polys = [tuple(p.vertices) for p in ob.data.polygons]
        bvh = BVHTree.FromPolygons([tuple(v) for v in V], polys, all_triangles=False)
        self._bvh[ob.name] = (bvh, key)
        return bvh

    # ------------------------------------------------------------------ main
    def annotate(self, objects: Optional[List[bpy.types.Object]] = None, s_train: Optional[float] = None) -> List[dict]:
        scene = self.scene
        self._skip = _non_occluding_names()
        if self._occ is None:
            self._occ = OccluderIndex(self._skip)
        dyn_tris = {}
        dg = bpy.context.evaluated_depsgraph_get()
        for o in _dynamic_objects():
            if o.type == "MESH" and not o.hide_render and o.name not in self._skip:
                me = o.evaluated_get(dg).data if o.modifiers else o.data
                dyn_tris[o.name] = _xf(_mesh_tris(me), np.array(o.matrix_world))
        W, H = self.resolution()
        M, Minv, K = self._camera_arrays()
        cam_pos = M[:3, 3]
        if s_train is None:
            s_train = float(scene.get("sbahn_train_s", 0.0))
        eye_back = float(scene.get("sbahn_cab_eye_back_m", 1.3))
        run_lat = float(scene.get("sbahn_running_track_offset_m", 2.0))
        objects = objects if objects is not None else [o for o in obstacles.all_obstacles() if not o.hide_render]
        out = []
        for ob in objects:
            V = self._world_verts(ob)
            centre = np.array(ob.matrix_world.translation)
            s_obj = float(ob.get("sbahn_s", 0.0))
            rec = dict(name=ob.name, class_id=int(ob["class_id"]), class_name=ob["class_name"], background=False,
                       variant=ob.get("variant", ""), note=ob.get("note", ""),
                       s=s_obj, lateral=float(ob.get("sbahn_lateral", 0.0)),
                       on_running_track=bool(abs(float(ob.get("sbahn_lateral", 0.0)) - run_lat) < RUNNING_TRACK_HALF_WIDTH
                                             or ob["class_name"] == "fallen_tree"),
                       world_xyz=[round(float(x), 3) for x in centre])
            if self.tp is not None:
                d_front = self.tp.distance_ahead(s_train, s_obj)
                if d_front > self.tp.length_m / 2:
                    d_front -= self.tp.length_m                   # negative = behind the train front
                rec["dist_along_track_m"] = round(d_front, 3)
                rec["dist_along_track_cam_m"] = round(d_front + eye_back, 3)
            rec["dist_euclid_cam_m"] = round(float(np.linalg.norm(centre - cam_pos)), 3)
            extra = [t for n, t in dyn_tris.items() if n != ob.name]
            self._measure(rec, V, lambda: self._bvh_for(ob), extra, W, H, K)
            out.append(rec)
        if self.background_vehicles:
            out += self._background_vehicles(dyn_tris, W, H, K, cam_pos, s_train)
        return out

    def _measure(self, rec, V, get_bvh, extra, W, H, K, grid=None):
        """Fill size, depth, amodal box, truncation, visibility and the labelled flag from world vertices V."""
        dims = V.max(0) - V.min(0)
        rec["size_m"] = [round(float(x), 3) for x in dims]
        u, v, depth = self.project(V)
        front = depth > K["clip_start"]
        rec["depth_cam_m"] = round(float(np.median(depth)), 3)
        rec.update(in_frame=False, bbox_amodal=None, bbox_visible=None, truncation=1.0,
                   object_px=0, visible_px=0, visible_fraction=0.0, labelled=False)
        if front.sum() == 0 or rec["depth_cam_m"] > K["clip_end"]:
            return
        uu, vv = u[front], v[front]
        x0, x1, y0, y1 = uu.min(), uu.max(), vv.min(), vv.max()
        cx0, cx1, cy0, cy1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        full = max((x1 - x0) * (y1 - y0), 1e-9)
        rec["in_frame"] = True
        rec["bbox_amodal"] = [round(float(a), 2) for a in (cx0, cy0, cx1, cy1)]
        rec["bbox_amodal_unclipped"] = [round(float(a), 2) for a in (x0, y0, x1, y1)]
        rec["truncation"] = round(float(1 - (cx1 - cx0) * (cy1 - cy0) / full), 4)
        self._visibility(rec, get_bvh(), extra, (cx0, cy0, cx1, cy1), float(depth[front].max()), grid)
        rec["labelled"] = bool(rec["visible_px"] >= self.min_visible_px and rec["visible_fraction"] >= self.min_visible_fraction
                               and rec["bbox_visible"] is not None)

    def _background_vehicles(self, dyn_tris, W, H, K, cam_pos, s_train) -> List[dict]:
        """Parked cars baked into the house plots: labelled as class car (background=True) so a
        detector is not penalised for finding them. Only those in front within bg_max_depth_m."""
        occ = self._occ
        out = []
        extra = list(dyn_tris.values())
        for key, cars in occ.bg_cars.items():
            Ms = getattr(occ, "bg_matrices", {}).get(key)
            if Ms is None:
                continue
            for ci, car in enumerate(cars):
                lc = car.reshape(-1, 3).mean(0)
                centres = np.einsum("nij,j->ni", Ms[:, :3, :3], lc) + Ms[:, :3, 3]
                u, v, depth = self.project(centres)
                sel = np.nonzero((depth > 1.0) & (depth < self.bg_max_depth_m) & (u > -0.2 * W) & (u < 1.2 * W)
                                 & (v > -0.2 * H) & (v < 1.2 * H))[0]
                for i in sel:
                    tris = _xf(car, Ms[i])
                    V = tris.reshape(-1, 3).astype(np.float64)
                    centre = V.mean(0)
                    rec = dict(name=f"bg_vehicle_{key[0]}_{ci}_{i}", class_id=obstacles.CLASS_ID["car"], class_name="car",
                               background=True, variant="parked", note=f"parked at {key[0]}", on_running_track=False,
                               world_xyz=[round(float(x), 3) for x in centre])
                    if self.tp is not None:
                        s_obj = float(self.tp.nearest_s([centre[0]], [centre[1]])[0])
                        rec["s"] = s_obj
                        d_front = self.tp.distance_ahead(s_train, s_obj)
                        if d_front > self.tp.length_m / 2:
                            d_front -= self.tp.length_m
                        rec["dist_along_track_m"] = round(d_front, 3)
                        rec["lateral_dist_m"] = round(float(np.linalg.norm(self.tp.point(s_obj)[0][:2] - centre[:2])), 2)
                    rec["dist_euclid_cam_m"] = round(float(np.linalg.norm(centre - cam_pos)), 3)

                    def get_bvh(t=tris):
                        n = len(t)
                        return BVHTree.FromPolygons(t.reshape(-1, 3).tolist(), np.arange(n * 3).reshape(-1, 3).tolist(),
                                                    all_triangles=True)
                    self._measure(rec, V, get_bvh, extra, W, H, K, grid=max(8, self.grid // 2))
                    if rec["in_frame"]:
                        out.append(rec)
        return out

    def _visibility(self, rec, bvh, extra, box, dmax, grid=None):
        grid = grid or self.grid
        cx0, cy0, cx1, cy1 = box
        # pixel-centre grid over the pixels the box touches
        px0, py0 = int(math.floor(cx0)), int(math.floor(cy0))
        px1, py1 = int(math.ceil(cx1)), int(math.ceil(cy1))
        # up to 4 x 4 rays per pixel for small objects, at most grid x grid for large ones
        nx = max(1, min(grid, 4 * (px1 - px0)))
        ny = max(1, min(grid, 4 * (py1 - py0)))
        gx = px0 + (np.arange(nx) + 0.5) * (px1 - px0) / nx
        gy = py0 + (np.arange(ny) + 0.5) * (py1 - py0) / ny
        cell_w, cell_h = (px1 - px0) / nx, (py1 - py0) / ny
        UU, VV = np.meshgrid(gx, gy)
        origin, dirs = self.pixel_ray(UU.ravel(), VV.ravel())
        o = Vector(origin)
        occ, n_occ = self._occ.query(self.project, (px0, py0, px1, py1), dmax, extra)
        rec["occluder_tris"] = n_occ
        obj_hits, vis_pts = 0, []
        for (uq, vq), d in zip(zip(UU.ravel(), VV.ravel()), dirs):
            dv = Vector(d)
            loc, nrm, idx, dist = bvh.ray_cast(o, dv)
            if loc is None:
                continue
            obj_hits += 1
            hit = occ.ray_cast(o, dv, max(dist - 0.02, 0.0)) if occ else (None,)
            if hit[0] is None:
                vis_pts.append((uq, vq))
        # scale grid counts to pixels
        px_per_cell = cell_w * cell_h
        rec["object_px"] = int(round(obj_hits * px_per_cell))
        rec["visible_px"] = int(round(len(vis_pts) * px_per_cell))
        rec["visible_fraction"] = round(len(vis_pts) / obj_hits, 4) if obj_hits else 0.0
        rec["grid"] = [nx, ny]
        if vis_pts:
            a = np.array(vis_pts)
            vb = [max(a[:, 0].min() - cell_w / 2, cx0), max(a[:, 1].min() - cell_h / 2, cy0),
                  min(a[:, 0].max() + cell_w / 2, cx1), min(a[:, 1].max() + cell_h / 2, cy1)]
            rec["bbox_visible"] = [round(float(x), 2) for x in vb]


def yolo_lines(objs: List[dict], W: int, H: int, box_key: str = "bbox_visible") -> List[str]:
    """YOLO rows (class cx cy w h, normalised) for labelled objects."""
    rows = []
    for o in objs:
        b = o.get(box_key)
        if not o.get("labelled") or not b:
            continue
        x0, y0, x1, y1 = b
        rows.append(f"{o['class_id']} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return rows

"""Instance-source collections for the scatter systems, built from the
procedural model library (replaces the low-poly proxies in scenery.py).

    SRC_Trees      near LOD trees (ribbon, within ~60 m of the track)
    SRC_TreesFar   far LOD trees (terrain grid)
    SRC_Bushes     shrubs and saplings
    SRC_Grass      grass tufts, tall grass, wildflowers, bracken
    SRC_Houses     house plots: building + driveway + parked car(s) / garage

External models override a category when files exist in
<asset_dir>/models/{trees,houses,vehicles}/ (.glb .gltf .obj .fbx .blend);
trees also feed SRC_TreesFar, vehicles replace the parked cars in the plots.
Parked cars carry the object property sbahn_background_vehicles (count) so
label code can tell them apart from the obstacle car.
"""
from __future__ import annotations

import glob
import math
import os
from typing import Dict, List

import bpy
import numpy as np

from .. import util
from . import houses as H
from . import materials as M
from . import trees as T
from . import vegetation as VG
from . import vehicles as VH
from .geom import MeshBuilder, box, mat4, rot_z

# (species, count) per LOD: weights of the uniform instance pick
TREE_MIX = [("spruce", 4), ("pine", 2), ("beech", 3), ("oak", 2), ("birch", 2), ("maple", 1)]
BUSH_MIX = [("shrub", 3), ("sapling", 2)]
HOUSE_MIX = [("efh", 5), ("dhh", 2), ("mfh", 2), ("barn", 1), ("hall", 1)]
PARKED = ["small", "hatchback", "hatchback", "estate", "sedan", "suv", "van"]


def _tree_mats(sp):
    return [M.bark(sp), M.foliage(sp)]


def import_models(folder: str, col, prefix: str) -> List[bpy.types.Object]:
    """Import every model file in folder as one joined mesh object each (origin at 0)."""
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        ext = path.lower().rsplit(".", 1)[-1]
        before = set(bpy.data.objects)
        try:
            if ext in ("glb", "gltf"):
                bpy.ops.import_scene.gltf(filepath=path)
            elif ext == "obj":
                bpy.ops.wm.obj_import(filepath=path)
            elif ext == "fbx":
                bpy.ops.import_scene.fbx(filepath=path)
            elif ext == "blend":
                with bpy.data.libraries.load(path, link=False) as (src, dst):
                    dst.objects = list(src.objects)
                for ob in dst.objects:
                    if ob is not None:
                        bpy.context.scene.collection.objects.link(ob)
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"[models] could not import {path}: {exc}")
            continue
        new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        for o in [o for o in bpy.data.objects if o not in before and o.type != "MESH"]:
            bpy.data.objects.remove(o, do_unlink=True)
        if not new:
            continue
        name = f"{prefix}_{os.path.basename(path).rsplit('.', 1)[0]}"
        ob = util.join_meshes(name, new, col) if len(new) > 1 else new[0]
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        col.objects.link(ob)
        ob.location = (0, 0, 0)
        out.append(ob)
    return out


def build_trees(cfg, rng, near_col, far_col):
    folder = os.path.join(cfg.asset_dir, "models", "trees")
    ext = import_models(folder, near_col, "Tree") if cfg.use_downloaded_assets and os.path.isdir(folder) else []
    if ext:
        for ob in ext:                                  # same meshes for the far field
            o2 = bpy.data.objects.new(ob.name + "_far", ob.data)
            far_col.objects.link(o2)
        return
    for sp, n in TREE_MIX:
        for i in range(n):
            seed = int(rng.integers(1 << 30))
            T.make_tree(sp, np.random.default_rng(seed), 0).to_object(f"Tree_{sp}_{i}", near_col, _tree_mats(sp))
            T.make_tree(sp, np.random.default_rng(seed + 1), 1).to_object(f"TreeFar_{sp}_{i}", far_col, _tree_mats(sp))


def build_bushes(cfg, rng, col):
    for sp, n in BUSH_MIX:
        for i in range(n):
            T.make_tree(sp, np.random.default_rng(int(rng.integers(1 << 30))), 0).to_object(f"Bush_{sp}_{i}", col, _tree_mats(sp))


def build_grass(cfg, rng, col):
    mats = [M.grass_blades(), M.flower_heads()]
    items = [("GrassTuft", lambda r: VG.grass_tuft(r))] * 10 + [("GrassTall", lambda r: VG.grass_tuft(r, tall=True))] * 4
    items += [(f"Flowers_{k}", (lambda k: lambda r: VG.flower_clump(r, k))(k)) for k in VG.FLOWER_KINDS]
    items += [("Fern", lambda r: VG.fern(r))]
    for i, (name, fn) in enumerate(items):
        fn(np.random.default_rng(int(rng.integers(1 << 30)))).to_object(f"{name}_{i}", col, mats)


def _parked(mb_plot: MeshBuilder, kind, rng, x, y, yaw, n_slots_house, veh_objs=None):
    car = VH.make_vehicle(kind, rng)
    car.transform(mat4(rot_z(yaw), (x, y, 0.0)))
    mb_plot.extend(car, mat_offset=n_slots_house)


def build_houses(cfg, rng, col):
    folder = os.path.join(cfg.asset_dir, "models", "houses")
    ext = import_models(folder, col, "House") if cfg.use_downloaded_assets and os.path.isdir(folder) else []
    if ext:
        return
    hm = M.house_materials()
    vm = VH.vehicle_materials(None)
    nh = len(H.SLOTS)
    k = 0
    for kind, n in HOUSE_MIX:
        for i in range(n):
            r = np.random.default_rng(int(rng.integers(1 << 30)))
            mb, (L, D) = H.make_building(kind, r)
            cars = 0
            if kind in ("efh", "dhh"):
                # driveway beside the house (+x end), paved, with a parked car; garage on half of the EFH
                dx = L / 2 + 1.8
                v, f, uv = box((3.2, D + 7.0, 0.06), (dx + 0.8, -3.0, 0.03))
                mb.add(v, f, H.S["paving"], uv=uv)
                if kind == "efh" and r.random() < 0.5:
                    g, (gl, gd) = H.garage(r)
                    g.transform(mat4(None, (dx + 0.8, D / 2 - gd / 2 + 0.5, 0)))
                    mb.extend(g)
                    _parked(mb, PARKED[r.integers(len(PARKED))], r, dx + 0.8, -D / 2 - 3.2, math.pi / 2 * 0 + r.normal(0, 0.05), nh)
                else:
                    _parked(mb, PARKED[r.integers(len(PARKED))], r, dx + 0.8, -D / 2 - 0.5, r.normal(0, 0.05), nh)
                cars = 1
                if kind == "dhh":
                    v, f, uv = box((3.2, D + 7.0, 0.06), (-dx - 0.8, -3.0, 0.03))
                    mb.add(v, f, H.S["paving"], uv=uv)
                    _parked(mb, PARKED[r.integers(len(PARKED))], r, -dx - 0.8, -D / 2 - 0.5, math.pi + r.normal(0, 0.05), nh)
                    cars = 2
            elif kind == "mfh":
                # parking row in front of the block
                v, f, uv = box((L, 6.0, 0.06), (0, -D / 2 - 4.5, 0.03))
                mb.add(v, f, H.S["paving"], uv=uv)
                xs = np.arange(-L / 2 + 1.5, L / 2 - 1.5, 2.6)
                for x in xs:
                    if r.random() < 0.6:
                        _parked(mb, PARKED[r.integers(len(PARKED))], r, x, -D / 2 - 4.5, math.pi / 2 + (math.pi if r.random() < 0.5 else 0) + r.normal(0, 0.03), nh)
                        cars += 1
            elif kind in ("barn", "hall"):
                v, f, uv = box((L * 0.6, 10.0, 0.06), (0, -D / 2 - 5.0, 0.03))
                mb.add(v, f, H.S["paving"], uv=uv)
                if r.random() < 0.7:
                    _parked(mb, "sprinter" if kind == "hall" else "van", r, -L / 4, -D / 2 - 5.0, math.pi / 2 + r.normal(0, 0.05), nh)
                    cars += 1
                if kind == "hall" and r.random() < 0.6:
                    _parked(mb, "truck", r, L / 5, -D / 2 - 5.5, math.pi / 2, nh)
                    cars += 1
            ob = mb.to_object(f"House_{kind}_{i}", col, hm + vm)
            ob["sbahn_background_vehicles"] = cars
            k += 1


def build_instance_sources(cfg, mats=None) -> Dict[str, bpy.types.Collection]:
    rng = np.random.default_rng(cfg.seed + 20)
    trees = util.ensure_collection("SRC_Trees")
    trees_far = util.ensure_collection("SRC_TreesFar")
    bushes = util.ensure_collection("SRC_Bushes")
    grass = util.ensure_collection("SRC_Grass")
    houses = util.ensure_collection("SRC_Houses")
    build_trees(cfg, rng, trees, trees_far)
    build_bushes(cfg, rng, bushes)
    build_grass(cfg, rng, grass)
    build_houses(cfg, rng, houses)
    for c in (trees, trees_far, bushes, grass, houses):
        util.hide_collection_instances_source(c)
    return dict(trees=trees, trees_far=trees_far, bushes=bushes, grass=grass, houses=houses)


def fallen_tree_parts(rng, species="spruce", length=None):
    """A wind-thrown tree lying along +X (root end at x=0) in the (parts, mats) format of
    obstacles.BUILDERS. Far LOD keeps the mesh light (~20-35k faces); the crown is pressed
    flat below the trunk and squashed above it, the trunk rests ~0.4 m above the ground."""
    mb = T.make_tree(species, rng, 1)
    V = np.concatenate(mb.V)
    k = (length or float(rng.uniform(12.0, 18.0))) / max(np.ptp(V[:, 2]), 1e-3)
    R = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]) * k    # +Z (up the trunk) -> +X
    mb.transform(mat4(R, (0, 0, 0)))
    out = []
    for v in mb.V:
        z = v[:, 2].copy()
        z = np.where(z > 0.4, 0.4 + 0.3 * (z - 0.4), np.maximum(z, -0.4))
        v = v.copy(); v[:, 2] = z + 0.4
        out.append(v)
    mb.V = out
    return _parts(mb), _tree_mats(species)


def _parts(mb: MeshBuilder):
    V = np.concatenate(mb.V)
    groups = list(zip(mb.T, mb.TM)) + list(zip(mb.Q, mb.QM))
    parts = []
    for m in sorted({int(x) for _, M_ in groups for x in np.unique(M_)}):
        faces = [F[M_ == m] for F, M_ in groups if (M_ == m).any()]
        used = np.unique(np.concatenate([f.ravel() for f in faces]))
        remap = np.full(len(V), -1)
        remap[used] = np.arange(len(used))
        parts.append((V[used], [list(remap[f]) for fs in faces for f in fs], m))
    return parts

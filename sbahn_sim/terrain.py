"""Terrain: rolling heightfield with the railway corridor cut into it
(formation, bench, 1:1.5 embankments and cuttings), road corridors for level
crossings flattened in, and land-use masks stored as mesh attributes for the
scatter systems and the terrain shader.

Two meshes:
  TerrainRibbon  fine strip (+-ribbon_half_w) built along the track frames
  TerrainGrid    coarse grid over the whole loop, with the ribbon area removed
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import util
from .config import Config
from .noise import Perlin2D, fbm
from .track_path import TrackPath


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


@dataclass
class FlattenLine:
    """A road or path: polyline (N,3) with target heights; terrain is blended
    to the line height within half_w and feathered over blend_w."""
    points: np.ndarray
    half_w: float = 4.0
    blend_w: float = 6.0

    def distance(self, x, y):
        P = self.points
        best_d = np.full(len(x), np.inf)
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            ab = b[:2] - a[:2]
            L2 = float(ab @ ab)
            t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(L2, 1e-9), 0, 1)
            d = np.hypot(x - (a[0] + t * ab[0]), y - (a[1] + t * ab[1]))
            best_d = np.minimum(best_d, d)
        return best_d

    def apply(self, x, y, z):
        P = self.points
        best_d = np.full(len(x), np.inf)
        best_z = np.zeros(len(x))
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            ab = b[:2] - a[:2]
            L2 = float(ab @ ab)
            t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(L2, 1e-9), 0, 1)
            px = a[0] + t * ab[0]; py = a[1] + t * ab[1]
            d = np.hypot(x - px, y - py)
            m = d < best_d
            best_d[m] = d[m]
            best_z[m] = a[2] + t[m] * (b[2] - a[2])
        w = 1.0 - smoothstep(self.half_w, self.half_w + self.blend_w, best_d)
        return z * (1 - w) + best_z * w


class Terrain:
    def __init__(self, cfg: Config, tp: TrackPath):
        self.cfg = cfg
        self.tp = tp
        tc = cfg.terrain
        self.p_hills = Perlin2D(cfg.seed + 100)
        self.p_detail = Perlin2D(cfg.seed + 101)
        self.p_lu = Perlin2D(cfg.seed + 102)
        self.flatten: List[FlattenLine] = []
        self._kd = None
        # sector boundaries along s
        shares = np.array([sh for _, sh in tc.sectors], dtype=np.float64)
        shares /= shares.sum()
        self.sector_types = [t for t, _ in tc.sectors]
        self.sector_s1 = np.cumsum(shares) * tp.length_m
        self.sector_s0 = np.concatenate([[0.0], self.sector_s1[:-1]])

    # ---- functions of (x, y) ----------------------------------------------
    def natural_height(self, x, y) -> np.ndarray:
        tc = self.cfg.terrain
        h = tc.hill_amplitude_m * fbm(self.p_hills, x, y, tc.hill_wavelength_m, 4, gain=0.45)
        h += tc.detail_amplitude_m * fbm(self.p_detail, x, y, tc.detail_wavelength_m, 3)
        return h

    def natural_height_fn(self):
        return lambda x, y: self.natural_height(np.asarray(x), np.asarray(y))

    def sector_of(self, s) -> np.ndarray:
        s = self.tp.wrap(np.asarray(s, dtype=np.float64))
        idx = np.clip(np.searchsorted(self.sector_s1, s, side="right"), 0, len(self.sector_types) - 1)
        return idx

    def sectors_table(self) -> List[dict]:
        return [dict(type=t, s0=float(a), s1=float(b)) for t, a, b in
                zip(self.sector_types, self.sector_s0, self.sector_s1)]

    def track_sd(self, x, y):
        """(s, signed lateral d) of the nearest centreline sample for each (x,y)."""
        if self._kd is None:
            self._kd = self.tp.kdtree()
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        n = len(x)
        idx = np.empty(n, dtype=np.int64)
        find = self._kd.find
        for i in range(n):
            idx[i] = find((x[i], y[i], 0.0))[1]
        tp = self.tp
        s = tp.s[idx]
        h = tp.heading[idx]
        dx = x - tp.x[idx]; dy = y - tp.y[idx]
        d = dx * np.sin(h) - dy * np.cos(h)        # dot with right vector (sin h, -cos h)
        return s, d

    def corridor_height(self, z_nat, s, d):
        """Apply formation / bench / embankment / cutting to natural heights."""
        pw = self.cfg.pway
        z_track = self.tp._interp("z", s)
        toe = self._ballast_toe()
        bench_end = toe + pw.bench_w_m
        ad = np.abs(d)
        z_bench = z_track + pw.formation_h - 0.02 * np.clip(ad - toe, 0, None)   # slight fall for drainage
        z = z_nat.copy()
        inside = ad <= bench_end
        z[inside] = z_bench[inside]
        out = ~inside
        run = (ad[out] - bench_end) / pw.embankment_slope
        fill = np.maximum(z_nat[out], z_bench[out] - run)
        cut = np.minimum(z_nat[out], z_bench[out] + run)
        z[out] = np.where(z_nat[out] < z_bench[out], fill, cut)
        return z

    def _ballast_toe(self) -> float:
        from .track_geometry import ballast_profile
        half = self.cfg.track.track_spacing_m / 2.0
        return ballast_profile(self.cfg, (half, -half))[1]

    def height(self, x, y, s=None, d=None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        if s is None:
            s, d = self.track_sd(x, y)
        z = self.corridor_height(self.natural_height(x, y), s, d)
        for fl in self.flatten:
            z = fl.apply(x, y, z)
        return z

    # ---- land use ----------------------------------------------------------
    def landuse(self, x, y, s, d) -> Dict[str, np.ndarray]:
        tc, sc = self.cfg.terrain, self.cfg.scatter
        ad = np.abs(d)
        toe = self._ballast_toe()
        bench_end = toe + self.cfg.pway.bench_w_m
        sec = self.sector_of(s)
        stype = np.array(self.sector_types, dtype=object)[sec]
        bias = np.where(stype == "forest", 0.45, np.where(stype == "field", -0.35, -0.30))
        falloff = 1.0 - smoothstep(350.0, 700.0, ad)
        lu = fbm(self.p_lu, x, y, tc.landuse_wavelength_m, 3) + bias * falloff
        forest = smoothstep(-0.05, 0.25, lu)
        suburban = (stype == "suburban").astype(np.float64) * (1.0 - smoothstep(300.0, 420.0, ad))
        forest = forest * (1.0 - 0.85 * suburban)
        # keep the immediate corridor open (embankments are grass)
        forest *= smoothstep(tc.tree_clearance_m, tc.tree_clearance_m + 6.0, ad)
        field = (1.0 - forest) * (1.0 - suburban)
        grassy = 1.0 - smoothstep(14.0, 40.0, ad)          # embankment / lineside grass
        bench = 1.0 - smoothstep(bench_end, bench_end + 2.5, ad)
        far = smoothstep(tc.scatter_max_dist_m - 80.0, tc.scatter_max_dist_m, ad)
        tree_density = (forest * sc.forest_tree_density
                        + field * sc.field_tree_density * (ad > 25)
                        + suburban * sc.suburban_tree_density) * (1 - far) * (ad > tc.tree_clearance_m)
        grass_density = sc.grass_density * (ad > toe - 0.3) * (1.0 - smoothstep(sc.grass_max_dist_m * 0.7, sc.grass_max_dist_m, ad))
        house_density = suburban * (1 - forest) * sc.house_density * smoothstep(22.0, 32.0, ad) * (1 - far)
        # keep roads (level-crossing approaches) free of trees, houses and grass
        for fl in self.flatten:
            road = 1.0 - smoothstep(fl.half_w + 1.0, fl.half_w + 4.0, fl.distance(x, y))
            tree_density = tree_density * (1 - road)
            house_density = house_density * (1 - smoothstep(0, 1, road) * 1.0) * (fl.distance(x, y) > 14.0)
            grass_density = grass_density * (1 - road)
        return dict(lu_forest=forest, lu_field=field, lu_suburban=suburban, grassy=grassy, bench=bench,
                    dist_track=ad, tree_density=tree_density, grass_density=grass_density,
                    house_density=house_density)

    # ---- meshes ------------------------------------------------------------
    def build_ribbon(self, mats, col):
        tc = self.cfg.terrain
        tp = self.tp
        W = tc.ribbon_half_w_m
        toe = self._ballast_toe()
        bench_end = toe + self.cfg.pway.bench_w_m
        lat_pos = np.array([toe, bench_end, bench_end + 1.5, bench_end + 3.0, bench_end + 5.0, 13.5, 17.0, 21.0, 26.0, 32.0, 39.0, 47.0, 55.0, W])
        lat = np.concatenate([-lat_pos[::-1], [0.0], lat_pos])
        n = int(round(tp.length_m / tc.ribbon_step_m))
        s = np.arange(n) * (tp.length_m / n)
        fr = tp.sample(s)
        k = len(lat)
        P = fr["pos"][:, None, :] + lat[None, :, None] * fr["right"][:, None, :]
        x = P[..., 0].ravel(); y = P[..., 1].ravel()
        S = np.repeat(s, k); D = np.tile(lat, n)
        z = self.corridor_height(self.natural_height(x, y), S, D)
        for fl in self.flatten:
            z = fl.apply(x, y, z)
        # sink the outer 10 m so the ribbon dives under the coarse grid
        sink = smoothstep(W - 12.0, W, np.abs(D)) * 0.6
        z = z - sink
        V = np.stack([x, y, z], axis=1)
        ii = np.arange(n); jj = np.arange(k - 1)
        i0 = ii[:, None] * k + jj[None, :]
        i1 = ((ii + 1) % n)[:, None] * k + jj[None, :]
        F = np.stack([i0, i1, i1 + 1, i0 + 1], axis=-1).reshape(-1, 4)
        ob = util.new_mesh_object("TerrainRibbon", V, F, col, [mats.get("terrain")], smooth=True)
        self._write_attrs(ob.data, x, y, S, D)
        return ob

    def build_grid(self, mats, col):
        tc = self.cfg.terrain
        x0, y0, x1, y1 = self.tp.meta["bbox"]
        m = tc.margin_m
        gx = np.arange(x0 - m, x1 + m + tc.grid_step_m, tc.grid_step_m)
        gy = np.arange(y0 - m, y1 + m + tc.grid_step_m, tc.grid_step_m)
        X, Y = np.meshgrid(gx, gy)
        x = X.ravel(); y = Y.ravel()
        s, d = self.track_sd(x, y)
        z = self.corridor_height(self.natural_height(x, y), s, d)
        for fl in self.flatten:
            z = fl.apply(x, y, z)
        V = np.stack([x, y, z], axis=1)
        ny, nx = X.shape
        idx = np.arange(nx * ny).reshape(ny, nx)
        F = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1).reshape(-1, 4)
        inner = np.abs(d) < (tc.ribbon_half_w_m - 10.0)
        keep = ~np.all(inner[F], axis=1)
        F = F[keep]
        ob = util.new_mesh_object("TerrainGrid", V, F, col, [mats.get("terrain")], smooth=True)
        self._write_attrs(ob.data, x, y, s, d)
        return ob

    def _write_attrs(self, me, x, y, s, d):
        lu = self.landuse(x, y, s, d)
        sc = self.cfg.scatter
        lu["tree_density_norm"] = np.clip(lu["tree_density"] / sc.forest_tree_density, 0, 1)
        lu["house_density_norm"] = np.clip(lu["house_density"] / sc.house_density, 0, 1)
        lu["bush_density_norm"] = np.clip(0.6 * lu["lu_field"] * (lu["dist_track"] > 12) * (1 - lu["lu_suburban"])
                                          * (1 - smoothstep(60.0, 140.0, lu["dist_track"]))
                                          + 0.5 * lu["lu_forest"] * (lu["dist_track"] > 9), 0, 1)
        for k, v in lu.items():
            util.set_point_attr(me, k, "FLOAT", v)
        util.set_point_attr(me, "track_s", "FLOAT", s)
        util.set_point_attr(me, "track_d", "FLOAT", d)


def terrain_material(name, mats_lib):
    """One shader for both terrain meshes, blending layers by point attributes."""
    from . import materials as M
    import bpy
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat, nt, bsdf = M._new(name)
    nt.nodes.remove(bsdf)
    out = nt.nodes["Material Output"]
    mp = M._objcoords(nt, 1.0, (-1500, 0))

    def layer(fn, loc, **kw):
        b = M._n(nt, "ShaderNodeBsdfPrincipled", loc)
        fn(nt, b, mp, loc, **kw)
        return b

    def textured_or(role, fn, loc, tile_m, tint=(1, 1, 1, 1), **kw):
        ts = M.find_texture_set(mats_lib.asset_dir, role) if getattr(mats_lib, "use_textures", False) else None
        if ts:
            b = M._n(nt, "ShaderNodeBsdfPrincipled", loc)
            M.layer_textured(nt, b, mp, ts, tile_m, loc, tint=tint)
            return b
        return layer(fn, loc, **kw)

    field = layer(M.layer_field, (-600, 1200))
    # photo scans tinted to summer albedos: the grass scan is a dry autumn meadow, the forest-floor scan
    # is orange leaf litter
    grass = textured_or("grass", M.layer_grass, (-600, 600), 1.4)          # ambientCG Grass001 is 1.4 m
    dry_ts = M.find_texture_set(mats_lib.asset_dir, "grass_dry") if getattr(mats_lib, "use_textures", False) else None
    if dry_ts and M.find_texture_set(mats_lib.asset_dir, "grass"):
        # dry, thin patches (about a quarter of the area, 10-20 m across) break up the green
        dry = M._n(nt, "ShaderNodeBsdfPrincipled", (-600, 1800))
        M.layer_textured(nt, dry, mp, dry_ts, 2.5, (-600, 1800), tint=(0.80, 0.90, 0.80, 1))
        dn = M._n(nt, "ShaderNodeTexNoise", (-300, 1500))
        dn.inputs["Scale"].default_value = 0.07
        dn.inputs["Detail"].default_value = 5.0
        nt.links.new(mp.outputs["Vector"], dn.inputs["Vector"])
        dm = M._n(nt, "ShaderNodeMapRange", (-150, 1500), interpolation_type="SMOOTHSTEP")
        dm.inputs["From Min"].default_value = 0.54
        dm.inputs["From Max"].default_value = 0.66
        nt.links.new(dn.outputs["Fac"], dm.inputs["Value"])
        gm = M._n(nt, "ShaderNodeMixShader", (0, 1500))
        nt.links.new(dm.outputs["Result"], gm.inputs["Fac"])
        nt.links.new(grass.outputs[0], gm.inputs[1])
        nt.links.new(dry.outputs[0], gm.inputs[2])
        grass = gm
    forest = textured_or("forest_floor", M.layer_grass, (-600, 0), 3.0, tint=(0.45, 0.60, 0.60, 1),
                         base=(0.11, 0.09, 0.045, 1))
    # Randweg beside the ballast: trodden fine gravel and soil (was a sand colour)
    soil = textured_or("walkway", M.layer_concrete, (-600, -600), 2.0, tint=(0.38, 0.38, 0.40, 1),
                       base=(0.13, 0.12, 0.10, 1), rough=0.92, scale=30.0)

    def attr(nm, loc):
        a = M._n(nt, "ShaderNodeAttribute", loc)
        a.attribute_name = nm
        a.attribute_type = "GEOMETRY"
        return a

    def mix(a, b, fac_node, loc):
        mx = M._n(nt, "ShaderNodeMixShader", loc)
        nt.links.new(fac_node.outputs["Fac"], mx.inputs["Fac"])
        nt.links.new(a.outputs[0], mx.inputs[1])
        nt.links.new(b.outputs[0], mx.inputs[2])
        return mx

    m1 = mix(field, grass, attr("grassy", (100, 900)), (300, 900))
    m2 = mix(m1, forest, attr("lu_forest", (300, 300)), (500, 300))
    # grass and weeds creep onto the walkway in patches
    bench = attr("bench", (300, -300))
    pn = M._n(nt, "ShaderNodeTexNoise", (300, -550))
    pn.inputs["Scale"].default_value = 0.6
    pn.inputs["Detail"].default_value = 4.0
    nt.links.new(mp.outputs["Vector"], pn.inputs["Vector"])
    pr = M._n(nt, "ShaderNodeMapRange", (450, -550), interpolation_type="SMOOTHSTEP")
    pr.inputs["From Min"].default_value = 0.42
    pr.inputs["From Max"].default_value = 0.62
    pr.inputs["To Min"].default_value = 1.0
    pr.inputs["To Max"].default_value = 0.15
    nt.links.new(pn.outputs["Fac"], pr.inputs["Value"])
    bm = M._n(nt, "ShaderNodeMath", (550, -400), operation="MULTIPLY")
    nt.links.new(bench.outputs["Fac"], bm.inputs[0])
    nt.links.new(pr.outputs["Result"], bm.inputs[1])
    m3 = M._n(nt, "ShaderNodeMixShader", (700, -300))
    nt.links.new(bm.outputs[0], m3.inputs["Fac"])
    nt.links.new(m2.outputs[0], m3.inputs[1])
    nt.links.new(soil.outputs[0], m3.inputs[2])
    nt.links.new(m3.outputs[0], out.inputs["Surface"])
    out.location = (900, 0)
    return mat


def build(cfg: Config, tp: TrackPath, mats, terrain: Terrain):
    col = util.ensure_collection("Terrain")
    mats._cache["terrain"] = terrain_material("SB_terrain", mats)
    ribbon = terrain.build_ribbon(mats, col)
    grid = terrain.build_grid(mats, col)
    return dict(ribbon=ribbon, grid=grid)

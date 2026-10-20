"""Geometry Nodes scatter systems on the terrain meshes.

Each modifier: Distribute Points on Faces (density from a point attribute the
terrain module wrote) -> Instance on Points (random pick / yaw / scale from a
source collection) -> Join with the terrain mesh.
"""
from __future__ import annotations

import bpy

from . import util
from .config import Config


def add_scatter(ob, name: str, source_col, density_attr: str, seed: int, dist_min: float = 0.0,
                density_max: float = 1.0, scale_range=(0.8, 1.25), poisson: bool = True,
                selection_attr: str = None, selection_min: float = 0.0):
    tree, gi, go = util.gn_tree(f"GN_Scatter_{ob.name}_{name}")
    dpf = util.node(tree, "GeometryNodeDistributePointsOnFaces", (-200, 200),
                    distribute_method="POISSON" if poisson else "RANDOM")
    dens = util.named_attr(tree, density_attr, "FLOAT", (-500, 100))
    if poisson:
        dpf.inputs["Distance Min"].default_value = dist_min
        dpf.inputs["Density Max"].default_value = density_max
        util.gn_link(tree, dens, "Attribute", dpf, "Density Factor")
    else:
        mul = util.node(tree, "ShaderNodeMath", (-350, 100), operation="MULTIPLY")
        mul.inputs[1].default_value = density_max
        util.gn_link(tree, dens, "Attribute", mul, 0)
        util.gn_link(tree, mul, "Value", dpf, "Density")
    dpf.inputs["Seed"].default_value = seed
    util.gn_link(tree, gi, "Geometry", dpf, "Mesh")
    if selection_attr:
        sel = util.named_attr(tree, selection_attr, "FLOAT", (-500, 350))
        cmp = util.node(tree, "FunctionNodeCompare", (-350, 350), data_type="FLOAT", operation="GREATER_THAN")
        cmp.inputs[1].default_value = selection_min
        util.gn_link(tree, sel, "Attribute", cmp, 0)
        util.gn_link(tree, cmp, "Result", dpf, "Selection")

    iop = util.node(tree, "GeometryNodeInstanceOnPoints", (250, 200))
    ci = util.node(tree, "GeometryNodeCollectionInfo", (-200, -100), transform_space="ORIGINAL")
    ci.inputs["Collection"].default_value = source_col
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    iop.inputs["Pick Instance"].default_value = True
    n_children = max(1, len(source_col.objects))
    ridx = util.node(tree, "FunctionNodeRandomValue", (-200, -300), data_type="INT")
    ridx.inputs[4].default_value = 0                 # int Min
    ridx.inputs[5].default_value = n_children - 1    # int Max
    ridx.inputs["Seed"].default_value = seed + 1
    ryaw = util.node(tree, "FunctionNodeRandomValue", (-200, -500), data_type="FLOAT")
    ryaw.inputs[2].default_value = 0.0               # float Min
    ryaw.inputs[3].default_value = 6.2831853         # float Max
    ryaw.inputs["Seed"].default_value = seed + 2
    cxyz = util.node(tree, "ShaderNodeCombineXYZ", (0, -500))
    util.gn_link(tree, ryaw, 1, cxyz, "Z")
    rscl = util.node(tree, "FunctionNodeRandomValue", (-200, -700), data_type="FLOAT")
    rscl.inputs[2].default_value = scale_range[0]
    rscl.inputs[3].default_value = scale_range[1]
    rscl.inputs["Seed"].default_value = seed + 3
    sxyz = util.node(tree, "ShaderNodeCombineXYZ", (0, -700))
    for ax in ("X", "Y", "Z"):
        util.gn_link(tree, rscl, 1, sxyz, ax)
    util.gn_link(tree, dpf, "Points", iop, "Points")
    util.gn_link(tree, ci, "Instances", iop, "Instance")
    util.gn_link(tree, ridx, 2, iop, "Instance Index")
    util.gn_link(tree, cxyz, "Vector", iop, "Rotation")
    util.gn_link(tree, sxyz, "Vector", iop, "Scale")
    join = util.node(tree, "GeometryNodeJoinGeometry", (450, 0))
    util.gn_link(tree, gi, "Geometry", join, "Geometry")
    util.gn_link(tree, iop, "Instances", join, "Geometry")
    util.gn_link(tree, join, "Geometry", go, "Geometry")
    mod = ob.modifiers.new(f"Scatter_{name}", "NODES")
    mod.node_group = tree
    return mod


def build(cfg: Config, terrain_objs: dict, sources: dict):
    sc = cfg.scatter
    seed = cfg.seed
    ribbon, grid = terrain_objs["ribbon"], terrain_objs["grid"]
    mods = []
    hd = "trees_far" in sources
    for ob in (ribbon, grid):
        # tree_density attribute is trees/m^2 (0..~0.02): use it as the factor of Density Max.
        # The detailed library has true-size trees and a lighter LOD for the far grid.
        src = sources["trees_far"] if (hd and ob is grid) else sources["trees"]
        mods.append(add_scatter(ob, "Trees", src, "tree_density_norm", seed, dist_min=3.5,
                                density_max=sc.forest_tree_density, scale_range=(0.8, 1.2) if hd else (0.7, 1.35)))
        mods.append(add_scatter(ob, "Bushes", sources["bushes"], "bush_density_norm", seed + 7, dist_min=2.5,
                                density_max=0.004, scale_range=(0.6, 1.4)))
    mods.append(add_scatter(ribbon, "Grass", sources["grass"], "grass_density", seed + 3, poisson=False,
                            density_max=1.0, scale_range=(0.6, 1.6)))
    mods.append(add_scatter(grid, "Houses", sources["houses"], "house_density_norm", seed + 5, dist_min=sc.house_min_dist_m,
                            density_max=sc.house_density, scale_range=(0.97, 1.03) if hd else (0.85, 1.15)))
    return mods

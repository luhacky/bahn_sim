"""Scene orchestration: build_scene(cfg) creates the whole loop, stores the
track LUT in the .blend, writes sidecar JSON files and saves the .blend."""
from __future__ import annotations

import json
import math
import os
import time

import bpy
import numpy as np

from . import catenary, scatter, scenery, signals, terrain as terrain_mod, track_geometry, train_rig, util
from .config import Config
from .materials import MaterialLibrary
from .track_path import TrackPath


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.render.fps = 30


def set_engine(cfg: Config):
    scene = bpy.context.scene
    engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    if cfg.render_engine.upper().startswith("EEVEE"):
        for cand in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            if cand in engines:
                scene.render.engine = cand
                break
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = cfg.preview_samples
        scene.cycles.use_denoising = True
        try:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        except Exception:
            pass
    scene.render.resolution_x, scene.render.resolution_y = cfg.preview_res
    scene.render.image_settings.file_format = "PNG"


def preview_lighting():
    """Placeholder daylight: Nishita sky + sun. sbahn_sim.conditions replaces this with the
    weather / day-night system; everything here lives in the 'Lighting' collection."""
    scene = bpy.context.scene
    world = bpy.data.worlds.get("SBahnWorld") or bpy.data.worlds.new("SBahnWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky_types = [e.identifier for e in sky.bl_rna.properties["sky_type"].enum_items]
    sky.sky_type = "NISHITA" if "NISHITA" in sky_types else ("MULTIPLE_SCATTERING" if "MULTIPLE_SCATTERING" in sky_types else sky_types[0])
    sky.sun_elevation = math.radians(38.0)
    sky.sun_rotation = math.radians(140.0)
    for attr, val in (("sun_intensity", 1.0), ("altitude", 200), ("air_density", 1.2), ("dust_density", 1.5)):
        if hasattr(sky, attr):
            setattr(sky, attr, val)
    bg.inputs["Strength"].default_value = 0.04   # physical sky models are bright: keep this low
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    col = util.ensure_collection("Lighting")
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 3.5
    sun.angle = math.radians(0.53)
    so = bpy.data.objects.new("Sun", sun)
    col.objects.link(so)
    so.rotation_euler = (math.radians(90 - 38.0), 0, math.radians(140.0 + 180.0))
    return so


def build_scene(cfg: Config) -> dict:
    t0 = time.time()
    log = lambda msg: print(f"[build {time.time() - t0:6.1f}s] {msg}")
    os.makedirs(cfg.out_path, exist_ok=True)
    reset_scene()
    set_engine(cfg)
    mats = MaterialLibrary(cfg.asset_dir, cfg.use_downloaded_assets)

    # 1. terrain function first so the alignment can follow the relief
    terr = None
    ground_fn = None
    if cfg.build_terrain:
        # a throw-away Terrain to expose the natural height function
        dummy = terrain_mod.Terrain.__new__(terrain_mod.Terrain)
        from .noise import Perlin2D
        dummy.cfg = cfg
        dummy.p_hills, dummy.p_detail = Perlin2D(cfg.seed + 100), Perlin2D(cfg.seed + 101)
        ground_fn = dummy.natural_height_fn()
    tp = TrackPath.build(cfg, ground_fn)
    tp.store_in_scene()
    tp.save_json(os.path.join(cfg.out_path, "track_lut.json"))
    log(f"track: L={tp.length_m:.0f} m, Rmin={tp.meta['min_radius_m']:.0f} m, "
        f"max gradient={tp.meta['max_gradient'] * 1000:.1f} permille, closure={tp.meta['closure_err_m']:.2f} m")
    tcol = util.ensure_collection("Track")
    tp.to_curve_object("TrackCentreline", 0.0, 0.0, collection=tcol)
    half = cfg.track.track_spacing_m / 2.0
    tp.to_curve_object("TrackRunning", +half, 0.0, collection=tcol)
    tp.to_curve_object("TrackOpposite", -half, 0.0, collection=tcol)
    for ob in ("TrackCentreline", "TrackRunning", "TrackOpposite"):
        bpy.data.objects[ob].hide_render = True

    terr = terrain_mod.Terrain(cfg, tp)
    sectors = terr.sectors_table()
    station = scenery.plan_station(cfg, tp, sectors) if cfg.build_station else None
    crossings = signals.plan_crossings(cfg, tp, station) if cfg.build_crossings else []
    if crossings:
        terr.flatten += signals.crossing_flatten_lines(cfg, tp, terr, crossings)

    # 2. permanent way
    tg = track_geometry.build(cfg, tp, mats)
    log("permanent way done")
    offsets = tg["track_offsets"]

    # 3. catenary + signals
    cat = dict(mast_s=[], mast_lateral_m=half + cfg.catenary.mast_offset_m)
    if cfg.build_catenary:
        cat = catenary.build(cfg, tp, mats, offsets)
        log(f"catenary: {len(cat['mast_s'])} mast pairs")
    sig_plan = []
    if cfg.build_signals:
        sig_plan = signals.plan_signals(cfg, tp, cat["mast_s"] or [0.0], offsets)
        signals.build_signals(cfg, tp, mats, sig_plan)
        if cat["mast_s"]:
            signals.build_hektometer(cfg, tp, mats, cat["mast_s"], cat["mast_lateral_m"])
        log(f"signals: {len(sig_plan)} Ks signals")

    # 4. terrain + scenery + scatter
    terrain_objs = None
    if cfg.build_terrain:
        terrain_objs = terrain_mod.build(cfg, tp, mats, terr)
        log(f"terrain: ribbon {len(terrain_objs['ribbon'].data.vertices)} verts, grid {len(terrain_objs['grid'].data.vertices)} verts")
    if crossings:
        signals.build_crossings(cfg, tp, mats, terr, crossings, offsets)
    barriers = dict(runs=[])
    if cfg.build_scatter or cfg.build_station:
        sources = scenery.build_instance_sources(cfg, mats)
        clear = [(c["s"] - 14.0, c["s"] + 14.0) for c in crossings]
        if station:
            clear.append((station["s0"] - 10.0, station["s1"] + 10.0))
        barriers = scenery.build_noise_barriers(cfg, tp, mats, sectors, cat["mast_s"], keep_clear=clear)
        if station:
            scenery.build_station(cfg, tp, mats, station)
        if cfg.build_scatter and terrain_objs:
            scatter.build(cfg, terrain_objs, sources)
            log("scatter modifiers added")

    # 5. train + camera + placeholder lighting
    if cfg.build_train:
        train_rig.build(cfg, tp, mats)
    preview_lighting()

    # 6. manifest for the weather / obstacle / dataset modules
    manifest = dict(
        version=1,
        seed=cfg.seed,
        blend=os.path.join(cfg.out_path, cfg.blend_name),
        track_lut="track_lut.json",
        length_m=tp.length_m,
        track_spacing_m=cfg.track.track_spacing_m,
        running_track_offset_m=cfg.track.running_track_offset_m,
        design_speed_kmh=cfg.track.design_speed_kmh,
        segments=tp.meta["segments"],
        straights=[dict(s0=s["s0"], s1=s["s1"], length_m=round(s["s1"] - s["s0"], 1))
                   for s in tp.meta["segments"] if s["kind"] == "straight"],
        sectors=sectors,
        masts=dict(s=cat["mast_s"], lateral_m=cat["mast_lateral_m"]),
        signals=sig_plan,
        crossings=[{k: v for k, v in c.items() if k != "road_profile"} for c in crossings],
        station=station,
        noise_barriers=barriers["runs"],
        camera=dict(name=train_rig.CAMERA_NAME, eye_height_m=cfg.train.cab_eye_h, eye_back_m=cfg.train.cab_eye_back_m,
                    focal_mm=cfg.train.cam_focal_mm, sensor_mm=cfg.train.cam_sensor_mm),
        train=dict(cars=cfg.train.cars, car_len_m=cfg.train.car_len_m, lateral_m=cfg.track.running_track_offset_m),
        heights=dict(rail_top=0.0, formation=cfg.pway.formation_h, contact_wire=cfg.catenary.contact_wire_h),
        conventions="s = arc length (m) along the centreline in running direction; lateral + = right; "
                    "height from rail top; use TrackPath.from_scene() / track_lut.json",
    )
    with open(os.path.join(cfg.out_path, "scene_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    cfg.save(os.path.join(cfg.out_path, "config_used.json"))
    blend_path = os.path.join(cfg.out_path, cfg.blend_name)
    # texture paths relative to the .blend, so the project folder can be moved or copied to another machine
    for img in bpy.data.images:
        if img.filepath and not img.filepath.startswith("//") and os.path.isabs(img.filepath):
            try:
                img.filepath = bpy.path.relpath(img.filepath, start=os.path.dirname(os.path.abspath(blend_path)))
            except ValueError:            # different drive on Windows: keep the absolute path
                pass
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
    log(f"saved {blend_path}")
    manifest["build_seconds"] = round(time.time() - t0, 1)
    return manifest

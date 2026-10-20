"""Snapshot renderer: steps the train towards fixed-distance obstacle test
cases under a list of condition sets and writes a YOLO dataset.

    from sbahn_sim.dataset import DatasetConfig, run
    run(DatasetConfig(out_dir="out/dataset", mode="approach", conditions=["clear_noon", "fog_dense"]))

Modes
-----
approach    per test case: the train drives towards the obstacle at speed_kmh
            from approach_from_m to approach_to_m ahead of it, one snapshot
            every dt_s seconds, plus an extra snapshot at every exact
            milestone distance (50/100/200/300/500 m by default). Only the
            obstacle(s) of that case are shown.
milestones  only the exact milestone snapshots (5 per case and condition).
drive       all cases active at once, the train drives the loop from
            drive_s0 for drive_duration_s, one snapshot every dt_s.

Output (out_dir)
----------------
images/<id>.png         rendered cab view
labels/<id>.txt         YOLO rows "class cx cy w h" (normalised, visible box)
meta/<id>.json          per-frame metadata (see README "Dataset")
frames.jsonl            all meta records, one per line (rebuilt by collect())
frames.csv              one row per (frame, obstacle) for quick analysis
dataset.yaml            YOLO dataset file (path, names)
cases.json, plan.json   the test cases and the frame plan that were used

Frames are independent: a finished frame (image + meta) is skipped on the
next run, so an interrupted run resumes, and `shard=(i, n)` splits the plan
across processes.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np

from . import camera_fx
from . import labels as labels_mod
from . import obstacles
from .track_path import TrackPath
from .train_rig import place_train

# ---------------------------------------------------------------------------
# conditions: implemented in sbahn_sim.conditions. A condition
# set is a preset name from conditions.PRESETS or a dict of condition keys;
# "name" is ours (used in frame ids) and is stripped before the call.
# ---------------------------------------------------------------------------
def condition_presets() -> Dict[str, dict]:
    try:
        from .conditions import PRESETS
        return dict(PRESETS)
    except ImportError:
        return {}


def _apply_conditions_fn():
    """sbahn_sim.conditions.apply_conditions, or a logging stub."""
    try:
        from .conditions import apply_conditions
        return apply_conditions, "sbahn_sim.conditions"
    except ImportError:
        def stub(scene, cond):
            return dict(cond, _stub=True)
        return stub, "stub (sbahn_sim.conditions not installed; scene lighting unchanged)"


def measured_sun(scene=None) -> dict:
    """Sun elevation / azimuth (deg, azimuth clockwise from +Y = north) read from the scene's sun lamp."""
    sun = next((o for o in bpy.data.objects if o.type == "LIGHT" and o.data.type == "SUN" and not o.hide_render), None)
    if sun is None:
        return dict(sun_object=None)
    z = np.array(sun.matrix_world.col[2][:3])         # a sun lamp shines along its local -Z
    z /= np.linalg.norm(z)
    return dict(sun_object=sun.name, sun_elevation_deg=round(math.degrees(math.asin(np.clip(z[2], -1, 1))), 3),
                sun_azimuth_deg=round(math.degrees(math.atan2(z[0], z[1])) % 360, 3),
                sun_strength=round(float(getattr(sun.data, "energy", 0.0)), 4))


# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    out_dir: str = "out/dataset"
    mode: str = "approach"                       # approach / milestones / drive
    conditions: List = field(default_factory=lambda: ["clear_noon"])   # preset names or dicts with 'name'
    cases: Optional[List[str]] = None            # names from obstacles.default_cases(); None = all
    milestones_m: List[float] = field(default_factory=lambda: [50.0, 100.0, 200.0, 300.0, 500.0])
    speed_kmh: float = 100.0
    dt_s: float = 0.5
    approach_from_m: float = 520.0
    approach_to_m: float = 25.0
    drive_s0: float = 0.0
    drive_duration_s: float = 300.0
    resolution: Tuple[int, int] = (1920, 1080)
    samples: int = 64
    engine: Optional[str] = None                 # CYCLES / BLENDER_EEVEE(_NEXT); None = keep the .blend setting
    denoise: bool = True
    image_format: str = "PNG"                    # PNG / JPEG
    label_grid: int = 48
    min_label_px: int = 4                        # visible pixels needed for a YOLO row (metadata keeps everything)
    background_vehicles: bool = True             # label parked cars of the detailed scenery as class car
    render: bool = True                          # False = labels + metadata only (fast plan check)
    reapply_conditions_per_frame: bool = True    # wiper phase, rain streaks, sun relative to heading follow the frame
    skip_existing: bool = True
    shard: Tuple[int, int] = (0, 1)
    max_frames: Optional[int] = None
    camera_fx: bool = True                       # lens / sensor model on the rendered image (sbahn_sim.camera_fx)
    motion_blur: bool = True                     # shutter-time motion blur of the moving train
    jpeg_quality: int = 90                       # in-camera compression of the camera model (0 = off)

    @property
    def speed_ms(self):
        return self.speed_kmh / 3.6


# condition keys used by the planner / camera model, not passed to apply_conditions
PLAN_KEYS = ("name", "cases", "milestones", "camera_artifacts")


def resolve_conditions(conds) -> List[dict]:
    out = []
    presets = condition_presets()
    for c in conds:
        if isinstance(c, str):
            if presets and c not in presets:
                raise ValueError(f"unknown condition preset {c!r}; one of {sorted(presets)}")
            out.append(dict(preset=c, name=c))
        else:
            c = dict(c)
            c.setdefault("name", f"cond{len(out):02d}")
            bad = sorted(set(c.get("camera_artifacts") or {}) - set(camera_fx.ARTIFACTS))
            if bad:
                raise ValueError(f"condition {c['name']}: unknown camera artifact(s) {bad}; known: {sorted(camera_fx.ARTIFACTS)}")
            out.append(c)
    return out


def plan_frames(cfg: DatasetConfig, cases: List[dict], conditions: List[dict], L: float) -> List[dict]:
    """Deterministic list of frames: {id, cond, case, seq, t, s_front, milestone_m}."""
    frames = []
    v = cfg.speed_ms
    for cond in conditions:
        if cfg.mode == "drive":
            n = int(round(cfg.drive_duration_s / cfg.dt_s)) + 1
            for k in range(n):
                t = k * cfg.dt_s
                frames.append(dict(id=f"{cond['name']}__drive__{k:05d}", cond=cond["name"], case="drive", seq=f"{cond['name']}__drive",
                                   t=round(t, 4), s_front=float((cfg.drive_s0 + v * t) % L), milestone_m=None))
            continue
        only = cond.get("cases")                  # a condition may restrict the cases / distances
        for case in cases:
            if only and case["name"] not in only:
                continue
            s_obj = case["specs"][0].s
            seq = f"{cond['name']}__{case['name']}"
            items = []
            if cfg.mode == "approach":
                k = 0
                while True:
                    d = cfg.approach_from_m - v * k * cfg.dt_s
                    if d < cfg.approach_to_m - 1e-6:
                        break
                    items.append((k * cfg.dt_s, d, None))
                    k += 1
            for m in cond.get("milestones") or cfg.milestones_m:
                if cfg.mode == "milestones" or cfg.approach_to_m <= m <= cfg.approach_from_m:
                    # same clock as the approach sequence, so a milestone frame is the same frame in both modes
                    t = (max(cfg.approach_from_m, m) - m) / v
                    items.append((t, m, m))
            items.sort(key=lambda x: (x[0], x[2] is None))
            for i, (t, d, ms) in enumerate(items):
                tag = f"m{int(ms):04d}" if ms is not None else f"{i:04d}"
                frames.append(dict(id=f"{seq}__{tag}", cond=cond["name"], case=case["name"], seq=seq,
                                   t=round(t, 4), s_front=float((s_obj - d) % L), milestone_m=ms, target_dist_m=round(d, 4)))
    return frames


def _setup_render(scene, cfg: DatasetConfig):
    r = scene.render
    if cfg.engine:
        eng = cfg.engine
        if eng.startswith("EEVEE") or eng.startswith("BLENDER_EEVEE"):
            items = [e.identifier for e in r.bl_rna.properties["engine"].enum_items]
            eng = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in items else "BLENDER_EEVEE"
        r.engine = eng
    r.resolution_x, r.resolution_y = cfg.resolution
    r.resolution_percentage = 100
    if r.engine == "CYCLES":
        scene.cycles.samples = cfg.samples
        scene.cycles.use_denoising = cfg.denoise
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = cfg.samples
    r.image_settings.file_format = cfg.image_format
    if cfg.image_format == "PNG":
        r.image_settings.color_mode = "RGB"
        r.image_settings.compression = 30
    else:
        r.image_settings.quality = 92
    r.use_motion_blur = False
    return r.engine


def _render_info(scene) -> dict:
    r = scene.render
    info = dict(engine=r.engine, resolution=[r.resolution_x, r.resolution_y])
    if r.engine == "CYCLES":
        info.update(samples=scene.cycles.samples, denoise=bool(scene.cycles.use_denoising))
    elif hasattr(scene, "eevee"):
        info.update(samples=scene.eevee.taa_render_samples)
    return info


def build_cases(tp: TrackPath, manifest: dict, names=None) -> List[dict]:
    obstacles.clear()
    run_lat = float(bpy.context.scene.get("sbahn_running_track_offset_m", 2.0))
    cases = obstacles.default_cases(manifest, run_lat)
    if names:
        unknown = set(names) - {c["name"] for c in cases}
        if unknown:
            raise ValueError(f"unknown case(s) {sorted(unknown)}; available: {[c['name'] for c in cases]}")
    # object index counts over ALL default cases, so an obstacle keeps its name (Obst_<i>_<kind>) and
    # pass_index whichever --cases subset a run uses, and metadata from different runs can be joined
    i = 0
    picked = []
    for c in cases:
        c["objects"] = []
        for sp in c["specs"]:
            if not names or c["name"] in names:
                ob = obstacles.place(tp, sp, index=i)
                c["objects"].append(ob.name)
            i += 1
        if not names or c["name"] in names:
            picked.append(c)
    return picked


def run(cfg: DatasetConfig, manifest: Optional[dict] = None) -> dict:
    scene = bpy.context.scene
    tp = TrackPath.from_scene()
    if manifest is None:
        mpath = os.path.join(os.path.dirname(os.path.abspath(bpy.data.filepath)), "scene_manifest.json")
        manifest = json.load(open(mpath))
    out = os.path.abspath(cfg.out_dir)
    for sub in ("images", "labels", "meta"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    engine = _setup_render(scene, cfg)
    cases = build_cases(tp, manifest, cfg.cases)
    conditions = resolve_conditions(cfg.conditions)
    frames = plan_frames(cfg, cases, conditions, tp.length_m)
    si, sn = cfg.shard
    mine = frames[si::sn]
    if cfg.max_frames:
        mine = mine[: cfg.max_frames]
    apply_conditions, cond_source = _apply_conditions_fn()
    _write_json(os.path.join(out, "cases.json"), [dict(name=c["name"], objects=c["objects"],
                                                       specs=[sp.to_dict() for sp in c["specs"]]) for c in cases])
    _write_json(os.path.join(out, "plan.json"), dict(config=_cfg_dict(cfg), n_frames=len(frames), frames=frames))
    write_dataset_yaml(out)

    lab = labels_mod.Labeller(scene, tp, grid=cfg.label_grid, min_visible_px=cfg.min_label_px,
                              background_vehicles=cfg.background_vehicles)
    intr = lab.intrinsics()
    ext = ".png" if cfg.image_format == "PNG" else ".jpg"
    case_by_name = {c["name"]: c for c in cases}
    cond_by_name = {c["name"]: c for c in conditions}
    applied_cond, effective = None, None
    done = skipped = 0
    t_start = time.time()
    print(f"[dataset] {len(mine)} of {len(frames)} frames (shard {si}/{sn}), engine {engine}, "
          f"{cfg.resolution[0]}x{cfg.resolution[1]}, conditions via {cond_source}")
    active = None
    frame_no = 0
    fx = camera_fx.CameraFX(enabled=cfg.camera_fx, motion_blur=cfg.motion_blur, jpeg_quality=cfg.jpeg_quality)
    depth_dir = os.path.join(out, "_depth_tmp")
    if fx.enabled and cfg.render:
        camera_fx.ensure_depth_output(scene, depth_dir)
    for fr in mine:
        img = os.path.join(out, "images", fr["id"] + ext)
        meta_path = os.path.join(out, "meta", fr["id"] + ".json")
        if cfg.skip_existing and os.path.exists(meta_path) and (os.path.exists(img) or not cfg.render):
            skipped += 1
            continue
        want = None if fr["case"] == "drive" else case_by_name[fr["case"]]["objects"]
        if want != active:
            obstacles.set_active(want)
            active = want
        place_train(fr["s_front"], tp=tp)
        scene["sbahn_time_s"] = fr["t"]
        frame_no += 1
        scene.frame_set(frame_no)
        if fr["cond"] != applied_cond or cfg.reapply_conditions_per_frame:
            cond = {k: v for k, v in cond_by_name[fr["cond"]].items() if k not in PLAN_KEYS}
            if not cond_source.startswith("stub"):
                cond.setdefault("time_s", fr["t"])
                cond.setdefault("train_speed_kmh", cfg.speed_kmh)
            effective = apply_conditions(scene, cond) or {}
            if fr["cond"] != applied_cond:
                lab.invalidate_if_changed()   # rebuild the occluder index only if occluding objects changed
            applied_cond = fr["cond"]
        bpy.context.view_layer.update()
        objs = lab.annotate(s_train=fr["s_front"])
        expo = camera_fx.exposure_from_conditions(fx, effective) if fx.enabled else None
        t0 = time.time()
        fx_info = None
        if cfg.render:
            scene.render.filepath = img
            bpy.ops.render.render(write_still=True)
            if fx.enabled:
                travel = (camera_fx.camera_travel(scene, tp, fr["s_front"], cfg.speed_ms, expo["shutter_s"])
                          if fx.motion_blur else None)
                fx_info = camera_fx.postprocess(img, fx, expo, fr["id"], intr=intr,
                                                depth_path=camera_fx.depth_file(scene, depth_dir), travel=travel,
                                                artifacts=cond_by_name[fr["cond"]].get("camera_artifacts"),
                                                sun=camera_fx.sun_in_camera(scene, intr), artifact_seed=fr["cond"])
        render_s = time.time() - t0
        rows = labels_mod.yolo_lines(objs, intr["width"], intr["height"])
        with open(os.path.join(out, "labels", fr["id"] + ".txt"), "w") as fh:
            fh.write("\n".join(rows) + ("\n" if rows else ""))
        cam = scene.camera
        meta = dict(
            id=fr["id"], image=os.path.relpath(img, out) if cfg.render else None,
            label=os.path.join("labels", fr["id"] + ".txt"),
            sequence=fr["seq"], case=fr["case"], condition=fr["cond"], mode=cfg.mode,
            timestamp_s=fr["t"], dt_s=cfg.dt_s, speed_kmh=cfg.speed_kmh,
            milestone_m=fr.get("milestone_m"), target_dist_m=fr.get("target_dist_m"),
            train=dict(s_front=round(fr["s_front"], 4), loop_length_m=tp.length_m,
                       curvature=float(tp.sample(fr["s_front"])["curvature"][0]),
                       gradient=float(tp.sample(fr["s_front"])["gradient"][0])),
            camera=dict(name=cam.name, matrix_world=[[round(x, 5) for x in row] for row in cam.matrix_world], **intr),
            conditions_requested=cond_by_name[fr["cond"]], conditions_effective=_jsonable(effective),
            conditions_source=cond_source, sun=measured_sun(scene),
            render=dict(_render_info(scene), seconds=round(render_s, 2)) if cfg.render else None,
            camera_fx=dict(fx.as_dict(), **(fx_info or expo or {})) if fx.enabled else None,
            objects=objs, n_labels=len(rows),
        )
        _write_json(meta_path, meta)
        done += 1
        el = time.time() - t_start
        print(f"[dataset] {done}/{len(mine) - skipped} {fr['id']} s={fr['s_front']:.1f} labels={len(rows)} "
              f"render={render_s:.1f}s elapsed={el / 60:.1f}min", flush=True)
    summary = collect(out)
    summary.update(rendered=done, skipped=skipped, minutes=round((time.time() - t_start) / 60, 2))
    return summary


# ---------------------------------------------------------------------------
def write_dataset_yaml(out):
    with open(os.path.join(out, "dataset.yaml"), "w") as fh:
        fh.write(f"# YOLO dataset written by sbahn_sim.dataset (benchmark / test split only)\n")
        # no 'path' key: Ultralytics then resolves images/ next to this file, so the folder can be moved or zipped
        fh.write(f"# written at {out}\ntrain: images\nval: images\ntest: images\n")
        fh.write(f"nc: {len(obstacles.CLASSES)}\nnames:\n")
        for k, v in obstacles.CLASSES.items():
            fh.write(f"  {k}: {v}\n")


def collect(out: str) -> dict:
    """Rebuild frames.jsonl / frames.csv from meta/*.json (safe after sharded runs)."""
    metas = []
    for p in sorted(glob.glob(os.path.join(out, "meta", "*.json"))):
        try:
            metas.append(json.load(open(p)))
        except Exception:
            pass
    with open(os.path.join(out, "frames.jsonl"), "w") as fh:
        for m in metas:
            fh.write(json.dumps(m) + "\n")
    cols = ["frame", "image", "sequence", "case", "condition", "timestamp_s", "milestone_m", "s_front",
            "fog_density", "rain_rate_mm_h", "time_of_day", "sun_elevation_deg", "sun_azimuth_deg",
            "cloud_cover", "visibility_m", "camera_artifacts", "shutter_s", "gain_ev",
            "object", "class_id", "class_name", "background", "on_running_track", "dist_along_track_m", "dist_euclid_cam_m",
            "in_frame", "labelled", "visible_fraction", "truncation", "visible_px",
            "bbox_visible_x0", "bbox_visible_y0", "bbox_visible_x1", "bbox_visible_y1",
            "bbox_amodal_x0", "bbox_amodal_y0", "bbox_amodal_x1", "bbox_amodal_y1"]
    n_obj = n_lab = 0
    with open(os.path.join(out, "frames.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in metas:
            ce = m.get("conditions_effective") or {}
            cr = m.get("conditions_requested") or {}
            sun = m.get("sun") or {}
            cfx = m.get("camera_fx") or {}
            arts = ";".join(f"{k}={v}" for k, v in sorted((cfx.get("artifacts") or {}).items()))
            cget = lambda k: ce.get(k, cr.get(k))
            for o in m["objects"]:
                n_obj += 1
                n_lab += bool(o["labelled"])
                bv = o.get("bbox_visible") or [None] * 4
                ba = o.get("bbox_amodal") or [None] * 4
                w.writerow([m["id"], m["image"], m["sequence"], m["case"], m["condition"], m["timestamp_s"], m["milestone_m"],
                            m["train"]["s_front"], cget("fog_density_per_m") if cget("fog_density_per_m") is not None else cget("fog_density"),
                            cget("rain_rate_mm_h"), cget("time_of_day"),
                            ce.get("sun_elevation_deg", sun.get("sun_elevation_deg")),
                            ce.get("sun_azimuth_deg", sun.get("sun_azimuth_deg")),
                            cget("cloud_cover"), cget("visibility_m"), arts, cfx.get("shutter_s"), cfx.get("gain_ev_total", cfx.get("gain_ev")),
                            o["name"], o["class_id"], o["class_name"], o.get("background", False), o["on_running_track"], o.get("dist_along_track_m"),
                            o["dist_euclid_cam_m"], o["in_frame"], o["labelled"], o["visible_fraction"], o["truncation"],
                            o["visible_px"], *bv, *ba])
    return dict(frames=len(metas), object_records=n_obj, labelled=n_lab, out=out)


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_jsonable(v) for v in x]
        try:
            return float(x)
        except Exception:
            return str(x)


def _cfg_dict(cfg):
    d = asdict(cfg)
    d["conditions"] = [c if isinstance(c, str) else dict(c) for c in cfg.conditions]
    return d


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)

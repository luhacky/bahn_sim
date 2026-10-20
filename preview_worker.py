"""Blender side of preview_server.py: keeps the scene loaded and renders one preview per request.

    blender -b out/scene/sbahn_loop.blend -P gpu_enable.py -P preview_worker.py -- <queue_dir>

A request is <queue_dir>/request.json (one condition dict plus "case", "distance_m", "res", "samples");
the answer is <queue_dir>/preview.png and <queue_dir>/result.json. No labels are computed (speed).
"""
import json
import os
import sys
import time
import traceback

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sbahn_sim import camera_fx, labels, obstacles  # noqa: E402
from sbahn_sim.conditions import apply_conditions  # noqa: E402
from sbahn_sim.dataset import PLAN_KEYS, build_cases  # noqa: E402
from sbahn_sim.track_path import TrackPath  # noqa: E402
from sbahn_sim.train_rig import place_train  # noqa: E402

queue = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
os.makedirs(queue, exist_ok=True)
scene = bpy.context.scene
tp = TrackPath.from_scene()
manifest = json.load(open(os.path.join(os.path.dirname(bpy.data.filepath), "scene_manifest.json")))
cases = {c["name"]: c for c in build_cases(tp, manifest)}
depth_dir = os.path.join(queue, "_depth")
camera_fx.ensure_depth_output(scene, depth_dir)
fx = camera_fx.CameraFX()
with open(os.path.join(queue, "ready.json"), "w") as fh:
    json.dump(dict(cases=sorted(cases)), fh)
print("[preview] ready", flush=True)

req_path = os.path.join(queue, "request.json")
frame = 0
while True:
    if not os.path.exists(req_path):
        time.sleep(0.3)
        continue
    t0 = time.time()
    try:
        req = json.load(open(req_path))
        os.remove(req_path)
        case = cases[req["case"]]
        obstacles.set_active(case["objects"])
        frame += 1
        place_train(case["specs"][0].s - float(req["distance_m"]), tp=tp)
        scene.frame_set(frame)
        scene.render.resolution_x, scene.render.resolution_y = req.get("res", [640, 360])
        scene.render.resolution_percentage = 100
        scene.cycles.samples = int(req.get("samples", 16))
        scene.render.image_settings.file_format = "PNG"
        cond = {k: v for k, v in req.items() if k not in PLAN_KEYS and k not in ("case", "distance_m", "res", "samples", "camera_fx")}
        eff = apply_conditions(scene, cond) or {}
        img = os.path.join(queue, "render.png")
        scene.render.filepath = img
        bpy.ops.render.render(write_still=True)
        intr = labels.Labeller(scene, tp).intrinsics()
        info = {}
        if req.get("camera_fx", True):
            expo = camera_fx.exposure_from_conditions(fx, eff)
            travel = camera_fx.camera_travel(scene, tp, case["specs"][0].s - float(req["distance_m"]), 100 / 3.6,
                                             expo["shutter_s"])
            info = camera_fx.postprocess(img, fx, expo, f"preview{frame}", intr=intr,
                                         depth_path=camera_fx.depth_file(scene, depth_dir), travel=travel,
                                         artifacts=req.get("camera_artifacts"), sun=camera_fx.sun_in_camera(scene, intr))
        os.replace(img, os.path.join(queue, "preview.png"))
        result = dict(ok=True, seconds=round(time.time() - t0, 1), effective=eff, camera=info)
    except Exception:
        result = dict(ok=False, error=traceback.format_exc()[-1500:])
    with open(os.path.join(queue, "result.json.tmp"), "w") as fh:
        json.dump(result, fh, default=str)
    os.replace(os.path.join(queue, "result.json.tmp"), os.path.join(queue, "result.json"))
    print(f"[preview] done in {time.time() - t0:.1f}s ok={result['ok']}", flush=True)

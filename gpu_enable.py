"""Enable Cycles GPU rendering (OptiX / CUDA / HIP / Metal / oneAPI, CPU if none) for every scene,
including after a .blend is opened.

    blender -b -P gpu_enable.py -P render_dataset.py -- ...
"""
import bpy
from bpy.app.handlers import persistent

prefs = bpy.context.preferences.addons["cycles"].preferences
DEVICE = "CPU"
for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):      # NVIDIA RTX, NVIDIA, AMD, Apple, Intel
    try:
        prefs.compute_device_type = backend
    except TypeError:
        continue
    prefs.refresh_devices()
    gpus = [d for d in prefs.devices if d.type == backend]
    if gpus:
        for d in prefs.devices:
            d.use = d.type == backend
        print(f"[gpu] {backend}: " + ", ".join(d.name for d in gpus))
        DEVICE = "GPU"
        break
else:
    prefs.compute_device_type = "NONE"
    print("[gpu] no GPU backend found, rendering on the CPU (much slower)")


@persistent
def _use_gpu(*_):
    for sc in bpy.data.scenes:
        sc.cycles.device = DEVICE


_use_gpu()
bpy.app.handlers.load_post.append(_use_gpu)


# --- speed-ups: skip scenery the cab camera cannot see, keep scene data between frames ---
KEEP = ("Train", "Lighting", "Weather")       # never cull the train, lights, fog/rain


@persistent
def _cull(*_):
    for sc in bpy.data.scenes:
        c = sc.cycles
        c.use_camera_cull = True
        c.camera_cull_margin = 0.15             # a little beyond the frame edge, for shadows and reflections
        c.use_distance_cull = True
        c.distance_cull_margin = 40.0           # everything within 40 m stays, visible or not
        sc.render.use_persistent_data = True
    for ob in bpy.data.objects:
        if any(col.name.startswith(KEEP) for col in ob.users_collection) or ob.name.startswith("Obst_"):
            continue
        if ob.parent and ob.parent.name.startswith(("Car_", "CabCamera")):
            continue
        ob.cycles.use_camera_cull = True
        ob.cycles.use_distance_cull = True
    print("[gpu] camera + distance culling on, persistent data on")


_cull()
bpy.app.handlers.load_post.append(_cull)

"""How the weather, obstacle and renderer modules use the built scene.

    blender -b out/sbahn_loop.blend -P examples/consumer_example.py
    python examples/consumer_example.py            # with the bpy wheel
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bpy
import numpy as np

from sbahn_sim.track_path import TrackPath
from sbahn_sim.train_rig import place_train
from sbahn_sim import signals

if not bpy.data.filepath:
    bpy.ops.wm.open_mainfile(filepath=os.path.join(os.path.dirname(HERE), "out", "sbahn_loop.blend"))

tp = TrackPath.from_scene()                       # LUT stored inside the .blend
print("loop length", round(tp.length_m, 1), "m")

# 1. put the train somewhere and place an obstacle exactly 200 m ahead on the running track
s_train = 4100.0
place_train(s_train, tp=tp)
s_obst = tp.wrap(s_train + 200.0)
lateral = float(bpy.context.scene["sbahn_running_track_offset_m"])
m = tp.matrix(s_obst, lateral=lateral, height=0.0, upright=True)   # local X right, Y forward, Z up
cube = bpy.data.objects.new("Obstacle", bpy.data.meshes.new("Obstacle"))
bpy.context.scene.collection.objects.link(cube)
cube.matrix_world = m
print("obstacle at", np.round(cube.matrix_world.translation, 2),
      "along-track distance", tp.distance_ahead(s_train, s_obst),
      "straight-line", round(float(np.linalg.norm(tp.point(s_obst, lateral) - tp.point(s_train, lateral))), 2))

# 2. geometry queries
f = tp.sample(s_obst)
print("curvature 1/m", float(f["curvature"][0]), "gradient", float(f["gradient"][0]), "cant deg", np.degrees(float(f["cant"][0])))

# 3. nearest arc length for an arbitrary world point (e.g. a scattered object)
print("nearest s of (500, 200):", tp.nearest_s([500.0], [200.0]))

# 4. switch a signal aspect
sig = bpy.data.objects.get("KsSignal_T0_00")
if sig:
    signals.set_aspect(sig, "Hp0")
    print(sig.name, "->", sig["aspect"], "at s =", sig["sbahn_s"])

# 5. the cab camera is the scene camera; frame-stepping is just place_train(s) per frame
cam = bpy.data.objects["CabCamera"]
print("camera", np.round(cam.matrix_world.translation, 2), "current s", cam["sbahn_s"])

"""Proxy Class 423 S-Bahn EMU + cab camera (Führerstand view).

    from sbahn_sim.train_rig import place_train
    place_train(s_front)      # moves all cars and the cab camera to arc length s_front

The train runs on the running track (lateral +track_spacing/2), front of the
first car at `s_front`, travelling in +s direction. The camera object is
"CabCamera"; its 'sbahn_s' property stores the current arc length.
"""
from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Matrix

from . import util
from .config import Config
from .materials import MaterialLibrary
from .track_path import TrackPath

CAMERA_NAME = "CabCamera"
TRAIN_COLLECTION = "Train"


def _car_mesh(cfg: Config, mats: MaterialLibrary, col, index: int, is_end: bool):
    tr = cfg.train
    L, W = tr.car_len_m, tr.width_m
    z0, z1 = tr.floor_h, tr.roof_h
    parts = []
    cab = 2.8 if is_end else 0.0            # cab length left open at the +Y end (camera sits inside)
    Lb = L - 0.3 - cab                      # body length behind the cab
    yb = -cab / 2.0                         # body centre shifts backwards
    # body, slightly narrower at the roof (tumblehome) is approximated with two boxes
    v, f = util.box((W, Lb, (z1 - z0) * 0.72), (0, yb, z0 + (z1 - z0) * 0.36)); parts.append((v, f, 0))
    v, f = util.box((W * 0.92, Lb - 0.2, (z1 - z0) * 0.30), (0, yb, z1 - (z1 - z0) * 0.15)); parts.append((v, f, 0))
    # red lower band (DB verkehrsrot)
    v, f = util.box((W + 0.02, Lb, 0.9), (0, yb, z0 + 0.45)); parts.append((v, f, 1))
    # window band
    v, f = util.box((W + 0.02, Lb - 0.9, 1.0), (0, yb, z0 + 1.85)); parts.append((v, f, 2))
    # underframe + bogies
    v, f = util.box((W * 0.85, L - 0.6, 0.45), (0, 0, z0 - 0.25)); parts.append((v, f, 3))
    for y in (-L * 0.32, L * 0.32):
        v, f = util.box((2.6, 2.8, 0.7), (0, y, 0.55)); parts.append((v, f, 3))
    # roof equipment
    v, f = util.box((1.6, 3.0, 0.35), (0, L * 0.15 - cab, z1 + 0.15)); parts.append((v, f, 3))
    if is_end:
        yc = L / 2 - 0.15 - cab / 2          # cab centre
        # cab shell: floor, roof, thin side walls, front below and above the windscreen (window left open)
        v, f = util.box((W, cab, 0.12), (0, yc, z0 + 0.06)); parts.append((v, f, 3))
        v, f = util.box((W * 0.95, cab, 0.12), (0, yc, z1 - 0.06)); parts.append((v, f, 0))
        for sx in (-1, 1):
            v, f = util.box((0.06, cab, z1 - z0), (sx * (W / 2 - 0.03), yc, (z0 + z1) / 2)); parts.append((v, f, 0))
        v, f = util.box((W, 0.35, 0.75), (0, L / 2 - 0.3, z0 + 0.37)); parts.append((v, f, 1))   # front below window (sill ~1.75 m)
        v, f = util.box((W * 0.95, 0.35, 0.45), (0, L / 2 - 0.3, z1 - 0.22)); parts.append((v, f, 0))  # front above window
        for x, z in ((-1.0, z0 + 0.35), (1.0, z0 + 0.35), (0.0, z1 - 0.45)):   # Dreilicht-Spitzensignal
            hv, hf = util.cylinder((x, L / 2 - 0.2, z), (x, L / 2 + 0.05, z), 0.13, 12)
            parts.append((hv, hf, 4))
    V, F, M = util.merge(parts)
    return util.new_mesh_object(f"Car_{index}", V, F, col, [mats.get("train_body"), mats.get("train_red"),
                                                            mats.get("glass_dark"), mats.get("galvanised"),
                                                            mats.get("lamp_white")], M)


def _cab_frame(cfg: Config, mats: MaterialLibrary, col):
    """Windscreen frame / pillars / dashboard as seen from the camera (parented to it)."""
    parts = []
    d = 1.15          # distance in front of the eye
    v, f = util.box((0.14, 0.05, 1.9), (-1.05, d, 0.25)); parts.append((v, f, 0))   # A pillars
    v, f = util.box((0.14, 0.05, 1.9), (1.05, d, 0.25)); parts.append((v, f, 0))
    v, f = util.box((2.4, 0.3, 0.16), (0, d, 0.98)); parts.append((v, f, 0))         # roof lip / sun visor
    v, f = util.box((2.4, 0.55, 0.35), (0, d - 0.15, -0.72)); parts.append((v, f, 1))  # desk
    v, f = util.box((1.0, 0.015, 0.02), (0.15, d - 0.02, -0.45)); parts.append((v, f, 0))  # wiper (parked along the sill)
    V, F, M = util.merge(parts)
    ob = util.new_mesh_object("CabFrame", V, F, col, [mats.get("signal_black"), mats.get("mast_paint")], M)
    return ob


def build(cfg: Config, tp: TrackPath, mats: MaterialLibrary):
    col = util.ensure_collection(TRAIN_COLLECTION)
    cars = [_car_mesh(cfg, mats, col, i, is_end=(i == 0 or i == cfg.train.cars - 1)) for i in range(cfg.train.cars)]
    cam_data = bpy.data.cameras.new(CAMERA_NAME)
    cam_data.lens = cfg.train.cam_focal_mm
    cam_data.sensor_width = cfg.train.cam_sensor_mm
    cam_data.clip_end = 5000.0
    cam = bpy.data.objects.new(CAMERA_NAME, cam_data)
    col.objects.link(cam)
    bpy.context.scene.camera = cam
    frame = _cab_frame(cfg, mats, col)
    frame.parent = cam
    frame.matrix_parent_inverse = Matrix.Identity(4)
    # the frame is modelled with +Y forward / +Z up; the camera looks down -Z, so undo the camera rotation
    frame.matrix_basis = Matrix.Rotation(-math.pi / 2, 4, "X")
    place_train(0.0, tp=tp, cfg=cfg)
    return dict(cars=cars, camera=cam, frame=frame)


def place_train(s_front: float, tp: TrackPath = None, cfg: Config = None, pitch_deg: float = -1.5):
    """Move the train (front of car 0 at s_front) and the cab camera along the loop."""
    tp = tp or TrackPath.from_scene()
    scene = bpy.context.scene
    lateral = float(scene.get("sbahn_running_track_offset_m", 2.0))
    n_cars = len([o for o in bpy.data.objects if o.name.startswith("Car_")])
    car_len = cfg.train.car_len_m if cfg else float(scene.get("sbahn_car_len_m", 16.85))
    eye_h = cfg.train.cab_eye_h if cfg else float(scene.get("sbahn_cab_eye_h", 2.45))
    eye_back = cfg.train.cab_eye_back_m if cfg else float(scene.get("sbahn_cab_eye_back_m", 1.3))
    for i in range(n_cars):
        ob = bpy.data.objects.get(f"Car_{i}")
        if ob:
            s_c = s_front - (i + 0.5) * car_len
            m = tp.matrix(s_c, lateral=lateral, height=0.0)
            if i == n_cars - 1 and n_cars > 1:
                m = m @ Matrix.Rotation(math.pi, 4, "Z")   # rear cab faces backwards
            ob.matrix_world = m
    cam = bpy.data.objects.get(CAMERA_NAME)
    if cam:
        m = tp.matrix(s_front - eye_back, lateral=lateral, height=eye_h)
        # camera looks down its local -Z: rotate so -Z = +Y(forward), +Y = up
        cam.matrix_world = m @ Matrix.Rotation(math.pi / 2 + math.radians(pitch_deg), 4, "X")
        cam["sbahn_s"] = float(s_front)
    scene["sbahn_train_s"] = float(s_front)
    if cfg:
        scene["sbahn_car_len_m"] = car_len
        scene["sbahn_cab_eye_h"] = eye_h
        scene["sbahn_cab_eye_back_m"] = eye_back

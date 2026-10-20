"""Draw the ground-truth boxes onto rendered frames for visual checking.

    python -m sbahn_sim.overlay out/dataset          (needs bpy for image IO)

Green = visible box (the YOLO label), cyan = labelled parked background car, thin grey = amodal box, red = obstacle in
frame but not labelled (hidden or too small); hidden parked cars are not drawn. Writes <dataset>/overlays/<id>.png.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

COLOURS = {"visible": (0.1, 1.0, 0.2), "background": (0.1, 0.8, 1.0), "amodal": (0.7, 0.7, 0.7), "hidden": (1.0, 0.1, 0.1)}


def _rect(px, box, col, t=2):
    H, W = px.shape[:2]
    # drawn just outside the box so small objects stay visible
    x0, y0, x1, y1 = int(math.floor(box[0])) - 1, int(math.floor(box[1])) - 1, int(math.ceil(box[2])), int(math.ceil(box[3]))
    x0, x1 = np.clip([x0, x1], 0, W - 1)
    y0, y1 = np.clip([y0, y1], 0, H - 1)
    for k in range(t):
        for yy in (y0 - k, y1 + k):
            if 0 <= yy < H:
                px[yy, max(x0 - k, 0):min(x1 + k, W - 1) + 1, :3] = col
        for xx in (x0 - k, x1 + k):
            if 0 <= xx < W:
                px[max(y0 - k, 0):min(y1 + k, H - 1) + 1, xx, :3] = col


def draw(meta: dict, root: str, out_dir: str) -> str:
    import bpy
    img_path = os.path.join(root, meta["image"])
    img = bpy.data.images.load(img_path)
    W, H = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)[::-1].copy()   # row 0 = top
    for o in meta["objects"]:
        if not o["in_frame"] or (o.get("background") and not o["labelled"]):
            continue
        if o.get("bbox_amodal"):
            _rect(px, o["bbox_amodal"], COLOURS["amodal"], 1)
        if o["labelled"]:
            _rect(px, o["bbox_visible"], COLOURS["background" if o.get("background") else "visible"], 2)
        else:
            _rect(px, o["bbox_amodal"], COLOURS["hidden"], 2)
    out = bpy.data.images.new("overlay", W, H, alpha=False)
    out.pixels.foreach_set(px[::-1].ravel())
    path = os.path.join(out_dir, meta["id"] + ".png")
    out.filepath_raw = path
    out.file_format = "PNG"
    out.save()
    bpy.data.images.remove(out)
    bpy.data.images.remove(img)
    return path


def draw_all(root: str) -> int:
    out_dir = os.path.join(root, "overlays")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(root, "meta", "*.json"))):
        meta = json.load(open(p))
        if meta.get("image") and os.path.exists(os.path.join(root, meta["image"])):
            draw(meta, root, out_dir)
            n += 1
    return n


if __name__ == "__main__":
    print(draw_all(os.path.abspath(sys.argv[1])), "overlays written")

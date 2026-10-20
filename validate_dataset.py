#!/usr/bin/env python3
"""Independent check of the YOLO labels against what Cycles actually renders.

For a small set of frames this script
  1. renders them with the normal pipeline (sbahn_sim.dataset.run, beauty + labels + meta + overlays), then
  2. re-renders every frame as a 1-sample Cycles pass with the Object Index pass (obstacles carry
     pass_index 100+i) and the Material Index pass (vehicle materials of parked cars get index 7),
     with the non-occluding weather objects (fog domain, rain streaks, windscreen glass) hidden,
  3. compares each label's visible box with the bounding box of the rendered mask pixels.

    python validate_dataset.py --blend out/sbahn_loop.blend --out out/validation/proxy
    python validate_dataset.py --blend out/hd/sbahn_loop.blend --out out/validation/hd --frames-preset hd

Writes <out>/validation_report.json (per object: IoU, edge errors in px, rendered vs labelled pixel
counts) and prints a summary. Needs the OpenEXR python package (pip install OpenEXR).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BG_VEHICLE_INDEX = 7

# (conditions, cases, milestones): small, covers every class, near/far, the curve and the wiper
FRAME_SETS = {
    "proxy": [
        (["clear_noon"], None, [100.0]),
        (["clear_noon"], ["person_standing_on_track", "car_on_crossing_00", "branches_in_curve"], [50.0, 300.0]),
        (["rain_heavy"], ["person_standing_on_track", "deer_on_track"], [50.0]),
        (["night_clear"], ["worker_on_opposite_track"], [100.0]),
    ],
    "smoke": [(["clear_noon"], ["person_standing_on_track"], [100.0])],
    "hd": [
        (["clear_noon"], ["car_on_crossing_01", "fallen_tree_across_tracks", "person_crossing_track"], [50.0, 200.0]),
    ],
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-preset", default="proxy", choices=sorted(FRAME_SETS))
    ap.add_argument("--res", type=int, nargs=2, default=[960, 540])
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--skip-beauty", action="store_true", help="only the mask pass (frames must exist in --out)")
    return ap.parse_args(argv)


def read_index_passes(path):
    import OpenEXR
    import numpy as np
    f = OpenEXR.File(path, separate_channels=True)
    ch = {n: c for part in f.parts for n, c in part.channels.items()}      # one part per pass in Blender 5
    def get(*tags):
        k = [n for n in ch if n.endswith(tags)]
        if not k:
            raise KeyError(f"{tags} not in {list(ch)}")
        return np.rint(np.asarray(ch[k[0]].pixels)).astype(int)
    return get("IndexOB.X", "Object Index.X"), get("IndexMA.X", "Material Index.X")


def box_of(mask):
    import numpy as np
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def mask_pass(out, scratch):
    import bpy
    import numpy as np
    from sbahn_sim import dataset, labels as labels_mod, obstacles
    from sbahn_sim.track_path import TrackPath
    from sbahn_sim.train_rig import place_train
    scene = bpy.context.scene
    tp = TrackPath.from_scene()
    vl = bpy.context.view_layer
    vl.use_pass_object_index = True
    vl.use_pass_material_index = True
    for m in bpy.data.materials:
        m.pass_index = BG_VEHICLE_INDEX if any(t in m.name for t in labels_mod.VEHICLE_MATERIAL_TAGS) else 0
    r = scene.render
    r.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    r.use_compositing = False                         # else the multilayer file only holds the composite
    if hasattr(r.image_settings, "media_type"):          # Blender 5: multilayer is a media type
        r.image_settings.media_type = "MULTI_LAYER_IMAGE"
    r.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    r.image_settings.color_depth = "32"
    apply_conditions, _ = dataset._apply_conditions_fn()
    report = []
    metas = [json.load(open(os.path.join(out, "meta", f))) for f in sorted(os.listdir(os.path.join(out, "meta")))]
    manifest = json.load(open(os.path.join(os.path.dirname(os.path.abspath(bpy.data.filepath)), "scene_manifest.json")))
    built = None
    for meta in metas:
        if meta["case"] != built:
            built = meta["case"]
            if built == "drive":                       # drive mode: every case at once, names are stable
                cs = dataset.build_cases(tp, manifest, None)
                case = dict(objects=[n for c in cs for n in c["objects"]])
            else:
                case = dataset.build_cases(tp, manifest, [meta["case"]])[0]
        obstacles.set_active(case["objects"])
        fg = [o["name"] for o in meta["objects"] if not o.get("background")]
        obj_for = dict(zip(fg, case["objects"])) if built != "drive" else {n: n for n in fg}
        place_train(meta["train"]["s_front"], tp=tp)
        scene["sbahn_time_s"] = meta["timestamp_s"]
        cond = {k: v for k, v in meta["conditions_requested"].items() if k != "name"}
        cond.setdefault("time_s", meta["timestamp_s"])
        cond.setdefault("train_speed_kmh", meta["speed_kmh"])
        apply_conditions(scene, cond)
        for name in labels_mod._non_occluding_names():
            bpy.data.objects[name].hide_render = True
        bpy.context.view_layer.update()
        r.resolution_x, r.resolution_y = meta["camera"]["width"], meta["camera"]["height"]
        r.resolution_percentage = 100
        exr = os.path.join(scratch, meta["id"] + ".exr")
        r.filepath = exr
        bpy.ops.render.render(write_still=True)
        idx_ob, idx_ma = read_index_passes(exr)
        H, W = idx_ob.shape
        for o in meta["objects"]:
            rec = dict(frame=meta["id"], object=o["name"], class_name=o["class_name"], background=o.get("background", False),
                       labelled=o["labelled"], dist_m=o.get("dist_along_track_m"), label_box=o.get("bbox_visible"),
                       label_visible_px=o["visible_px"], label_visible_fraction=o["visible_fraction"])
            if not o.get("background"):
                pi = bpy.data.objects[obj_for[o["name"]]].pass_index
                mask = idx_ob == pi
            else:
                if not o.get("bbox_amodal"):
                    continue
                x0, y0, x1, y1 = [int(round(v)) for v in o["bbox_amodal"]]
                mask = np.zeros_like(idx_ob, bool)
                mask[max(y0 - 1, 0):y1 + 1, max(x0 - 1, 0):x1 + 1] = True
                mask &= idx_ma == BG_VEHICLE_INDEX
            rb = box_of(mask)
            rec["render_box"] = rb
            rec["render_px"] = int(mask.sum())
            if rb and rec["label_box"]:
                lb = rec["label_box"]
                rec["iou"] = round(iou(lb, rb), 4)
                rec["edge_err_px"] = [round(lb[i] - rb[i], 2) for i in range(4)]
                rec["max_edge_err_px"] = round(max(abs(e) for e in rec["edge_err_px"]), 2)
            report.append(rec)
        print(f"[validate] {meta['id']}: " + ", ".join(
            f"{x['object']} iou={x.get('iou')} px {x['label_visible_px']}/{x['render_px']}"
            for x in report if x["frame"] == meta["id"] and not x["background"]), flush=True)
    return report


def summarise(report):
    import numpy as np
    fg = [r for r in report if not r["background"]]
    bg = [r for r in report if r["background"]]
    def stats(rs):
        both = [r for r in rs if "iou" in r]
        missing_label = [r for r in rs if r["render_px"] >= 4 and not r["labelled"]]
        ghost_label = [r for r in rs if r["labelled"] and r["render_px"] == 0]
        ious = np.array([r["iou"] for r in both]) if both else np.zeros(0)
        errs = np.array([r["max_edge_err_px"] for r in both]) if both else np.zeros(0)
        return dict(objects=len(rs), compared=len(both),
                    iou_mean=round(float(ious.mean()), 4) if len(ious) else None,
                    iou_min=round(float(ious.min()), 4) if len(ious) else None,
                    iou_below_0_7=[(r["frame"], r["object"], r["iou"]) for r in both if r["iou"] < 0.7],
                    max_edge_err_px_median=round(float(np.median(errs)), 2) if len(errs) else None,
                    rendered_but_unlabelled=[(r["frame"], r["object"], r["render_px"]) for r in missing_label],
                    labelled_but_not_rendered=[(r["frame"], r["object"]) for r in ghost_label])
    return dict(foreground=stats(fg), background_vehicles=stats(bg))


def main():
    args = parse_args()
    args.out = os.path.abspath(args.out)
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
    from sbahn_sim import dataset
    if not args.skip_beauty:
        for conds, cases, ms in FRAME_SETS[args.frames_preset]:
            cfg = dataset.DatasetConfig(out_dir=args.out, mode="milestones", conditions=conds, cases=cases,
                                        milestones_m=ms, resolution=tuple(args.res), samples=args.samples, engine="CYCLES")
            print("[validate] beauty:", dataset.run(cfg), flush=True)
        from sbahn_sim.overlay import draw_all
        draw_all(args.out)
    scratch = os.path.join(args.out, "_masks")
    os.makedirs(scratch, exist_ok=True)
    report = mask_pass(args.out, scratch)
    summary = summarise(report)
    with open(os.path.join(args.out, "validation_report.json"), "w") as fh:
        json.dump(dict(summary=summary, objects=report), fh, indent=1)
    print("[validate] summary:", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

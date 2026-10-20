#!/usr/bin/env python3
"""Render the YOLO benchmark snapshots (obstacles at fixed distances).

    blender -b out/sbahn_loop.blend -P render_dataset.py -- --out out/dataset --mode approach \
        --conditions clear_noon fog_dense night --res 1920 1080 --samples 64
    python render_dataset.py --blend out/sbahn_loop.blend --mode milestones --res 960 540 --samples 32

Useful flags
    --mode approach|milestones|drive   see sbahn_sim/dataset.py
    --dt 0.5 --speed 100               snapshot interval (s) and train speed (km/h)
    --milestones 50 100 200 300 500    exact distances (m, train front to obstacle)
    --cases person_standing_on_track car_on_crossing_00 ...   (default: all; --list shows them)
    --conditions clear_noon rain ...   presets from sbahn_sim.conditions.PRESETS or a JSON file
    --labels-only                      plan + labels + metadata without rendering (seconds, not hours)
    --shard 0 4                        this process renders frames 0, 4, 8, ...
    --overlay                          also write overlays/<id>.png with the boxes drawn in
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend", help="open this .blend first (default out/sbahn_loop.blend when not run inside Blender)")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "dataset"))
    ap.add_argument("--mode", default="approach", choices=["approach", "milestones", "drive"])
    ap.add_argument("--conditions", nargs="+", default=["clear_noon"],
                    help="preset names from sbahn_sim.conditions.PRESETS, or one .json file with a list of condition dicts (optional 'name')")
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--milestones", nargs="+", type=float)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--speed", type=float, default=100.0)
    ap.add_argument("--from-m", type=float, default=520.0)
    ap.add_argument("--to-m", type=float, default=25.0)
    ap.add_argument("--drive-s0", type=float, default=0.0)
    ap.add_argument("--drive-seconds", type=float, default=300.0)
    ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--engine", choices=["CYCLES", "EEVEE"])
    ap.add_argument("--format", default="PNG", choices=["PNG", "JPEG"])
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--min-px", type=int, default=4, help="visible pixels an object needs to get a YOLO row")
    ap.add_argument("--no-background-vehicles", action="store_true", help="do not label parked cars")
    ap.add_argument("--no-skip", action="store_true", help="re-render frames that already exist")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--list", action="store_true", help="list cases and condition presets, then exit")
    ap.add_argument("--no-camera-fx", action="store_true", help="skip the lens / sensor model (clean render)")
    ap.add_argument("--no-motion-blur", action="store_true", help="no shutter motion blur")
    ap.add_argument("--jpeg-quality", type=int, default=90, help="in-camera JPEG compression of the camera model, 0 = off")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    args.out = os.path.abspath(args.out)       # resolve now: the working directory can vanish on network mounts
    import bpy
    if args.blend or not bpy.data.filepath:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend or os.path.join(HERE, "out", "sbahn_loop.blend")))
    from sbahn_sim import dataset, obstacles
    if args.list:
        manifest = json.load(open(os.path.join(os.path.dirname(bpy.data.filepath), "scene_manifest.json")))
        for c in obstacles.default_cases(manifest):
            sp = c["specs"][0]
            print(f"case {c['name']:28s} {sp.kind:12s} s={sp.s:8.1f} lateral={sp.lateral:+.1f}  {sp.note}")
        for k, v in dataset.condition_presets().items():
            print(f"condition {k:12s} {v}")
        return
    conds = args.conditions
    if len(conds) == 1 and conds[0].endswith(".json"):
        conds = json.load(open(conds[0]))
    cfg = dataset.DatasetConfig(
        out_dir=args.out, mode=args.mode, conditions=conds, cases=args.cases,
        speed_kmh=args.speed, dt_s=args.dt, approach_from_m=args.from_m, approach_to_m=args.to_m,
        drive_s0=args.drive_s0, drive_duration_s=args.drive_seconds,
        resolution=tuple(args.res), samples=args.samples, engine=args.engine, image_format=args.format,
        render=not args.labels_only, skip_existing=not args.no_skip, shard=tuple(args.shard), max_frames=args.max_frames,
        min_label_px=args.min_px, background_vehicles=not args.no_background_vehicles,
        camera_fx=not args.no_camera_fx, motion_blur=not args.no_motion_blur, jpeg_quality=args.jpeg_quality)
    if args.milestones:
        cfg.milestones_m = args.milestones
    summary = dataset.run(cfg)
    if args.overlay and not args.labels_only:
        from sbahn_sim.overlay import draw_all
        n = draw_all(args.out)
        summary["overlays"] = n
    print("[dataset] done:", json.dumps(summary))


if __name__ == "__main__":
    main()

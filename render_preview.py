#!/usr/bin/env python3
"""Render validation frames from the cab camera at a few characteristic spots
(straight, curve, signal, crossing, station) to check the build visually.

    blender -b out/sbahn_loop.blend -P render_preview.py -- --out out --samples 48
    python render_preview.py --blend out/sbahn_loop.blend
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def pick_spots(manifest, n_extra=0):
    spots = []
    L = manifest["length_m"]
    straights = sorted(manifest["straights"], key=lambda s: -s["length_m"])
    if straights:
        s = straights[0]
        spots.append(("straight", s["s0"] + 60.0))
    curves = [c for c in manifest["segments"] if c["kind"] == "arc"]
    if curves:
        c = max(curves, key=lambda c: c["s1"] - c["s0"])
        spots.append(("curve", 0.5 * (c["s0"] + c["s1"])))
    for sig in manifest["signals"]:
        if sig["track"] == 0:
            spots.append(("signal", (sig["s"] - 140.0) % L))
            break
    for cr in manifest["crossings"][:1]:
        spots.append(("crossing", (cr["s"] - 120.0) % L))
    if manifest.get("station"):
        spots.append(("station", (manifest["station"]["s0"] - 150.0) % L))
    for nb in manifest.get("noise_barriers", [])[:1]:
        spots.append(("noise_barrier", (nb["s0"] + 40.0) % L))
    return spots


def render_previews(out_dir, manifest=None, samples=None, res=None, spots=None):
    import bpy
    from sbahn_sim.train_rig import place_train
    from sbahn_sim.track_path import TrackPath
    if manifest is None and spots is None:
        mpath = os.path.join(out_dir, "scene_manifest.json")
        if not os.path.exists(mpath):
            mpath = os.path.join(os.path.dirname(os.path.abspath(bpy.data.filepath)), "scene_manifest.json")
        manifest = json.load(open(mpath))
    os.makedirs(out_dir, exist_ok=True)
    tp = TrackPath.from_scene()
    scene = bpy.context.scene
    if samples and scene.render.engine == "CYCLES":
        scene.cycles.samples = samples
    if res:
        scene.render.resolution_x, scene.render.resolution_y = res
    spots = spots or pick_spots(manifest)
    for name, s in spots:
        place_train(float(s), tp=tp)
        scene.render.filepath = os.path.join(out_dir, f"preview_{name}_s{int(s):05d}.png")
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[preview] {name:14s} s={s:7.1f} -> {scene.render.filepath} ({time.time() - t:.0f} s)")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--blend", help="open this .blend first (not needed when Blender opened it)")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--res", type=int, nargs=2, default=None)
    ap.add_argument("--spot", action="append", default=[], help="name=s_metres, repeatable")
    args = ap.parse_args(argv)
    import bpy
    if args.blend:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
    spots = [(sp.split("=")[0], float(sp.split("=")[1])) for sp in args.spot] or None
    render_previews(os.path.abspath(args.out), samples=args.samples, res=tuple(args.res) if args.res else None, spots=spots)


if __name__ == "__main__":
    main()

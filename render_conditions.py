#!/usr/bin/env python3
"""Render the cab view under every weather / lighting preset (weather validation).

    python render_conditions.py --blend out/sbahn_loop.blend --out out/conditions --res 480 270 --samples 32
    blender -b out/sbahn_loop.blend -P render_conditions.py -- --preset fog_dense --preset night_rain --s 60

Writes <preset>_s<s>.png plus conditions_log.jsonl (one line of effective
parameters per image) and a contact sheet when Pillow is available.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", help="open this .blend first (not needed when Blender opened it)")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "conditions"))
    ap.add_argument("--preset", action="append", default=[], help="repeatable; default: all presets")
    ap.add_argument("--cond", action="append", default=[], help="extra JSON condition dicts, repeatable")
    ap.add_argument("--s", type=float, default=60.0, help="train front arc length (m)")
    ap.add_argument("--res", type=int, nargs=2, default=(480, 270))
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--time-s", type=float, default=0.5, help="clock for the wiper phase")
    args = ap.parse_args(argv)
    import bpy
    if args.blend:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
    from sbahn_sim.conditions import PRESETS, apply_conditions
    from sbahn_sim.train_rig import place_train
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = args.res
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
    os.makedirs(args.out, exist_ok=True)
    place_train(args.s)
    runs = [dict(preset=p) for p in (args.preset or ([] if args.cond else list(PRESETS)))]
    runs += [json.loads(c) for c in args.cond]
    files = []
    with open(os.path.join(args.out, "conditions_log.jsonl"), "a") as log:
        for i, cond in enumerate(runs):
            cond.setdefault("time_s", args.time_s)
            eff = apply_conditions(scene, cond)
            name = cond.get("preset") or f"custom{i:02d}"
            path = os.path.join(args.out, f"{name}_s{int(args.s):05d}.png")
            scene.render.filepath = path
            t = time.time()
            bpy.ops.render.render(write_still=True)
            eff.update(image=os.path.basename(path), s_train=args.s, render_seconds=round(time.time() - t, 1))
            log.write(json.dumps(eff) + "\n")
            files.append((name, path))
            print(f"[conditions] {name:15s} vis={eff['visibility_m']} ev={eff['exposure_ev']:+.2f} "
                  f"({eff['render_seconds']} s) -> {path}")
    try:
        from PIL import Image, ImageDraw
        ims = [Image.open(p).convert("RGB") for _, p in files]
        w, h = ims[0].size
        cols = 3
        sheet = Image.new("RGB", (w * cols, h * ((len(ims) + cols - 1) // cols)))
        for k, ((name, _), im) in enumerate(zip(files, ims)):
            x, y = (k % cols) * w, (k // cols) * h
            sheet.paste(im, (x, y))
            ImageDraw.Draw(sheet).text((x + 6, y + 6), name, fill=(255, 255, 0))
        sheet.save(os.path.join(args.out, "contact_sheet.png"))
    except ImportError:
        pass


if __name__ == "__main__":
    main()

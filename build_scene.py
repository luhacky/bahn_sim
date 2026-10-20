#!/usr/bin/env python3
"""Build the procedural S-Bahn loop.

    blender -b -P build_scene.py -- --out out --seed 7 --template stadium --length 8000
    python build_scene.py --out out            # with the `bpy` wheel (pip install bpy)

Any Config field can be overridden with --set a.b=value, e.g.
    --set terrain.grid_step_m=12 --set scatter.forest_tree_density=0.01 --set build_scatter=false
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--config", help="JSON config to start from (see out/config_used.json)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--template", choices=["stadium", "bean", "twist"])
    ap.add_argument("--length", type=float, help="target loop length in metres")
    ap.add_argument("--engine", choices=["CYCLES", "EEVEE"])
    ap.add_argument("--no-scatter", action="store_true")
    ap.add_argument("--no-terrain", action="store_true")
    ap.add_argument("--preview", action="store_true", help="render validation frames after building")
    ap.add_argument("--set", action="append", default=[], help="override: a.b=value")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    from sbahn_sim.config import Config
    cfg = Config.load(args.config) if args.config else Config()
    cfg.out_dir = args.out
    cfg.asset_dir = os.path.join(HERE, "assets") if cfg.asset_dir == "assets" else cfg.asset_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.template:
        cfg.track.template = args.template
    if args.length:
        cfg.track.target_length_m = args.length
    if args.engine:
        cfg.render_engine = args.engine
    if args.no_scatter:
        cfg.build_scatter = False
    if args.no_terrain:
        cfg.build_terrain = False
    cfg.apply_overrides(args.set)
    from sbahn_sim.build import build_scene
    manifest = build_scene(cfg)
    print(f"loop length {manifest['length_m']:.0f} m, {len(manifest['signals'])} signals, "
          f"{len(manifest['masts']['s'])} mast pairs, {len(manifest['crossings'])} crossings, "
          f"built in {manifest['build_seconds']} s")
    if args.preview:
        from render_preview import render_previews
        render_previews(cfg.out_path, manifest)


if __name__ == "__main__":
    main()

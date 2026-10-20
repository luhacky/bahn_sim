#!/usr/bin/env python3
"""Render the standard validation batch: 50 combinations of weather, light, fog and camera artifacts.

    python render_batch.py --dry-run      # list the combinations, frame count and time estimate
    python render_batch.py --test         # 8 frames from representative combinations (a few minutes)
    python render_batch.py                # the full batch into out/batch50 (resumes when restarted)

Useful options
    --out DIR            output folder (default out/batch50; --test uses out/batch50_test)
    --blend FILE         scene (default: out/scene/sbahn_loop.blend, built by setup.py)
    --res W H            resolution (default 1280 720)
    --samples N          path-tracing samples (default 32)
    --only NAME ...      render only these combinations (names from --dry-run)
    --frames-per-case 3  distances per case: 2 = default pairs, 3 = 50/100/200 m everywhere
    --blender PATH       Blender executable (otherwise found automatically, or set BLENDER)

The batch
    15 x every weather / light preset as it is (baseline)
    10 x fog sweep over a partly cloudy sky, 10 % to 100 % (100 % = 150 m visibility)
    25 x camera artifacts: lens flare, windscreen dirt, rain smears, lens smudge, worn camera,
         exposure errors (see sbahn_sim/camera_fx.py ARTIFACTS)
Every combination renders 3 of the 8 obstacle cases (rotating, so each case appears about 19 times)
at 2 distances; the fog sweep uses the same 3 cases at 50, 100 and 200 m so the levels compare directly.
Runs only need plain Python 3.8+; Blender does the rendering.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sbahn_sim.launch import find_blender, run_blender  # noqa: E402

CASES = ["person_standing_on_track", "car_on_crossing_00", "deer_on_track", "worker_on_opposite_track",
         "fallen_tree_across_tracks", "person_crossing_track", "branches_in_curve", "car_on_crossing_01"]
FOG_CASES = ["person_standing_on_track", "car_on_crossing_00", "worker_on_opposite_track"]
DIST_SETS = ([50, 150], [100, 250])
FULL_FOG = 0.026                           # 1/m: 100 % fog = 150 m visibility (fog_dense preset)

WORN = {"sensor_gain_ev": 1.5, "hot_pixels": 250, "jpeg_quality": 45, "defocus_px": 0.8}
ARTIFACT_COMBOS = [
    # name, preset, artifacts, extra condition keys
    ("flare_low_sun", "low_sun_glare", {"lens_flare": 1.0}, {}),
    ("flare_sunset", "sunset", {"lens_flare": 0.8}, {}),
    ("flare_morning", "clear_morning", {"lens_flare": 0.7}, {}),
    ("flare_partly_cloudy", "partly_cloudy", {"lens_flare": 0.6}, {}),
    ("flare_twilight", "twilight", {"lens_flare": 0.6}, {}),
    ("dirt_noon", "clear_noon", {"windscreen_dirt": 0.6}, {}),
    ("dirt_low_sun", "low_sun_glare", {"windscreen_dirt": 0.8, "lens_flare": 0.6}, {}),
    ("dirt_overcast", "overcast", {"windscreen_dirt": 0.5}, {}),
    ("dirt_night", "night_clear", {"windscreen_dirt": 0.7}, {}),
    ("dirt_sunset", "sunset", {"windscreen_dirt": 0.7}, {}),
    ("smear_rain_light", "rain_light", {"rain_smear": 0.6}, {}),
    ("smear_rain_heavy", "rain_heavy", {"rain_smear": 0.9}, {}),
    ("smear_night_rain", "night_rain", {"rain_smear": 0.8}, {}),
    ("smear_after_rain", "overcast", {"rain_smear": 0.5}, {"wetness": 0.8}),
    ("smear_fog", "fog_light", {"rain_smear": 0.5}, {}),
    ("smudge_partly_cloudy", "partly_cloudy", {"lens_smudge": 0.8}, {}),
    ("smudge_low_sun", "low_sun_glare", {"lens_smudge": 0.7}, {}),
    ("smudge_night", "night_clear", {"lens_smudge": 0.8}, {}),
    ("smudge_overcast", "overcast", {"lens_smudge": 0.6}, {}),
    ("worn_night", "night_clear", WORN, {}),
    ("worn_twilight", "twilight", WORN, {}),
    ("worn_overcast", "overcast", WORN, {}),
    ("overexposed_noon", "clear_noon", {"exposure_bias_ev": 1.0}, {}),
    ("underexposed_overcast", "overcast", {"exposure_bias_ev": -1.0}, {}),
    ("dirty_camera_sunset", "sunset", {"lens_flare": 0.8, "windscreen_dirt": 0.6, "lens_smudge": 0.5}, {}),
]
BASELINE = ["clear_noon", "clear_morning", "partly_cloudy", "overcast", "low_sun_glare", "sunset", "twilight",
            "night_clear", "night_overcast", "fog_light", "fog_dense", "night_fog", "rain_light", "rain_heavy",
            "night_rain"]
TEST_PICK = ["base_partly_cloudy", "fog_050", "flare_low_sun", "dirt_low_sun", "smear_rain_heavy",
             "smudge_partly_cloudy", "worn_night", "dirty_camera_sunset"]

SEC_PER_FRAME = 31.0      # 1280x720, 32 samples, RTX 2070 Super, incl. labels and camera model (measured)
SEC_PER_CONDITION = 10.0  # switching weather / light
SEC_STARTUP = 120.0       # loading the scene, first label index


def combinations(frames_per_case=2):
    combos, k = [], 0

    def cases_for(i):
        return [CASES[(3 * i + j) % len(CASES)] for j in range(3)]

    def dists_for(i):
        return [50, 100, 200] if frames_per_case >= 3 else DIST_SETS[i % 2]

    for p in BASELINE:
        combos.append(dict(name=f"base_{p}", preset=p, cases=cases_for(k), milestones=dists_for(k)))
        k += 1
    for pct in range(10, 101, 10):
        combos.append(dict(name=f"fog_{pct:03d}", preset="partly_cloudy", fog_percent=pct,
                           fog_density=round(FULL_FOG * pct / 100.0, 6), fog_top_m=80.0, fog_patchiness=0.0,
                           cases=list(FOG_CASES), milestones=[50, 100, 200]))
    for name, preset, arts, extra in ARTIFACT_COMBOS:
        c = dict(name=name, preset=preset, camera_artifacts=dict(arts), cases=cases_for(k), milestones=dists_for(k))
        c.update(extra)
        combos.append(c)
        k += 1
    return combos


def default_blend():
    for d in ("scene", "real3", "real2", "real1", "lit", ""):
        p = os.path.join(HERE, "out", d, "sbahn_loop.blend")
        if os.path.exists(p):
            return p
    raise SystemExit("no scene found: run setup.py (or build_scene.py) first")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--blend")
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--frames-per-case", type=int, default=2, choices=[2, 3])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--blender")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    combos = combinations(args.frames_per_case)
    if args.test:
        combos = [dict(c, cases=c["cases"][:1], milestones=[c["milestones"][0]]) for c in combos if c["name"] in TEST_PICK]
    if args.only:
        unknown = set(args.only) - {c["name"] for c in combos}
        if unknown:
            raise SystemExit(f"unknown combination(s) {sorted(unknown)}; see --dry-run")
        combos = [c for c in combos if c["name"] in args.only]
    n_frames = sum(len(c["cases"]) * len(c["milestones"]) for c in combos)
    scale = (args.res[0] * args.res[1]) / (1280 * 720) * (0.35 + 0.65 * args.samples / 32)
    est = SEC_STARTUP + len(combos) * SEC_PER_CONDITION + n_frames * SEC_PER_FRAME * scale
    print(f"{len(combos)} combinations, {n_frames} frames, estimated {est / 3600:.1f} h "
          f"at {args.res[0]}x{args.res[1]} / {args.samples} samples (RTX 2070 Super reference)")
    if args.dry_run:
        for c in combos:
            arts = ", ".join(f"{k}={v}" for k, v in (c.get("camera_artifacts") or {}).items()) or "-"
            fog = f"fog {c['fog_percent']}%" if "fog_percent" in c else ""
            print(f"  {c['name']:24s} {c['preset']:14s} {fog:9s} artifacts: {arts:55s} "
                  f"{len(c['cases'])} cases x {c['milestones']} m")
        return

    out = os.path.abspath(args.out or os.path.join(HERE, "out", "batch50_test" if args.test else "batch50"))
    os.makedirs(out, exist_ok=True)
    cond_path = os.path.join(out, "batch_conditions.json")
    with open(cond_path, "w", encoding="utf-8") as fh:
        json.dump(combos, fh, indent=1)
    blender, _ = find_blender(args.blender)
    blend = os.path.abspath(args.blend or default_blend())
    cases = sorted({c for combo in combos for c in combo["cases"]})
    rd_args = ["--blend", blend, "--mode", "milestones", "--conditions", cond_path, "--cases", *cases,
               "--res", str(args.res[0]), str(args.res[1]), "--samples", str(args.samples), "--out", out]
    if not args.no_overlay:
        rd_args.append("--overlay")
    t0 = time.time()
    rc = run_blender(blender, [os.path.join(HERE, "gpu_enable.py"), os.path.join(HERE, "render_dataset.py")],
                     rd_args, log_path=os.path.join(out, "batch.log"), cwd=HERE)
    print(f"[batch] finished with exit code {rc} after {(time.time() - t0) / 3600:.2f} h; output in {out}")
    sys.exit(rc)


if __name__ == "__main__":
    main()

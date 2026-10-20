#!/usr/bin/env python3
"""Set up the S-Bahn cab-view simulation on this machine (Windows, Linux or macOS).

    python setup.py                  # check Blender and GPU, get textures, build the scene, render a test frame
    python setup.py --get-blender    # also download a portable Blender 4.4 into tools/ if none is installed

Steps (each is skipped when its result already exists, so the script can be re-run safely)
    1. find Blender 4.2 or newer   (PATH, standard install folders, the BLENDER variable, tools/, or --blender)
    2. check the GPU                (OptiX / CUDA / HIP / Metal / oneAPI; CPU works but is ~10x slower)
    3. download the CC0 textures    (Poly Haven / ambientCG, ~150 MB, into assets/textures; skipped if present)
    4. build the scene              (out/scene/sbahn_loop.blend, about 2 minutes, deterministic from the seed)
    5. render one test frame        (out/setup_test/images, with labels and overlay)

Options
    --blender PATH       use this Blender
    --skip-textures      do not download textures (procedural materials are used for missing ones)
    --rebuild            rebuild the scene even if it exists
    --skip-test          no test frame
Only the Python standard library is needed to run this script (Python 3.8+).
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sbahn_sim.launch import find_blender, run_blender  # noqa: E402

BLENDER_VERSION = "4.4.3"
BLENDER_URLS = {
    "windows": f"https://download.blender.org/release/Blender4.4/blender-{BLENDER_VERSION}-windows-x64.zip",
    "linux": f"https://download.blender.org/release/Blender4.4/blender-{BLENDER_VERSION}-linux-x64.tar.xz",
}
SCENE_DIR = os.path.join(HERE, "out", "scene")


def step(n, text):
    print(f"\n=== {n}. {text}", flush=True)


def get_blender():
    key = "windows" if sys.platform.startswith("win") else ("linux" if sys.platform.startswith("linux") else None)
    if key is None:
        raise SystemExit("Automatic download is only set up for Windows and Linux. On macOS install Blender "
                         "4.4 from https://www.blender.org/download/ and re-run.")
    url = BLENDER_URLS[key]
    tools = os.path.join(HERE, "tools")
    os.makedirs(tools, exist_ok=True)
    archive = os.path.join(tools, url.rsplit("/", 1)[-1])
    if not os.path.exists(archive):
        print(f"downloading {url} (~400 MB) ...", flush=True)
        urllib.request.urlretrieve(url, archive)
    print("extracting ...", flush=True)
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(tools)
    else:
        with tarfile.open(archive) as t:
            t.extractall(tools)
    os.remove(archive)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blender")
    ap.add_argument("--get-blender", action="store_true")
    ap.add_argument("--skip-textures", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()
    if sys.version_info < (3, 8):
        raise SystemExit("Python 3.8 or newer is needed for this script")
    print(f"S-Bahn simulation setup in {HERE} ({platform.system()} {platform.machine()}, Python {platform.python_version()})")
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)

    step(1, "Blender")
    try:
        blender, ver = find_blender(args.blender)
    except SystemExit:
        if not args.get_blender:
            raise
        get_blender()
        blender, ver = find_blender(args.blender)

    step(2, "GPU")
    out = subprocess.run([blender, "-b", "--factory-startup", "-P", os.path.join(HERE, "gpu_enable.py")],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    gpu_lines = [ln for ln in out.splitlines() if ln.startswith("[gpu]")]
    print("\n".join(gpu_lines) or "[gpu] (no report)")
    if shutil.which("nvidia-smi"):
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
        print(f"nvidia-smi: {q}")

    step(3, "Textures (CC0, Poly Haven / ambientCG)")
    if args.skip_textures:
        print("skipped")
    else:
        from sbahn_sim.assets import download_manifest
        report = download_manifest(os.path.join(HERE, "assets"), "2k")
        for role, res in report.items():
            print(f"  {role:14s} {res}")
        if any(v == "failed" for v in report.values()):
            print("  some downloads failed: those materials fall back to procedural ones (check the network and re-run)")

    step(4, "Scene")
    blend = os.path.join(SCENE_DIR, "sbahn_loop.blend")
    if os.path.exists(blend) and not args.rebuild:
        print(f"present: {blend} (use --rebuild to build it again)")
    else:
        t0 = time.time()
        rc = run_blender(blender, [os.path.join(HERE, "build_scene.py")], ["--out", SCENE_DIR],
                         log_path=os.path.join(HERE, "out", "setup_build.log"), cwd=HERE)
        if rc != 0 or not os.path.exists(blend):
            raise SystemExit(f"scene build failed (exit {rc}), see out/setup_build.log")
        print(f"built in {time.time() - t0:.0f} s: {blend}")

    if not args.skip_test:
        step(5, "Test frame")
        test_out = os.path.join(HERE, "out", "setup_test")
        t0 = time.time()
        rc = run_blender(blender, [os.path.join(HERE, "gpu_enable.py"), os.path.join(HERE, "render_dataset.py")],
                         ["--blend", blend, "--mode", "milestones", "--milestones", "50",
                          "--cases", "person_standing_on_track", "--conditions", "partly_cloudy",
                          "--res", "1280", "720", "--samples", "32", "--out", test_out, "--overlay", "--no-skip"],
                         log_path=os.path.join(HERE, "out", "setup_test.log"), cwd=HERE)
        img = os.path.join(test_out, "images", "partly_cloudy__person_standing_on_track__m0050.png")
        if rc != 0 or not os.path.exists(img):
            raise SystemExit(f"test render failed (exit {rc}), see out/setup_test.log")
        print(f"test frame in {time.time() - t0:.0f} s (incl. loading): {img}")

    print("\nReady. Next steps (see HOWTO.md):\n"
          "  python render_batch.py --dry-run     list the 50 combinations and the time estimate\n"
          "  python render_batch.py --test        8 test frames\n"
          "  python render_batch.py               the full batch (resumes when restarted)")


if __name__ == "__main__":
    main()

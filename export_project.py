#!/usr/bin/env python3
"""Pack the project into one zip to move it to another machine (then run setup.py there).

    python export_project.py                   # code, docs, condition files and textures (~150 MB)
    python export_project.py --no-textures     # code only (~1 MB); setup.py downloads the textures again
    python export_project.py --with-scene      # also out/scene (the built .blend, ~145 MB; else setup rebuilds it)
    python export_project.py --with out/batch50 out/fog_validation   # also these result folders
    python export_project.py --out D:/transfer/sbahn.zip

Not included: out/ (unless asked for), tools/ (portable Blender), logs, caches and the original
download zips. Unpack anywhere and run  python setup.py  (or setup.bat / setup.sh).
"""
import argparse
import datetime
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {"__pycache__", ".git", "tools", "out", ".idea", ".vscode"}
SKIP_EXT = (".pyc", ".log", ".zip", ".bak", ".blend1")
PROJECT_DIRS = {"sbahn_sim", "examples", "conditions", "assets"}
PROJECT_EXT = (".py", ".md")
PROJECT_FILES = {"setup.bat", "setup.sh", ".gitignore", ".gitattributes"}


def walk(root, rel_base, skip_dirs=SKIP_DIRS):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for f in filenames:
            if f.lower().endswith(SKIP_EXT):
                continue
            full = os.path.join(dirpath, f)
            yield full, os.path.relpath(full, rel_base)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--no-textures", action="store_true")
    ap.add_argument("--with-scene", action="store_true")
    ap.add_argument("--with", dest="with_dirs", nargs="+", default=[])
    args = ap.parse_args()

    stamp = datetime.date.today().strftime("%Y%m%d")
    out = os.path.abspath(args.out or os.path.join(os.path.dirname(HERE), f"sbahn_sim_export_{stamp}.zip"))
    files = []
    skip = set(SKIP_DIRS)
    if args.no_textures:
        skip.add("textures")
    # only the project's own files: other things kept in this folder (personal documents etc.) never leave it
    for name in sorted(os.listdir(HERE)):
        full = os.path.join(HERE, name)
        if os.path.isdir(full) and name in PROJECT_DIRS:
            files += list(walk(full, HERE, skip))
        elif os.path.isfile(full) and (name.endswith(PROJECT_EXT) or name in PROJECT_FILES):
            files.append((full, name))
    extra_dirs = list(args.with_dirs)
    if args.with_scene:
        scene = os.path.join(HERE, "out", "scene")
        if not os.path.isdir(scene):
            sys.exit("out/scene not found: run setup.py first (or pass --with out/<folder>)")
        extra_dirs.append(scene)
    for d in extra_dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            sys.exit(f"not a folder: {d}")
        for full, rel in walk(d, HERE, {"__pycache__", "_depth_tmp"}):
            files.append((full, rel))
    top = "sbahn_sim_project"
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for full, rel in files:
            z.write(full, os.path.join(top, rel))
            total += os.path.getsize(full)
    print(f"{len(files)} files ({total / 1e6:.1f} MB unpacked) -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print("On the other machine: unzip, then run  python setup.py  inside sbahn_sim_project/")


if __name__ == "__main__":
    main()

"""Optional CC0 asset download hooks (Poly Haven, ambientCG).

Nothing here is required: every material has a procedural fallback. When a
texture set exists in <asset_dir>/textures/<role>/ the MaterialLibrary picks
it up automatically (ambientCG *_Color/_Roughness/_NormalGL naming and Poly
Haven *_diff/_rough/_nor_gl naming are both recognised).

Usage (needs outbound HTTPS):
    python -m sbahn_sim.assets --asset-dir assets            # download the manifest
    python -m sbahn_sim.assets --asset-dir assets --res 4k

Manifest entries map a material *role* (see materials._SPECS) to candidate
sources; the first that downloads wins. Edit ASSET_MANIFEST to taste.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile

POLYHAVEN_API = "https://api.polyhaven.com/files/{slug}"
AMBIENTCG_ZIP = "https://ambientcg.com/get?file={asset_id}_{res}-JPG.zip"

# role -> list of (source, id). Poly Haven ids are asset slugs, ambientCG ids are asset ids;
# ("copy", role) reuses another role's files. This is the set the current scenes are tuned for
# (tints in materials.py / terrain.py, list in assets/textures/SOURCES.md). All CC0.
ASSET_MANIFEST = {
    "ballast":      [("polyhaven", "gravel_stones"), ("ambientcg", "Gravel040")],      # angular crushed stone
    "sleeper":      [("polyhaven", "concrete_floor_worn_001"), ("ambientcg", "Concrete034")],
    "concrete":     [("copy", "sleeper")],                                               # foundations, shelters
    "grass":        [("ambientcg", "Grass001"), ("ambientcg", "Grass004")],             # dense green ground
    "grass_dry":    [("polyhaven", "grass_ground")],                                     # dry patches in the grass
    "walkway":      [("polyhaven", "gravel_floor_02")],                                  # Randweg beside the ballast
    "forest_floor": [("polyhaven", "forest_leaves_02"), ("polyhaven", "brown_mud_leaves_01")],
    "asphalt":      [("polyhaven", "asphalt_02"), ("ambientcg", "Asphalt012")],          # level crossings
    "rail_body":    [("polyhaven", "rust_coarse_01")],                                   # rail web and foot
    "bark_pine":    [("polyhaven", "pine_bark")],
    "bark_spruce":  [("polyhaven", "bark_brown_02")],
    "bark_oak":     [("polyhaven", "jolcham_oak_bark_01")],
}
# Further roles the material library understands but the scenes do not use (they would replace a tuned
# procedural material): field, soil, noise_wall, house_wall, roof_tile, platform, bark (proxies).
EXTRA_MANIFEST = {
    "field":        [("polyhaven", "aerial_mud_1"), ("ambientcg", "Ground037")],
    "noise_wall":   [("ambientcg", "Concrete031"), ("polyhaven", "concrete_wall_006")],
    "platform":     [("ambientcg", "PavingStones131"), ("polyhaven", "concrete_pavement_02")],
}
_JUNK = (".blend", ".usdc", ".usda", ".mtlx", ".tres")


def _get(url: str, timeout=60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sbahn-sim/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_polyhaven_texture(slug: str, dest: str, res: str = "2k") -> bool:
    info = json.loads(_get(POLYHAVEN_API.format(slug=slug)))
    wanted = {"Diffuse": "diff", "Rough": "rough", "nor_gl": "nor_gl", "Displacement": "disp"}
    os.makedirs(dest, exist_ok=True)
    got = 0
    for key, tag in wanted.items():
        entry = info.get(key, {}).get(res)
        if not entry:
            continue
        fmt = "jpg" if "jpg" in entry else next(iter(entry))
        url = entry[fmt]["url"]
        path = os.path.join(dest, f"{slug}_{tag}_{res}.{fmt}")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(_get(url, timeout=300))
        got += 1
    return got > 0


def download_ambientcg_texture(asset_id: str, dest: str, res: str = "2K") -> bool:
    data = _get(AMBIENTCG_ZIP.format(asset_id=asset_id, res=res.upper()), timeout=300)
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dest)
    for f in os.listdir(dest):                      # keep the maps only
        if f.lower().endswith(_JUNK) or "NormalDX" in f or f.lower() == f"{asset_id.lower()}.png":
            os.remove(os.path.join(dest, f))
    return True


def download_manifest(asset_dir: str, res: str = "2k", roles=None, manifest=None) -> dict:
    manifest = manifest or ASSET_MANIFEST
    report = {}
    for role, candidates in manifest.items():
        if roles and role not in roles:
            continue
        dest = os.path.join(asset_dir, "textures", role)
        if os.path.isdir(dest) and os.listdir(dest):
            report[role] = "present"
            continue
        report[role] = "failed"
        for source, ident in candidates:
            try:
                if source == "copy":
                    src = os.path.join(asset_dir, "textures", ident)
                    if not (os.path.isdir(src) and os.listdir(src)):
                        download_manifest(asset_dir, res, [ident], manifest)
                    os.makedirs(dest, exist_ok=True)
                    for f in os.listdir(src):
                        shutil.copy(os.path.join(src, f), os.path.join(dest, f))
                    ok = True
                else:
                    ok = (download_polyhaven_texture(ident, dest, res.lower()) if source == "polyhaven"
                          else download_ambientcg_texture(ident, dest, res.upper()))
                if ok:
                    report[role] = f"{source}:{ident}"
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"[assets] {role}: {source}:{ident} failed: {exc}", file=sys.stderr)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset-dir", default="assets")
    ap.add_argument("--res", default="2k", help="2k or 4k (Poly Haven) / 2K, 4K (ambientCG)")
    ap.add_argument("--roles", nargs="*", help="only these material roles")
    args = ap.parse_args(argv)
    rep = download_manifest(args.asset_dir, args.res, args.roles)
    for k, v in rep.items():
        print(f"{k:14s} {v}")


if __name__ == "__main__":
    main()

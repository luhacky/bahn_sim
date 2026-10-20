"""Find and run Blender from plain Python (no bpy needed): used by render_batch.py and setup.py."""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

MIN_VERSION = (4, 2)


def _candidates(explicit: Optional[str] = None) -> List[str]:
    c = [explicit, os.environ.get("BLENDER"), shutil.which("blender"), shutil.which("blender.exe")]
    if sys.platform.startswith("win"):
        for root in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", "")):
            if root:
                c += sorted(glob.glob(os.path.join(root, "Blender Foundation", "Blender*", "blender.exe")), reverse=True)
        c += sorted(glob.glob(os.path.expanduser(r"~\blender*\blender.exe")), reverse=True)
    elif sys.platform == "darwin":
        c += ["/Applications/Blender.app/Contents/MacOS/Blender",
              os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender")]
    else:
        c += sorted(glob.glob(os.path.expanduser("~/blender*/blender")), reverse=True)
        c += sorted(glob.glob("/opt/blender*/blender"), reverse=True)
        c += ["/snap/bin/blender", "/usr/local/bin/blender", "/usr/bin/blender"]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe = "blender.exe" if sys.platform.startswith("win") else "blender"
    c += sorted(glob.glob(os.path.join(here, "tools", "blender*", exe)), reverse=True)        # setup.py --get-blender
    return [p for p in c if p]


def blender_version(path: str) -> Optional[Tuple[int, int, int]]:
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"Blender (\d+)\.(\d+)(?:\.(\d+))?", out)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) if m else None


def find_blender(explicit: Optional[str] = None, quiet: bool = False) -> Tuple[str, Tuple[int, int, int]]:
    """Return (path, version) of the first Blender >= MIN_VERSION, or raise SystemExit with a hint."""
    seen = set()
    for p in _candidates(explicit):
        p = os.path.abspath(p)
        if p in seen or not os.path.isfile(p):
            continue
        seen.add(p)
        v = blender_version(p)
        if v and v[:2] >= MIN_VERSION:
            if not quiet:
                print(f"[launch] Blender {'.'.join(map(str, v))}: {p}")
            return p, v
    raise SystemExit("Blender 4.2 or newer not found. Install it from https://www.blender.org/download/ "
                     "(or run setup.py --get-blender), or pass --blender PATH / set the BLENDER variable.")


def run_blender(blender: str, scripts: List[str], args: List[str], blend: Optional[str] = None,
                log_path: Optional[str] = None, cwd: Optional[str] = None) -> int:
    """blender -b [blend] -P script1 -P script2 ... -- args ; output is echoed and optionally logged."""
    cmd = [blender, "-b"] + ([blend] if blend else [])
    for s in scripts:
        cmd += ["-P", s]
    cmd += ["--"] + list(args)
    print("[launch]", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", cwd=cwd)
        for line in proc.stdout:
            if log:
                log.write(line)
            # keep the console readable: progress and problems only
            if line.startswith(("[dataset]", "[gpu]", "[build", "[launch]", "Traceback", "Error", "loop length")) \
                    or "Error" in line or "error:" in line:
                print(line.rstrip(), flush=True)
        return proc.wait()
    finally:
        if log:
            log.close()

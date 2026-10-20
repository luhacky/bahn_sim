"""Obstacle test cases for the YOLO benchmark: procedural proxies placed at
exact arc lengths on (or next to) the running track.

    from sbahn_sim import obstacles
    ob = obstacles.place(tp, obstacles.ObstacleSpec("person", s=1100.0, lateral=2.0, yaw_deg=90))
    obstacles.CLASSES          # YOLO class id -> name (also written to dataset.yaml)

Every obstacle is one mesh object in the "Obstacles" collection with custom
properties used by the labeller:

    sbahn_obstacle = True, class_id, class_name, sbahn_s, sbahn_lateral, sbahn_height
    pass_index     = unique per obstacle (Cycles Object Index pass)

Obstacles are modelled facing +Y (towards the train when yaw_deg = 180) with
the origin on the ground under their centre, like the rest of the package
(local X right, Y forward, Z up; see track_path.TrackPath.matrix).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import bpy
import numpy as np

from . import util

COLLECTION = "Obstacles"

# YOLO class ids. Keep stable: dataset.yaml and the metadata use them.
CLASSES: Dict[int, str] = {
    0: "person",
    1: "car",
    2: "fallen_tree",
    3: "branches",
    4: "animal",
}
CLASS_ID = {v: k for k, v in CLASSES.items()}

# Heights (m above rail top, along the upright frame) at which things stand.
# Rail top is 0; sleeper tops / ballast between the rails are ~0.17 m lower,
# a level-crossing deck is level with the rail head.
GROUND_BETWEEN_RAILS = -0.17
GROUND_DECK = -0.012


@dataclass
class ObstacleSpec:
    kind: str                       # person / car / fallen_tree / branches / animal
    s: float                        # arc length of the obstacle origin (m)
    lateral: float = 2.0            # m to the right of travel (+2.0 = running track)
    yaw_deg: float = 0.0            # rotation about Z; 0 = modelled +Y along the running direction
    height: Optional[float] = None  # m above rail top; None = default for the kind
    variant: str = ""               # e.g. person: standing / walking / worker; car colour name
    seed: int = 0
    name: str = ""
    note: str = ""                  # free text for the metadata (e.g. "stalled on Crossing_00")

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# small mesh primitives (numpy, closed)
# --------------------------------------------------------------------------
def ellipsoid(center, radii, rings=8, segments=12):
    """Closed UV ellipsoid, (verts, faces)."""
    cx, cy, cz = center
    rx, ry, rz = radii
    V = [(cx, cy, cz - rz)]
    for i in range(1, rings):
        th = math.pi * i / rings - math.pi / 2
        for j in range(segments):
            ph = 2 * math.pi * j / segments
            V.append((cx + rx * math.cos(th) * math.cos(ph), cy + ry * math.cos(th) * math.sin(ph), cz + rz * math.sin(th)))
    V.append((cx, cy, cz + rz))
    F = []
    top = len(V) - 1
    for j in range(segments):
        F.append([0, 1 + (j + 1) % segments, 1 + j])
    for i in range(rings - 2):
        a, b = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            jn = (j + 1) % segments
            F.append([a + j, a + jn, b + jn, b + j])
    last = 1 + (rings - 2) * segments
    for j in range(segments):
        F.append([last + j, last + (j + 1) % segments, top])
    return np.array(V), F


def capped_cylinder(p0, p1, radius, segments=10, radius1=None):
    v, f = util.cylinder(p0, p1, radius, segments, radius1)
    f = [list(x) for x in f]
    n = segments
    f.append(list(range(n))[::-1])
    f.append(list(range(n, 2 * n)))
    return v, f


def prism(profile_yz, width, x0=0.0):
    """Extrude a closed (y, z) polygon along X over [x0 - w/2, x0 + w/2] (car bodies)."""
    p = np.asarray(profile_yz, dtype=np.float64)
    n = len(p)
    left = np.column_stack([np.full(n, x0 - width / 2), p])
    right = np.column_stack([np.full(n, x0 + width / 2), p])
    V = np.concatenate([left, right])
    F = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    F.append(list(range(n))[::-1])
    F.append(list(range(n, 2 * n)))
    return V, F


def _mat(name, color, rough=0.6, metallic=0.0, emission=0.0, transmission=0.0):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    if emission:
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = emission
    mat.diffuse_color = color
    return mat


CAR_COLOURS = {
    "red": (0.45, 0.03, 0.02, 1), "blue": (0.03, 0.08, 0.30, 1), "silver": (0.55, 0.56, 0.58, 1),
    "white": (0.80, 0.80, 0.78, 1), "black": (0.02, 0.02, 0.02, 1), "yellow": (0.70, 0.50, 0.02, 1),
}
CLOTHES = [(0.05, 0.06, 0.10, 1), (0.25, 0.04, 0.03, 1), (0.10, 0.18, 0.08, 1), (0.30, 0.28, 0.25, 1),
           (0.02, 0.02, 0.02, 1), (0.35, 0.20, 0.08, 1)]
SKIN = [(0.62, 0.45, 0.35, 1), (0.40, 0.26, 0.17, 1), (0.22, 0.13, 0.08, 1), (0.75, 0.58, 0.47, 1)]


def _band(center_xy, radii_xy, z0, z1, segments=14):
    """Open elliptic band (no caps) for reflective stripes wrapped round a body part."""
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.column_stack([center_xy[0] + radii_xy[0] * np.cos(ang), center_xy[1] + radii_xy[1] * np.sin(ang)])
    V = np.concatenate([np.column_stack([ring, np.full(segments, z0)]), np.column_stack([ring, np.full(segments, z1)])])
    i = np.arange(segments); j = (i + 1) % segments
    return V, [[a, b, b + segments, a + segments] for a, b in zip(i, j)]


def _limb_band(p0, p1, r0, r1, t, width):
    """Short ring round a tapered limb from p0 to p1 at fraction t (radius slightly proud of the limb)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a, b = p0 + (p1 - p0) * (t - width / 2), p0 + (p1 - p0) * (t + width / 2)
    r = (r0 + (r1 - r0) * t) * 1.06
    v, f = util.cylinder(a, b, r, 12)
    return v, [list(x) for x in f]


def _fabric(name, color, rough=0.85, sheen=0.3):
    """Cloth: colour with fold / weave variation and a little sheen."""
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = _mat(name, color, rough)
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 18.0
    nz.inputs["Detail"].default_value = 3.0
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 0.35
    mix.inputs[6].default_value = color
    nt.links.new(nz.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    for key in ("Sheen Weight", "Sheen"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = sheen
            break
    return mat


# --------------------------------------------------------------------------
# builders: each returns (verts, faces, mat_index list, materials)
# --------------------------------------------------------------------------
def build_car(rng, variant=""):
    """Compact hatchback (~4.25 x 1.80 x 1.47 m), front towards +Y."""
    col = CAR_COLOURS.get(variant) or list(CAR_COLOURS.values())[rng.integers(len(CAR_COLOURS))]
    mats = [_mat(f"obst_car_paint_{variant or 'x'}_{int(col[0]*100)}_{int(col[2]*100)}", col, 0.25, 0.5),
            _mat("obst_car_glass", (0.02, 0.025, 0.03, 1), 0.05),
            _mat("obst_tyre", (0.015, 0.015, 0.015, 1), 0.9),
            _mat("obst_car_trim", (0.05, 0.05, 0.05, 1), 0.5),
            _mat("obst_car_light", (0.9, 0.9, 0.85, 1), 0.2),
            _mat("obst_car_tail", (0.5, 0.02, 0.02, 1), 0.3)]
    L, W = 4.25, 1.80
    parts = []
    # lower body: side profile (y, z)
    lower = [(-L / 2, 0.30), (L / 2, 0.30), (L / 2, 0.62), (L / 2 - 0.35, 0.92), (-L / 2 + 0.05, 0.98), (-L / 2, 0.85)]
    v, f = prism(lower, W); parts.append((v, f, 0))
    # greenhouse
    gh = [(-L / 2 + 0.15, 0.97), (L / 2 - 0.55, 0.93), (L / 2 - 1.55, 1.44), (-L / 2 + 0.45, 1.46)]
    v, f = prism(gh, W - 0.16); parts.append((v, f, 1))
    # roof and pillars in body colour over the glass
    roof = [(-L / 2 + 0.45, 1.43), (L / 2 - 1.55, 1.41), (L / 2 - 1.62, 1.47), (-L / 2 + 0.48, 1.47)]
    v, f = prism(roof, W - 0.14); parts.append((v, f, 0))
    for y0 in (L / 2 - 0.62, -L / 2 + 0.2):      # A / C pillars
        v, f = util.box((W - 0.12, 0.10, 0.48), (0, y0 - (0.45 if y0 > 0 else -0.12), 1.2)); parts.append((v, f, 0))
    # wheels
    for x in (-W / 2 + 0.12, W / 2 - 0.12):
        for y in (-L / 2 + 0.72, L / 2 - 0.80):
            v, f = capped_cylinder((x - 0.11, y, 0.31), (x + 0.11, y, 0.31), 0.31, 16); parts.append((v, f, 2))
    # bumpers, lights
    v, f = util.box((W + 0.02, 0.12, 0.22), (0, L / 2 - 0.02, 0.40)); parts.append((v, f, 3))
    v, f = util.box((W + 0.02, 0.12, 0.22), (0, -L / 2 + 0.02, 0.40)); parts.append((v, f, 3))
    for x in (-0.62, 0.62):
        v, f = util.box((0.35, 0.05, 0.12), (x, L / 2 + 0.01, 0.72)); parts.append((v, f, 4))
        v, f = util.box((0.30, 0.05, 0.12), (x, -L / 2 - 0.01, 0.80)); parts.append((v, f, 5))
    return parts, mats


def build_person(rng, variant=""):
    """1.60-1.90 m human. variants: standing (facing +Y), walking, worker (DB hi-vis), child.

    The worker wears what DB track workers wear: fluorescent orange jacket and trousers with
    silver retro-reflective bands (EN ISO 20471 class 3) and a white safety helmet."""
    variant = variant or ["standing", "walking", "worker"][rng.integers(3)]
    h = float(rng.uniform(1.62, 1.90)) if variant != "child" else float(rng.uniform(1.10, 1.35))
    k = h / 1.75
    worker = variant == "worker"
    hivis = (0.95, 0.20, 0.015, 1)
    cloth = hivis if worker else CLOTHES[rng.integers(len(CLOTHES))]
    top = hivis if worker else CLOTHES[rng.integers(len(CLOTHES))]
    skin = SKIN[rng.integers(len(SKIN))]
    mats = [_fabric(f"obst_cloth_{int(cloth[0]*100)}_{int(cloth[1]*100)}", cloth, 0.85),
            _fabric(f"obst_top_{int(top[0]*100)}_{int(top[1]*100)}", top, 0.75),
            _mat(f"obst_skin_{int(skin[0]*100)}", skin, 0.55),
            _mat("obst_hair", (0.03, 0.02, 0.015, 1), 0.6),
            _mat("obst_reflective", (0.62, 0.63, 0.64, 1), 0.35, 0.4),
            _mat("obst_helmet", (0.85, 0.85, 0.82, 1), 0.3)]
    parts = []
    stride = 0.28 * k if variant == "walking" else 0.0
    arm_swing = 0.22 * k if variant == "walking" else 0.0
    hip = 0.93 * k
    for sx, sy in ((-0.1, stride), (0.1, -stride)):          # legs
        p0, p1 = (sx * k, sy, 0.05), (sx * k * 0.9, 0, hip)
        v, f = capped_cylinder(p0, p1, 0.075 * k, 10, 0.085 * k); parts.append((v, f, 0))
        v, f = util.box((0.11 * k, 0.26 * k, 0.09), (sx * k, sy + 0.05, 0.045)); parts.append((v, f, 3))   # shoes
        if worker:                                            # two reflective bands round each leg
            for t in (0.28, 0.42):
                v, f = _limb_band(p0, p1, 0.075 * k, 0.085 * k, t, 0.055); parts.append((v, f, 4))
    tz, trz = hip + 0.30 * k, 0.36 * k
    v, f = ellipsoid((0, 0, tz), (0.21 * k, 0.13 * k, trz), 8, 14); parts.append((v, f, 1))   # torso
    if worker:                                                # chest and waist bands
        for zc in (tz - 0.12 * k, tz + 0.12 * k):
            fac = math.sqrt(max(1 - ((zc - tz) / trz) ** 2, 0.05)) * 1.035
            v, f = _band((0, 0), (0.21 * k * fac, 0.13 * k * fac), zc - 0.03 * k, zc + 0.03 * k); parts.append((v, f, 4))
    shoulder = hip + 0.52 * k
    for sx, sy in ((-0.24, -arm_swing), (0.24, arm_swing)):  # arms
        p0, p1 = (sx * k, 0, shoulder), (sx * k * 1.08, sy, shoulder - 0.62 * k)
        v, f = capped_cylinder(p0, p1, 0.052 * k, 8, 0.042 * k); parts.append((v, f, 1))
        v, f = ellipsoid((sx * k * 1.08, sy, shoulder - 0.66 * k), (0.045 * k, 0.045 * k, 0.07 * k), 6, 8); parts.append((v, f, 2))
        if worker:
            v, f = _limb_band(p0, p1, 0.052 * k, 0.042 * k, 0.55, 0.08); parts.append((v, f, 4))
    v, f = capped_cylinder((0, 0, shoulder - 0.02), (0, 0, shoulder + 0.10 * k), 0.055 * k, 8); parts.append((v, f, 2))  # neck
    v, f = ellipsoid((0, 0.01, shoulder + 0.2 * k), (0.095 * k, 0.11 * k, 0.125 * k), 8, 12); parts.append((v, f, 2))    # head
    if worker:                                                # safety helmet with a brim
        v, f = ellipsoid((0, 0.0, shoulder + 0.25 * k), (0.125 * k, 0.14 * k, 0.11 * k), 6, 14); parts.append((v, f, 5))
        v, f = ellipsoid((0, 0.03, shoulder + 0.22 * k), (0.14 * k, 0.17 * k, 0.018 * k), 4, 14); parts.append((v, f, 5))
    else:
        v, f = ellipsoid((0, -0.015, shoulder + 0.24 * k), (0.10 * k, 0.11 * k, 0.10 * k), 6, 12); parts.append((v, f, 3))  # hair
    return parts, mats, variant


def build_fallen_tree(rng, variant=""):
    """Trunk lying along local X (across the track when yaw = 0), crown towards +X."""
    L = float(rng.uniform(9.0, 14.0))
    r = float(rng.uniform(0.18, 0.30))
    mats = [_mat("obst_bark", (0.20, 0.15, 0.10, 1), 0.9),
            _mat("obst_leaves_fallen", (0.08, 0.16, 0.04, 1), 0.75),
            _mat("obst_wood_cut", (0.55, 0.42, 0.26, 1), 0.8)]
    x0 = -L * 0.35                                              # root end on the -X side
    parts = []
    v, f = capped_cylinder((x0, 0, r), (x0 + L, 0, r * 0.5), r, 14, r * 0.25); parts.append((v, f, 0))
    v, f = ellipsoid((x0 - 0.05, 0, r), (0.08, r * 1.3, r * 1.3), 4, 10); parts.append((v, f, 2))   # broken end
    for i in range(int(rng.integers(6, 10))):                  # branches from the upper half
        t = float(rng.uniform(0.35, 0.95))
        px = x0 + t * L
        pr = r * (1 - 0.7 * t)
        ang = float(rng.uniform(-1.2, 1.2)) + (0 if i % 2 else math.pi)
        up = float(rng.uniform(0.2, 1.3))
        bl = float(rng.uniform(1.5, 3.5)) * (1.2 - t * 0.5)
        end = (px + bl * 0.6, math.sin(ang) * bl, max(0.1, r + up * bl * 0.5))
        v, f = capped_cylinder((px, 0, pr + 0.02), end, max(0.03, pr * 0.45), 7, 0.02); parts.append((v, f, 0))
        v, f = ellipsoid(end, (0.9, 0.8, 0.6), 5, 8)
        v = v + rng.normal(0, 0.08, v.shape); parts.append((v, f, 1))
    for i in range(3):                                          # crown mass
        c = (x0 + L * (0.85 + 0.08 * i), float(rng.uniform(-1, 1)), 0.9 + 0.3 * i)
        v, f = ellipsoid(c, (1.6, 1.5, 1.0), 6, 10)
        v = v + rng.normal(0, 0.12, v.shape); parts.append((v, f, 1))
    return parts, mats


def build_branches(rng, variant=""):
    """Pile of broken branches (~3 x 2 x 1 m) lying on the track."""
    mats = [_mat("obst_bark", (0.20, 0.15, 0.10, 1), 0.9),
            _mat("obst_leaves_fallen", (0.08, 0.16, 0.04, 1), 0.75)]
    parts = []
    for i in range(int(rng.integers(7, 12))):
        a = rng.uniform(-1.6, 1.6, 3) * np.array([1.0, 0.8, 0]) + np.array([0, 0, rng.uniform(0.05, 0.4)])
        d = rng.normal(0, 1, 3); d[2] *= 0.3; d /= np.linalg.norm(d)
        b = a + d * rng.uniform(1.0, 2.2)
        b[2] = max(0.05, b[2])
        v, f = capped_cylinder(a, b, float(rng.uniform(0.03, 0.07)), 6, 0.015); parts.append((v, f, 0))
    for i in range(int(rng.integers(3, 6))):
        c = (float(rng.uniform(-1.2, 1.2)), float(rng.uniform(-0.8, 0.8)), float(rng.uniform(0.3, 0.6)))
        v, f = ellipsoid(c, (0.6, 0.5, 0.4), 5, 8)
        v = v + rng.normal(0, 0.06, v.shape); parts.append((v, f, 1))
    return parts, mats


def build_animal(rng, variant=""):
    """Roe deer (Reh), ~1.2 m long, 0.75 m at the shoulder, facing +X (broadside to the train at yaw 0)."""
    mats = [_mat("obst_deer_fur", (0.30, 0.17, 0.08, 1), 0.85),
            _mat("obst_deer_dark", (0.05, 0.04, 0.03, 1), 0.7),
            _mat("obst_deer_white", (0.75, 0.72, 0.65, 1), 0.8)]
    parts = []
    v, f = ellipsoid((0, 0, 0.80), (0.55, 0.20, 0.22), 8, 12); parts.append((v, f, 0))            # body
    for x in (-0.38, 0.38):
        for y in (-0.08, 0.08):
            v, f = capped_cylinder((x, y, 0.78), (x + 0.03, y, 0.02), 0.035, 6, 0.02); parts.append((v, f, 0))
    v, f = capped_cylinder((0.45, 0, 0.88), (0.62, 0, 1.18), 0.07, 8, 0.06); parts.append((v, f, 0))  # neck
    v, f = ellipsoid((0.70, 0, 1.20), (0.16, 0.07, 0.08), 6, 10); parts.append((v, f, 0))          # head
    v, f = ellipsoid((0.84, 0, 1.18), (0.03, 0.03, 0.03), 4, 6); parts.append((v, f, 1))           # nose
    for y in (-0.05, 0.05):
        v, f = ellipsoid((0.62, y * 1.6, 1.30), (0.03, 0.02, 0.07), 4, 6); parts.append((v, f, 1))  # ears
    v, f = ellipsoid((-0.55, 0, 0.82), (0.06, 0.13, 0.12), 5, 8); parts.append((v, f, 2))          # rump patch
    return parts, mats


def build_car_detailed(rng, variant=""):
    """Detailed hatchback from sbahn_sim.models (detailed model library); proxy fallback."""
    try:
        from .models import vehicles
    except ImportError:
        return build_car(rng, variant)
    kind = "hatchback"
    paint = variant if variant in vehicles.CAR_COLOURS else None
    if variant and ":" in variant:                    # e.g. "suv:blue"
        kind, paint = variant.split(":", 1)
        paint = paint if paint in vehicles.CAR_COLOURS else None
    if paint is None:
        paint = list(vehicles.CAR_COLOURS)[rng.integers(len(vehicles.CAR_COLOURS))]
    return vehicles.obstacle_parts(kind, rng, paint)


def build_fallen_tree_detailed(rng, variant=""):
    """Detailed wind-thrown tree from sbahn_sim.models, root end shifted to x = -0.3 L like the proxy."""
    try:
        from .models import library
    except ImportError:
        return build_fallen_tree(rng, variant)
    parts, mats = library.fallen_tree_parts(rng, species=variant or "spruce", length=float(rng.uniform(11.0, 15.0)))
    xmax = max(float(np.asarray(v)[:, 0].max()) for v, f, m in parts)
    shifted = []
    for v, f, m in parts:
        v = np.asarray(v, dtype=np.float64).copy()
        v[:, 0] -= 0.3 * xmax
        shifted.append((v, f, m))
    return shifted, mats


BUILDERS = {"car": build_car_detailed, "person": build_person, "fallen_tree": build_fallen_tree_detailed,
            "branches": build_branches, "animal": build_animal,
            "car_proxy": build_car, "fallen_tree_proxy": build_fallen_tree}
DEFAULT_HEIGHT = {"car": GROUND_DECK, "person": GROUND_BETWEEN_RAILS, "fallen_tree": GROUND_BETWEEN_RAILS,
                  "branches": GROUND_BETWEEN_RAILS, "animal": GROUND_BETWEEN_RAILS,
                  "car_proxy": GROUND_DECK, "fallen_tree_proxy": 0.0}


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------
def collection():
    return util.ensure_collection(COLLECTION)


def clear():
    """Remove every obstacle object (meshes are removed with them)."""
    col = bpy.data.collections.get(COLLECTION)
    if not col:
        return
    for ob in list(col.objects):
        me = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if me and me.users == 0:
            bpy.data.meshes.remove(me)


def all_obstacles() -> List[bpy.types.Object]:
    return [o for o in bpy.data.objects if o.get("sbahn_obstacle")]


def place(tp, spec: ObstacleSpec, index: Optional[int] = None) -> bpy.types.Object:
    """Build the proxy for spec and put it at (s, lateral, height) in an upright track frame."""
    if spec.kind not in BUILDERS:
        raise ValueError(f"unknown obstacle kind {spec.kind!r}; one of {sorted(BUILDERS)}")
    rng = np.random.default_rng(spec.seed)
    out = BUILDERS[spec.kind](rng, spec.variant)
    if spec.kind == "person":
        parts, mats, spec.variant = out
    else:
        parts, mats = out
    V, F, M = util.merge(parts)
    existing = all_obstacles()
    index = index if index is not None else len(existing)
    name = spec.name or f"Obst_{index:02d}_{spec.kind}"
    spec.name = name
    ob = util.new_mesh_object(name, V, F, collection(), mats, M)
    for poly in ob.data.polygons:
        poly.use_smooth = spec.kind in ("person", "animal", "car", "fallen_tree")
    h = DEFAULT_HEIGHT[spec.kind] if spec.height is None else spec.height
    ob.matrix_world = tp.matrix(spec.s, lateral=spec.lateral, height=h, yaw=math.radians(spec.yaw_deg), upright=True)
    ob["sbahn_obstacle"] = True
    cls = spec.kind.replace("_proxy", "")
    ob["class_id"] = CLASS_ID[cls]
    ob["class_name"] = cls
    ob["sbahn_s"] = float(tp.wrap(spec.s))
    ob["sbahn_lateral"] = float(spec.lateral)
    ob["sbahn_height"] = float(h)
    ob["variant"] = spec.variant
    ob["note"] = spec.note
    ob.pass_index = 100 + index
    return ob


def set_active(names) -> None:
    """Show only the obstacles whose names are in `names` (None = all)."""
    for ob in all_obstacles():
        on = names is None or ob.name in names
        ob.hide_render = not on
        ob.hide_viewport = not on


# --------------------------------------------------------------------------
# default test cases for the stadium template (from scene_manifest.json)
# --------------------------------------------------------------------------
def default_cases(manifest: dict, running_lateral: float = 2.0) -> List[dict]:
    """One test case per obstacle: {'name', 'specs': [ObstacleSpec], 'target_s'}.

    Placed where the manifest says there is sight distance: the two long
    straights host the 300 / 500 m cases, one case sits in an R 750 curve to
    cover the "leaves the windscreen" regime, and the car sits on a level
    crossing as the brief asks."""
    L = manifest["length_m"]
    straights = sorted(manifest["straights"], key=lambda s: -s["length_m"])
    arcs = sorted([sg for sg in manifest["segments"] if sg["kind"] == "arc"], key=lambda sg: -(sg["s1"] - sg["s0"]))
    lat = running_lateral
    cases = []
    for i, cr in enumerate(manifest.get("crossings", [])[:2]):
        cases.append(dict(name=f"car_on_{cr['name'].lower()}", specs=[
            ObstacleSpec("car", s=cr["s"], lateral=lat + (0.4 if i == 0 else -0.3), yaw_deg=90 if i == 0 else -90,
                         variant=["red", "silver"][i % 2], seed=11 + i, note=f"stalled on {cr['name']}")]))
    A, B = straights[0], straights[1] if len(straights) > 1 else straights[0]
    cases.append(dict(name="person_standing_on_track", specs=[
        ObstacleSpec("person", s=A["s1"] - 30.0, lateral=lat, yaw_deg=180, variant="standing", seed=21,
                     note="person facing the train between the rails")]))
    cases.append(dict(name="person_crossing_track", specs=[
        ObstacleSpec("person", s=B["s0"] + 700.0, lateral=lat + 0.6, yaw_deg=-90, variant="walking", seed=22,
                     note="person walking across the running track")]))
    cases.append(dict(name="worker_on_opposite_track", specs=[
        ObstacleSpec("person", s=B["s0"] + 560.0, lateral=-lat, yaw_deg=0, variant="worker", seed=23,
                     note="track worker in hi-vis on the opposing track")]))
    cases.append(dict(name="fallen_tree_across_tracks", specs=[
        ObstacleSpec("fallen_tree", s=B["s1"] - 40.0, lateral=lat, yaw_deg=8, seed=31,
                     note="trunk across both tracks, crown to the right")]))
    if arcs:
        a = arcs[0]
        cases.append(dict(name="branches_in_curve", specs=[
            ObstacleSpec("branches", s=a["s0"] + 0.6 * (a["s1"] - a["s0"]), lateral=lat, yaw_deg=20, seed=41,
                         note=f"branch pile in an R {a['radius']:.0f} m curve")]))
    cases.append(dict(name="deer_on_track", specs=[
        ObstacleSpec("animal", s=A["s0"] + 900.0 if A["length_m"] > 950 else A["s1"] - 60, lateral=lat + 0.3,
                     yaw_deg=0, variant="roe_deer", seed=51, note="roe deer broadside on the running track")]))
    for c in cases:
        for sp in c["specs"]:
            sp.s = float(sp.s % L)
    return cases

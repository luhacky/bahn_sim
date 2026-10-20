"""Camera realism for the cab camera: motion blur over the shutter interval plus a lens / sensor model.

A clean path-traced frame differs from real cab-camera footage in a few systematic ways. Each one is
modelled here with conservative parameters that are logged per frame (meta["camera_fx"]):

* exposure: the shutter time follows the auto-exposure push of the conditions (about 1/400 s in
  sunshine, up to 1/30 s at dusk and at night); whatever the shutter cannot cover becomes sensor gain
* motion blur: from the depth pass and the camera travel during the exposure (near sleepers and
  ballast smear, distant scenery stays sharp; the cab itself moves with the camera and stays sharp)
* optics: slight softness (Gaussian PSF, softer towards the corners), lateral chromatic aberration,
  veiling glare (lifts the blacks), bloom around bright areas, natural vignetting
* sensor and in-camera processing: Poisson-Gaussian noise in linear light scaled by the gain, partly
  coloured and slightly correlated (demosaicing), unsharp-mask sharpening, JPEG compression

Optional camera artifacts can be switched on per condition (key "camera_artifacts", see ARTIFACTS):
lens flare from the sun, dirt and insects on the windscreen, wiper rain smears, a smudged lens, defocus,
exposure errors, extra sensor gain, hot pixels and heavy compression.

There is no geometric distortion, so the pinhole labels stay valid; the chromatic aberration moves the
red and blue channels by less than one pixel at the image corners.
"""
from __future__ import annotations

import math
import os
import zlib
from dataclasses import asdict, dataclass

import bpy
import numpy as np


@dataclass
class CameraFX:
    enabled: bool = True
    motion_blur: bool = True
    base_shutter_s: float = 1 / 400.0     # sunny daylight
    max_shutter_s: float = 1 / 30.0       # video camera limit
    psf_sigma_px: float = 0.55            # optical softness in the image centre (at 720 lines)
    psf_sigma_edge_px: float = 1.0        # ... and in the corners
    chroma_aberration: float = 0.0012     # relative magnification difference between red and blue
    veiling_glare: float = 0.025          # share of the (very blurred) image added as lens flare
    bloom: float = 0.35
    bloom_threshold: float = 0.80         # display value above which light blooms
    vignette: float = 0.40                # share of the natural cos^4 falloff left by the lens design
    shot_noise: float = 1 / 5000.0        # noise variance per unit linear signal at base gain
    read_noise: float = 0.0008            # linear noise sigma at base gain
    chroma_noise: float = 0.45            # share of per-channel (colour) noise
    sharpen: float = 0.40                 # in-camera unsharp mask
    jpeg_quality: int = 90                # 0 = keep lossless
    ref_height_px: int = 720              # PSF and blur sizes scale with the image height

    def as_dict(self):
        return asdict(self)


# ------------------------------------------------------------------------------------- exposure
def exposure_from_conditions(fx: CameraFX, eff: dict) -> dict:
    """Shutter time and gain for the auto-exposure push the conditions applied (eff['exposure_ev'])."""
    push = max(float((eff or {}).get("exposure_ev", 0.0) or 0.0), 0.0)
    shutter = min(fx.base_shutter_s * 2.0 ** push, fx.max_shutter_s)
    gain_ev = max(push - math.log2(shutter / fx.base_shutter_s), 0.0)
    el = (eff or {}).get("sun_elevation_deg")
    if el is not None:
        # The conditions keep dusk and night dark on purpose; a real camera would already be at its
        # gain limit there, which shows as noise rather than brightness.
        gain_ev += float(np.interp(float(el), [-12.0, -4.0, 3.0], [3.0, 1.5, 0.0]))
    return dict(shutter_s=round(shutter, 6), gain_ev=round(gain_ev, 3))


# ---------------------------------------------------------------------------------- motion blur
# Cycles' own motion blur re-synchronises every scatter instance at each motion step, which more than
# doubled the render time. The world is static and only the camera moves, so the blur is computed from
# the depth pass instead: a point at depth D moves by (-(x-cx) Tz - fx Tx) / D, (-(y-cy) Tz + fy Ty) / D
# pixels while the camera travels T (camera coordinates) during the exposure.
DEPTH_NODE = "SBahnDepthOut"
NEAR_LIMIT_M = 3.0          # closer surfaces belong to the cab (they move with the camera): no blur


def ensure_depth_output(scene, directory: str):
    """Write the depth pass of every render to <directory>/depth_<frame>.exr via the compositor."""
    for vl in scene.view_layers:
        vl.use_pass_z = True
    scene.use_nodes = True
    scene.render.use_compositing = True
    nt = scene.node_tree
    rl = next((n for n in nt.nodes if n.bl_idname == "CompositorNodeRLayers"), None) or nt.nodes.new("CompositorNodeRLayers")
    comp = next((n for n in nt.nodes if n.bl_idname == "CompositorNodeComposite"), None) or nt.nodes.new("CompositorNodeComposite")
    if not comp.inputs["Image"].is_linked:
        nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
    fo = nt.nodes.get(DEPTH_NODE) or nt.nodes.new("CompositorNodeOutputFile")
    fo.name = DEPTH_NODE
    fo.base_path = directory
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    fo.file_slots[0].path = "depth_"
    nt.links.new(rl.outputs["Depth"], fo.inputs[0])
    os.makedirs(directory, exist_ok=True)


def depth_file(scene, directory: str) -> str:
    return os.path.join(directory, f"depth_{scene.frame_current:04d}.exr")


def camera_travel(scene, tp, s_front: float, speed_mps: float, shutter_s: float):
    """Camera translation during the exposure, in camera coordinates (x right, y up, -z forward)."""
    t = np.asarray(tp.sample(s_front)["tangent"]).reshape(-1)[:3]
    R = np.array(scene.camera.matrix_world)[:3, :3]
    R = R / np.linalg.norm(R, axis=0)
    return (R.T @ (t / np.linalg.norm(t) * speed_mps * shutter_s)).tolist()


def _motion_blur(lin, depth, T, intr, max_px=80.0):
    h, w = depth.shape
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    D = np.where((depth > NEAR_LIMIT_M) & (depth < 1e6), depth, np.inf).astype(np.float32)
    dx = (-(xx - cx) * T[2] - fx * T[0]) / D
    dy = (-(yy - cy) * T[2] + fy * T[1]) / D
    dx, dy = np.clip(dx, -max_px, max_px), np.clip(dy, -max_px, max_px)
    n = int(np.clip(np.ceil(np.percentile(np.hypot(dx, dy), 99.5) / 1.5), 1, 32))
    if n <= 1:
        return lin, float(np.hypot(dx, dy).max())
    acc = np.zeros_like(lin)
    for t in np.linspace(-0.5, 0.5, n):
        x0, x1, u = _axis_interp(w, xx + t * dx)
        y0, y1, v = _axis_interp(h, yy + t * dy)
        u, v = u[..., None], v[..., None]
        acc += ((lin[y0, x0] * (1 - u) + lin[y0, x1] * u) * (1 - v)
                + (lin[y1, x0] * (1 - u) + lin[y1, x1] * u) * v)
    return acc / n, float(np.hypot(dx, dy).max())


def _read_depth(path):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except TypeError:
        pass
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)[::-1, :, 0].copy()


# ----------------------------------------------------------------------------- image helpers
def _srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _kernel(sigma):
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32), r


def _blur(img, sigma):
    """Separable Gaussian blur, edge padding; img (H, W) or (H, W, C)."""
    if sigma < 0.05:
        return img
    k, r = _kernel(sigma)
    pad_y = ((r, r), (0, 0)) + (((0, 0),) if img.ndim == 3 else ())
    p = np.pad(img, pad_y, mode="edge")
    out = np.zeros_like(img)
    for i in range(2 * r + 1):
        out += k[i] * p[i:i + img.shape[0]]
    pad_x = ((0, 0), (r, r)) + (((0, 0),) if img.ndim == 3 else ())
    p = np.pad(out, pad_x, mode="edge")
    out = np.zeros_like(img)
    for i in range(2 * r + 1):
        out += k[i] * p[:, i:i + img.shape[1]]
    return out


def _axis_interp(n_in, coords):
    x = np.clip(coords, 0, n_in - 1)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, (x - i0).astype(np.float32)


def _resize(img, h, w):
    """Bilinear resize of (H, W, C)."""
    i0, i1, t = _axis_interp(img.shape[0], (np.arange(h) + 0.5) * img.shape[0] / h - 0.5)
    img = img[i0] * (1 - t)[:, None, None] + img[i1] * t[:, None, None]
    j0, j1, u = _axis_interp(img.shape[1], (np.arange(w) + 0.5) * img.shape[1] / w - 0.5)
    return img[:, j0] * (1 - u)[None, :, None] + img[:, j1] * u[None, :, None]


def _wide_blur(img, factor=16, sigma=3.0):
    h, w = img.shape[:2]
    h2, w2 = max(h // factor, 1), max(w // factor, 1)
    small = img[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor, -1).mean(axis=(1, 3))
    return _resize(_blur(small, sigma), h, w)


def _magnify(ch, s):
    """Scale one channel (H, W) about the image centre by s (bilinear)."""
    h, w = ch.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    i0, i1, t = _axis_interp(h, (np.arange(h) - cy) / s + cy)
    ch = ch[i0] * (1 - t)[:, None] + ch[i1] * t[:, None]
    j0, j1, u = _axis_interp(w, (np.arange(w) - cx) / s + cx)
    return ch[:, j0] * (1 - u)[None, :] + ch[:, j1] * u[None, :]


def _read(path):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except TypeError:
        pass
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)[::-1, :, :3].copy()      # Blender stores rows bottom-up


def _write(path, rgb, fmt, quality=90):
    h, w = rgb.shape[:2]
    img = bpy.data.images.new("_sbahn_fx", w, h, alpha=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except TypeError:
        pass
    px = np.ones((h, w, 4), dtype=np.float32)
    px[..., :3] = rgb[::-1]
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = fmt
    try:
        img.save(quality=quality) if fmt == "JPEG" else img.save()
    except TypeError:                     # older API without the quality argument
        img.save()
    bpy.data.images.remove(img)


# ------------------------------------------------------------------------------ camera artifacts
# Optional degradations, switched on per condition with the key "camera_artifacts", e.g.
#   {"name": "sunset_flare", "preset": "sunset", "camera_artifacts": {"lens_flare": 1.0, "windscreen_dirt": 0.5}}
# Strengths are 0..1 unless noted. None of them moves geometry, so the labels stay valid (an obstacle
# hidden behind a smear or a flare keeps its box, like one hidden in fog; check visible_px and the
# logged artifacts when scoring).
ARTIFACTS = {
    "lens_flare": "ghosts mirrored through the image centre, halo and a 6-ray starburst from the sun "
                  "(when it is in or near the frame and not behind trees) and other clipped highlights",
    "windscreen_dirt": "dust film, out-of-focus specks and insect splats on the glass; they darken the view "
                       "and glow when the light (low sun) comes towards the camera",
    "rain_smear": "wiper arcs of smeared water: local blur, refraction and scattered light, plus residual drops",
    "lens_smudge": "grease / fingerprint on the lens: soft veiling glow over part of the frame and streaks "
                   "around bright light",
    "defocus_px": "extra lens blur sigma in pixels (at 720 lines), e.g. a misadjusted or fogged-up camera",
    "exposure_bias_ev": "auto-exposure error in EV (+1 = one stop over-exposed, -1 = under)",
    "sensor_gain_ev": "extra sensor gain in EV (more noise), e.g. a cheap or ageing sensor",
    "hot_pixels": "number of stuck bright pixels",
    "jpeg_quality": "stronger compression than the default 90 (e.g. 40)",
}


def sun_in_camera(scene, intr: dict) -> dict:
    """Where the sun is relative to the cab camera: pixel position (may lie outside the frame), angle from
    the optical axis and elevation. Uses the 'Sun' lamp (its +Z axis points at the sun)."""
    so, cam = bpy.data.objects.get("Sun"), scene.camera
    if so is None or cam is None or so.hide_render:
        return {}
    d = np.array(so.matrix_world)[:3, 2]
    d = d / np.linalg.norm(d)
    R = np.array(cam.matrix_world)[:3, :3]
    R = R / np.linalg.norm(R, axis=0)
    dc = R.T @ d
    out = dict(sun_angle_from_axis_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, -dc[2])))), 2),
               sun_elevation_deg=round(math.degrees(math.asin(max(-1.0, min(1.0, d[2])))), 2))
    if dc[2] < -1e-3:
        out["sun_px"] = [round(intr["cx"] + intr["fx"] * dc[0] / -dc[2], 1),
                         round(intr["cy"] - intr["fy"] * dc[1] / -dc[2], 1)]
    return out


def _soft_blur(img, sigma):
    """Gaussian blur that switches to a down-sampled blur for wide kernels (cheap glows)."""
    if sigma <= 6.0:
        return _blur(img, sigma)
    f = int(min(16, max(2, sigma // 3)))
    h, w = img.shape[:2]
    small = img[:h - h % f, :w - w % f]
    small = small.reshape(h // f, f, w // f, f, -1).mean(axis=(1, 3))
    return _resize(_blur(small, sigma / f), h, w)


def _resample_zero(img, s, cx, cy):
    """Scale (H, W, C) by s about (cx, cy); negative s mirrors through that point; zero outside."""
    h, w = img.shape[:2]
    ys = (np.arange(h) - cy) / s + cy
    xs = (np.arange(w) - cx) / s + cx
    i0, i1, t = _axis_interp(h, ys)
    j0, j1, u = _axis_interp(w, xs)
    out = img[i0] * (1 - t)[:, None, None] + img[i1] * t[:, None, None]
    out = out[:, j0] * (1 - u)[None, :, None] + out[:, j1] * u[None, :, None]
    vy = ((ys >= 0) & (ys <= h - 1)).astype(np.float32)
    vx = ((xs >= 0) & (xs <= w - 1)).astype(np.float32)
    return out * vy[:, None, None] * vx[None, :, None]


def _shift(img, dx, dy):
    """Integer shift with zero fill."""
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    dx, dy = int(round(dx)), int(round(dy))
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


def _value_noise(h, w, rng, cell_px):
    """Smooth random field 0..1 with features of about cell_px."""
    gh, gw = max(2, int(h / cell_px) + 2), max(2, int(w / cell_px) + 2)
    g = rng.random((gh, gw, 1)).astype(np.float32)
    return _resize(g, h, w)[..., 0]


def _impulses(h, w, rng, n, sigma, amp=(0.3, 1.0)):
    """n soft blobs of size sigma (pixels) at random positions, values up to ~1."""
    img = np.zeros((h, w, 1), np.float32)
    if n <= 0:
        return img[..., 0]
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    np.add.at(img, (ys, xs, 0), rng.uniform(*amp, n).astype(np.float32) * (2 * math.pi * sigma ** 2))
    return _blur(img, sigma)[..., 0]


MAX_GAIN_EV = 4.5            # beyond this a real camera's image is unusable anyway

FLARE_GHOSTS = (    # scale about the image centre (negative = mirrored), defocus sigma at 720 p, tint, weight
    (-0.42, 5.0, (1.0, 0.70, 0.35), 0.22),
    (-0.78, 9.0, (0.45, 0.95, 0.55), 0.14),
    (-1.20, 15.0, (0.65, 0.50, 1.00), 0.09),
    (0.30, 3.5, (1.0, 0.85, 0.55), 0.10),
    (-1.90, 24.0, (0.50, 0.75, 1.00), 0.05),
)


def _lens_flare(lin, strength, sun, depth, sc):
    h, w = lin.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    src = np.clip(lin - 0.75, 0.0, None)              # clipped highlights act as flare sources
    vis, sx, sy = 0.0, None, None
    if sun.get("sun_px") and sun.get("sun_elevation_deg", -90) > -1.0:
        sx, sy = sun["sun_px"]
        if -0.25 * w < sx < 1.25 * w and -0.25 * h < sy < 1.25 * h:
            vis = 1.0
            if 0 <= sx < w and 0 <= sy < h and depth is not None:
                # sun behind trees or masts: only the sky fraction round the disc lets it through
                y0, y1 = int(max(0, sy - 6 * sc)), int(min(h, sy + 6 * sc + 1))
                x0, x1 = int(max(0, sx - 6 * sc)), int(min(w, sx + 6 * sc + 1))
                vis = float((depth[y0:y1, x0:x1] > 1e6).mean())
            elif not (0 <= sx < w and 0 <= sy < h):
                vis = 0.5                                  # just outside the frame: veiling flare only
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if vis > 0:
        # the sun is far brighter than the clipped sky: put a hot core back at its position
        core = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * (4.0 * sc) ** 2)) * 30.0 * vis
        src = src + core[..., None] * np.array([1.0, 0.95, 0.85], np.float32)
    if src.max() <= 0:
        return lin, 0.0
    flare = np.zeros_like(lin)
    for s, sig, tint, wgt in FLARE_GHOSTS:
        flare += wgt * _soft_blur(_resample_zero(src, s, cx, cy), sig * sc) * np.array(tint, np.float32)
    flare += 0.12 * _soft_blur(src, 30.0 * sc)          # halo
    if vis > 0:
        dx, dy = xx - sx, yy - sy
        r = np.hypot(dx, dy) + 1.0
        ang = np.arctan2(dy, dx)
        burst = np.zeros((h, w), np.float32)
        for k in range(6):                               # 6 aperture blades -> 6 rays
            th = 0.35 + k * math.pi / 3.0
            dperp = r * np.abs(np.sin(ang - th))
            burst += np.exp(-(dperp / (1.3 * sc)) ** 2)
        burst *= np.clip(r / (10.0 * sc), 1.0, None) ** -1.4 * 0.9 * vis
        flare += burst[..., None] * np.array([1.0, 0.97, 0.9], np.float32)
    return lin + strength * flare, vis


def _windscreen_dirt(lin, strength, rng, sc, sun_prox):
    h, w = lin.shape[:2]
    specks = (_impulses(h, w, rng, int(1600 * strength), 1.6 * sc)
              + _impulses(h, w, rng, int(450 * strength), 4.0 * sc)
              + _impulses(h, w, rng, int(60 * strength), 11.0 * sc, (0.2, 0.6)))
    specks = np.clip(specks, 0.0, 1.0)
    film = np.clip(_value_noise(h, w, rng, 90 * sc) * 1.4 - 0.4, 0.0, 1.0) * strength
    insects = np.clip(_impulses(h, w, rng, int(14 * strength), 1.3 * sc, (0.8, 1.0)) * 1.6, 0.0, 1.0)
    light = _soft_blur(lin, 18.0 * sc)                  # light falling on the glass
    glow = 1.0 + 6.0 * sun_prox                          # forward scattering towards a low sun
    absorb = np.clip(0.55 * specks + 0.08 * film + 0.85 * insects, 0.0, 0.95)[..., None]
    lin = lin * (1 - absorb * np.array([1.0, 1.0, 1.08], np.float32))
    lin = lin + (0.35 * specks + 0.10 * film)[..., None] * light * glow * np.array([1.0, 0.96, 0.88], np.float32)
    return lin


def _rain_smear(lin, strength, rng, sc):
    """Water left by a worn wiper blade: thin arcs at random radii round the pivot (each fading in and out
    along its length), a light water film, and residual drops."""
    h, w = lin.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = 0.70 * w, 1.15 * h                          # wiper pivot below the bottom edge
    r = np.hypot(xx - px, yy - py)
    ang = np.arctan2(py - yy, xx - px)                   # 0 = right, pi/2 = up
    sector = np.clip((ang - 0.35) / 0.25, 0.0, 1.0) * np.clip((2.85 - ang) / 0.25, 0.0, 1.0)
    rmax = float(r[0].max())
    arcs = np.zeros((h, w), np.float32)
    for _ in range(int(10 + 14 * strength)):
        ri = rng.uniform(0.25, 1.0) * rmax
        wi = rng.uniform(0.6, 2.8) * sc
        nodes = np.clip(rng.normal(0.35, 0.45, 9), 0.0, 1.0)        # opacity along the arc
        along = np.interp(ang, np.linspace(0.3, 2.9, 9), nodes).astype(np.float32)
        arcs += np.exp(-((r - ri) / wi) ** 2) * along * rng.uniform(0.4, 1.0)
    film = np.clip(_value_noise(h, w, rng, 160 * sc) * 1.5 - 0.45, 0.0, 1.0) * 0.35
    m = (np.clip(arcs, 0.0, 1.0) * 0.9 + film) * sector * strength
    m = m[..., None]
    nx = (_value_noise(h, w, rng, 25 * sc) - 0.5) * 6.0 * sc
    ny = (_value_noise(h, w, rng, 25 * sc) - 0.5) * 6.0 * sc
    x0, x1, u = _axis_interp(w, xx + nx)
    y0, y1, v = _axis_interp(h, yy + ny)
    u, v = u[..., None], v[..., None]
    refr = (lin[y0, x0] * (1 - u) + lin[y0, x1] * u) * (1 - v) + (lin[y1, x0] * (1 - u) + lin[y1, x1] * u) * v
    smeared = 0.5 * _blur(lin, 2.0 * sc) + 0.5 * _blur(refr, 1.0 * sc)
    lin = lin * (1 - m) + smeared * m + 0.12 * m * _soft_blur(lin, 20.0 * sc)
    drops = np.clip(_impulses(h, w, rng, int(220 * strength), 2.0 * sc) * 1.3, 0.0, 1.0)[..., None]
    lin = lin * (1 - drops) + drops * (1.06 * _blur(lin, 4.0 * sc))
    return lin


def _lens_smudge(lin, strength, rng, sc):
    h, w = lin.shape[:2]
    m = np.clip(_value_noise(h, w, rng, 220 * sc) * 1.8 - 0.5, 0.0, 1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    m = m * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.35 * w) ** 2)) * strength
    lin = lin * (1 - 0.35 * m[..., None]) + 0.5 * m[..., None] * _soft_blur(lin, 24.0 * sc)
    bright = np.clip(lin - 0.6, 0.0, None)
    if bright.max() > 0:
        th = rng.uniform(0, math.pi)
        streak = np.zeros_like(lin)
        for i in range(-12, 13):
            streak += _shift(bright, i * 3.0 * sc * math.cos(th), i * 3.0 * sc * math.sin(th)) * math.exp(-abs(i) / 6.0)
        lin = lin + 0.12 * strength * streak
    return lin


# ---------------------------------------------------------------------------------- post model
def postprocess(path: str, fx: CameraFX, expo: dict, seed_key: str, intr: dict = None,
                depth_path: str = None, travel=None, artifacts: dict = None, sun: dict = None,
                artifact_seed: str = None):
    """Apply motion blur, optional camera artifacts and the lens / sensor model to a rendered 8-bit image
    file in place. Returns what was applied (for the frame metadata)."""
    artifacts = {k: v for k, v in (artifacts or {}).items() if v}
    unknown = sorted(set(artifacts) - set(ARTIFACTS))
    if unknown:
        raise ValueError(f"unknown camera artifact(s) {unknown}; known: {sorted(ARTIFACTS)}")
    sun = sun or {}
    rgb = _read(path)
    h, w = rgb.shape[:2]
    sc = h / float(fx.ref_height_px)
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()) & 0xFFFFFFFF)
    # windscreen dirt stays the same for all frames of one condition (same glass)
    rng_glass = np.random.default_rng(zlib.crc32((artifact_seed or seed_key).encode()) & 0xFFFFFFFF)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0)
    r2 = (rr / rr.max()) ** 2
    info = dict(expo, seed=seed_key)

    lin = _srgb_to_lin(rgb).astype(np.float32)
    depth = None
    if depth_path and os.path.exists(depth_path):
        depth = _read_depth(depth_path)
        os.remove(depth_path)
        if depth.shape != lin.shape[:2]:
            depth = None
    blur_px = 0.0
    if fx.motion_blur and depth is not None and travel is not None:
        lin, blur_px = _motion_blur(lin, depth, travel, intr)
    info.update(motion_blur_max_px=round(blur_px, 2),
                camera_travel_m=[round(float(v), 4) for v in travel] if travel is not None else None)
    # exposure error of the auto exposure
    if artifacts.get("exposure_bias_ev"):
        lin = lin * 2.0 ** float(artifacts["exposure_bias_ev"])
    # on the windscreen (moves with the camera, out of focus)
    sun_prox = 0.0
    if "sun_angle_from_axis_deg" in sun and sun.get("sun_elevation_deg", -90) > -1.0:
        sun_prox = float(np.clip((math.cos(math.radians(sun["sun_angle_from_axis_deg"])) - 0.5) / 0.5, 0.0, 1.0))
    if artifacts.get("windscreen_dirt"):
        lin = _windscreen_dirt(lin, float(artifacts["windscreen_dirt"]), rng_glass, sc, sun_prox)
    if artifacts.get("rain_smear"):
        lin = _rain_smear(lin, float(artifacts["rain_smear"]), rng, sc)
    # in the lens
    if artifacts.get("lens_smudge"):
        lin = _lens_smudge(lin, float(artifacts["lens_smudge"]), rng_glass, sc)
    if artifacts.get("lens_flare"):
        lin, vis = _lens_flare(lin, float(artifacts["lens_flare"]), sun, depth, sc)
        info["sun_flare_visibility"] = round(vis, 3)
    # optics: softness growing towards the corners (+ optional defocus)
    extra = float(artifacts.get("defocus_px", 0.0))
    sharp = _blur(lin, math.hypot(fx.psf_sigma_px, extra) * sc)
    soft = _blur(lin, math.hypot(fx.psf_sigma_edge_px, extra) * sc)
    lin = sharp * (1 - r2)[..., None] + soft * r2[..., None]
    # lateral chromatic aberration
    if fx.chroma_aberration:
        lin[..., 0] = _magnify(lin[..., 0], 1.0 + fx.chroma_aberration)
        lin[..., 2] = _magnify(lin[..., 2], 1.0 - fx.chroma_aberration)
    # bloom around bright areas and veiling glare over the whole frame
    thr = float(_srgb_to_lin(np.float32(fx.bloom_threshold)))
    bright = np.clip(lin - thr, 0.0, None)
    if fx.bloom and bright.max() > 0:
        lin = lin + fx.bloom * (0.6 * _blur(bright, 5.0 * sc) + 0.4 * _wide_blur(bright, 8, 3.0))
    if fx.veiling_glare:
        lin = lin * (1 - fx.veiling_glare) + fx.veiling_glare * _wide_blur(lin, 16, 4.0)
    # natural vignetting (cos^4 of the field angle, partly corrected by the lens design)
    if fx.vignette:
        f = (intr or {}).get("fx") or (0.5 * w / math.tan(math.radians(36.0)))
        cos4 = np.cos(np.arctan(rr / f)) ** 4
        lin = lin * (1 - fx.vignette * (1 - cos4))[..., None]
    # sensor noise: Poisson-Gaussian in linear light, scaled by the gain, slightly correlated
    gain_ev = min(float(expo.get("gain_ev", 0.0)) + float(artifacts.get("sensor_gain_ev", 0.0)), MAX_GAIN_EV)
    g = 2.0 ** gain_ev
    sigma = np.sqrt(np.clip(fx.shot_noise * g * np.clip(lin, 0, None) + (fx.read_noise * g) ** 2, 0, None))
    n = (math.sqrt(1 - fx.chroma_noise) * rng.standard_normal((h, w, 1), dtype=np.float32)
         + math.sqrt(fx.chroma_noise) * rng.standard_normal((h, w, 3), dtype=np.float32))
    k, _ = _kernel(0.5)
    n = _blur(n, 0.5) / float((k ** 2).sum())
    lin = lin + sigma * n
    if artifacts.get("hot_pixels"):
        cnt = int(artifacts["hot_pixels"])
        hy, hx = rng_glass.integers(0, h, cnt), rng_glass.integers(0, w, cnt)
        lin[hy, hx] += rng_glass.uniform(0.3, 1.0, (cnt, 3)).astype(np.float32)
    # in-camera processing: back to display values, sharpening, 8 bit, compression
    out = _lin_to_srgb(lin)
    if fx.sharpen:
        out = out + fx.sharpen * (out - _blur(out, 1.0 * sc))
    out = np.round(np.clip(out, 0.0, 1.0) * 255.0) / 255.0
    quality = int(artifacts.get("jpeg_quality", fx.jpeg_quality) or 0)
    ext = os.path.splitext(path)[1].lower()
    if quality and ext == ".png":
        tmp = path[:-4] + "__fx_tmp.jpg"
        _write(tmp, out, "JPEG", quality)
        out = _read(tmp)
        os.remove(tmp)
    _write(path, out, "JPEG" if ext in (".jpg", ".jpeg") else "PNG", quality or 95)
    info.update(sun, artifacts=artifacts or None, gain_ev_total=round(gain_ev, 3), jpeg_quality_used=quality)
    return info

"""Seeded 2-D Perlin / fBm noise in pure numpy (no Blender dependency).

Used for terrain relief, land-use masks and small placement jitter so the
scene is reproducible from Config.seed alone.
"""
from __future__ import annotations

import numpy as np


class Perlin2D:
    def __init__(self, seed: int, period: int = 1024):
        rng = np.random.default_rng(seed)
        self.period = period
        self.perm = rng.permutation(period)
        angles = rng.uniform(0, 2 * np.pi, period)
        self.grad = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def _g(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        p = self.perm
        idx = p[(p[ix % self.period] + iy) % self.period]
        return self.grad[idx]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx = x - x0
        fy = y - y0
        u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)

        def dot(gx, gy, dx, dy):
            g = self._g(gx, gy)
            return g[..., 0] * dx + g[..., 1] * dy

        n00 = dot(x0, y0, fx, fy)
        n10 = dot(x0 + 1, y0, fx - 1, fy)
        n01 = dot(x0, y0 + 1, fx, fy - 1)
        n11 = dot(x0 + 1, y0 + 1, fx - 1, fy - 1)
        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        return (nx0 + v * (nx1 - nx0)) * 1.41  # normalise to ~[-1, 1]


def fbm(perlin: Perlin2D, x, y, wavelength: float, octaves: int = 4,
        lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """Fractal Brownian motion, output roughly in [-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    total = np.zeros(np.broadcast(x, y).shape)
    amp = 1.0
    freq = 1.0 / wavelength
    norm = 0.0
    for i in range(octaves):
        total += amp * perlin(x * freq + 13.7 * i, y * freq - 7.1 * i)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm

"""Central configuration for the S-Bahn loop builder.

Every module reads from a Config instance; nothing is hard-coded elsewhere.
Distances are metres, angles radians unless the field name says otherwise.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TrackConfig:
    target_length_m: float = 8000.0       # closed-loop length (6-10 km)
    design_speed_kmh: float = 120.0       # sets cant and radii
    template: str = "stadium"             # "stadium" (two ~1.1 km straights) | "bean" (S-curve) | "twist"
    jitter: float = 0.06                  # seeded jitter on the plan template
    min_radius_m: float = 750.0           # tightest curve (120 km/h with 160 mm cant)
    max_radius_m: float = 3000.0
    transition_length_m: float = 140.0    # clothoid length (Übergangsbogen)
    min_transition_m: float = 80.0        # shortest cant ramp the designer may use
    min_straight_m: float = 120.0
    max_gradient: float = 0.0125          # 12.5 permille (DB Hauptbahn)
    max_cant_mm: float = 160.0            # DB max Überhöhung
    gauge_m: float = 1.435
    track_spacing_m: float = 4.0          # Gleisabstand double track
    sample_step_m: float = 1.0            # LUT resolution
    running_track_offset_m: float = 2.0   # +right of centreline = train's track
    elevation_smooth_m: float = 450.0     # Gaussian smoothing of terrain along s
    elevation_scale: float = 0.9         # how much of terrain relief the track follows


@dataclass
class PermanentWayConfig:
    rail_height_m: float = 0.172          # 60E1
    rail_centre_offset_m: float = 0.7535  # gauge/2 + half head width
    sleeper_spacing_m: float = 0.60       # B70 on main lines
    sleeper_len_m: float = 2.60
    sleeper_w_m: float = 0.30
    sleeper_h_m: float = 0.20
    ballast_top_h: float = -0.19          # relative to rail top (SOK)
    ballast_shoulder_m: float = 0.45      # beyond sleeper end
    ballast_slope: float = 1.5            # 1:1.5
    formation_h: float = -0.70            # Planum relative to SOK
    bench_w_m: float = 2.0                # Randweg width beyond ballast toe (masts, cable troughs, walls)
    embankment_slope: float = 1.5         # 1:1.5


@dataclass
class CatenaryConfig:
    contact_wire_h: float = 5.30          # DB standard Fahrdrahthöhe
    system_height: float = 1.80           # Re 200
    stagger_m: float = 0.40               # zigzag
    mast_offset_m: float = 3.30           # from track axis
    mast_height_m: float = 8.60
    span_straight_m: float = 65.0
    span_curve_m: float = 50.0
    dropper_spacing_m: float = 9.0
    midspan_sag_m: float = 1.15           # messenger wire sag at 65 m span
    contact_wire_radius: float = 0.0060
    messenger_radius: float = 0.0055


@dataclass
class SignalConfig:
    block_length_m: float = 1200.0        # Ks-Hauptsignal spacing per track
    signal_offset_m: float = 2.55         # right of track axis
    hektometer_spacing_m: float = 200.0
    crossings: int = 2                    # level crossings (Bahnübergänge)
    crossing_road_w_m: float = 6.5


@dataclass
class TerrainConfig:
    margin_m: float = 900.0               # around the loop bounding box
    grid_step_m: float = 8.0
    ribbon_half_w_m: float = 60.0
    ribbon_step_m: float = 2.5
    hill_amplitude_m: float = 28.0
    hill_wavelength_m: float = 1800.0
    detail_amplitude_m: float = 1.2
    detail_wavelength_m: float = 90.0
    landuse_wavelength_m: float = 700.0
    tree_clearance_m: float = 12.0        # no trunks closer to the centreline (crowns stay clear of the gauge)
    scatter_max_dist_m: float = 550.0     # nothing scattered further away
    # Sector layout along s (fractions must sum to ~1): type, share
    sectors: List[Tuple[str, float]] = field(default_factory=lambda: [
        ("suburban", 0.14), ("forest", 0.22), ("field", 0.18),
        ("forest", 0.13), ("field", 0.16), ("suburban", 0.17)])


@dataclass
class ScatterConfig:
    forest_tree_density: float = 0.018    # trees / m² (≈ 1 per 55 m²)
    field_tree_density: float = 0.0004
    suburban_tree_density: float = 0.003
    grass_density: float = 3.6            # tufts / m² near the track (was 1.8: ground showed between tufts)
    grass_max_dist_m: float = 28.0
    house_min_dist_m: float = 30.0
    house_density: float = 0.00035
    noise_barrier_share: float = 0.6      # of suburban length that gets walls
    noise_barrier_offset_m: float = 6.2   # from centreline (both sides), on the bench behind the masts
    noise_barrier_h_m: float = 4.0
    panel_len_m: float = 4.0
    hd_models: bool = True                # detailed model library (models/); False = low-poly proxies


@dataclass
class TrainConfig:
    cars: int = 4                          # Class 423: 4-car EMU
    car_len_m: float = 16.85
    width_m: float = 3.02
    floor_h: float = 1.0
    roof_h: float = 4.20
    cab_eye_h: float = 2.45               # driver's eye above rail top
    cab_eye_back_m: float = 1.3           # behind the windscreen glass
    cam_focal_mm: float = 24.0
    cam_sensor_mm: float = 36.0


@dataclass
class Config:
    seed: int = 7
    out_dir: str = "out"
    blend_name: str = "sbahn_loop.blend"
    render_engine: str = "CYCLES"         # or "EEVEE"
    build_terrain: bool = True
    build_scatter: bool = True
    build_catenary: bool = True
    build_signals: bool = True
    build_crossings: bool = True
    build_station: bool = True
    build_train: bool = True
    preview_render: bool = False          # render a few validation frames
    preview_res: Tuple[int, int] = (960, 540)
    preview_samples: int = 48
    asset_dir: str = "assets"
    use_downloaded_assets: bool = True    # use textures in asset_dir when present
    track: TrackConfig = field(default_factory=TrackConfig)
    pway: PermanentWayConfig = field(default_factory=PermanentWayConfig)
    catenary: CatenaryConfig = field(default_factory=CatenaryConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    scatter: ScatterConfig = field(default_factory=ScatterConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = cls()
        for k, v in d.items():
            if not hasattr(cfg, k):
                continue
            cur = getattr(cfg, k)
            if dataclasses.is_dataclass(cur) and isinstance(v, dict):
                for kk, vv in v.items():
                    if hasattr(cur, kk):
                        setattr(cur, kk, vv)
            else:
                setattr(cfg, k, v)
        return cfg

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def apply_overrides(self, overrides: List[str]) -> None:
        """Apply 'a.b=value' strings (CLI)."""
        for ov in overrides:
            key, _, val = ov.partition("=")
            obj = self
            parts = key.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            cur = getattr(obj, parts[-1])
            if isinstance(cur, bool):
                val = val.lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int):
                val = int(val)
            elif isinstance(cur, float):
                val = float(val)
            elif isinstance(cur, (list, tuple)):
                val = json.loads(val)
            setattr(obj, parts[-1], val)

    @property
    def out_path(self) -> str:
        return os.path.abspath(self.out_dir)

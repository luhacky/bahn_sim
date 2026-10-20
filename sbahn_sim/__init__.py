"""
sbahn_sim - procedural German S-Bahn loop for synthetic cab-view data.

Modules (build order):
    config       central parameters (Config dataclass, seedable)
    noise        numpy Perlin / fBm used by terrain and land-use masks
    track_path   closed-loop alignment (straights, clothoids, arcs), gradients,
                 cant, arc-length lookup table (TrackPath)
    util         Blender helpers (collections, meshes, instancers, GN helpers)
    materials    procedural PBR materials + texture-set loader
    assets       download hooks for CC0 textures / models (Poly Haven, ambientCG)
    track_geometry   rails (60E1 profile), sleepers (B70), ballast bed
    catenary     masts, cantilevers, contact + messenger wire, droppers
    signals      Ks signals, Hektometertafeln, level crossings
    terrain      heightfield, corridor ribbon, embankments/cuttings, land-use
    scatter      Geometry Nodes scatter: forest, grass, houses, noise barriers
    train_rig    proxy Class 423 + cab camera, place_train(s)
    build        orchestration: build_scene(cfg) -> writes .blend + sidecars

Consumers (weather, obstacles, dataset renderer) should only need:
    from sbahn_sim.track_path import TrackPath
    tp = TrackPath.from_scene()          # reads the LUT stored in the .blend
    tp.length_m, tp.sample(s), tp.matrix(s, lateral=2.0, height=0.0)
    from sbahn_sim.train_rig import place_train
"""
__version__ = "0.1.0"

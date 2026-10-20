"""Procedural model library: German trees / shrubs / grass, buildings and road vehicles.

    from sbahn_sim.models import trees, houses, vehicles, vegetation, library
    trees.make_tree("beech", rng, lod=0).to_object("Beech", collection, [materials.bark("beech"), materials.foliage("beech")])
    vehicles.make_vehicle("estate", rng).to_object("Car", collection, vehicles.vehicle_materials("silver"))
    parts, mats = vehicles.obstacle_parts("hatchback", rng, "red")     # obstacles.BUILDERS format
    parts, mats = library.fallen_tree_parts(rng, "spruce")              # obstacles.BUILDERS format
    library.build_instance_sources(cfg)                                 # SRC_* collections for scatter.py
"""

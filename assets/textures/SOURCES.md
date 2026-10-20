# Texture sources

All sets are CC0 (public domain) from Poly Haven and ambientCG, 2k JPG, downloaded 2026-10-02
with `sbahn_sim/assets.py`. Folder name = material role (see `sbahn_sim/materials.py`).

| folder | source asset | used for | tint in code |
|---|---|---|---|
| ballast | [gravel_stones](https://polyhaven.com/a/gravel_stones) | track ballast (tiled at 6 m so stones are 3-6 cm) | x1.45 |
| sleeper | [concrete_floor_worn_001](https://polyhaven.com/a/concrete_floor_worn_001) | B70 concrete sleepers | x1.7 |
| concrete | copy of `sleeper` | mast foundations, shelters | x2.1 |
| grass | [ambientCG Grass001](https://ambientcg.com/view?id=Grass001) | dense green ground under the grass tufts (1.4 m tile) | none |
| grass_dry | [grass_ground](https://polyhaven.com/a/grass_ground) | dry patches mixed into the grass (about a quarter) | (0.80, 0.90, 0.80) |
| walkway | [gravel_floor_02](https://polyhaven.com/a/gravel_floor_02) | walkway strip beside the ballast | x0.38 |
| forest_floor | [forest_leaves_02](https://polyhaven.com/a/forest_leaves_02) | forest floor | (0.45, 0.60, 0.60) |
| asphalt | [asphalt_02](https://polyhaven.com/a/asphalt_02) | level-crossing roads | none |
| rail_body | [rust_coarse_01](https://polyhaven.com/a/rust_coarse_01) | rail web and foot | x0.6 |
| bark_pine | [pine_bark](https://polyhaven.com/a/pine_bark) | pine trunks | none |
| bark_spruce | [bark_brown_02](https://polyhaven.com/a/bark_brown_02) | spruce trunks | none |
| bark_oak | [jolcham_oak_bark_01](https://polyhaven.com/a/jolcham_oak_bark_01) | oak trunks | none |

Remove a folder to fall back to the procedural material for that role (rebuild the scene afterwards).

# How to render with the S-Bahn cab-view simulation

This guide covers everything from a fresh machine to a finished YOLO validation set.
All commands are run in the project folder (the one containing this file).

---

## 1. What you need

| | |
|---|---|
| Blender | 4.2 or newer (tested with 4.4.3). `setup.py --get-blender` can fetch a portable 4.4.3 on Windows / Linux. |
| GPU | NVIDIA (OptiX / CUDA), AMD (HIP), Apple (Metal) or Intel (oneAPI). CPU works but is about 10x slower. |
| Python | 3.8+ for the helper scripts (`setup.py`, `render_batch.py`, `export_project.py`); standard library only. |
| Disk | about 3 GB (scene 145 MB, textures 150 MB, renders ~0.7 MB per frame). |

---

## 2. Set up on a new machine

```bash
python setup.py              # Windows: setup.bat     Linux / macOS: ./setup.sh
```

It runs five steps and skips what already exists, so it is safe to re-run:

1. finds Blender (PATH, standard install folders, the `BLENDER` variable, or `--blender PATH`)
2. reports the GPU Blender will use
3. downloads the CC0 photo textures into `assets/textures/` (list and licences: `assets/textures/SOURCES.md`)
4. builds the scene into `out/scene/sbahn_loop.blend` (about 2 minutes, identical on every machine)
5. renders one test frame into `out/setup_test/`

Options: `--get-blender`, `--blender PATH`, `--skip-textures`, `--rebuild`, `--skip-test`.

### Moving the project to another machine

```bash
python export_project.py                 # code, docs, conditions and textures -> ../sbahn_sim_export_<date>.zip
python export_project.py --no-textures   # ~1 MB, setup.py downloads the textures again
python export_project.py --with-scene    # also the built scene (otherwise setup.py rebuilds it)
python export_project.py --with out/batch50   # also result folders, e.g. to continue a batch
```

Unzip on the other machine and run `python setup.py` inside `sbahn_sim_project/`. A batch whose
output folder was copied along continues where it stopped (finished frames are skipped).

On a server where nothing can be installed and there is no SSH: put the zip on MEGA, copy `mega_get.py`
to the server (paste it into `cat > mega_get.py`, finish with Ctrl+D) and run

```bash
python3 mega_get.py "https://mega.nz/folder/XXXX#KEY" ~/sbahn       # needs only python3 and openssl
cd ~/sbahn && python3 -m zipfile -e sbahn_sim_export_*.zip . && cd sbahn_sim_project
python3 setup.py --get-blender                                     # portable Blender, no admin rights
```

---

## 3. Render the standard batch (50 combinations)

```bash
python render_batch.py --dry-run     # the 50 combinations, frame count, time estimate
python render_batch.py --test        # 8 frames from representative combinations (~7 min)
python render_batch.py               # the full batch -> out/batch50 (about 3 h on an RTX 2070 Super)
```

The batch is:

| group | combinations | what |
|---|---|---|
| baseline | 15 | every weather / light preset as it is |
| fog sweep | 10 | partly cloudy sky with ground fog 10 %, 20 % ... 100 % (100 % = 150 m visibility) |
| camera artifacts | 25 | lens flare, windscreen dirt, rain smears, lens smudge, worn camera, over / under exposure |

Each combination renders 3 of the 8 obstacle cases (rotating, so every case appears about 19 times) at
2 distances (50 + 150 m or 100 + 250 m); the fog sweep always uses the same 3 cases at 50, 100 and
200 m so the fog levels compare directly. That is 330 frames.

Options: `--out DIR`, `--blend FILE`, `--res W H`, `--samples N`, `--only NAME ...`,
`--frames-per-case 3` (50 / 100 / 200 m for every combination, 480 frames), `--blender PATH`.
Interrupted? Run the same command again: it resumes. The combinations are written to
`<out>/batch_conditions.json`; edit `render_batch.py` (lists at the top) to change them.

---

## 3b. Try settings in the preview page

```bash
python preview_server.py        # open http://localhost:8765 (on a server: --host 0.0.0.0, or tunnel the port)
```

Pick obstacle, distance, preset, time, clouds, fog %, rain and camera artifacts, press **Render preview**
(~20 s at 640x360 once Blender has loaded the scene), and **Add to custom.json** for the ones you like.
The page shows the command that renders `conditions/custom.json` at full quality.

## 4. Render your own selection

Everything goes through one command:

```bash
blender -b -P gpu_enable.py -P render_dataset.py -- --blend out/scene/sbahn_loop.blend [options]
```

Example: two obstacles, every distance milestone, two weather presets, 1280x720:

```bash
blender -b -P gpu_enable.py -P render_dataset.py -- --blend out/scene/sbahn_loop.blend \
    --mode milestones --cases worker_on_opposite_track person_crossing_track \
    --conditions partly_cloudy fog_dense --res 1280 720 --samples 32 --overlay --out out/my_run
```

### 4.1 Mode and distances

| option | meaning |
|---|---|
| `--mode milestones` | one frame per obstacle at each distance in `--milestones` (default 50 100 200 300 500 m) |
| `--mode approach` | the train drives from `--from-m 520` to `--to-m 25` m before the obstacle at `--speed 100` km/h, a frame every `--dt 0.5` s, plus the milestone frames |
| `--mode drive` | all obstacles present, the train drives the loop from `--drive-s0` for `--drive-seconds` (300) s, a frame every `--dt` |
| `--milestones 25 50 75 ...` | any list of distances (train front to obstacle, metres) |

### 4.2 Obstacles (`--cases`, default all 8; `--list` shows positions)

| case | YOLO class | |
|---|---|---|
| `car_on_crossing_00` | 1 car | red hatchback stalled on level crossing A |
| `car_on_crossing_01` | 1 car | silver car on level crossing B |
| `person_standing_on_track` | 0 person | facing the train between the rails |
| `person_crossing_track` | 0 person | walking across the running track |
| `worker_on_opposite_track` | 0 person | track worker in DB hi-vis with helmet, on the other track |
| `fallen_tree_across_tracks` | 2 fallen_tree | spruce across both tracks |
| `branches_in_curve` | 3 branches | branch pile in a 750 m radius curve |
| `deer_on_track` | 4 animal | roe deer on the running track |

Parked cars at houses are labelled class 1 with `background: true` (`--no-background-vehicles` turns that off).

### 4.3 Weather, light and time of day (`--conditions`)

Preset names (each listed preset is one pass over all selected cases):

| preset | | preset | |
|---|---|---|---|
| `clear_noon` | 12:30, almost cloudless | `night_clear` | night, headlights and sky glow only |
| `clear_morning` | 07:30, low sun | `night_overcast` | dark night |
| `partly_cloudy` | 14:00, scattered clouds | `fog_light` | 600 m visibility |
| `overcast` | closed cloud deck | `fog_dense` | 150 m visibility |
| `low_sun_glare` | sun 4 deg high, straight ahead | `night_fog` | night, 250 m |
| `sunset` | 21:10 | `rain_light` | 2.5 mm/h, drops on the glass |
| `twilight` | 21:45 | `rain_heavy` / `night_rain` | 25 / 8 mm/h, wiper running |

Or a JSON file with a list of conditions (examples in `conditions/`: `fog_sweep.json` = fog 0-100 % in 5 % steps, `camera_artifacts_examples.json`). Each entry is a
dict; every key is optional, a `preset` fills the rest:

```json
[
  {"name": "dusk_fog_flare", "preset": "sunset", "fog_density": 0.008,
   "camera_artifacts": {"lens_flare": 0.8, "windscreen_dirt": 0.5},
   "cases": ["person_standing_on_track"], "milestones": [50, 100]}
]
```

| key | meaning |
|---|---|
| `name` | used in the file names |
| `time_of_day`, `date`, `latitude`, `longitude` | real sun position (default Munich, 21 June) |
| `sun_elevation_deg`, `sun_azimuth_deg`, `sun_azimuth_rel_deg` | force the sun (`rel` = relative to the train, 0 = straight ahead) |
| `cloud_cover` | 0..1 |
| `fog_density` (1/m) or `fog_visibility_m` | ground fog; 0.026 /m = 150 m visibility (100 % in the fog sweep) |
| `fog_top_m`, `fog_patchiness` | fog layer height (80 m), fog banks (0 = even) |
| `haze_visibility_m` | background haze (14 km clear, 8 km overcast / rain) |
| `rain_rate_mm_h`, `wind_ms`, `wetness` | rain, crosswind, wet surfaces |
| `windscreen`, `wiper`, `wiper_period_s` | drops on the glass, wiper on / off / parked |
| `headlights`, `headlight_mode`, `headlight_power` | three-light head code, dipped / high |
| `exposure_ev` | `auto` (default) or a fixed EV |
| `seed` | varies clouds, fog banks, drop layout |
| `camera_artifacts` | see 4.4 |
| `cases`, `milestones` | restrict this condition to these cases / distances (default: the command-line ones) |

### 4.4 Camera model and artifacts

Every frame passes through a model of a real cab camera (`sbahn_sim/camera_fx.py`): the shutter time
follows the light (1/400 s sunny, ~1/120 s overcast, 1/30 s night) and blurs the near track
accordingly; lens softness, slight colour fringing, darker corners, glare, sensor noise (more in low
light), sharpening and JPEG compression. `--no-camera-fx` gives the clean render, `--no-motion-blur`
and `--jpeg-quality Q` (0 = off) switch single parts.

Optional artifacts per condition (`"camera_artifacts": {...}`, strengths 0..1 unless noted):

| artifact | effect |
|---|---|
| `lens_flare` | ghosts mirrored through the image centre, halo and a 6-ray star from the sun (when in or near the frame and not behind trees) and other clipped highlights |
| `windscreen_dirt` | dust film, out-of-focus specks and insect splats; they darken the view and glow against a low sun |
| `rain_smear` | arcs left by a worn wiper blade (blur, refraction, scattered light) and residual drops |
| `lens_smudge` | grease on the lens: soft glow over part of the frame, streaks around bright light |
| `defocus_px` | extra blur in pixels (misadjusted or fogged-up camera) |
| `exposure_bias_ev` | auto-exposure error, e.g. `1.0` = one stop over-exposed, `-1.0` under |
| `sensor_gain_ev` | extra sensor gain (more noise), e.g. `1.5` |
| `hot_pixels` | number of stuck bright pixels, e.g. `250` |
| `jpeg_quality` | stronger compression, e.g. `45` |

None of these moves anything, so the label boxes stay exact. Labels are geometric: an obstacle hidden
by fog, glare or a smear keeps its box. Use `visible_px`, the distance columns and the logged
conditions to decide what a detector can be expected to find.

### 4.5 Image and run settings

| option | default | |
|---|---|---|
| `--res W H` | 1920 1080 | we use 1280 720 |
| `--samples N` | 64 | 32 is enough with the denoiser |
| `--engine` | Cycles | `EEVEE` works but was not faster here |
| `--format` | PNG | or JPEG |
| `--overlay` | off | also write `overlays/` with the boxes drawn |
| `--labels-only` | off | plan, labels and metadata without rendering (seconds) |
| `--min-px N` | 4 | visible pixels an object needs for a YOLO row |
| `--max-frames N` | all | stop after N frames (quick tests) |
| `--shard i n` | 0 1 | split a run across several processes |
| `--no-skip` | off | re-render frames that already exist (normally runs resume) |

---

## 5. Output

```
<out>/images/<id>.png         camera frame            <id> = <condition>__<case>__m<distance>
<out>/labels/<id>.txt         YOLO rows "class cx cy w h" (normalised, visible box)
<out>/meta/<id>.json          everything about the frame (conditions, camera, every object)
<out>/overlays/<id>.png       boxes drawn in (with --overlay)
<out>/frames.csv              one row per frame and object
<out>/dataset.yaml            YOLO dataset file (classes person, car, fallen_tree, branches, animal)
```

`frames.csv` columns include: condition, case, distance milestone, fog density, rain, time of day,
sun elevation / azimuth, cloud cover, visibility, camera artifacts, shutter time, sensor gain, and per
object the class, distance along the track and straight-line, visible and full (amodal) box, visible
fraction, truncation and visible pixels. `python fog_summary.py <out>` makes a per-fog-level table.

---

## 6. Planning time (RTX 2070 Super, 1280x720, 32 samples)

| | |
|---|---|
| per frame | ~30 s (23 s render + labels + camera model) |
| per change of condition | ~10 s |
| loading the scene | ~1-2 min per run |

Pixels and samples scale the render part roughly linearly. Other programs that use all CPU cores
slow down the scene preparation noticeably.

---

## 7. Changing the world

```bash
blender -b -P build_scene.py -- --out out/scene [--seed N] [--template stadium|bean|twist] [--length 6000-10000]
```

`--set a.b=value` overrides any field of `sbahn_sim/config.py` (e.g. `--set scatter.grass_density=2.0`).
Textures: delete a folder in `assets/textures/` to fall back to the procedural material for that role,
or add a set (ambientCG / Poly Haven naming) to use it; rebuild the scene afterwards.

---

## 8. Troubleshooting

| problem | fix |
|---|---|
| "Blender 4.2 or newer not found" | install Blender, pass `--blender PATH`, set `BLENDER`, or `setup.py --get-blender` |
| `[gpu] no GPU backend found` | update the GPU driver; rendering still works on the CPU, just slower |
| out of GPU memory | lower `--res`, or set `--set scatter.grass_density=1.8` and rebuild |
| textures missing (grey / procedural look) | re-run `setup.py` (it only downloads what is missing) |
| a run stopped | run the same command again; finished frames are skipped |

# Real-photo hCaptcha-tile dataset (60 classes)

Every tile in this folder is a **real photograph** (centre-cropped to a
square and resized to 96×96 px), organised one folder per class. The
images were gathered via workspace image search and curated by query slug;
`_build_real_data.py` performs the crop/resize and writes the manifest.

## Classes (60)

| id | class | prompt |
|---:|---|---|
| 0 | `bus` | Please click each image containing a bus |
| 1 | `car` | Please click each image containing a car |
| 2 | `truck` | Please click each image containing a truck |
| 3 | `train` | Please click each image containing a train |
| 4 | `bicycle` | Please click each image containing a bicycle |
| 5 | `motorcycle` | Please click each image containing a motorcycle |
| 6 | `boat` | Please click each image containing a boat |
| 7 | `airplane` | Please click each image containing an airplane |
| 8 | `traffic_light` | Please click each image containing a traffic light |
| 9 | `red_light` | Please click each image containing a red light |
| 10 | `crosswalk` | Please click each image containing a crosswalk |
| 11 | `fire_hydrant` | Please click each image containing a fire hydrant |
| 12 | `parking_meter` | Please click each image containing a parking meter |
| 13 | `dog` | Please click each image containing a dog |
| 14 | `cat` | Please click each image containing a cat |
| 15 | `rabbit` | Please click each image containing a rabbit |
| 16 | `horse` | Please click each image containing a horse |
| 17 | `elephant` | Please click each image containing an elephant |
| 18 | `cow` | Please click each image containing a cow |
| 19 | `bird` | Please click each image containing a bird |
| 20 | `frog` | Please click each image containing a frog |
| 21 | `turtle` | Please click each image containing a turtle |
| 22 | `snail` | Please click each image containing a snail |
| 23 | `kangaroo` | Please click each image containing a kangaroo |
| 24 | `hammer` | Please click each image containing a hammer |
| 25 | `drill` | Please click each image containing a drill |
| 26 | `saw` | Please click each image containing a saw |
| 27 | `paintbrush` | Please click each image containing a paintbrush |
| 28 | `wrench` | Please click each image containing a wrench |
| 29 | `screwdriver` | Please click each image containing a screwdriver |
| 30 | `wood` | Please click each image containing a wood |
| 31 | `nail` | Please click each image containing a nail |
| 32 | `screw` | Please click each image containing a screw |
| 33 | `bolt` | Please click each image containing a bolt |
| 34 | `wall` | Please click each image containing a wall |
| 35 | `canvas` | Please click each image containing a canvas |
| 36 | `apple` | Please click each image containing an apple |
| 37 | `pizza` | Please click each image containing a pizza |
| 38 | `table` | Please click each image containing a table |
| 39 | `chair` | Please click each image containing a chair |
| 40 | `cup` | Please click each image containing a cup |
| 41 | `book` | Please click each image containing a book |
| 42 | `clock` | Please click each image containing a clock |
| 43 | `umbrella` | Please click each image containing an umbrella |
| 44 | `tree` | Please click each image containing a tree |
| 45 | `flower` | Please click each image containing a flower |
| 46 | `house` | Please click each image containing a house |
| 47 | `mountain` | Please click each image containing a mountain |
| 48 | `boot` | Please click each image containing a boot |
| 49 | `zebra` | Please click each image containing a zebra |
| 50 | `giraffe` | Please click each image containing a giraffe |
| 51 | `lion` | Please click each image containing a lion |
| 52 | `bear` | Please click each image containing a bear |
| 53 | `sheep` | Please click each image containing a sheep |
| 54 | `duck` | Please click each image containing a duck |
| 55 | `fish` | Please click each image containing a fish |
| 56 | `butterfly` | Please click each image containing a butterfly |
| 57 | `banana` | Please click each image containing a banana |
| 58 | `guitar` | Please click each image containing a guitar |
| 59 | `cactus` | Please click each image containing a cactus |

### Label rules

* **red_light** — a traffic light whose **red lamp is lit**.
* **traffic_light** — a traffic light whose red lamp is **not** lit
  (dim 3-lamp signal, or yellow/green lit).
* **crosswalk** — white zebra stripes painted across a road band.

## Layout

```
data/<class>/<class>_00000.jpg   # one folder per class (real photo tiles)
data/manifest.jsonl              # one JSON object per image
data/_preview.jpg                # 60-column contact sheet
data/README.md
```

Each manifest line:

```json
{"image": "data/bus/bus_00000.jpg", "label": "bus", "class_id": 0, "prompt": "Please click each image containing a bus"}
```

## Rebuild

```bash
pip install Pillow
# 1. download photos per class into image-search/ (query slug -> class map
#    lives at the top of _build_real_data.py)
# 2. crop/resize + manifest + preview:
python _build_real_data.py
```

The procedural Pillow painters in `make_dataset.py` + `synth_shapes.py`
still draw all 60 classes (deterministic per-class seeds) and are used to
augment / backfill when a class has fewer real photos; the real corpus is
what teaches the offline models photograph appearance.

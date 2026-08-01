# SVG Visualization Legend

Reference for `interim/svg_visualizations/*.png` — these images carry no on-screen legend by design, so this file is the mapping.

## Node colors

Ego is a **black star** (`*`), independent of the class palette below — it isn't a segmentation class.

Every other node is drawn as a colored circle using its class's **official Mapillary Vistas v1.2 color** — the same color the segmentation overlay uses for that class, so the graph layer and segmentation layer visually agree.

| Node type | Class | RGB | Hex |
|---|---|---|---|
| building | Building | (70, 70, 70) | #464646 |
| light_pole | Pole | (153, 153, 153) | #999999 |
| light_pole | Street Light | (210, 170, 100) | #d2aa64 |
| light_pole | Traffic Light | (250, 170, 30) | #faaa1e |
| light_pole | Utility Pole | (0, 0, 80) | #000050 |
| road_marking | Crosswalk - Plain | (140, 140, 200) | #8c8cc8 |
| road_marking | Lane Marking - General | (255, 255, 255) | #ffffff |
| signage | Traffic Sign (Back) | (192, 192, 192) | #c0c0c0 |
| signage | Traffic Sign (Front) | (220, 220, 0) | #dcdc00 |
| signage | Traffic Sign Frame | (128, 128, 128) | #808080 |
| vegetation | Vegetation | (107, 142, 35) | #6b8e23 |

## Edge styles

| Relation | Line style | Opacity | Color | Meaning |
|---|---|---|---|---|
| `sees` | solid | 0.5 | white | Ego → object, always present |
| `mounted_with` | dashed | 0.7 | yellow | Signage/Light_pole instances sharing one physical support |
| `near` | dotted | 0.28 | cyan | General spatial adjacency, distance ≤ `near_cutoff_d` |

## Segmentation overlay

Opacity 0.5–0.6 over the base image. Only the 11 node-relevant classes above are highlighted — everything else (sky, road, vehicles, people, etc.) is left as plain, unhighlighted original image, since the overlay's purpose is QC-ing graph construction specifically, not general segmentation review.

## Provisional parameters in effect when these images were generated

Check `configs/svg_schema.yaml` for the current values — `near_cutoff_d`, `mask_confidence_threshold`, `mask_min_area_fraction`, and `mounted_with_distance_threshold` are all provisional until confirmed by visual inspection of these exact images.
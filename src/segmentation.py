"""
Mask2Former panoptic model load + inference. Adapted from the project's
Mask2Former playground notebook; class lists preserved verbatim from
there so segmentation behavior matches what was already validated.

No resizing is applied — images are processed at native resolution, so
segmentation output dimensions always match the original SVI exactly.
This matters for 03's SVG visualization step, which overlays the graph
directly onto the unmodified input image.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation


# ─── Mapillary Vistas v1.2 class names, exact order matching the checkpoint's
#     id2label indices — verbatim from the validated playground notebook. ────
MAPILLARY_NAMES = [
    'Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Wall',
    'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking', 'Pedestrian Area',
    'Rail Track', 'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building',
    'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider',
    'Lane Marking - Crosswalk', 'Lane Marking - General', 'Mountain', 'Sand',
    'Sky', 'Snow', 'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench',
    'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant',
    'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole',
    'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole',
    'Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)',
    'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle',
    'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow',
    'Car Mount', 'Ego Vehicle',
]

# Official Mapillary Vistas v1.2 color palette, positionally aligned to
# MAPILLARY_NAMES (index i in one list corresponds to index i in the
# other) — verbatim from the validated playground notebook, verified
# against the official Mapillary Vistas config / mmseg reference.
MAPILLARY_PALETTE = [
    [165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
    [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255],
    [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
    [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
    [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
    [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
    [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
    [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
    [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
    [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
    [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
    [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
    [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
    [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90],
    [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70],
    [0, 0, 192], [32, 32, 32], [120, 10, 10],
]


# 'Stuff' = amorphous background classes fused into ONE segment per class.
# 'Thing' = countable objects that stay separate per-instance.
STUFF_CLASS_NAMES = {
    'Sky', 'Road', 'Sidewalk', 'Terrain', 'Vegetation', 'Water',
    'Building', 'Wall', 'Fence', 'Guard Rail', 'Barrier', 'Curb',
    'Bridge', 'Tunnel', 'Mountain', 'Snow', 'Sand',
    'Parking', 'Rail Track', 'Crosswalk - Plain', 'Curb Cut',
    'Bike Lane', 'Service Lane', 'Pedestrian Area',
    'Lane Marking - Crosswalk', 'Lane Marking - General',
}


def load_model(checkpoint: str, device: str = None):
    """Loads processor + model, resolves class names and stuff/thing split.

    Returns: (processor, model, device, class_names, stuff_ids, id2label)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint).to(device).eval()

    id2label = model.config.id2label
    num_classes = len(id2label)

    # Safety check from the original playground: only trust the curated
    # Mapillary name list if it actually matches this checkpoint's class
    # count — otherwise fall back to the model's own (less friendly) labels
    # rather than silently mis-mapping names to the wrong indices.
    if len(MAPILLARY_NAMES) == num_classes:
        class_names = MAPILLARY_NAMES
    else:
        print(f"⚠️  Checkpoint has {num_classes} classes, expected {len(MAPILLARY_NAMES)} "
              f"(Mapillary Vistas v1.2) — falling back to the checkpoint's own id2label "
              f"strings. STUFF/THING matching below may be unreliable — check manually.")
        class_names = [id2label[i] for i in range(num_classes)]

    stuff_ids = {i for i, name in enumerate(class_names) if name in STUFF_CLASS_NAMES}

    return processor, model, device, class_names, stuff_ids, id2label


def validate_against_schema(class_names, stuff_ids, tier1_thing_classes, stuff_classes_for_island_labeling):
    """Cross-checks the loaded checkpoint's actual stuff/thing split against
    configs/svg_schema.yaml's expectations. Returns a list of problem
    strings — empty list means everything lines up.
    """
    stuff_names_actual = {class_names[i] for i in stuff_ids}
    problems = []

    for cls in tier1_thing_classes:
        if cls not in class_names:
            problems.append(f"'{cls}' not found in model class list at all")
        elif cls in stuff_names_actual:
            problems.append(f"'{cls}' expected THING but is classified as STUFF (fused)")

    for cls in stuff_classes_for_island_labeling:
        if cls not in class_names:
            problems.append(f"'{cls}' not found in model class list at all")
        elif cls not in stuff_names_actual:
            problems.append(f"'{cls}' expected STUFF (fused) but is classified as THING")

    return problems


def infer_panoptic(image: Image.Image, processor, model, device, stuff_ids, confidence_threshold: float):
    """Runs panoptic inference at the image's native resolution (no resize).

    Returns: (seg_map: np.ndarray[int32] of shape (H, W), segments_info: list[dict])
    Each segments_info dict has: id, label_id, class_name, is_stuff, score.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_panoptic_segmentation(
        outputs,
        threshold=confidence_threshold,
        label_ids_to_fuse=stuff_ids,
        target_sizes=[image.size[::-1]],  # (H, W) — matches the ORIGINAL image, no resize
    )[0]

    seg_map = result["segmentation"].cpu().numpy().astype(np.int32)

    segments_info = []
    for seg in result["segments_info"]:
        label_id = seg["label_id"]
        score = seg.get("score", None)
        segments_info.append({
            "id": int(seg["id"]),
            "label_id": int(label_id),
            "class_name": MAPILLARY_NAMES[label_id] if label_id < len(MAPILLARY_NAMES) else f"class_{label_id}",
            "is_stuff": label_id in stuff_ids,
            "score": float(score) if score is not None else None,
        })

    return seg_map, segments_info


def save_result(out_dir, point_id, seg_map, segments_info):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{point_id}.npz", seg_map=seg_map)
    with open(out_dir / f"{point_id}.json", "w") as f:
        json.dump(segments_info, f, indent=1)


def load_result(out_dir, point_id):
    out_dir = Path(out_dir)
    seg_map = np.load(out_dir / f"{point_id}.npz")["seg_map"]
    with open(out_dir / f"{point_id}.json") as f:
        segments_info = json.load(f)
    return seg_map, segments_info


# ─── Color palette for light QC visualization (not the full graph-aware
#     overlay — that belongs to 03) ───────────────────────────────────────
def get_palette(class_names):
    """Returns the official Mapillary palette if class_names matches the
    standard v1.2 taxonomy exactly (same fallback logic as load_model's
    class-name resolution); otherwise falls back to an auto-generated one
    so this never hard-fails on an unexpected checkpoint.
    """
    if class_names == MAPILLARY_NAMES:
        return [tuple(c) for c in MAPILLARY_PALETTE]
    return _auto_palette(len(class_names))


def _auto_palette(n_classes: int, seed: int = 42):
    import colorsys
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i / max(n_classes, 1)) % 1.0, 0.75, 0.90))
        for i in range(n_classes)
    ]

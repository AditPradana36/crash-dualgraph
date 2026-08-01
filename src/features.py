"""
Scene-level SVG feature computation: SVF, Enclosure Index, Shannon entropy.
Operate directly on a (seg_map, segments_info) pair from 02.

IMPORTANT: class name lists passed in here (e.g. enclosure_classes from
svg_schema.yaml) must use the EXACT Mapillary Vistas class name strings
(e.g. "Building", "Traffic Light", "Traffic Sign (Front)") — not a
lowercase/snake_case version. A mismatch here silently zeroes out the
affected metric rather than raising an error, since an empty class-name
match is not treated as an exception.
"""
import numpy as np


def _class_mask(seg_map, segments_info, class_names):
    mask = np.zeros(seg_map.shape, dtype=bool)
    class_set = set(class_names)
    for s in segments_info:
        if s["class_name"] in class_set:
            mask |= (seg_map == s["id"])
    return mask


def compute_svf(seg_map, segments_info):
    """Sky View Factor — sky pixel proportion over the WHOLE image."""
    sky_mask = _class_mask(seg_map, segments_info, {"Sky"})
    return float(sky_mask.sum() / seg_map.size)


def compute_enclosure(seg_map, segments_info, enclosure_classes, crop_fraction=0.40):
    """Enclosure Index — proportion of enclosure-class pixels within the
    TOP crop_fraction of image height. Denominator is the whole crop area
    (not enclosure+sky), per the locked redefinition."""
    h, w = seg_map.shape
    crop_h = int(round(h * crop_fraction))
    crop = seg_map[:crop_h, :]
    if crop.size == 0:
        return 0.0
    mask = _class_mask(crop, segments_info, enclosure_classes)
    return float(mask.sum() / crop.size)


def compute_entropy(seg_map, segments_info):
    """Shannon entropy over CLASS (not per-instance) pixel-count
    proportions, whole image."""
    class_pixel_counts = {}
    for s in segments_info:
        cnt = int((seg_map == s["id"]).sum())
        class_pixel_counts[s["class_name"]] = class_pixel_counts.get(s["class_name"], 0) + cnt
    total = seg_map.size
    probs = np.array([c / total for c in class_pixel_counts.values() if c > 0])
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))

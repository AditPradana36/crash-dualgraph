"""
Builds one SVG (Street View Graph) HeteroData object per point.

Requires torch + torch_geometric (basic Data/HeteroData objects only —
no compiled extensions needed just to construct and save these graphs;
those are only required later, for actual GNN layers in 06/07).
"""
import math

import numpy as np
from scipy import ndimage
import torch
from torch_geometric.data import HeteroData


CLASS_TO_NODE_TYPE = {
    "Traffic Sign Frame": "signage",
    "Traffic Sign (Back)": "signage",
    "Traffic Sign (Front)": "signage",
    "Traffic Light": "light_pole",
    "Street Light": "light_pole",
    "Pole": "light_pole",
    "Banner": "banner",
    "Billboard": "billboard",
    "Utility Pole": "light_pole",
    "Crosswalk - Plain": "road_marking",
    "Lane Marking - General": "road_marking",
    "Building": "building",
    "Vegetation": "vegetation",
}

# 'Island' classes: fused stuff segments needing connected-component
# splitting into individual instances (per-object, no aggregation).
ISLAND_CLASSES = {"Crosswalk - Plain", "Lane Marking - General", "Building", "Vegetation"}

# 'Thing' classes: already individually instanced by the model's own
# panoptic post-processing — no further splitting needed.
THING_CLASSES = {
    "Traffic Sign Frame", "Traffic Sign (Back)", "Traffic Sign (Front)",
    "Traffic Light", "Street Light", "Pole", "Utility Pole", "Banner", "Billboard"
}

OBJECT_NODE_TYPES = ["signage", "light_pole", "road_marking", "building", "vegetation"]
POLE_FAMILY_TYPES = ["signage", "light_pole"]

# Stable per-node-type class vocabularies — the integer index stored per
# node is looked up against these; the actual embedding happens in the
# model (06), not here.
CLASS_IDX = {
    "signage": {"Traffic Sign Frame": 0, "Traffic Sign (Back)": 1, "Traffic Sign (Front)": 2, "Banner": 3, "Billboard": 4},
    "light_pole": {"Traffic Light": 0, "Street Light": 1, "Pole": 2, "Utility Pole": 3},
    "road_marking": {"Crosswalk - Plain": 0, "Lane Marking - General": 1},
    "building": {"Building": 0},
    "vegetation": {"Vegetation": 0},
}


def extract_objects(seg_map, segments_info, min_area_fraction, min_area_fraction_by_type=None):
    """min_area_fraction_by_type: optional dict node_type -> override fraction.
    Falls back to min_area_fraction if a node_type isn't in the dict."""
    total_px = seg_map.size
    min_area_fraction_by_type = min_area_fraction_by_type or {}
    min_area_px_default = min_area_fraction * total_px
    objects = {nt: [] for nt in OBJECT_NODE_TYPES}

    def _record(node_type, class_name, mask):
        area = int(mask.sum())
        threshold_frac = min_area_fraction_by_type.get(node_type, min_area_fraction)
        min_area_px = threshold_frac * total_px
        if area < min_area_px:
            return
        ys, xs = np.nonzero(mask)
        objects[node_type].append({
            "class_name": class_name,
            "cx_px": float(xs.mean()), "cy_px": float(ys.mean()),
            "area_px": area,
            "bbox": (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        })

    # Native "thing" instances — one segment already = one instance.
    for s in segments_info:
        cls = s["class_name"]
        if cls in THING_CLASSES and not s["is_stuff"]:
            mask = (seg_map == s["id"])
            _record(CLASS_TO_NODE_TYPE[cls], cls, mask)

    # Fused "stuff" classes — split via connected-component labeling
    # (no dilation: dashed lane markings intentionally fragment into
    # multiple nodes rather than being merged into one).
    for s in segments_info:
        cls = s["class_name"]
        if cls in ISLAND_CLASSES and s["is_stuff"]:
            base_mask = (seg_map == s["id"])
            labeled, n_islands = ndimage.label(base_mask)
            for island_id in range(1, n_islands + 1):
                _record(CLASS_TO_NODE_TYPE[cls], cls, labeled == island_id)

    return objects


def _bbox_overlap_ratio(bbox_a, bbox_b):
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter_area = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    smaller = min(area_a, area_b)
    return float(inter_area / smaller) if smaller > 0 else 0.0


def build_sees_edges(objects, w, h, ego_pos_frac=(0.5, 0.93)):
    """Ego -> every object, bidirectional. Edge features: [area_norm, dist_norm].
    Distance uses raw pixel distance / diagonal (aspect-ratio safe) —
    NOT per-axis-normalized distance, which distorts on non-square images.
    """
    diag = math.hypot(w, h)
    total_px = w * h
    ego_x_px, ego_y_px = ego_pos_frac[0] * w, ego_pos_frac[1] * h

    edges = {}
    for node_type in OBJECT_NODE_TYPES:
        objs = objects[node_type]
        n = len(objs)
        if n == 0:
            edges[node_type] = (
                torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 2), dtype=torch.float),
                torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 2), dtype=torch.float),
            )
            continue

        area_norm = [o["area_px"] / total_px for o in objs]
        dist_norm = [math.hypot(o["cx_px"] - ego_x_px, o["cy_px"] - ego_y_px) / diag for o in objs]
        attr = torch.tensor(list(zip(area_norm, dist_norm)), dtype=torch.float)

        fwd_index = torch.tensor([[0] * n, list(range(n))], dtype=torch.long)   # ego -> obj
        rev_index = torch.tensor([list(range(n)), [0] * n], dtype=torch.long)   # obj -> ego

        edges[node_type] = (fwd_index, attr, rev_index, attr)

    return edges


def build_mounted_with_edges(objects, w, h, distance_threshold):
    """Signage<->Signage, Light_pole<->Light_pole, Signage<->Light_pole.
    Triggered by bbox overlap OR centroid distance <= threshold (fraction
    of diagonal). Edge feature: bbox overlap ratio (0.0 if distance-only)."""
    diag = math.hypot(w, h)
    results = {}

    def _pairs(type_a, type_b):
        objs_a, objs_b = objects[type_a], objects[type_b]
        src, dst, attr = [], [], []
        for i, oa in enumerate(objs_a):
            for j, ob in enumerate(objs_b):
                if type_a == type_b and i == j:
                    continue
                dist = math.hypot(oa["cx_px"] - ob["cx_px"], oa["cy_px"] - ob["cy_px"]) / diag
                overlap = _bbox_overlap_ratio(oa["bbox"], ob["bbox"])
                if overlap > 0 or dist <= distance_threshold:
                    src.append(i); dst.append(j); attr.append(overlap)
        if src:
            return (torch.tensor([src, dst], dtype=torch.long),
                    torch.tensor(attr, dtype=torch.float).unsqueeze(1))
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float)

    results[("signage", "signage")] = _pairs("signage", "signage")
    results[("light_pole", "light_pole")] = _pairs("light_pole", "light_pole")
    results[("signage", "light_pole")] = _pairs("signage", "light_pole")
    return results


def build_near_edges(objects, w, h, cutoff_d):
    """All pairwise + self combinations among the 5 object node types.
    Triggered by centroid distance <= cutoff_d (fraction of diagonal).
    Edge feature: [dist_norm]."""
    diag = math.hypot(w, h)
    results = {}

    def _pairs(type_a, type_b):
        objs_a, objs_b = objects[type_a], objects[type_b]
        src, dst, attr = [], [], []
        for i, oa in enumerate(objs_a):
            for j, ob in enumerate(objs_b):
                if type_a == type_b and i == j:
                    continue
                dist = math.hypot(oa["cx_px"] - ob["cx_px"], oa["cy_px"] - ob["cy_px"]) / diag
                if dist <= cutoff_d:
                    src.append(i); dst.append(j); attr.append(dist)
        if src:
            return (torch.tensor([src, dst], dtype=torch.long),
                    torch.tensor(attr, dtype=torch.float).unsqueeze(1))
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float)

    for i, ta in enumerate(OBJECT_NODE_TYPES):
        for tb in OBJECT_NODE_TYPES[i:]:
            results[(ta, tb)] = _pairs(ta, tb)
    return results


def assemble_svg(objects, w, h, svf, enclosure, entropy, near_cutoff_d, mounted_with_threshold):
    """Builds and returns the full SVG HeteroData object for one point."""
    data = HeteroData()

    data["ego"].x = torch.tensor([[svf, enclosure, entropy, 0.5, 0.93]], dtype=torch.float)

    total_px = w * h
    for node_type in OBJECT_NODE_TYPES:
        objs = objects[node_type]
        if len(objs) == 0:
            data[node_type].x = torch.zeros((0, 2), dtype=torch.float)
            data[node_type].class_idx = torch.zeros((0,), dtype=torch.long)
            continue
        feats = [[o["cx_px"] / w, o["cy_px"] / h] for o in objs]  # per-axis-normalized position (feature only)
        idxs = [CLASS_IDX[node_type][o["class_name"]] for o in objs]
        data[node_type].x = torch.tensor(feats, dtype=torch.float)
        data[node_type].class_idx = torch.tensor(idxs, dtype=torch.long)
        data[node_type].area_norm = torch.tensor(
            [o["area_px"] / total_px for o in objs], dtype=torch.float
        ).unsqueeze(1)

    sees = build_sees_edges(objects, w, h)
    for node_type, (fwd_idx, fwd_attr, rev_idx, rev_attr) in sees.items():
        data["ego", "sees", node_type].edge_index = fwd_idx
        data["ego", "sees", node_type].edge_attr = fwd_attr
        data[node_type, "sees", "ego"].edge_index = rev_idx
        data[node_type, "sees", "ego"].edge_attr = rev_attr

    mounted = build_mounted_with_edges(objects, w, h, mounted_with_threshold)
    for (ta, tb), (idx, attr) in mounted.items():
        data[ta, "mounted_with", tb].edge_index = idx
        data[ta, "mounted_with", tb].edge_attr = attr
        if ta != tb:
            data[tb, "mounted_with", ta].edge_index = idx.flip(0)
            data[tb, "mounted_with", ta].edge_attr = attr

    near = build_near_edges(objects, w, h, near_cutoff_d)
    for (ta, tb), (idx, attr) in near.items():
        data[ta, "near", tb].edge_index = idx
        data[ta, "near", tb].edge_attr = attr
        if ta != tb:
            data[tb, "near", ta].edge_index = idx.flip(0)
            data[tb, "near", ta].edge_attr = attr

    return data

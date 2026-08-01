"""
SVG QC overlay: segmentation + graph drawn directly on top of the
unmodified original SVI. No title, no legend — mapping documented in
docs/svg_visualization_legend.md instead.

Edges are drawn directly from the already-assembled HeteroData object's
edge_index tensors, not recomputed here — guarantees the visualization
can never silently drift from the actual graph being saved.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import segmentation as seg


NODE_RELEVANT_CLASSES = {
    "Building", "Vegetation", "Crosswalk - Plain", "Lane Marking - General",
    "Traffic Sign Frame", "Traffic Sign (Back)", "Traffic Sign (Front)",
    "Traffic Light", "Street Light", "Pole", "Utility Pole",
}

# class_name -> official Mapillary RGB (0-1 float), reused so the graph
# layer visually matches the segmentation layer beneath it.
_NAME_TO_IDX = {name: i for i, name in enumerate(seg.MAPILLARY_NAMES)}
NODE_COLOR = {
    name: tuple(c / 255 for c in seg.MAPILLARY_PALETTE[_NAME_TO_IDX[name]])
    for name in NODE_RELEVANT_CLASSES
}

def _contrasting_outline(rgb_0_1):
    """Picks black or white outline based on the fill color's perceived
    luminance, so the outline stays visible whether the node's own class
    color is dark (e.g. Utility Pole) or light/white (e.g. Lane Marking -
    General) — a fixed single outline color fails on one end or the other."""
    r, g, b = rgb_0_1
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.5 else "white"


EGO_COLOR = "black"
EGO_MARKER = "*"
EGO_MARKER_SIZE = 650
EGO_OUTLINE = _contrasting_outline((0, 0, 0))  # white

EDGE_STYLE = {
    "sees": dict(linestyle="-", alpha=0.75, linewidth=2.0, color="white"),       # was 1.0
    "mounted_with": dict(linestyle="--", alpha=0.9, linewidth=2.6, color="yellow"),  # was 1.3
    "near": dict(linestyle=":", alpha=0.35, linewidth=1.6, color="cyan"),       # was 0.8
}

OBJECT_NODE_TYPES = ["signage", "light_pole", "road_marking", "building", "vegetation"]


def render_svg_overlay(image, seg_map, segments_info, objects, data, w, h, seg_alpha=0.55):
    """
    image: PIL image (original SVI, unmodified)
    objects: dict from svg_builder.extract_objects — gives node positions/classes
    data: the assembled HeteroData for this point — gives real edge_index tensors
    """
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.imshow(np.array(image))

    # ── Segmentation overlay: only node-relevant classes ───────────────
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for s in segments_info:
        if s["class_name"] in NODE_RELEVANT_CLASSES:
            color = NODE_COLOR[s["class_name"]]
            mask = seg_map == s["id"]
            overlay[mask, 0], overlay[mask, 1], overlay[mask, 2] = color
            overlay[mask, 3] = seg_alpha
    ax.imshow(overlay)

    ego_xy = (0.5 * w, 0.93 * h)

    def _pos(node_type, local_idx):
        o = objects[node_type][local_idx]
        return o["cx_px"], o["cy_px"]

    # ── Edges — drawn from the real, already-built graph's edge_index ──
    for node_type in OBJECT_NODE_TYPES:
        key = ("ego", "sees", node_type)
        if key in data.edge_index_dict:
            idx = data.edge_index_dict[key].t().tolist()
            for _, obj_i in idx:  # ego is always index 0; only draw once, not the reverse too
                ox, oy = _pos(node_type, obj_i)
                ax.plot([ego_xy[0], ox], [ego_xy[1], oy], **EDGE_STYLE["sees"], zorder=2)

    for relation in ["mounted_with", "near"]:
        for i, ta in enumerate(OBJECT_NODE_TYPES):
            for tb in OBJECT_NODE_TYPES[i:]:
                key = (ta, relation, tb)
                if key in data.edge_index_dict:
                    idx = data.edge_index_dict[key].t().tolist()
                    for a, b in idx:
                        if ta == tb and a >= b:
                            continue  # same-type self-pairs stored both directions — draw once
                        ax_, ay_ = _pos(ta, a)
                        bx_, by_ = _pos(tb, b)
                        ax.plot([ax_, bx_], [ay_, by_], **EDGE_STYLE[relation], zorder=3)

    # ── Nodes — solid opaque fill, contrast-aware outline so the marker
    #    never blends into its own (matching-colored) segmentation region
    #    or into a dark/light background ─────────────────────────────────
    ax.scatter([ego_xy[0]], [ego_xy[1]], marker=EGO_MARKER, s=EGO_MARKER_SIZE,
               color=EGO_COLOR, zorder=5, edgecolors=EGO_OUTLINE, linewidths=1.6)

    for node_type in OBJECT_NODE_TYPES:
        for o in objects[node_type]:
            color = NODE_COLOR.get(o["class_name"], (1, 0, 0))
            outline = _contrasting_outline(color)
            ax.scatter([o["cx_px"]], [o["cy_px"]], marker="o", s=160,
                       color=color, zorder=4, edgecolors=outline, linewidths=1.5)

    return fig


def save_overlay(fig, out_dir, point_id):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{point_id}.png", dpi=100)
    plt.close(fig)

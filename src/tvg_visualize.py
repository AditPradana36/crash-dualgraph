"""
TVG QC visualization: a fixed-extent, fixed-pixel-size geographic map per
incident. Deliberately minimal (incident, building footprints, isovist,
peer incidents only — no intersections, no edges) for rendering speed
across the whole dataset.

Fast by construction: buildings are queried from the already-cached,
in-memory GeoDataFrame's spatial index for a small fixed window, never
re-fetched or re-projected per point. No basemap tiles (no network calls).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

PLOT_HALF_EXTENT_M = 75  # fixed real-world window: incident +/- 75m each direction
FIGSIZE_PX = 500         # fixed output size for every point, regardless of content
DPI = 100

INCLUDED_BUILDING_COLOR = "red"
EXCLUDED_BUILDING_COLOR = "lightgray"
ISOVIST_COLOR = "#1f78ff"
INCIDENT_POSITIVE_COLOR = "red"
INCIDENT_NEGATIVE_COLOR = "blue"
PEER_COLOR = "red"  # peers are always positive-labeled — never a separate color


def render_tvg_overlay(origin_xy, label, polygon, included_building_ids,
                        buildings_gdf, buildings_sindex, peer_xy_list):
    """
    origin_xy: (x, y) of the incident, in the same projected CRS as buildings_gdf.
    label: 1 (positive/red star) or 0 (negative/blue star).
    polygon: the incident's isovist polygon (shapely).
    included_building_ids: indices into buildings_gdf that bounded the isovist.
    buildings_sindex: buildings_gdf.sindex — reused across every call, not rebuilt.
    peer_xy_list: list of (x, y) for this incident's crash_history peers
                  (already guaranteed positive-only upstream in tvg_builder.py).
    """
    ox_, oy_ = origin_xy
    half = PLOT_HALF_EXTENT_M

    fig = plt.figure(figsize=(FIGSIZE_PX / DPI, FIGSIZE_PX / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(ox_ - half, ox_ + half)
    ax.set_ylim(oy_ - half, oy_ + half)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Fast local building query: bbox against the spatial index, not the
    #    whole city layer — this is what keeps per-point rendering fast
    #    regardless of total dataset size. ─────────────────────────────
    window_bbox = (ox_ - half, oy_ - half, ox_ + half, oy_ + half)
    candidate_idx = list(buildings_sindex.intersection(window_bbox))
    included_set = set(included_building_ids)

    for idx in candidate_idx:
        geom = buildings_gdf.geometry.iloc[idx]
        color = INCLUDED_BUILDING_COLOR if idx in included_set else EXCLUDED_BUILDING_COLOR
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for p in polys:
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, facecolor=color, edgecolor="none", alpha=0.7, zorder=2)

    # ── Isovist ──────────────────────────────────────────────────────────
    if polygon is not None and not polygon.is_empty:
        ix, iy = polygon.exterior.xy
        ax.fill(ix, iy, facecolor=ISOVIST_COLOR, edgecolor=ISOVIST_COLOR, alpha=0.20,
                 linewidth=1.2, zorder=1)

    # ── Peer incidents — always drawn red; negatives can never be peers,
    #    so this is also a visible sanity check on that rule. ─────────────
    if peer_xy_list:
        px = [p[0] for p in peer_xy_list]
        py = [p[1] for p in peer_xy_list]
        ax.scatter(px, py, marker="o", s=25, color=PEER_COLOR, edgecolors="black",
                   linewidths=0.5, zorder=3)

    # ── The incident itself ────────────────────────────────────────────
    incident_color = INCIDENT_POSITIVE_COLOR if label == 1 else INCIDENT_NEGATIVE_COLOR
    ax.scatter([ox_], [oy_], marker="*", s=300, color=incident_color,
               edgecolors="black", linewidths=0.8, zorder=4)

    return fig


def save_overlay(fig, out_dir, point_id):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{point_id}.png", dpi=DPI)
    plt.close(fig)

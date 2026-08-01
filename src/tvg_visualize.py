"""
TVG QC visualization: a fixed-extent, fixed-pixel-size geographic map per
incident. Draws incident, building footprints, isovist, peer incidents,
included intersections, and the HeteroData graph edges (anchors, adjacent,
connects, fronts, on_segment) so the QC image reflects the actual graph
that was built, not just the underlying geometry.

Fast by construction: buildings are queried from the already-cached,
in-memory GeoDataFrame's spatial index for a small fixed window, never
re-fetched or re-projected per point. No basemap tiles (no network calls).
Edge drawing is O(edges in this point's subgraph), which is small and
bounded by isovist_radius_m — negligible next to the building fill cost.
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

INTERSECTION_COLOR = "black"
BACKGROUND_ROAD_COLOR = "#cccccc"  # full underlying road network, context only
ROAD_EDGE_COLOR = "#666666"       # connects: real OSM road edges (included subgraph)
ANCHOR_EDGE_COLOR = "#999999"     # anchors: incident <-> building/intersection
ADJACENT_EDGE_COLOR = "orange"    # adjacent: building <-> building clique
FRONTS_EDGE_COLOR = "green"       # fronts: building -> nearest intersection
ON_SEGMENT_COLOR = "black"        # on_segment: incident <-> u, v (the road it sits on)


def render_tvg_overlay(origin_xy, label, polygon, included_building_ids,
                        buildings_gdf, buildings_sindex, peer_xy_list,
                        G=None, data=None, included_intersections=None,
                        u=None, v=None):
    """
    origin_xy: (x, y) of the incident, in the same projected CRS as buildings_gdf.
    label: 1 (positive/red star) or 0 (negative/blue star).
    polygon: the incident's isovist polygon (shapely).
    included_building_ids: indices into buildings_gdf that bounded the isovist.
    buildings_sindex: buildings_gdf.sindex — reused across every call, not rebuilt.
    peer_xy_list: list of (x, y) for this incident's crash_history peers
                  (already guaranteed positive-only upstream in tvg_builder.py).

    Optional graph-overlay args (all-or-nothing: pass all four of G, data,
    included_intersections, u, v to draw intersections + HeteroData edges;
    omitting them reproduces the original minimal overlay unchanged):
    G: the OSMnx/networkx road graph (same object passed to process_incident).
    data: the HeteroData object returned by process_incident for this point.
    included_intersections: meta["included_intersections"] — node IDs, in the
                             same order used to build data's intersection nodes.
    u, v: meta["u"], meta["v"] — the road segment endpoints the incident sits on.
    """
    ox_, oy_ = origin_xy
    half = PLOT_HALF_EXTENT_M

    fig = plt.figure(figsize=(FIGSIZE_PX / DPI, FIGSIZE_PX / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(ox_ - half, ox_ + half)
    ax.set_ylim(oy_ - half, oy_ + half)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_graph = G is not None and data is not None and included_intersections is not None

    # Local xy lookup for every included intersection, keyed by its position
    # in included_intersections — this order matches the node index used
    # throughout data["intersection", ...] edge_index tensors.
    inter_xy = {}
    if draw_graph:
        for nid in included_intersections:
            node_d = G.nodes[nid]
            inter_xy[nid] = (node_d["x"], node_d["y"])
        inter_local_xy = [inter_xy[nid] for nid in included_intersections]

    # ── Fast local building query: bbox against the spatial index, not the
    #    whole city layer — this is what keeps per-point rendering fast
    #    regardless of total dataset size. ─────────────────────────────
    window_bbox = (ox_ - half, oy_ - half, ox_ + half, oy_ + half)
    candidate_idx = list(buildings_sindex.intersection(window_bbox))
    included_set = set(included_building_ids)

    building_centroid = {}
    for idx in candidate_idx:
        geom = buildings_gdf.geometry.iloc[idx]
        color = INCLUDED_BUILDING_COLOR if idx in included_set else EXCLUDED_BUILDING_COLOR
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for p in polys:
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, facecolor=color, edgecolor="none", alpha=0.7, zorder=2)
        if idx in included_set:
            c = geom.centroid
            building_centroid[idx] = (c.x, c.y)

    # ── Full underlying road network (context only) — every edge in G whose
    #    endpoints fall in this point's plot window, regardless of whether it
    #    made it into the HeteroData subgraph. Drawn first / lowest z-order so
    #    the "included" connects edges still stand out on top of it. Filtering
    #    by node coordinate bbox (not a spatial index) is cheap: G.nodes is a
    #    plain dict, and this window is tiny relative to the whole city graph.
    if G is not None:
        xmin, xmax = ox_ - half, ox_ + half
        ymin, ymax = oy_ - half, oy_ + half
        for a, b, ed in G.edges(data=True):
            ax_, ay_ = G.nodes[a]["x"], G.nodes[a]["y"]
            bx_, by_ = G.nodes[b]["x"], G.nodes[b]["y"]
            if (xmin <= ax_ <= xmax or xmin <= bx_ <= xmax) and \
               (ymin <= ay_ <= ymax or ymin <= by_ <= ymax):
                ax.plot([ax_, bx_], [ay_, by_], color=BACKGROUND_ROAD_COLOR,
                        linewidth=1.0, alpha=0.9, zorder=0.5)

    # ── Isovist ──────────────────────────────────────────────────────────
    if polygon is not None and not polygon.is_empty:
        ix, iy = polygon.exterior.xy
        ax.fill(ix, iy, facecolor=ISOVIST_COLOR, edgecolor=ISOVIST_COLOR, alpha=0.20,
                 linewidth=1.2, zorder=1)

    # ── Graph edges (drawn under nodes, above isovist fill) ───────────────
    if draw_graph:
        # connects: real OSM road edges among included intersections
        conn_key = ("intersection", "connects", "intersection")
        if conn_key in data.edge_types:
            ei = data[conn_key].edge_index
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                sx, sy = inter_local_xy[s]
                dx, dy = inter_local_xy[d]
                ax.plot([sx, dx], [sy, dy], color=ROAD_EDGE_COLOR, linewidth=1.2,
                        alpha=0.8, zorder=1.5)

        # anchors: incident <-> building
        anchor_b_key = ("incident", "anchors", "building")
        if anchor_b_key in data.edge_types and included_building_ids:
            ei = data[anchor_b_key].edge_index
            for _, d in zip(ei[0].tolist(), ei[1].tolist()):
                bid = included_building_ids[d]
                if bid in building_centroid:
                    bx, by = building_centroid[bid]
                    ax.plot([ox_, bx], [oy_, by], color=ANCHOR_EDGE_COLOR, linewidth=0.6,
                            alpha=0.5, zorder=1.5, linestyle=":")

        # anchors: incident <-> intersection
        anchor_i_key = ("incident", "anchors", "intersection")
        if anchor_i_key in data.edge_types and included_intersections:
            ei = data[anchor_i_key].edge_index
            for _, d in zip(ei[0].tolist(), ei[1].tolist()):
                ix_, iy_ = inter_local_xy[d]
                ax.plot([ox_, ix_], [oy_, iy_], color=ANCHOR_EDGE_COLOR, linewidth=0.6,
                        alpha=0.5, zorder=1.5, linestyle=":")

        # adjacent: building <-> building clique
        adj_key = ("building", "adjacent", "building")
        if adj_key in data.edge_types and included_building_ids:
            ei = data[adj_key].edge_index
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                if s < d:  # clique is symmetric; draw each pair once
                    bs, bd = included_building_ids[s], included_building_ids[d]
                    if bs in building_centroid and bd in building_centroid:
                        sx, sy = building_centroid[bs]
                        dx, dy = building_centroid[bd]
                        ax.plot([sx, dx], [sy, dy], color=ADJACENT_EDGE_COLOR, linewidth=0.5,
                                alpha=0.4, zorder=1.5)

        # fronts: building -> nearest included intersection
        fronts_key = ("building", "fronts", "intersection")
        if fronts_key in data.edge_types and included_building_ids:
            ei = data[fronts_key].edge_index
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                bid = included_building_ids[s]
                if bid in building_centroid:
                    bx, by = building_centroid[bid]
                    ix_, iy_ = inter_local_xy[d]
                    ax.plot([bx, ix_], [by, iy_], color=FRONTS_EDGE_COLOR, linewidth=0.6,
                            alpha=0.5, zorder=1.5, linestyle="--")

        # on_segment: incident <-> u, v (the road segment it sits on)
        if u is not None and v is not None and u in inter_xy and v in inter_xy:
            ux, uy = inter_xy[u]
            vx, vy = inter_xy[v]
            ax.plot([ox_, ux], [oy_, uy], color=ON_SEGMENT_COLOR, linewidth=1.5,
                    alpha=0.9, zorder=1.6)
            ax.plot([ox_, vx], [oy_, vy], color=ON_SEGMENT_COLOR, linewidth=1.5,
                    alpha=0.9, zorder=1.6)

        # intersection nodes themselves
        if inter_local_xy:
            nx_, ny_ = zip(*inter_local_xy)
            ax.scatter(nx_, ny_, marker="s", s=18, color=INTERSECTION_COLOR,
                       edgecolors="white", linewidths=0.4, zorder=3)

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

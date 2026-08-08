"""
Builds one TVG (Top View Graph) HeteroData object per point, using
osm_fetch.py's cached global layer and isovist.py's ray-casting.

Node types: incident, building, intersection, peer_incident.
"""
import math

import numpy as np
import torch
from torch_geometric.data import HeteroData
from shapely.geometry import Point

import geo_utils
import isovist as iso


def _safe_float(value, default=0.0):
    """Parses an OSM tag value (often a messy string) to float. Returns
    (value, is_missing_flag) — never raises, never silently fabricates
    a plausible-looking number beyond the flagged default."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default, 1.0
    try:
        return float(str(value).split(";")[0].replace("m", "").strip()), 0.0
    except (ValueError, TypeError):
        return default, 1.0


def process_incident(
    incident_row, reconciled_df, buildings_gdf, building_type_vocab, highway_vocab, G,
    betweenness, orientation_entropy, tree, boundaries, utm_crs,
    isovist_radius_m, n_rays, crash_history_threshold_m, adjacent_k_nearest=5,
):
    """Returns (data: HeteroData, meta: dict) for one incident point.
    meta carries plotting-relevant info reused by tvg_visualize.py so
    nothing needs recomputing for the QC image.
    """
    lat, lon = incident_row["input_lat"], incident_row["input_lon"]
    point_utm = geo_utils.project_point(lat, lon, utm_crs)
    origin_xy = (point_utm.x, point_utm.y)

    # ── Road projection ──────────────────────────────────────────────
    u, v, key, edge_geom, edge_data = geo_utils.nearest_road_edge(G, origin_xy)
    _, fraction_along = geo_utils.project_onto_edge(origin_xy, edge_geom)

    # ── Isovist ──────────────────────────────────────────────────────
    polygon, hit_building_idx, ray_hit_flags = iso.compute_isovist(
        origin_xy, tree, boundaries, isovist_radius_m, n_rays
    )
    isovist_area = iso.isovist_area(polygon)
    isovist_compactness = iso.isovist_compactness(polygon)
    isovist_occlusivity = iso.isovist_occlusivity(ray_hit_flags)

    # ── Included buildings: exactly what bounded the isovist ─────────
    included_building_ids = sorted(hit_building_idx)

    # ── Included intersections: STRICTLY inside the isovist polygon ──
    # REVISED: u/v (the nearest road edge's endpoints) are NO LONGER
    # force-included when they fall outside the isovist. "intersection"
    # now means exactly what it says -- a node the isovist itself can
    # see -- matching the same inclusion rule already used for buildings,
    # with no special exemption for the road-projection anchors. This can
    # leave u and/or v out of the graph entirely; on_segment (below)
    # handles that by only connecting to whichever of the two actually
    # qualify, rather than assuming both always do.
    included_intersections = set()
    for node_id, data_n in G.nodes(data=True):
        pt = Point(data_n["x"], data_n["y"])
        if polygon.covers(pt):
            included_intersections.add(node_id)
    included_intersections = sorted(included_intersections)
    node_local_idx = {nid: i for i, nid in enumerate(included_intersections)}

    # ── Peer incidents: same-city, POSITIVE-labeled only, within threshold ──
    # Precomputed once here, at dataset-construction time (04) -- baked
    # into the saved .pt graph, same as every other TVG relation, not
    # deferred to training time.
    #
    # No fold/split constraint (01 no longer assigns fold_rep{r} columns
    # at all). This DOES mean a point's peer_incident set can straddle
    # whatever train/val/test split a downstream 07 run later draws --
    # crash_history/peer_incident is opt-in (ablation scenarios B+
    # through F+ only; never touched by the default A-F/G scenarios),
    # and that leakage tradeoff was accepted explicitly rather than left
    # as an unexamined side effect of removing folds.
    #
    # same-city: without this, e.g. a Bogor point and a Somerville point
    # could be matched as "peers" purely by both being positive-labeled
    # within threshold -- UTM coordinates aren't comparable across
    # cities/CRSes, so this filter isn't optional the way it might look.
    candidates = reconciled_df[
        (reconciled_df["point_id"] != incident_row["point_id"])
        & (reconciled_df["label"] == 1)  # HARD RULE: peers are never negative points
    ]
    if "city" in reconciled_df.columns:
        candidates = candidates[candidates["city"] == incident_row["city"]]

    if len(candidates):
        dists = np.hypot(candidates["_utm_x"] - origin_xy[0], candidates["_utm_y"] - origin_xy[1])
        peers = candidates[dists.values <= crash_history_threshold_m]
    else:
        peers = candidates

    # ── Assemble HeteroData ──────────────────────────────────────────
    data = HeteroData()

    data["incident"].x = torch.tensor(
        [[fraction_along, isovist_area, isovist_compactness, isovist_occlusivity]],
        dtype=torch.float,
    )
    # NEW: the incident's own road type — a direct node feature, not left to
    # arrive solely via on_segment's edge attribute. Not an ablation: negatives
    # were deliberately sampled to match positives' highway-type distribution,
    # so this doesn't carry the circularity risk that put crash_history behind
    # an ablation gate.
    incident_hw = edge_data.get("highway", "unclassified")
    if isinstance(incident_hw, list):
        incident_hw = incident_hw[0]
    incident_hw_idx = highway_vocab.get(incident_hw, highway_vocab["unclassified"])
    data["incident"].highway_type_idx = torch.tensor([incident_hw_idx], dtype=torch.long)

    if included_building_ids:
        rows = buildings_gdf.loc[included_building_ids]
        feats, type_idx, height_vals, height_flags, levels_vals, levels_flags = [], [], [], [], [], []
        for _, r in rows.iterrows():
            feats.append([r["area"], r["perimeter"], r["circular_compactness"],
                          r["elongation"], r["orientation"], r["shape_index"]])
            type_idx.append(int(r["type_idx"]))
            hv, hf = _safe_float(r.get("height"))
            lv, lf = _safe_float(r.get("building:levels"))
            height_vals.append(hv); height_flags.append(hf)
            levels_vals.append(lv); levels_flags.append(lf)

        data["building"].x = torch.tensor(feats, dtype=torch.float)
        data["building"].type_idx = torch.tensor(type_idx, dtype=torch.long)
        data["building"].height = torch.tensor(height_vals, dtype=torch.float).unsqueeze(1)
        data["building"].height_missing = torch.tensor(height_flags, dtype=torch.float).unsqueeze(1)
        data["building"].levels = torch.tensor(levels_vals, dtype=torch.float).unsqueeze(1)
        data["building"].levels_missing = torch.tensor(levels_flags, dtype=torch.float).unsqueeze(1)
    else:
        data["building"].x = torch.zeros((0, 6), dtype=torch.float)
        data["building"].type_idx = torch.zeros((0,), dtype=torch.long)
        for attr in ["height", "height_missing", "levels", "levels_missing"]:
            data["building"][attr] = torch.zeros((0, 1), dtype=torch.float)

    inter_feats = [[betweenness.get(nid, 0.0), orientation_entropy.get(nid, 0.0)]
                   for nid in included_intersections]
    data["intersection"].x = torch.tensor(inter_feats, dtype=torch.float) if inter_feats else torch.zeros((0, 2), dtype=torch.float)

    n_peers = len(peers)
    data["peer_incident"].x = torch.ones((n_peers, 1), dtype=torch.float)  # constant placeholder — no real attributes

    # ── Edges ─────────────────────────────────────────────────────────
    diag_placeholder = 1.0  # distances here are real meters, no diagonal normalization needed (metric CRS)

    def _dist(ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    # anchors: incident <-> every included building & intersection, bidirectional
    if included_building_ids:
        b_dists = [_dist(origin_xy[0], origin_xy[1], buildings_gdf.loc[bid].geometry.centroid.x,
                          buildings_gdf.loc[bid].geometry.centroid.y) for bid in included_building_ids]
        fwd = torch.tensor([[0] * len(included_building_ids), list(range(len(included_building_ids)))], dtype=torch.long)
        attr = torch.tensor(b_dists, dtype=torch.float).unsqueeze(1)
        data["incident", "anchors", "building"].edge_index = fwd
        data["incident", "anchors", "building"].edge_attr = attr
        data["building", "anchors", "incident"].edge_index = fwd.flip(0)
        data["building", "anchors", "incident"].edge_attr = attr
    else:
        data["incident", "anchors", "building"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["incident", "anchors", "building"].edge_attr = torch.zeros((0, 1), dtype=torch.float)
        data["building", "anchors", "incident"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["building", "anchors", "incident"].edge_attr = torch.zeros((0, 1), dtype=torch.float)

    if included_intersections:
        i_dists = [_dist(origin_xy[0], origin_xy[1], G.nodes[nid]["x"], G.nodes[nid]["y"])
                   for nid in included_intersections]
        fwd = torch.tensor([[0] * len(included_intersections), list(range(len(included_intersections)))], dtype=torch.long)
        attr = torch.tensor(i_dists, dtype=torch.float).unsqueeze(1)
        data["incident", "anchors", "intersection"].edge_index = fwd
        data["incident", "anchors", "intersection"].edge_attr = attr
        data["intersection", "anchors", "incident"].edge_index = fwd.flip(0)
        data["intersection", "anchors", "incident"].edge_attr = attr
    else:
        data["incident", "anchors", "intersection"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["incident", "anchors", "intersection"].edge_attr = torch.zeros((0, 1), dtype=torch.float)
        data["intersection", "anchors", "incident"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["intersection", "anchors", "incident"].edge_attr = torch.zeros((0, 1), dtype=torch.float)

    # adjacent: each building <-> its adjacent_k_nearest nearest OTHER
    # included buildings (REVISED from the old full clique among every
    # isovist-included building). A full clique scales quadratically and
    # treats "next door" the same as "across the isovist"; a k-NN graph
    # keeps that distinction, and real centroid distance (meters) replaces
    # the old constant 1.0 flag as the edge feature -- distance is exactly
    # what defines this neighborhood now, so it's worth recording per edge
    # rather than discarding. Directed per building (i's k nearest get an
    # edge i->neighbor); NOT forced symmetric -- if j is also among i's k
    # nearest and i is among j's, both directions appear naturally, but
    # neither is added just to mirror the other.
    n_b = len(included_building_ids)
    if n_b > 1:
        centroids = [(buildings_gdf.loc[bid].geometry.centroid.x, buildings_gdf.loc[bid].geometry.centroid.y)
                     for bid in included_building_ids]
        src, dst, attrs = [], [], []
        for i in range(n_b):
            neighbor_dists = sorted(
                ((_dist(*centroids[i], *centroids[j]), j) for j in range(n_b) if j != i),
                key=lambda t: t[0],
            )
            for d, j in neighbor_dists[:adjacent_k_nearest]:
                src.append(i); dst.append(j); attrs.append(d)
        data["building", "adjacent", "building"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["building", "adjacent", "building"].edge_attr = torch.tensor(attrs, dtype=torch.float).unsqueeze(1)
    else:
        data["building", "adjacent", "building"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["building", "adjacent", "building"].edge_attr = torch.zeros((0, 1), dtype=torch.float)

    # connects: REAL OSM edges among included intersections only
    src, dst, attrs = [], [], []
    for i, a in enumerate(included_intersections):
        for j, b in enumerate(included_intersections):
            if i == j:
                continue
            if G.has_edge(a, b):
                for key_ in G[a][b]:
                    ed = G[a][b][key_]
                    hw = ed.get("highway", "unclassified")
                    if isinstance(hw, list):
                        hw = hw[0]
                    hw_idx = highway_vocab.get(hw, highway_vocab["unclassified"])
                    maxspeed_val, maxspeed_missing = _safe_float(ed.get("maxspeed"))
                    oneway = 1.0 if ed.get("oneway") else 0.0
                    src.append(i); dst.append(j)
                    attrs.append([float(hw_idx), maxspeed_val, maxspeed_missing, oneway])
    if src:
        data["intersection", "connects", "intersection"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["intersection", "connects", "intersection"].edge_attr = torch.tensor(attrs, dtype=torch.float)
    else:
        data["intersection", "connects", "intersection"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["intersection", "connects", "intersection"].edge_attr = torch.zeros((0, 4), dtype=torch.float)

    # fronts: each building -> its nearest INCLUDED intersection
    if included_building_ids and included_intersections:
        src, dst, attrs = [], [], []
        for bi, bid in enumerate(included_building_ids):
            bx, by = buildings_gdf.loc[bid].geometry.centroid.x, buildings_gdf.loc[bid].geometry.centroid.y
            dists = [_dist(bx, by, G.nodes[nid]["x"], G.nodes[nid]["y"]) for nid in included_intersections]
            nearest_j = int(np.argmin(dists))
            src.append(bi); dst.append(nearest_j); attrs.append(dists[nearest_j])
        data["building", "fronts", "intersection"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["building", "fronts", "intersection"].edge_attr = torch.tensor(attrs, dtype=torch.float).unsqueeze(1)
    else:
        data["building", "fronts", "intersection"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["building", "fronts", "intersection"].edge_attr = torch.zeros((0, 1), dtype=torch.float)

    # on_segment: incident <-> whichever of u/v (the nearest road edge's
    # endpoints) actually fall inside the isovist. REVISED: u/v are no
    # longer force-included as intersection nodes (see above), so this
    # can now produce 0, 1, or 2 edges instead of always exactly 2 --
    # never crashes on a missing endpoint, it just has nothing to connect
    # to on that side. hw/maxspeed/oneway describe the road segment
    # itself (same for both endpoints); only distance and target node
    # differ per endpoint.
    hw = edge_data.get("highway", "unclassified")
    if isinstance(hw, list):
        hw = hw[0]
    hw_idx = highway_vocab.get(hw, highway_vocab["unclassified"])
    maxspeed_val, maxspeed_missing = _safe_float(edge_data.get("maxspeed"))
    oneway = 1.0 if edge_data.get("oneway") else 0.0

    seg_src, seg_dst, seg_attrs = [], [], []
    for endpoint in (u, v):
        if endpoint in node_local_idx:
            dist_endpoint = _dist(origin_xy[0], origin_xy[1], G.nodes[endpoint]["x"], G.nodes[endpoint]["y"])
            seg_src.append(0)
            seg_dst.append(node_local_idx[endpoint])
            seg_attrs.append([dist_endpoint, float(hw_idx), maxspeed_val, maxspeed_missing, oneway])

    if seg_src:
        seg_fwd = torch.tensor([seg_src, seg_dst], dtype=torch.long)
        seg_attr = torch.tensor(seg_attrs, dtype=torch.float)
        data["incident", "on_segment", "intersection"].edge_index = seg_fwd
        data["incident", "on_segment", "intersection"].edge_attr = seg_attr
        data["intersection", "on_segment", "incident"].edge_index = seg_fwd.flip(0)
        data["intersection", "on_segment", "incident"].edge_attr = seg_attr
    else:
        data["incident", "on_segment", "intersection"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["incident", "on_segment", "intersection"].edge_attr = torch.zeros((0, 5), dtype=torch.float)
        data["intersection", "on_segment", "incident"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["intersection", "on_segment", "incident"].edge_attr = torch.zeros((0, 5), dtype=torch.float)

    # crash_history (ablation): incident -> each peer, distance-weighted
    if n_peers:
        peer_dists = np.hypot(peers["_utm_x"].values - origin_xy[0], peers["_utm_y"].values - origin_xy[1])
        data["incident", "crash_history", "peer_incident"].edge_index = torch.tensor(
            [[0] * n_peers, list(range(n_peers))], dtype=torch.long
        )
        data["incident", "crash_history", "peer_incident"].edge_attr = torch.tensor(
            peer_dists, dtype=torch.float
        ).unsqueeze(1)
    else:
        data["incident", "crash_history", "peer_incident"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["incident", "crash_history", "peer_incident"].edge_attr = torch.zeros((0, 1), dtype=torch.float)

    meta = {
        "origin_xy": origin_xy, "polygon": polygon,
        "included_building_ids": included_building_ids,
        "included_intersections": included_intersections,
        "peer_point_ids": peers["point_id"].tolist() if n_peers else [],
        "u": u, "v": v,
    }
    return data, meta

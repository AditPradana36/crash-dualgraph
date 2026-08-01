"""
ONE-TIME fetch of the whole study area's building footprints and street
network (bounding-box extent, not per point), plus everything that can be
precomputed globally rather than recomputed per incident: building shape
metrics, building-type vocabulary, intersection betweenness centrality
and per-node orientation entropy, and the shared STRtree used by
isovist.py's ray-casting.

Called once at the top of 04_tvg_construction; every incident point
re-uses the cached result via load_cache().
"""
import json
import math
import pickle
from pathlib import Path

import numpy as np
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely.strtree import STRtree
from shapely.geometry import box


def fetch_study_area_layers(boundary_geojson, bbox_padding_m, network_type="drive"):
    """Fetches buildings + street network across the study area's BOUNDING
    BOX (not polygon), padded by bbox_padding_m. Returns everything
    projected to a shared local UTM CRS."""
    boundary = gpd.read_file(boundary_geojson)
    if boundary.crs is None:
        boundary = boundary.set_crs(epsg=4326)
    elif boundary.crs.to_epsg() != 4326:
        boundary = boundary.to_crs(epsg=4326)

    utm_crs = boundary.estimate_utm_crs()
    boundary_utm = boundary.to_crs(utm_crs)

    minx, miny, maxx, maxy = boundary_utm.total_bounds
    minx -= bbox_padding_m; miny -= bbox_padding_m
    maxx += bbox_padding_m; maxy += bbox_padding_m

    bbox_polygon_utm = box(minx, miny, maxx, maxy)
    bbox_polygon_wgs84 = gpd.GeoSeries([bbox_polygon_utm], crs=utm_crs).to_crs(epsg=4326).iloc[0]

    buildings = ox.features_from_polygon(bbox_polygon_wgs84, tags={"building": True})
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    buildings_utm = buildings.to_crs(utm_crs)

    G = ox.graph_from_polygon(bbox_polygon_wgs84, network_type=network_type)
    G_utm = ox.project_graph(G, to_crs=utm_crs)

    return buildings_utm, G_utm, utm_crs


def compute_building_shape_metrics(buildings_gdf):
    """Adds area, perimeter, circular_compactness, elongation, orientation,
    shape_index columns — computed once for every fetched building."""
    geoms = buildings_gdf.geometry

    buildings_gdf["area"] = geoms.area
    buildings_gdf["perimeter"] = geoms.length
    buildings_gdf["circular_compactness"] = (
        4 * math.pi * buildings_gdf["area"] / buildings_gdf["perimeter"].clip(lower=1e-9) ** 2
    )

    min_rect = geoms.apply(lambda g: g.minimum_rotated_rectangle)
    def _rect_dims(rect):
        coords = list(rect.exterior.coords)
        side_a = math.dist(coords[0], coords[1])
        side_b = math.dist(coords[1], coords[2])
        return (min(side_a, side_b), max(side_a, side_b))

    dims = min_rect.apply(_rect_dims)
    buildings_gdf["elongation"] = dims.apply(lambda d: d[0] / d[1] if d[1] > 0 else 0.0)

    def _orientation(rect):
        coords = list(rect.exterior.coords)
        dx, dy = coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]
        return math.degrees(math.atan2(dy, dx)) % 180

    buildings_gdf["orientation"] = min_rect.apply(_orientation)

    rect_area = min_rect.area
    buildings_gdf["shape_index"] = buildings_gdf["area"] / rect_area.clip(lower=1e-9)

    return buildings_gdf


def build_building_type_vocab(buildings_gdf, tag_col="building"):
    """Stable categorical vocabulary across ALL fetched buildings, with
    an explicit 'unknown' category for untagged/missing values."""
    values = buildings_gdf[tag_col].fillna("unknown").astype(str)
    unique_vals = sorted(values.unique())
    if "unknown" not in unique_vals:
        unique_vals.append("unknown")
    vocab = {v: i for i, v in enumerate(unique_vals)}
    buildings_gdf["type_idx"] = values.map(vocab).fillna(vocab["unknown"]).astype(int)
    return buildings_gdf, vocab


def build_highway_vocab(G):
    """Stable categorical vocabulary across every 'highway' tag value in
    the fetched street network — built ONCE, globally. Must never be
    rebuilt per-incident: a locally-built vocab would assign a different
    integer to the same road type in different graphs, silently corrupting
    a shared embedding table downstream."""
    values = set()
    for _, _, data in G.edges(data=True):
        hw = data.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0]
        values.add(hw)
    unique_vals = sorted(values)
    if "unclassified" not in unique_vals:
        unique_vals.append("unclassified")
    return {hw: i for i, hw in enumerate(unique_vals)}


def compute_intersection_metrics(G, n_bearing_bins=8):
    """Betweenness centrality (exact, whole network) + per-node orientation
    entropy (over that node's own incident edge bearings) — both computed
    once for every intersection in the fetched network."""
    G_undirected = G.to_undirected() if G.is_directed() else G
    betweenness = nx.betweenness_centrality(G_undirected)

    orientation_entropy = {}
    for node in G.nodes:
        bearings = []
        for _, nbr, data in G.edges(node, data=True):
            x1, y1 = G.nodes[node]["x"], G.nodes[node]["y"]
            x2, y2 = G.nodes[nbr]["x"], G.nodes[nbr]["y"]
            bearing = math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360
            bearings.append(bearing)

        if not bearings:
            orientation_entropy[node] = 0.0
            continue

        bin_size = 360 / n_bearing_bins
        counts = [0] * n_bearing_bins
        for b in bearings:
            counts[min(int(b / bin_size), n_bearing_bins - 1)] += 1
        total = sum(counts)
        probs = np.array([c / total for c in counts if c > 0])
        orientation_entropy[node] = float(-np.sum(probs * np.log(probs))) if len(probs) else 0.0

    return betweenness, orientation_entropy


def build_building_strtree(buildings_gdf):
    boundaries = [geom.boundary for geom in buildings_gdf.geometry]
    tree = STRtree(boundaries)
    return tree, boundaries


def save_cache(cache_dir, buildings_gdf, building_type_vocab, highway_vocab, G, betweenness, orientation_entropy, utm_crs):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    buildings_gdf.to_parquet(cache_dir / "buildings.parquet")
    with open(cache_dir / "building_type_vocab.json", "w") as f:
        json.dump(building_type_vocab, f)
    with open(cache_dir / "highway_vocab.json", "w") as f:
        json.dump(highway_vocab, f)
    with open(cache_dir / "street_graph.pkl", "wb") as f:
        pickle.dump(G, f)
    with open(cache_dir / "intersection_metrics.json", "w") as f:
        json.dump({"betweenness": {str(k): v for k, v in betweenness.items()},
                   "orientation_entropy": {str(k): v for k, v in orientation_entropy.items()}}, f)
    with open(cache_dir / "utm_crs.txt", "w") as f:
        f.write(str(utm_crs))


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    buildings_gdf = gpd.read_parquet(cache_dir / "buildings.parquet")
    with open(cache_dir / "building_type_vocab.json") as f:
        building_type_vocab = json.load(f)
    with open(cache_dir / "highway_vocab.json") as f:
        highway_vocab = json.load(f)
    with open(cache_dir / "street_graph.pkl", "rb") as f:
        G = pickle.load(f)
    with open(cache_dir / "intersection_metrics.json") as f:
        raw = json.load(f)
        betweenness = {int(k): v for k, v in raw["betweenness"].items()}
        orientation_entropy = {int(k): v for k, v in raw["orientation_entropy"].items()}
    with open(cache_dir / "utm_crs.txt") as f:
        utm_crs = f.read().strip()
    return buildings_gdf, building_type_vocab, highway_vocab, G, betweenness, orientation_entropy, utm_crs


def cache_exists(cache_dir):
    cache_dir = Path(cache_dir)
    required = ["buildings.parquet", "building_type_vocab.json", "highway_vocab.json",
                "street_graph.pkl", "intersection_metrics.json", "utm_crs.txt"]
    return all((cache_dir / f).exists() for f in required)

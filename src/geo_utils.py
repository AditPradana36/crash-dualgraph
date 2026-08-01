"""
Shared CRS/geometry helpers used by both 01 (spatial fold assignment) and
04 (TVG construction). Extracted here per the "wait until 04 needs it"
plan — 01 gets refactored to import from here too, rather than keeping
its own inline copy.
"""
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import osmnx as ox


def estimate_utm_crs(gdf):
    """Auto-detects the appropriate local UTM CRS from a GeoDataFrame's extent."""
    return gdf.estimate_utm_crs()


def project_point(lat, lon, to_crs):
    """Projects a single lat/lon point into the given metric CRS."""
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(to_crs)
    return pt.iloc[0]


def nearest_road_edge(G_projected, point_xy):
    """point_xy: (x, y) already in the graph's projected CRS.
    Returns (u, v, key, edge_geometry_linestring)."""
    x, y = point_xy
    u, v, key = ox.distance.nearest_edges(G_projected, X=[x], Y=[y])[0]
    edge_data = G_projected.edges[u, v, key]
    if "geometry" in edge_data:
        geom = edge_data["geometry"]
    else:
        geom = LineString([
            (G_projected.nodes[u]["x"], G_projected.nodes[u]["y"]),
            (G_projected.nodes[v]["x"], G_projected.nodes[v]["y"]),
        ])
    return u, v, key, geom, edge_data


def project_onto_edge(point_xy, edge_geometry):
    """Returns (snapped_point, fraction_along_from_start)."""
    pt = Point(point_xy)
    snapped = nearest_points(edge_geometry, pt)[0]
    fraction = edge_geometry.project(snapped, normalized=True)
    return snapped, float(fraction)

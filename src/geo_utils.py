"""
Shared CRS/geometry helpers used by both 01 and 04.

Functions to implement:
- estimate_utm_crs(gdf) -> CRS
- project_point(lat, lon, to_crs) -> (x, y)
- nearest_road_edge(graph, point) -> (u, v, key, geometry)
- project_onto_edge(point, edge_geometry) -> (snapped_point, fraction_along)
"""

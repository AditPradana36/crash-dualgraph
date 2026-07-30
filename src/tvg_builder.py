"""
Builds one TVG (Top View Graph) HeteroData object per point, using
osm_fetch.py's cached layer and isovist.py's ray-casting.

Functions to implement:
- project_incident(point, street_graph) -> (u, v, fraction_along)
- included_buildings(hit_building_idx, buildings_gdf) -> GeoDataFrame
- included_intersections(isovist_polygon, street_nodes, u, v) -> GeoDataFrame
- build_anchors_edges(...) / build_adjacent_edges(...) / build_connects_edges(...)
- build_fronts_edges(...) / build_on_segment_edges(...)
- build_crash_history_edges(...)   # ablation only, same-fold, positive peers only
- assemble_tvg(...) -> HeteroData
"""

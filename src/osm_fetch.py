"""
ONE-TIME fetch of the whole study area's building footprints and street
network (bounding-box extent, not per point). Called once at the top of
04_tvg_construction; every incident point re-uses the cached result.

Functions to implement:
- fetch_study_area_buildings(boundary_geojson, bbox_padding_m) -> GeoDataFrame
- fetch_study_area_streets(boundary_geojson) -> networkx.MultiDiGraph
- build_building_strtree(buildings_gdf) -> STRtree
- cache_to_disk(...) / load_from_cache(...)   # interim/osm_cache/
"""

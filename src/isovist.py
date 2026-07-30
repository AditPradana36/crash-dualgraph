"""
Ray-casting isovist computation against the cached building STRtree
(see osm_fetch.py). One call per incident point.

Functions to implement:
- compute_isovist(origin_xy, tree, boundaries, radius_m, n_rays) -> (polygon, hit_building_idx)
- isovist_area(polygon) -> float
- isovist_compactness(polygon) -> float          # circularity ratio
- isovist_occlusivity(hit_building_idx, n_rays) -> float  # fraction of rays that hit a building
"""

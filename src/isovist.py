"""
Ray-casting isovist computation against a shared building STRtree, exactly
matching the user's validated check script — generalized into reusable
functions, with shape metrics (area, compactness, occlusivity) added.
"""
import math

import numpy as np
from shapely.geometry import Point, Polygon, LineString


def compute_isovist(origin_xy, tree, boundaries, radius, n_rays):
    """
    origin_xy: (x, y) in the same projected CRS as the building layer.
    tree: shapely.strtree.STRtree built over building boundaries (once, globally).
    boundaries: the list of boundary geometries the tree was built from
                (needed to map STRtree query results back to building indices).

    Returns: (isovist_polygon, hit_building_idx: set[int], ray_hit_flags: list[bool])
    """
    ox_, oy_ = origin_xy
    isovist_points = []
    hit_building_idx = set()
    ray_hit_flags = []

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    for ang in angles:
        dx, dy = math.cos(ang), math.sin(ang)
        far_x, far_y = ox_ + radius * dx, oy_ + radius * dy
        ray = LineString([(ox_, oy_), (far_x, far_y)])

        closest_x, closest_y = far_x, far_y
        closest_dist2 = radius * radius
        closest_idx = None

        candidate_idx = tree.query(ray)
        for i in candidate_idx:
            boundary = boundaries[i]
            inter = ray.intersection(boundary)
            if inter.is_empty:
                continue

            if inter.geom_type == "Point":
                pts = [inter]
            elif inter.geom_type == "MultiPoint":
                pts = list(inter.geoms)
            elif inter.geom_type == "LineString":
                pts = [Point(c) for c in inter.coords]
            elif inter.geom_type == "MultiLineString":
                pts = [Point(c) for ls in inter.geoms for c in ls.coords]
            else:
                continue

            for p in pts:
                d2 = (p.x - ox_) ** 2 + (p.y - oy_) ** 2
                if d2 < closest_dist2:
                    closest_dist2 = d2
                    closest_x, closest_y = p.x, p.y
                    closest_idx = i

        isovist_points.append((closest_x, closest_y))
        ray_hit_flags.append(closest_idx is not None)
        if closest_idx is not None:
            hit_building_idx.add(closest_idx)

    poly = Polygon(isovist_points)
    if not poly.is_valid:
        poly = poly.buffer(0)

    return poly, hit_building_idx, ray_hit_flags


def isovist_area(polygon):
    return float(polygon.area)


def isovist_compactness(polygon):
    """Circularity ratio: 4*pi*Area / Perimeter^2. 1.0 = perfect circle,
    lower = more elongated/irregular."""
    perimeter = polygon.length
    if perimeter <= 0:
        return 0.0
    return float(4 * math.pi * polygon.area / (perimeter ** 2))


def isovist_occlusivity(ray_hit_flags):
    """Fraction of cast rays that were stopped by a building before
    reaching the full radius, vs. reaching it unobstructed."""
    if not ray_hit_flags:
        return 0.0
    return float(sum(ray_hit_flags) / len(ray_hit_flags))

"""
Feature computation shared across svg_builder.py and tvg_builder.py.

Functions to implement:
- compute_svf(segmentation_map) -> float                      # whole image
- compute_enclosure(segmentation_map, crop_fraction=0.40) -> float  # top-N% crop
- compute_entropy(segmentation_map) -> float
- building_shape_metrics(polygon) -> dict   # area, perimeter, compactness, elongation, orientation, shape_index
- learned_placeholder_encode(value, is_missing) -> tensor   # missing-data convention
"""

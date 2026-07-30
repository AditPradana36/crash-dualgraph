"""
Builds one SVG (Street View Graph) HeteroData object per point.

Functions to implement:
- label_islands(segmentation_map, class_ids) -> list[island_masks]   # connected-component labeling
- build_ego_node(svf, enclosure, entropy) -> features
- build_object_nodes(islands, class_map) -> dict[node_type, features]
- build_sees_edges(ego, objects) -> edge_index, edge_attr
- build_mounted_with_edges(objects) -> edge_index, edge_attr
- build_near_edges(objects, cutoff_d) -> edge_index, edge_attr
- assemble_svg(...) -> HeteroData
"""

"""
Scenario F — merged SVG+TVG graph, built ONCE per point (not per-batch,
per-conv, or anywhere in models.py) so the merge itself stays a single,
inspectable, testable step ahead of training.

Two problems this solves, both flagged as blockers in models.py's old
ScenarioUnifiedGraph placeholder:

1. NAME COLLISION: SVG has a "building" node type (facades visible in
   street-view imagery — signage/light_pole/road_marking/vegetation all
   live at roughly this same visual scale) and TVG separately has its
   OWN "building" node type (OSM footprint polygons, with height/levels/
   type_idx features entirely unrelated to SVG's building nodes). Naively
   unioning both HeteroData objects' node stores would silently overwrite
   one "building" with the other — same key, two different feature
   schemas, two different meanings. Renamed here to "svg_building" and
   "tvg_building" respectively; every edge type mentioning either is
   renamed to match. All other SVG/TVG node types (ego, signage,
   light_pole, road_marking, vegetation, incident, intersection,
   peer_incident) keep their original names — no collision exists there.

2. NO CROSS-GRAPH MESSAGE PASSING: without a new edge, "unified" would
   just be two disjoint subgraphs sharing one HeteroData container for
   batching convenience — the SVG half and TVG half could never actually
   exchange information, which defeats the point of scenario F as a
   distinct hypothesis from C (concat) or D (late fusion). A new
   bidirectional "same_location" edge connects SVG's "ego" node (the
   street-view viewpoint) to TVG's "incident" node (the top-down anchor)
   — both exist exactly once per graph, UNCONDITIONALLY regardless of
   label. Positive (real crash) and negative (sampled non-crash) points
   both get this edge the same way: "incident" is TVG's per-point anchor
   node name regardless of the point's label (see tvg_builder.py — the
   label==1 restriction there applies only to peer_incident sampling,
   never to the anchor itself), and "ego" is SVG's per-point viewpoint
   node, also always present. So same_location links ego<->incident for
   every point in the dataset, positive or negative alike.
"""
import torch
from torch_geometric.data import HeteroData


# Renamed to avoid the SVG-building / TVG-building collision described
# above. Every other node type keeps its original name.
SVG_RENAME = {"building": "svg_building"}
TVG_RENAME = {"building": "tvg_building"}

UNIFIED_SVG_NODE_TYPES = ["ego", "signage", "light_pole", "road_marking", "svg_building", "vegetation"]
UNIFIED_TVG_NODE_TYPES = ["incident", "tvg_building", "intersection", "peer_incident"]
UNIFIED_NODE_TYPES = UNIFIED_SVG_NODE_TYPES + UNIFIED_TVG_NODE_TYPES

# The one new edge type this module adds. Bidirectional, unconditional
# (every point has exactly one ego and one incident node, positive or
# negative), no edge_attr needed (the relation itself — "these two
# anchors describe the same physical point" — carries the signal; adding
# a redundant distance-0 attr would just be a constant column).
SAME_LOCATION_EDGE_TYPES = [
    ("ego", "same_location", "incident"),
    ("incident", "same_location", "ego"),
]


def _renamed_node_types(data, rename_map):
    """All node types actually present on `data`, with rename_map applied."""
    return [rename_map.get(nt, nt) for nt in data.node_types]


def _renamed_edge_key(key, rename_map):
    src, rel, dst = key
    return (rename_map.get(src, src), rel, rename_map.get(dst, dst))


def merge_svg_tvg(svg_data, tvg_data):
    """Build ONE merged HeteroData from a point's (svg_data, tvg_data) pair.

    Copies every node/edge store from both graphs across untouched EXCEPT
    for the "building" name collision (see module docstring) — those get
    renamed to "svg_building" / "tvg_building" on both node stores and
    every edge type that references them. Adds the new same_location
    edges linking ego<->incident.

    Does NOT mutate svg_data / tvg_data in place — builds a fresh
    HeteroData and copies tensors by reference (same tensors, new
    container), matching graph_datasets.apply_normalization's existing
    contract of normalizing svg_data/tvg_data separately BEFORE this is
    called (see run_forward_unified in train.py — normalization happens
    on the original pair, merge happens after)."""
    merged = HeteroData()

    for nt in svg_data.node_types:
        new_nt = SVG_RENAME.get(nt, nt)
        for attr_name, value in svg_data[nt].items():
            merged[new_nt][attr_name] = value

    for nt in tvg_data.node_types:
        new_nt = TVG_RENAME.get(nt, nt)
        for attr_name, value in tvg_data[nt].items():
            merged[new_nt][attr_name] = value

    for key in svg_data.edge_types:
        new_key = _renamed_edge_key(key, SVG_RENAME)
        for attr_name, value in svg_data[key].items():
            merged[new_key][attr_name] = value

    for key in tvg_data.edge_types:
        new_key = _renamed_edge_key(key, TVG_RENAME)
        for attr_name, value in tvg_data[key].items():
            merged[new_key][attr_name] = value

    # same_location: ego<->incident, exactly one of each per graph,
    # always present regardless of label (see module docstring point 2).
    device = svg_data["ego"].x.device
    fwd = torch.tensor([[0], [0]], dtype=torch.long, device=device)  # ego idx 0 -> incident idx 0
    merged["ego", "same_location", "incident"].edge_index = fwd
    merged["incident", "same_location", "ego"].edge_index = fwd.flip(0)

    return merged


def merge_batch(svg_batch, tvg_batch, point_ids):
    """Per-item merge for an already-collated PyG Batch pair, used when a
    per-graph merge before batching isn't convenient. NOT used by the
    default training path (graph_datasets.collate_pairs_unified merges
    BEFORE batching, which is simpler and cheaper — see its docstring for
    why). Kept here only as a documented alternative / for tests that
    want to merge post-hoc without re-deriving per-item HeteroData
    objects from a Batch."""
    raise NotImplementedError(
        "Merging after PyG batching requires re-deriving per-graph "
        "edge_index offsets per node type — not needed by the current "
        "training path (which merges before batching via "
        "graph_datasets.collate_pairs_unified). Left unimplemented "
        "rather than shipped untested."
    )

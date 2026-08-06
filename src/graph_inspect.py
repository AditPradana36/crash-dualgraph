"""
graph_inspect.py — expose every node type, edge type, and their attribute
tensors on a saved SVG/TVG/unified HeteroData object as plain pandas
tables, with real column names (not x_0/x_1/...) and categorical indices
decoded back to their string labels where a vocab is supplied.

Written to answer one question: "what does this .pt file actually
contain, node by node and edge by edge, before I start changing the
encoder." Nothing here trains or touches a model — it's read-only
introspection, meant to run in seconds on a single point.

Usage
-----
    import graph_inspect as gi
    import torch

    svg_data = torch.load("processed/svg_graphs/some_point.pt", weights_only=False)
    tvg_data = torch.load("processed/tvg_graphs/some_point.pt", weights_only=False)

    vocabs = gi.load_vocabs(
        building_vocab_path=cache_dir / "building_type_vocab.json",
        highway_vocab_path=cache_dir / "highway_vocab.json",
    )

    gi.describe_graph(svg_data, schema="svg")               # summary tables
    gi.node_table(svg_data, "signage")                       # per-node detail
    gi.edge_table(tvg_data, ("intersection", "connects", "intersection"),
                  vocabs=vocabs)                              # per-edge detail, decoded

    # or, straight from a dataset object (DualGraphDataset / PooledDualGraphDataset):
    dump = gi.inspect_point(dataset, 0, vocabs=vocabs, export_dir="inspect_out/point0")

Column-name provenance
-----------------------
Feature layouts below are read directly off svg_builder.assemble_svg and
tvg_builder.process_incident's tensor construction — not guessed. If
either builder changes its column order, update SVG_X_NAMES / TVG_X_NAMES
/ SVG_EDGE_NAMES / TVG_EDGE_NAMES below to match; everything else in this
module is generic and needs no further changes.
"""
import json
from pathlib import Path

import pandas as pd
import torch


# ─────────────────────────────────────────────────────────────────────────
# Feature name tables — the one place that encodes "column k means what"
# ─────────────────────────────────────────────────────────────────────────

# node_type -> ordered names for that node type's `.x` tensor columns
SVG_X_NAMES = {
    "ego":          ["svf", "enclosure", "entropy", "pos_x_frac", "pos_y_frac"],
    "signage":      ["cx_frac", "cy_frac"],
    "light_pole":   ["cx_frac", "cy_frac"],
    "road_marking": ["cx_frac", "cy_frac"],
    "building":     ["cx_frac", "cy_frac"],      # pre-merge SVG building (facade instance)
    "svg_building": ["cx_frac", "cy_frac"],      # post-merge (unified_graph rename)
    "vegetation":   ["cx_frac", "cy_frac"],
}

TVG_X_NAMES = {
    "incident":     ["fraction_along", "isovist_area", "isovist_compactness", "isovist_occlusivity"],
    "building":     ["area", "perimeter", "circular_compactness", "elongation", "orientation", "shape_index"],
    "tvg_building": ["area", "perimeter", "circular_compactness", "elongation", "orientation", "shape_index"],
    "intersection": ["betweenness", "orientation_entropy"],
    "peer_incident": ["const_placeholder"],      # torch.ones — no real attribute, see tvg_builder docstring
}

# Non-"x" per-node attributes stored separately (all node types share this
# lookup; only the attrs actually present on a given store get used).
EXTRA_NODE_ATTR_NAMES = {
    "area_norm":       ["area_norm"],
    "class_idx":       ["class_idx"],             # categorical -> decode via vocabs["svg_class"][node_type]
    "type_idx":        ["type_idx"],               # categorical -> decode via vocabs["building_type"]
    "highway_type_idx": ["highway_type_idx"],       # categorical -> decode via vocabs["highway"]
    "height":          ["height_m"],
    "height_missing":  ["height_missing"],
    "levels":          ["levels"],
    "levels_missing":  ["levels_missing"],
}

# rel -> ordered names for that edge type's `edge_attr` columns
SVG_EDGE_NAMES = {
    "sees":         ["area_norm", "dist_norm"],
    "mounted_with": ["bbox_overlap_ratio"],
    "near":         ["dist_norm"],
}

TVG_EDGE_NAMES = {
    "anchors":       ["dist_m"],
    "adjacent":      ["const_clique_flag"],
    "connects":      ["highway_idx", "maxspeed", "maxspeed_missing", "oneway"],
    "fronts":        ["dist_m"],
    "on_segment":    ["dist_m", "highway_idx", "maxspeed", "maxspeed_missing", "oneway"],
    "crash_history": ["dist_m"],                  # ablation-only edge type
    "same_location": [],                           # unified_graph — no edge_attr at all
}

# Edge relations whose named column list includes a highway vocab index —
# used by edge_table to know which column to decode.
_HIGHWAY_EDGE_COLS = {"connects": ["highway_idx"], "on_segment": ["highway_idx"]}


# Kept in sync manually with svg_builder.CLASS_IDX — imported directly
# when available so there's only one real source of truth; this is only
# the fallback for environments where svg_builder can't be imported
# (e.g. missing scipy) but torch/pandas are fine for inspection.
_FALLBACK_CLASS_IDX = {
    "signage": {"Traffic Sign Frame": 0, "Traffic Sign (Back)": 1, "Traffic Sign (Front)": 2,
                "Banner": 3, "Billboard": 4},
    "light_pole": {"Traffic Light": 0, "Street Light": 1, "Pole": 2, "Utility Pole": 3},
    "road_marking": {"Crosswalk - Plain": 0, "Lane Marking - General": 1},
    "building": {"Building": 0},
    "vegetation": {"Vegetation": 0},
}


def _svg_class_idx():
    try:
        from svg_builder import CLASS_IDX
        return CLASS_IDX
    except Exception:
        return _FALLBACK_CLASS_IDX


def default_svg_class_vocab():
    """node_type -> {class_idx: class_name}, inverted from svg_builder.CLASS_IDX
    (or the fallback copy above if svg_builder can't be imported)."""
    return {nt: {i: c for c, i in d.items()} for nt, d in _svg_class_idx().items()}


def invert_vocab(str_to_idx):
    """{"category_str": idx, ...} -> {idx: "category_str", ...} — the JSON
    files written by osm_fetch.py / vocab_merge.py store the forward
    direction; every decode here needs the inverse."""
    return {v: k for k, v in str_to_idx.items()}


def load_vocabs(building_vocab_path=None, highway_vocab_path=None, svg_class_vocab=None):
    """Build the `vocabs` dict describe_graph/node_table/edge_table accept.

    building_vocab_path / highway_vocab_path: paths to the JSON files
    written under <city>/interim/osm_cache/ (pre- or post-04b unification
    — this just reads whatever's there, it doesn't check pooled agreement;
    use dataset_audit.load_unified_vocab_sizes for that check separately).
    svg_class_vocab: override for default_svg_class_vocab(), if needed.
    """
    vocabs = {"svg_class": svg_class_vocab or default_svg_class_vocab()}
    if building_vocab_path is not None:
        with open(building_vocab_path) as f:
            vocabs["building_type"] = invert_vocab(json.load(f))
    if highway_vocab_path is not None:
        with open(highway_vocab_path) as f:
            vocabs["highway"] = invert_vocab(json.load(f))
    return vocabs


# ─────────────────────────────────────────────────────────────────────────
# Schema inference
# ─────────────────────────────────────────────────────────────────────────

def _infer_schema(data):
    node_types = set(data.node_types)
    has_svg = "ego" in node_types
    has_tvg = "incident" in node_types or "peer_incident" in node_types
    if has_svg and has_tvg:
        return "unified"
    if has_svg:
        return "svg"
    if has_tvg:
        return "tvg"
    raise ValueError(
        f"Can't infer schema from node types {sorted(node_types)} — "
        "pass schema='svg'/'tvg'/'unified' explicitly.")


def _x_names_table(schema):
    if schema == "svg":
        return SVG_X_NAMES
    if schema == "tvg":
        return TVG_X_NAMES
    if schema == "unified":
        return {**SVG_X_NAMES, **TVG_X_NAMES}
    raise ValueError(f"Unknown schema: {schema!r}")


def _edge_names_table(schema):
    if schema == "svg":
        return SVG_EDGE_NAMES
    if schema == "tvg":
        return TVG_EDGE_NAMES
    if schema == "unified":
        return {**SVG_EDGE_NAMES, **TVG_EDGE_NAMES}
    raise ValueError(f"Unknown schema: {schema!r}")


def _names_for(names_list, k, fallback_prefix):
    if names_list and len(names_list) == k:
        return list(names_list)
    return [f"{fallback_prefix}_{i}" for i in range(k)]


# ─────────────────────────────────────────────────────────────────────────
# Summaries — one row per node type / edge type
# ─────────────────────────────────────────────────────────────────────────

def summarize_nodes(data, schema="auto"):
    """One row per node type present on `data`: count, x width, and every
    extra attribute name stored on that node store (class_idx, area_norm,
    height, ...). Empty node types (count 0) are included, not skipped —
    an empty type is itself informative (e.g. no signage detected)."""
    schema = _infer_schema(data) if schema == "auto" else schema
    x_names = _x_names_table(schema)

    rows = []
    for nt in data.node_types:
        store = data[nt]
        n = store.num_nodes if store.num_nodes is not None else 0
        x_dim = store.x.shape[1] if "x" in store and store.x.dim() == 2 else 0
        extra_attrs = sorted(k for k in store.keys() if k != "x" and torch.is_tensor(store[k]))
        rows.append({
            "node_type": nt, "n_nodes": n, "x_dim": x_dim,
            "x_columns": ", ".join(x_names.get(nt, [f"x_{i}" for i in range(x_dim)])),
            "extra_attrs": ", ".join(extra_attrs) if extra_attrs else "(none)",
        })
    return pd.DataFrame(rows).sort_values("node_type").reset_index(drop=True)


def summarize_edges(data, schema="auto"):
    """One row per edge type present on `data`: edge count, edge_attr
    width, and whether edge_attr exists at all (same_location, added by
    unified_graph.merge_svg_tvg, deliberately has none)."""
    schema = _infer_schema(data) if schema == "auto" else schema
    edge_names = _edge_names_table(schema)

    rows = []
    for key in data.edge_types:
        src, rel, dst = key
        store = data[key]
        n_edges = store.edge_index.shape[1] if "edge_index" in store else 0
        has_attr = "edge_attr" in store and torch.is_tensor(store.edge_attr) and store.edge_attr.numel() > 0
        attr_dim = store.edge_attr.shape[1] if has_attr and store.edge_attr.dim() == 2 else 0
        rows.append({
            "src": src, "rel": rel, "dst": dst, "n_edges": n_edges,
            "edge_attr_dim": attr_dim,
            "edge_attr_columns": ", ".join(edge_names.get(rel, [f"attr_{i}" for i in range(attr_dim)]))
                                 if attr_dim else "(none)",
        })
    return pd.DataFrame(rows).sort_values(["src", "rel", "dst"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# Full per-node / per-edge tables, with categorical decoding
# ─────────────────────────────────────────────────────────────────────────

def node_table(data, node_type, schema="auto", vocabs=None):
    """Every node of `node_type` as one row, every stored attribute as
    named columns. Adds a `<attr>_label` column next to any categorical
    index column (class_idx, type_idx, highway_type_idx) a vocab was
    supplied for."""
    schema = _infer_schema(data) if schema == "auto" else schema
    if node_type not in data.node_types:
        raise KeyError(f"node_type {node_type!r} not on this graph. "
                        f"Available: {sorted(data.node_types)}")
    store = data[node_type]
    n = store.num_nodes if store.num_nodes is not None else 0
    vocabs = vocabs or {}

    cols = {}
    x_names_table = _x_names_table(schema)

    if "x" in store and torch.is_tensor(store.x) and store.x.dim() == 2 and store.x.shape[0] == n:
        names = _names_for(x_names_table.get(node_type), store.x.shape[1], "x")
        for k, name in enumerate(names):
            cols[name] = store.x[:, k].detach().cpu().numpy()

    for attr in sorted(store.keys()):
        if attr in ("x",) or not torch.is_tensor(store[attr]):
            continue
        t = store[attr]
        if t.dim() == 0 or t.shape[0] != n:
            continue
        names = EXTRA_NODE_ATTR_NAMES.get(attr, [attr])
        if t.dim() == 1:
            names = _names_for(names, 1, attr)
            cols[names[0]] = t.detach().cpu().numpy()
        else:
            names = _names_for(names, t.shape[1], attr)
            for k, name in enumerate(names):
                cols[name] = t[:, k].detach().cpu().numpy()

    df = pd.DataFrame(cols)
    df.insert(0, "node_idx", range(n))

    # Decode categorical columns in place, right next to the raw index.
    if "class_idx" in cols and "svg_class" in vocabs and node_type in vocabs["svg_class"]:
        lut = vocabs["svg_class"][node_type]
        df.insert(df.columns.get_loc("class_idx") + 1, "class_label",
                   df["class_idx"].map(lambda i: lut.get(int(i), f"<unknown:{int(i)}>")))
    if "type_idx" in cols and "building_type" in vocabs:
        lut = vocabs["building_type"]
        df.insert(df.columns.get_loc("type_idx") + 1, "type_label",
                   df["type_idx"].map(lambda i: lut.get(int(i), f"<unknown:{int(i)}>")))
    if "highway_type_idx" in cols and "highway" in vocabs:
        lut = vocabs["highway"]
        df.insert(df.columns.get_loc("highway_type_idx") + 1, "highway_label",
                   df["highway_type_idx"].map(lambda i: lut.get(int(i), f"<unknown:{int(i)}>")))

    return df


def edge_table(data, edge_type, schema="auto", vocabs=None):
    """Every edge of `edge_type` (a (src, rel, dst) tuple) as one row:
    src_idx, dst_idx, then every edge_attr column by name. Decodes a
    highway_idx column into `highway_label` if `vocabs["highway"]` is
    given and this relation carries one (connects / on_segment)."""
    schema = _infer_schema(data) if schema == "auto" else schema
    if edge_type not in data.edge_types:
        raise KeyError(f"edge_type {edge_type!r} not on this graph. "
                        f"Available: {sorted(data.edge_types)}")
    src, rel, dst = edge_type
    store = data[edge_type]
    vocabs = vocabs or {}

    ei = store.edge_index
    n_edges = ei.shape[1]
    df = pd.DataFrame({"src_idx": ei[0].detach().cpu().numpy(),
                        "dst_idx": ei[1].detach().cpu().numpy()})

    ea = store.get("edge_attr", None)
    if torch.is_tensor(ea) and ea.numel() > 0:
        edge_names_table = _edge_names_table(schema)
        names = _names_for(edge_names_table.get(rel), ea.shape[1], "attr")
        for k, name in enumerate(names):
            df[name] = ea[:, k].detach().cpu().numpy()

        highway_col = _HIGHWAY_EDGE_COLS.get(rel)
        if highway_col and "highway" in vocabs and highway_col[0] in df.columns:
            lut = vocabs["highway"]
            col = highway_col[0]
            df.insert(df.columns.get_loc(col) + 1, "highway_label",
                       df[col].map(lambda i: lut.get(int(round(i)), f"<unknown:{int(round(i))}>")))

    df.insert(0, "edge_idx", range(n_edges))
    return df


# ─────────────────────────────────────────────────────────────────────────
# Combined summary + full dump + convenience wrappers
# ─────────────────────────────────────────────────────────────────────────

def describe_graph(data, schema="auto", vocabs=None, verbose=True):
    """Prints (if verbose) and returns {"nodes": df, "edges": df} —
    the two summary tables side by side, the fast first look."""
    schema = _infer_schema(data) if schema == "auto" else schema
    nodes_df = summarize_nodes(data, schema=schema)
    edges_df = summarize_edges(data, schema=schema)
    if verbose:
        print(f"schema={schema}")
        print("\n-- nodes --")
        print(nodes_df.to_string(index=False))
        print("\n-- edges --")
        print(edges_df.to_string(index=False))
    return {"nodes": nodes_df, "edges": edges_df}


def full_dump(data, schema="auto", vocabs=None):
    """Every node type and edge type as its own full DataFrame (not just
    the summary counts) — the complete, human-readable contents of one
    graph. Returns {"nodes": {node_type: df, ...}, "edges": {(s,r,d): df, ...}}."""
    schema = _infer_schema(data) if schema == "auto" else schema
    node_tables = {nt: node_table(data, nt, schema=schema, vocabs=vocabs) for nt in data.node_types}
    edge_tables = {ek: edge_table(data, ek, schema=schema, vocabs=vocabs) for ek in data.edge_types}
    return {"nodes": node_tables, "edges": edge_tables}


def _export_dump(dump, export_dir, tag):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    for nt, df in dump["nodes"].items():
        df.to_csv(export_dir / f"{tag}_node_{nt}.csv", index=False)
    for (s, r, d), df in dump["edges"].items():
        df.to_csv(export_dir / f"{tag}_edge_{s}__{r}__{d}.csv", index=False)


def inspect_point(dataset, i, vocabs=None, verbose=True, export_dir=None):
    """Convenience wrapper for a DualGraphDataset / PooledDualGraphDataset
    item: `i` may be a positional index or, if the dataset's index_df has
    a matching 'point_id'/'uid' column, that identifier directly.

    Prints (if verbose) the SVG and TVG summary tables, and if
    `export_dir` is given, writes every node/edge type's full table out
    as CSVs under that directory (svg_*.csv / tvg_*.csv) for opening in
    a spreadsheet.

    Returns {"uid": ..., "label": ..., "svg": full_dump(...), "tvg": full_dump(...)}.
    """
    index_df = dataset.index_df
    if isinstance(i, (int,)) and i < len(dataset):
        idx = i
    else:
        id_col = "uid" if "uid" in index_df.columns else "point_id"
        matches = index_df.index[index_df[id_col] == i]
        if len(matches) == 0:
            raise KeyError(f"{i!r} not found in dataset.index_df['{id_col}']")
        idx = matches[0]

    svg_data, tvg_data, label, uid = dataset[idx]

    if verbose:
        print(f"===== point uid={uid}  label={float(label):.0f} =====")
        print("\n----- SVG -----")
    svg_summary = describe_graph(svg_data, schema="svg", vocabs=vocabs, verbose=verbose)
    if verbose:
        print("\n----- TVG -----")
    tvg_summary = describe_graph(tvg_data, schema="tvg", vocabs=vocabs, verbose=verbose)

    result = {"uid": uid, "label": float(label),
              "svg": full_dump(svg_data, schema="svg", vocabs=vocabs),
              "tvg": full_dump(tvg_data, schema="tvg", vocabs=vocabs),
              "svg_summary": svg_summary, "tvg_summary": tvg_summary}

    if export_dir is not None:
        _export_dump(result["svg"], export_dir, "svg")
        _export_dump(result["tvg"], export_dir, "tvg")
        if verbose:
            print(f"\nExported per-type CSVs to {export_dir}")

    return result


def browse_points(dataset, city=None, label=None, uid_contains=None,
                  n=10, random_state=None, sort_by=None):
    """Filter/sample dataset.index_df down to candidate points, without
    writing the same pandas filter by hand each time.

    city: str or list of str, matches the 'city' column if present.
    label: 0 or 1, matches the 'label' column.
    uid_contains: substring match against 'uid' (falls back to 'point_id'
        if no 'uid' column exists — e.g. a single-city DualGraphDataset).
    n: max rows returned. None returns everything matching the filters.
    random_state: if given, `n` rows are a reproducible random sample;
        if None, the first `n` rows (optionally after sort_by) are used.
    sort_by: column name to sort by before taking the first `n` (ignored
        if random_state is given). Useful with a joined-in prediction
        column, e.g. sort_by="pred_prob" to look at the most confident
        points first.

    Returns a DataFrame with the original index preserved as a column
    named 'row_idx' — pass any of THOSE values as `i` to inspect_point /
    plot_point, or pass 'uid' directly (both are accepted).
    """
    df = dataset.index_df

    if city is not None and "city" in df.columns:
        cities = [city] if isinstance(city, str) else list(city)
        df = df[df["city"].isin(cities)]
    if label is not None and "label" in df.columns:
        df = df[df["label"] == label]
    if uid_contains is not None:
        id_col = "uid" if "uid" in df.columns else "point_id"
        df = df[df[id_col].astype(str).str.contains(uid_contains, na=False)]

    if sort_by is not None and random_state is None:
        df = df.sort_values(sort_by)

    if random_state is not None:
        df = df.sample(min(n, len(df)), random_state=random_state) if n else df.sample(frac=1, random_state=random_state)
    elif n is not None:
        df = df.head(n)

    out = df.copy()
    out.insert(0, "row_idx", out.index)
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# Visualization — abstract graph TOPOLOGY only (which nodes exist, which
# edges connect them, what relation each edge is). Deliberately NOT
# spatially/visually grounded — svg_visualize.py already draws nodes at
# their real pixel position over the source image, and tvg_visualize.py
# already draws them at real UTM coordinates over the isovist/buildings.
# Both of those need assets this module doesn't load (the source image,
# the buildings GeoDataFrame). This needs only the saved .pt tensors, so
# it works everywhere graph_inspect's tables already work — including
# straight off a checkpointed graph with no image/OSM cache at hand.
#
# matplotlib/networkx are imported lazily, inside these functions only,
# so `import graph_inspect` for the table functions above never requires
# them.
# ─────────────────────────────────────────────────────────────────────────

# Fixed canonical ordering per schema, so repeated calls lay out the same
# node type in the same angular slot every time — makes two points'
# diagrams visually comparable rather than shuffled per-call.
SVG_NODE_ORDER = ["signage", "light_pole", "road_marking", "building", "svg_building", "vegetation"]
TVG_NODE_ORDER = ["tvg_building", "building", "intersection", "peer_incident"]


def _anchor_types(schema):
    if schema == "svg":
        return ["ego"]
    if schema == "tvg":
        return ["incident"]
    if schema == "unified":
        return ["ego", "incident"]
    raise ValueError(f"Unknown schema: {schema!r}")


def _node_order(schema):
    if schema == "svg":
        return SVG_NODE_ORDER
    if schema == "tvg":
        return TVG_NODE_ORDER
    if schema == "unified":
        return SVG_NODE_ORDER + TVG_NODE_ORDER
    raise ValueError(f"Unknown schema: {schema!r}")


def _radial_layout(data, schema):
    """Deterministic hub-and-spoke layout: anchor node(s) at the center
    (ego/incident — the one node type guaranteed exactly-one-per-graph),
    every other node type given a fixed angular slot around it, multiple
    nodes of the same type fanned out along a short arc within their
    slot so they don't stack on top of each other.

    Returns {global_id: (x, y)} where global_id is f"{node_type}_{i}".
    """
    import math

    anchors = [a for a in _anchor_types(schema) if a in data.node_types]
    order = [nt for nt in _node_order(schema) if nt in data.node_types]
    # anything present but not in the canonical order list (shouldn't
    # normally happen) still gets placed, appended at the end.
    order += [nt for nt in data.node_types if nt not in order and nt not in anchors]

    pos = {}
    for ai, a in enumerate(anchors):
        # two anchors (unified graph): place side by side, close together,
        # since same_location connects exactly this pair.
        x = 0.0 if len(anchors) == 1 else (ai - 0.5) * 0.5
        pos[f"{a}_0"] = (x, 0.0)

    n_types = max(len(order), 1)
    R = 1.8
    for ti, nt in enumerate(order):
        theta = 2 * math.pi * ti / n_types
        n = data[nt].num_nodes or 0
        if n == 0:
            continue
        spread = min(0.30, 0.9 / n)
        for i in range(n):
            offset = (i - (n - 1) / 2) * spread
            a = theta + offset
            pos[f"{nt}_{i}"] = (R * math.cos(a), R * math.sin(a))

    return pos


def _short_label(nt, i, data, schema, vocabs):
    """One short line per node for the diagram: node type + a decoded
    category if one applies, else just the index."""
    store = data[nt]
    vocabs = vocabs or {}
    if "class_idx" in store and "svg_class" in vocabs and nt in vocabs["svg_class"]:
        idx = int(store.class_idx[i])
        return vocabs["svg_class"][nt].get(idx, f"{nt}?{idx}")
    if "type_idx" in store and "building_type" in vocabs:
        idx = int(store.type_idx[i])
        return vocabs["building_type"].get(idx, f"{nt}?{idx}")
    if nt in ("ego", "incident"):
        return nt
    return f"{nt}[{i}]"


def to_networkx(data, schema="auto", vocabs=None):
    """Convert to a networkx.MultiGraph — one node per graph instance
    (global id f"{node_type}_{i}"), one edge per (instance pair, relation),
    with type/rel metadata as node/edge attributes. Reciprocal edge types
    (e.g. ego->signage and signage->ego, which svg_builder always writes
    as a mirrored pair) collapse to a single undirected edge; a relation
    that legitimately differs between the same pair (e.g. signage and
    light_pole connected by BOTH 'near' and 'mounted_with') keeps both as
    separate parallel edges.

    Needs networkx installed; raises with a clear message if it isn't.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("to_networkx requires networkx (`pip install networkx`).") from e

    schema = _infer_schema(data) if schema == "auto" else schema
    G = nx.MultiGraph()

    for nt in data.node_types:
        n = data[nt].num_nodes or 0
        for i in range(n):
            gid = f"{nt}_{i}"
            G.add_node(gid, node_type=nt, local_idx=i,
                       label=_short_label(nt, i, data, schema, vocabs))

    seen = set()
    for key in data.edge_types:
        s, rel, d = key
        ei = data[key].edge_index
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            u, v = f"{s}_{i}", f"{d}_{j}"
            dedup_key = (frozenset((u, v)), rel)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            G.add_edge(u, v, rel=rel)

    return G


def plot_graph(data, schema="auto", vocabs=None, title=None, figsize=(9, 8),
              node_size=550, font_size=7, ax=None, save_path=None):
    """Draw the graph's topology: node-type-colored dots in a fixed
    radial layout (anchor node at the center, every other node type in
    its own angular slot), edges colored by relation. Returns the
    matplotlib Figure.

    This is a structural diagram, not a spatial one — node position here
    encodes "which type, which slot," not real pixel/UTM coordinates.
    Use svg_visualize.py / tvg_visualize.py instead when you want nodes
    drawn at their true position over the source image or map.

    Needs matplotlib installed; raises with a clear message if it isn't.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
    except ImportError as e:
        raise ImportError("plot_graph requires matplotlib (`pip install matplotlib`).") from e

    schema = _infer_schema(data) if schema == "auto" else schema
    pos = _radial_layout(data, schema)
    anchors = set(_anchor_types(schema))

    # Build the deduplicated edge list once (see to_networkx's docstring
    # for why dedup is needed — svg/tvg builders write reciprocal edge
    # types for every symmetric relation).
    edges = []
    seen = set()
    for key in data.edge_types:
        s, rel, d = key
        ei = data[key].edge_index
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            u, v = f"{s}_{i}", f"{d}_{j}"
            dedup_key = (frozenset((u, v)), rel)
            if dedup_key in seen or u not in pos or v not in pos:
                continue
            seen.add(dedup_key)
            edges.append((u, v, rel))

    node_types_present = sorted({nt for nt in data.node_types if (data[nt].num_nodes or 0) > 0})
    rels_present = sorted({rel for _, _, rel in edges})

    node_cmap = plt.get_cmap("tab20")
    node_colors = {nt: node_cmap(i / max(len(node_types_present) - 1, 1))
                   for i, nt in enumerate(node_types_present)}
    edge_cmap = plt.get_cmap("tab10")
    edge_colors = {rel: edge_cmap(i / max(len(rels_present) - 1, 1))
                   for i, rel in enumerate(rels_present)}

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for u, v, rel in edges:
        (x0, y0), (x1, y1) = pos[u], pos[v]
        ax.plot([x0, x1], [y0, y1], color=edge_colors[rel], alpha=0.55,
                linewidth=1.3, zorder=1)

    for nt in data.node_types:
        n = data[nt].num_nodes or 0
        if n == 0:
            continue
        xs = [pos[f"{nt}_{i}"][0] for i in range(n) if f"{nt}_{i}" in pos]
        ys = [pos[f"{nt}_{i}"][1] for i in range(n) if f"{nt}_{i}" in pos]
        is_anchor = nt in anchors
        ax.scatter(xs, ys, s=node_size * (1.6 if is_anchor else 1.0),
                   color=node_colors[nt], edgecolors="black",
                   linewidths=1.4 if is_anchor else 0.6,
                   marker="*" if is_anchor else "o", zorder=3, label=None)
        for i in range(n):
            gid = f"{nt}_{i}"
            if gid not in pos:
                continue
            x, y = pos[gid]
            ax.annotate(_short_label(nt, i, data, schema, vocabs), (x, y),
                       textcoords="offset points", xytext=(0, 9),
                       ha="center", fontsize=font_size, zorder=4)

    node_legend = [mpatches.Patch(color=node_colors[nt],
                                   label=f"{nt} (n={data[nt].num_nodes or 0})")
                   for nt in node_types_present]
    edge_legend = [mlines.Line2D([], [], color=edge_colors[rel], label=rel)
                   for rel in rels_present]
    leg1 = ax.legend(handles=node_legend, title="node type", loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=edge_legend, title="edge relation", loc="lower left",
             bbox_to_anchor=(1.02, 0.0), fontsize=8, title_fontsize=9)

    ax.set_title(title or f"schema={schema}")
    ax.set_aspect("equal")
    ax.axis("off")
    if own_fig:
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_point(dataset, i, vocabs=None, save_dir=None):
    """SVG and TVG topology diagrams for one dataset item, side by side
    in one figure. Same `i` semantics as inspect_point (positional index
    or a uid/point_id present in dataset.index_df). Returns the Figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("plot_point requires matplotlib (`pip install matplotlib`).") from e

    index_df = dataset.index_df
    if isinstance(i, int) and i < len(dataset):
        idx = i
    else:
        id_col = "uid" if "uid" in index_df.columns else "point_id"
        matches = index_df.index[index_df[id_col] == i]
        if len(matches) == 0:
            raise KeyError(f"{i!r} not found in dataset.index_df['{id_col}']")
        idx = matches[0]

    svg_data, tvg_data, label, uid = dataset[idx]

    fig, (ax_svg, ax_tvg) = plt.subplots(1, 2, figsize=(18, 8))
    plot_graph(svg_data, schema="svg", vocabs=vocabs,
              title=f"SVG — uid={uid} label={float(label):.0f}", ax=ax_svg)
    plot_graph(tvg_data, schema="tvg", vocabs=vocabs,
              title=f"TVG — uid={uid} label={float(label):.0f}", ax=ax_tvg)
    fig.tight_layout()

    if save_dir is not None:
        path = Path(save_dir) / f"{uid}_graph.png"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    return fig

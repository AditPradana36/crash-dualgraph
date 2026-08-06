"""
Scenario G: flattens an SVG+TVG graph pair into tabular summary features
for XGBoost — deliberately independent of models.py's encoders, since its
whole purpose is testing whether graph structure earns its keep over
flat features.
"""
import torch


def flatten_pair(svg_data, tvg_data):
    row = {}

    row["svf"] = svg_data["ego"].x[0, 0].item()
    row["enclosure"] = svg_data["ego"].x[0, 1].item()
    row["entropy"] = svg_data["ego"].x[0, 2].item()

    for nt in ["signage", "light_pole", "road_marking", "building", "vegetation"]:
        row[f"n_{nt}"] = svg_data[nt].x.shape[0]

    row["isovist_area"] = tvg_data["incident"].x[0, 1].item()
    row["isovist_compactness"] = tvg_data["incident"].x[0, 2].item()
    row["isovist_occlusivity"] = tvg_data["incident"].x[0, 3].item()
    row["highway_type_idx"] = tvg_data["incident"].highway_type_idx[0].item()

    n_buildings = tvg_data["building"].x.shape[0]
    row["n_buildings_tvg"] = n_buildings
    row["n_intersections"] = tvg_data["intersection"].x.shape[0]
    row["building_density"] = n_buildings / max(tvg_data["incident"].x[0, 1].item(), 1e-6)  # per isovist area

    return row


def build_feature_table(point_ids, svg_dir, tvg_dir, torch_module):
    from pathlib import Path
    import pandas as pd
    svg_dir, tvg_dir = Path(svg_dir), Path(tvg_dir)
    rows = []
    for pid in point_ids:
        svg_data = torch_module.load(svg_dir / f"{pid}.pt", weights_only=False)
        tvg_data = torch_module.load(tvg_dir / f"{pid}.pt", weights_only=False)
        row = flatten_pair(svg_data, tvg_data)
        row["point_id"] = pid
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_table_pooled(index_df, torch_module):
    """Like build_feature_table, but reads svg_dir/tvg_dir PER ROW from
    index_df (columns added by 05b_dataset_assembly_pooled.ipynb) instead
    of one shared dir -- same reason as graph_datasets.PooledDualGraphDataset:
    Bogor and Warsaw's graphs live in different folders. Keys the
    returned table by 'uid' (e.g. bog_positive_12, war_negative_34 --
    city AND label encoded, see 05b), not bare point_id, matching
    PooledDualGraphDataset's convention -- callers merge back onto
    index_df via 'uid', not 'point_id'."""
    from pathlib import Path
    import pandas as pd
    rows = []
    for _, r in index_df.iterrows():
        svg_data = torch_module.load(Path(r["svg_dir"]) / f"{r['point_id']}.pt", weights_only=False)
        tvg_data = torch_module.load(Path(r["tvg_dir"]) / f"{r['point_id']}.pt", weights_only=False)
        row = flatten_pair(svg_data, tvg_data)
        row["uid"] = r["uid"]
        rows.append(row)
    return pd.DataFrame(rows)

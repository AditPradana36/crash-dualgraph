"""
Post-construction dataset assembly checks: cross-checks 01's reconciled
points against what 02/03/04 actually produced on disk, builds the final
training-ready index with a full drop report, and audits node/edge-type
coverage across the whole dataset.

Single-pass by design: scan_dataset() loads every SVG/TVG pair exactly
once, catching both missing files AND corrupted/truncated ones (e.g. from
a Colab session killed mid-torch.save), while simultaneously accumulating
type-presence stats — no second pass needed just for the audit.
"""
from pathlib import Path

import pandas as pd


def scan_dataset(point_ids, svg_dir, tvg_dir, torch_module):
    """Returns (status_df, node_stats, edge_stats).

    status_df columns: point_id, svg_ok, tvg_ok, svg_error, tvg_error
    node_stats / edge_stats: dict keyed by node type / (src, rel, dst),
        each -> {"total_count": int, "n_graphs_present": int}
    """
    svg_dir, tvg_dir = Path(svg_dir), Path(tvg_dir)
    records = []
    node_stats, edge_stats = {}, {}

    for pid in point_ids:
        svg_path, tvg_path = svg_dir / f"{pid}.pt", tvg_dir / f"{pid}.pt"
        svg_ok, svg_err, svg_data = False, None, None
        tvg_ok, tvg_err, tvg_data = False, None, None

        if svg_path.exists():
            try:
                svg_data = torch_module.load(svg_path, weights_only=False)
                svg_ok = True
            except Exception as e:
                svg_err = str(e)
        else:
            svg_err = "file not found"

        if tvg_path.exists():
            try:
                tvg_data = torch_module.load(tvg_path, weights_only=False)
                tvg_ok = True
            except Exception as e:
                tvg_err = str(e)
        else:
            tvg_err = "file not found"

        records.append({
            "point_id": pid, "svg_ok": svg_ok, "tvg_ok": tvg_ok,
            "svg_error": svg_err, "tvg_error": tvg_err,
        })

        for data in (svg_data, tvg_data):
            if data is None:
                continue
            for nt in data.node_types:
                n = data[nt].x.shape[0] if hasattr(data[nt], "x") else 0
                s = node_stats.setdefault(nt, {"total_count": 0, "n_graphs_present": 0})
                s["total_count"] += n
                if n > 0:
                    s["n_graphs_present"] += 1
            for ek in data.edge_index_dict:
                n = data.edge_index_dict[ek].shape[1]
                s = edge_stats.setdefault(ek, {"total_count": 0, "n_graphs_present": 0})
                s["total_count"] += n
                if n > 0:
                    s["n_graphs_present"] += 1

    return pd.DataFrame(records), node_stats, edge_stats


def build_drop_report(status_df):
    total = len(status_df)
    complete = status_df[status_df["svg_ok"] & status_df["tvg_ok"]]
    svg_only_missing = status_df[~status_df["svg_ok"] & status_df["tvg_ok"]]
    tvg_only_missing = status_df[status_df["svg_ok"] & ~status_df["tvg_ok"]]
    both_missing = status_df[~status_df["svg_ok"] & ~status_df["tvg_ok"]]

    return {
        "total_points": total,
        "complete_pairs": len(complete),
        "missing_svg_only": len(svg_only_missing),
        "missing_tvg_only": len(tvg_only_missing),
        "missing_both": len(both_missing),
        "missing_svg_only_ids": svg_only_missing["point_id"].tolist(),
        "missing_tvg_only_ids": tvg_only_missing["point_id"].tolist(),
        "missing_both_ids": both_missing["point_id"].tolist(),
    }


def filter_complete_points(reconciled_df, status_df):
    complete_ids = set(status_df[status_df["svg_ok"] & status_df["tvg_ok"]]["point_id"])
    return reconciled_df[reconciled_df["point_id"].isin(complete_ids)].reset_index(drop=True)


def fold_balance_report(df, fold_cols):
    return {
        col: df.groupby(col)["label"].agg(["count", "sum"]).rename(columns={"sum": "n_positive"})
        for col in fold_cols
    }

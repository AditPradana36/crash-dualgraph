"""
Loads paired SVG/TVG graphs per point, PyG-compatible batching, and
per-fold normalization — fit on the training partition only, applied
without refitting to val/test. Never fit globally (that would leak fold
information back in).

SCOPE NOTE: normalization covers node .x tensors only (SVG's are already
fraction/ratio-bounded by construction; TVG's building shape metrics and
intersection metrics are the ones that actually need it, being in real
units). Edge-attribute continuous columns (distances in meters) are NOT
normalized here — a known simplification, worth revisiting if training
shows instability on TVG-containing scenarios specifically.
"""
from pathlib import Path

import torch
from torch_geometric.data import Batch


class DualGraphDataset:
    def __init__(self, index_df, svg_dir, tvg_dir):
        self.index_df = index_df.reset_index(drop=True)
        self.svg_dir = Path(svg_dir)
        self.tvg_dir = Path(tvg_dir)

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, i):
        row = self.index_df.iloc[i]
        pid = row["point_id"]
        svg_data = torch.load(self.svg_dir / f"{pid}.pt", weights_only=False)
        tvg_data = torch.load(self.tvg_dir / f"{pid}.pt", weights_only=False)
        label = torch.tensor(float(row["label"]))
        return svg_data, tvg_data, label, pid


def collate_pairs(batch):
    svg_list, tvg_list, labels, pids = zip(*batch)
    svg_batch = Batch.from_data_list(list(svg_list))
    tvg_batch = Batch.from_data_list(list(tvg_list))
    return svg_batch, tvg_batch, torch.stack(labels), list(pids)


# Explicit, not auto-detected — avoids ever accidentally normalizing a
# categorical index or a missing-value flag.
NORMALIZE_TARGETS = {
    "svg": {
        "ego": (0, 5), "signage": (0, 2), "light_pole": (0, 2),
        "road_marking": (0, 2), "building": (0, 2), "vegetation": (0, 2),
    },
    "tvg": {
        "incident": (0, 4), "building": (0, 6), "intersection": (0, 2),
    },
}


def _fit_stats(values_list):
    if not values_list:
        return None, None
    stacked = torch.cat(values_list, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0).clamp(min=1e-6)  # guards constant columns, e.g. ego's fixed position
    return mean, std


def fit_normalization(train_svg_list, train_tvg_list):
    """Fit on ONE fold's training partition only."""
    stats = {"svg": {}, "tvg": {}}
    for nt, (s, e) in NORMALIZE_TARGETS["svg"].items():
        vals = [d[nt].x[:, s:e] for d in train_svg_list if d[nt].x.shape[0] > 0]
        stats["svg"][nt] = _fit_stats(vals)
    for nt, (s, e) in NORMALIZE_TARGETS["tvg"].items():
        vals = [d[nt].x[:, s:e] for d in train_tvg_list if d[nt].x.shape[0] > 0]
        stats["tvg"][nt] = _fit_stats(vals)
    return stats


def apply_normalization(svg_data, tvg_data, stats):
    for nt, (s, e) in NORMALIZE_TARGETS["svg"].items():
        mean, std = stats["svg"][nt]
        if mean is not None and svg_data[nt].x.shape[0] > 0:
            svg_data[nt].x[:, s:e] = (svg_data[nt].x[:, s:e] - mean) / std
    for nt, (s, e) in NORMALIZE_TARGETS["tvg"].items():
        mean, std = stats["tvg"][nt]
        if mean is not None and tvg_data[nt].x.shape[0] > 0:
            tvg_data[nt].x[:, s:e] = (tvg_data[nt].x[:, s:e] - mean) / std
    return svg_data, tvg_data

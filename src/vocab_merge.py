"""
Unifies per-city highway/building-type vocabularies AFTER both cities'
04_tvg_construction has already run independently, and patches the
categorical indices already baked into saved .pt TVG graphs to match
the unified vocab — WITHOUT recomputing isovist geometry.

Why this is safe: highway_vocab / building_type_vocab are just
{category_str: int} lookup tables. The graphs only ever store the
resulting integer. So "fixing" a graph is: look up which category
string an old integer meant (via that city's OWN old vocab), then
look up the new integer for that same string in the unified vocab,
and overwrite the tensor value. No geometry, no OSM refetch.

Convention preserved from osm_fetch.py: category strings are sorted,
then the fallback token ("unclassified" for highway, "unknown" for
building type) is appended last if not already present. The unified
vocab follows the exact same convention so it stays a drop-in
replacement for any future single-city rerun too.

Usage (per city, run once):

    from vocab_merge import *

    hw_a = load_vocab(city_a_cache_dir / "highway_vocab.json")
    hw_b = load_vocab(city_b_cache_dir / "highway_vocab.json")
    bt_a = load_vocab(city_a_cache_dir / "building_type_vocab.json")
    bt_b = load_vocab(city_b_cache_dir / "building_type_vocab.json")

    unified_hw = build_unified_vocab(hw_a, hw_b, fallback="unclassified")
    unified_bt = build_unified_vocab(bt_a, bt_b, fallback="unknown")

    # Remap EACH city's saved graphs (yes, including the one whose vocab
    # "won" most of its slots — union can still shift order at the
    # boundary, never assume identity mapping without checking).
    remap_city_graphs(
        tvg_dir=city_a_tvg_dir, out_dir=city_a_tvg_dir_remapped,
        old_highway_vocab=hw_a, new_highway_vocab=unified_hw,
        old_building_vocab=bt_a, new_building_vocab=unified_bt,
    )
    remap_city_graphs(
        tvg_dir=city_b_tvg_dir, out_dir=city_b_tvg_dir_remapped,
        old_highway_vocab=hw_b, new_highway_vocab=unified_hw,
        old_building_vocab=bt_b, new_building_vocab=unified_bt,
    )

    # Then update each city's cached vocab JSON + buildings parquet
    # type_idx column too, so any FUTURE single-city rerun (e.g. a
    # third city added later) starts from the unified vocab already.
    save_vocab(unified_hw, city_a_cache_dir / "highway_vocab.json")
    save_vocab(unified_hw, city_b_cache_dir / "highway_vocab.json")
    save_vocab(unified_bt, city_a_cache_dir / "building_type_vocab.json")
    save_vocab(unified_bt, city_b_cache_dir / "building_type_vocab.json")
    remap_buildings_parquet_type_idx(city_a_cache_dir / "buildings.parquet", bt_a, unified_bt)
    remap_buildings_parquet_type_idx(city_b_cache_dir / "buildings.parquet", bt_b, unified_bt)
"""
import json
import shutil
from pathlib import Path

import torch
import geopandas as gpd


def load_vocab(path):
    with open(path) as f:
        return json.load(f)


def save_vocab(vocab, path):
    with open(path, "w") as f:
        json.dump(vocab, f)


def build_unified_vocab(*vocabs, fallback):
    """Union of category strings across all given vocabs, sorted, with
    `fallback` (e.g. "unclassified"/"unknown") pinned last — same
    convention as osm_fetch.build_highway_vocab / build_building_type_vocab.
    """
    all_cats = set()
    for v in vocabs:
        all_cats.update(v.keys())
    unique_vals = sorted(c for c in all_cats if c != fallback)
    unique_vals.append(fallback)
    return {c: i for i, c in enumerate(unique_vals)}


def _remap_map(old_vocab, new_vocab, fallback):
    """old_idx -> new_idx lookup array-as-dict. Any old category not
    found in new_vocab (shouldn't happen if new_vocab is a true union,
    but checked defensively) falls back to new_vocab[fallback]."""
    idx_to_cat = {i: c for c, i in old_vocab.items()}
    fallback_new_idx = new_vocab[fallback]
    return {
        old_idx: new_vocab.get(cat, fallback_new_idx)
        for old_idx, cat in idx_to_cat.items()
    }


def _apply_remap_to_long_tensor(t, remap):
    if t.numel() == 0:
        return t
    out = t.clone()
    for old_idx, new_idx in remap.items():
        out[t == old_idx] = new_idx
    return out


def _apply_remap_to_float_column(t, col, remap):
    if t.numel() == 0:
        return t
    out = t.clone()
    col_vals = t[:, col]
    for old_idx, new_idx in remap.items():
        out[col_vals == float(old_idx), col] = float(new_idx)
    return out


def remap_one_graph(data, hw_remap, bt_remap):
    """Mutates a loaded HeteroData in place, returns it. Every site listed
    in the module docstring is patched; anything else in the graph never
    stored a vocab index and is left untouched."""
    if "incident" in data.node_types and hasattr(data["incident"], "highway_type_idx"):
        data["incident"].highway_type_idx = _apply_remap_to_long_tensor(
            data["incident"].highway_type_idx, hw_remap
        )

    if "building" in data.node_types and hasattr(data["building"], "type_idx"):
        data["building"].type_idx = _apply_remap_to_long_tensor(
            data["building"].type_idx, bt_remap
        )

    connects_key = ("intersection", "connects", "intersection")
    if connects_key in data.edge_types:
        data[connects_key].edge_attr = _apply_remap_to_float_column(
            data[connects_key].edge_attr, col=0, remap=hw_remap
        )

    for seg_key in [("incident", "on_segment", "intersection"),
                     ("intersection", "on_segment", "incident")]:
        if seg_key in data.edge_types:
            data[seg_key].edge_attr = _apply_remap_to_float_column(
                data[seg_key].edge_attr, col=1, remap=hw_remap
            )

    return data


def remap_city_graphs(tvg_dir, out_dir, old_highway_vocab, new_highway_vocab,
                       old_building_vocab, new_building_vocab,
                       highway_fallback="unclassified", building_fallback="unknown",
                       point_ids=None):
    """Reads every .pt in tvg_dir (or, if point_ids is given, only those
    specific {point_id}.pt files), remaps vocab indices, writes to out_dir
    (kept SEPARATE from tvg_dir on purpose — never overwrite the originals
    in place; verify the remapped copies first, then swap directories).

    point_ids: needed once multiple cities share ONE combined tvg_dir
    (see 04b_vocab_unification.ipynb's multi-city setup) — each city's
    remap pass must only touch its own files, not every *.pt in the
    shared directory. None (default) preserves the original
    glob-everything behavior, for a tvg_dir that only ever holds one
    city's graphs.
    """
    tvg_dir = Path(tvg_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hw_remap = _remap_map(old_highway_vocab, new_highway_vocab, highway_fallback)
    bt_remap = _remap_map(old_building_vocab, new_building_vocab, building_fallback)

    if point_ids is not None:
        pt_files = [tvg_dir / f"{pid}.pt" for pid in point_ids]
        missing = [p for p in pt_files if not p.exists()]
        if missing:
            print(f"  ⚠️  {len(missing)} expected file(s) not found in {tvg_dir} "
                  f"(first 5): {[p.name for p in missing[:5]]}")
        pt_files = [p for p in pt_files if p.exists()]
    else:
        pt_files = sorted(tvg_dir.glob("*.pt"))

    n_ok, n_err = 0, 0
    for pt_path in pt_files:
        try:
            data = torch.load(pt_path, weights_only=False)
            data = remap_one_graph(data, hw_remap, bt_remap)
            torch.save(data, out_dir / pt_path.name)
            n_ok += 1
        except Exception as e:
            print(f"  ⚠️  {pt_path.name}: {e}")
            n_err += 1

    print(f"Remapped {n_ok}/{len(pt_files)} graphs from {tvg_dir} -> {out_dir} "
          f"({n_err} errors)")
    return n_ok, n_err


def remap_buildings_parquet_type_idx(parquet_path, old_building_vocab, new_building_vocab,
                                      building_fallback="unknown"):
    """Updates the cached buildings.parquet's type_idx column so any FUTURE
    per-incident processing for this city (e.g. new points added later)
    reads consistent indices without needing this script re-run."""
    bt_remap = _remap_map(old_building_vocab, new_building_vocab, building_fallback)
    gdf = gpd.read_parquet(parquet_path)
    gdf["type_idx"] = gdf["type_idx"].map(bt_remap).fillna(new_building_vocab[building_fallback]).astype(int)
    gdf.to_parquet(parquet_path)
    return gdf


def swap_in_remapped_graphs(tvg_dir, remapped_dir, backup_suffix="_pre_unification_backup"):
    """Only call this AFTER spot-checking remapped_dir. Moves the original
    tvg_dir aside as a backup, then moves remapped_dir into its place."""
    tvg_dir = Path(tvg_dir)
    remapped_dir = Path(remapped_dir)
    backup_dir = tvg_dir.parent / f"{tvg_dir.name}{backup_suffix}"
    if backup_dir.exists():
        raise FileExistsError(f"Backup dir already exists, refusing to overwrite: {backup_dir}")
    shutil.move(str(tvg_dir), str(backup_dir))
    shutil.move(str(remapped_dir), str(tvg_dir))
    print(f"Swapped in remapped graphs. Original preserved at: {backup_dir}")

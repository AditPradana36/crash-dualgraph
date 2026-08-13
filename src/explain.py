"""
Post-hoc explainability for the SVG/TVG/Unified encoders: native GATv2
attention-weight extraction, a GNNExplainer-style learned mask
explanation, and a tidy aggregator over many explained points.

Both techniques are implemented directly against this codebase's own
encoder.forward(data, batch_dict) signature rather than forced through
PyTorch Geometric's generic Explainer wrapper -- that wrapper expects a
single homogeneous (x, edge_index) forward signature, not a multi-
relation HeteroData graph plus this codebase's own
assemble_inputs/assemble_edge_attrs input-assembly step. Reimplementing
the (small) core ideas directly stays correct and in sync with
src/models.py rather than fighting a library abstraction built for a
different input shape.

Functions:
- extract_attention_weights(encoder, data) -> dict
- run_gnnexplainer(encoder, data, predict_fn, ...) -> (node_mask_dict, edge_mask_dict, final_fidelity_loss)
- explain_scenario_point(model, scenario, svg_data, tvg_data, ...) -> list[dict]
- aggregate_explanations(records) -> pandas.DataFrame
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_edge_dicts(encoder, data):
    """Reproduces exactly what encoder.forward() itself uses for
    edge_index_dict/edge_attr_dict -- SVGEncoder reads data's own
    edge_index_dict/edge_attr_dict properties directly; TVGEncoder/
    UnifiedEncoder route edge_attr through their own
    assemble_edge_attrs() (categorical-embedding folding for
    connects/on_segment). Keeping this one place in sync with
    models.py's forward() methods avoids extract_attention_weights/
    run_gnnexplainer silently drifting out of step with real inference."""
    if hasattr(encoder, "assemble_edge_attrs"):
        edge_index_dict = {k: data[k].edge_index for k in encoder._edge_types}
        edge_attr_dict = encoder.assemble_edge_attrs(data)
    else:
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
    return edge_index_dict, edge_attr_dict


def extract_attention_weights(encoder, data):
    """Runs `encoder`'s forward pass layer by layer, calling each
    relation's GATv2Conv directly with return_attention_weights=True
    (rather than through HeteroConv, which doesn't expose it) so the
    real per-neighbor attention weights the trained model actually used
    can be read out. Every layer's LayerNorm/ELU/Dropout(eval-mode) is
    replayed in between so layer 2's attention is computed on the exact
    input layer 1 produces during real inference, not a diverged
    approximation.

    Only defined for conv_type="gatv2" -- GINEConv has no attention
    mechanism to extract; raises ValueError rather than returning an
    empty/misleading result.

    Returns {layer_idx: {(src, rel, dst): (edge_index, alpha)}} --
    edge_index includes any self-loops GATv2Conv added internally
    (authoritative for interpreting alpha's rows), alpha has shape
    [num_edges, heads]. Both tensors detached, left on their original
    device."""
    conv_type = getattr(encoder, "conv_type", "gatv2")
    if conv_type != "gatv2":
        raise ValueError(
            f"extract_attention_weights only applies to conv_type='gatv2' "
            f"(got {conv_type!r}) -- GINEConv has no attention weights to extract."
        )

    encoder.eval()
    x_dict = encoder.assemble_inputs(data)
    edge_index_dict, edge_attr_dict = _get_edge_dicts(encoder, data)

    attention_by_layer = {}
    with torch.no_grad():
        for layer_idx, (hetero_conv, norm) in enumerate(zip(encoder.convs, encoder.norms)):
            layer_attn = {}
            out_parts = {}
            for edge_type, conv in hetero_conv.convs.items():
                if edge_type not in edge_index_dict:
                    continue
                src, rel, dst = edge_type
                edge_index = edge_index_dict[edge_type]
                edge_attr = edge_attr_dict.get(edge_type) if edge_attr_dict is not None else None
                x_in = x_dict[src] if src == dst else (x_dict[src], x_dict[dst])

                out, (ei, alpha) = conv(x_in, edge_index, edge_attr, return_attention_weights=True)
                layer_attn[edge_type] = (ei.detach(), alpha.detach())
                out_parts.setdefault(dst, []).append(out)

            # Same aggregation HeteroConv itself uses (every encoder builds
            # its HeteroConv with aggr="sum") -- replayed here since we
            # called each relation's conv directly instead of through
            # HeteroConv.
            x_dict = {
                nt: (torch.stack(parts, dim=0).sum(dim=0) if len(parts) > 1 else parts[0])
                for nt, parts in out_parts.items()
            }
            x_dict = {
                nt: F.dropout(F.elu(norm[nt](x)), p=encoder.dropout, training=False)
                for nt, x in x_dict.items()
            }
            attention_by_layer[layer_idx] = layer_attn

    return attention_by_layer


def run_gnnexplainer(encoder, data, predict_fn, epochs=100, lr=0.05,
                      edge_size_coef=0.005, node_feat_size_coef=0.005,
                      entropy_coef=0.1, verbose=False):
    """GNNExplainer-style post-hoc explanation (Ying et al. 2019), adapted
    to this codebase's custom multi-relation forward pass.

    Learns a per-node soft importance mask (applied to `encoder`'s own
    INPUT projection, i.e. right after assemble_inputs) and -- for
    encoders that route edge features through their own
    assemble_edge_attrs() (TVGEncoder, UnifiedEncoder) -- a per-edge-type
    soft mask on edge_attr, so that the masked prediction reproduces the
    model's own ORIGINAL, unmasked prediction as closely as possible
    while both masks are pushed toward sparse/confident (near-0 or
    near-1) values. This is GNNExplainer's core mutual-information-style
    objective, implemented here as fidelity (BCE to the original
    prediction) + L1 sparsity + entropy regularization.

    SVGEncoder has NO edge masking available: its forward() reads
    data.edge_attr_dict directly (no assemble_edge_attrs() indirection
    to hook), so scaling edge_attr there would require mutating the
    input HeteroData itself. Rather than do that (risks leaving a shared
    `data` object mutated for later cells), SVGEncoder explanations are
    NODE-MASK-ONLY -- explicit, not a silent gap.

    encoder: the SVGEncoder/TVGEncoder/UnifiedEncoder instance whose
        INPUT this call explains (one branch at a time -- for dual-graph
        scenarios C/D/E, call this once per branch; the other branch's
        data must already be fixed, unmasked, inside `predict_fn`'s
        closure).
    data: the (already-normalized) HeteroData graph fed to `encoder`,
        matching what the trained model actually saw at inference.
    predict_fn: callable(data) -> raw logit (0-d tensor) using the SAME
        trained scenario model this explanation is for. Any other
        branch's data must already be captured in this closure. Called
        once with the unmodified `data` to get the original prediction,
        then once per optimization epoch with `encoder`'s
        assemble_inputs/assemble_edge_attrs transparently masked.

    Returns (node_mask_dict, edge_mask_dict, final_fidelity_loss):
    node_mask_dict / edge_mask_dict map node/edge type -> a detached,
    post-sigmoid [0, 1] importance tensor (higher = more important to
    the prediction). edge_mask_dict is {} for SVGEncoder. Model weights
    are never updated -- only the mask parameters are in the optimizer."""
    encoder.eval()

    base_x_dict = encoder.assemble_inputs(data)
    has_edge_hook = hasattr(encoder, "assemble_edge_attrs")
    base_edge_attr_dict = encoder.assemble_edge_attrs(data) if has_edge_hook else {}

    device = next(iter(base_x_dict.values())).device if base_x_dict else torch.device("cpu")

    node_mask_params = {
        nt: nn.Parameter(torch.zeros(x.shape[0], device=device))
        for nt, x in base_x_dict.items() if x.shape[0] > 0
    }
    edge_mask_params = {}
    if has_edge_hook:
        for key, attr in base_edge_attr_dict.items():
            if attr is None or attr.shape[0] == 0:
                continue
            edge_mask_params[key] = nn.Parameter(torch.zeros(attr.shape[0], device=device))

    if not node_mask_params and not edge_mask_params:
        return {}, {}, float("nan")

    with torch.no_grad():
        original_logit = predict_fn(data).detach()
        original_pred = (torch.sigmoid(original_logit) > 0.5).float()

    optimizer = torch.optim.Adam(
        list(node_mask_params.values()) + list(edge_mask_params.values()), lr=lr
    )

    orig_assemble_inputs = encoder.assemble_inputs
    orig_assemble_edge_attrs = encoder.assemble_edge_attrs if has_edge_hook else None

    def masked_assemble_inputs(d):
        x_dict = orig_assemble_inputs(d)
        return {
            nt: (x * torch.sigmoid(node_mask_params[nt]).unsqueeze(-1) if nt in node_mask_params else x)
            for nt, x in x_dict.items()
        }

    def masked_assemble_edge_attrs(d):
        edge_attr_dict = orig_assemble_edge_attrs(d)
        return {
            key: (attr * torch.sigmoid(edge_mask_params[key]).unsqueeze(-1)
                  if key in edge_mask_params and attr is not None and attr.shape[0] > 0 else attr)
            for key, attr in edge_attr_dict.items()
        }

    final_fidelity = None
    try:
        encoder.assemble_inputs = masked_assemble_inputs
        if has_edge_hook:
            encoder.assemble_edge_attrs = masked_assemble_edge_attrs

        for epoch in range(epochs):
            optimizer.zero_grad()
            masked_logit = predict_fn(data)
            fidelity_loss = F.binary_cross_entropy_with_logits(masked_logit, original_pred)

            node_size_loss = sum((torch.sigmoid(p).sum() for p in node_mask_params.values()),
                                  torch.tensor(0.0, device=device))
            edge_size_loss = sum((torch.sigmoid(p).sum() for p in edge_mask_params.values()),
                                  torch.tensor(0.0, device=device))
            all_params = list(node_mask_params.values()) + list(edge_mask_params.values())
            entropy_loss = sum(
                (-torch.sigmoid(p) * torch.log(torch.sigmoid(p) + 1e-8)
                 - (1 - torch.sigmoid(p)) * torch.log(1 - torch.sigmoid(p) + 1e-8)).mean()
                for p in all_params
            ) / max(len(all_params), 1)

            loss = (fidelity_loss + node_feat_size_coef * node_size_loss
                    + edge_size_coef * edge_size_loss + entropy_coef * entropy_loss)
            loss.backward()
            optimizer.step()
            final_fidelity = fidelity_loss.item()

            if verbose and epochs >= 5 and epoch % (epochs // 5) == 0:
                print(f"    gnnexplainer epoch {epoch:3d} | fidelity_loss={fidelity_loss.item():.4f}")
    finally:
        encoder.assemble_inputs = orig_assemble_inputs
        if has_edge_hook:
            encoder.assemble_edge_attrs = orig_assemble_edge_attrs

    node_mask_dict = {nt: torch.sigmoid(p).detach().cpu() for nt, p in node_mask_params.items()}
    edge_mask_dict = {key: torch.sigmoid(p).detach().cpu() for key, p in edge_mask_params.items()}
    return node_mask_dict, edge_mask_dict, final_fidelity


def explain_scenario_point(model, scenario, svg_data, tvg_data, point_id, category=None,
                            gnnexplainer_epochs=100, gnnexplainer_lr=0.05, extract_attention=True):
    """Runs both explanation techniques for ONE already-normalized point
    against an already-trained scenario `model` (any of build_model()'s
    A/B/C/D/E/F outputs -- G has no GNN encoder, not supported here).
    Handles the per-scenario branch structure (single-encoder A/B/F vs.
    dual-encoder C/D/E) so callers (08g/08i notebooks) don't need to
    special-case scenario types themselves.

    For dual-graph scenarios (C/D/E), each branch is explained
    SEPARATELY, holding the other branch's data fixed/unmasked -- see
    run_gnnexplainer's own docstring for why masking both branches at
    once isn't attempted. Returned records tag the branch into
    `scenario` itself (e.g. "C_svg"/"C_tvg") so aggregate_explanations'
    output stays self-descriptive without a schema change.

    extract_attention: skipped automatically (not silently) whenever the
    branch's encoder isn't conv_type="gatv2" -- see
    extract_attention_weights' own ValueError for GINEConv.

    Returns a list of record dicts (see aggregate_explanations), one per
    explained branch: 1 for A/B/F, 2 for C/D/E."""
    import unified_graph as ug

    records = []

    def _record(sub_scenario, encoder, data, predict_fn):
        rec = {"point_id": point_id, "category": category, "scenario": sub_scenario}
        if extract_attention and getattr(encoder, "conv_type", "gatv2") == "gatv2":
            rec["attention"] = extract_attention_weights(encoder, data)
        node_mask, edge_mask, _fidelity = run_gnnexplainer(
            encoder, data, predict_fn, epochs=gnnexplainer_epochs, lr=gnnexplainer_lr)
        rec["node_mask"] = node_mask
        rec["edge_mask"] = edge_mask
        records.append(rec)

    if scenario == "A":
        _record("A", model.encoder, svg_data, lambda d: model(d))
    elif scenario == "B":
        _record("B", model.encoder, tvg_data, lambda d: model(d))
    elif scenario == "F":
        merged = ug.merge_svg_tvg(svg_data, tvg_data)
        _record("F", model.encoder, merged, lambda d: model(d))
    elif scenario in ("C", "D", "E"):
        _record(f"{scenario}_svg", model.svg_encoder, svg_data,
                 lambda d: model(d, tvg_data, None, None))
        _record(f"{scenario}_tvg", model.tvg_encoder, tvg_data,
                 lambda d: model(svg_data, d, None, None))
    else:
        raise ValueError(
            f"explain_scenario_point: unsupported scenario {scenario!r} -- "
            f"G has no GNN encoder; pass an ablation's base letter (e.g. 'B' "
            f"for 'B_ablation'), not the '_ablation' suffix itself."
        )

    return records


def aggregate_explanations(records):
    """Tidies many explained points' results -- from run_gnnexplainer
    and/or extract_attention_weights -- into ONE long-format
    pandas.DataFrame, so GNNExplainer's node/edge importance and GATv2's
    raw attention weights can be compared side by side on a shared
    schema, rather than needing two separate aggregation passes.

    records: list of dicts, each describing ONE explained point:
      {"point_id": str, "category": "TP"/"TN"/"FP"/"FN" (optional),
       "scenario": str (optional),
       "node_mask": {node_type: Tensor} or None/absent,   # run_gnnexplainer
       "edge_mask": {edge_type: Tensor} or None/absent,   # run_gnnexplainer
       "attention": {layer_idx: {edge_type: (edge_index, alpha)}} or None/absent}  # extract_attention_weights

    Returns a DataFrame with columns: point_id, category, scenario,
    source ("gnnexplainer" or "attention_layer{N}"), kind ("node" or
    "edge"), type, mean_value, max_value, n -- one row per
    (point, source, type). Empty tensors are skipped, not zero-filled."""
    import pandas as pd

    rows = []
    for rec in records:
        pid, cat, scen = rec.get("point_id"), rec.get("category"), rec.get("scenario")

        for nt, mask in (rec.get("node_mask") or {}).items():
            if mask.numel() == 0:
                continue
            rows.append({"point_id": pid, "category": cat, "scenario": scen,
                         "source": "gnnexplainer", "kind": "node", "type": nt,
                         "mean_value": mask.mean().item(), "max_value": mask.max().item(),
                         "n": mask.numel()})

        for et, mask in (rec.get("edge_mask") or {}).items():
            if mask.numel() == 0:
                continue
            rows.append({"point_id": pid, "category": cat, "scenario": scen,
                         "source": "gnnexplainer", "kind": "edge", "type": str(et),
                         "mean_value": mask.mean().item(), "max_value": mask.max().item(),
                         "n": mask.numel()})

        for layer_idx, layer_attn in (rec.get("attention") or {}).items():
            for et, (_edge_index, alpha) in layer_attn.items():
                if alpha.numel() == 0:
                    continue
                # alpha: [num_edges, heads] -- averaged over heads to one
                # scalar per edge, matching the granularity node_mask/
                # edge_mask already report at.
                alpha_per_edge = alpha.mean(dim=-1) if alpha.dim() > 1 else alpha
                rows.append({"point_id": pid, "category": cat, "scenario": scen,
                             "source": f"attention_layer{layer_idx}", "kind": "edge", "type": str(et),
                             "mean_value": alpha_per_edge.mean().item(), "max_value": alpha_per_edge.max().item(),
                             "n": alpha_per_edge.numel()})

    return pd.DataFrame(rows)


def build_type_importance_report(explain_df, top_n=5):
    """Summarizes aggregate_explanations' per-point output into a
    per-scheme (scenario) x per-type report -- averaged across every
    explained point, so scenarios/branches become directly comparable
    ("does C_svg lean on signage more than A does") instead of only
    being readable one point at a time.

    All 'attention_layerN' sources are folded into one 'attention'
    source first (mean across whatever layers that point/type has),
    so the two importance signals -- GNNExplainer's learned mask vs.
    GATv2's raw attention -- line up as exactly two comparable columns
    rather than one column per layer.

    Returns (pivot_df, topn_df):
      pivot_df: one row per (scenario, kind, type), columns
        'gnnexplainer' / 'attention' (mean importance across all
        explained points that had that type present; NaN where a
        source has no rows for that type), plus 'n_points'.
      topn_df: for each (scenario, source), the top_n types by mean
        importance, ranked -- columns scenario, source, rank, kind,
        type, mean_importance.
    """
    import pandas as pd

    if explain_df.empty:
        empty_pivot = pd.DataFrame(columns=["scenario", "kind", "type", "gnnexplainer", "attention", "n_points"])
        empty_topn = pd.DataFrame(columns=["scenario", "source", "rank", "kind", "type", "mean_importance"])
        return empty_pivot, empty_topn

    df = explain_df.copy()
    df["source"] = df["source"].apply(lambda s: "attention" if s.startswith("attention_layer") else s)

    grouped = (
        df.groupby(["scenario", "kind", "type", "source"])
        .agg(mean_importance=("mean_value", "mean"), n_points=("point_id", "nunique"))
        .reset_index()
    )

    pivot_df = grouped.pivot_table(
        index=["scenario", "kind", "type"], columns="source",
        values="mean_importance", aggfunc="mean",
    ).reset_index()
    n_points_df = grouped.groupby(["scenario", "kind", "type"])["n_points"].max().reset_index()
    pivot_df = pivot_df.merge(n_points_df, on=["scenario", "kind", "type"], how="left")
    for col in ("gnnexplainer", "attention"):
        if col not in pivot_df.columns:
            pivot_df[col] = float("nan")
    pivot_df = pivot_df[["scenario", "kind", "type", "gnnexplainer", "attention", "n_points"]]
    pivot_df = pivot_df.sort_values(["scenario", "kind", "gnnexplainer"], ascending=[True, True, False])

    topn_rows = []
    for (scenario, source), sub in grouped.groupby(["scenario", "source"]):
        ranked = sub.sort_values("mean_importance", ascending=False).head(top_n)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            topn_rows.append({
                "scenario": scenario, "source": source, "rank": rank,
                "kind": row.kind, "type": row.type, "mean_importance": row.mean_importance,
            })
    topn_df = pd.DataFrame(topn_rows)

    return pivot_df, topn_df


# ─────────────────────────────────────────────────────────────────────────
# Feature-level explanation (v2) -- masks individual NAMED FEATURE
# COMPONENTS (position, area, class embedding, building height, ...)
# per node, instead of one scalar per node. Requires each encoder's
# _assemble_raw_features(data) hook (src/models.py) -- the raw,
# pre-input_proj concatenated feature vector per node type.
# ─────────────────────────────────────────────────────────────────────────

def get_feature_components(encoder, node_type):
    """Named (component_name, width) segments describing node_type's raw
    feature vector BEFORE input_proj, in the exact concatenation order
    encoder._assemble_raw_features uses. Widths are read from the
    encoder's own embedding/projection modules (not hardcoded), so this
    stays correct if cat_embed_dim / building_type_embed_dim / etc change.

    Granularity: every continuous raw field gets its own named slot; a
    categorical field's embedding block is ONE named component -- masking
    individual embedding dimensions isn't semantically interpretable,
    only "was this categorical field used at all" is.

    Note: SVGEncoder's "building" (pos+area, 3 dims) and TVGEncoder's
    "building" (6 geometric fields + type/height/levels, richer) share a
    name but are NOT the same schema -- distinguished here by whether
    `encoder` has a `building_type_embed` attribute (TVG/Unified only).
    Always call this with the SAME encoder instance the data belongs to."""
    if node_type == "ego":
        return [("svf", 1), ("enclosure", 1), ("entropy", 1), ("pos_x", 1), ("pos_y", 1)]
    if node_type in ("signage", "light_pole", "road_marking"):
        embedder = {"signage": encoder.signage_embed, "light_pole": encoder.light_pole_embed,
                    "road_marking": encoder.road_marking_embed}[node_type]
        return [("pos_x", 1), ("pos_y", 1), ("area_norm", 1),
                ("class_embed", embedder.embed.embedding_dim)]
    if node_type == "incident":
        return [("fraction_along", 1), ("isovist_area", 1), ("isovist_compactness", 1),
                ("isovist_occlusivity", 1), ("highway_embed", encoder.highway_embed.embed.embedding_dim)]
    if node_type in ("building", "tvg_building") and hasattr(encoder, "building_type_embed"):
        return [("area", 1), ("perimeter", 1), ("circular_compactness", 1), ("elongation", 1),
                ("orientation", 1), ("shape_index", 1),
                ("type_embed", encoder.building_type_embed.embed.embedding_dim),
                ("height_feat", encoder.height_module.proj.out_features),
                ("levels_feat", encoder.levels_module.proj.out_features)]
    if node_type in ("building", "svg_building", "vegetation"):
        return [("pos_x", 1), ("pos_y", 1), ("area_norm", 1)]
    if node_type == "intersection":
        return [("betweenness", 1), ("orientation_entropy", 1)]
    if node_type == "peer_incident":
        return [("constant_placeholder", 1)]
    raise ValueError(f"get_feature_components: unknown node_type {node_type!r}")


def run_gnnexplainer_features(encoder, data, predict_fn, epochs=100, lr=0.05,
                               edge_size_coef=0.005, feature_size_coef=0.005,
                               entropy_coef=0.1, verbose=False):
    """Feature-level variant of run_gnnexplainer: learns a soft mask per
    NAMED FEATURE COMPONENT per node (see get_feature_components), not
    one scalar per node. Hooks encoder._assemble_raw_features (the raw,
    pre-input_proj feature vector) instead of assemble_inputs, so masking
    lands on interpretable named fields (e.g. "is this building's HEIGHT
    driving the prediction, or just its footprint AREA") rather than the
    opaque post-projection hidden_dim embedding.

    Edge masking is UNCHANGED from run_gnnexplainer (still one scalar per
    edge, not per edge-attribute-dimension) -- true per-edge-attribute
    feature masking would need an analogous raw/projected split inside
    assemble_edge_attrs, out of scope here (see explain.py's module-level
    notes / the conversation this was scoped in).

    Returns (feature_mask_dict, edge_mask_dict, final_fidelity_loss):
    feature_mask_dict[node_type] = {component_name: Tensor[n_nodes]},
    detached, post-sigmoid [0, 1] importance per named component.
    edge_mask_dict is {} for SVGEncoder (no edge hook), same as
    run_gnnexplainer."""
    encoder.eval()

    base_raw_dict = encoder._assemble_raw_features(data)
    has_edge_hook = hasattr(encoder, "assemble_edge_attrs")
    base_edge_attr_dict = encoder.assemble_edge_attrs(data) if has_edge_hook else {}

    device = next(iter(base_raw_dict.values())).device if base_raw_dict else torch.device("cpu")

    components_by_type = {}
    feature_mask_params = {}
    for nt, raw in base_raw_dict.items():
        if raw.shape[0] == 0:
            continue
        comps = get_feature_components(encoder, nt)
        components_by_type[nt] = comps
        feature_mask_params[nt] = nn.Parameter(torch.zeros(raw.shape[0], len(comps), device=device))

    edge_mask_params = {}
    if has_edge_hook:
        for key, attr in base_edge_attr_dict.items():
            if attr is None or attr.shape[0] == 0:
                continue
            edge_mask_params[key] = nn.Parameter(torch.zeros(attr.shape[0], device=device))

    if not feature_mask_params and not edge_mask_params:
        return {}, {}, float("nan")

    with torch.no_grad():
        original_logit = predict_fn(data).detach()
        original_pred = (torch.sigmoid(original_logit) > 0.5).float()

    optimizer = torch.optim.Adam(
        list(feature_mask_params.values()) + list(edge_mask_params.values()), lr=lr)

    orig_assemble_raw = encoder._assemble_raw_features
    orig_assemble_edge_attrs = encoder.assemble_edge_attrs if has_edge_hook else None

    def masked_assemble_raw(d):
        raw_dict = orig_assemble_raw(d)
        out = {}
        for nt, raw in raw_dict.items():
            if nt not in feature_mask_params:
                out[nt] = raw
                continue
            mask = torch.sigmoid(feature_mask_params[nt])  # [n_nodes, n_components]
            pieces = []
            offset = 0
            for c_idx, (_name, width) in enumerate(components_by_type[nt]):
                pieces.append(raw[:, offset:offset + width] * mask[:, c_idx:c_idx + 1])
                offset += width
            out[nt] = torch.cat(pieces, dim=1)
        return out

    def masked_assemble_edge_attrs(d):
        edge_attr_dict = orig_assemble_edge_attrs(d)
        return {
            key: (attr * torch.sigmoid(edge_mask_params[key]).unsqueeze(-1)
                  if key in edge_mask_params and attr is not None and attr.shape[0] > 0 else attr)
            for key, attr in edge_attr_dict.items()
        }

    final_fidelity = None
    try:
        encoder._assemble_raw_features = masked_assemble_raw
        if has_edge_hook:
            encoder.assemble_edge_attrs = masked_assemble_edge_attrs

        for epoch in range(epochs):
            optimizer.zero_grad()
            masked_logit = predict_fn(data)
            fidelity_loss = F.binary_cross_entropy_with_logits(masked_logit, original_pred)

            feat_size_loss = sum((torch.sigmoid(p).sum() for p in feature_mask_params.values()),
                                  torch.tensor(0.0, device=device))
            edge_size_loss = sum((torch.sigmoid(p).sum() for p in edge_mask_params.values()),
                                  torch.tensor(0.0, device=device))
            all_params = list(feature_mask_params.values()) + list(edge_mask_params.values())
            entropy_loss = sum(
                (-torch.sigmoid(p) * torch.log(torch.sigmoid(p) + 1e-8)
                 - (1 - torch.sigmoid(p)) * torch.log(1 - torch.sigmoid(p) + 1e-8)).mean()
                for p in all_params
            ) / max(len(all_params), 1)

            loss = (fidelity_loss + feature_size_coef * feat_size_loss
                    + edge_size_coef * edge_size_loss + entropy_coef * entropy_loss)
            loss.backward()
            optimizer.step()
            final_fidelity = fidelity_loss.item()

            if verbose and epochs >= 5 and epoch % (epochs // 5) == 0:
                print(f"    gnnexplainer(features) epoch {epoch:3d} | fidelity_loss={fidelity_loss.item():.4f}")
    finally:
        encoder._assemble_raw_features = orig_assemble_raw
        if has_edge_hook:
            encoder.assemble_edge_attrs = orig_assemble_edge_attrs

    feature_mask_dict = {
        nt: {name: torch.sigmoid(feature_mask_params[nt][:, i]).detach().cpu()
             for i, (name, _w) in enumerate(components_by_type[nt])}
        for nt in feature_mask_params
    }
    edge_mask_dict = {key: torch.sigmoid(p).detach().cpu() for key, p in edge_mask_params.items()}
    return feature_mask_dict, edge_mask_dict, final_fidelity


def explain_scenario_point_features(model, scenario, svg_data, tvg_data, point_id, category=None,
                                     gnnexplainer_epochs=100, gnnexplainer_lr=0.05):
    """Feature-level counterpart to explain_scenario_point -- same
    per-scenario branch structure (single-encoder A/B/F vs. dual-encoder
    C/D/E), but calls run_gnnexplainer_features instead of
    run_gnnexplainer, so node importance comes back broken down by named
    feature component rather than one scalar per node. Attention is NOT
    re-extracted here (see explain_scenario_point/extract_attention_weights
    for that) -- this function is scoped to feature-level GNNExplainer
    masks only, meant to run ALONGSIDE explain_scenario_point's output,
    not replace it.

    Returns a list of record dicts (see aggregate_feature_explanations),
    one per explained branch: 1 for A/B/F, 2 for C/D/E."""
    import unified_graph as ug

    records = []

    def _record(sub_scenario, encoder, data, predict_fn):
        rec = {"point_id": point_id, "category": category, "scenario": sub_scenario}
        feature_mask, edge_mask, _fidelity = run_gnnexplainer_features(
            encoder, data, predict_fn, epochs=gnnexplainer_epochs, lr=gnnexplainer_lr)
        rec["feature_mask"] = feature_mask
        rec["edge_mask"] = edge_mask
        records.append(rec)

    if scenario == "A":
        _record("A", model.encoder, svg_data, lambda d: model(d))
    elif scenario == "B":
        _record("B", model.encoder, tvg_data, lambda d: model(d))
    elif scenario == "F":
        merged = ug.merge_svg_tvg(svg_data, tvg_data)
        _record("F", model.encoder, merged, lambda d: model(d))
    elif scenario in ("C", "D", "E"):
        _record(f"{scenario}_svg", model.svg_encoder, svg_data,
                 lambda d: model(d, tvg_data, None, None))
        _record(f"{scenario}_tvg", model.tvg_encoder, tvg_data,
                 lambda d: model(svg_data, d, None, None))
    else:
        raise ValueError(
            f"explain_scenario_point_features: unsupported scenario {scenario!r} -- "
            f"G has no GNN encoder; pass an ablation's base letter (e.g. 'B' "
            f"for 'B_ablation'), not the '_ablation' suffix itself."
        )

    return records


def aggregate_feature_explanations(records):
    """Tidies run_gnnexplainer_features' output across many explained
    points into ONE long-format pandas.DataFrame -- feature-level
    counterpart to aggregate_explanations. Edge-mask rows are NOT
    included (unchanged from aggregate_explanations' own edge rows --
    avoid duplicating that report); this is scoped to per-node-type-
    per-feature-component importance only.

    records: list of dicts from explain_scenario_point_features:
      {"point_id": str, "category": ..., "scenario": str,
       "feature_mask": {node_type: {feature_name: Tensor}}, "edge_mask": ...}

    Returns a DataFrame: point_id, category, scenario, node_type,
    feature, mean_value, max_value, n -- one row per
    (point, node_type, feature component)."""
    import pandas as pd

    rows = []
    for rec in records:
        pid, cat, scen = rec.get("point_id"), rec.get("category"), rec.get("scenario")
        for nt, comp_masks in (rec.get("feature_mask") or {}).items():
            for feat_name, mask in comp_masks.items():
                if mask.numel() == 0:
                    continue
                rows.append({"point_id": pid, "category": cat, "scenario": scen,
                             "node_type": nt, "feature": feat_name,
                             "mean_value": mask.mean().item(), "max_value": mask.max().item(),
                             "n": mask.numel()})
    return pd.DataFrame(rows)


def build_feature_importance_report(feature_df, top_n=5):
    """Per-scheme (scenario) summary of aggregate_feature_explanations'
    output, analogous to build_type_importance_report but for named
    feature components instead of node/edge types.

    Returns (pivot_df, topn_df):
      pivot_df: one row per (scenario, node_type, feature), averaged
        across every explained point -- columns scenario, node_type,
        feature, mean_importance, n_points.
      topn_df: top_n features by mean importance per scenario, ranked --
        columns scenario, rank, node_type, feature, mean_importance."""
    import pandas as pd

    if feature_df.empty:
        empty_pivot = pd.DataFrame(columns=["scenario", "node_type", "feature", "mean_importance", "n_points"])
        empty_topn = pd.DataFrame(columns=["scenario", "rank", "node_type", "feature", "mean_importance"])
        return empty_pivot, empty_topn

    grouped = (
        feature_df.groupby(["scenario", "node_type", "feature"])
        .agg(mean_importance=("mean_value", "mean"), n_points=("point_id", "nunique"))
        .reset_index()
    )
    pivot_df = grouped.sort_values(["scenario", "mean_importance"], ascending=[True, False])

    topn_rows = []
    for scenario, sub in grouped.groupby("scenario"):
        ranked = sub.sort_values("mean_importance", ascending=False).head(top_n)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            topn_rows.append({"scenario": scenario, "rank": rank, "node_type": row.node_type,
                              "feature": row.feature, "mean_importance": row.mean_importance})
    topn_df = pd.DataFrame(topn_rows)

    return pivot_df, topn_df

"""
HeteroConv + GATv2Conv encoders for SVG/TVG, fusion heads for scenarios
A-G, and the missing-data / categorical-embedding building blocks both
encoders share.

NOTE: torch/torch_geometric aren't available in the sandbox this was
written in — syntax-checked only. First real validation happens in 06's
own QC cell (one forward pass against a real graph pair from 05).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, global_mean_pool


# ─────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────────

class MissingAwareContinuous(nn.Module):
    """height / building:levels / maxspeed: linear projection when present,
    ONE shared learned placeholder vector when the missing flag is set —
    never a fabricated numeric value, never mean-imputed."""
    def __init__(self, out_dim=4):
        super().__init__()
        self.proj = nn.Linear(1, out_dim)
        self.placeholder = nn.Parameter(torch.randn(out_dim) * 0.01)

    def forward(self, value, missing_flag):
        projected = self.proj(value)
        placeholder = self.placeholder.unsqueeze(0).expand(value.shape[0], -1)
        mask = (missing_flag > 0.5).expand(-1, projected.shape[1])
        return torch.where(mask, placeholder, projected)


class CategoricalEmbedder(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

    def forward(self, idx):
        return self.embed(idx)


def embed_edge_attr_with_categorical(edge_attr, cat_col, embedder):
    """FIX: several edge_attr tensors as saved (connects, on_segment) carry
    a raw highway-type INDEX sitting in a continuous float column — feeding
    that straight into GATv2Conv's edge_dim would treat category index 3 as
    numerically 'more' than index 1, which is meaningless for road types.
    This splits the categorical column out, embeds it properly, and
    reconcatenates with the genuinely continuous columns."""
    if edge_attr.shape[0] == 0:
        emb_dim = embedder.embed.embedding_dim
        remaining_dim = edge_attr.shape[1] - 1
        return torch.zeros((0, emb_dim + remaining_dim), device=edge_attr.device)
    cat_idx = edge_attr[:, cat_col].long()
    continuous = torch.cat([edge_attr[:, :cat_col], edge_attr[:, cat_col + 1:]], dim=1)
    embedded = embedder(cat_idx)
    return torch.cat([embedded, continuous], dim=1)


def _pool_and_anchor(x_dict, batch_dict, anchor_type):
    """Readout: concat[global_mean_pool(all nodes), anchor node embedding].
    batch_dict: optional dict of per-node-type batch-index tensors (from
    PyG's HeteroData batching in 07); None = treat as a single graph.

    FIX: global_mean_pool only emits a row for graph-indices that actually
    appear in `batch`. When some graphs in the batch have zero nodes of a
    given type (e.g. a point with no signage detected), that graph's row
    goes missing entirely instead of coming back as zeros -- silently
    shrinking that node type's pooled tensor and breaking the later
    torch.stack across node types (mismatched batch dim). We force every
    node type to align to the full batch size, taken from anchor_type
    (assumed always present, exactly one per graph).

    Every torch.zeros/torch.arange fallback below is pinned to the anchor's
    device explicitly -- torch.zeros(...) with no device= always defaults
    to CPU, which crashes the moment it's concatenated with a CUDA tensor
    (e.g. inside a downstream nn.Linear) once training actually runs on GPU."""
    anchor = x_dict[anchor_type]
    device = anchor.device
    num_graphs = batch_dict[anchor_type].max().item() + 1 if batch_dict is not None else 1

    pooled_parts = []
    for nt, x in x_dict.items():
        if x.shape[0] == 0:
            # No nodes of this type anywhere in the batch -> contribute zeros
            # for every graph rather than dropping this node type entirely.
            pooled_parts.append(torch.zeros(num_graphs, x.shape[1], device=x.device))
            continue
        batch = batch_dict[nt] if batch_dict is not None else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        pooled_parts.append(global_mean_pool(x, batch, size=num_graphs))
    pooled = (torch.stack(pooled_parts, dim=0).sum(dim=0) if pooled_parts
              else torch.zeros(num_graphs, list(x_dict.values())[0].shape[1], device=device))

    if batch_dict is not None:
        anchor_batch = batch_dict[anchor_type]
        anchor_pooled = global_mean_pool(anchor, anchor_batch, size=num_graphs)  # anchor is always exactly 1 per graph
    else:
        anchor_pooled = anchor.mean(dim=0, keepdim=True)

    return torch.cat([pooled, anchor_pooled], dim=1)


# ─────────────────────────────────────────────────────────────────────────
# SVG Encoder
# ─────────────────────────────────────────────────────────────────────────

SVG_NODE_TYPES = ["ego", "signage", "light_pole", "road_marking", "building", "vegetation"]

SVG_EDGE_TYPES = (
    [("ego", "sees", nt) for nt in ["signage", "light_pole", "road_marking", "building", "vegetation"]]
    + [(nt, "sees", "ego") for nt in ["signage", "light_pole", "road_marking", "building", "vegetation"]]
    + [("signage", "mounted_with", "signage"), ("light_pole", "mounted_with", "light_pole"),
       ("signage", "mounted_with", "light_pole"), ("light_pole", "mounted_with", "signage")]
    + [(a, "near", b) for i, a in enumerate(["signage", "light_pole", "road_marking", "building", "vegetation"])
       for b in ["signage", "light_pole", "road_marking", "building", "vegetation"][i:]]
    + [(b, "near", a) for i, a in enumerate(["signage", "light_pole", "road_marking", "building", "vegetation"])
       for b in ["signage", "light_pole", "road_marking", "building", "vegetation"][i:] if a != b]
)

SVG_EDGE_DIM = {"sees": 2, "mounted_with": 1, "near": 1}


class SVGEncoder(nn.Module):
    def __init__(self, hidden_dim=64, heads=4, num_layers=2, dropout=0.35,
                 signage_vocab=5, light_pole_vocab=4, road_marking_vocab=2, cat_embed_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.signage_embed = CategoricalEmbedder(signage_vocab, cat_embed_dim)
        self.light_pole_embed = CategoricalEmbedder(light_pole_vocab, cat_embed_dim)
        self.road_marking_embed = CategoricalEmbedder(road_marking_vocab, cat_embed_dim)

        self.input_proj = nn.ModuleDict({
            "ego": nn.Linear(5, hidden_dim),                          # svf, enclosure, entropy, pos_x, pos_y
            "signage": nn.Linear(2 + 1 + cat_embed_dim, hidden_dim),  # pos(2) + area(1) + class_embed
            "light_pole": nn.Linear(2 + 1 + cat_embed_dim, hidden_dim),
            "road_marking": nn.Linear(2 + 1 + cat_embed_dim, hidden_dim),
            "building": nn.Linear(2 + 1, hidden_dim),                 # pos(2) + area(1), single class
            "vegetation": nn.Linear(2 + 1, hidden_dim),
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for (src, rel, dst) in SVG_EDGE_TYPES:
                conv_dict[(src, rel, dst)] = GATv2Conv(
                    hidden_dim, hidden_dim // heads, heads=heads, concat=True,
                    edge_dim=SVG_EDGE_DIM[rel], add_self_loops=(src == dst),
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in SVG_NODE_TYPES}))
        self.dropout = dropout

    def assemble_inputs(self, data):
        x_dict = {"ego": data["ego"].x}

        for nt, embedder in [("signage", self.signage_embed), ("light_pole", self.light_pole_embed),
                              ("road_marking", self.road_marking_embed)]:
            pos_area = torch.cat([data[nt].x, data[nt].area_norm], dim=1)
            if data[nt].class_idx.shape[0] > 0:
                cls_emb = embedder(data[nt].class_idx)
                x_dict[nt] = torch.cat([pos_area, cls_emb], dim=1)
            else:
                x_dict[nt] = torch.zeros((0, pos_area.shape[1] + embedder.embed.embedding_dim), device=pos_area.device)

        for nt in ["building", "vegetation"]:
            x_dict[nt] = torch.cat([data[nt].x, data[nt].area_norm], dim=1)

        return {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}

    def forward(self, data, batch_dict=None):
        x_dict = self.assemble_inputs(data)
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {nt: F.dropout(F.elu(norm[nt](x)), p=self.dropout, training=self.training)
                       for nt, x in x_dict.items()}

        return _pool_and_anchor(x_dict, batch_dict, anchor_type="ego")

    @property
    def out_dim(self):
        return self.hidden_dim * 2  # pool + anchor concat


# ─────────────────────────────────────────────────────────────────────────
# TVG Encoder
# ─────────────────────────────────────────────────────────────────────────

TVG_NODE_TYPES = ["incident", "building", "intersection", "peer_incident"]

TVG_EDGE_TYPES = [
    ("incident", "anchors", "building"), ("building", "anchors", "incident"),
    ("incident", "anchors", "intersection"), ("intersection", "anchors", "incident"),
    ("building", "adjacent", "building"),
    ("intersection", "connects", "intersection"),
    ("building", "fronts", "intersection"),
    ("incident", "on_segment", "intersection"), ("intersection", "on_segment", "incident"),
    ("incident", "crash_history", "peer_incident"),  # ablation only
]

TVG_RAW_EDGE_DIM = {
    "anchors": 1, "adjacent": 1, "connects": 4, "fronts": 1, "on_segment": 5, "crash_history": 1,
}


class TVGEncoder(nn.Module):
    def __init__(self, hidden_dim=64, heads=4, num_layers=2, dropout=0.35,
                 building_type_vocab=58, highway_vocab=13,
                 building_type_embed_dim=8, highway_embed_dim=4,
                 continuous_missing_dim=4, use_ablation_edges=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_ablation_edges = use_ablation_edges

        self.building_type_embed = CategoricalEmbedder(building_type_vocab, building_type_embed_dim)
        # ONE shared highway embedding table — reused for the incident node's
        # own highway_type_idx feature AND for connects/on_segment edge_attr,
        # since both represent the same category vocabulary.
        self.highway_embed = CategoricalEmbedder(highway_vocab, highway_embed_dim)

        self.height_module = MissingAwareContinuous(continuous_missing_dim)
        self.levels_module = MissingAwareContinuous(continuous_missing_dim)
        self.maxspeed_module = MissingAwareContinuous(continuous_missing_dim)

        self.input_proj = nn.ModuleDict({
            "incident": nn.Linear(4 + highway_embed_dim, hidden_dim),
            "building": nn.Linear(6 + building_type_embed_dim + continuous_missing_dim * 2, hidden_dim),
            "intersection": nn.Linear(2, hidden_dim),
            "peer_incident": nn.Linear(1, hidden_dim),
        })

        edge_types = TVG_EDGE_TYPES if use_ablation_edges else \
            [e for e in TVG_EDGE_TYPES if e[1] != "crash_history"]
        if use_ablation_edges:
            self.input_proj["peer_incident"] = nn.Linear(1, hidden_dim)

        edge_dim_for = {}
        for (src, rel, dst) in edge_types:
            if rel == "connects":
                edge_dim_for[(src, rel, dst)] = (TVG_RAW_EDGE_DIM[rel] - 1) + highway_embed_dim
            elif rel == "on_segment":
                edge_dim_for[(src, rel, dst)] = (TVG_RAW_EDGE_DIM[rel] - 1) + highway_embed_dim
            else:
                edge_dim_for[(src, rel, dst)] = TVG_RAW_EDGE_DIM[rel]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for (src, rel, dst) in edge_types:
                conv_dict[(src, rel, dst)] = GATv2Conv(
                    hidden_dim, hidden_dim // heads, heads=heads, concat=True,
                    edge_dim=edge_dim_for[(src, rel, dst)], add_self_loops=(src == dst),
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            node_types = TVG_NODE_TYPES if use_ablation_edges else \
                [nt for nt in TVG_NODE_TYPES if nt != "peer_incident"]
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types}))
        self.dropout = dropout
        self._edge_types = edge_types

    def assemble_inputs(self, data):
        hw_idx = data["incident"].highway_type_idx
        hw_emb = self.highway_embed(hw_idx)
        incident_x = torch.cat([data["incident"].x, hw_emb], dim=1)

        n_b = data["building"].x.shape[0]
        if n_b > 0:
            type_emb = self.building_type_embed(data["building"].type_idx)
            height_feat = self.height_module(data["building"].height, data["building"].height_missing)
            levels_feat = self.levels_module(data["building"].levels, data["building"].levels_missing)
            building_x = torch.cat([data["building"].x, type_emb, height_feat, levels_feat], dim=1)
        else:
            dim = 6 + self.building_type_embed.embed.embedding_dim + self.height_module.proj.out_features * 2
            building_x = torch.zeros((0, dim), device=incident_x.device)

        x_dict = {
            "incident": incident_x,
            "building": building_x,
            "intersection": data["intersection"].x,
        }
        if self.use_ablation_edges:
            x_dict["peer_incident"] = data["peer_incident"].x

        return {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}

    def assemble_edge_attrs(self, data):
        edge_attr_dict = {}
        for (src, rel, dst) in self._edge_types:
            key = (src, rel, dst)
            raw = data[key].edge_attr
            if rel in ("connects", "on_segment"):
                cat_col = 0 if rel == "connects" else 1  # on_segment: [dist, hw_idx, ...]
                edge_attr_dict[key] = embed_edge_attr_with_categorical(raw, cat_col, self.highway_embed)
            else:
                edge_attr_dict[key] = raw
        return edge_attr_dict

    def forward(self, data, batch_dict=None):
        x_dict = self.assemble_inputs(data)
        edge_index_dict = {k: data[k].edge_index for k in self._edge_types}
        edge_attr_dict = self.assemble_edge_attrs(data)

        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {nt: F.dropout(F.elu(norm[nt](x)), p=self.dropout, training=self.training)
                       for nt, x in x_dict.items()}

        return _pool_and_anchor(x_dict, batch_dict, anchor_type="incident")

    @property
    def out_dim(self):
        return self.hidden_dim * 2


# ─────────────────────────────────────────────────────────────────────────
# Classifier head — depth is an open comparison (linear vs 2-layer MLP),
# NOT assumed; both get built, same depth used across every scenario.
# ─────────────────────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, depth="linear", hidden=32, dropout=0.35):
        super().__init__()
        if depth == "linear":
            self.net = nn.Linear(in_dim, 1)
        elif depth == "mlp2":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            raise ValueError(f"Unknown head depth: {depth}")

    def forward(self, x):
        return self.net(x)  # raw logit — sigmoid applied in the loss (BCEWithLogitsLoss)


# ─────────────────────────────────────────────────────────────────────────
# Fusion scenarios A-G
# ─────────────────────────────────────────────────────────────────────────

class ScenarioSingleBranch(nn.Module):
    """A (SVG only) or B (TVG only)."""
    def __init__(self, encoder, fusion_dim, head_depth="linear"):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(encoder.out_dim, fusion_dim)
        self.head = ClassifierHead(fusion_dim, depth=head_depth)

    def forward(self, data, batch_dict=None):
        emb = self.proj(self.encoder(data, batch_dict))
        return self.head(emb).squeeze(-1)


class ScenarioConcat(nn.Module):
    """C — early/feature fusion."""
    def __init__(self, svg_encoder, tvg_encoder, fusion_dim, head_depth="linear"):
        super().__init__()
        self.svg_encoder, self.tvg_encoder = svg_encoder, tvg_encoder
        self.proj_svg = nn.Linear(svg_encoder.out_dim, fusion_dim)
        self.proj_tvg = nn.Linear(tvg_encoder.out_dim, fusion_dim)
        self.head = ClassifierHead(fusion_dim * 2, depth=head_depth)

    def forward(self, svg_data, tvg_data, svg_batch=None, tvg_batch=None):
        s = self.proj_svg(self.svg_encoder(svg_data, svg_batch))
        t = self.proj_tvg(self.tvg_encoder(tvg_data, tvg_batch))
        return self.head(torch.cat([s, t], dim=1)).squeeze(-1)


class ScenarioLateFusion(nn.Module):
    """D — separate heads, LEARNED (not fixed) combination weight."""
    def __init__(self, svg_encoder, tvg_encoder, fusion_dim, head_depth="linear"):
        super().__init__()
        self.svg_encoder, self.tvg_encoder = svg_encoder, tvg_encoder
        self.proj_svg = nn.Linear(svg_encoder.out_dim, fusion_dim)
        self.proj_tvg = nn.Linear(tvg_encoder.out_dim, fusion_dim)
        self.head_svg = ClassifierHead(fusion_dim, depth=head_depth)
        self.head_tvg = ClassifierHead(fusion_dim, depth=head_depth)
        self.combine = nn.Linear(2, 1)  # learns how much to trust each branch's logit

    def forward(self, svg_data, tvg_data, svg_batch=None, tvg_batch=None):
        s = self.proj_svg(self.svg_encoder(svg_data, svg_batch))
        t = self.proj_tvg(self.tvg_encoder(tvg_data, tvg_batch))
        logit_s = self.head_svg(s)
        logit_t = self.head_tvg(t)
        return self.combine(torch.cat([logit_s, logit_t], dim=1)).squeeze(-1)


class ScenarioCrossAttention(nn.Module):
    """E — SVG and TVG embeddings as two tokens, attending to each other
    before the final decision. Most directly tests the 'double scheme'
    hypothesis: does each branch's relevance depend on the other's context."""
    def __init__(self, svg_encoder, tvg_encoder, fusion_dim, head_depth="linear", n_heads=2):
        super().__init__()
        self.svg_encoder, self.tvg_encoder = svg_encoder, tvg_encoder
        self.proj_svg = nn.Linear(svg_encoder.out_dim, fusion_dim)
        self.proj_tvg = nn.Linear(tvg_encoder.out_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(fusion_dim, n_heads, batch_first=True)
        self.head = ClassifierHead(fusion_dim * 2, depth=head_depth)

    def forward(self, svg_data, tvg_data, svg_batch=None, tvg_batch=None):
        s = self.proj_svg(self.svg_encoder(svg_data, svg_batch)).unsqueeze(1)   # (B, 1, F)
        t = self.proj_tvg(self.tvg_encoder(tvg_data, tvg_batch)).unsqueeze(1)   # (B, 1, F)
        tokens = torch.cat([s, t], dim=1)  # (B, 2, F)
        attended, _ = self.cross_attn(tokens, tokens, tokens)
        combined = attended.reshape(attended.shape[0], -1)  # (B, 2*F)
        return self.head(combined).squeeze(-1)


# Scenario F (unified merged graph via a new same_location edge) is
# deliberately NOT implemented yet — build and validate A-E first, per
# the established plan. Placeholder left here as the next thing to add.
class ScenarioUnifiedGraph(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Scenario F builds on a merged SVG+TVG node/edge namespace "
            "(needs svg_building/tvg_building renamed to avoid collision) "
            "— intentionally deferred until A-E give a working baseline."
        )


def build_model(scenario, fusion_dim=64, head_depth="linear", use_ablation=False,
                 svg_kwargs=None, tvg_kwargs=None):
    svg_kwargs = svg_kwargs or {}
    tvg_kwargs = dict(tvg_kwargs or {})
    tvg_kwargs["use_ablation_edges"] = use_ablation

    svg_enc = SVGEncoder(**svg_kwargs) if scenario in ("A", "C", "D", "E") else None
    tvg_enc = TVGEncoder(**tvg_kwargs) if scenario in ("B", "C", "D", "E") else None

    if scenario == "A":
        return ScenarioSingleBranch(svg_enc, fusion_dim, head_depth)
    if scenario == "B":
        return ScenarioSingleBranch(tvg_enc, fusion_dim, head_depth)
    if scenario == "C":
        return ScenarioConcat(svg_enc, tvg_enc, fusion_dim, head_depth)
    if scenario == "D":
        return ScenarioLateFusion(svg_enc, tvg_enc, fusion_dim, head_depth)
    if scenario == "E":
        return ScenarioCrossAttention(svg_enc, tvg_enc, fusion_dim, head_depth)
    if scenario == "F":
        return ScenarioUnifiedGraph()
    raise ValueError(f"Unknown scenario: {scenario} (G is XGBoost — see baseline_features.py, no torch model)")

# Fusion Scenario Architectures — Input to Output, A Through G

One diagram per scenario, each traced from raw graph input to the final
prediction. Every scenario shares the same building blocks (defined
once here, referenced by name in every diagram below) — only how they're
wired together differs.

## Shared building blocks

**GATv2 Encoder** (`SVGEncoder` / `TVGEncoder` / `UnifiedEncoder`,
`src/models.py`): 2 `HeteroConv` layers, one `GATv2Conv` per edge
relation per layer, `hidden_dim=128`, 4 attention heads, categorical
fields (object class, building type, highway type) through learned
embeddings, `LayerNorm → ELU → Dropout(0.3)` per layer.

**Readout**: collapses the encoder's per-node output into one
`z ∈ ℝ²⁵⁶` graph-level vector. Two interchangeable options (the only
axis that differs between `07g` and `07i` — see
`docs/07g_07i_architecture.md`):
- `pool_anchor` (default): `[Σ mean-pool per node type ‖ anchor node's own embedding]`
- `dgcnn` (SortPooling): rank all nodes by salience, keep top-`k`, `Conv1d → MaxPool1d → Conv1d → Linear`

**Classifier Head** (`ClassifierHead`, `depth="mlp2"` in `07g`/`07i`):
`Linear(in, 256) → ReLU → Dropout(0.5) → Linear(256, 1)` → one raw logit.

**Loss**: `BCEWithLogitsLoss` against the binary incident/non-incident label.

```mermaid
flowchart LR
    subgraph LEGEND["Legend — used inside every scenario diagram below"]
        direction LR
        L1["Graph input"] -.-> L2["GATv2 Encoder<br/>(2 layers, hidden_dim=128)"]
        L2 -.-> L3["Readout<br/>(pool_anchor or SortPooling)<br/>→ z ∈ ℝ²⁵⁶"]
        L3 -.-> L4["Fusion projection<br/>Linear(256, 256)"]
        L4 -.-> L5["Classifier Head<br/>(linear or mlp2)"]
        L5 -.-> L6["Logit → BCEWithLogitsLoss"]
    end
```

---

## Scenario A — SVG only

Single branch. The street-view graph alone decides the prediction.

```mermaid
flowchart LR
    SVG["SVG Graph<br/>(ego, signage, light_pole,<br/>road_marking, building, vegetation)"] --> ENC["SVGEncoder"]
    ENC --> RO["Readout → z_svg ∈ ℝ²⁵⁶"]
    RO --> PROJ["proj_svg: Linear(256,256)"]
    PROJ --> HEAD["ClassifierHead"]
    HEAD --> LOGIT["logit"]
    LOGIT --> LOSS["BCEWithLogitsLoss"]
```

## Scenario B — TVG only

Single branch. The top-view graph alone decides the prediction.

```mermaid
flowchart LR
    TVG["TVG Graph<br/>(incident, building, intersection,<br/>peer_incident[ablation])"] --> ENC["TVGEncoder"]
    ENC --> RO["Readout → z_tvg ∈ ℝ²⁵⁶"]
    RO --> PROJ["proj_tvg: Linear(256,256)"]
    PROJ --> HEAD["ClassifierHead"]
    HEAD --> LOGIT["logit"]
    LOGIT --> LOSS["BCEWithLogitsLoss"]
```

## Scenario C — Dual graph, concatenation (early fusion)

Both branches encoded independently, then concatenated before the head sees either one.

```mermaid
flowchart LR
    SVG["SVG Graph"] --> ENC1["SVGEncoder"] --> RO1["Readout → z_svg"]
    TVG["TVG Graph"] --> ENC2["TVGEncoder"] --> RO2["Readout → z_tvg"]
    RO1 --> P1["proj_svg → 256"]
    RO2 --> P2["proj_tvg → 256"]
    P1 --> CAT["Concat<br/>[proj_svg ‖ proj_tvg] ∈ ℝ⁵¹²"]
    P2 --> CAT
    CAT --> HEAD["ClassifierHead(in=512)"]
    HEAD --> LOGIT["logit"]
    LOGIT --> LOSS["BCEWithLogitsLoss"]
```

## Scenario D — Dual graph, late fusion

Both branches get their **own independent classifier head**, producing
two logits; a small learned layer decides how much to trust each.

```mermaid
flowchart LR
    SVG["SVG Graph"] --> ENC1["SVGEncoder"] --> RO1["Readout → z_svg"] --> P1["proj_svg → 256"] --> H1["head_svg (ClassifierHead)"] --> LS["logit_s"]
    TVG["TVG Graph"] --> ENC2["TVGEncoder"] --> RO2["Readout → z_tvg"] --> P2["proj_tvg → 256"] --> H2["head_tvg (ClassifierHead)"] --> LT["logit_t"]
    LS --> COMBINE["Linear(2,1)<br/>learned w1·logit_s + w2·logit_t"]
    LT --> COMBINE
    COMBINE --> LOGIT["final logit"]
    LOGIT --> LOSS["BCEWithLogitsLoss"]
```

## Scenario E — Dual graph, cross-attention

Both branches' embeddings become two tokens that attend to each other
before the head — tests whether each branch's relevance depends on the
other's context.

```mermaid
flowchart LR
    SVG["SVG Graph"] --> ENC1["SVGEncoder"] --> RO1["Readout → z_svg"] --> P1["proj_svg → 256<br/>(token 1)"]
    TVG["TVG Graph"] --> ENC2["TVGEncoder"] --> RO2["Readout → z_tvg"] --> P2["proj_tvg → 256<br/>(token 2)"]
    P1 --> TOK["Stack tokens<br/>[token1; token2] ∈ ℝ²ˣ²⁵⁶"]
    P2 --> TOK
    TOK --> ATTN["MultiheadAttention<br/>(self-attend, 2 heads)"]
    ATTN --> FLAT["Reshape → ℝ⁵¹²"]
    FLAT --> HEAD["ClassifierHead(in=512)"]
    HEAD --> LOGIT["logit"]
    LOGIT --> LOSS["BCEWithLogitsLoss"]
```

## Scenario F — Unified merged graph

SVG and TVG are merged into **one heterogeneous graph** *before*
encoding (not after) — `building` renamed `svg_building`/`tvg_building`
to avoid a name collision, plus a new `same_location` edge connecting
each point's `ego` node to its own `incident` node. One shared encoder,
one readout, one head.

```mermaid
flowchart LR
    SVG["SVG Graph"] --> MERGE["merge_svg_tvg()<br/>10 node types combined,<br/>+ same_location edge (ego ↔ incident)"]
    TVG["TVG Graph"] --> MERGE
    MERGE --> ENC["UnifiedEncoder"]
    ENC --> RO["Readout → z_unified ∈ ℝ²⁵⁶"]
    RO --> PROJ["proj: Linear(256,256)"]
    PROJ --> HEAD["ClassifierHead"]
    HEAD --> LOGIT["logit"]
    LOGIT --> LOSS["BCEWithLogitsLoss"]
```

## Scenario G — Tabular baseline (no graph, no GNN)

The only non-GNN scheme. Every node's features across both graphs are
flattened into one fixed-length row per point; a gradient-boosted tree
ensemble replaces the entire encoder/readout/head stack. Exists to test
whether graph *structure* adds value over the same raw information
presented flat.

```mermaid
flowchart LR
    SVG["SVG Graph"] --> FLAT["baseline_features.build_feature_table()<br/>flatten every node's features<br/>(counts, means, areas, positions, ...)"]
    TVG["TVG Graph"] --> FLAT
    FLAT --> ROW["One fixed-length feature vector per point"]
    ROW --> XGB["XGBClassifier<br/>(200 trees, max_depth=4)"]
    XGB --> PROB["predicted probability"]
```

---

## Summary table

| Scenario | Input graph(s) | Encoder(s) | Fusion mechanism | Head |
|---|---|---|---|---|
| A | SVG only | 1× SVGEncoder | — | ClassifierHead(256) |
| B | TVG only | 1× TVGEncoder | — | ClassifierHead(256) |
| C | SVG + TVG | SVGEncoder + TVGEncoder | Concatenation | ClassifierHead(512) |
| D | SVG + TVG | SVGEncoder + TVGEncoder | Learned logit combination | 2× ClassifierHead(256) + Linear(2,1) |
| E | SVG + TVG | SVGEncoder + TVGEncoder | Cross-attention (2 tokens) | ClassifierHead(512) |
| F | SVG ∪ TVG (merged) | 1× UnifiedEncoder | Pre-encoding graph merge | ClassifierHead(256) |
| G | SVG + TVG (flattened) | — (no GNN) | Feature concatenation | XGBoost (200 trees) |

Every scenario except G shares: `hidden_dim=128`, 2 GATv2 layers,
`fusion_dim=256`, `head_depth="mlp2"`, `head_hidden=256`,
`head_dropout=0.5`, and the same `readout` choice (`pool_anchor` for
`07g`, SortPooling for `07i`) — the fusion mechanism in the table above
is the only structural difference between A–F.

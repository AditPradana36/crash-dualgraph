# 07g / 07i — Pipeline and Architecture

`07g_train_eval_capacity_revision.ipynb` and `07i_train_eval_dgcnn.ipynb`
are a matched pair: identical data, identical hyperparameters, identical
training procedure — the **only** thing that differs between them is the
graph **readout** (how a variable-size set of node embeddings collapses
into one fixed-size graph vector). This document covers what both share,
then what each does differently.

## 1. Shared data pipeline

Both branches read from the same combined, multi-city dataset assembled
by `01`–`05`:

1. **`01`** samples/reconciles positive (crash) and negative points per
   city, assigns a globally city-prefixed `point_id` (`bog_positive_12`,
   `war_negative_34`, …).
2. **`02`/`03`** build the **Street View Graph (SVG)** per point:
   Mask2Former panoptic segmentation → 6 node types (`ego`, `signage`,
   `light_pole`, `road_marking`, `building`, `vegetation`) connected by
   `sees` / `mounted_with` / `near` edges.
3. **`04`** builds the **Top View Graph (TVG)** per point: OSM
   buildings/streets + isovist ray-casting → 4 node types (`incident`,
   `building`, `intersection`, `peer_incident`) connected by `anchors` /
   `adjacent` / `connects` / `fronts` / `on_segment` / (`crash_history`,
   ablation-only) edges.
4. **`04b`** unions each city's building-type/highway-type vocabularies
   into one shared vocabulary and remaps every graph's categorical
   indices — required before pooling cities into one embedding table.
5. **`05`** assembles the combined `dataset_index.parquet` (one row per
   point, `city` + `label` columns) and one shared `svg_graphs/` /
   `tvg_graphs/` directory across all cities.

Both `07g` and `07i` load this via `graph_datasets.DualGraphDataset` and
train scenarios **A–F** (G, the XGBoost tabular baseline, has no GNN
encoder and is only run in `07g`).

## 2. Shared encoder architecture

Each of SVG and TVG is encoded by its own heterogeneous GNN
(`src/models.py`: `SVGEncoder`, `TVGEncoder`; scenario F merges both into
one `UnifiedEncoder` over a combined 10-node-type graph):

- Per-node-type **input projection**: a `Linear` per node type into a
  shared `hidden_dim` (=128). Categorical fields (object class, building
  type, highway type) pass through learned embeddings first, never raw
  integers. Missing continuous fields (height, levels, maxspeed) use a
  linear projection when present, or one shared learned placeholder
  vector when flagged missing.
- **2 `HeteroConv` layers** (PyTorch Geometric), one `GATv2Conv` per edge
  relation per layer — 4 attention heads, head dim = `hidden_dim/heads`,
  edge features fed into attention where available. `LayerNorm → ELU →
  Dropout(0.3)` after each layer.
- **Readout** — the one point where `07g` and `07i` diverge (§4).

$$h^{(0)}_v = \text{InputProj}_{\tau(v)}(x_v), \qquad
h^{(\ell)} = \text{Dropout}\big(\text{ELU}(\text{LayerNorm}(\text{HeteroConv}^{(\ell)}(h^{(\ell-1)})))\big), \quad \ell = 1, 2$$

where $\tau(v)$ is node $v$'s type.

## 3. Shared fusion scenarios (A–F) and classifier head

| Scenario | Fusion mechanism |
|---|---|
| A | SVG only |
| B | TVG only |
| C | Concat: $[\text{proj}(z_{\text{svg}}) \Vert \text{proj}(z_{\text{tvg}})] \to$ head |
| D | Late fusion: separate heads → logits $\ell_s, \ell_t$; learned $w_1\ell_s + w_2\ell_t$ |
| E | Cross-attention: $z_{\text{svg}}, z_{\text{tvg}}$ as 2 tokens, self-attend, concat → head |
| F | Unified graph: SVG+TVG merged pre-encoding into one `HeteroConv` stack (`building` renamed `svg_building`/`tvg_building`; new `same_location` edge, ego↔incident) |

Every scenario's encoder output $z$ is projected to `fusion_dim` (=256),
then passed through a **2-layer MLP classifier head**
(`hidden=256, dropout=0.3` → 1 logit; `head_depth="mlp2"`, fixed, not
swept against the bare-linear alternative this run):

$$\text{logit} = W_2\,\text{Dropout}\big(\text{ReLU}(W_1 z + b_1)\big) + b_2$$

Scenarios B–F each have an ablation variant (`+`) adding
`crash_history`/`peer_incident`, gated behind `use_ablation`.

## 4. The one difference: readout

### 07g — `pool_anchor` (default)

Sum of per-node-type mean-pools, concatenated with the anchor node's own
embedding (`ego` for SVG, `incident` for TVG/unified):

$$z = \Big[\ \sum_{\tau} \text{MeanPool}\big(\{h^{(2)}_v : \tau(v)=\tau\}\big)\ \Big\Vert\ \text{MeanPool}\big(\{h^{(2)}_v : v \in \text{anchor}\}\big)\ \Big], \qquad \dim(z) = 2\cdot\text{hidden\_dim}$$

The anchor's own representation is **always** present in $z$ by
construction.

### 07i — `dgcnn` (SortPooling, Zhang et al. 2018)

`DGCNNReadout` (`src/models.py`) instead concatenates **every layer's**
per-node output (including the pre-message-passing $h^{(0)}$), sorts
nodes by salience, and runs 1-D convolutions over the fixed-size sorted
sequence:

1. **Cross-layer concat** per node: $H_v = [h^{(0)}_v \Vert h^{(1)}_v \Vert h^{(2)}_v]$,
   $\dim(H_v) = 3\cdot\text{hidden\_dim}$ (the "+1" layer is the paper's
   own convention — raw features count as the first "layer").
2. **SortPooling**: sort all nodes descending by their value in the
   *last* channel of $h^{(2)}$; truncate to the top $k$, or zero-pad if
   fewer than $k$ nodes exist. Output: $R \in \mathbb{R}^{k \times 3\text{hidden\_dim}}$,
   flattened to one length-$k{\cdot}3\text{hidden\_dim}$ sequence.
3. **1-D conv stack**:
   $$\text{Conv1d}(1,\ c_1,\ \text{kernel}{=}3\text{hidden\_dim},\ \text{stride}{=}3\text{hidden\_dim}) \to \text{MaxPool1d}(2) \to \text{Conv1d}(c_1,\ c_2,\ \text{kernel}{=}k_2) \to \text{Flatten} \to \text{Linear} \to z$$
   ($c_1{=}16$, $c_2{=}32$, $\dim(z) = 2\cdot\text{hidden\_dim}$ — same
   output width as `pool_anchor`, so both readouts plug into the same
   downstream fusion/head code unchanged.)

**Choosing $k$ — not a guess.** Zhang et al.'s own rule: $k$ = the 40th
percentile of the per-graph total node-count distribution (60% of graphs
get truncated to their most salient $k$ nodes, the smaller 40% are fully
retained). The post-SortPool conv stack imposes a hard floor,
$k \ge (k_2-1)\cdot2+2$, so the actual rule applied per encoder in `07i`'s
diagnostic cell is:

$$k = \max\Big(\ (k_2-1)\cdot 2 + 2,\ \ \big\lceil P_{40}(\text{node counts})\big\rceil\ \Big)$$

TVG's real graphs are small (mean ≈ 8.1 total nodes: `incident` + a
handful of `building`/`intersection`), well below the floor a paper-default
$k_2{=}5$ would impose ($k\ge10$) — so TVG uses a relaxed $k_2{=}3$
($k\ge6$) while SVG/Unified keep $k_2{=}5$. Expect the floor, not the
percentile rule, to decide TVG's $k$ more often than SVG's/Unified's;
that reflects TVG's small graph scale, not a bug.

**No anchor guarantee** — unlike `pool_anchor`, the anchor node can rank
outside the top-$k$ and be dropped from $z$ entirely. A genuine
architectural difference, tested empirically rather than assumed benign.

## 5. Shared training procedure and hyperparameters

Both branches read `configs/eval_capacity_revision.yaml` **unchanged**
(`07i` does not fork it) and the same capacity numbers
(`model_capacity_revision.yaml` / `model_dgcnn_comparison.yaml` differ
only by the `readout`-specific keys `07i` needs):

| | Value |
|---|---|
| `hidden_dim` | 128 |
| `fusion_dim` | 256 |
| `head_hidden` / `head_dropout` | 256 / 0.3 |
| encoder `dropout` | 0.3 |
| `svg_layers` / `tvg_layers` | 2 / 2 |
| `cat_embed_dim` / `building_type_embed_dim` / `highway_embed_dim` | 4 / 16 / 8 |
| `head_depth` | `mlp2` (fixed, not swept) |
| `conv_type` | `gatv2` (both branches — unchanged) |
| split | 70/15/15, plain random, stratified by `label` only |
| **`epoch_cap`** | **150, strictly** — `patience=150 ≥ epoch_cap` structurally prevents early stopping from firing within the loop; every repeat trains the full 150 epochs, not "at most 150" |
| `warmup_epochs` | 20 |
| **`primary_metric`** | **`accuracy`** (`train.py`'s own default now — inherited, not set explicitly in either config) — drives checkpoint selection, the `ReduceLROnPlateau` scheduler, and early-stop comparisons |
| optimizer | AdamW, lr $5\times10^{-3}$, weight decay $1\times10^{-4}$ |
| loss | `BCEWithLogitsLoss` (numerically-stable logit-space binary cross-entropy) |
| threshold | fixed at 0.5 |
| `n_repeats` | 5 (`run_scenario_random_repeats`, plain random re-splits) |

Because every one of these is identical between `07g` and `07i`, any
difference in reported metrics is attributable to the readout swap
alone — a genuine single-variable comparison, not a multi-factor one.

## 6. Per-point raw predictions

Every `train_one_fold` call (every scenario × repeat, both branches)
writes, alongside its per-epoch history JSON, a sibling
`{tag}_history/repeat{N}_test_predictions.json`: one record per test
point — `point_id`, `label`, `prob` (raw sigmoid output), `pred`
(thresholded), `category` (TP/TN/FP/FN at the chosen threshold). Both
notebooks end with a cell that walks every such file under their
`CHECKPOINT_DIR`, tags each row with its scenario/repeat, and writes one
combined `raw_test_predictions_all_scenarios.csv` to `METRICS_DIR` —
enabling point-by-point comparison (e.g. "which points does `07g`'s
`pool_anchor` get right that `07i`'s `dgcnn` gets wrong") on top of the
usual aggregate PR-AUC/AUROC/accuracy tables.

## 7. Software

Python 3.11, PyTorch + PyTorch Geometric (`GATv2Conv`, `HeteroConv`,
`SortAggregation`), scikit-learn, SciPy, XGBoost (scenario G, `07g`
only). Trained on Google Colab GPUs (bfloat16 mixed precision). Global
seed 42, per-repeat offset.

# Materials and Methods

> Numbers marked **[REPORT FROM RUN LOGS]** are configured targets, not
> measured outputs — this repository's notebooks are not executed here,
> so final sample sizes, class balance, and result figures must be
> pulled from actual run logs (`dataset_index.parquet`,
> `metrics_dgcnn_comparison/*.csv`) before submission. Every threshold,
> formula, and hyperparameter below was verified against the current
> source code, not reconstructed from memory.

## Overview

This study proposes a dual-graph heterogeneous graph attention network
for point-level traffic-crash risk classification (incident vs.
non-incident), combining an egocentric **Street View Graph (SVG)** and
an allocentric **Top View Graph (TVG)** per candidate location. Both
graphs are encoded by a Graph Attention Network v2 (GATv2) stack and
collapsed to a fixed-size representation via **SortPooling**, a
learned, node-salience-ranking readout (Zhang et al., 2018), rather
than the conventional pooling-plus-anchor-node concatenation used as
this study's own internal baseline. The proposed configuration is
referred to throughout as the **SortPooling variant**; the baselivne
sharing every other hyperparameter is referred to as the
**anchor-pooling baseline**.

```mermaid
flowchart TD
    A["Point Sampling & Reconciliation<br/>4 cities, globally unique point_id"] --> B["Panoptic Segmentation<br/>Mask2Former / Mapillary Vistas"]
    B --> C["Street View Graph Construction"]
    A --> D["Top View Graph Construction<br/>OSM buildings/streets + isovist"]
    C --> E["Cross-City Vocabulary Harmonization"]
    D --> E
    E --> F["Dataset Assembly<br/>combined index, shared graph directories"]
    F --> G["Architecture QC<br/>one forward pass per scenario"]
    G --> H["Model Training & Evaluation<br/>GATv2 encoders + SortPooling readout"]
    H --> I["Post-hoc Interpretability<br/>Attention weights + GNNExplainer"]
```
*Figure 1. End-to-end pipeline, from raw point sampling to the final
interpretability pass.*

## 1. Study Areas and Point Sampling

Four urban study areas were used: Bogor (Indonesia), Warsaw (Poland),
Kraków (Poland), and Somerville (Massachusetts, USA), selected to span
distinct street-network typologies, building densities, and traffic
contexts. Each candidate location contributed one incident coordinate
(used for all geometric processing) and one paired street-view image (a
single, forward-facing crop, not a full panorama).

Every candidate point was reconciled against its metadata and confirmed
image availability via inner join; points failing any check were
dropped. Negative (non-crash) points were sampled to match the positive
(crash) set's road/highway-type distribution rather than uniformly at
random, avoiding a road-type confound given that highway type is later
used as a model feature. Every point received a globally unique,
city-prefixed identifier (`bog_`, `war_`, `kra_`, `som_`) at this stage,
so that all four cities' data could subsequently be pooled into one
combined dataset without collision.

**Target sample size** [REPORT FROM RUN LOGS]: ≈2,200 positive and
≈2,200 negative points, pooled across all four cities; final N depends
on reconciliation yield and downstream graph-construction success
(§6).

## 2. Street-View Graph (SVG) Construction

Panoptic segmentation was performed with Mask2Former
(`facebook/mask2former-swin-large-mapillary-vistas-panoptic`), trained
on the Mapillary Vistas v1.2 taxonomy (65 classes) (Cheng et al., 2022;
Neuhold et al., 2017).

**Nodes (6 types).** `ego` (one per graph; features: sky-view factor,
enclosure index, Shannon entropy of the segmentation, fixed viewpoint
position) and five object types — `signage`, `light_pole`,
`road_marking`, `building`, `vegetation` — each carrying normalized
centroid position and normalized area. Object instances were split from
Mask2Former's "thing" detections directly, or from "stuff" segments via
connected-component labeling. Minimum retained object area was 0.08% of
image area for signage/light_pole/road_marking, and 5% for
building/vegetation.

**Edges (3 relations, bidirectional where noted).**

| Relation | Trigger | Edge feature |
|---|---|---|
| `sees` (bidirectional) | `ego` ↔ every object node | area, distance/diagonal |
| `mounted_with` | centroid distance ≤ 0.04 × image diagonal (bounding-box overlap is **not** required) | bounding-box overlap ratio |
| `near` (bidirectional) | centroid distance ≤ 0.175 × image diagonal | distance/diagonal |

`mounted_with` connects `signage`↔`signage`, `light_pole`↔`light_pole`,
and `signage`↔`light_pole` pairs only.

## 3. Top-View Graph (TVG) Construction

Each point's allocentric surroundings were represented from
OpenStreetMap building footprints and street topology, combined with an
isovist computed at the incident coordinate: 360 rays cast to a 50 m
radius against a spatial index of building boundaries.

**Nodes (4 types).** `incident` (one per graph; fractional position
along its nearest road segment, isovist area/compactness/openness,
highway type) · `building` (area, perimeter, circular compactness,
elongation, orientation, shape index, building type, height/level
count with explicit missing-value flags) · `intersection` (betweenness
centrality, orientation entropy) · `peer_incident`
(**ablation-only**, constant placeholder).

**Inclusion rule.** A building or intersection enters the graph if and
only if it is spatially covered by the incident's isovist polygon — no
road-edge endpoints are force-included.

**Edges (6 relations).**

| Relation | Trigger | Edge feature |
|---|---|---|
| `anchors` (bidirectional) | `incident` ↔ every included building/intersection | distance (m) |
| `adjacent` | 5 nearest other included buildings, by real centroid distance (not a full clique) | distance (m) |
| `connects` | real OSM street edges between included intersections | highway type, maxspeed, oneway |
| `fronts` | building → its nearest included intersection | distance (m) |
| `on_segment` (bidirectional) | `incident` ↔ its nearest road edge's two endpoints | distance, highway type, maxspeed, oneway |
| `crash_history` (**ablation-only**) | `incident` → every other **positive-label, same-city** point within 100 m | distance (m) |

The `crash_history` relation deliberately excludes negative-label
points (a non-crash point can never be "peer evidence" for crash
history) and carries no train/validation/test split awareness — it is
computed once, at dataset-construction time, from the full reconciled
point table.

## 4. Cross-City Vocabulary Harmonization

Building-type and highway-type categorical vocabularies were built
independently per city, then unioned into one shared vocabulary before
any pooled experiment. Every already-constructed TVG graph's stored
categorical indices were remapped to the unified vocabulary (geometry
was not recomputed); every city's cache was spot-checked for agreement
before use.

## 5. Dataset Assembly

A point entered the final dataset only if both its SVG and TVG graphs
constructed and loaded successfully. All four cities' graphs were
written into one shared directory pair, indexed by one combined table
carrying `city` and `label` columns and keyed by each point's globally
unique `point_id`.

## 6. Model Architecture

Both SVG and TVG (and, for the unified-graph configuration, their
merge) are encoded by an independent heterogeneous graph attention
network, then collapsed to a fixed-size vector by SortPooling, projected
into a shared fusion space, and classified by a small MLP head.

```mermaid
flowchart TD
    RAW["Heterogeneous input graph<br/>(SVG, TVG, or merged Unified graph)"] --> L0["Per-node-type input projection h(0)<br/>(categorical fields: learned embeddings;<br/>missing continuous fields: learned placeholder)"]
    L0 --> L1["GATv2 layer 1 (HeteroConv)<br/>one GATv2Conv per edge relation, 4 heads<br/>LayerNorm -> ELU -> Dropout(0.3)"]
    L1 --> L2["GATv2 layer 2 (HeteroConv)<br/>LayerNorm -> ELU -> Dropout(0.3)"]
    L0 -.-> CAT["Cross-layer concatenation<br/>H = [h(0) || h(1) || h(2)],  dim = 3 x hidden_dim"]
    L1 -.-> CAT
    L2 -.-> CAT
    CAT --> SORT["SortPooling<br/>rank nodes by h(2)'s final channel,<br/>keep top-k, zero-pad if fewer than k"]
    SORT --> CONV1["Conv1d(1 -> 16, kernel = stride = 3 x hidden_dim)"]
    CONV1 --> POOL["MaxPool1d(2)"]
    POOL --> CONV2["Conv1d(16 -> 32, kernel = k2)"]
    CONV2 --> FLAT["Flatten -> Linear -> z<br/>dim(z) = 2 x hidden_dim"]
    FLAT --> FUSE["Fusion projection<br/>(scenario-dependent: single branch, concat, late-fusion,<br/>cross-attention, or unified graph)"]
    FUSE --> MLP["MLP classification head<br/>Linear -> ReLU -> Dropout(0.3) -> Linear"]
    MLP --> LOGIT["Raw logit"]
    LOGIT --> LOSS["Binary cross-entropy (logit-space)<br/>vs. ground-truth label"]
```
*Figure 2. Per-branch encoder-to-loss data flow for the proposed
SortPooling variant.*

### 6.1 Graph attention encoding

For each layer $\ell = 1, 2$:

$$h^{(0)}_v = \text{InputProj}_{\tau(v)}(x_v), \qquad
h^{(\ell)} = \text{Dropout}_{0.3}\Big(\text{ELU}\big(\text{LayerNorm}(\text{HeteroConv}^{(\ell)}(h^{(\ell-1)}))\big)\Big)$$

where $\tau(v)$ is node $v$'s type and each `HeteroConv` layer applies
one GATv2Conv (Brody et al., 2022) per edge relation (4 attention
heads, head dimension = hidden_dim/4, edge features included where
available), with per-relation outputs summed at shared destination node
types. Hidden dimension was 128 for every encoder.

### 6.2 SortPooling readout

Following Zhang et al. (2018), every layer's per-node output — including
the pre-message-passing projection $h^{(0)}$ — is concatenated per node:

$$H_v = \big[h^{(0)}_v \,\Vert\, h^{(1)}_v \,\Vert\, h^{(2)}_v\big], \qquad \dim(H_v) = 3\cdot\text{hidden\_dim} = 384$$

Nodes are ranked in descending order of their value in $h^{(2)}_v$'s
final channel, then truncated to the top $k$ (or zero-padded if the
graph has fewer than $k$ nodes), giving $R \in \mathbb{R}^{k \times
384}$. $R$ is flattened to one sequence and passed through:

$$\text{Conv1d}(1,\,16,\,\text{kernel}{=}\text{stride}{=}384) \to \text{MaxPool1d}(2) \to \text{Conv1d}(16,\,32,\,\text{kernel}{=}k_2) \to \text{Flatten} \to \text{Linear} \to z \in \mathbb{R}^{256}$$

**Selection of $k$.** Following Zhang et al.'s own rule, $k$ was set to
the 40th percentile of each encoder's per-graph total node-count
distribution (computed from a random 500-point sample of the pooled
dataset), so that 60% of graphs are truncated to their most salient $k$
nodes and the remaining 40% are retained in full. The post-SortPool
convolution stack imposes a hard architectural floor
($k \ge (k_2{-}1)\!\cdot\!2+2$), so the rule actually applied was:

$$k = \max\Big(\ (k_2-1)\cdot 2 + 2,\ \ \big\lceil P_{40}(\text{node counts})\big\rceil\ \Big)$$

The TVG encoder's total node count excludes the ablation-only
`peer_incident` type. Because TVG's graphs are small (mean total node
count ≈ 8, summing `incident` + retained `building`/`intersection`
nodes), its second convolution kernel was relaxed to $k_2{=}3$ (floor
$k\ge6$); SVG and the unified-graph encoder retained $k_2{=}5$ (floor
$k\ge10$). This is a data-driven, per-encoder decision, not a shared
constant.

### 6.3 Fusion and classification

Every branch's SortPooling output $z \in \mathbb{R}^{256}$ was projected
into a 256-dimensional fusion space, combined according to the
evaluated fusion scheme (Table 1), and classified by a 2-layer MLP:

$$\text{logit} = W_2\,\text{Dropout}_{0.3}\big(\text{ReLU}(W_1 z_{\text{fused}} + b_1)\big) + b_2, \qquad W_1 \in \mathbb{R}^{256\times256}$$

*Table 1. Fusion schemes evaluated.*

| Scheme | Mechanism |
|---|---|
| SVG only | Single branch |
| TVG only | Single branch |
| Concatenation | $[\text{proj}(z_{\text{svg}}) \Vert \text{proj}(z_{\text{tvg}})] \to$ head |
| Late fusion | Independent heads → logits $\ell_s,\ell_t$; learned combination $w_1\ell_s + w_2\ell_t$ |
| Cross-attention | $z_{\text{svg}}, z_{\text{tvg}}$ as two tokens, self-attend, concatenate → head |
| Unified graph | SVG and TVG merged into one heterogeneous graph pre-encoding, single encoder |

## 7. Training Procedure

Every fusion scheme was trained independently, end to end, for 5
independent repeats of a 70/15/15 train/validation/test split
(stratified by label only, drawn fresh each repeat; no city-level
stratification). Continuous node features were $z$-scored using
statistics fit on each repeat's training partition only, never
refit on validation/test data.

*Table 2. Training hyperparameters.*

| Parameter | Value |
|---|---|
| Optimizer | AdamW, learning rate $5\times10^{-3}$, weight decay $1\times10^{-4}$ |
| Loss | Binary cross-entropy on raw logits (numerically stable logit-space formulation) |
| LR schedule | Reduce-on-plateau (factor 0.5), monitoring the primary metric, patience 8 epochs, after a 20-epoch warm-up |
| Epoch budget | **150 epochs, strictly** — early stopping was structurally disabled (patience ≥ epoch cap) so training duration could not confound the comparison against the baseline |
| Model-selection / primary metric | **Accuracy**, evaluated on the held-out validation split each epoch |
| Decision threshold | Fixed at 0.5 |
| Batch size | 128 |
| Repeats | 5, independently re-split |

Checkpoint selection used only validation-split performance; test data
never influenced which epoch's weights were kept.

## 8. Baseline Comparison

The SortPooling variant was compared against an anchor-pooling baseline
sharing every hyperparameter in Table 2 and every architectural
component except the readout (§6.2): the baseline instead concatenates a
sum of per-node-type mean pools with the anchor node's (`ego`/`incident`)
own embedding, guaranteeing that node's representation always reaches
the classifier — a guarantee SortPooling does not provide, since the
anchor can be ranked outside the retained top-$k$ nodes. Holding every
other hyperparameter identical isolates the readout as the only
explanatory variable between the two conditions.

## 9. Model Interpretability

Two complementary post-hoc techniques were applied to trained
checkpoints, sampled across true-positive, true-negative,
false-positive, and false-negative predictions:

1. **Attention-weight extraction** — GATv2's learned per-neighbor
   attention coefficients were read directly from each trained layer,
   at no additional computational cost.
2. **A GNNExplainer-style learned mask** (Ying et al., 2019) — a
   per-point optimization that learns soft node- and edge-importance
   masks maximizing agreement between the masked and original
   predictions while encouraging sparsity, answering which specific
   nodes/edges drove a given prediction rather than only which
   neighbors received attention at each layer. This was used
   specifically to test whether the anchor node's importance
   empirically collapses under SortPooling relative to the
   anchor-pooling baseline.

## 10. Software and Computational Environment

Python 3.11, PyTorch and PyTorch Geometric (`GATv2Conv`, `HeteroConv`,
`SortAggregation`), scikit-learn, SciPy, GeoPandas/Shapely/OSMnx,
Hugging Face Transformers (Mask2Former). Training was performed on
Google Colab GPUs with bfloat16 mixed precision; graph construction was
CPU-only. A global random seed of 42 was used throughout, with a
per-repeat offset for every repeated procedure.

## References

- Brody, S., Alon, U., & Yahav, E. (2022). How Attentive are Graph
  Attention Networks? *International Conference on Learning
  Representations (ICLR)*.
- Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R.
  (2022). Masked-attention Mask Transformer for Universal Image
  Segmentation. *CVPR*.
- Neuhold, G., Ollmann, T., Rota Bulò, S., & Kontschieder, P. (2017).
  The Mapillary Vistas Dataset for Semantic Understanding of Street
  Scenes. *ICCV*.
- Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019).
  GNNExplainer: Generating Explanations for Graph Neural Networks.
  *NeurIPS*.
- Zhang, M., Cui, Z., Neumann, M., & Chen, Y. (2018). An End-to-End
  Deep Learning Architecture for Graph Classification. *AAAI*.

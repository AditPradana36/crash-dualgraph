# Materials and Methods

> Numbers marked **[REPORT FROM RUN LOGS]** are config targets, not
> measured outputs — the pipeline notebooks have no saved execution
> output in this repository, so final N, class balance, and result
> figures must be pulled from actual run logs / `dataset_index.parquet`
> / `metrics_*.csv`. All thresholds, formulas, and splits below are as
> implemented, not approximated.

## 1. Study Areas and Point Sampling

Two study areas: Bogor, Indonesia and Warsaw, Poland. Each candidate
point has a paired street-view image (a single ~90°-FOV "dashcam-style"
crop, not the full panorama) and an incident coordinate (used for all
geometric processing, distinct from the source panorama's own GPS fix).

Each city's positive (crash) and negative (non-crash) points were
reconciled via a three-way inner join (points table ∩ metadata table ∩
image present on disk); failing points were dropped. Negative points
were sampled to match the positive set's road/highway-type distribution
(not uniform-random), avoiding a road-type confound given highway type
is later used as a model feature.

**Target N** [REPORT FROM RUN LOGS]: single-city ≈460 positive / 460
negative; pooled two-city ≈1,100 / 1,100. Final N depends on
reconciliation yield and graph-construction success (Section 6).

## 2. Street-View Segmentation and Scene Features

Panoptic segmentation: Mask2Former
(`facebook/mask2former-swin-large-mapillary-vistas-panoptic`, Mapillary
Vistas v1.2, 65 classes), native resolution, confidence threshold 0.5.

Three scene-level features, computed per image:

| Feature | Formula | Domain |
|---|---|---|
| Sky View Factor | $\text{SVF} = A_{\text{sky}} / A_{\text{image}}$ | whole image |
| Enclosure Index | $\text{EI} = A_{\text{enclosure}} / A_{\text{crop}}$ | top 40% of image height only |
| Shannon entropy | $H = -\sum_c p_c \ln p_c$, $p_c$ = pixel share of class $c$ | whole image |

## 3. Street View Graph (SVG)

Egocentric graph, one per point, from the cached segmentation.

**Nodes (6 types):** `ego` (1; features = SVF, EI, $H$, fixed viewpoint
position) · `signage` (5 classes) · `light_pole` (4 classes) ·
`road_marking` (2 classes) · `building` (1 class) · `vegetation` (1
class) — object nodes carry normalized centroid + normalized area, split
from Mask2Former "thing" instances directly or from "stuff" segments via
connected-component labeling. Minimum object area: 0.08% of image area
(5% for building/vegetation).

**Edges (3 relations, bidirectional):**

| Relation | Trigger | Feature |
|---|---|---|
| `sees` | ego → every object | area, distance/diagonal |
| `mounted_with` | bbox overlap OR centroid dist ≤ 0.04·diagonal | bbox overlap ratio |
| `near` | centroid dist ≤ 0.175·diagonal | distance/diagonal |

## 4. Top View Graph (TVG)

Allocentric graph, one per point, from OSM buildings/streets + isovist
geometry (per-city building/street data fetched once, cached).

**Isovist:** 360 rays cast from the incident coordinate to a 50 m radius
against a building-boundary spatial index.

$$A_{\text{iso}} = \text{polygon area}, \quad
C_{\text{iso}} = \frac{4\pi A_{\text{iso}}}{P_{\text{iso}}^2}, \quad
O_{\text{iso}} = \frac{\#\{\text{rays stopped by a building}\}}{360}$$

**Building shape descriptors** (computed once per city):

$$C_b = \frac{4\pi A}{P^2} \ \text{(circular compactness)}, \quad
E = \frac{w_{\text{short}}}{l_{\text{long}}} \ \text{(elongation, min. rotated rect.)}, \quad
S = \frac{A}{A_{\text{rect}}} \ \text{(shape index)}$$

**Inclusion rule:** a building/intersection enters the graph iff it
intersects the isovist polygon, plus the two street nodes bounding the
nearest road edge (always included).

**Nodes (4 types):** `incident` (1; fraction-along-edge, $A_{\text{iso}}$,
$C_{\text{iso}}$, $O_{\text{iso}}$, + highway type) · `building` (area,
perimeter, $C_b$, $E$, orientation, $S$, + type, height/levels with
missing-flags) · `intersection` (betweenness, orientation entropy) ·
`peer_incident` (ablation-only, constant placeholder).

**Edges (6 relations):**

| Relation | Trigger | Feature |
|---|---|---|
| `anchors` | incident ↔ every included building/intersection | distance (m) |
| `adjacent` | full clique among included buildings | constant flag |
| `connects` | real OSM street edges among included intersections | highway type, maxspeed, oneway |
| `fronts` | building → nearest included intersection | distance (m) |
| `on_segment` | incident ↔ nearest edge's two endpoints | distance, highway type, maxspeed, oneway |
| `crash_history` (**ablation-only**) | incident → same-fold, positive-only peers within 100 m | distance (m) |

## 5. Cross-City Vocabulary Unification

Building-type and highway-type vocabularies were built independently per
city, then unioned into one shared vocabulary before any pooled
experiment; every already-built TVG graph's stored categorical indices
were remapped to the unified vocabulary (no geometry recomputation),
spot-checked, with pre-unification graphs kept as backup.

## 6. Dataset Assembly and Cross-Validation Folds

A point entered the final dataset only if both its SVG and TVG graphs
built and loaded successfully. Pooled (two-city) rows carry a `city`
tag and a collision-safe key `uid = {city}_{label}_{n}`.

**Spatial folds:** K-means on UTM-projected incident coordinates, $k=5$
clusters, 3 independent repeats (`fold_rep0..2`), positive/negative
points clustered jointly. This is **leave-one-cluster-out**, not random
k-fold — held-out folds are geographically distinct regions. Each city's
clustering is independent (different UTM zones); pooled fold $i$ = union
of city A's and city B's cluster-$i$ points.

## 7. Model Architecture

**Encoders:** SVG and TVG each encoded by a heterogeneous GNN: 2
`HeteroConv` layers (PyTorch Geometric), one `GATv2Conv` per edge
relation per layer (4 heads, head dim = hidden/4, edge features fed to
attention), LayerNorm + ELU + dropout per layer. Categorical fields
(object class, building type, highway type) pass through learned
embeddings, never raw integers; missing continuous fields (height,
levels, maxspeed) use a linear projection when present or one shared
learned placeholder vector when flagged missing — never imputed.
Readout: $[\text{mean-pool over all node types} \Vert \text{anchor node
embedding}]$ (anchor = `ego` for SVG, `incident` for TVG/unified).

**Fusion scenarios (A–G):** shared encoder design; head = linear or
2-layer MLP (both trained and compared).

| Scenario | Fusion mechanism |
|---|---|
| A | SVG only |
| B | TVG only |
| C | Concat: $[\text{proj}(z_{\text{svg}}) \Vert \text{proj}(z_{\text{tvg}})] \to$ head |
| D | Late fusion: separate heads → logits $\ell_s, \ell_t$; learned $w_1\ell_s + w_2\ell_t$ |
| E | Cross-attention: $z_{\text{svg}}, z_{\text{tvg}}$ as 2 tokens, self-attend, concat → head |
| F | Unified graph: SVG+TVG merged pre-encoding into one `HeteroConv` stack (`building` renamed `svg_building`/`tvg_building`; new `same_location` edge, ego↔incident) |
| G | Tabular baseline: flattened scene/isovist/count features → XGBoost (200 trees, depth 4); tests whether graph structure adds value over flat features |

Scenarios B–F each have an ablation variant (`+`) adding
`crash_history`/`peer_incident`, gated behind a flag given its
leakage risk.

## 8. Training Procedure

Every (scenario, head depth, ablation, split) trained independently,
end to end. Continuous features z-scored per fold/repeat, fit on the
training partition only. AdamW (lr $5\times10^{-3}$, weight decay
$1$–$3\times10^{-4}$), `BCEWithLogitsLoss`, `ReduceLROnPlateau` on val
PR-AUC after a warm-up window, bfloat16 mixed precision on GPU.
Checkpoint = highest **validation** PR-AUC epoch (test never influences
selection).

**Decision threshold** — fixed at 0.5, or adaptive (fit on val, frozen
for test):

$$t_{F1} = \arg\max_t F_1(t), \qquad
t_{Youden} = \arg\max_t \big(\text{TPR}(t) - \text{FPR}(t)\big), \qquad
t_{cost} = \arg\min_t \big(c_{FN}\cdot FN(t) + c_{FP}\cdot FP(t)\big)$$

with default cost ratio $c_{FN}\!:\!c_{FP} = 10\!:\!1$.

## 9. Evaluation Protocols

Six protocols, kept separate because their repeated scores are not the
same statistical object. What differs between them:

| # | Protocol | Split (train/val/test) | Repeats | Stratification | Threshold |
|---|---|---|---|---|---|
| 1 | Spatial k-fold (primary) | leave-1-cluster-out; 85/15 of remainder | 5 folds × 3 = 15 | spatial (K-means) | fixed |
| 2 | Bootstrap | 70/15/15, resample w/ replacement + dedup | 20 | label (post-dedup) | fixed |
| 3 | Random re-splits | 60/20/20 | 20 | label | adaptive (cost-sensitive default) |
| 4 | 70:30, no val | 70/–/30 (test doubles as val) | 10 | label | fixed |
| 5 | Fixed-composition CV, pooled | 65/15/20 | 5 | label × city (joint) | fixed |
| 6 | Cross-city generalization | train = 100% city A, test = 100% city B (both directions) | $N$ seeds | none (whole-city split) | scheme default |

Protocol 4's test-as-validation design is an intentional, explicit
train/test contract relaxation (test data influences model selection),
used only there. Protocol 6 answers cross-city generalization; protocol
5 answers whether pooling more, city-diverse data helps in-distribution.
Only protocol 1 feeds the significance tests below.

## 10. Evaluation Metrics and Statistical Analysis

Primary (co-equal): PR-AUC, AUROC. Descriptive: accuracy, F1, precision,
recall, BCE loss, confusion counts, threshold used.

**Significance testing** (protocol 1 only), per pre-specified scenario
pair, on matched fold-level score vectors:

$$\text{Wilcoxon signed-rank (primary)}, \qquad
t = \frac{\bar d}{\sqrt{\left(\tfrac{1}{k} + \tfrac{n_{test}}{n_{train}}\right) s_d^2}}
\ \text{(Nadeau–Bengio corrected t-test, secondary)}$$

where $\bar d$, $s_d^2$ are the mean and variance of the $k$ paired
fold-score differences; the correction term accounts for the
non-independence of repeated k-fold estimates. p-values corrected via
Holm–Bonferroni (step-down, per metric family), significance at
corrected $\alpha = 0.05$.

**Pairs (20):** A–B; C vs. {A,B,D,E,F}; D–E, D–F, E–F; G vs.
{A,B,C,D,E,F}; each ablation vs. its base scenario (B/C/D/E/F vs. B+/C+/D+/E+/F+).

**Interpretability:** attention-weight extraction; GNNExplainer.

## 11. Software

Python 3.11, PyTorch + PyTorch Geometric, Hugging Face Transformers
(Mask2Former), GeoPandas/Shapely/OSMnx, scikit-learn, SciPy, XGBoost.
Trained on Google Colab GPUs (bfloat16); graph construction is CPU-only.
Global seed 42, with a per-repeat offset for every repeated procedure.

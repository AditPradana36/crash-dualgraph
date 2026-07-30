# Design Reference — Frozen Checkpoint

This file should hold the full node/edge schema, locked parameters, and
design rationale as last summarized in the project's planning conversation.
Paste the latest comprehensive checkpoint here so it lives with the code,
not only in chat history.

Locked parameters (do not re-derive, just confirm against here):
- Isovist / candidate buffer radius: 50m
- Crash history distance threshold: 100m
- N_RAYS (isovist angular resolution): 360
- Enclosure Index: computed over top 40%-height image crop only
- SVF: computed over whole image (unchanged)

Still open (see configs/*.yaml for null markers):
- near cutoff D (SVG)
- mask confidence threshold
- mask minimum area
- bbox padding margin (OSM fetch)
- batch size / epoch cap / patience

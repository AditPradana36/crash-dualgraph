# crash-dualgraph

Dual-graph heterogeneous GNN for traffic accident prediction, integrating
egocentric street-view imagery (SVG) and allocentric top-view urban
structure (TVG).

## Structure
- `notebooks/` — numbered pipeline stages, `00` through `08`. Run in Colab.
- `src/` — all functions/defs. Notebooks import from here; never define
  core logic inline.
- `configs/` — YAML config. `paths.yaml` is machine-specific and gitignored;
  the rest (`svg_schema.yaml`, `tvg_schema.yaml`, `model.yaml`, `eval.yaml`)
  are version-controlled — editing a cutoff value and pushing it is how a
  research decision becomes permanent.
- `docs/design_reference.md` — frozen methodology/design checkpoint.

## Running in Colab
Every notebook's first cell clones this repo into `/content` and adds
`src/` to `sys.path`. Data lives on Google Drive, never in this repo.

## Data on Drive is not included here
See `configs/paths.yaml` (generated locally by `00_setup_config`, gitignored)
for the expected Drive folder structure.

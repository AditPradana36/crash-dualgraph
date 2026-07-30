"""
Per-item checkpoint/resume utility, used by 01, 02, and 04 wherever a
per-point loop needs to survive a Colab disconnect without redoing
already-completed work.

Functions to implement:
- load_manifest(path) -> pd.DataFrame          # (point_id, stage, status, timestamp)
- is_done(manifest, point_id, stage) -> bool
- mark_done(manifest, point_id, stage)
- pending_items(manifest, all_ids, stage) -> list
"""

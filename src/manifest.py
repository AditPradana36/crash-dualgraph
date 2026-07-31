"""
Per-item checkpoint/resume utility, used wherever a per-point loop needs
to survive a Colab disconnect without redoing already-completed work.

Design choice: the PRIMARY checkpoint signal is whether the actual output
file exists on disk — not a separate tracked status. This avoids the two
ever going out of sync with each other. The log is a secondary, append-only
record (CSV, not parquet — safe to append to one row at a time without
rewriting the whole file, which matters for a loop that might be
interrupted mid-run) kept for auditing status/errors, not for the resume
decision itself.
"""
import csv
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def output_path(out_dir, point_id, ext=".npz"):
    return Path(out_dir) / f"{point_id}{ext}"


def is_done(out_dir, point_id, ext=".npz"):
    return output_path(out_dir, point_id, ext).exists()


def pending_items(all_point_ids, out_dir, ext=".npz"):
    return [pid for pid in all_point_ids if not is_done(out_dir, pid, ext)]


def append_log(log_path, point_id, stage, status, error=None):
    log_path = Path(log_path)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["point_id", "stage", "status", "timestamp", "error"])
        writer.writerow([
            point_id, stage, status,
            datetime.now(timezone.utc).isoformat(),
            error or "",
        ])


def load_log(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        return pd.DataFrame(columns=["point_id", "stage", "status", "timestamp", "error"])
    return pd.read_csv(log_path)

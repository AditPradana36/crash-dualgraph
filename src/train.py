"""
Training loop + repeated spatial k-fold orchestration. Encoders trained
independently, end-to-end, per scenario, per fold — never shared/frozen
across scenarios or folds. Checkpointed per (scenario, head_depth,
ablation, fold, repeat) combination so a Colab disconnect resumes rather
than restarting.
"""
import copy
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def _to_jsonable(obj):
    """Recursively convert numpy/torch scalar types to native Python types.

    evaluate.py's metrics (and comparisons like `> best_val_prauc`) often
    come back as numpy.float64 / numpy.bool_ rather than plain float/bool.
    json.dumps doesn't know how to serialize those -- this walks dicts/
    lists and casts anything with a numpy-style .item() method."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar types (float64, bool_, int64, ...)
        return obj.item()
    return obj

import graph_datasets as ds
import models
import evaluate as ev


def run_forward(model, scenario, svg_batch, tvg_batch, svg_node_types, tvg_node_types):
    svg_bd = {nt: svg_batch[nt].batch for nt in svg_node_types} if svg_batch is not None else None
    tvg_bd = {nt: tvg_batch[nt].batch for nt in tvg_node_types} if tvg_batch is not None else None
    if scenario == "A":
        return model(svg_batch, svg_bd)
    if scenario == "B":
        return model(tvg_batch, tvg_bd)
    return model(svg_batch, tvg_batch, svg_bd, tvg_bd)


def _run_epoch_eval(model, scenario, loader, device, svg_nt, tvg_nt, is_cuda=False, use_amp=False):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for svg_batch, tvg_batch, labels, _ in loader:
            svg_batch = svg_batch.to(device, non_blocking=is_cuda)
            tvg_batch = tvg_batch.to(device, non_blocking=is_cuda)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = run_forward(model, scenario, svg_batch, tvg_batch, svg_nt, tvg_nt)
            y_prob.extend(torch.sigmoid(logits).float().cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_prob


def train_one_fold(scenario, head_depth, use_ablation, train_items, val_items, test_items,
                    svg_kwargs, tvg_kwargs, config, device, verbose=True, history_path=None):
    """*_items: list of (svg_data, tvg_data, label, point_id) for that split.

    verbose: print train_loss / val_pr_auc / val_auroc each epoch.
    history_path: if given, write the full per-epoch history to this JSON
    path after training (for later plotting -- see plot_fold_history)."""
    stats = ds.fit_normalization([i[0] for i in train_items], [i[1] for i in train_items])

    def _norm(items):
        return [(*ds.apply_normalization(copy.deepcopy(s), copy.deepcopy(t), stats), l, p)
                for s, t, l, p in items]

    train_items, val_items, test_items = _norm(train_items), _norm(val_items), _norm(test_items)

    # device may be passed in as a string ("cuda") or a torch.device object
    # depending on the caller -- normalize once so downstream checks work
    # either way instead of crashing on str.
    is_cuda = (device.type == "cuda") if isinstance(device, torch.device) else (str(device).startswith("cuda"))

    # num_workers>0 spawns subprocesses that re-import this module and
    # pickle every item in the DataLoader's dataset argument (here, the
    # normalized (svg, tvg, label, point_id) tuples). If graph_datasets'
    # HeteroData objects or collate_pairs don't pickle cleanly, or if the
    # environment's multiprocessing start method is unusual (common in
    # Colab), worker spawn can hang indefinitely with zero error output --
    # exactly the "0% GPU util forever" symptom. Default to 0 (safe,
    # single-process loading) and let config opt into >0 explicitly once
    # verified to work in this environment.
    num_workers = config.get("num_workers", 2)

    def _loader(items, shuffle):
        return DataLoader(
            [(s, t, l, p) for s, t, l, p in items], batch_size=config["batch_size"],
            shuffle=shuffle, collate_fn=ds.collate_pairs,
            num_workers=num_workers,
            pin_memory=is_cuda,
            persistent_workers=(num_workers > 0),
        )

    train_loader, val_loader, test_loader = _loader(train_items, True), _loader(val_items, False), _loader(test_items, False)

    model = models.build_model(scenario, fusion_dim=64, head_depth=head_depth, use_ablation=use_ablation,
                                svg_kwargs=svg_kwargs, tvg_kwargs=tvg_kwargs).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    criterion = torch.nn.BCEWithLogitsLoss()

    # bf16 autocast engages the A100's Tensor Cores for matmuls/convs during
    # the forward pass, which is the main thing that actually raises GPU
    # utilization/throughput on top of just "not being blocked by the
    # dataloader". bf16 (not fp16) is used because it has fp32's dynamic
    # range, so no GradScaler is needed -- fp16 would risk over/underflow
    # in the loss without one. use_amp is gated on CUDA since autocast on
    # CPU has no benefit here and just adds overhead.
    use_amp = config.get("use_amp", True) and is_cuda

    svg_nt = models.SVG_NODE_TYPES
    tvg_nt = models.TVG_NODE_TYPES if use_ablation else [nt for nt in models.TVG_NODE_TYPES if nt != "peer_incident"]

    best_val_prauc, best_state, no_improve = -1.0, None, 0
    best_train_loss, best_val_loss = None, None
    history = []

    for epoch in range(config["epoch_cap"]):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for svg_batch, tvg_batch, labels, _ in train_loader:
            # non_blocking only actually overlaps with compute when paired
            # with pin_memory=True on the loader (set above) -- otherwise
            # it's a no-op fallback to a normal blocking copy.
            svg_batch = svg_batch.to(device, non_blocking=is_cuda)
            tvg_batch = tvg_batch.to(device, non_blocking=is_cuda)
            labels = labels.to(device, non_blocking=is_cuda)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = run_forward(model, scenario, svg_batch, tvg_batch, svg_nt, tvg_nt)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        val_true, val_prob = _run_epoch_eval(model, scenario, val_loader, device, svg_nt, tvg_nt,
                                              is_cuda=is_cuda, use_amp=use_amp)
        val_metrics = ev.compute_metrics(val_true, val_prob)
        scheduler.step(val_metrics["pr_auc"])

        improved = val_metrics["pr_auc"] > best_val_prauc
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_pr_auc": val_metrics["pr_auc"], "val_auroc": val_metrics.get("auroc"),
                         "lr": optimizer.param_groups[0]["lr"], "improved": improved})

        if verbose:
            marker = " *" if improved else ""
            print(f"    epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_pr_auc {val_metrics['pr_auc']:.4f} | val_auroc {val_metrics.get('auroc', float('nan')):.4f} | "
                  f"no_improve {no_improve}/{config['patience']}{marker}")

        if improved:
            best_val_prauc, best_state, no_improve = val_metrics["pr_auc"], copy.deepcopy(model.state_dict()), 0
            best_train_loss, best_val_loss = train_loss, val_metrics.get("bce_loss")
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                if verbose:
                    print(f"    early stop at epoch {epoch} (best val_pr_auc={best_val_prauc:.4f})")
                break

    model.load_state_dict(best_state)
    test_true, test_prob = _run_epoch_eval(model, scenario, test_loader, device, svg_nt, tvg_nt,
                                            is_cuda=is_cuda, use_amp=use_amp)
    test_metrics = ev.compute_metrics(test_true, test_prob)
    # loss at the best (early-stopped) epoch, kept separate from
    # test_metrics["bce_loss"] which is BCE computed on the test split
    test_metrics["train_loss"] = best_train_loss
    test_metrics["val_loss"] = best_val_loss
    # val_pr_auc is what model/checkpoint selection should use (see
    # run_scenario) -- surfaced here rather than changing the return
    # signature, since it's the metric that picked best_state in the
    # first place.
    test_metrics["val_pr_auc"] = best_val_prauc

    if history_path is not None:
        Path(history_path).write_text(json.dumps(_to_jsonable(history), indent=1))

    return test_metrics, best_state


def run_scenario(scenario, head_depth, use_ablation, dataset, fold_cols, config, svg_kwargs, tvg_kwargs,
                  device, checkpoint_dir, verbose=True):
    """Repeated spatial k-fold: iterates every (fold_col, fold_id) combination,
    checkpointed so completed fold-runs are skipped on re-run."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{scenario}_{head_depth}{'_ablation' if use_ablation else ''}"
    results_path = checkpoint_dir / f"{tag}_results.json"
    history_dir = checkpoint_dir / f"{tag}_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done_keys = {(r["fold_col"], r["fold_id"]) for r in results}

    # Only the single best fold's weights are kept on disk per (scenario,
    # depth, ablation) tag -- "best" = highest VAL pr_auc (not test). Using
    # test here would mean test influenced which checkpoint gets kept as
    # "the" model, which biases any test number later reported for it.
    # Re-evaluated every time a fold finishes so the saved model always
    # reflects the best fold seen so far, even across resumed/interrupted
    # runs. test_metrics still gets appended to `results` for every fold
    # regardless, so the aggregated (mean +/- std) test performance across
    # all folds -- the number that should be reported -- is unaffected by
    # which single fold's weights happen to be saved here.
    best_model_path = checkpoint_dir / f"{tag}_best_model.pt"
    best_model_meta_path = checkpoint_dir / f"{tag}_best_model_meta.json"
    best_so_far = (json.loads(best_model_meta_path.read_text())
                   if best_model_meta_path.exists() else None)

    index_df = dataset.index_df
    for fold_col in fold_cols:
        for fold_id in sorted(index_df[fold_col].unique()):
            if (fold_col, fold_id) in done_keys:
                continue

            test_mask = index_df[fold_col] == fold_id
            train_val_df = index_df[~test_mask]
            val_frac = 0.15
            val_df = train_val_df.sample(frac=val_frac, random_state=42)
            train_df = train_val_df.drop(val_df.index)
            test_df = index_df[test_mask]

            def _items(df):
                return [dataset[i] for i in df.index]

            n_train = len(train_df)
            if verbose:
                print(f"  -- {fold_col} fold {fold_id} (n_train={n_train}, n_test={len(test_df)}) --")

            fold_history_path = history_dir / f"{fold_col}_fold{fold_id}.json"
            test_metrics, best_state = train_one_fold(
                scenario, head_depth, use_ablation, _items(train_df), _items(val_df), _items(test_df),
                svg_kwargs, tvg_kwargs, config, device,
                verbose=verbose, history_path=fold_history_path,
            )

            if best_so_far is None or test_metrics["val_pr_auc"] > best_so_far["val_pr_auc"]:
                torch.save(best_state, best_model_path)
                best_so_far = {"fold_col": fold_col, "fold_id": int(fold_id),
                                "val_pr_auc": test_metrics["val_pr_auc"],
                                "test_pr_auc": test_metrics["pr_auc"],
                                "test_auroc": test_metrics.get("auroc")}
                best_model_meta_path.write_text(json.dumps(_to_jsonable(best_so_far), indent=1))
                if verbose:
                    print(f"    -> new best model for {tag} "
                          f"(val pr_auc={test_metrics['val_pr_auc']:.4f}, "
                          f"test pr_auc={test_metrics['pr_auc']:.4f}), saved.")

            results.append({"fold_col": fold_col, "fold_id": int(fold_id), "n_train": n_train,
                             "n_test": len(test_df), **test_metrics})
            results_path.write_text(json.dumps(_to_jsonable(results), indent=1))

    return results

"""
Training loop, plus THREE separate orchestration schemes on top of it:

  run_scenario()              -- repeated spatial k-fold (the FORMAL
                                  pipeline, used by 07_train_eval.ipynb;
                                  feeds evaluate.py's pre-registered
                                  Wilcoxon/Nadeau-Bengio tests).
  run_scenario_bootstrap()    -- classic bootstrap resampling WITH
                                  replacement + dedup (used by
                                  07b_train_eval_bootstrap.ipynb).
  run_scenario_random_repeats() -- repeated random re-splits WITHOUT
                                  replacement, no dedup needed (every
                                  point used exactly once per repeat).

All three are separate, parallel schemes -- none replaces another.
run_scenario_bootstrap and run_scenario_random_repeats are NOT the same
thing despite both being "repeated": the former resamples with
replacement (duplicates possible, hence the dedup step and a smaller
effective n per repeat); the latter just re-shuffles and re-splits the
full dataset fresh each repeat (no duplicates, no dedup, full n retained
every repeat).

Encoders trained independently, end-to-end, per scenario, per fold/repeat
-- never shared/frozen across scenarios or folds/repeats. Checkpointed
per (scenario, head_depth, ablation, fold-or-repeat) combination so a
Colab disconnect resumes rather than restarting.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    path after training (for later plotting -- see plot_fold_history).

    Decision threshold: config["threshold_method"] controls how the
    classification threshold is chosen (default "fixed", i.e. the old
    hardcoded 0.5 behavior, preserved so existing 07/07b runs are
    unaffected unless they opt in). "f1" / "youden" / "cost_sensitive"
    learn a threshold from the VAL split's predictions at the best
    (early-stopped) epoch and freeze it for the test-split evaluation --
    see evaluate.find_optimal_threshold's docstring for what each method
    optimizes. config["threshold_fn_cost"] / config["threshold_fp_cost"]
    are only used by "cost_sensitive" (default 10:1, i.e. missing a
    crash-risk point is weighted like ten false alarms)."""
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
    num_workers = config.get("num_workers", 0)

    def _loader(items, shuffle):
        return DataLoader(
            [(s, t, l, p) for s, t, l, p in items], batch_size=config["batch_size"],
            shuffle=shuffle, collate_fn=ds.collate_pairs,
            num_workers=num_workers,
            pin_memory=is_cuda,
            persistent_workers=(num_workers > 0),
        )

    train_loader, val_loader, test_loader = _loader(train_items, True), _loader(val_items, False), _loader(test_items, False)

    model = models.build_model(scenario, fusion_dim=config.get("fusion_dim", 128), head_depth=head_depth,
                                use_ablation=use_ablation, svg_kwargs=svg_kwargs, tvg_kwargs=tvg_kwargs).to(device)
    optimizer = AdamW(model.parameters(), lr=config.get("lr", 5e-3), weight_decay=config.get("weight_decay", 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=config.get("lr_patience", 5), factor=0.5)
    criterion = torch.nn.BCEWithLogitsLoss()
    warmup_epochs = config.get("warmup_epochs", 0)

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
        if epoch >= warmup_epochs:
            scheduler.step(val_metrics["pr_auc"])

        improved = val_metrics["pr_auc"] > best_val_prauc
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_pr_auc": val_metrics["pr_auc"], "val_auroc": val_metrics.get("auroc"),
                         "lr": optimizer.param_groups[0]["lr"], "improved": improved})

        if improved:
            best_val_prauc = val_metrics["pr_auc"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            best_train_loss = train_loss
            best_val_loss = val_metrics.get("bce_loss")
        else:
            if epoch >= warmup_epochs:
                no_improve += 1

                if no_improve >= config["patience"]:
                    if verbose:
                        print(
                            f"    early stop at epoch {epoch} "
                            f"(best val_pr_auc={best_val_prauc:.4f})"
                        )
                    break

    model.load_state_dict(best_state)

    # Threshold: re-run the best (early-stopped) epoch's VAL predictions
    # once more here (cheap -- val is small, one forward pass) to fit the
    # threshold on the exact weights being evaluated. Frozen and reused
    # unchanged on test below -- test-set data never touches threshold
    # selection, only best_state weights + val predictions do.
    threshold_method = config.get("threshold_method", "fixed")
    if threshold_method == "fixed":
        chosen_threshold, threshold_method_used, threshold_score = config.get("threshold", 0.5), "fixed", float("nan")
    else:
        val_true_best, val_prob_best = _run_epoch_eval(model, scenario, val_loader, device, svg_nt, tvg_nt,
                                                         is_cuda=is_cuda, use_amp=use_amp)
        chosen_threshold, threshold_method_used, threshold_score = ev.find_optimal_threshold(
            val_true_best, val_prob_best, method=threshold_method,
            fn_cost=config.get("threshold_fn_cost", 10.0),
            fp_cost=config.get("threshold_fp_cost", 1.0),
        )
        if verbose:
            print(f"    threshold ({threshold_method_used}) = {chosen_threshold:.4f} "
                  f"(val score={threshold_score:.4f})")

    test_true, test_prob = _run_epoch_eval(model, scenario, test_loader, device, svg_nt, tvg_nt,
                                            is_cuda=is_cuda, use_amp=use_amp)
    test_metrics = ev.compute_metrics(test_true, test_prob, threshold=chosen_threshold)
    test_metrics["threshold_method"] = threshold_method_used
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
    checkpointed so completed fold-runs are skipped on re-run.

    This is the FORMAL pipeline's orchestration function, used by
    07_train_eval.ipynb. Its fold-level scores are what feed evaluate.py's
    Wilcoxon / Nadeau-Bengio significance tests, which are specifically
    designed for paired, correlated scores from repeated k-fold CV --
    don't swap this out for run_scenario_bootstrap's output without also
    reconsidering whether those tests still apply (see run_scenario_bootstrap
    docstring for why bootstrap-repeat scores aren't the same statistical
    object as k-fold scores).
    """
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


def run_scenario_bootstrap(scenario, head_depth, use_ablation, dataset, n_repeats, config, svg_kwargs, tvg_kwargs,
                            device, checkpoint_dir, verbose=True, id_col="point_id", seed=42):
    """Bootstrap-repeats orchestration -- a SEPARATE, parallel scheme from
    run_scenario's spatial k-fold above, used by 07b_train_eval_bootstrap.ipynb.
    Not a replacement for run_scenario / the formal k-fold pipeline: see
    that function's docstring for why the two produce statistically
    different objects (k-fold partition vs. resampled-with-replacement),
    which matters for evaluate.py's Wilcoxon/Nadeau-Bengio tests.

    Each repeat:
      1. Sample len(index_df) rows WITH REPLACEMENT from the full dataset
         (classic bootstrap resample).
      2. Dedup by id_col before splitting into train/val/test. Sampling
         with replacement means the same point can be drawn more than
         once within a repeat -- if a duplicate lands in both train and
         test, that's the same graph the model already saw during
         training showing up again in the "held-out" test set, which
         would silently inflate test performance. Dedup keeps train/val/
         test disjoint by point identity even though the resample itself
         has repeats.
      3. train/val/test split via the SAME percentages every repeat
         (config["val_frac"], config["test_frac"]), each repeat's split
         randomized independently via repeat_seed = seed + repeat_idx so
         results are reproducible but not identical across repeats.

    Checkpointed per repeat, same resume-on-reconnect behavior as the old
    fold-based version: repeats already present in results.json are
    skipped on re-run.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{scenario}_{head_depth}{'_ablation' if use_ablation else ''}"
    results_path = checkpoint_dir / f"{tag}_results.json"
    history_dir = checkpoint_dir / f"{tag}_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done_repeats = {r["repeat"] for r in results}

    # Same val-based (not test-based) selection rule as the fold version --
    # see run_scenario's docstring/comments for the reasoning. Only the
    # single best repeat's weights are kept on disk.
    best_model_path = checkpoint_dir / f"{tag}_best_model.pt"
    best_model_meta_path = checkpoint_dir / f"{tag}_best_model_meta.json"
    best_so_far = (json.loads(best_model_meta_path.read_text())
                   if best_model_meta_path.exists() else None)

    index_df = dataset.index_df
    n_total = len(index_df)
    val_frac = config.get("val_frac", 0.15)
    test_frac = config.get("test_frac", 0.15)

    for repeat in range(n_repeats):
        if repeat in done_repeats:
            continue

        repeat_seed = seed + repeat
        rng = np.random.default_rng(repeat_seed)

        # classic bootstrap: sample n_total rows WITH replacement
        boot_idx = rng.integers(0, n_total, size=n_total)
        boot_df = index_df.iloc[boot_idx]
        # dedup by id_col -- see docstring. keep='first' just needs to be
        # deterministic; which duplicate survives doesn't matter since
        # they're the same underlying point.
        boot_df = boot_df.drop_duplicates(subset=id_col, keep="first")

        # shuffle then split by percentage -- .sample(frac=1) with the
        # repeat-specific seed gives an independent random ordering per
        # repeat, distinct from the resampling RNG above.
        shuffled = boot_df.sample(frac=1.0, random_state=repeat_seed)
        n = len(shuffled)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))

        val_df = shuffled.iloc[:n_val]
        test_df = shuffled.iloc[n_val:n_val + n_test]
        train_df = shuffled.iloc[n_val + n_test:]

        def _items(df):
            return [dataset[i] for i in df.index]

        if verbose:
            print(f"  -- repeat {repeat} (bootstrap n={n} after dedup from {n_total}, "
                  f"n_train={len(train_df)}, n_val={len(val_df)}, n_test={len(test_df)}) --")

        repeat_history_path = history_dir / f"repeat{repeat}.json"
        test_metrics, best_state = train_one_fold(
            scenario, head_depth, use_ablation, _items(train_df), _items(val_df), _items(test_df),
            svg_kwargs, tvg_kwargs, config, device,
            verbose=verbose, history_path=repeat_history_path,
        )

        if best_so_far is None or test_metrics["val_pr_auc"] > best_so_far["val_pr_auc"]:
            torch.save(best_state, best_model_path)
            best_so_far = {"repeat": repeat,
                            "val_pr_auc": test_metrics["val_pr_auc"],
                            "test_pr_auc": test_metrics["pr_auc"],
                            "test_auroc": test_metrics.get("auroc")}
            best_model_meta_path.write_text(json.dumps(_to_jsonable(best_so_far), indent=1))
            if verbose:
                print(f"    -> new best model for {tag} "
                      f"(val pr_auc={test_metrics['val_pr_auc']:.4f}, "
                      f"test pr_auc={test_metrics['pr_auc']:.4f}), saved.")

        results.append({"repeat": repeat, "n_train": len(train_df), "n_val": len(val_df),
                         "n_test": len(test_df), **test_metrics})
        results_path.write_text(json.dumps(_to_jsonable(results), indent=1))

    return results


def _stratified_split(df, label_col, val_frac, test_frac, seed, target_pos_frac=None):
    """Split df into train/val/test, preserving (or gently nudging) the
    positive-class fraction in every split -- unlike a plain
    `.sample(frac=1)` shuffle, which can drift the positive rate between
    splits by chance on an imbalanced dataset (small positive counts make
    that drift proportionally large).

    Splits each class SEPARATELY by the same val_frac/test_frac
    percentages, then concatenates -- so every split's positive rate
    matches the class's split ratio by construction, not by luck.

    target_pos_frac: None (default) preserves the natural/global positive
    rate as-is in every split -- true stratification, no rebalancing.
    If given (e.g. 0.3), each split's positive class is gently upsampled
    WITH replacement within that split only (never crossing into another
    split, so no train/test leakage) until it reaches roughly that
    fraction -- "flexible but not too extreme" means this stays a knob
    you dial in later rather than forcing anything now; leave it None
    until you have a concrete target ratio in mind.
    """
    rng = np.random.default_rng(seed)
    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] == 0]

    def _split_one(class_df):
        shuffled = class_df.sample(frac=1.0, random_state=seed)
        n = len(shuffled)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        return shuffled.iloc[:n_val], shuffled.iloc[n_val:n_val + n_test], shuffled.iloc[n_val + n_test:]

    pos_val, pos_test, pos_train = _split_one(pos_df)
    neg_val, neg_test, neg_train = _split_one(neg_df)

    def _maybe_upsample(pos_part, neg_part):
        if target_pos_frac is None or len(pos_part) == 0:
            return pos_part
        # solve for target positive COUNT given the (fixed) negative count
        # in this split: pos_target / (pos_target + n_neg) = target_pos_frac
        n_neg = len(neg_part)
        target_n_pos = int(round((target_pos_frac * n_neg) / max(1 - target_pos_frac, 1e-6)))
        if target_n_pos <= len(pos_part):
            return pos_part  # already at/above target -- never downsample here
        extra_idx = rng.choice(pos_part.index, size=target_n_pos - len(pos_part), replace=True)
        return pd.concat([pos_part, pos_part.loc[extra_idx]])

    val_df = pd.concat([_maybe_upsample(pos_val, neg_val), neg_val]).sample(frac=1.0, random_state=seed)
    test_df = pd.concat([_maybe_upsample(pos_test, neg_test), neg_test]).sample(frac=1.0, random_state=seed + 1)
    train_df = pd.concat([_maybe_upsample(pos_train, neg_train), neg_train]).sample(frac=1.0, random_state=seed + 2)
    return train_df, val_df, test_df


def run_scenario_random_repeats(scenario, head_depth, use_ablation, dataset, n_repeats, config, svg_kwargs, tvg_kwargs,
                                 device, checkpoint_dir, verbose=True, seed=42):
    """Repeated RANDOM train/val/test re-splits -- a THIRD, separate scheme
    from both run_scenario (spatial k-fold) and run_scenario_bootstrap
    (bootstrap-with-replacement) above.

    NOT bootstrap: no sampling with replacement, no duplicates, no dedup
    needed. Every point appears in exactly train, val, OR test within a
    given repeat -- same disjointness guarantee as k-fold, just re-drawn
    with a fresh random split each repeat instead of iterating fixed folds.

    Each repeat:
      1. Split by percentage into train/val/test (config["val_frac"],
         config["test_frac"]) -- every point used exactly once, nothing
         dropped, nothing duplicated.
      2. STRATIFIED by config["label_col"] (default "label") when that
         column exists on dataset.index_df: positives and negatives are
         each split by the same percentages independently, so every
         split's positive rate matches the source class ratio instead of
         drifting by chance the way a plain shuffle-then-slice can on a
         small/imbalanced positive class. Falls back to the old plain
         shuffle if label_col isn't present. config["target_pos_frac"]
         (default None = preserve natural ratio) optionally nudges the
         positive fraction per split via in-split upsampling -- see
         _stratified_split's docstring.
      3. Train fresh, evaluate on that repeat's held-out test split.

    Across repeats, WHICH points land in train vs. val vs. test changes
    each time (since the shuffle seed changes), so this still gives you a
    distribution of scores to look at spread across repeats -- same
    motivation as bootstrap repeats (a single split's score could be a
    fluke of which points got shuffled where), just without resampling
    with replacement.

    Checkpointed per repeat, same resume-on-reconnect behavior as the
    other two schemes.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{scenario}_{head_depth}{'_ablation' if use_ablation else ''}"
    results_path = checkpoint_dir / f"{tag}_results.json"
    history_dir = checkpoint_dir / f"{tag}_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done_repeats = {r["repeat"] for r in results}

    best_model_path = checkpoint_dir / f"{tag}_best_model.pt"
    best_model_meta_path = checkpoint_dir / f"{tag}_best_model_meta.json"
    best_so_far = (json.loads(best_model_meta_path.read_text())
                   if best_model_meta_path.exists() else None)

    index_df = dataset.index_df
    val_frac = config.get("val_frac", 0.2)
    test_frac = config.get("test_frac", 0.2)
    label_col = config.get("label_col", "label")
    target_pos_frac = config.get("target_pos_frac", None)
    stratify = label_col in index_df.columns

    for repeat in range(n_repeats):
        if repeat in done_repeats:
            continue

        repeat_seed = seed + repeat

        if stratify:
            train_df, val_df, test_df = _stratified_split(
                index_df, label_col, val_frac, test_frac, repeat_seed, target_pos_frac)
        else:
            # fallback: plain shuffle, no resampling with replacement --
            # every point appears exactly once in this ordering
            shuffled = index_df.sample(frac=1.0, random_state=repeat_seed)
            n = len(shuffled)
            n_val = int(round(n * val_frac))
            n_test = int(round(n * test_frac))
            val_df = shuffled.iloc[:n_val]
            test_df = shuffled.iloc[n_val:n_val + n_test]
            train_df = shuffled.iloc[n_val + n_test:]

        n = len(train_df) + len(val_df) + len(test_df)

        def _items(df):
            return [dataset[i] for i in df.index]

        if verbose:
            split_kind = f"stratified by '{label_col}'" if stratify else "plain shuffle (no label_col found)"
            pos_rates = (f", pos_rate train/val/test="
                         f"{train_df[label_col].mean():.3f}/{val_df[label_col].mean():.3f}/{test_df[label_col].mean():.3f}"
                         if stratify else "")
            print(f"  -- repeat {repeat} (random re-split, {split_kind}, n={n}, "
                  f"n_train={len(train_df)}, n_val={len(val_df)}, n_test={len(test_df)}{pos_rates}) --")

        repeat_history_path = history_dir / f"repeat{repeat}.json"
        test_metrics, best_state = train_one_fold(
            scenario, head_depth, use_ablation, _items(train_df), _items(val_df), _items(test_df),
            svg_kwargs, tvg_kwargs, config, device,
            verbose=verbose, history_path=repeat_history_path,
        )

        if best_so_far is None or test_metrics["val_pr_auc"] > best_so_far["val_pr_auc"]:
            torch.save(best_state, best_model_path)
            best_so_far = {"repeat": repeat,
                            "val_pr_auc": test_metrics["val_pr_auc"],
                            "test_pr_auc": test_metrics["pr_auc"],
                            "test_auroc": test_metrics.get("auroc")}
            best_model_meta_path.write_text(json.dumps(_to_jsonable(best_so_far), indent=1))
            if verbose:
                print(f"    -> new best model for {tag} "
                      f"(val pr_auc={test_metrics['val_pr_auc']:.4f}, "
                      f"test pr_auc={test_metrics['pr_auc']:.4f}), saved.")

        results.append({"repeat": repeat, "n_train": len(train_df), "n_val": len(val_df),
                         "n_test": len(test_df), **test_metrics})
        results_path.write_text(json.dumps(_to_jsonable(results), indent=1))

    return results


def run_scenario_train_test_repeats(scenario, head_depth, use_ablation, dataset, n_repeats, config, svg_kwargs, tvg_kwargs,
                                     device, checkpoint_dir, verbose=True, seed=42):
    """Repeated 70:30 TRAIN/TEST split -- a FOURTH, separate scheme from
    run_scenario (spatial k-fold), run_scenario_bootstrap, and
    run_scenario_random_repeats above. Matches BOTH the split ratio AND the
    protocol reported in Lei et al. 2024 (Computers, Environment and Urban
    Systems): there is NO separate val split here -- only train (70%) and
    test (30%). Test plays double duty as val, exactly like the paper's
    own script (gnn_building_prediction.ipynb: `if test_rmse <
    best_test_rmse: ... best_model = model.state_dict()`).

    WHAT THIS MEANS, EXPLICITLY: the 30% test set is used to (a) drive
    early stopping, (b) select which epoch's weights become best_state,
    and (c) [if threshold_method != "fixed"] fit the decision threshold --
    then the SAME 30% is what test_metrics reports. That is a real
    train/test contract violation (test data influences model selection),
    not a stricter evaluation than the other three schemes -- it is a
    weaker one, deliberately chosen here to mirror the paper's reported
    numbers on a like-for-like protocol basis, not just a like-for-like
    split ratio. Do not treat this branch's numbers as directly comparable
    to 07/07b/07c's val-driven-selection test scores without accounting
    for this difference -- see the earlier discussion of this exact issue
    in the paper's own notebook.

    Threshold: config["threshold_method"] defaults to "fixed" (0.5) via
    train_one_fold's own default -- set explicitly in configs/eval_70_30.yaml
    so it's visible rather than implicit. PR-AUC is the primary metric
    (config-driven, same as the other schemes); compute_metrics also
    returns accuracy/auroc/f1/etc. every repeat, so nothing is lost.

    Same checkpoint/resume-on-reconnect contract as the other three
    schemes: repeats already present in results.json are skipped on re-run.
    Descriptive aggregates only (mean +/- std across repeats) -- like
    bootstrap/random-repeats, this doesn't feed evaluate.py's paired
    k-fold significance tests (see run_scenario's docstring for why).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{scenario}_{head_depth}{'_ablation' if use_ablation else ''}"
    results_path = checkpoint_dir / f"{tag}_results.json"
    history_dir = checkpoint_dir / f"{tag}_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done_repeats = {r["repeat"] for r in results}

    # "val_pr_auc" below is still the name train_one_fold returns -- but
    # since val IS test in this scheme, it's numerically identical to
    # test_metrics["pr_auc"] for the SAME epoch, not an independent check.
    best_model_path = checkpoint_dir / f"{tag}_best_model.pt"
    best_model_meta_path = checkpoint_dir / f"{tag}_best_model_meta.json"
    best_so_far = (json.loads(best_model_meta_path.read_text())
                   if best_model_meta_path.exists() else None)

    index_df = dataset.index_df
    test_frac = config.get("test_frac", 0.30)
    label_col = config.get("label_col", "label")
    target_pos_frac = config.get("target_pos_frac", None)
    stratify = label_col in index_df.columns

    for repeat in range(n_repeats):
        if repeat in done_repeats:
            continue

        repeat_seed = seed + repeat

        if stratify:
            # Plain 70/30 split -- val_frac=0.0 means _stratified_split's
            # "val" bucket comes back empty, so this is a two-way split.
            train_df, _, test_df = _stratified_split(
                index_df, label_col, val_frac=0.0, test_frac=test_frac,
                seed=repeat_seed, target_pos_frac=target_pos_frac)
        else:
            shuffled = index_df.sample(frac=1.0, random_state=repeat_seed)
            n = len(shuffled)
            n_test = int(round(n * test_frac))
            test_df = shuffled.iloc[:n_test]
            train_df = shuffled.iloc[n_test:]

        n = len(train_df) + len(test_df)

        def _items(df):
            return [dataset[i] for i in df.index]

        if verbose:
            split_kind = f"stratified by '{label_col}'" if stratify else "plain shuffle (no label_col found)"
            pos_rates = (f", pos_rate train/test="
                         f"{train_df[label_col].mean():.3f}/{test_df[label_col].mean():.3f}"
                         if stratify else "")
            print(f"  -- repeat {repeat} (70:30 train/test, NO separate val -- "
                  f"test doubles as val, {split_kind}, n={n}, "
                  f"n_train={len(train_df)}, n_test={len(test_df)}{pos_rates}) --")

        repeat_history_path = history_dir / f"repeat{repeat}.json"
        # test_items passed as BOTH val_items and test_items -- this is the
        # explicit, intentional leak described in the docstring above.
        test_items_list = _items(test_df)
        test_metrics, best_state = train_one_fold(
            scenario, head_depth, use_ablation, _items(train_df), test_items_list, test_items_list,
            svg_kwargs, tvg_kwargs, config, device,
            verbose=verbose, history_path=repeat_history_path,
        )

        if best_so_far is None or test_metrics["val_pr_auc"] > best_so_far["val_pr_auc"]:
            torch.save(best_state, best_model_path)
            best_so_far = {"repeat": repeat,
                            "val_pr_auc": test_metrics["val_pr_auc"],
                            "test_pr_auc": test_metrics["pr_auc"],
                            "test_accuracy": test_metrics.get("accuracy"),
                            "test_auroc": test_metrics.get("auroc")}
            best_model_meta_path.write_text(json.dumps(_to_jsonable(best_so_far), indent=1))
            if verbose:
                print(f"    -> new best model for {tag} "
                      f"(test/val pr_auc={test_metrics['val_pr_auc']:.4f}, "
                      f"test accuracy={test_metrics.get('accuracy', float('nan')):.4f}), saved.")

        results.append({"repeat": repeat, "n_train": len(train_df), "n_test": len(test_df),
                         **test_metrics})
        results_path.write_text(json.dumps(_to_jsonable(results), indent=1))

    return results

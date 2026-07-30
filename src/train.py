"""
Training loop + repeated spatial k-fold orchestration.

Functions to implement:
- train_one_fold(model, train_loader, val_loader, config) -> best_checkpoint, history
- run_scenario(scenario, dataset, folds, config) -> list[fold_results]
"""

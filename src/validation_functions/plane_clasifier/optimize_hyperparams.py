"""Simple Optuna hyperparameter optimization"""

import optuna
import numpy as np
import tensorflow as tf
from pathlib import Path

from train import train_plane_classifier
from config import TrainingConfig


def objective(trial, train_df, val_df, test_df):
    """Optuna objective function"""

    # Create config with suggested hyperparameters
    config = TrainingConfig()

    # Suggest key hyperparameters
    config.phase1_lr = trial.suggest_float("phase1_lr", 1e-4, 1e-2, log=True)
    config.phase2_lr = trial.suggest_float("phase2_lr", 1e-6, 1e-4, log=True)
    config.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    config.dropout_rate_1 = trial.suggest_float("dropout_rate_1", 0.2, 0.5)
    config.dropout_rate_2 = trial.suggest_float("dropout_rate_2", 0.1, 0.4)
    config.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    config.class_weight_mode = trial.suggest_categorical(
        "class_weight_mode", ["sqrt_balanced", "balanced", None]
    )

    # Reduce epochs for faster optimization
    config.phase1_epochs = 15
    config.phase2_epochs = 10
    config.bootstrap_enabled = False

    # Unique directories for this trial
    trial_dir = Path(f"./optuna_trials/trial_{trial.number}")
    config.checkpoint_dir = str(trial_dir / "checkpoints")
    config.log_dir = str(trial_dir / "logs")

    try:
        _, _, hist2, _ = train_plane_classifier(train_df, val_df, test_df, config)

        # Return best validation PR-AUC
        best_pr_auc = max(hist2["val_pr_auc"])

        # Clean up
        tf.keras.backend.clear_session()

        return best_pr_auc

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def optimize(train_df, val_df, test_df, n_trials=20):
    """Run hyperparameter optimization

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        n_trials: Number of trials (default: 20)

    Returns:
        Optuna study with results
    """
    print("=" * 70)
    print(f"RUNNING OPTUNA OPTIMIZATION ({n_trials} trials)")
    print("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        study_name="plane_classifier",
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        lambda trial: objective(trial, train_df, val_df, test_df),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best PR-AUC: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study


def get_best_config(study):
    """Get TrainingConfig with optimized hyperparameters"""
    config = TrainingConfig()

    # Apply best parameters
    for key, value in study.best_params.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Restore full training epochs
    config.phase1_epochs = 30
    config.phase2_epochs = 20
    config.bootstrap_enabled = True
    config.checkpoint_dir = "./checkpoints_optimized"
    config.log_dir = "./logs_optimized"

    return config

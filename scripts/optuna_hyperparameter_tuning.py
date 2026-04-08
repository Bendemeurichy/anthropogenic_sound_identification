"""
Lightweight Optuna hyperparameter tuning for COI separation models.

This script tunes the most important hyperparameters for each model (sudormrf, tuss, clapsep)
on the birds dataset and saves the best configurations for model comparison.

Usage:
    # Tune all models
    python scripts/optuna_hyperparameter_tuning.py --n-trials 20

    # Tune specific model
    python scripts/optuna_hyperparameter_tuning.py --model sudormrf --n-trials 50
    
    # Quick test
    python scripts/optuna_hyperparameter_tuning.py --n-trials 5 --max-epochs 10
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import optuna
import torch
import yaml

# Add parent directories to path for imports
_src_root = Path(__file__).parent.parent / "src"
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))


def create_sudormrf_trial_config(trial: optuna.Trial, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create SuDoRMRF config with trial hyperparameters."""
    config = base_config.copy()
    
    # Most important hyperparameters for SuDoRMRF
    config["model"]["out_channels"] = trial.suggest_categorical("out_channels", [128, 256, 512])
    config["model"]["in_channels"] = trial.suggest_categorical("in_channels", [256, 512, 768])
    config["model"]["num_blocks"] = trial.suggest_int("num_blocks", 8, 20)
    config["model"]["enc_num_basis"] = trial.suggest_categorical("enc_num_basis", [512, 1024, 2048])
    config["model"]["num_head_conv_blocks"] = trial.suggest_int("num_head_conv_blocks", 1, 3)
    
    config["training"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    config["training"]["class_weight"] = trial.suggest_float("class_weight", 1.0, 3.0)
    config["training"]["warmup_steps"] = trial.suggest_int("warmup_steps", 100, 500)
    
    return config


def create_tuss_trial_config(trial: optuna.Trial, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create TUSS config with trial hyperparameters."""
    config = base_config.copy()
    
    # Most important hyperparameters for TUSS
    config["training"]["lr"] = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    config["training"]["coi_weight"] = trial.suggest_float("coi_weight", 1.0, 3.0)
    config["training"]["zero_ref_loss_weight"] = trial.suggest_float("zero_ref_loss_weight", 0.01, 0.5, log=True)
    config["training"]["warmup_steps"] = trial.suggest_int("warmup_steps", 100, 500)
    config["training"]["grad_accum_steps"] = trial.suggest_categorical("grad_accum_steps", [4, 8, 16])
    
    # Freeze backbone option
    config["model"]["freeze_backbone"] = trial.suggest_categorical("freeze_backbone", [True, False])
    
    return config


def create_clapsep_trial_config(trial: optuna.Trial, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create CLAPSep config with trial hyperparameters."""
    config = base_config.copy()
    
    # Most important hyperparameters for CLAPSep
    config["model"]["embed_dim"] = trial.suggest_categorical("embed_dim", [64, 128, 256])
    config["model"]["encoder_embed_dim"] = trial.suggest_categorical("encoder_embed_dim", [64, 128, 256])
    config["model"]["n_masker_layer"] = trial.suggest_int("n_masker_layer", 2, 5)
    config["model"]["d_attn"] = trial.suggest_categorical("d_attn", [320, 640, 1024])
    
    config["training"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    config["training"]["class_weight"] = trial.suggest_float("class_weight", 1.0, 3.0)
    
    # Freeze encoder option
    config["model"]["freeze_encoder"] = trial.suggest_categorical("freeze_encoder", [True, False])
    
    return config


def train_and_evaluate_sudormrf(config: Dict[str, Any], trial_number: int, max_epochs: int = None) -> float:
    """Train SuDoRMRF model and return validation metric."""
    import subprocess
    import tempfile
    
    # Override epochs if specified
    if max_epochs is not None:
        config["training"]["num_epochs"] = max_epochs
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # Run training script
        result = subprocess.run(
            [
                sys.executable,
                "src/models/sudormrf/train.py",
                "--config", config_path,
                "--trial-number", str(trial_number),
            ],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        
        # Parse best validation metric from output
        # Look for lines like "Best Val SI-SNR: 12.34 dB"
        for line in result.stdout.split('\n'):
            if "Best Val SI-SNR" in line or "best_val_sisnr" in line:
                try:
                    # Extract the metric value
                    metric = float(line.split(':')[-1].replace('dB', '').strip())
                    return metric
                except:
                    pass
        
        # If we can't find the metric, return a poor value
        print(f"Warning: Could not parse validation metric from training output")
        return -100.0
        
    except subprocess.TimeoutExpired:
        print(f"Trial {trial_number} timed out")
        return -100.0
    except Exception as e:
        print(f"Error running trial {trial_number}: {e}")
        return -100.0
    finally:
        # Clean up temp file
        Path(config_path).unlink(missing_ok=True)


def train_and_evaluate_tuss(config: Dict[str, Any], trial_number: int, max_epochs: int = None) -> float:
    """Train TUSS model and return validation metric."""
    import subprocess
    import tempfile
    
    if max_epochs is not None:
        config["training"]["num_epochs"] = max_epochs
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        result = subprocess.run(
            [
                sys.executable,
                "src/models/tuss/train.py",
                "--config", config_path,
                "--trial-number", str(trial_number),
            ],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        
        for line in result.stdout.split('\n'):
            if "Best Val SNR" in line or "best_val_snr" in line:
                try:
                    metric = float(line.split(':')[-1].replace('dB', '').strip())
                    return metric
                except:
                    pass
        
        return -100.0
        
    except subprocess.TimeoutExpired:
        print(f"Trial {trial_number} timed out")
        return -100.0
    except Exception as e:
        print(f"Error running trial {trial_number}: {e}")
        return -100.0
    finally:
        Path(config_path).unlink(missing_ok=True)


def train_and_evaluate_clapsep(config: Dict[str, Any], trial_number: int, max_epochs: int = None) -> float:
    """Train CLAPSep model and return validation metric."""
    import subprocess
    import tempfile
    
    if max_epochs is not None:
        config["training"]["num_epochs"] = max_epochs
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        result = subprocess.run(
            [
                sys.executable,
                "src/models/clapsep/train_coi.py",
                "--config", config_path,
                "--trial-number", str(trial_number),
            ],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        
        for line in result.stdout.split('\n'):
            if "Best Val SI-SNR" in line or "best_val_sisnr" in line:
                try:
                    metric = float(line.split(':')[-1].replace('dB', '').strip())
                    return metric
                except:
                    pass
        
        return -100.0
        
    except subprocess.TimeoutExpired:
        print(f"Trial {trial_number} timed out")
        return -100.0
    except Exception as e:
        print(f"Error running trial {trial_number}: {e}")
        return -100.0
    finally:
        Path(config_path).unlink(missing_ok=True)


def objective_sudormrf(trial: optuna.Trial, base_config: Dict[str, Any], max_epochs: int = None) -> float:
    """Optuna objective function for SuDoRMRF."""
    config = create_sudormrf_trial_config(trial, base_config)
    metric = train_and_evaluate_sudormrf(config, trial.number, max_epochs)
    return metric


def objective_tuss(trial: optuna.Trial, base_config: Dict[str, Any], max_epochs: int = None) -> float:
    """Optuna objective function for TUSS."""
    config = create_tuss_trial_config(trial, base_config)
    metric = train_and_evaluate_tuss(config, trial.number, max_epochs)
    return metric


def objective_clapsep(trial: optuna.Trial, base_config: Dict[str, Any], max_epochs: int = None) -> float:
    """Optuna objective function for CLAPSep."""
    config = create_clapsep_trial_config(trial, base_config)
    metric = train_and_evaluate_clapsep(config, trial.number, max_epochs)
    return metric


def tune_model(
    model_name: str,
    n_trials: int = 20,
    max_epochs: int = None,
    storage: str = None,
    study_name: str = None,
) -> Dict[str, Any]:
    """
    Tune hyperparameters for a specific model.
    
    Args:
        model_name: One of 'sudormrf', 'tuss', 'clapsep'
        n_trials: Number of Optuna trials to run
        max_epochs: Maximum epochs per trial (for quick testing)
        storage: Optuna storage URL (e.g., sqlite:///optuna.db)
        study_name: Name for the Optuna study
        
    Returns:
        Best hyperparameters dictionary
    """
    # Load base config
    config_path = Path(f"src/models/{model_name}/training_config.yaml")
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    
    # Update to use birds dataset
    base_config["data"]["df_path"] = "data/birds_data.csv"
    base_config["data"]["target_classes"] = [["bird", "Bird"]]
    
    # Create study
    if study_name is None:
        study_name = f"{model_name}_birds_tuning"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # Maximize SI-SNR or SNR
        load_if_exists=True,
    )
    
    # Select objective function
    if model_name == "sudormrf":
        objective = lambda trial: objective_sudormrf(trial, base_config, max_epochs)
    elif model_name == "tuss":
        objective = lambda trial: objective_tuss(trial, base_config, max_epochs)
    elif model_name == "clapsep":
        objective = lambda trial: objective_clapsep(trial, base_config, max_epochs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Run optimization
    print(f"\n{'='*80}")
    print(f"Starting Optuna hyperparameter tuning for {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Number of trials: {n_trials}")
    print(f"Max epochs per trial: {max_epochs if max_epochs else 'from config'}")
    print(f"Study name: {study_name}")
    print(f"{'='*80}\n")
    
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Optimization completed for {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (SI-SNR/SNR): {study.best_trial.value:.2f} dB")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Save best config
    best_config = base_config.copy()
    if model_name == "sudormrf":
        best_config = create_sudormrf_trial_config(study.best_trial, base_config)
    elif model_name == "tuss":
        best_config = create_tuss_trial_config(study.best_trial, base_config)
    elif model_name == "clapsep":
        best_config = create_clapsep_trial_config(study.best_trial, base_config)
    
    # Save to file
    output_dir = Path("configs/tuned")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_birds_best.yaml"
    
    with open(output_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"Best config saved to: {output_path}")
    
    # Also save study statistics
    stats_path = output_dir / f"{model_name}_birds_tuning_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Optuna Study Statistics for {model_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Study name: {study_name}\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_trial.value:.2f} dB\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("All trials:\n")
        for trial in study.trials:
            f.write(f"Trial {trial.number}: {trial.value:.2f} dB\n")
    
    print(f"Study statistics saved to: {stats_path}")
    
    return study.best_trial.params


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for COI separation models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sudormrf", "tuss", "clapsep", "all"],
        default="all",
        help="Which model to tune (default: all)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials per model (default: 20)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum epochs per trial (for quick testing)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_coi_tuning.db",
        help="Optuna storage URL (default: sqlite:///optuna_coi_tuning.db)",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Run data preparation script before tuning",
    )
    
    args = parser.parse_args()
    
    # Prepare data if requested
    if args.prepare_data:
        print("Preparing birds dataset...")
        import subprocess
        subprocess.run([
            sys.executable,
            "scripts/prepare_birds_dataset.py",
        ])
        print("Data preparation complete.\n")
    
    # Check if birds data exists
    birds_data_path = Path("data/birds_data.csv")
    if not birds_data_path.exists():
        print(f"Error: Birds dataset not found at {birds_data_path}")
        print("Run with --prepare-data flag to create it first.")
        return
    
    # Tune models
    models_to_tune = ["sudormrf", "tuss", "clapsep"] if args.model == "all" else [args.model]
    
    results = {}
    for model_name in models_to_tune:
        best_params = tune_model(
            model_name=model_name,
            n_trials=args.n_trials,
            max_epochs=args.max_epochs,
            storage=args.storage,
        )
        results[model_name] = best_params
    
    # Print summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 80)
    for model_name, params in results.items():
        print(f"\n{model_name.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    print("\n" + "=" * 80)
    print("\nTuned configs saved to: configs/tuned/")
    print("You can now train the models with these optimized hyperparameters!")
    print("=" * 80)


if __name__ == "__main__":
    main()

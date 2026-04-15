"""Training pipeline and utilities for PANN plane classifier"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from tqdm import tqdm
import json
import time

from .model import PlaneClassifierPANN
from .model_loader import save_checkpoint
from .config import TrainingConfig


class BootstrapPRAUCCallback:
    """Custom callback to compute bootstrap confidence intervals for validation PR-AUC"""
    
    def __init__(
        self,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        log_frequency: int = 1,
    ):
        super().__init__()
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.log_frequency = log_frequency
        self.bootstrap_results = []
    
    def compute(self, model: PlaneClassifierPANN, val_loader: DataLoader, epoch: int, device: str):
        """Compute bootstrap CI for current epoch"""
        # Only compute every N epochs
        if (epoch + 1) % self.log_frequency != 0:
            return None
        
        # Collect all validation predictions and labels
        model.eval()
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                y_true_list.append(batch_y.numpy())
                y_pred_list.append(probs.flatten())
        
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        
        # Compute bootstrap confidence interval
        n_samples = len(y_true)
        bootstrap_scores = []
        
        np.random.seed(42 + epoch)  # Deterministic but different per epoch
        
        for _ in range(self.n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Compute PR-AUC for this bootstrap sample
            try:
                precision, recall, _ = precision_recall_curve(y_true_boot, y_pred_boot)
                pr_auc = auc(recall, precision)
                bootstrap_scores.append(pr_auc)
            except Exception:
                continue
        
        if bootstrap_scores:
            bootstrap_scores = np.array(bootstrap_scores)
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_scores, lower_percentile)
            ci_upper = np.percentile(bootstrap_scores, upper_percentile)
            mean_score = np.mean(bootstrap_scores)
            
            # Store results
            result = {
                "epoch": epoch + 1,
                "mean": float(mean_score),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            }
            self.bootstrap_results.append(result)
            
            # Log to console
            print(
                f"\n  Bootstrap PR-AUC (n={self.n_iterations}): "
                f"{mean_score:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] "
                f"(CI width: {ci_upper - ci_lower:.4f})"
            )
            
            return result
        
        return None


def compute_metrics(
    model: PlaneClassifierPANN,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """
    Compute comprehensive metrics on a dataset.
    
    Args:
        model: PlaneClassifierPANN model
        data_loader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_logits = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device).unsqueeze(1)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item() * len(batch_x)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.append(batch_y.cpu().numpy())
            all_probs.append(probs)
            all_logits.append(logits.cpu().numpy())
    
    all_labels = np.concatenate(all_labels).flatten()
    all_probs = np.concatenate(all_probs).flatten()
    all_logits = np.concatenate(all_logits).flatten()
    
    avg_loss = total_loss / len(all_labels)
    
    # Compute PR-AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    # Compute ROC-AUC
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    # Metrics at threshold=0.5
    preds_05 = (all_probs >= 0.5).astype(int)
    acc_05 = (preds_05 == all_labels).mean()
    prec_05 = precision_score(all_labels, preds_05, zero_division=0)
    rec_05 = recall_score(all_labels, preds_05, zero_division=0)
    f1_05 = f1_score(all_labels, preds_05, zero_division=0)
    
    return {
        'loss': avg_loss,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': acc_05,
        'precision': prec_05,
        'recall': rec_05,
        'f1_score': f1_05,
    }


def compute_optimal_threshold(
    model: PlaneClassifierPANN,
    val_loader: DataLoader,
    device: str,
    metric: str = 'f1'
) -> Tuple[float, Dict[str, float]]:
    """
    Compute optimal threshold on validation set based on PR curve.
    
    Args:
        model: Trained model
        val_loader: Validation DataLoader
        device: Device to run on
        metric: 'f1' for max F1-score
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            y_true_list.append(batch_y.numpy())
            y_pred_list.append(probs.flatten())
    
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    if metric == 'f1':
        # Compute F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        )
        
        metrics = {
            'threshold': float(optimal_threshold),
            'precision': float(precision[best_idx]),
            'recall': float(recall[best_idx]),
            'f1_score': float(f1_scores[best_idx]),
        }
    else:
        # Default to max F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        )
        metrics = {
            'threshold': float(optimal_threshold),
            'precision': float(precision[best_idx]),
            'recall': float(recall[best_idx]),
            'f1_score': float(f1_scores[best_idx]),
        }
    
    return optimal_threshold, metrics


def train_epoch(
    model: PlaneClassifierPANN,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    gradient_clip_norm: float = 1.0
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device).unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        
        optimizer.step()
        
        total_loss += loss.item() * len(batch_x)
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


def train_phase(
    model: PlaneClassifierPANN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    phase_name: str,
    device: str,
    checkpoint_dir: Path,
    bootstrap_callback: Optional[BootstrapPRAUCCallback] = None
) -> Dict[str, Any]:
    """
    Train a single phase (phase1 or phase2).
    
    Args:
        model: PlaneClassifierPANN model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: TrainingConfig
        phase_name: 'phase1' or 'phase2'
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        bootstrap_callback: Optional bootstrap CI callback
        
    Returns:
        Training history dictionary
    """
    is_phase1 = phase_name == 'phase1'
    epochs = config.phase1_epochs if is_phase1 else config.phase2_epochs
    lr = config.phase1_lr if is_phase1 else config.phase2_lr
    patience = config.phase1_patience if is_phase1 else config.phase2_patience
    reduce_lr_patience = (
        config.phase1_reduce_lr_patience if is_phase1
        else config.phase2_reduce_lr_patience
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.weight_decay,
        betas=(config.beta_1, config.beta_2)
    )
    
    # Setup loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=reduce_lr_patience,
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_pr_auc': [],
        'val_roc_auc': [],
        'val_accuracy': [],
        'val_f1': [],
        'lr': [],
    }
    
    best_pr_auc = 0.0
    epochs_no_improve = 0
    
    print(f"\n{'='*70}")
    print(f"{phase_name.upper()}: {'TRAINING CLASSIFIER HEAD (CNN14 frozen)' if is_phase1 else 'FINE-TUNING ENTIRE MODEL'}")
    print(f"{'='*70}")
    print(f"Epochs: {epochs}, LR: {lr}, Patience: {patience}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            config.gradient_clip_norm
        )
        
        # Validate
        val_metrics = compute_metrics(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_pr_auc'].append(val_metrics['pr_auc'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_metrics['loss']:.4f}")
        print(f"Val PR-AUC: {val_metrics['pr_auc']:.4f}")
        print(f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1_score']:.4f}")
        
        # Bootstrap CI
        if bootstrap_callback is not None:
            bootstrap_callback.compute(model, val_loader, epoch, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['pr_auc'])
        
        # Save best model
        if val_metrics['pr_auc'] > best_pr_auc:
            best_pr_auc = val_metrics['pr_auc']
            epochs_no_improve = 0
            
            save_path = checkpoint_dir / f"best_model_{phase_name}.pth"
            save_checkpoint(model, optimizer, epoch, save_path, val_metrics)
            print(f"✓ New best model saved (PR-AUC: {best_pr_auc:.4f})")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation PR-AUC: {best_pr_auc:.4f}")
            break
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch time: {epoch_time:.1f}s")
    
    print(f"\n{phase_name.upper()} completed!")
    print(f"Best validation PR-AUC: {best_pr_auc:.4f}")
    
    # Load best model
    best_checkpoint = checkpoint_dir / f"best_model_{phase_name}.pth"
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from {best_checkpoint}")
    
    return history


def train_plane_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model: PlaneClassifierPANN,
    config: TrainingConfig,
    device: str
) -> Tuple[PlaneClassifierPANN, Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    """
    Complete two-phase training pipeline.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        model: PlaneClassifierPANN model
        config: TrainingConfig
        device: Device to train on
        
    Returns:
        Tuple of (trained_model, phase1_history, phase2_history, test_results)
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Bootstrap callback
    bootstrap_cb = BootstrapPRAUCCallback(
        n_iterations=config.bootstrap_n_iterations,
        confidence_level=config.bootstrap_confidence_level,
        log_frequency=config.bootstrap_log_frequency
    ) if config.bootstrap_enabled else None
    
    # PHASE 1: Train classifier head with frozen CNN14
    model.set_fine_tune(False)
    history_phase1 = train_phase(
        model, train_loader, val_loader, config,
        'phase1', device, checkpoint_dir, bootstrap_cb
    )
    
    # Save phase1 history
    with open(checkpoint_dir / 'history_phase1.json', 'w') as f:
        json.dump(history_phase1, f, indent=2)
    
    # PHASE 2: Fine-tune entire model
    model.set_fine_tune(True)
    history_phase2 = train_phase(
        model, train_loader, val_loader, config,
        'phase2', device, checkpoint_dir, bootstrap_cb
    )
    
    # Save phase2 history
    with open(checkpoint_dir / 'history_phase2.json', 'w') as f:
        json.dump(history_phase2, f, indent=2)
    
    # Compute optimal threshold
    print(f"\n{'='*70}")
    print("COMPUTING OPTIMAL THRESHOLD ON VALIDATION SET")
    print(f"{'='*70}")
    
    optimal_threshold, threshold_metrics = compute_optimal_threshold(
        model, val_loader, device, metric='f1'
    )
    
    print(f"\nOptimal threshold (max F1): {optimal_threshold:.4f}")
    print(f"  Precision: {threshold_metrics['precision']:.4f}")
    print(f"  Recall:    {threshold_metrics['recall']:.4f}")
    print(f"  F1 Score:  {threshold_metrics['f1_score']:.4f}")
    
    # Save optimal threshold
    with open(checkpoint_dir / 'optimal_threshold.json', 'w') as f:
        json.dump(threshold_metrics, f, indent=2)
    
    # Final evaluation on test set
    print(f"\n{'='*70}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*70}")
    
    criterion = nn.BCEWithLogitsLoss()
    test_results = compute_metrics(model, test_loader, criterion, device)
    
    print("\nTest Results:")
    for metric_name, value in test_results.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pth'
    save_checkpoint(model, None, -1, final_path, test_results)
    print(f"\nFinal model saved to {final_path}")
    
    return model, history_phase1, history_phase2, test_results

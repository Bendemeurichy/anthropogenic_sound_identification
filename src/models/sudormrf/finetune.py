"""
Finetuning script for pretrained SuDoRM-RF models with a custom COI separation head.

Loads a pretrained checkpoint, replaces the mask network with a COI-specific
separation head, and finetunes with configurable freeze strategies:
    - "head_only":  Only the new COI separation head is trained
    - "partial":    Head + last N separation module blocks (+ optionally bottleneck)
    - "full":       All parameters with differential learning rates

Supports staged unfreezing: head_only -> partial -> full across epochs.

Usage:
    python finetune.py --config finetune_config.yaml
"""

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Add src/ to path for label_loading imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Add src/models/sudormrf/ to path for local imports (base, config, train, etc.)
sys.path.insert(0, str(Path(__file__).parent))
# Add base/ to path so pickled checkpoints can find sudo_rm_rf module
sys.path.insert(0, str(Path(__file__).parent / "base"))

from base.sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 import GroupCommSudoRmRf
from base.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF
from config import Config, DataConfig, ModelConfig, TrainingConfig
from seperation_head import wrap_model_for_coi
from train import (
    COIWeightedLoss,
    create_dataloader,
    set_seed,
    train_epoch,
    validate_epoch,
)

from label_loading.metadata_loader import (
    load_metadata_datasets,
    split_seperation_classification,
)
from label_loading.sampler import get_coi, sample_non_coi

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PretrainedConfig:
    checkpoint_path: str = ""
    model_type: str = "improved"
    out_channels: int = 256
    in_channels: int = 512
    num_blocks: int = 16
    upsampling_depth: int = 5
    enc_kernel_size: int = 21
    enc_num_basis: int = 512
    num_sources: int = 2


@dataclass
class FinetuningStrategyConfig:
    freeze_strategy: str = "head_only"
    unfreeze_last_n_blocks: int = 4
    unfreeze_bottleneck: bool = True
    unfreeze_encoder: bool = False
    unfreeze_decoder: bool = False
    staged_unfreeze: bool = False
    stage1_epochs: int = 10
    stage2_epochs: int = 25
    backbone_lr_multiplier: float = 0.1
    reinit_decoder_bias: bool = False


@dataclass
class HeadConfig:
    num_conv_blocks: int = 0
    upsampling_depth: Optional[int] = None
    expanded_channels: Optional[int] = None


@dataclass
class SchedulerConfig:
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-7


@dataclass
class FinetuneConfig:
    pretrained: PretrainedConfig = field(default_factory=PretrainedConfig)
    finetuning: FinetuningStrategyConfig = field(
        default_factory=FinetuningStrategyConfig
    )
    head: HeadConfig = field(default_factory=HeadConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        pt = PretrainedConfig(**cfg_dict.get("pretrained", {}))
        ft = FinetuningStrategyConfig(**cfg_dict.get("finetuning", {}))
        head = HeadConfig(**cfg_dict.get("head", {}))
        data = DataConfig(**cfg_dict.get("data", {}))
        training = TrainingConfig(**cfg_dict.get("training", {}))
        sched = SchedulerConfig(**cfg_dict.get("scheduler", {}))
        return cls(
            pretrained=pt,
            finetuning=ft,
            head=head,
            data=data,
            training=training,
            scheduler=sched,
        )

    def save(self, path: str | Path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_train_config(self) -> Config:
        """Convert to a standard Config for reuse with train.py utilities."""
        model_cfg = ModelConfig(
            type=self.pretrained.model_type,
            out_channels=self.pretrained.out_channels,
            in_channels=self.pretrained.in_channels,
            num_blocks=self.pretrained.num_blocks,
            upsampling_depth=self.pretrained.upsampling_depth,
            enc_kernel_size=self.pretrained.enc_kernel_size,
            enc_num_basis=self.pretrained.enc_num_basis,
            num_head_conv_blocks=self.head.num_conv_blocks,
        )
        return Config(data=self.data, model=model_cfg, training=self.training)


# =============================================================================
# Model Loading, Head Replacement & Freezing
# =============================================================================


def create_finetunable_model(config: FinetuneConfig) -> nn.Module:
    """Load pretrained weights, replace mask_net with COI head, apply freeze."""
    pt = config.pretrained

    # Resolve checkpoint path
    ckpt_path = Path(pt.checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).parent.parent.parent.parent / pt.checkpoint_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    print(f"Loading pretrained checkpoint: {ckpt_path}")

    # Build base model and load weights
    ModelClass = SuDORMRF if pt.model_type == "improved" else GroupCommSudoRmRf
    model = ModelClass(
        out_channels=pt.out_channels,
        in_channels=pt.in_channels,
        num_blocks=pt.num_blocks,
        upsampling_depth=pt.upsampling_depth,
        enc_kernel_size=pt.enc_kernel_size,
        enc_num_basis=pt.enc_num_basis,
        num_sources=pt.num_sources,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract state_dict from various checkpoint formats
    if isinstance(checkpoint, nn.Module):
        print("  Checkpoint is a full model object")
        model = checkpoint
    elif isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
            else:
                raise ValueError(
                    f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}"
                )

        # Strip DataParallel / torch.compile prefixes
        cleaned = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }
        result = model.load_state_dict(cleaned, strict=False)
        if result.missing_keys:
            print(f"  Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"  Unexpected keys: {result.unexpected_keys}")
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")

    print(
        f"  Loaded {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
    )

    # Compatibility attribute
    if not hasattr(model, "n_least_samples_req"):
        model.n_least_samples_req = (pt.enc_kernel_size // 2) * (2**pt.upsampling_depth)

    # Replace mask_net with COI separation head
    hd = config.head
    print(f"  Replacing mask_net (num_conv_blocks={hd.num_conv_blocks})")
    model = wrap_model_for_coi(
        model,
        replace_head=True,
        num_conv_blocks=hd.num_conv_blocks,
        upsampling_depth=hd.upsampling_depth or pt.upsampling_depth,
        expanded_channels=hd.expanded_channels or pt.in_channels,
    )

    # Optionally reinitialise decoder bias
    if config.finetuning.reinit_decoder_bias and hasattr(model, "decoder"):
        if model.decoder.bias is not None:
            nn.init.zeros_(model.decoder.bias)

    # Apply initial freeze strategy
    apply_freeze_strategy(model, config)
    print_param_summary(model)
    return model


def apply_freeze_strategy(
    model: nn.Module, config: FinetuneConfig, strategy: str | None = None
) -> None:
    """Freeze / unfreeze parameters according to the chosen strategy."""
    strategy = strategy or config.finetuning.freeze_strategy
    ft = config.finetuning
    num_blocks = config.pretrained.num_blocks

    if strategy == "head_only":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("mask_net.")
        print("Freeze: head_only")

    elif strategy == "partial":
        # Start frozen, selectively unfreeze
        for p in model.parameters():
            p.requires_grad = False

        # Determine which top-level prefixes to unfreeze
        unfreeze_prefixes = ["mask_net."]
        if ft.unfreeze_bottleneck:
            unfreeze_prefixes.append("bottleneck.")
        if ft.unfreeze_encoder:
            unfreeze_prefixes.append("encoder.")
        if ft.unfreeze_decoder:
            unfreeze_prefixes.append("decoder.")

        # Last N separation-module blocks
        n_unfreeze = min(ft.unfreeze_last_n_blocks, num_blocks)
        sm_prefixes = [f"sm.{i}." for i in range(num_blocks - n_unfreeze, num_blocks)]
        unfreeze_prefixes.extend(sm_prefixes)

        for name, p in model.named_parameters():
            if any(name.startswith(pfx) for pfx in unfreeze_prefixes):
                p.requires_grad = True
        print(f"Freeze: partial (last {n_unfreeze}/{num_blocks} SM blocks)")

    elif strategy == "full":
        for p in model.parameters():
            p.requires_grad = True
        print("Freeze: full (all trainable)")

    else:
        raise ValueError(f"Unknown freeze strategy: {strategy!r}")


def print_param_summary(model: nn.Module) -> None:
    """Print trainable vs frozen parameter counts per top-level module."""
    stats: dict[str, list[int]] = {}  # prefix -> [trainable, frozen]
    for name, p in model.named_parameters():
        prefix = name.split(".")[0]
        if prefix not in stats:
            stats[prefix] = [0, 0]
        stats[prefix][0 if p.requires_grad else 1] += p.numel()

    total_t = sum(v[0] for v in stats.values())
    total_f = sum(v[1] for v in stats.values())
    total = total_t + total_f
    print(
        f"  Parameters: {total / 1e6:.2f}M total, "
        f"{total_t / 1e6:.2f}M trainable ({100 * total_t / max(total, 1):.0f}%), "
        f"{total_f / 1e6:.2f}M frozen"
    )
    for prefix, (t, f) in sorted(stats.items()):
        tag = "TRAIN" if f == 0 else ("FROZEN" if t == 0 else "MIXED")
        print(f"    {prefix:20s} {(t + f) / 1e6:7.2f}M  [{tag}]")


# =============================================================================
# Optimizer with Differential Learning Rates
# =============================================================================


def create_optimizer(model: nn.Module, config: FinetuneConfig) -> optim.Adam:
    """Adam with separate LR for head (base LR) and backbone (reduced LR)."""
    base_lr = float(config.training.lr)
    backbone_lr = base_lr * config.finetuning.backbone_lr_multiplier

    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if name.startswith("mask_net.") else backbone_params).append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": base_lr, "name": "head"})
    if backbone_params:
        groups.append(
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"}
        )
    if not groups:
        raise RuntimeError("No trainable parameters — check freeze strategy.")

    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  Optim '{g['name']}': {n / 1e6:.2f}M params, lr={g['lr']:.2e}")
    return optim.Adam(groups)


def _restore_group_lrs(optimizer: optim.Optimizer, target_lrs: dict[str, float]):
    """Reset param-group LRs after train_epoch's uniform warmup overwrites them."""
    for pg in optimizer.param_groups:
        name = pg.get("name")
        if name and name in target_lrs:
            pg["lr"] = target_lrs[name]


# =============================================================================
# Checkpoint Helpers
# =============================================================================


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    global_step: int,
    val_loss: float,
    train_config: Config,
    finetune_config: FinetuneConfig,
    history: dict,
    strategy: str,
):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": train_config.to_dict(),
            "finetune_config": finetune_config.to_dict(),
            "history": history,
            "freeze_strategy": strategy,
            "pretrained_checkpoint": str(finetune_config.pretrained.checkpoint_path),
        },
        path,
    )


# =============================================================================
# Main Finetuning Loop
# =============================================================================


def finetune(config: FinetuneConfig, timestamp: str | None = None):
    """Load pretrained model, replace head, train with optional staged unfreezing."""
    set_seed(config.training.seed)

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(config.training.checkpoint_dir) / timestamp
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    config.save(ckpt_dir / "finetune_config.yaml")

    train_config = config.to_train_config()
    train_config.save(ckpt_dir / "config.yaml")

    # Model
    model = create_finetunable_model(config)
    device = config.training.device
    try:
        model = model.to(device)
    except Exception as e:
        print(f"Error moving to {device}: {e}. Using CPU.")
        device = "cpu"
        model = model.to(device)

    if hasattr(torch, "compile") and config.training.compile_model:
        try:
            model = torch.compile(model, backend=config.training.compile_backend)
        except Exception as e:
            print(f"Warning: torch.compile failed ({e})")

    # Data — reuse create_dataloader from train.py via the converted Config
    print("\nCreating train dataloader...")
    train_loader, _ = create_dataloader(train_config, "train")

    # Training components
    criterion = COIWeightedLoss(class_weight=config.training.class_weight)
    base_lr = float(config.training.lr)
    backbone_lr = base_lr * config.finetuning.backbone_lr_multiplier
    target_lrs = {"head": base_lr, "backbone": backbone_lr}

    optimizer = create_optimizer(model, config)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler.factor,
        patience=config.scheduler.patience,
        min_lr=config.scheduler.min_lr,
    )

    warmup_steps = config.training.warmup_steps
    validate_every_n = config.training.validate_every_n_epochs

    ft = config.finetuning
    current_strategy = ft.freeze_strategy

    best_val_loss = float("inf")
    epochs_no_improve = 0
    global_step = 0
    epoch = 0
    val_loss = float("inf")
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "grad_norms": [],
        "freeze_strategy": [],
    }

    print(
        f"\nFinetuning: {config.training.num_epochs} epochs, "
        f"head_lr={base_lr:.2e}, backbone_lr={backbone_lr:.2e}, "
        f"strategy={current_strategy}\n"
    )

    for epoch in range(1, config.training.num_epochs + 1):
        # --- Staged Unfreezing ---
        if ft.staged_unfreeze:
            new = current_strategy
            if epoch == ft.stage1_epochs + 1 and current_strategy == "head_only":
                new = "partial"
            elif epoch == ft.stage2_epochs + 1 and current_strategy == "partial":
                new = "full"
            if new != current_strategy:
                print(f"\n--- Stage transition: {current_strategy} -> {new} ---")
                current_strategy = new
                apply_freeze_strategy(model, config, current_strategy)
                print_param_summary(model)
                optimizer = create_optimizer(model, config)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=config.scheduler.factor,
                    patience=config.scheduler.patience,
                    min_lr=config.scheduler.min_lr,
                )

        history["freeze_strategy"].append(current_strategy)
        print(f"Epoch {epoch}/{config.training.num_epochs} [{current_strategy}]")

        # Train — reuse train_epoch from train.py (uses base_lr for uniform warmup)
        train_loss, global_step, epoch_norms = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            clip_grad_norm=config.training.clip_grad_norm,
            grad_accum_steps=config.training.grad_accum_steps,
            use_amp=config.training.use_amp,
            warmup_steps=warmup_steps,
            global_step=global_step,
            base_lr=base_lr,
            snr_range=tuple(config.data.snr_range),
        )

        # Restore per-group LRs (train_epoch's warmup sets all groups to base_lr)
        _restore_group_lrs(optimizer, target_lrs)

        history["train_loss"].append(train_loss)
        valid_norms = [n for n in epoch_norms if not np.isnan(n)]
        history["grad_norms"].append(
            np.mean(valid_norms) if valid_norms else float("nan")
        )

        # Validation
        should_validate = (
            epoch % validate_every_n == 0
            or epoch == 1
            or epoch == config.training.num_epochs
        )
        if should_validate:
            val_loader, _ = create_dataloader(train_config, "val")
            val_loss = validate_epoch(
                model,
                val_loader,
                criterion,
                device,
                use_amp=config.training.use_amp,
                snr_range=tuple(config.data.snr_range),
            )
            del val_loader
            gc.collect()

            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)
            # Sync target_lrs with scheduler updates
            for pg in optimizer.param_groups:
                name = pg.get("name")
                if name:
                    target_lrs[name] = pg["lr"]

            lr_info = ", ".join(
                f"{pg.get('name', '?')}={pg['lr']:.2e}" for pg in optimizer.param_groups
            )
            print(f"  Train: {train_loss:.4f}, Val: {val_loss:.4f}  LR: {lr_info}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                _save_checkpoint(
                    ckpt_dir / "best_model.pt",
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    val_loss,
                    train_config,
                    config,
                    history,
                    current_strategy,
                )
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= config.training.patience:
                print(f"\nEarly stopping after {epoch} epochs")
                break
        else:
            val_loss = history["val_loss"][-1] if history["val_loss"] else float("inf")
            history["val_loss"].append(val_loss)
            print(f"  Train: {train_loss:.4f}")

    # Final save
    _save_checkpoint(
        ckpt_dir / "final_model.pt",
        model,
        optimizer,
        epoch,
        global_step,
        val_loss,
        train_config,
        config,
        history,
        current_strategy,
    )
    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone! Best val loss: {best_val_loss:.4f}  Checkpoints: {ckpt_dir}")


# =============================================================================
# Dataset Preparation (shared with train.py main)
# =============================================================================


def prepare_dataset(config: FinetuneConfig, checkpoint_dir: Path) -> str:
    """Load metadata, sample COI/non-COI, save CSV. Returns CSV path."""
    project_root = Path(__file__).parent.parent.parent.parent
    all_metadata = load_metadata_datasets(
        str(project_root / "data"), str(project_root.parent / "datasets")
    )
    sep_metadata, _ = split_seperation_classification(all_metadata)
    print(f"Loaded {len(all_metadata)} samples, {len(sep_metadata)} for separation")

    # pull the list of semantic labels from the configuration object rather than
    # keeping it hard‑coded here.  The dataclass default means the attribute
    # will always exist, but we enforce that it is non‑empty so the user is
    # reminded to specify something sensible.
    target_classes = getattr(config.data, "target_classes", None)
    if not target_classes:
        raise ValueError(
            "No target_classes specified in config.data – please add a list of "
            "labels to the YAML configuration"
        )

    coi_df = get_coi(sep_metadata, target_classes)
    sampled_df = sample_non_coi(sep_metadata, coi_df, coi_ratio=0.25)
    sampled_df["label"] = sampled_df["label"].apply(
        lambda x: (
            1
            if (isinstance(x, list) and any(l in target_classes for l in x))
            or (isinstance(x, str) and x in target_classes)
            else 0
        )
    )

    # Drop missing files
    sampled_df["exists"] = sampled_df["filename"].apply(lambda f: Path(f).exists())
    n_missing = (~sampled_df["exists"]).sum()
    if n_missing:
        print(f"  Dropping {n_missing} missing files")
        sampled_df = sampled_df[sampled_df["exists"]]
    sampled_df = sampled_df.drop(columns=["exists"])

    for split in ("train", "val", "test"):
        s = sampled_df[sampled_df["split"] == split]
        print(
            f"  {split}: {len(s)} (COI: {(s['label'] == 1).sum()}, "
            f"non-COI: {(s['label'] == 0).sum()})"
        )

    sampled_df["label"] = sampled_df["label"].astype("uint8")
    sampled_df["split"] = sampled_df["split"].astype("category")
    df_path = checkpoint_dir / "separation_dataset.csv"
    sampled_df.to_csv(df_path, index=False)
    print(f"  Saved dataset: {df_path}")

    del all_metadata, sep_metadata, coi_df, sampled_df
    gc.collect()
    return str(df_path)


# =============================================================================
# Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Finetune a pretrained SuDoRM-RF model for COI separation"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to finetune config YAML"
    )
    config = FinetuneConfig.from_yaml(str(parser.parse_args().config))

    print(
        f"Pretrained: {config.pretrained.checkpoint_path} "
        f"({config.pretrained.model_type}, {config.pretrained.num_blocks} blocks)"
    )
    print(
        f"Strategy: {config.finetuning.freeze_strategy}, "
        f"head_blocks: {config.head.num_conv_blocks}, "
        f"device: {config.training.device}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(config.training.checkpoint_dir) / timestamp
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    config.data.df_path = prepare_dataset(config, ckpt_dir)
    finetune(config, timestamp=timestamp)


if __name__ == "__main__":
    main()

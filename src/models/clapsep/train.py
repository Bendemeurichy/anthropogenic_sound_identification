"""
Training script for CLAPSep model adapted for COI (Class of Interest) separation.

Supports two conditioning modes:
    --text-conditioning     : decoder conditioned on CLAP text embeddings
                              (inference-time flexible prompts)
    (default)               : decoder conditioned on learned embeddings
                              (no text dependency, COI class fixed at train time)

Uses the shared COI training utilities from src/common/coi_training.py.
Supports optional LoRA fine-tuning of the audio encoder.

Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for COI, 0 for background (non-COI)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch

# Add parent directories to path for imports
_src_root = Path(__file__).parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from common.coi_training import create_coi_dataloader

# Import CLAPSep components
try:
    import laion_clap

    from models.clapsep.base.model.CLAPSep_decoder import HTSAT_Decoder

    HAS_CLAP = True
except ImportError:
    HAS_CLAP = False
    print("Warning: laion_clap not available. Install with: pip install laion-clap")

from .coi_model import COICLAPSepDecoder, COICLAPSep
from .text_model import TextPromptCLAPSep


# =============================================================================
# Training function
# =============================================================================


def train(
    df_path: str,
    clap_checkpoint: str,
    checkpoint_dir: str = "checkpoints/clapsep",
    sample_rate: int = 32000,
    segment_length: float = 5.0,
    snr_range: tuple = (-5, 5),
    batch_size: int = 16,
    num_epochs: int = 150,
    lr: float = 1e-4,
    device: str = "cuda",
    num_workers: int = 4,
    seed: int = 42,
    nfft: int = 1024,
    freeze_encoder: bool = True,
    use_lora: bool = False,
    lora_rank: int = 8,
    class_weight: float = 1.5,
    precision: str = "bf16-mixed",
    embed_dim: int = 128,
    encoder_embed_dim: int = 128,
    d_attn: int = 640,
    n_masker_layer: int = 3,
    use_webdataset: bool = False,
    webdataset_path: str = "",
    text_conditioning: bool = False,
    coi_text_prompts: Optional[List[List[str]]] = None,
    bg_text_prompts: Optional[List[str]] = None,
):
    """Main training function."""
    if not HAS_CLAP:
        raise RuntimeError(
            "laion_clap and CLAPSep decoder are required. "
            "Install laion-clap with: pip install laion-clap"
        )

    import laion_clap as _laion_clap
    import yaml

    from models.clapsep.base.model.CLAPSep_decoder import (
        HTSAT_Decoder as _HTSAT_Decoder,
    )

    pl.seed_everything(seed)

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "text_prompt_" if text_conditioning else ""
    checkpoint_path = Path(checkpoint_dir) / f"{prefix}{timestamp}"
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    # Load config for target_classes / text prompts
    config_path = Path(__file__).parent / "training_config.yaml"
    target_classes: List[str] = []
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        target_classes = config.get("data", {}).get("target_classes", []) or []
        if target_classes:
            print(f"Loaded target_classes from config: {target_classes}")

        # Load text prompts when in text-conditioning mode
        if text_conditioning:
            if coi_text_prompts is None:
                coi_text_prompts = config.get("model", {}).get(
                    "coi_text_prompts", [["airplane engine", "aircraft noise"]]
                )
            if bg_text_prompts is None:
                bg_text_prompts = config.get("model", {}).get(
                    "background_text_prompts",
                    ["ambient noise", "background sounds"],
                )
            print("Text-prompt configuration:")
            for i, prompts in enumerate(coi_text_prompts):
                print(f"  COI class {i}: {prompts}")
            print(f"  Background:    {bg_text_prompts}")
    elif text_conditioning:
        coi_text_prompts = coi_text_prompts or [["airplane engine", "aircraft noise"]]
        bg_text_prompts = bg_text_prompts or ["ambient noise", "background sounds"]

    # Create dataloaders
    print("Creating dataloaders...")
    dataloader_kwargs = dict(
        df_path=df_path,
        batch_size=batch_size,
        sample_rate=sample_rate,
        segment_length=segment_length,
        snr_range=snr_range,
        stereo=False,
        num_workers=num_workers,
        seed=seed,
        use_webdataset=use_webdataset,
        webdataset_path=webdataset_path if use_webdataset else None,
        target_classes=target_classes if use_webdataset else None,
        coi_ratio=0.25,
    )

    train_loader, _ = create_coi_dataloader(split="train", **dataloader_kwargs)
    val_loader, _ = create_coi_dataloader(split="val", **dataloader_kwargs)

    # Load pretrained CLAP model
    print(f"Loading CLAP model from {clap_checkpoint}...")
    clap_model = _laion_clap.CLAP_Module(
        enable_fusion=False, amodel="HTSAT-base", device="cpu"
    )
    clap_model.load_ckpt(clap_checkpoint)

    # Create decoder
    print("Creating decoder...")
    decoder = _HTSAT_Decoder(
        lan_embed_dim=1024,
        depths=[1, 1, 1, 1],
        embed_dim=embed_dim,
        encoder_embed_dim=encoder_embed_dim,
        phase=False,
        spec_factor=8,
        d_attn=d_attn,
        n_masker_layer=n_masker_layer,
        conv=False,
    )

    # Build Lightning module based on conditioning mode
    print("Creating model...")
    if text_conditioning:
        model = TextPromptCLAPSep(
            clap_model=clap_model,
            decoder_model=decoder,
            coi_text_prompts=coi_text_prompts
            or [["airplane engine", "aircraft noise"]],
            background_text_prompts=bg_text_prompts
            or ["ambient noise", "background sounds"],
            lr=lr,
            nfft=nfft,
            sample_rate=sample_rate,
            resample_rate=48000,
            class_weight=class_weight,
            freeze_encoder=freeze_encoder,
            use_lora=use_lora,
            lora_rank=lora_rank,
        )
    else:
        coi_decoder = COICLAPSepDecoder(decoder=decoder, embed_dim=512, num_sources=2)
        model = COICLAPSep(
            clap_model=clap_model,
            decoder_model=coi_decoder,
            lr=lr,
            nfft=nfft,
            sample_rate=sample_rate,
            resample_rate=48000,
            class_weight=class_weight,
            freeze_encoder=freeze_encoder,
            use_lora=use_lora,
            lora_rank=lora_rank,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(
        f"Trainable parameters: {trainable_params / 1e6:.2f}M "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    if use_lora:
        print(f"  Using LoRA fine-tuning (rank={lora_rank})")
    elif freeze_encoder:
        print("  Encoder frozen (decoder-only training)")
    else:
        print("  Full encoder+decoder fine-tuning")

    # Setup trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    trainer = pl.Trainer(
        default_root_dir=str(checkpoint_path),
        devices=1 if device == "cuda" and torch.cuda.is_available() else "auto",
        accelerator="gpu" if device == "cuda" and torch.cuda.is_available() else "cpu",
        benchmark=True,
        gradient_clip_val=5.0,
        precision=precision,
        max_epochs=num_epochs,
        callbacks=[
            checkpoint_callback,
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=30,
                mode="min",
                check_on_train_epoch_end=False,
            ),
        ],
    )

    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_val_loss = (
        checkpoint_callback.best_model_score.item()
        if checkpoint_callback.best_model_score is not None
        else float("inf")
    )
    best_val_sisnr = -best_val_loss
    print(f"\nTraining completed! Checkpoints saved to {checkpoint_path}")
    print(f"Best Val SI-SNR: {best_val_sisnr:.2f} dB")
    return checkpoint_path


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train CLAPSep for COI separation"
    )
    parser.add_argument("--df-path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument(
        "--clap-checkpoint", type=str, required=True,
        help="Path to pretrained CLAP checkpoint",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/clapsep")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--segment-length", type=float, default=5.0)
    parser.add_argument("--snr-min", type=float, default=-5)
    parser.add_argument("--snr-max", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--no-freeze-encoder", action="store_true",
                        help="Allow encoder fine-tuning (full or LoRA)")
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRA for parameter-efficient encoder fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--class-weight", type=float, default=1.5)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--encoder-embed-dim", type=int, default=128)
    parser.add_argument("--d-attn", type=int, default=640)
    parser.add_argument("--n-masker-layer", type=int, default=3)
    parser.add_argument("--use-webdataset", action="store_true",
                        help="Use WebDataset tar shards instead of individual files")
    parser.add_argument("--webdataset-path", type=str, default="",
                        help="Path to WebDataset tar shards directory")
    parser.add_argument("--text-conditioning", action="store_true",
                        help="Use CLAP text embeddings for conditioning (flexible at inference)")

    args = parser.parse_args()

    if args.use_lora and not args.no_freeze_encoder:
        print("Warning: --use-lora requires --no-freeze-encoder. Setting --no-freeze-encoder=True")
        args.no_freeze_encoder = True

    if args.use_webdataset and not args.webdataset_path:
        parser.error("--webdataset-path is required when --use-webdataset is set")

    train(
        df_path=args.df_path,
        clap_checkpoint=args.clap_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        snr_range=(args.snr_min, args.snr_max),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        nfft=args.nfft,
        freeze_encoder=not args.no_freeze_encoder,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        class_weight=args.class_weight,
        precision=args.precision,
        embed_dim=args.embed_dim,
        encoder_embed_dim=args.encoder_embed_dim,
        d_attn=args.d_attn,
        n_masker_layer=args.n_masker_layer,
        use_webdataset=args.use_webdataset,
        webdataset_path=args.webdataset_path,
        text_conditioning=args.text_conditioning,
    )


if __name__ == "__main__":
    main()

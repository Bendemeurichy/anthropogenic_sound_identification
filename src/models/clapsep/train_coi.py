"""
Training script for CLAPSep model adapted for COI (Class of Interest) separation.

This script fine-tunes the pretrained CLAPSep model for single-target + residue
separation, using the shared COI training utilities from src/common/coi_training.py.

The approach uses the pretrained CLAP audio encoder with optional LoRA fine-tuning,
and trains a decoder to separate COI from background without text conditioning.

Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for COI, 0 for background (non-COI)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchlibrosa import ISTFT, STFT
from torchlibrosa.stft import magphase

# Add parent directories to path for imports
_src_root = Path(__file__).parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

# Import shared COI training utilities
from common.coi_training import (
    COIWeightedLoss,
    create_coi_dataloader,
    prepare_batch,
    sisnr,
)

# Import CLAPSep components and LoRA
try:
    import laion_clap
    from models.clapsep.base.model.CLAPSep_decoder import HTSAT_Decoder
    HAS_CLAP = True
except ImportError:
    HAS_CLAP = False
    print("Warning: laion_clap not available. Install with: pip install laion-clap")

try:
    import loralib as lora
    HAS_LORA = True
except ImportError:
    HAS_LORA = False
    print("Warning: loralib not available. Install with: pip install loralib")
    print("  LoRA fine-tuning will be disabled. Set use_lora=False or install loralib.")


# =============================================================================
# LoRA utilities (from original CLAPSep implementation)
# =============================================================================


def set_module(model, submodule_key, module):
    """Set a submodule in a model by its key path."""
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def apply_lora_to_model(model, rank: int = 8):
    """
    Replace linear layers in WindowAttention modules with LoRA layers.
    
    This follows the original CLAPSep implementation which applies LoRA
    specifically to attention layers in the HTSAT encoder.
    
    Args:
        model: The audio encoder model
        rank: LoRA rank (lower = fewer parameters, typically 4-16)
        
    Returns:
        Model with LoRA layers applied
    """
    if not HAS_LORA:
        raise RuntimeError(
            "loralib is required for LoRA fine-tuning. "
            "Install with: pip install loralib"
        )
    
    for module_name, module in model.named_modules():
        # Apply LoRA only to WindowAttention layers (as in original CLAPSep)
        if 'WindowAttention' in str(type(module)):
            for layer_name, layer in module.named_modules():
                if isinstance(layer, torch.nn.Linear):
                    # Create LoRA layer with same dimensions
                    lora_layer = lora.Linear(
                        layer.in_features,
                        layer.out_features,
                        r=rank,
                        bias=hasattr(layer, 'bias'),
                        merge_weights=False
                    )
                    # Copy pretrained weights
                    lora_layer.weight = layer.weight
                    if hasattr(layer, 'bias'):
                        lora_layer.bias = layer.bias
                    # Replace the layer
                    full_path = f"{module_name}.{layer_name}" if module_name else layer_name
                    set_module(model, full_path, lora_layer)
    
    # Mark only LoRA parameters as trainable
    lora.mark_only_lora_as_trainable(model, bias='lora_only')
    
    return model


# =============================================================================
# COI-adapted CLAPSep Model
# =============================================================================


class COICLAPSepDecoder(nn.Module):
    """Modified CLAPSep decoder for COI separation (2 sources: COI + background).

    Instead of using text embeddings, this model learns a fixed embedding
    for the COI class during training.
    """

    def __init__(
        self,
        decoder: nn.Module,
        embed_dim: int = 1024,
        num_sources: int = 2,
    ):
        super().__init__()
        self.decoder = decoder
        self.num_sources = num_sources

        # Learnable embeddings for COI and background
        # These replace the text/audio embeddings from CLAP
        self.coi_embedding = nn.Parameter(torch.randn(1, embed_dim))
        self.bg_embedding = nn.Parameter(torch.randn(1, embed_dim))

        # Initialize with small values
        nn.init.normal_(self.coi_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.bg_embedding, mean=0.0, std=0.02)

    def forward(self, hidden_state, skip_features, batch_size: int = 1):
        """
        Generate masks for COI and background sources.

        Args:
            hidden_state: Features from CLAP audio encoder
            skip_features: Skip connection features from encoder
            batch_size: Current batch size

        Returns:
            masks: (num_sources, B, 1, F, T) separation masks
        """
        masks = []

        # Generate mask for COI using learned embedding
        coi_embed = self.coi_embedding.expand(batch_size, -1)
        # Normalize and concatenate positive/negative embeddings as in original CLAPSep
        embed_coi = torch.nn.functional.normalize(
            torch.cat([coi_embed, self.bg_embedding.expand(batch_size, -1)], dim=-1),
            dim=-1,
        )
        mask_coi = self.decoder(
            hidden_state=hidden_state, skip_features=skip_features, embed=embed_coi
        )
        masks.append(mask_coi)

        # Generate mask for background using learned embedding
        embed_bg = torch.nn.functional.normalize(
            torch.cat([self.bg_embedding.expand(batch_size, -1), coi_embed], dim=-1),
            dim=-1,
        )
        mask_bg = self.decoder(
            hidden_state=hidden_state, skip_features=skip_features, embed=embed_bg
        )
        masks.append(mask_bg)

        return masks


class COICLAPSep(pl.LightningModule):
    """PyTorch Lightning module for COI separation using CLAPSep architecture.
    
    Supports three encoder modes:
    1. freeze_encoder=True, use_lora=False: Encoder completely frozen (fastest, least memory)
    2. freeze_encoder=False, use_lora=False: Full encoder fine-tuning (most parameters)
    3. freeze_encoder=False, use_lora=True: LoRA fine-tuning (parameter-efficient, recommended)
    """

    def __init__(
        self,
        clap_model,
        decoder_model: nn.Module,
        lr: float = 1e-4,
        nfft: int = 1024,
        sample_rate: int = 32000,
        resample_rate: int = 48000,
        class_weight: float = 1.5,
        freeze_encoder: bool = True,
        use_lora: bool = False,
        lora_rank: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["clap_model", "decoder_model"])

        self.lr = lr
        self.class_weight = class_weight
        self.sample_rate = sample_rate
        self.resample_rate = resample_rate
        self.use_lora = use_lora
        self.lora_rank = lora_rank

        # CLAP model (always frozen, only used for reference)
        self.clap_model = clap_model
        for p in self.clap_model.parameters():
            p.requires_grad = False

        # Copy audio branch for fine-tuning
        import copy
        self.audio_branch = copy.deepcopy(self.clap_model.model.audio_branch)

        if freeze_encoder:
            # Mode 1: Completely frozen encoder
            for p in self.audio_branch.parameters():
                p.requires_grad = False
        elif use_lora:
            # Mode 2: LoRA fine-tuning (parameter-efficient)
            if not HAS_LORA:
                raise RuntimeError(
                    "loralib is required for LoRA fine-tuning. "
                    "Install with: pip install loralib\n"
                    "Or set use_lora=False to use full fine-tuning or frozen encoder."
                )
            print(f"Applying LoRA (rank={lora_rank}) to audio encoder...")
            self.audio_branch = apply_lora_to_model(self.audio_branch, rank=lora_rank)
            # Count trainable params
            lora_params = sum(p.numel() for p in self.audio_branch.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.audio_branch.parameters())
            print(f"  LoRA trainable: {lora_params/1e6:.2f}M / {total_params/1e6:.2f}M total "
                  f"({100*lora_params/total_params:.2f}%)")
        else:
            # Mode 3: Full fine-tuning (all parameters trainable)
            print("Using full encoder fine-tuning (all parameters trainable)")

        # COI-adapted decoder
        self.decoder = decoder_model

        # STFT for time-frequency processing
        self.stft = STFT(
            n_fft=nfft,
            hop_length=320,
            win_length=nfft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.istft = ISTFT(
            n_fft=nfft,
            hop_length=320,
            win_length=nfft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )

        # Resampler for CLAP input (expects 48kHz)
        import torchaudio

        self.resampler = torchaudio.transforms.Resample(sample_rate, resample_rate)

        # Install forward hooks for skip connections
        self.features = self._install_forward_hooks()

        # Loss function
        self.criterion = COIWeightedLoss(class_weight=class_weight)

    def _install_forward_hooks(self):
        """Install forward hooks to capture intermediate features."""
        features = []

        def get_features_list(_, __, output):
            features.append(output)

        def get_features_list_basic_layer(_, __, output):
            features.append(output[0])

        def spectrogram_padding(_, __, out):
            return torch.nn.functional.pad(out, (0, 0, 0, 1024 - out.size(2)))

        self.audio_branch.spectrogram_extractor.register_forward_hook(
            spectrogram_padding
        )
        self.audio_branch.patch_embed.register_forward_hook(get_features_list)
        for module in self.audio_branch.layers:
            module.register_forward_hook(get_features_list_basic_layer)

        return features

    def wav_reconstruct(self, mask, mag_x, cos_x, sin_x, length):
        """Reconstruct waveform from mask and STFT components."""
        if isinstance(mask, (list, tuple)):
            # Phase-aware mask
            mag_y = torch.nn.functional.relu_(mag_x * mask[0])
            _, mask_cos, mask_sin = magphase(mask[1], mask[2])
            cos_y = cos_x * mask_cos - sin_x * mask_sin
            sin_y = sin_x * mask_cos + cos_x * mask_sin
        else:
            mag_y = torch.nn.functional.relu_(mag_x * mask)
            cos_y = cos_x
            sin_y = sin_x

        pred = self.istft(mag_y * cos_y, mag_y * sin_y, length=length)
        return pred

    def forward(self, mixture):
        """
        Separate mixture into COI and background.

        Args:
            mixture: (B, T) input mixture waveform at self.sample_rate

        Returns:
            separated: (B, 2, T) with [COI, background]
        """
        B = mixture.shape[0]
        length = mixture.shape[-1]

        # STFT
        real, imag = self.stft(mixture)
        mag, cos, sin = magphase(real, imag)

        # Clear features from previous forward pass
        del self.features[:]
        self.features.append(mag)

        # Get features from audio encoder
        mixture_resampled = self.resampler(mixture)
        with torch.no_grad():
            self.audio_branch({"waveform": mixture_resampled})

        # Generate masks using COI decoder
        masks = self.decoder(
            hidden_state=self.features[-1],
            skip_features=self.features[:-1],
            batch_size=B,
        )

        # Reconstruct separated sources
        separated = []
        for mask in masks:
            pred = self.wav_reconstruct(mask, mag, cos, sin, length=length)
            separated.append(pred)

        del self.features[:]

        # Stack: (B, n_src, T)
        return torch.stack(separated, dim=1)

    def training_step(self, batch, batch_idx):
        mixture, sources = batch

        # Prepare batch with SNR mixing (sources: B, n_src, T)
        snr_range = (-5, 5)
        mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=False)

        # Forward pass
        separated = self(mixture)

        # Compute loss
        loss = self.criterion(separated, clean_wavs)

        self.log(
            "train_loss", loss.item(), on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        mixture, sources = batch

        snr_range = (-5, 5)
        mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=True)

        with torch.no_grad():
            separated = self(mixture)
            loss = self.criterion(separated, clean_wavs)

            # Compute SI-SNR metrics
            coi_sisnr = sisnr(separated[:, 0], clean_wavs[:, 0]).mean()
            bg_sisnr = sisnr(separated[:, -1], clean_wavs[:, -1]).mean()

        self.log("val_loss", loss.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "val_coi_sisnr",
            coi_sisnr.item(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_bg_sisnr",
            bg_sisnr.item(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"val_loss": loss}

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=10, verbose=True, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }


# =============================================================================
# Training Function
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
):
    """Main training function."""
    if not HAS_CLAP:
        raise RuntimeError(
            "laion_clap and CLAPSep decoder are required. "
            "Install laion-clap with: pip install laion-clap"
        )

    # Import here to avoid unbound errors when HAS_CLAP is False
    import laion_clap as _laion_clap

    from models.clapsep.base.model.CLAPSep_decoder import (
        HTSAT_Decoder as _HTSAT_Decoder,
    )

    # Set seed
    pl.seed_everything(seed)

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = Path(checkpoint_dir) / timestamp
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, _ = create_coi_dataloader(
        df_path=df_path,
        split="train",
        batch_size=batch_size,
        sample_rate=sample_rate,
        segment_length=segment_length,
        snr_range=snr_range,
        stereo=False,  # CLAPSep uses mono
        num_workers=num_workers,
        seed=seed,
    )

    val_loader, _ = create_coi_dataloader(
        df_path=df_path,
        split="val",
        batch_size=batch_size,
        sample_rate=sample_rate,
        segment_length=segment_length,
        snr_range=snr_range,
        stereo=False,
        num_workers=num_workers,
        seed=seed,
    )

    # Load pretrained CLAP model
    print(f"Loading CLAP model from {clap_checkpoint}...")
    clap_model = _laion_clap.CLAP_Module(
        enable_fusion=False, amodel="HTSAT-base", device="cpu"
    )
    clap_model.load_ckpt(clap_checkpoint)

    # Create decoder
    print("Creating decoder...")
    base_decoder = _HTSAT_Decoder(
        lan_embed_dim=1024,
        depths=[1, 1, 1, 1],
        embed_dim=128,
        encoder_embed_dim=128,
        phase=False,
        spec_factor=8,
        d_attn=640,
        n_masker_layer=3,
        conv=False,
    )

    # Wrap decoder for COI separation
    coi_decoder = COICLAPSepDecoder(
        decoder=base_decoder,
        embed_dim=1024,
        num_sources=2,
    )

    # Create Lightning module
    print("Creating model...")
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
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")
    
    if use_lora:
        print(f"  Using LoRA fine-tuning (rank={lora_rank})")
    elif freeze_encoder:
        print(f"  Encoder frozen (decoder-only training)")
    else:
        print(f"  Full encoder+decoder fine-tuning")

    # Setup trainer
    trainer = pl.Trainer(
        default_root_dir=str(checkpoint_path),
        devices=1 if device == "cuda" and torch.cuda.is_available() else "auto",
        accelerator="gpu" if device == "cuda" and torch.cuda.is_available() else "cpu",
        benchmark=True,
        gradient_clip_val=5.0,
        precision=precision,
        max_epochs=num_epochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_path,
                filename="best-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=30,
                mode="min",
            ),
        ],
    )

    # Train
    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"\nTraining completed! Checkpoints saved to {checkpoint_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train CLAPSep for COI separation")
    parser.add_argument(
        "--df-path", type=str, required=True, help="Path to dataset CSV"
    )
    parser.add_argument(
        "--clap-checkpoint",
        type=str,
        required=True,
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
                        help="Use LoRA for parameter-efficient encoder fine-tuning (requires --no-freeze-encoder)")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank (4-16 typical, lower=fewer params)")
    parser.add_argument("--class-weight", type=float, default=1.5)
    parser.add_argument("--precision", type=str, default="bf16-mixed")

    args = parser.parse_args()
    
    # Validate LoRA usage
    if args.use_lora and not args.no_freeze_encoder:
        print("Warning: --use-lora requires --no-freeze-encoder. Setting --no-freeze-encoder=True")
        args.no_freeze_encoder = True

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
    )


if __name__ == "__main__":
    main()

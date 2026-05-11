"""
Training script for CLAPSep with TEXT-PROMPT conditioning for COI separation.

Mirrors the working ``train_coi.py`` pipeline (file-based + WebDataset, shared
``coi_training`` utilities, LoRA, bf16-mixed precision) but conditions the
decoder on CLAP text embeddings instead of learned embeddings. This preserves
inference-time flexibility: the prompt can be changed without retraining.

Key design choices (chosen by user):
  - One random (pos, neg) prompt pair sampled per training step, shared across
    the batch (1 text-encoder call/step, matches base CLAPSep).
  - Decoder is called twice per step with swapped embeddings to produce both
    COI and background masks, yielding proper supervised background training
    (matches ``COICLAPSepDecoder`` semantics in ``train_coi.py``).
  - No wrong-prompt branch (kept simple; revisit only if collapse appears).

Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for COI, 0 for background (non-COI)
"""

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List

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
# LoRA utilities (identical to train_coi.py)
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
    """Replace linear layers in WindowAttention modules with LoRA layers."""
    if not HAS_LORA:
        raise RuntimeError(
            "loralib is required for LoRA fine-tuning. "
            "Install with: pip install loralib"
        )

    for module_name, module in model.named_modules():
        if 'WindowAttention' in str(type(module)):
            for layer_name, layer in module.named_modules():
                if isinstance(layer, torch.nn.Linear):
                    lora_layer = lora.Linear(
                        layer.in_features,
                        layer.out_features,
                        r=rank,
                        bias=hasattr(layer, 'bias'),
                        merge_weights=False,
                    )
                    lora_layer.weight = layer.weight
                    if hasattr(layer, 'bias'):
                        lora_layer.bias = layer.bias
                    full_path = f"{module_name}.{layer_name}" if module_name else layer_name
                    set_module(model, full_path, lora_layer)

    lora.mark_only_lora_as_trainable(model, bias='lora_only')
    return model


# =============================================================================
# Text-Prompt CLAPSep model
# =============================================================================


class TextPromptCLAPSep(pl.LightningModule):
    """COI separation with CLAP text-prompt conditioning.

    Decoder is called twice per forward (with swapped pos/neg embeddings) so
    that both the COI head and the background head receive direct supervision
    from ``COIWeightedLoss``, matching the structure of ``COICLAPSepDecoder``.
    """

    def __init__(
        self,
        clap_model,
        decoder_model: nn.Module,
        coi_text_prompts: List[List[str]],
        background_text_prompts: List[str],
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
        # Save hyperparameters (including coi_text_prompts) so inference.py can
        # auto-route to _TextPromptModelAdapter via the "coi_text_prompts" key.
        self.save_hyperparameters(ignore=["clap_model", "decoder_model"])

        self.lr = lr
        self.class_weight = class_weight
        self.sample_rate = sample_rate
        self.resample_rate = resample_rate
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.coi_text_prompts = coi_text_prompts
        self.background_text_prompts = background_text_prompts

        # CLAP model is needed at runtime for get_text_embedding (always frozen).
        # Stored as a regular attribute (not submodule) to avoid shipping its
        # weights in the checkpoint — inference.py rebuilds it from the CLAP
        # checkpoint on load.
        self.clap_model = clap_model
        for p in self.clap_model.parameters():
            p.requires_grad = False
        self.clap_model.eval()

        # Copy audio branch for fine-tuning
        import copy
        self.audio_branch = copy.deepcopy(clap_model.model.audio_branch)

        if freeze_encoder:
            for p in self.audio_branch.parameters():
                p.requires_grad = False
            print("Audio encoder: FROZEN (decoder-only training)")
        elif use_lora:
            if not HAS_LORA:
                raise RuntimeError(
                    "loralib is required for LoRA fine-tuning. "
                    "Install with: pip install loralib"
                )
            print(f"Applying LoRA (rank={lora_rank}) to audio encoder...")
            self.audio_branch = apply_lora_to_model(self.audio_branch, rank=lora_rank)
            lora_params = sum(p.numel() for p in self.audio_branch.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.audio_branch.parameters())
            print(
                f"  LoRA trainable: {lora_params/1e6:.2f}M / {total_params/1e6:.2f}M total "
                f"({100*lora_params/total_params:.2f}%)"
            )
        else:
            print("Audio encoder: FULL FINE-TUNING (all parameters trainable)")

        self.decoder = decoder_model

        # STFT/iSTFT
        self.stft = STFT(
            n_fft=nfft, hop_length=320, win_length=nfft,
            window="hann", center=True, pad_mode="reflect", freeze_parameters=True,
        )
        self.istft = ISTFT(
            n_fft=nfft, hop_length=320, win_length=nfft,
            window="hann", center=True, pad_mode="reflect", freeze_parameters=True,
        )

        # Resampler for CLAP input (expects 48kHz)
        import torchaudio
        self.resampler = torchaudio.transforms.Resample(sample_rate, resample_rate)

        # Forward hooks for skip connections
        self.features = self._install_forward_hooks()

        # Loss
        self.criterion = COIWeightedLoss(class_weight=class_weight)

    def _install_forward_hooks(self):
        features = []

        def get_features_list(_, __, output):
            features.append(output)

        def get_features_list_basic_layer(_, __, output):
            features.append(output[0])

        def spectrogram_padding(_, __, out):
            return torch.nn.functional.pad(out, (0, 0, 0, 1024 - out.size(2)))

        self.audio_branch.spectrogram_extractor.register_forward_hook(spectrogram_padding)
        self.audio_branch.patch_embed.register_forward_hook(get_features_list)
        for module in self.audio_branch.layers:
            module.register_forward_hook(get_features_list_basic_layer)

        return features

    def wav_reconstruct(self, mask, mag_x, cos_x, sin_x, length):
        """Reconstruct waveform from mask and STFT components."""
        if isinstance(mask, (list, tuple)):
            mag_y = torch.nn.functional.relu_(mag_x * mask[0])
            _, mask_cos, mask_sin = magphase(mask[1], mask[2])
            cos_y = cos_x * mask_cos - sin_x * mask_sin
            sin_y = sin_x * mask_cos + cos_x * mask_sin
        else:
            mag_y = torch.nn.functional.relu_(mag_x * mask)
            cos_y = cos_x
            sin_y = sin_x
        return self.istft(mag_y * cos_y, mag_y * sin_y, length=length)

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get a single CLAP text embedding (1, embed_dim) on self.device."""
        with torch.no_grad():
            emb = self.clap_model.get_text_embedding([text], use_tensor=True)
        return emb.to(self.device)

    def forward(self, mixture: torch.Tensor, embed_pos: torch.Tensor, embed_neg: torch.Tensor) -> torch.Tensor:
        """Separate mixture using precomputed text embeddings.

        Args:
            mixture: (B, T) input mixture at self.sample_rate
            embed_pos: (1, E) or (B, E) CLAP embedding for COI prompt
            embed_neg: (1, E) or (B, E) CLAP embedding for background prompt

        Returns:
            (B, 2, T) tensor with [COI, background]
        """
        B = mixture.shape[0]
        length = mixture.shape[-1]

        # STFT
        real, imag = self.stft(mixture)
        mag, cos, sin = magphase(real, imag)

        # Reset features and prime with mag (matches base CLAPSep)
        del self.features[:]
        self.features.append(mag)

        # Audio encoder
        mixture_resampled = self.resampler(mixture)
        if self.use_lora or not all(
            not p.requires_grad for p in self.audio_branch.parameters()
        ):
            self.audio_branch({"waveform": mixture_resampled})
        else:
            with torch.no_grad():
                self.audio_branch({"waveform": mixture_resampled})

        hidden_state = self.features[-1]
        skip_features = self.features[:-1]

        # Broadcast embeddings to batch size if needed
        if embed_pos.shape[0] == 1:
            embed_pos = embed_pos.expand(B, -1)
        if embed_neg.shape[0] == 1:
            embed_neg = embed_neg.expand(B, -1)

        # COI head: [pos, neg]
        embed_coi = torch.nn.functional.normalize(
            torch.cat([embed_pos, embed_neg], dim=-1), dim=-1
        )
        mask_coi = self.decoder(
            hidden_state=hidden_state, skip_features=skip_features, embed=embed_coi
        )
        pred_coi = self.wav_reconstruct(mask_coi, mag, cos, sin, length=length)

        # Background head: [neg, pos] (swapped). The base HTSAT_Decoder does
        # not mutate ``hidden_state`` or ``skip_features``, so we can call it
        # again without re-running the encoder (same pattern as COICLAPSepDecoder
        # in train_coi.py).
        embed_bg = torch.nn.functional.normalize(
            torch.cat([embed_neg, embed_pos], dim=-1), dim=-1
        )
        mask_bg = self.decoder(
            hidden_state=hidden_state, skip_features=skip_features, embed=embed_bg
        )
        pred_bg = self.wav_reconstruct(mask_bg, mag, cos, sin, length=length)

        # Clear hook buffer for next step
        del self.features[:]

        return torch.stack([pred_coi, pred_bg], dim=1)

    # -- Per-step prompt sampling --------------------------------------------

    def _sample_prompt_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one (pos, neg) prompt pair, return their CLAP embeddings."""
        # For now, single-class COI: use first prompt list. Extend to multi-class
        # by sampling per-batch class index if n_coi_classes > 1.
        coi_class_prompts = self.coi_text_prompts[
            random.randrange(len(self.coi_text_prompts))
        ]
        pos_text = random.choice(coi_class_prompts)
        neg_text = random.choice(self.background_text_prompts)
        return self._get_text_embedding(pos_text), self._get_text_embedding(neg_text)

    # -- Lightning hooks ------------------------------------------------------

    def training_step(self, batch, batch_idx):
        # WebDataset yields a plain sources tensor; file-based yields (mixture, sources)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            _, sources = batch
        else:
            sources = batch

        snr_range = (-5, 5)
        mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=False)

        embed_pos, embed_neg = self._sample_prompt_pair()
        separated = self(mixture, embed_pos, embed_neg)

        loss = self.criterion(separated, clean_wavs)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            _, sources = batch
        else:
            sources = batch

        snr_range = (-5, 5)
        mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=True)

        # Use the first (canonical) prompt pair for deterministic validation.
        with torch.no_grad():
            pos_text = self.coi_text_prompts[0][0]
            neg_text = self.background_text_prompts[0]
            embed_pos = self._get_text_embedding(pos_text)
            embed_neg = self._get_text_embedding(neg_text)
            separated = self(mixture, embed_pos, embed_neg)
            loss = self.criterion(separated, clean_wavs)
            coi_sisnr = sisnr(separated[:, 0], clean_wavs[:, 0]).mean()
            bg_sisnr = sisnr(separated[:, -1], clean_wavs[:, -1]).mean()

        self.log("val_loss", loss.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_coi_sisnr", coi_sisnr.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_bg_sisnr", bg_sisnr.item(), on_epoch=True, prog_bar=True, sync_dist=True)
        return {"val_loss": loss}

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=10, min_lr=1e-6
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
):
    """Main training function for text-prompt CLAPSep."""
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
    checkpoint_path = Path(checkpoint_dir) / f"text_prompt_{timestamp}"
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    # Load text prompts (and target_classes for WebDataset filtering) from config
    config_path = Path(__file__).parent / "training_config.yaml"
    coi_text_prompts: List[List[str]] = [["airplane engine", "aircraft noise"]]
    background_text_prompts: List[str] = ["ambient noise", "background sounds"]
    target_classes: List[List[str]] = []
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        coi_text_prompts = config.get("model", {}).get("coi_text_prompts", coi_text_prompts)
        background_text_prompts = config.get("model", {}).get(
            "background_text_prompts", background_text_prompts
        )
        target_classes = config.get("data", {}).get("target_classes", []) or []
        print(f"Loaded prompt config from {config_path}")
    else:
        print(f"  Warning: config file not found at {config_path}, using defaults")

    print("Text-prompt configuration:")
    for i, prompts in enumerate(coi_text_prompts):
        print(f"  COI class {i}: {prompts}")
    print(f"  Background:    {background_text_prompts}")

    # Dataloaders (mirror train_coi.py exactly)
    print("Creating dataloaders...")
    train_loader, _ = create_coi_dataloader(
        df_path=df_path,
        split="train",
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
        use_webdataset=use_webdataset,
        webdataset_path=webdataset_path if use_webdataset else None,
        target_classes=target_classes if use_webdataset else None,
        coi_ratio=0.25,
    )

    # Load pretrained CLAP
    print(f"Loading CLAP model from {clap_checkpoint}...")
    clap_model = _laion_clap.CLAP_Module(
        enable_fusion=False, amodel="HTSAT-base", device="cpu"
    )
    clap_model.load_ckpt(clap_checkpoint)

    # Decoder
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

    # Lightning module
    print("Creating model...")
    model = TextPromptCLAPSep(
        clap_model=clap_model,
        decoder_model=decoder,
        coi_text_prompts=coi_text_prompts,
        background_text_prompts=background_text_prompts,
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
        f"({100*trainable_params/total_params:.2f}%)"
    )
    if use_lora:
        print(f"  Using LoRA fine-tuning (rank={lora_rank})")
    elif freeze_encoder:
        print("  Encoder frozen (decoder-only training)")
    else:
        print("  Full encoder+decoder fine-tuning")

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
        description="Train CLAPSep with text-prompt conditioning for COI separation"
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
    )


if __name__ == "__main__":
    main()

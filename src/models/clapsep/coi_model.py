"""
COI (Class of Interest) adapted CLAPSep model classes — decoder and Lightning module.
"""

import copy
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
from torchlibrosa.stft import magphase

_src_root = Path(__file__).parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from common.coi_training import (
    COIWeightedLoss,
    prepare_batch,
    sisnr,
)

from .clapsep_utils import (
    apply_lora_to_model,
    configure_adamw_plateau,
    install_forward_hooks,
    make_stft_istft,
    wav_reconstruct,
)

try:
    import loralib as lora
    HAS_LORA = True
except ImportError:
    HAS_LORA = False
    print("Warning: loralib not available. Install with: pip install loralib")
    print("  LoRA fine-tuning will be disabled. Set use_lora=False or install loralib.")


class COICLAPSepDecoder(nn.Module):
    """Modified CLAPSep decoder for COI separation (2 sources: COI + background).

    Instead of using text embeddings, this model learns a fixed embedding
    for the COI class during training.
    """

    def __init__(
        self,
        decoder: nn.Module,
        embed_dim: int = 512,
        num_sources: int = 2,
    ):
        super().__init__()
        self.decoder = decoder
        self.num_sources = num_sources

        self.coi_embedding = nn.Parameter(torch.randn(1, embed_dim))
        self.bg_embedding = nn.Parameter(torch.randn(1, embed_dim))

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

        coi_embed = self.coi_embedding.expand(batch_size, -1)
        embed_coi = torch.nn.functional.normalize(
            torch.cat([coi_embed, self.bg_embedding.expand(batch_size, -1)], dim=-1),
            dim=-1,
        )
        mask_coi = self.decoder(
            hidden_state=hidden_state, skip_features=skip_features, embed=embed_coi
        )
        masks.append(mask_coi)

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

        self.audio_branch = copy.deepcopy(clap_model.model.audio_branch)
        del clap_model

        if freeze_encoder:
            for p in self.audio_branch.parameters():
                p.requires_grad = False
        elif use_lora:
            if not HAS_LORA:
                raise RuntimeError(
                    "loralib is required for LoRA fine-tuning. "
                    "Install with: pip install loralib\n"
                    "Or set use_lora=False to use full fine-tuning or frozen encoder."
                )
            print(f"Applying LoRA (rank={lora_rank}) to audio encoder...")
            self.audio_branch = apply_lora_to_model(self.audio_branch, rank=lora_rank)
            lora_params = sum(p.numel() for p in self.audio_branch.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.audio_branch.parameters())
            print(f"  LoRA trainable: {lora_params/1e6:.2f}M / {total_params/1e6:.2f}M total "
                  f"({100*lora_params/total_params:.2f}%)")
        else:
            print("Using full encoder fine-tuning (all parameters trainable)")

        self.decoder = decoder_model

        self.stft, self.istft = make_stft_istft(nfft=nfft, hop_length=320)

        self.resampler = torchaudio.transforms.Resample(sample_rate, resample_rate)

        self._feature_collector = install_forward_hooks(self.audio_branch)

        self.criterion = COIWeightedLoss(class_weight=class_weight)

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

        real, imag = self.stft(mixture)
        mag, cos, sin = magphase(real, imag)

        self._feature_collector.clear()
        self._feature_collector.append(mag)

        mixture_resampled = self.resampler(mixture)
        if self.use_lora or not all(
            not p.requires_grad for p in self.audio_branch.parameters()
        ):
            self.audio_branch({"waveform": mixture_resampled})
        else:
            with torch.no_grad():
                self.audio_branch({"waveform": mixture_resampled})

        hidden_state, skip_features = self._feature_collector.get()

        masks = self.decoder(
            hidden_state=hidden_state,
            skip_features=skip_features,
            batch_size=B,
        )

        separated = []
        for mask in masks:
            pred = wav_reconstruct(mask, mag, cos, sin, length=length, istft=self.istft)
            separated.append(pred)

        return torch.stack(separated, dim=1)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            _, sources = batch
        else:
            sources = batch

        snr_range = (-5, 5)
        mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=False)

        separated = self(mixture)

        loss = self.criterion(separated, clean_wavs)

        self.log(
            "train_loss", loss.item(), on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            _, sources = batch
        else:
            sources = batch

        snr_range = (-5, 5)
        mixture, clean_wavs = prepare_batch(sources, snr_range, deterministic=True)

        with torch.no_grad():
            separated = self(mixture)
            loss = self.criterion(separated, clean_wavs)

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

    def configure_optimizers(self):
        return configure_adamw_plateau(self, lr=self.lr)

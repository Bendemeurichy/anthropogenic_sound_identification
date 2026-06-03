"""TextPromptCLAPSep - Lightning module for COI separation with CLAP text-prompt conditioning."""

import random
import sys
from pathlib import Path
from typing import Any, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import copy
import torchaudio
from torchlibrosa.stft import magphase

# Add parent directories to path for imports
_src_root = Path(__file__).parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from common.coi_training import (
    COIWeightedLoss,
    prepare_batch,
    sisnr,
)

from models.clapsep.clapsep_utils import (
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
        self.stft, self.istft = make_stft_istft(nfft=nfft)

        # Resampler for CLAP input (expects 48kHz)
        self.resampler = torchaudio.transforms.Resample(sample_rate, resample_rate)

        # Forward hooks for skip connections
        self._feature_collector = install_forward_hooks(self.audio_branch)

        # Loss
        self.criterion = COIWeightedLoss(class_weight=class_weight)

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
        self._feature_collector.clear()
        self._feature_collector.append(mag)

        # Audio encoder
        mixture_resampled = self.resampler(mixture)
        if self.use_lora or not all(
            not p.requires_grad for p in self.audio_branch.parameters()
        ):
            self.audio_branch({"waveform": mixture_resampled})
        else:
            with torch.no_grad():
                self.audio_branch({"waveform": mixture_resampled})

        hidden_state, skip_features = self._feature_collector.get()

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
        pred_coi = wav_reconstruct(mask_coi, mag, cos, sin, length, istft=self.istft)

        # Background head: [neg, pos] (swapped). The base HTSAT_Decoder does
        # not mutate ``hidden_state`` or ``skip_features``, so we can call it
        # again without re-running the encoder (same pattern as COICLAPSepDecoder
        # in train.py).
        embed_bg = torch.nn.functional.normalize(
            torch.cat([embed_neg, embed_pos], dim=-1), dim=-1
        )
        mask_bg = self.decoder(
            hidden_state=hidden_state, skip_features=skip_features, embed=embed_bg
        )
        pred_bg = wav_reconstruct(mask_bg, mag, cos, sin, length, istft=self.istft)

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
        return configure_adamw_plateau(self, lr=self.lr)

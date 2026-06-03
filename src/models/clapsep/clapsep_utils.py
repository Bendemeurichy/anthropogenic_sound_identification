"""
Shared utilities for CLAPSep training scripts.

Consolidates duplicated code from coi_model.py and text_model.py:
    - LoRA helpers (set_module, apply_lora_to_model)
    - Forward hooks for skip connection capture
    - Waveform reconstruction from masks
    - STFT/iSTFT setup
    - Optimizer/scheduler configuration
"""

from typing import Any

import torch
import torch.nn as nn
from torchlibrosa import ISTFT, STFT
from torchlibrosa.stft import magphase


# =============================================================================
# LoRA utilities
# =============================================================================


def set_module(model, submodule_key, module):
    """Set a submodule in a model by its key path."""
    tokens = submodule_key.split(".")
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def apply_lora_to_model(model, rank: int = 8):
    """Replace linear layers in WindowAttention modules with LoRA layers.

    Args:
        model: The audio encoder model
        rank: LoRA rank (lower = fewer parameters, typically 4-16)

    Returns:
        Model with LoRA layers applied
    """
    try:
        import loralib as lora
    except ImportError:
        raise RuntimeError(
            "loralib is required for LoRA fine-tuning. "
            "Install with: pip install loralib"
        )

    for module_name, module in model.named_modules():
        if "WindowAttention" in str(type(module)):
            for layer_name, layer in module.named_modules():
                if isinstance(layer, torch.nn.Linear):
                    lora_layer = lora.Linear(
                        layer.in_features,
                        layer.out_features,
                        r=rank,
                        bias=hasattr(layer, "bias"),
                        merge_weights=False,
                    )
                    lora_layer.weight = layer.weight
                    if hasattr(layer, "bias"):
                        lora_layer.bias = layer.bias
                    full_path = (
                        f"{module_name}.{layer_name}"
                        if module_name
                        else layer_name
                    )
                    set_module(model, full_path, lora_layer)

    lora.mark_only_lora_as_trainable(model, bias="lora_only")
    return model


# =============================================================================
# Forward hooks
# =============================================================================


class _FeatureCollector:
    """Thread-safe collector for intermediate features captured by forward hooks.

    Hooks append to an internal list during forward passes.  Callers prime
    the list with ``append()``, run the encoder, then call ``get()`` to
    retrieve (hidden_state, skip_features) and atomically clear the list.
    """

    def __init__(self):
        self._features: list = []

    def clear(self) -> None:
        self._features.clear()

    def append(self, item) -> None:
        self._features.append(item)

    def get(self) -> tuple:
        """Return (hidden_state, skip_features) and clear the internal list."""
        hidden = self._features[-1]
        skip = self._features[:-1]
        self._features.clear()
        return hidden, skip

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx):
        return self._features[idx]


def install_forward_hooks(audio_branch: nn.Module) -> _FeatureCollector:
    """Install forward hooks to capture intermediate features from skip connections.

    Args:
        audio_branch: The audio encoder module

    Returns:
        _FeatureCollector that is populated by hooks during every forward pass.
    """
    collector = _FeatureCollector()

    def hook_append(_, __, output):
        collector.append(output)

    def hook_append_first(_, __, output):
        collector.append(output[0])

    def hook_spectrogram_padding(_, __, out):
        return torch.nn.functional.pad(out, (0, 0, 0, 1024 - out.size(2)))

    audio_branch.spectrogram_extractor.register_forward_hook(
        hook_spectrogram_padding
    )
    audio_branch.patch_embed.register_forward_hook(hook_append)
    for module in audio_branch.layers:
        module.register_forward_hook(hook_append_first)

    return collector


# =============================================================================
# Waveform reconstruction
# =============================================================================


def wav_reconstruct(mask, mag_x, cos_x, sin_x, length, istft: ISTFT):
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

    return istft(mag_y * cos_y, mag_y * sin_y, length=length)


# =============================================================================
# STFT/iSTFT setup
# =============================================================================


def make_stft_istft(
    nfft: int = 1024, hop_length: int = 320
) -> tuple[STFT, ISTFT]:
    """Create STFT and iSTFT transforms with consistent parameters."""
    stft_kwargs = dict(
        n_fft=nfft,
        hop_length=hop_length,
        win_length=nfft,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    )
    return STFT(**stft_kwargs), ISTFT(**stft_kwargs)


# =============================================================================
# Optimizer / scheduler
# =============================================================================


def configure_adamw_plateau(
    module: nn.Module, lr: float = 1e-4
) -> dict[str, Any]:
    """Return a standard AdamW + ReduceLROnPlateau config for Lightning."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, module.parameters()),
        lr=lr,
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

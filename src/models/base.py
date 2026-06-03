"""Base class for all separation model inference wrappers.

Provides a shared interface that all three separation models
(TUSS, SuDoRM-RF, CLAPSep) implement, plus common utilities.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import torch
import torchaudio


class BaseSeparator(ABC):
    """Abstract base class for audio separation model inference.

    Subclasses must implement:
        from_checkpoint()  — factory classmethod
        separate()          — separate from file path
        separate_waveform() — separate from waveform tensor
    """

    def __init__(self, model, config, device: str, sample_rate: int = 16000):
        self.model = model
        self.config = config
        self.device = device
        self.sample_rate = sample_rate

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls, checkpoint_path: Union[str, Path], device: str = None, **kwargs
    ) -> "BaseSeparator":
        """Load model from a checkpoint and return an inference wrapper."""
        ...

    @abstractmethod
    def separate(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Load audio from disk, run separation, return source tensor (n_src, T)."""
        ...

    @abstractmethod
    def separate_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run separation on an in-memory waveform, return source tensor (n_src, T)."""
        ...

    def get_coi_audio(self, sources: torch.Tensor) -> torch.Tensor:
        """Extract the Class of Interest audio from separated sources.

        Default implementation: returns the first non-background source.
        Override in subclasses for model-specific COI selection.
        """
        return sources[0]

    def get_background_audio(self, sources: torch.Tensor) -> torch.Tensor:
        """Extract the background audio from separated sources.

        Default implementation: returns the last source.
        Override in subclasses for model-specific background selection.
        """
        return sources[-1]

    def save_audio(self, waveform: torch.Tensor, path: Union[str, Path]) -> None:
        """Save waveform to file. Handles dimensionality and falls back to soundfile."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        try:
            torchaudio.save(str(path), waveform.cpu(), self.sample_rate)
        except Exception:
            try:
                import soundfile as sf
                sf.write(str(path), waveform.squeeze().cpu().numpy().T, self.sample_rate)
            except Exception:
                raise RuntimeError(
                    "Failed to save audio via torchaudio and 'soundfile' is not installed."
                )
        print(f"Saved: {path}")

    def _load_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Load audio file and return waveform as (C, T) tensor."""
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform

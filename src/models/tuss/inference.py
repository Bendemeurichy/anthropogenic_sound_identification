"""
Inference module for trained TUSS separation model.

This module provides inference capabilities for the TUSS model with prompt-based
separation. The model outputs are guaranteed to have consistent head assignment:
    - Head 0 (COI_HEAD_INDEX): Class of Interest (e.g., airplane) audio
    - Head 1 (BACKGROUND_HEAD_INDEX): Background audio

The interface mirrors SeparationInference from sudormrf so validation code works
without modification.

Usage:
    from models.tuss.inference import TUSSInference
    
    inferencer = TUSSInference.from_checkpoint(
        "path/to/checkpoint_dir",
        device="cuda",
        coi_prompt="airplane",
        bg_prompt="background"
    )
    
    # Separate a waveform
    sources = inferencer.separate_waveform(waveform)  # (2, T) tensor
    coi_audio = sources[0]  # airplane
    bg_audio = sources[1]   # background
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import yaml

# Add paths for imports
_SCRIPT_DIR = Path(__file__).parent.resolve()
_BASE_DIR = _SCRIPT_DIR / "base"
_SRC_DIR = _SCRIPT_DIR.parent.parent  # code/src

for _p in [str(_BASE_DIR), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nets.model_wrapper import SeparationModel

# Head index constants for consistent output access (matching sudormrf interface)
COI_HEAD_INDEX: int = 0
BACKGROUND_HEAD_INDEX: int = 1
NUM_SOURCES: int = 2


def robust_load_audio(path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    """Load audio robustly: try torchaudio first, then switch backend to
    soundfile, and finally fall back to the soundfile Python API.

    Returns (waveform, sample_rate) where waveform has shape (channels, frames).
    """
    p = str(path)
    # 1) Preferred: torchaudio.load (may use torchcodec internally)
    try:
        return torchaudio.load(p)
    except Exception as e:  # pragma: no cover - runtime environment dependent
        # Attempt to switch torchaudio backend to soundfile and retry
        try:
            torchaudio.set_audio_backend("soundfile")
            return torchaudio.load(p)
        except Exception:
            # Final fallback: use pysoundfile directly (pure-python)
            try:
                import soundfile as sf
            except Exception:
                raise RuntimeError(
                    "Failed to load audio with torchaudio and 'soundfile' is not installed. "
                    "Install pysoundfile (`pip install soundfile`) or ensure FFmpeg/torchcodec compatibility."
                ) from e

            data, sr = sf.read(p, always_2d=True)
            import numpy as _np

            # data: (frames, channels) -> waveform: (channels, frames)
            wav = torch.from_numpy(_np.asarray(data).T).to(torch.float32)
            return wav, int(sr)


class TUSSInference:
    """Inference wrapper for trained TUSS separation model.

    This class provides methods for loading a trained model and performing
    audio source separation with guaranteed head assignment:
        - output[:, COI_HEAD_INDEX, :] = COI (e.g., Airplane) audio
        - output[:, BACKGROUND_HEAD_INDEX, :] = Background audio

    The interface mirrors SeparationInference from sudormrf for compatibility
    with existing validation code.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sample_rate: int,
        segment_samples: int,
        device: str,
        coi_prompt: str,
        bg_prompt: str,
        config: Optional[dict] = None,
    ):
        """Initialize TUSSInference.

        Args:
            model: SeparationModel instance
            sample_rate: Model's native sample rate (typically 48000 Hz)
            segment_samples: Segment length in samples
            device: Device to run inference on
            coi_prompt: Prompt name for Class of Interest (e.g., "airplane")
            bg_prompt: Prompt name for background (e.g., "background")
            config: Optional training config dict for metadata
        """
        self.model = model.to(device)
        self.model.eval()
        # Freeze weights for inference
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.coi_prompt = coi_prompt
        self.bg_prompt = bg_prompt
        # Store config for downstream code (e.g., validation pipelines)
        self.config = config
        # Number of sources is always 2 for single-COI mode
        self.num_sources = NUM_SOURCES

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        coi_prompt: str = "airplane",
        bg_prompt: str = "background",
    ) -> "TUSSInference":
        """Load model from training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory or .pth file
            device: Device to load on (default: auto-detect cuda/cpu)
            coi_prompt: Prompt name for COI (default: "airplane")
            bg_prompt: Prompt name for background (default: "background")

        Returns:
            TUSSInference instance ready for separation
        """
        checkpoint_path = Path(checkpoint_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading TUSS checkpoint from: {checkpoint_path}")

        # Determine checkpoint format: directory or .pth file
        if checkpoint_path.is_dir():
            # Directory format: checkpoint_dir/hparams.yaml + checkpoint_dir/checkpoints/model.pth
            config_path = checkpoint_path / "hparams.yaml"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"hparams.yaml not found in checkpoint directory: {checkpoint_path}"
                )

            # Find model checkpoint file
            ckpt_dir = checkpoint_path / "checkpoints"
            if not ckpt_dir.exists():
                raise FileNotFoundError(
                    f"checkpoints subdirectory not found in: {checkpoint_path}"
                )

            # Look for model.pth (or .ckpt files, excluding last.ckpt)
            allowed_suffix = [".pth", ".ckpt"]
            ckpt_files = [
                p
                for p in ckpt_dir.iterdir()
                if p.suffix in allowed_suffix and p.name != "last.ckpt"
            ]

            if not ckpt_files:
                raise FileNotFoundError(
                    f"No checkpoint files found in {ckpt_dir}. "
                    f"Expected files with suffix {allowed_suffix}"
                )

            # Use first checkpoint (typically model.pth)
            ckpt_path = ckpt_files[0]
            print(f"  Loading weights from: {ckpt_path}")

        else:
            # Single file format: path/to/model.pth
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            ckpt_path = checkpoint_path
            # Config should be in parent.parent/hparams.yaml
            config_path = checkpoint_path.parent.parent / "hparams.yaml"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"hparams.yaml not found at: {config_path}. "
                    f"Expected structure: checkpoint_dir/hparams.yaml and "
                    f"checkpoint_dir/checkpoints/model.pth"
                )

        # Load hparams
        print(f"  Loading config from: {config_path}")
        with open(config_path) as f:
            hparams = yaml.safe_load(f)

        # Build model from hparams
        model = SeparationModel(
            encoder_name=hparams["encoder_name"],
            encoder_conf=hparams["encoder_conf"],
            decoder_name=hparams["decoder_name"],
            decoder_conf=hparams["decoder_conf"],
            separator_name=hparams["model_name"],
            separator_conf=hparams["model_conf"],
            css_conf=hparams["css_conf"],
            variance_normalization=hparams.get("variance_normalization", True),
        )

        # Load state dict
        print(f"  Loading model weights...")
        try:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            # Try with safe_globals for newer PyTorch versions
            msg = str(e)
            if "numpy._core.multiarray.scalar" in msg or "Unsupported global" in msg:
                try:
                    import numpy as _np
                    from torch.serialization import safe_globals

                    with safe_globals([_np._core.multiarray.scalar]):
                        state_dict = torch.load(
                            ckpt_path, map_location=device, weights_only=False
                        )
                except Exception:
                    raise
            else:
                raise

        # Strip 'model.' prefix that Lightning checkpoints add
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {
                (k[len("model.") :] if k.startswith("model.") else k): v
                for k, v in state_dict.items()
            }

        # Load with strict=False since we may have extra/missing prompt vectors
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (may be expected for new prompts): {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

        # Extract sample rate and segment length from hparams
        # TUSS typically uses 48kHz sample rate
        try:
            sample_rate = hparams.get("sample_rate", 48000)
            # Segment length may be in training config or top-level
            if "training" in hparams and "segment_length" in hparams["training"]:
                segment_length = hparams["training"]["segment_length"]
            elif "segment_length" in hparams:
                segment_length = hparams["segment_length"]
            else:
                segment_length = 4.0  # TUSS default from training_config.yaml
            segment_samples = int(sample_rate * segment_length)
        except Exception:
            # Fallback defaults
            sample_rate = 48000
            segment_samples = 192000  # 4 seconds at 48kHz

        print(
            f"  Model config: sample_rate={sample_rate} Hz, "
            f"segment_length={segment_samples / sample_rate:.2f}s "
            f"({segment_samples} samples)"
        )
        print(f"  Using prompts: COI='{coi_prompt}', Background='{bg_prompt}'")

        return cls(
            model=model,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            device=device,
            coi_prompt=coi_prompt,
            bg_prompt=bg_prompt,
            config=hparams,
        )

    @torch.inference_mode()
    def separate(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Separate audio file into component sources.

        Args:
            audio_path: Path to audio file

        Returns:
            sources: (n_sources, T) tensor where:
                     sources[COI_HEAD_INDEX, :] = COI (e.g., Airplane) audio
                     sources[BACKGROUND_HEAD_INDEX, :] = Background audio
        """
        waveform, sr = robust_load_audio(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        original_length = waveform.shape[0]

        # Process in overlapping chunks for long audio
        if waveform.shape[0] > self.segment_samples:
            return self._separate_long(waveform, original_length)

        # Pad if needed
        if waveform.shape[0] < self.segment_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_samples - waveform.shape[0])
            )

        return self._separate_segment(waveform)[:, :original_length]

    @torch.inference_mode()
    def separate_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Separate a waveform tensor directly.

        Args:
            waveform: Input waveform tensor (T,) or (1, T)

        Returns:
            sources: (n_sources, T) tensor where:
                     sources[COI_HEAD_INDEX, :] = COI audio
                     sources[BACKGROUND_HEAD_INDEX, :] = Background audio
        """
        if waveform.dim() == 2:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

        original_length = waveform.shape[0]

        # Process in overlapping chunks for long audio
        if waveform.shape[0] > self.segment_samples:
            return self._separate_long(waveform, original_length)

        # Pad if needed
        if waveform.shape[0] < self.segment_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_samples - waveform.shape[0])
            )

        return self._separate_segment(waveform)[:, :original_length]

    def _separate_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """Separate a single segment.

        TUSS uses variance normalization (normalize by std) matching training.
        The model expects normalized input and outputs normalized sources.

        Args:
            segment: Input waveform (T,) of length segment_samples

        Returns:
            sources: (n_sources, T) tensor [COI, background]
        """
        # Variance normalization (matching TUSS training)
        mean = segment.mean()
        std = segment.std() + 1e-8

        # Normalize input (zero-mean, unit-variance)
        x = ((segment - mean) / std).unsqueeze(0).to(self.device)  # (1, T)

        # Build prompts list: [[coi_prompt, bg_prompt]]
        # TUSS model expects prompts as List[List[str]] where outer list is batch
        prompts = [[self.coi_prompt, self.bg_prompt]]

        # Run model
        estimates, *_ = self.model(x, prompts)  # (1, n_sources, T)

        # Scale outputs by input std to restore reasonable amplitude
        # (model outputs normalized sources, we rescale to match input level)
        sources = estimates[0].cpu() * std  # (n_sources, T)

        return sources

    def _separate_long(
        self, waveform: torch.Tensor, original_length: int
    ) -> torch.Tensor:
        """Process long audio with overlap-add.

        Uses Hann windowing with 50% overlap for smooth transitions.

        Args:
            waveform: Input waveform (T,) where T > segment_samples
            original_length: Original length to trim output to

        Returns:
            sources: (n_sources, T) tensor
        """
        hop = self.segment_samples // 2
        window = torch.hann_window(self.segment_samples)

        # Initialize output buffers
        output = torch.zeros(NUM_SOURCES, original_length)
        weight = torch.zeros(original_length)

        # Process overlapping segments
        for start in range(0, waveform.shape[0], hop):
            chunk = waveform[start : start + self.segment_samples]
            if chunk.shape[0] < self.segment_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, self.segment_samples - chunk.shape[0])
                )

            # Separate this segment
            sources = self._separate_segment(chunk)  # (n_sources, segment_samples)

            end = min(start + self.segment_samples, original_length)
            length = end - start

            # Add windowed segment (broadcast window across sources)
            output[:, start:end] += sources[:, :length] * window[:length]
            weight[start:end] += window[:length]

        # Normalize by overlap weight
        return output / (weight + 1e-8)

    def get_coi_audio(self, sources: torch.Tensor) -> torch.Tensor:
        """Extract the Class of Interest audio from separated sources.

        Args:
            sources: Output from separate() with shape (n_sources, T)

        Returns:
            COI audio tensor with shape (T,)
        """
        return sources[COI_HEAD_INDEX]

    def get_background_audio(self, sources: torch.Tensor) -> torch.Tensor:
        """Extract the background audio from separated sources.

        Args:
            sources: Output from separate() with shape (n_sources, T)

        Returns:
            Background audio tensor with shape (T,)
        """
        return sources[BACKGROUND_HEAD_INDEX]

    def save_audio(self, waveform: torch.Tensor, path: Union[str, Path]):
        """Save waveform to file.

        Args:
            waveform: Audio tensor (T,) or (1, T)
            path: Output file path
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Try torchaudio.save first, fall back to soundfile
        try:
            torchaudio.save(str(path), waveform, self.sample_rate)
            print(f"Saved: {path}")
            return
        except Exception:
            try:
                import soundfile as sf
            except Exception:
                raise RuntimeError(
                    "Failed to save audio via torchaudio and 'soundfile' is not installed. "
                    "Install pysoundfile (`pip install soundfile`) or fix torchcodec/FFmpeg."
                )

            # waveform: (channels, frames) -> data: (frames, channels)
            data = waveform.detach().cpu().numpy().T
            sf.write(str(path), data, self.sample_rate)
            print(f"Saved: {path} (via soundfile)")

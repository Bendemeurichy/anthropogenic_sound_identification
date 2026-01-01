"""
Inference module for trained SuDORMRF separation model.

Usage:
    python inference.py --checkpoint path/to/best_model.pt --audio path/to/audio.wav
"""

import sys
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional, Union
import argparse
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.sudormrf.config import Config
from src.models.sudormrf.train import create_model

# Check for environment variable to use old separation head
USE_OLD_SEPARATION_HEAD = os.environ.get("USE_OLD_SEPARATION_HEAD", "0") == "1"


class SeparationInference:
    """Inference wrapper for trained SuDORMRF separation model."""

    def __init__(
        self,
        model: torch.nn.Module,
        sample_rate: int,
        segment_samples: int,
        device: str,
    ):
        self.model = model.to(device)
        self.model.eval()
        # Freeze weights for inference
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Union[str, Path], device: Optional[str] = None
    ):
        """Load model from training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading checkpoint from: {checkpoint_path}")
        # Try loading the checkpoint allowing pickled objects. Some saved
        # checkpoints require `weights_only=False` or an allowlist for
        # numpy scalars introduced in newer PyTorch versions.
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
        except Exception as e:
            # If torch raises about unsafe globals (e.g. numpy scalar), try
            # using the `safe_globals` context manager to allowlist the
            # offending global. Only do this for trusted checkpoints.
            msg = str(e)
            if "numpy._core.multiarray.scalar" in msg or "Unsupported global" in msg:
                try:
                    import numpy as _np
                    from torch.serialization import safe_globals

                    with safe_globals([_np._core.multiarray.scalar]):
                        checkpoint = torch.load(
                            checkpoint_path, map_location=device, weights_only=False
                        )
                except Exception:
                    raise
            else:
                # Re-raise original exception for unexpected failures
                raise

        # Determine checkpoint layout and extract config/state_dict/model
        config = None
        state_dict = None
        model_obj = None

        if isinstance(checkpoint, dict):
            # Typical training checkpoint: contains both config and model_state_dict
            if "config" in checkpoint and "model_state_dict" in checkpoint:
                cfg_obj = checkpoint["config"]
                if isinstance(cfg_obj, Config):
                    config = cfg_obj
                elif isinstance(cfg_obj, dict):
                    config = Config.from_dict(cfg_obj)
                else:
                    try:
                        config = Config.from_dict(dict(cfg_obj))
                    except Exception:
                        config = None

                state_dict = checkpoint["model_state_dict"]

            # Some checkpoints use 'state_dict' key
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            # Sometimes checkpoints are saved as the raw state_dict
            else:
                # Heuristic: if all values are tensors, treat as state_dict
                if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    state_dict = checkpoint
                else:
                    # Try to fallback to a config.yaml alongside the checkpoint file
                    cfg_path = checkpoint_path.parent / "config.yaml"
                    if cfg_path.exists():
                        config = Config.from_yaml(str(cfg_path))
                        # attempt to pick common keys for state_dict
                        if "model_state_dict" in checkpoint:
                            state_dict = checkpoint["model_state_dict"]
                        elif "state_dict" in checkpoint:
                            state_dict = checkpoint["state_dict"]
                        else:
                            # maybe checkpoint contains a wrapped object; leave to later
                            state_dict = None
                    else:
                        raise ValueError(
                            "Unsupported checkpoint format. Expected training checkpoint or state_dict."
                        )

        else:
            # Loaded object may be a full model
            if isinstance(checkpoint, torch.nn.Module):
                model_obj = checkpoint
            else:
                # Try loading a config.yaml next to the checkpoint
                cfg_path = checkpoint_path.parent / "config.yaml"
                if cfg_path.exists():
                    config = Config.from_yaml(str(cfg_path))
                else:
                    raise ValueError("Unsupported checkpoint format.")

        # If config not found in the checkpoint, try config.yaml in same folder
        if config is None:
            cfg_path = checkpoint_path.parent / "config.yaml"
            if cfg_path.exists():
                config = Config.from_yaml(str(cfg_path))

        # Build model (or use provided model object) and load weights
        if model_obj is not None:
            model = model_obj
        else:
            if config is None:
                raise ValueError(
                    "No config found in checkpoint and no config.yaml in checkpoint folder."
                )
            model = create_model(config)

        # Load state dict with error handling (if we have one)
        if state_dict is not None:
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(
                    "Warning: Error(s) in loading state_dict. Attempting to resolve mismatches."
                )

                # Filter state_dict to match model's keys
                model_state_dict = model.state_dict()
                filtered_state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if k in model_state_dict and model_state_dict[k].shape == v.shape
                }

                missing_keys = set(model_state_dict.keys()) - set(
                    filtered_state_dict.keys()
                )
                unexpected_keys = set(state_dict.keys()) - set(
                    filtered_state_dict.keys()
                )

                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")

                model.load_state_dict(filtered_state_dict, strict=False)

        # Compatibility: some checkpoints / model objects may lack attributes
        # expected by newer code (e.g., `n_least_samples_req`). Compute and
        # set a sane default on the model object to avoid AttributeError at
        # inference time without modifying the base model source.
        try:
            if not hasattr(model, "n_least_samples_req"):
                enc_k = getattr(model, "enc_kernel_size", None)
                up_d = getattr(model, "upsampling_depth", None)

                if enc_k is None and hasattr(model, "encoder"):
                    try:
                        enc_k = model.encoder.kernel_size
                        if isinstance(enc_k, (list, tuple)):
                            enc_k = enc_k[0]
                    except Exception:
                        enc_k = 21

                if up_d is None:
                    up_d = 4

                try:
                    n_least = (int(enc_k) // 2) * (2 ** int(up_d))
                except Exception:
                    n_least = 1024

                try:
                    model.n_least_samples_req = n_least
                except Exception:
                    pass
        except Exception:
            # Best-effort only; do not fail checkpoint loading for unexpected
            # attribute access issues.
            pass

        # If config still None, ensure values for return
        if config is None:
            # As a last resort, try to infer sample_rate/segment_samples from defaults
            sample_rate = getattr(model, "sample_rate", 16000)
            segment_samples = getattr(model, "segment_samples", 16000 * 5)
        else:
            sample_rate = (
                config.data.sample_rate
                if hasattr(config, "data")
                else config.sample_rate
            )
            # support older config shapes
            try:
                segment_length = config.data.segment_length
            except Exception:
                segment_length = getattr(config, "segment_length", 5.0)
            segment_samples = int(sample_rate * segment_length)

        return cls(
            model=model,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            device=device,
        )

    @torch.inference_mode()
    def separate(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Separate audio into component sources.
        Returns:
            sources: (n_sources, T) tensor
        """
        waveform, sr = torchaudio.load(audio_path)
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

    def _separate_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """Separate a single segment.

        The model was trained with independently normalized sources, so outputs
        are normalized waveforms. We rescale using mixture statistics to get
        outputs in a reasonable amplitude range matching the input.

        Returns: (n_sources, T)
        """
        mean = segment.mean()
        std = segment.std() + 1e-8

        # Normalize input (zero-mean, unit-variance) - matches training
        x = ((segment - mean) / std).unsqueeze(0).unsqueeze(0).to(self.device)

        estimates = self.model(x)
        # estimates shape: (1, n_sources, T) - each source is normalized

        # The model outputs normalized sources. To get usable audio:
        # Scale by mixture std to restore reasonable amplitude.
        # Don't add mean since sources should be zero-mean signals.
        sources = estimates[0].cpu() * std
        return sources

    def _separate_long(
        self, waveform: torch.Tensor, original_length: int
    ) -> torch.Tensor:
        """Process long audio with overlap-add.
        Returns: (n_sources, T)
        """
        hop = self.segment_samples // 2
        window = torch.hann_window(self.segment_samples)

        # Determine number of sources from a dummy pass or model attribute
        n_sources = self.model.num_sources

        output = torch.zeros(n_sources, original_length)
        weight = torch.zeros(original_length)

        for start in range(0, waveform.shape[0], hop):
            chunk = waveform[start : start + self.segment_samples]
            if chunk.shape[0] < self.segment_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, self.segment_samples - chunk.shape[0])
                )

            # sources shape: (n_sources, segment_samples)
            sources = self._separate_segment(chunk)

            end = min(start + self.segment_samples, original_length)
            length = end - start

            # Add windowed segment
            # sources[:, :length] shape is (n_sources, length)
            # window[:length] shape is (length) -> broadcasts to (n_sources, length)
            output[:, start:end] += sources[:, :length] * window[:length]
            weight[start:end] += window[:length]

        return output / (weight + 1e-8)

    def save_audio(self, waveform: torch.Tensor, path: Union[str, Path]):
        """Save waveform to file."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(str(path), waveform, self.sample_rate)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run SuDORMRF separation inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to best_model.pt"
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument(
        "--output-dir", type=str, help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (default: auto)"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_dir = Path(args.output_dir) if args.output_dir else audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    inferencer = SeparationInference.from_checkpoint(
        args.checkpoint, device=args.device
    )
    sources = inferencer.separate(audio_path)

    # Save all sources
    # Assuming last source is background, others are COI classes
    n_sources = sources.shape[0]
    for i in range(n_sources - 1):
        inferencer.save_audio(
            sources[i], output_dir / f"{audio_path.stem}_coi_class_{i}.wav"
        )

    inferencer.save_audio(sources[-1], output_dir / f"{audio_path.stem}_background.wav")

    print(f"Done! Extracted {n_sources} sources.")


if __name__ == "__main__":
    main()

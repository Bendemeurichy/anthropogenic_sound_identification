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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.sudormrf.config import Config
from src.models.sudormrf.train import create_model


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
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Handle multiple checkpoint formats:
        # - dict with keys: 'config' and 'model_state_dict' (training checkpoint)
        # - saved torch.nn.Module instance (model was saved directly)
        if isinstance(checkpoint, dict):
            cfg = checkpoint.get("config", {})
            config = Config.from_dict(cfg)
            config.training.device = device

            model = create_model(config)
            state_dict = checkpoint["model_state_dict"]
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                # Handle common key-naming mismatches for wrapped separation heads.
                msg = str(e)
                # If checkpoint used a bare COISeparationHead as `mask_net` (keys
                # like `mask_net.coi_branch.*`) while the model expects a
                # sequential wrapper (keys like `mask_net.1.coi_branch.*`),
                # remap those keys and retry loading with strict=False so that
                # missing top-level PReLU params are left at their initialized
                # values.
                if any(
                    ("mask_net.coi_branch" in k)
                    or ("mask_net.background_branch" in k)
                    or ("mask_net.shared_conv" in k)
                    for k in state_dict.keys()
                ):
                    new_state = {}
                    for k, v in state_dict.items():
                        if "mask_net." in k:
                            # Keep any prefix (e.g., "_orig_mod.") and insert the
                            # sequential index `1` after `mask_net.` so that keys
                            # like "mask_net.coi_branch..." or
                            # "_orig_mod.mask_net.coi_branch..." become
                            # "mask_net.1.coi_branch..." or
                            # "_orig_mod.mask_net.1.coi_branch..." respectively.
                            prefix, rest = k.split("mask_net.", 1)
                            first = rest.split(".")[0]
                            if first in {
                                "coi_branch",
                                "background_branch",
                                "shared_conv",
                            }:
                                new_key = prefix + "mask_net.1." + rest
                                new_state[new_key] = v
                                continue
                        new_state[k] = v

                    try:
                        model.load_state_dict(new_state, strict=False)
                        print(
                            "Loaded checkpoint after remapping legacy `mask_net` keys (non-strict)."
                        )
                    except Exception:
                        # Fall back to less strict loading of original dict
                        model.load_state_dict(state_dict, strict=False)
                        print(
                            "Loaded checkpoint with non-strict fallback (no remapping applied)."
                        )
                else:
                    # Re-raise if it's an unrelated issue
                    raise

            print(
                f"Loaded model from epoch {checkpoint.get('epoch', '?')}, val_loss: {checkpoint.get('val_loss', '?'):.4f}"
            )
            sample_rate = config.data.sample_rate
            segment_samples = int(config.data.segment_length * config.data.sample_rate)
        elif isinstance(checkpoint, torch.nn.Module):
            # checkpoint is the model object itself
            model = checkpoint
            # Ensure compatibility attributes expected by downstream code
            try:
                if not hasattr(model, "n_least_samples_req"):
                    # infer kernel size
                    k = getattr(model, "enc_kernel_size", None)
                    if k is None:
                        enc = getattr(model, "encoder", None)
                        if enc is not None and hasattr(enc, "kernel_size"):
                            k = enc.kernel_size
                            if isinstance(k, (tuple, list)):
                                k = k[0]

                    # infer upsampling depth
                    ups = getattr(model, "upsampling_depth", None)
                    if ups is None:
                        sm = getattr(model, "sm", None)
                        try:
                            first = sm[0]
                            ups = getattr(first, "depth", None)
                        except Exception:
                            ups = None

                    if k is not None and ups is not None:
                        model.n_least_samples_req = (int(k) // 2) * (2 ** int(ups))
            except Exception:
                pass
            # try to infer sample_rate/segment from model attributes if present,
            # otherwise fall back to sensible defaults
            sample_rate = getattr(model, "sample_rate", 16000)
            segment_samples = int(getattr(model, "segment_samples", sample_rate * 4))
            print("Loaded model object from checkpoint file.")
        else:
            # Unknown format — try to treat as state_dict
            try:
                config = Config.from_dict({})
                model = create_model(config)
                model.load_state_dict(checkpoint)
                sample_rate = config.data.sample_rate
                segment_samples = int(
                    config.data.segment_length * config.data.sample_rate
                )
                print("Loaded model from raw state_dict checkpoint.")
            except Exception:
                raise RuntimeError("Unrecognized checkpoint format")

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

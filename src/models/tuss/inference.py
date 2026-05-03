"""
Inference module for trained TUSS separation model.

This module provides inference capabilities for the TUSS model with prompt-based
separation.  The model supports an arbitrary number of sources determined by the
trained prompts:

    prompts_list = coi_prompts + [bg_prompt]
    output shape: (n_sources, T)  where n_sources = len(prompts_list)

Output head assignment matches the order of ``prompts_list``:
    - output[i, :] = audio for coi_prompts[i]
    - output[-1, :] = background audio

Use ``TUSSInference.target_coi_index`` to retrieve the correct head for the
requested COI rather than relying on the legacy ``COI_HEAD_INDEX`` constant.

Usage:
    from models.tuss.inference import TUSSInference

    inferencer = TUSSInference.from_checkpoint(
        "path/to/checkpoint_dir",
        device="cuda",
        coi_prompt="birds",   # must match the exact trained prompt name
        bg_prompt="background"
    )

    # Separate a waveform
    sources = inferencer.separate_waveform(waveform)  # (n_sources, T) tensor
    coi_audio = sources[inferencer.target_coi_index]
    bg_audio   = sources[-1]
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

from common.audio_utils import resample_with_padding
from nets.model_wrapper import SeparationModel

# Legacy head-index constants kept for backward compatibility with code that
# imports them.  For multi-COI checkpoints use TUSSInference.target_coi_index
# instead; background is always at sources[-1] regardless of how many COIs
# were trained.
COI_HEAD_INDEX: int = 0       # only correct for single-COI (airplane) checkpoints
BACKGROUND_HEAD_INDEX: int = 1  # only correct for 2-source (single-COI) checkpoints
NUM_SOURCES: int = 2            # only correct for 2-source (single-COI) checkpoints


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
        - output[:num_cois, :] = COI (e.g., Airplane, Bird) audio
        - output[-1, :] = Background audio

    The interface mirrors SeparationInference from sudormrf for compatibility
    with existing validation code.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sample_rate: int,
        segment_samples: int,
        device: str,
        coi_prompt: Union[str, List[str]],
        bg_prompt: str,
        config: Optional[dict] = None,
        target_coi: Optional[str] = None,
    ):
        """Initialize TUSSInference.

        Args:
            model: SeparationModel instance
            sample_rate: Model's native sample rate (typically 48000 Hz)
            segment_samples: Segment length in samples
            device: Device to run inference on
            coi_prompt: Prompt name(s) for Class of Interest (e.g., "airplane" or ["airplane", "bird"])
            bg_prompt: Prompt name for background (e.g., "background")
            config: Optional training config dict for metadata
            target_coi: The specific COI name to extract at inference time.  Used
                to resolve ``target_coi_index`` via fuzzy name matching against
                ``coi_prompts``.  If *None* the first element of ``coi_prompts``
                is used.
        """
        self.model = model.to(device)
        self.model.eval()
        # Freeze weights for inference
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        
        # Support multiple COIs
        self.coi_prompts = [coi_prompt] if isinstance(coi_prompt, str) else list(coi_prompt)
        self.coi_prompt = self.coi_prompts[0] if len(self.coi_prompts) == 1 else self.coi_prompts # for backwards compat
        self.bg_prompt = bg_prompt
        self.prompts_list = self.coi_prompts + [self.bg_prompt]
        
        # Store config for downstream code (e.g., validation pipelines)
        self.config = config
        
        # Dynamic number of sources
        self.num_sources = len(self.prompts_list)

        # Resolve which COI head to return at inference time.
        # ``target_coi`` may differ from the exact string in ``coi_prompts``
        # (e.g. caller passes "bird" but checkpoint was trained with "birds"),
        # so we use fuzzy prefix/substring matching with an exact-match preference.
        _target = target_coi if target_coi is not None else self.coi_prompts[0]
        self.target_coi_index: int = self._resolve_coi_index(_target)
        
        # Verify all model components are on the correct device
        self._verify_device_placement()
    
    def _resolve_coi_index(self, requested: str) -> int:
        """Resolve the COI head index by exact lower-case match against ``coi_prompts``.

        Args:
            requested: The COI name the caller wants (must match a trained prompt exactly).

        Returns:
            Index into ``self.coi_prompts`` for the requested COI.

        Raises:
            ValueError: If ``requested`` does not match any trained prompt.
        """
        req = requested.lower()
        prompts_lower = [p.lower() for p in self.coi_prompts]

        if req in prompts_lower:
            return prompts_lower.index(req)

        raise ValueError(
            f"Requested COI '{requested}' not found in trained coi_prompts "
            f"{self.coi_prompts}. Pass the exact trained prompt name."
        )

    def _verify_device_placement(self):
        """Verify all model parameters and buffers are on self.device."""
        devices_found = set()
        for name, param in self.model.named_parameters():
            devices_found.add(str(param.device))
        for name, buf in self.model.named_buffers():
            devices_found.add(str(buf.device))
        
        if len(devices_found) > 1:
            print(f"WARNING: Model has parameters/buffers on multiple devices: {devices_found}")
            print(f"  Expected all on: {self.device}")
            # Force move everything to the correct device
            self.model = self.model.to(self.device)
            print(f"  Forced model.to({self.device})")
        elif devices_found and str(self.device) not in str(list(devices_found)[0]):
            print(f"WARNING: Model on {devices_found} but expected {self.device}")
            self.model = self.model.to(self.device)
            print(f"  Forced model.to({self.device})")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        coi_prompt: Union[str, List[str]] = "airplane",
        bg_prompt: str = "background",
    ) -> "TUSSInference":
        """Load model from training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory or .pth file
            device: Device to load on (default: auto-detect cuda/cpu)
            coi_prompt: The desired COI prompt string (or list) used for head
                selection at inference time.  When a training checkpoint
                contains a ``coi_prompts`` list the model **always** runs with
                the full trained prompt list; ``coi_prompt`` is only used to
                determine *which head* to extract via fuzzy name matching.
            bg_prompt: Prompt name for background (default: "background")

        Returns:
            TUSSInference instance ready for separation
        """
        # Preserve the caller's intent for head selection before any overrides.
        if isinstance(coi_prompt, str):
            requested_coi: str = coi_prompt
        elif coi_prompt:
            requested_coi = coi_prompt[0]
        else:
            requested_coi = "airplane"
        checkpoint_path = Path(checkpoint_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading TUSS checkpoint from: {checkpoint_path}")

        # Determine checkpoint format: directory or .pt/.pth file
        if checkpoint_path.is_dir():
            # Directory format: checkpoint_dir/config.yaml + checkpoint_dir/best_model.pt
            config_path = checkpoint_path / "config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"config.yaml not found in checkpoint directory: {checkpoint_path}"
                )

            # Find model checkpoint file directly in checkpoint directory
            ckpt_path = checkpoint_path / "best_model.pt"
            if not ckpt_path.exists():
                # Fallback: look for any .pt or .pth file
                allowed_suffix = [".pt", ".pth"]
                ckpt_files = [
                    p
                    for p in checkpoint_path.iterdir()
                    if p.suffix in allowed_suffix
                ]
                if not ckpt_files:
                    raise FileNotFoundError(
                        f"No checkpoint files found in {checkpoint_path}. "
                        f"Expected best_model.pt or files with suffix {allowed_suffix}"
                    )
                ckpt_path = ckpt_files[0]

            print(f"  Loading weights from: {ckpt_path}")

        else:
            # Single file format: path/to/model.pt
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            ckpt_path = checkpoint_path
            # Config should be in same directory as the checkpoint file
            config_path = checkpoint_path.parent / "config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"config.yaml not found at: {config_path}. "
                    f"Expected structure: checkpoint_dir/config.yaml and "
                    f"checkpoint_dir/best_model.pt"
                )

        # Load config
        print(f"  Loading config from: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Extract nested config sections
        model_conf = config.get("model", {})
        data_conf = config.get("data", {})

        # Extract sample rate and segment length from data config
        sample_rate = data_conf.get("sample_rate", 48000)
        segment_length = data_conf.get("segment_length", 4.0)
        segment_samples = int(sample_rate * segment_length)

        # The saved config.yaml only has training config, not full model architecture.
        # We need to load the model architecture from the pretrained model's hparams.yaml
        pretrained_path = model_conf.get("pretrained_path")
        if pretrained_path:
            # Resolve relative path from checkpoint directory or script directory
            pretrained_resolved = config_path.parent / pretrained_path
            if not pretrained_resolved.exists():
                pretrained_resolved = _SCRIPT_DIR / pretrained_path
            if not pretrained_resolved.exists():
                raise FileNotFoundError(
                    f"Pretrained model not found at {pretrained_path}. "
                    f"Tried: {config_path.parent / pretrained_path} and {_SCRIPT_DIR / pretrained_path}"
                )
            
            pretrained_hparams_path = pretrained_resolved / "hparams.yaml"
            if not pretrained_hparams_path.exists():
                raise FileNotFoundError(
                    f"hparams.yaml not found in pretrained model: {pretrained_resolved}"
                )
            
            print(f"  Loading model architecture from: {pretrained_hparams_path}")
            with open(pretrained_hparams_path) as f:
                hparams = yaml.safe_load(f)
            
            # Use architecture from pretrained hparams
            encoder_name = hparams["encoder_name"]
            encoder_conf = hparams["encoder_conf"]
            decoder_name = hparams["decoder_name"]
            decoder_conf = hparams["decoder_conf"]
            separator_name = hparams["model_name"]
            separator_conf = hparams["model_conf"]
            css_conf = hparams["css_conf"]
            variance_normalization = hparams.get("variance_normalization", True)
        else:
            # No pretrained path - use config directly (requires full separator_conf)
            encoder_name = model_conf.get("encoder_name", "stft")
            encoder_conf = model_conf.get("encoder_conf", {})
            decoder_name = model_conf.get("decoder_name", "stft")
            decoder_conf = model_conf.get("decoder_conf", {})
            separator_name = model_conf.get("separator_name", "tuss")
            separator_conf = model_conf.get("separator_conf", {})
            css_conf_raw = model_conf.get("css_conf", {})
            css_conf = {
                "segment_size": css_conf_raw.get("segment_size", segment_length),
                "shift_size": css_conf_raw.get("shift_size", segment_length / 2),
                "normalize_segment_scale": css_conf_raw.get("normalize_segment_scale", True),
                "solve_perm": css_conf_raw.get("solve_perm", False),
                "sample_rate": css_conf_raw.get("sample_rate", sample_rate),
            }
            variance_normalization = model_conf.get("variance_normalization", True)

        # Use config prompts as defaults if not provided by caller
        config_coi_prompts = model_conf.get("coi_prompts", ["airplane"])
        config_bg_prompt = model_conf.get("bg_prompt", "background")
        
        if isinstance(coi_prompt, str) and coi_prompt == "airplane" and config_coi_prompts:
            coi_prompt = config_coi_prompts
        if bg_prompt == "background" and config_bg_prompt:
            bg_prompt = config_bg_prompt

        # Build model from architecture config
        model = SeparationModel(
            encoder_name=encoder_name,
            encoder_conf=encoder_conf,
            decoder_name=decoder_name,
            decoder_conf=decoder_conf,
            separator_name=separator_name,
            separator_conf=separator_conf,
            css_conf=css_conf,
            variance_normalization=variance_normalization,
        )

        # Move model to inference device BEFORE loading state dict
        # This ensures model structure and state dict are on the same device
        model = model.to(device)

        # Load checkpoint
        print(f"  Loading model weights...")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            # Try with safe_globals for newer PyTorch versions
            msg = str(e)
            if "numpy._core.multiarray.scalar" in msg or "Unsupported global" in msg:
                try:
                    import numpy as _np
                    from torch.serialization import safe_globals

                    with safe_globals([_np._core.multiarray.scalar]):
                        checkpoint = torch.load(
                            ckpt_path, map_location=device, weights_only=False
                        )
                except Exception:
                    raise
            else:
                raise

        # Handle nested checkpoint format from train.py
        # The checkpoint may contain: model_state_dict, optimizer_state_dict, etc.
        # or it may be a raw state dict
        if "model_state_dict" in checkpoint:
            print(f"  Detected training checkpoint format")
            state_dict = checkpoint["model_state_dict"]
            # Extract prompt info if available
            ckpt_coi_prompts = checkpoint.get("coi_prompts", [])
            ckpt_bg_prompt = checkpoint.get("bg_prompt", "")
            
            # Always use the full trained prompt list from the checkpoint so
            # the model runs with the embeddings it was actually trained on.
            # The caller's original string (requested_coi) is used only for
            # head selection via target_coi_index in __init__.
            if ckpt_coi_prompts:
                coi_prompt = ckpt_coi_prompts
                
            if ckpt_bg_prompt and bg_prompt == config_bg_prompt:
                bg_prompt = ckpt_bg_prompt
            print(f"  Checkpoint prompts: COI={ckpt_coi_prompts}, BG={ckpt_bg_prompt}")
        else:
            state_dict = checkpoint

        # Strip 'model.' prefix that Lightning checkpoints add
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {
                (k[len("model.") :] if k.startswith("model.") else k): v
                for k, v in state_dict.items()
            }

        # ------------------------------------------------------------------ #
        # Inject new prompt parameters before loading state dict.            #
        # The ParameterDict won't automatically create keys from state_dict, #
        # so we must add them explicitly for any prompts saved in the ckpt.  #
        # ------------------------------------------------------------------ #
        prompts_dict = model.separator.prompts
        prompt_prefix = "separator.prompts."
        for key in state_dict.keys():
            if key.startswith(prompt_prefix):
                # Extract prompt name (use replace with count=1 to be explicit)
                prompt_name = key.replace(prompt_prefix, "", 1)
                if prompt_name not in prompts_dict:
                    # Infer embedding dimension from the saved tensor shape
                    saved_tensor = state_dict[key]
                    prompts_dict[prompt_name] = torch.nn.Parameter(
                        torch.zeros_like(saved_tensor)
                    )
                    print(f"  Injected prompt '{prompt_name}' from checkpoint")

        # Load with strict=False since we may have extra/missing prompt vectors
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  WARNING: Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

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
            config=config,
            target_coi=requested_coi,
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
            # Use padded resampling to eliminate edge artifacts in spectrograms
            waveform = resample_with_padding(waveform, sr, self.sample_rate)
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

    @torch.inference_mode()
    def separate_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Separate a batch of waveform tensors.

        Args:
            waveforms: Input waveform tensor (B, T)

        Returns:
            sources: (B, n_sources, T) tensor
        """
        B, T = waveforms.shape

        # Match training: variance normalization only (no mean subtraction)
        std = waveforms.std(dim=1, keepdim=True) + 1e-8

        # Normalize input by its standard deviation only
        x = (waveforms / std).to(self.device)  # (B, T)

        prompts = [self.prompts_list] * B

        output = self.model(x, prompts)  # (B, n_sources, T)

        # Rescale outputs
        sources = output.cpu() * std.unsqueeze(2)  # (B, n_sources, T)

        return sources

    def _separate_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """Separate a single segment.

        TUSS uses variance normalization (normalize by std) matching training.
        The model expects normalized input and outputs normalized sources.

        Args:
            segment: Input waveform (T,) of length segment_samples

        Returns:
            sources: (n_sources, T) tensor [COI, background]
        """
        # Match training: variance normalization only (no mean subtraction)
        std = segment.std() + 1e-8

        # Normalize input by its standard deviation only
        x = (segment / std).unsqueeze(0).to(self.device)  # (1, T)

        # Build prompts list: [coi_prompts + [bg_prompt]]
        # TUSS model expects prompts as List[List[str]] where outer list is batch
        prompts = [self.prompts_list]

        # Run model
        output = self.model(x, prompts)  # (1, n_sources, T)

        # Scale outputs by input std to restore reasonable amplitude
        # (model outputs normalized sources, we rescale to match input level)
        sources = output[0].cpu() * std  # (n_sources, T)

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
        output = torch.zeros(self.num_sources, original_length)
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
            COI audio tensor with shape (T,) if single COI, or (num_cois, T) if multiple.
        """
        if len(self.coi_prompts) == 1:
            return sources[0]
        return sources[:-1]

    def get_background_audio(self, sources: torch.Tensor) -> torch.Tensor:
        """Extract the background audio from separated sources.

        Args:
            sources: Output from separate() with shape (n_sources, T)

        Returns:
            Background audio tensor with shape (T,)
        """
        return sources[-1]

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

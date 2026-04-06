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
# NOTE: With multi-COI support, COI_HEAD_INDEX now refers to the first COI class.
# Background is always the last index: sources[-1] or sources[BACKGROUND_HEAD_INDEX]
# where BACKGROUND_HEAD_INDEX = len(coi_prompts)
COI_HEAD_INDEX: int = 0
BACKGROUND_HEAD_INDEX: int = 1  # Only valid for single-COI mode
NUM_SOURCES: int = 2  # Only valid for single-COI mode


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
    audio source separation. Supports both single-COI and multi-COI modes:
    
    Single-COI mode (backward compatible):
        - coi_prompts = ["airplane"]
        - output shape: (2, T) = [COI, background]
        - Access: sources[COI_HEAD_INDEX], sources[BACKGROUND_HEAD_INDEX]
    
    Multi-COI mode:
        - coi_prompts = ["airplane", "bird", "car"]  (sorted alphabetically)
        - output shape: (4, T) = [airplane, bird, car, background]
        - Access: sources[0:3] for COIs, sources[-1] for background
        - Or use get_coi_by_name() for named access

    The interface is compatible with SeparationInference from sudormrf for
    single-COI usage, with extensions for multi-COI support.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sample_rate: int,
        segment_samples: int,
        device: str,
        coi_prompts: Union[str, List[str]],
        bg_prompt: str,
        config: Optional[dict] = None,
    ):
        """Initialize TUSSInference.

        Args:
            model: SeparationModel instance
            sample_rate: Model's native sample rate (typically 48000 Hz)
            segment_samples: Segment length in samples
            device: Device to run inference on
            coi_prompts: Prompt name(s) for Class(es) of Interest. Can be:
                        - Single string: "airplane" (backward compatible)
                        - List of strings: ["airplane", "bird", "car"]
                        Prompts will be sorted alphabetically for consistent ordering.
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
        
        # Support both string and list input for backward compatibility
        if isinstance(coi_prompts, str):
            coi_prompts = [coi_prompts]
        
        # CRITICAL: Do NOT sort prompts - order must match training config
        # The model learns prompt embeddings indexed by position, so changing
        # the order between training and inference breaks everything silently.
        self.coi_prompts = list(coi_prompts)
        self.bg_prompt = bg_prompt
        
        # Store config for downstream code (e.g., validation pipelines)
        self.config = config
        
        # Number of sources: N COI classes + 1 background
        self.num_sources = len(self.coi_prompts) + 1
        
        # For backward compatibility: single-COI mode
        if len(self.coi_prompts) == 1:
            self.coi_prompt = self.coi_prompts[0]
        
        # Verify all model components are on the correct device
        self._verify_device_placement()
    
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
        coi_prompts: Optional[Union[str, List[str]]] = None,
        bg_prompt: str = "background",
    ) -> "TUSSInference":
        """Load model from training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory or .pth file
            device: Device to load on (default: auto-detect cuda/cpu)
            coi_prompts: Prompt name(s) for COI. Can be:
                        - None: use config defaults
                        - String: "airplane" (single COI)
                        - List: ["airplane", "bird", "car"] (multi-COI)
                        Defaults to first COI in config if not specified.
            bg_prompt: Prompt name for background (default: "background")

        Returns:
            TUSSInference instance ready for separation
        """
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
        
        # Handle coi_prompts parameter
        if coi_prompts is None:
            # Use config defaults
            coi_prompts = config_coi_prompts
        elif isinstance(coi_prompts, str):
            # Single prompt as string
            coi_prompts = [coi_prompts]
        # else: already a list, use as-is
        
        # Update bg_prompt from config if using default
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
            if ckpt_coi_prompts and coi_prompts == config_coi_prompts:
                # Use checkpoint's prompts if caller used defaults
                coi_prompts = ckpt_coi_prompts
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
        print(f"  Using prompts: COI={coi_prompts}, Background='{bg_prompt}'")

        return cls(
            model=model,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            device=device,
            coi_prompts=coi_prompts,
            bg_prompt=bg_prompt,
            config=config,
        )

    @torch.inference_mode()
    def separate(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Separate audio file into component sources.

        Args:
            audio_path: Path to audio file

        Returns:
            sources: (n_sources, T) tensor where:
                     sources[0:len(coi_prompts)] = COI classes (alphabetically sorted)
                     sources[-1] = background
                     
                     Example with coi_prompts=["airplane", "bird"]:
                     sources[0] = airplane audio
                     sources[1] = bird audio
                     sources[2] = background audio
                     
                     Use get_coi_by_name() for named access.
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
                     sources[0:len(coi_prompts)] = COI classes (alphabetically sorted)
                     sources[-1] = background
                     
                     Use get_coi_by_name() for named access.
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
        """Separate a single segment with all COI prompts + background.

        TUSS uses variance normalization (normalize by std) matching training.
        The model expects normalized input and outputs normalized sources.

        Args:
            segment: Input waveform (T,) of length segment_samples

        Returns:
            sources: (n_sources, T) tensor where:
                    sources[0:len(coi_prompts)] = COI classes (alphabetically sorted)
                    sources[-1] = background
                    
                    Example with coi_prompts=["airplane", "bird", "car"]:
                    sources[0] = airplane
                    sources[1] = bird
                    sources[2] = car
                    sources[3] = background
        """
        # Variance normalization (matching TUSS training)
        mean = segment.mean()
        std = segment.std() + 1e-8

        # Normalize input (zero-mean, unit-variance)
        x = ((segment - mean) / std).unsqueeze(0).to(self.device)  # (1, T)

        # Build prompts list: [[coi1, coi2, ..., background]]
        # TUSS model expects prompts as List[List[str]] where outer list is batch
        # COI prompts are already sorted alphabetically in __init__
        prompts = [self.coi_prompts + [self.bg_prompt]]

        # Run model with all prompts in a single forward pass
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
            sources: (n_sources, T) tensor where:
                    sources[0:len(coi_prompts)] = COI classes
                    sources[-1] = background
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
        
        For backward compatibility with single-COI mode.
        In multi-COI mode, returns the first COI class.

        Args:
            sources: Output from separate() with shape (n_sources, T)

        Returns:
            COI audio tensor with shape (T,). In multi-COI mode, returns
            the first COI class (alphabetically first).
        """
        return sources[COI_HEAD_INDEX]
    
    def get_coi_by_name(self, sources: torch.Tensor, coi_name: str) -> torch.Tensor:
        """Extract a specific COI class by name.
        
        Args:
            sources: Output from separate() with shape (n_sources, T)
            coi_name: Name of the COI class (e.g., "airplane", "bird")
        
        Returns:
            COI audio tensor with shape (T,)
        
        Raises:
            ValueError: If coi_name is not in the configured COI prompts
        """
        if coi_name not in self.coi_prompts:
            raise ValueError(
                f"COI '{coi_name}' not found. Available COIs: {self.coi_prompts}"
            )
        idx = self.coi_prompts.index(coi_name)
        return sources[idx]
    
    def get_all_cois(self, sources: torch.Tensor) -> torch.Tensor:
        """Extract all COI classes (excluding background).
        
        Args:
            sources: Output from separate() with shape (n_sources, T)
        
        Returns:
            COI audio tensor with shape (n_coi_classes, T)
        """
        return sources[:-1]  # All except last (background)

    def get_background_audio(self, sources: torch.Tensor) -> torch.Tensor:
        """Extract the background audio from separated sources.

        Args:
            sources: Output from separate() with shape (n_sources, T)

        Returns:
            Background audio tensor with shape (T,)
        """
        return sources[-1]  # Background is always last
    
    def get_sources_dict(self, sources: torch.Tensor) -> dict:
        """Convert sources tensor to named dictionary.
        
        Args:
            sources: Output from separate() with shape (n_sources, T)
        
        Returns:
            Dictionary mapping prompt names to audio tensors:
            {"airplane": tensor(T,), "bird": tensor(T,), ..., "background": tensor(T,)}
        """
        result = {}
        for i, prompt in enumerate(self.coi_prompts):
            result[prompt] = sources[i]
        result[self.bg_prompt] = sources[-1]
        return result

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

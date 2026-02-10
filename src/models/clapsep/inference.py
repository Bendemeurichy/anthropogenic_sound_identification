"""
Inference module for CLAPSep separation model.

Supports:
  1. Pretrained text-query model (HuggingFace checkpoint)
  2. COI-trained models (from train_coi.py Lightning checkpoints)

Usage:
    sep = CLAPSepInference.from_pretrained(device="cuda")
    sources = sep.separate("audio.wav")          # (2, T)
    coi = sep.get_coi_audio(sources)             # (T,)

    sep = CLAPSepInference.from_checkpoint("best.ckpt")
    sources = sep.separate_waveform(waveform)    # (2, T)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torchaudio

_src_root = Path(__file__).resolve().parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

# Head indices (match sudormrf convention)
COI_HEAD_INDEX: int = 0
BACKGROUND_HEAD_INDEX: int = 1
NUM_SOURCES: int = 2

_THIS_DIR = Path(__file__).resolve().parent
_CKPT_MODEL_DIR = _THIS_DIR / "checkpoint" / "CLAPSep" / "model"
_DEFAULT_CLAP_PATH = _CKPT_MODEL_DIR / "music_audioset_epoch_15_esc_90.14.pt"
_DEFAULT_MODEL_CKPT = _CKPT_MODEL_DIR / "best_model.ckpt"

DEFAULT_MODEL_CONFIG = {
    "lan_embed_dim": 1024,
    "depths": [1, 1, 1, 1],
    "embed_dim": 128,
    "encoder_embed_dim": 128,
    "phase": False,
    "spec_factor": 8,
    "d_attn": 640,
    "n_masker_layer": 3,
    "conv": False,
}
DEFAULT_SAMPLE_RATE = 32000
DEFAULT_CHUNK_SAMPLES = 320000  # 10 s @ 32 kHz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_weight_file(path: Path, label: str = "Checkpoint") -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size < 512:
        try:
            with open(path, "r", encoding="utf-8") as f:
                if f.read(64).startswith("version https://git-lfs"):
                    raise RuntimeError(
                        f"{label} at '{path}' is a Git LFS pointer. "
                        f"Run 'git lfs pull' inside '{path.parent}'."
                    )
        except (UnicodeDecodeError, OSError):
            pass


def _import_clapsep_class():
    """Import CLAPSep from checkpoint dir, falling back to base dir.

    Uses importlib to load directly from file paths so that other ``model``
    modules on ``sys.path`` (e.g. ``plane_clasifier/model.py``) don't
    shadow the checkpoint's ``model/`` package.
    """
    import importlib.util

    # --- Attempt 1: checkpoint/CLAPSep/model/CLAPSep.py -------------------
    ckpt_clapsep = _CKPT_MODEL_DIR / "CLAPSep.py"
    ckpt_decoder = _CKPT_MODEL_DIR / "CLAPSep_decoder.py"
    ckpt_pkg = _CKPT_MODEL_DIR.parent
    if ckpt_clapsep.exists() and ckpt_decoder.exists() and ckpt_pkg.exists():
        added = str(ckpt_pkg) not in sys.path
        if added:
            sys.path.insert(0, str(ckpt_pkg))
        # Another library (e.g. plane_clasifier) may have registered a bare
        # ``model`` module in sys.modules.  That shadows the namespace
        # package we need here (``checkpoint/CLAPSep/model/``).  Temporarily
        # evict it so our import resolves to the correct directory.
        shadow = sys.modules.pop("model", None)
        try:
            from model.CLAPSep import CLAPSep  # type: ignore[import]

            return CLAPSep
        except Exception:
            pass
        finally:
            if added:
                try:
                    sys.path.remove(str(ckpt_pkg))
                except ValueError:
                    pass
            # Restore the evicted module so other code is unaffected
            if shadow is not None:
                sys.modules.setdefault("model", shadow)
    try:
        from models.clapsep.base.model.CLAPSep import CLAPSep  # type: ignore

        return CLAPSep
    except Exception:
        pass
    raise ImportError(
        "Cannot import CLAPSep from checkpoint/CLAPSep/model/ or base/model/. "
        "Ensure laion-clap, loralib, einops, torchlibrosa are installed."
    )


def _build_pretrained_clapsep(clap_path: Path, ckpt_path: Path, device: str):
    """Build and load the pretrained CLAPSep model."""
    _validate_weight_file(clap_path, "CLAP weights")
    _validate_weight_file(ckpt_path, "CLAPSep checkpoint")

    cls = _import_clapsep_class()
    print(f"[CLAPSep] Building model ({cls.__name__})")
    model = cls(DEFAULT_MODEL_CONFIG, str(clap_path))

    print(f"[CLAPSep] Loading weights from {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _text_embeddings(clapsep, text_pos: str, text_neg: str, device: str):
    """Compute CLAP text embeddings for positive/negative queries."""
    with torch.no_grad():
        clap = clapsep.clap_model  # type: ignore[attr-defined]
        both = clap.get_text_embedding([text_pos, text_neg], use_tensor=True)
        ep, en = both.chunk(2, dim=0)
        if text_pos == "":
            ep = torch.zeros_like(ep)
        if text_neg == "":
            en = torch.zeros_like(en)
    return ep.to(device), en.to(device)


# ---------------------------------------------------------------------------
# Wrapper: CLAPSep -> (B,1,T) -> (B,2,T)  (sudormrf-compatible)
# ---------------------------------------------------------------------------


class CLAPSepModelWrapper(nn.Module):
    """Makes CLAPSep callable as ``model(x)`` with shape ``(B,1,T)->(B,2,T)``."""

    num_sources: int = NUM_SOURCES

    def __init__(
        self, clapsep, embed_pos, embed_neg, chunk_samples: int = DEFAULT_CHUNK_SAMPLES
    ):
        super().__init__()
        self.clapsep = clapsep
        self.register_buffer("embed_pos", embed_pos)
        self.register_buffer("embed_neg", embed_neg)
        self.chunk_samples = chunk_samples

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mix = x.squeeze(1)  # (B, T)
        out = []
        for i in range(mix.shape[0]):
            coi = self._separate_single(mix[i])
            out.append(torch.stack([coi, mix[i] - coi]))
        return torch.stack(out)

    def _separate_single(self, wav: torch.Tensor) -> torch.Tensor:
        orig_len = wav.shape[0]
        mx = torch.max(torch.abs(wav))
        if mx > 1:
            wav = wav * (0.9 / mx)
        rem = orig_len % self.chunk_samples
        if rem:
            wav = torch.nn.functional.pad(wav, (0, self.chunk_samples - rem))
        parts = []
        for chunk in wav.split(self.chunk_samples):
            sep = self.clapsep.inference_from_data(  # type: ignore[attr-defined]
                chunk.unsqueeze(0), self.embed_pos, self.embed_neg
            )
            parts.append(sep.squeeze(0))
        return torch.cat(parts)[:orig_len]


# ---------------------------------------------------------------------------
# COI-trained model adapter
# ---------------------------------------------------------------------------


class _COIModelAdapter(nn.Module):
    """Wraps COICLAPSep: ``(B,1,T) -> (B,2,T)``."""

    num_sources: int = NUM_SOURCES

    def __init__(self, lightning_model, sample_rate=DEFAULT_SAMPLE_RATE):
        super().__init__()
        self.lm = lightning_model
        self.sample_rate = sample_rate

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm(x.squeeze(1))  # COICLAPSep.forward returns (B,2,T)


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------


class CLAPSepInference:
    """Drop-in replacement for ``SeparationInference`` from sudormrf.

    Attributes: model, sample_rate, segment_samples, device
    """

    def __init__(
        self, model: nn.Module, sample_rate: int, segment_samples: int, device: str
    ):
        self.model: nn.Module = model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.device = device

    # -- Factory: pretrained HF checkpoint ----------------------------------
    @classmethod
    def from_pretrained(
        cls,
        clap_path=None,
        model_ckpt_path=None,
        device=None,
        text_pos="airplane engine",
        text_neg="",
    ):
        """Load the pretrained CLAPSep from ``checkpoint/CLAPSep/model/``."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        clap_path = Path(clap_path) if clap_path else _DEFAULT_CLAP_PATH
        model_ckpt_path = (
            Path(model_ckpt_path) if model_ckpt_path else _DEFAULT_MODEL_CKPT
        )

        print(f"[CLAPSep] device={device}  CLAP={clap_path}  ckpt={model_ckpt_path}")
        clapsep = _build_pretrained_clapsep(clap_path, model_ckpt_path, device).to(
            device
        )
        ep, en = _text_embeddings(clapsep, text_pos, text_neg, device)
        wrapper = CLAPSepModelWrapper(clapsep, ep, en)
        return cls(wrapper, DEFAULT_SAMPLE_RATE, DEFAULT_CHUNK_SAMPLES, device)

    # -- Factory: COI-trained Lightning checkpoint --------------------------
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None):
        """Load a COI-trained CLAPSep (from ``train_coi.py``)."""
        import laion_clap

        from models.clapsep.base.model.CLAPSep_decoder import HTSAT_Decoder
        from models.clapsep.train_coi import COICLAPSep, COICLAPSepDecoder

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(checkpoint_path)
        _validate_weight_file(checkpoint_path, "COI checkpoint")

        print(f"[CLAPSep] Loading COI checkpoint: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        hp = ckpt.get("hyper_parameters", {})
        sr = hp.get("sample_rate", DEFAULT_SAMPLE_RATE)

        # Find CLAP weights
        clap_path = _DEFAULT_CLAP_PATH
        if not clap_path.exists():
            clap_path = checkpoint_path.parent / "music_audioset_epoch_15_esc_90.14.pt"
        _validate_weight_file(clap_path, "CLAP weights")

        clap_model = laion_clap.CLAP_Module(
            enable_fusion=False, amodel="HTSAT-base", device="cpu"
        )
        clap_model.load_ckpt(str(clap_path))

        coi_decoder = COICLAPSepDecoder(
            decoder=HTSAT_Decoder(**DEFAULT_MODEL_CONFIG),
            embed_dim=DEFAULT_MODEL_CONFIG["lan_embed_dim"],
            num_sources=NUM_SOURCES,
        )
        lm = COICLAPSep(
            clap_model=clap_model,
            decoder_model=coi_decoder,
            nfft=hp.get("nfft", 1024),
            sample_rate=sr,
            resample_rate=hp.get("resample_rate", 48000),
        )
        m, u = lm.load_state_dict(state_dict, strict=False)
        if m:
            print(f"[CLAPSep] Missing keys: {len(m)}")
        if u:
            print(f"[CLAPSep] Unexpected keys: {len(u)}")
        lm.eval().to(device)

        return cls(_COIModelAdapter(lm, sr), sr, int(sr * 10), device)

    # -- Inference ----------------------------------------------------------
    @torch.inference_mode()
    def separate(self, audio_path, text_pos=None, text_neg=None) -> torch.Tensor:
        """Separate audio file -> ``(2, T)`` tensor."""
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        waveform = waveform.mean(0) if waveform.shape[0] > 1 else waveform.squeeze(0)
        return self.separate_waveform(waveform, text_pos, text_neg)

    @torch.inference_mode()
    def separate_waveform(self, waveform, text_pos=None, text_neg=None) -> torch.Tensor:
        """Separate waveform ``(T,)`` or ``(1,T)`` -> ``(2, T)``."""
        if waveform.dim() == 2:
            waveform = (
                waveform.mean(0) if waveform.shape[0] > 1 else waveform.squeeze(0)
            )
        if text_pos is not None or text_neg is not None:
            self._update_text_embeddings(text_pos, text_neg)
        x = waveform.unsqueeze(0).unsqueeze(0).to(self.device)
        sources = self.model(x)[0].cpu()
        return sources[..., : waveform.shape[0]]

    def get_coi_audio(self, sources: torch.Tensor) -> torch.Tensor:
        return sources[COI_HEAD_INDEX]

    def get_background_audio(self, sources: torch.Tensor) -> torch.Tensor:
        return sources[BACKGROUND_HEAD_INDEX]

    def save_audio(self, waveform: torch.Tensor, path) -> None:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(str(path), waveform.cpu(), self.sample_rate)
        print(f"Saved: {path}")

    def _update_text_embeddings(self, text_pos, text_neg):
        if not isinstance(self.model, CLAPSepModelWrapper):
            return
        ep, en = _text_embeddings(
            self.model.clapsep, text_pos or "", text_neg or "", self.device
        )
        self.model.embed_pos = ep
        self.model.embed_neg = en


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="CLAPSep separation inference")
    p.add_argument("--audio", required=True, help="Input audio file")
    p.add_argument(
        "--checkpoint", default=None, help="COI-trained .ckpt (omit for pretrained)"
    )
    p.add_argument("--text-pos", default="airplane engine")
    p.add_argument("--text-neg", default="")
    p.add_argument("--clap-path", default=None)
    p.add_argument("--model-ckpt", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    audio_path = Path(args.audio)
    out = Path(args.output_dir) if args.output_dir else audio_path.parent
    out.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        sep = CLAPSepInference.from_checkpoint(args.checkpoint, device=args.device)
    else:
        sep = CLAPSepInference.from_pretrained(
            args.clap_path, args.model_ckpt, args.device, args.text_pos, args.text_neg
        )

    sources = sep.separate(audio_path)
    sep.save_audio(sep.get_coi_audio(sources), out / f"{audio_path.stem}_coi.wav")
    sep.save_audio(
        sep.get_background_audio(sources), out / f"{audio_path.stem}_background.wav"
    )
    print(f"\nSeparated into {NUM_SOURCES} sources in {out}")


if __name__ == "__main__":
    main()

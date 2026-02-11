"""
Demo separation script.

Loads honk.wav as background noise and plane.wav as the class-of-interest (COI)
sound, computes and saves spectrograms of both inputs, creates a mixture at a
configurable SNR, runs source separation through a specified model checkpoint,
and saves all outputs (WAVs + spectrogram plots).

This script uses the ValidationPipeline from test_pipeline.py to load both
the separation and classification models (exactly as done during validation),
but only exercises separation for this audible demo.

Usage:
    python demo_separation.py --checkpoint /path/to/best_model.pt [--cls_weights /path/to/weights.h5] [--snr 0] [--out_dir ./demo_output]
"""

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so plots save without display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Path bootstrapping - make sure project-level imports resolve
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # .../code

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR / "plane_clasifier"))

# Shim for old checkpoint pickles that reference moved modules
try:
    import src.models.base.sudo_rm_rf
except Exception:
    try:
        real_mod = importlib.import_module("models.sudormrf.base.sudo_rm_rf")
        sys.modules["src.models.base.sudo_rm_rf"] = real_mod
        sys.modules["sudo_rm_rf"] = real_mod
    except Exception:
        pass

from src.common.coi_training import prepare_batch_mono
from src.models.sudormrf.inference import (
    BACKGROUND_HEAD_INDEX,
    COI_HEAD_INDEX,
)
from src.validation_functions.test_pipeline import ValidationPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Sample rate will be set dynamically based on model type
SAMPLE_RATE = 16_000  # Default, will be updated in main()
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

# Default file locations (relative to this script)
DEFAULT_COI_PATH = _SCRIPT_DIR / "plane.wav"
DEFAULT_BG_PATH = _SCRIPT_DIR / "honk.wav"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def load_wav(path: Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load a WAV file, resample to *target_sr*, and return a 1-D tensor."""
    # Use soundfile directly to avoid torchcodec FFmpeg issues
    wav, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(wav)

    # Handle multi-channel: convert to (channels, samples) if needed
    if wav.ndim == 2:
        wav = wav.T  # (samples, channels) -> (channels, samples)
    else:
        wav = wav.unsqueeze(0)  # (samples,) -> (1, samples)

    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    # Mix to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    return wav


def create_mixture_from_sources(
    coi_wav: torch.Tensor, bg_wav: torch.Tensor, snr_db: float
) -> tuple:
    """Create a mixture using the shared prepare_batch_mono from src.common.

    Stacks the COI and background into a (1, 2, T) sources tensor, calls
    prepare_batch_mono with a fixed SNR, and computes the actual SNR from
    the scaled signals.

    Args:
        coi_wav: 1-D COI waveform (T,)
        bg_wav:  1-D background waveform (T,)
        snr_db:  Target SNR in dB

    Returns:
        (mixture, clean_sources, actual_snr_db) where
            mixture:       (T,) normalized mixture waveform
            clean_sources: (2, T) independently normalized [COI, background]
            actual_snr_db: float - achieved SNR after clamping
    """
    # prepare_batch_mono expects (B, n_src, T) with background as the last source
    sources = torch.stack([coi_wav, bg_wav], dim=0).unsqueeze(0)  # (1, 2, T)

    # Use a single-point SNR range so the requested value is used exactly
    mixture, clean_wavs = prepare_batch_mono(
        sources, snr_range=(snr_db, snr_db), deterministic=True
    )
    # mixture: (1, T), clean_wavs: (1, 2, T)
    mixture = mixture.squeeze(0)  # (T,)
    clean_wavs = clean_wavs.squeeze(0)  # (2, T)

    # Compute actual SNR from the independently normalised sources
    eps = 1e-8
    coi_power = clean_wavs[0].pow(2).mean() + eps
    bg_power = clean_wavs[1].pow(2).mean() + eps
    actual_snr = (10 * torch.log10(coi_power / bg_power)).item()

    return mixture, clean_wavs, actual_snr


def compute_spectrogram(wav: torch.Tensor) -> np.ndarray:
    """Return the log-magnitude spectrogram (freq x time) as a numpy array."""
    spec = torch.stft(
        wav,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=torch.hann_window(WIN_LENGTH),
        return_complex=True,
    )
    mag = torch.abs(spec)
    log_mag = 20 * torch.log10(mag + 1e-8)
    return log_mag.cpu().numpy()


def plot_spectrogram(
    spec: np.ndarray,
    title: str,
    save_path: Path,
    sr: int = SAMPLE_RATE,
) -> None:
    """Plot a single spectrogram and save to *save_path*."""
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=(0, spec.shape[1] * HOP_LENGTH / sr, 0, sr / 2),
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Saved spectrogram -> {save_path}")


def save_wav(wav: torch.Tensor, path: Path, sr: int = SAMPLE_RATE) -> None:
    t = wav.detach().cpu().numpy()
    # soundfile expects (samples,) or (samples, channels)
    if t.ndim == 2:
        if t.shape[0] == 1:
            t = t.squeeze(0)  # (1, samples) -> (samples,)
        else:
            t = t.T  # (channels, samples) -> (samples, channels)
    sf.write(str(path), t, sr)
    print(f"  Saved WAV -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global SAMPLE_RATE

    parser = argparse.ArgumentParser(
        description="Demo: mix honk (bg) + plane (COI), separate, save spectrograms"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the separation model checkpoint (.pt or .ckpt)",
    )
    parser.add_argument(
        "--cls_weights",
        type=str,
        default=None,
        help="Path to the classifier weights (.weights.h5). "
        "Defaults to ValidationPipeline.CLS_WEIGHTS.",
    )
    parser.add_argument(
        "--coi",
        type=str,
        default=str(DEFAULT_COI_PATH),
        help="Path to the COI (plane) WAV file",
    )
    parser.add_argument(
        "--bg",
        type=str,
        default=str(DEFAULT_BG_PATH),
        help="Path to the background (honk) WAV file",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=0.0,
        help="Target SNR in dB for the mixture (default: 0)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(_SCRIPT_DIR / "demo_output"),
        help="Output directory for WAVs and plots",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    coi_path = Path(args.coi)
    bg_path = Path(args.bg)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Detect model type and set appropriate sample rate
    checkpoint_str = str(checkpoint_path).lower()
    if checkpoint_path.suffix == ".ckpt" or "clapsep" in checkpoint_str:
        SAMPLE_RATE = 32_000  # ClapSep uses 32kHz
        model_type = "ClapSep"
        use_clapsep = True
    else:
        SAMPLE_RATE = 16_000  # SuDoRMRF uses 16kHz
        model_type = "SuDoRMRF"
        use_clapsep = False

    print(f"Detected model type: {model_type} (Sample rate: {SAMPLE_RATE} Hz)")
    # Create a model-specific subdirectory inside the requested output directory
    out_dir = out_root / model_type
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved to: {out_dir}")

    # ------------------------------------------------------------------
    # 1. Load input WAVs
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading COI (plane) from {coi_path}")
    coi_wav = load_wav(coi_path)
    print(
        f"       shape={tuple(coi_wav.shape)}, "
        f"duration={coi_wav.shape[0] / SAMPLE_RATE:.2f}s"
    )

    print(f"[1/6] Loading background (honk) from {bg_path}")
    bg_wav = load_wav(bg_path)
    print(
        f"       shape={tuple(bg_wav.shape)}, "
        f"duration={bg_wav.shape[0] / SAMPLE_RATE:.2f}s"
    )

    # ------------------------------------------------------------------
    # 2. Compute & save spectrograms for both inputs
    # ------------------------------------------------------------------
    print("\n[2/6] Computing and saving input spectrograms")
    coi_spec = compute_spectrogram(coi_wav)
    bg_spec = compute_spectrogram(bg_wav)

    plot_spectrogram(
        coi_spec,
        "COI Input - plane.wav",
        out_dir / f"spectrogram_coi_input_{ts}.png",
    )
    plot_spectrogram(
        bg_spec,
        "Background Input - honk.wav",
        out_dir / f"spectrogram_bg_input_{ts}.png",
    )

    # ------------------------------------------------------------------
    # 3. Load both models via ValidationPipeline (separator + classifier)
    # ------------------------------------------------------------------
    print(f"\n[3/6] Loading models via ValidationPipeline")
    print(f"       Separation checkpoint : {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    pipeline = ValidationPipeline()
    pipeline.load_models(
        sep_checkpoint=str(checkpoint_path),
        cls_weights=args.cls_weights,
        use_clapsep=use_clapsep,
    )
    print(f"       Classifier weights    : {pipeline.cls_checkpoint_path}")
    print(f"       Sample rate           : {pipeline.sample_rate}")
    print(f"       Device                : {pipeline.device}")

    # ------------------------------------------------------------------
    # 4. Align lengths and create mixture
    # ------------------------------------------------------------------
    print("\n[4/6] Creating mixture (using src.common.coi_training.prepare_batch_mono)")
    src_len = coi_wav.shape[0]
    noise_len = bg_wav.shape[0]
    if noise_len < src_len:
        repeats = int(np.ceil(src_len / noise_len))
        bg_wav = bg_wav.repeat(repeats)[:src_len]
    elif noise_len > src_len:
        bg_wav = bg_wav[:src_len]

    mixture, clean_sources, actual_snr = create_mixture_from_sources(
        coi_wav, bg_wav, args.snr
    )
    print(f"  Requested SNR: {args.snr:.1f} dB -> Actual SNR: {actual_snr:.1f} dB")

    mix_wav_path = out_dir / f"mixture_{ts}.wav"
    save_wav(mixture, mix_wav_path)

    # Save the independently-normalized clean sources produced by prepare_batch_mono
    clean_coi_path = out_dir / f"clean_coi_{ts}.wav"
    clean_bg_path = out_dir / f"clean_bg_{ts}.wav"
    save_wav(clean_sources[0], clean_coi_path)
    save_wav(clean_sources[1], clean_bg_path)

    mix_spec = compute_spectrogram(mixture)
    plot_spectrogram(
        mix_spec,
        f"Mixture (SNR={actual_snr:.1f} dB)",
        out_dir / f"spectrogram_mixture_{ts}.png",
    )

    # ------------------------------------------------------------------
    # 5. Run separation via ValidationPipeline._separate
    # ------------------------------------------------------------------
    print(
        f"\n[5/6] Running separation "
        f"(COI head={COI_HEAD_INDEX}, BG head={BACKGROUND_HEAD_INDEX})"
    )
    separated = pipeline._separate(mixture.to(pipeline.device))
    separated = separated.detach().cpu()
    print(f"  Separated tensor shape: {tuple(separated.shape)}")

    n_sources = separated.shape[0]
    separated_paths = []
    for i in range(n_sources):
        label = "coi" if i == COI_HEAD_INDEX else "background"
        wav_path = out_dir / f"separated_{label}_src{i}_{ts}.wav"
        save_wav(separated[i], wav_path)
        separated_paths.append(wav_path)

    # ------------------------------------------------------------------
    # 6. Compute & save spectrograms of separated outputs
    # ------------------------------------------------------------------
    print(f"\n[6/6] Computing and saving separated spectrograms")
    sep_specs = []
    for i in range(n_sources):
        label = "COI (plane)" if i == COI_HEAD_INDEX else "Background (honk)"
        spec = compute_spectrogram(separated[i])
        sep_specs.append(spec)
        plot_spectrogram(
            spec,
            f"Separated - {label} (src{i})",
            out_dir / f"spectrogram_separated_src{i}_{ts}.png",
        )

    # ------------------------------------------------------------------
    # Combined comparison figure
    # ------------------------------------------------------------------
    n_plots = 2 + 1 + n_sources  # 2 inputs + 1 mixture + n_sources separated
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots))

    titles = [
        "COI Input - plane.wav",
        "Background Input - honk.wav",
        f"Mixture (SNR={actual_snr:.1f} dB)",
    ]
    specs_all = [coi_spec, bg_spec, mix_spec]
    for i in range(n_sources):
        label = "COI (plane)" if i == COI_HEAD_INDEX else "Background (honk)"
        titles.append(f"Separated - {label} (src{i})")
        specs_all.append(sep_specs[i])

    for ax, spec, title in zip(axes, specs_all, titles):
        im = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=(
                0,
                spec.shape[1] * HOP_LENGTH / SAMPLE_RATE,
                0,
                SAMPLE_RATE / 2,
            ),
        )
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="dB")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    combined_path = out_dir / f"spectrograms_combined_{ts}.png"
    fig.savefig(str(combined_path), dpi=150)
    plt.close(fig)
    print(f"  Saved combined spectrogram -> {combined_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEMO COMPLETE - all outputs saved to:", out_dir)
    print("=" * 60)
    print(f"  COI input WAV        : {coi_path}")
    print(f"  Background input WAV : {bg_path}")
    print(f"  Clean COI WAV        : {clean_coi_path}")
    print(f"  Clean BG WAV         : {clean_bg_path}")
    print(f"  Mixture WAV          : {mix_wav_path}")
    for i, p in enumerate(separated_paths):
        label = "COI" if i == COI_HEAD_INDEX else "BG "
        print(f"  Separated {label} WAV    : {p}")
    print(f"  Combined plot        : {combined_path}")
    print(f"  Individual plots in  : {out_dir}/spectrogram_*_{ts}.png")


if __name__ == "__main__":
    main()

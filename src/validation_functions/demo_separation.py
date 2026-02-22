"""
Demo separation script.

Loads honk.wav as background noise and plane.wav as the class-of-interest (COI)
sound, computes and saves spectrograms of both inputs, creates a mixture at a
configurable SNR, runs source separation through a specified model checkpoint,
and saves all outputs (WAVs + spectrogram plots).

Several helper routines have been added to this module for
post‑hoc visualisation:

* ``plot_spectrogram_from_wav`` – load a WAV and write a single spectrogram
  PNG.
* ``plot_multiple_wav_spectrograms`` – batch version of the above.
* ``plot_combined_spectrograms`` / ``plot_combined_spectrograms_from_wavs`` –
  produce a vertically stacked comparison figure akin to the one generated
  by the full demo.

These utilities can be imported and used independently of the demo pipeline.

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
# human hearing range - used when plotting so that axes extend at least
# this far even if the underlying sample rate is lower than 40 kHz.
HUMAN_HEARING_MAX = 20_000

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


def compute_spectrogram(
    wav: torch.Tensor,
    sr: int = SAMPLE_RATE,
    display_sr: int | None = None,
    n_mels: int = 128,
) -> tuple[np.ndarray, int]:
    """
    Return a log-magnitude Mel-spectrogram (n_mels x time) as a numpy array.

    ``wav`` is assumed to be sampled at ``sr``.  We optionally resample the
    waveform to a higher rate before computing the Mel transform so the
    resulting figure can cover the full human hearing band.

    If ``display_sr`` is provided it is used as the target rate for the STFT.
    Otherwise we pick

        max(sr, HUMAN_HEARING_MAX * 2)

    which guarantees at least 20 kHz of frequency content in the result.  The
    function returns ``(mel_spec_db, used_sr)`` where ``used_sr`` is the sample
    rate actually employed (useful when labelling axes).
    """
    # choose a display rate if none was requested
    if display_sr is None:
        display_sr = max(sr, HUMAN_HEARING_MAX * 2)

    # resample waveform if required
    if display_sr != sr:
        wav = torchaudio.transforms.Resample(sr, display_sr)(wav.unsqueeze(0)).squeeze(
            0
        )
        sr = display_sr

    # Compute Mel-spectrogram (torchaudio returns power by default)
    # MelSpectrogram expects input shape (channel, samples)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=n_mels,
        power=2.0,
    )
    mel_spec = mel_transform(wav.unsqueeze(0))[0]  # (n_mels, time)

    # Convert power to dB for plotting
    mel_spec_db = 10.0 * torch.log10(mel_spec + 1e-8)

    return mel_spec_db.cpu().numpy(), sr


def plot_spectrogram(
    spec: np.ndarray,
    title: str,
    save_path: Path,
    sr: int = SAMPLE_RATE,
) -> None:
    """Plot a Mel-spectrogram (mel x time) with informative y-axis (Hz) and save to *save_path*."""
    fig, ax = plt.subplots(figsize=(10, 4))
    # time extent in seconds
    time_sec = spec.shape[1] * HOP_LENGTH / sr
    n_mels = spec.shape[0]

    # We display the matrix with vertical axis as Mel bands (0..n_mels)
    im = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=(0, time_sec, 0, n_mels),
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

    # Map a few Mel-band positions back to Hz for readable tick labels.
    # Uses the standard mel <-> hz conversion:
    #   mel = 2595 * log10(1 + f/700)
    #   f   = 700 * (10^(mel/2595) - 1)
    def mel_to_hz(mel_val: float) -> float:
        return 700.0 * (10.0 ** (mel_val / 2595.0) - 1.0)

    # mel_max corresponding to Nyquist (sr/2)
    mel_max = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)

    # Choose a small set of ticks evenly spaced on the mel axis
    num_ticks = 6
    mel_ticks = np.linspace(0.0, mel_max, num_ticks)
    # Convert mel values to corresponding bin indices in [0, n_mels-1]
    mel_bin_indices = (mel_ticks / mel_max) * (n_mels - 1)
    # Positions for imshow (extent uses 0..n_mels), so use those indices directly
    y_positions = mel_bin_indices

    # Format Hz labels (use k for thousands)
    def fmt_hz(hz: float) -> str:
        if hz >= 1000:
            return f"{hz / 1000:.1f}k"
        return f"{int(hz)}"

    hz_labels = [fmt_hz(mel_to_hz(m)) for m in mel_ticks]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(hz_labels)

    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Saved spectrogram -> {save_path}")


def plot_spectrogram_from_wav(
    wav_path: Path,
    save_path: Path,
    title: str | None = None,
    sr: int = SAMPLE_RATE,
) -> None:
    """Load a WAV file, compute its spectrogram, and save a plot.

    Convenience wrapper that handles the I/O and plotting in one call.  The
    audio is loaded and resampled to ``sr``; the spectrogram computation will
    upsample further if necessary so that the resulting figure contains actual
    values covering the full human hearing band (up to ~20 kHz).
    """
    wav = load_wav(wav_path, target_sr=sr)
    spec, used_sr = compute_spectrogram(wav, sr=sr)
    if title is None:
        title = f"Spectrogram - {wav_path.name}"
    plot_spectrogram(spec, title, save_path, sr=used_sr)


def plot_multiple_wav_spectrograms(
    wav_paths: list[Path],
    out_dir: Path,
    sr: int = SAMPLE_RATE,
) -> None:
    """Plot spectrograms for a list of WAV files and save them to *out_dir*.

    Each output file is named with a timestamp to avoid collisions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for wav_path in wav_paths:
        stem = wav_path.stem.replace(" ", "_")
        save_path = out_dir / f"spectrogram_{stem}_{ts}.png"
        plot_spectrogram_from_wav(wav_path, save_path, title=wav_path.name, sr=sr)


def plot_combined_spectrograms(
    specs: list[np.ndarray],
    titles: list[str],
    save_path: Path,
    sr: int = SAMPLE_RATE,
) -> None:
    """Create a combined vertical figure from a list of Mel-spectrograms.

    Each input in ``specs`` is expected to be a Mel-spectrogram (n_mels x time).
    The y-axis tick labels are shown in Hz for easier human interpretation.
    """
    n_plots = len(specs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots))
    # ``subplots`` returns a single Axes object when n_plots == 1, so
    # normalise to a list so the later loop works uniformly.
    if n_plots == 1:
        axes = [axes]

    for ax, spec, title in zip(axes, specs, titles):
        time_sec = spec.shape[1] * HOP_LENGTH / sr
        n_mels = spec.shape[0]
        im = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=(0, time_sec, 0, n_mels),
        )
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(title)

        # Create mel->Hz ticks as in single plot function
        def mel_to_hz(mel_val: float) -> float:
            return 700.0 * (10.0 ** (mel_val / 2595.0) - 1.0)

        mel_max = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
        num_ticks = 6
        mel_ticks = np.linspace(0.0, mel_max, num_ticks)
        mel_bin_indices = (mel_ticks / mel_max) * (n_mels - 1)
        y_positions = mel_bin_indices

        def fmt_hz(hz: float) -> str:
            if hz >= 1000:
                return f"{hz / 1000:.1f}k"
            return f"{int(hz)}"

        hz_labels = [fmt_hz(mel_to_hz(m)) for m in mel_ticks]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(hz_labels)

        fig.colorbar(im, ax=ax, label="dB")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Saved combined spectrogram -> {save_path}")


def plot_combined_spectrograms_from_wavs(
    wav_paths: list[Path],
    save_path: Path,
    titles: list[str] | None = None,
    sr: int = SAMPLE_RATE,
) -> None:
    """Load several WAVs, compute their spectrograms, and make a combined plot.

    Titles default to the basename of each file if not supplied.
    """
    specs = []
    used_sr: int | None = None
    for p in wav_paths:
        wav = load_wav(p, target_sr=sr)
        spec, used_sr = compute_spectrogram(wav, sr=sr)
        specs.append(spec)

    if titles is None:
        titles = [p.name for p in wav_paths]

    plot_combined_spectrograms(specs, titles, save_path, sr=(used_sr or sr))


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
    # determine display rate for plotting (at least twice the human-hearing
    # limit so the spec contains real values up to ~20 kHz)
    plot_sr = max(SAMPLE_RATE, HUMAN_HEARING_MAX * 2)

    coi_spec, _ = compute_spectrogram(coi_wav, sr=SAMPLE_RATE, display_sr=plot_sr)
    bg_spec, _ = compute_spectrogram(bg_wav, sr=SAMPLE_RATE, display_sr=plot_sr)

    plot_spectrogram(
        coi_spec,
        "COI Input - plane.wav",
        out_dir / f"spectrogram_coi_input_{ts}.png",
        sr=plot_sr,
    )
    plot_spectrogram(
        bg_spec,
        "Background Input - honk.wav",
        out_dir / f"spectrogram_bg_input_{ts}.png",
        sr=plot_sr,
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

    mix_spec, _ = compute_spectrogram(mixture, sr=SAMPLE_RATE, display_sr=plot_sr)
    plot_spectrogram(
        mix_spec,
        f"Mixture (SNR={actual_snr:.1f} dB)",
        out_dir / f"spectrogram_mixture_{ts}.png",
        sr=plot_sr,
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
        spec, _ = compute_spectrogram(separated[i], sr=SAMPLE_RATE, display_sr=plot_sr)
        sep_specs.append(spec)
        plot_spectrogram(
            spec,
            f"Separated - {label} (src{i})",
            out_dir / f"spectrogram_separated_src{i}_{ts}.png",
            sr=plot_sr,
        )

    # ------------------------------------------------------------------
    # Combined comparison figure
    # ------------------------------------------------------------------
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

    combined_path = out_dir / f"spectrograms_combined_{ts}.png"
    plot_combined_spectrograms(specs_all, titles, combined_path, sr=plot_sr)

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
    # main()
    wav_files = [
        Path(
            "/home/bendm/Thesis/project/code/src/validation_functions/validation_examples_train/mixture_sep/mixture_coi_clean_1.wav"
        ),
        Path(
            "/home/bendm/Thesis/project/code/src/validation_functions/validation_examples_train/mixture_sep/mixture_bg_clean_1.wav"
        ),
        Path(
            "/home/bendm/Thesis/project/code/src/validation_functions/validation_examples_train/mixture_sep/mixture_created_1.wav"
        ),
        Path(
            "/home/bendm/Thesis/project/code/src/validation_functions/validation_examples_train/mixture_sep/mixture_separated_coi_head_1.wav"
        ),
        Path(
            "/home/bendm/Thesis/project/code/src/validation_functions/validation_examples_train/mixture_sep/mixture_separated_src1_1.wav"
        ),
    ]
    plot_combined_spectrograms_from_wavs(
        wav_files,
        Path("separation_output_demo/combined_mixture.png"),
        titles=["Train", "Background", "Mixture", "Separated", "Background"],
    )

"""
plot_qualitative_gallery.py — Spectrogram gallery for dissertation.

Scans *_examples/ directories produced by runner scripts, finds sets of
saved WAV files (clean COI, mixture, separated outputs) and calls
demo_separation.plot_combined_spectrograms_from_wavs() to produce
one combined spectrogram figure per model.

Figures are saved as PNG to ./chapter_figures/gallery/.

Usage:
    python plot_qualitative_gallery.py

Requires:
    - One or more *_examples/ directories (from runner scripts with save_examples_dir set)
    - demo_separation.py in the same directory (provides plotting utilities)
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Try importing demo_separation utilities
try:
    from demo_separation import plot_combined_spectrograms_from_wavs
    HAS_DEMO = True
except ImportError as e:
    print(f"Warning: could not import demo_separation: {e}")
    HAS_DEMO = False

OUTPUT_DIR = SCRIPT_DIR / "chapter_figures" / "gallery"

# ── Gallery configuration ──────────────────────────────────────────────────
# Each entry: (display_label, examples_dir, wav_pattern)
# wav_pattern keys: clean_coi, mixture, separated_coi
GALLERY_CONFIGS = [
    {
        "label": "SuDoRM-RF — Airplane",
        "examples_dir": SCRIPT_DIR / "sudormrf_airplane_examples",
    },
    {
        "label": "TUSS single-class — Airplane",
        "examples_dir": SCRIPT_DIR / "tuss_singleclass_airplane_examples",
    },
    {
        "label": "CLAPSep — Airplane",
        "examples_dir": SCRIPT_DIR / "clapsep_airplane_examples",
    },
    {
        "label": "TUSS multi-class — Airplane head",
        "examples_dir": SCRIPT_DIR / "tuss_multiclass_airplane_examples",
    },
    {
        "label": "TUSS multi-class — Bird head",
        "examples_dir": SCRIPT_DIR / "tuss_multiclass_bird_examples",
    },
]


def _find_example_triplets(examples_dir: Path) -> list:
    """Return list of (clean_coi_path, mixture_path, separated_path) triplets."""
    if not examples_dir.exists():
        return []

    triplets = []
    # WAV files are named: clean_coi_<k>.wav, mixture_created_<k>.wav,
    #   mixture_separated_coi_head_<k>.wav  (TUSS/CLAPSep)
    #   mixture_separated_coi_est_<k>.wav   (SuDoRM-RF / no multi-source)
    clean_wavs = sorted(examples_dir.glob("*clean_coi_*.wav"))
    if not clean_wavs:
        # Fall back to scanning mixture_coi_clean_*.wav
        clean_wavs = sorted(examples_dir.glob("mixture_coi_clean_*.wav"))

    for clean_wav in clean_wavs:
        # Extract index k from filename
        stem = clean_wav.stem  # e.g. "clean_coi_0" or "mixture_coi_clean_0"
        k = stem.rsplit("_", 1)[-1]

        mixture_wav = examples_dir / f"mixture_created_{k}.wav"
        if not mixture_wav.exists():
            mixture_wav = examples_dir / f"mixture_{k}.wav"

        # Prefer COI head output; fall back to coi_est
        sep_wav = examples_dir / f"mixture_separated_coi_head_{k}.wav"
        if not sep_wav.exists():
            sep_wav = examples_dir / f"mixture_separated_coi_est_{k}.wav"
        if not sep_wav.exists():
            # Try any separated wav for this index
            candidates = list(examples_dir.glob(f"*separated*{k}*.wav"))
            sep_wav = candidates[0] if candidates else None

        if mixture_wav.exists() and sep_wav is not None and sep_wav.exists():
            triplets.append((clean_wav, mixture_wav, sep_wav))

    return triplets


def plot_gallery_entry(label: str, examples_dir: Path):
    """Generate and save spectrogram panels for one model's examples."""
    triplets = _find_example_triplets(examples_dir)
    if not triplets:
        print(f"  No example WAVs found in {examples_dir} — skipping.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "-").replace("—", "-")

    for i, (clean_wav, mixture_wav, sep_wav) in enumerate(triplets):
        out_path = OUTPUT_DIR / f"{safe_label}_example_{i}.png"
        print(f"  [{i+1}/{len(triplets)}] {clean_wav.name} → {out_path.name}")

        if HAS_DEMO:
            try:
                plot_combined_spectrograms_from_wavs(
                    clean_wav_path=str(clean_wav),
                    mixture_wav_path=str(mixture_wav),
                    separated_wav_path=str(sep_wav),
                    title=f"{label} — Example {i+1}",
                    output_path=str(out_path),
                )
            except Exception as e:
                print(f"    ERROR: {e}")
        else:
            # Fallback: try a minimal matplotlib spectrogram if demo_separation unavailable
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import torchaudio

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for ax, wav_path, title in zip(
                    axes,
                    [clean_wav, mixture_wav, sep_wav],
                    ["Clean COI", "Mixture", "Separated"],
                ):
                    waveform, sr = torchaudio.load(str(wav_path))
                    waveform = waveform[0]
                    ax.specgram(waveform.numpy(), Fs=sr, NFFT=1024, noverlap=512, cmap="inferno")
                    ax.set_title(title)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Frequency (Hz)")
                fig.suptitle(f"{label} — Example {i+1}", fontsize=13)
                plt.tight_layout()
                fig.savefig(str(out_path), dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"    Fallback matplotlib also failed: {e}")


def main():
    print("=" * 70)
    print("QUALITATIVE GALLERY — SPECTROGRAM FIGURES")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}\n")

    for cfg in GALLERY_CONFIGS:
        label = cfg["label"]
        examples_dir = cfg["examples_dir"]
        print(f"\n{'─'*50}")
        print(f"Model: {label}")
        print(f"Examples dir: {examples_dir}")
        plot_gallery_entry(label, examples_dir)

    print(f"\nGallery complete. Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

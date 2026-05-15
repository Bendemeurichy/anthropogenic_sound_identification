"""
plot_qualitative_gallery.py — Cross-model spectrogram gallery for the
dissertation Results chapter.

For each example index ``k`` shared across the model-specific
``*_examples/`` directories, this script produces a single combined
spectrogram figure that stacks:

  1. clean COI reference
  2. created mixture (clean + background)
  3. one panel per available separator, showing
     ``mixture_separated_coi_head_<k>.wav`` (or ``..._coi_est_<k>.wav`` as
     fallback for SuDoRM-RF).

Energy deltas (ΔRMS, ΔSEL) are reported relative to the clean COI panel so
the cross-model comparison highlights how each separator distorts the
energy envelope of the target.

Models whose example directories are missing or empty are silently
skipped, with a warning listing them once at the end.  Re-running after
populating the missing directories will regenerate the gallery with the
extra panels included.

Output: ``final_results/chapter_figures/gallery/example_<k>.png``
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from demo_separation import plot_combined_spectrograms_from_wavs  # noqa: E402

# ── Configuration ───────────────────────────────────────────────────────────
FINAL_RESULTS_DIR = SCRIPT_DIR / "final_results"
OUTPUT_DIR = FINAL_RESULTS_DIR / "chapter_figures" / "gallery"

# Map separator → (display label, examples-dir under final_results/, classifier
# subdir holding the WAVs).  The classifier subdir mirrors the runner layout
# `<run>_examples/<classifier>/{clean_sep,mixture_sep}/`.
GROUPS: Dict[str, List[Tuple[str, Path, str]]] = {
    "airplane": [
        ("SuDoRM-RF",     FINAL_RESULTS_DIR / "sudormrf_airplane_examples",        "pann_finetuned"),
        ("TUSS (single)", FINAL_RESULTS_DIR / "tuss_singleclass_airplane_examples", "pann_finetuned"),
        ("CLAPSep",       FINAL_RESULTS_DIR / "clapsep_airplane_examples",         "pann_finetuned"),
        ("TUSS (multi)",  FINAL_RESULTS_DIR / "tuss_multiclass_airplane_examples",  "pann_finetuned"),
    ],
    "bird": [
        ("TUSS (single)", FINAL_RESULTS_DIR / "tuss_singleclass_bird_examples",     "bird_mae"),
        ("TUSS (multi)",  FINAL_RESULTS_DIR / "tuss_multiclass_bird_examples",      "bird_mae"),
    ],
}

# Maximum number of example indices to render per group (keeps the chapter
# manageable; raise/lower freely).
MAX_EXAMPLES = 5


# ── Helpers ─────────────────────────────────────────────────────────────────

def _model_dirs(examples_dir: Path, classifier: str) -> Tuple[Path, Path]:
    """Return (clean_sep_dir, mixture_sep_dir) for a given examples root."""
    return (examples_dir / classifier / "clean_sep",
            examples_dir / classifier / "mixture_sep")


def _list_indices(examples_dir: Path, classifier: str) -> List[str]:
    """Indices of available examples in mixture_sep (the operational layout)."""
    _, mix_dir = _model_dirs(examples_dir, classifier)
    if not mix_dir.exists():
        return []
    idx = []
    for p in sorted(mix_dir.glob("mixture_created_*.wav")):
        idx.append(p.stem.rsplit("_", 1)[-1])
    return idx


def _separated_path(examples_dir: Path, classifier: str, k: str) -> Optional[Path]:
    """Resolve the separated-COI WAV for example k (head → est fallback)."""
    _, mix_dir = _model_dirs(examples_dir, classifier)
    candidates = [
        mix_dir / f"mixture_separated_coi_head_{k}.wav",
        mix_dir / f"mixture_separated_coi_est_{k}.wav",
    ]
    for c in candidates:
        if c.exists():
            return c
    extras = list(mix_dir.glob(f"*separated*_{k}.wav"))
    return extras[0] if extras else None


def _shared_indices(group: List[Tuple[str, Path, str]]) -> List[str]:
    """Indices present in *every* model's mixture_sep dir."""
    sets = []
    for _, examples_dir, classifier in group:
        idx = _list_indices(examples_dir, classifier)
        if idx:
            sets.append(set(idx))
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common, key=lambda s: int(s) if s.isdigit() else s)


def _all_indices(group: List[Tuple[str, Path, str]]) -> List[str]:
    """Union of indices — used as fallback if no shared index exists."""
    seen = set()
    for _, examples_dir, classifier in group:
        seen.update(_list_indices(examples_dir, classifier))
    return sorted(seen, key=lambda s: int(s) if s.isdigit() else s)


# ── Main rendering ──────────────────────────────────────────────────────────

def render_group(group_name: str, group: List[Tuple[str, Path, str]]) -> Tuple[int, List[str]]:
    """Render the cross-model gallery for one COI group.

    Returns (n_figures_written, missing_models)."""
    available = [(label, ed, cls) for label, ed, cls in group
                 if _list_indices(ed, cls)]
    missing   = [label for label, ed, cls in group
                 if not _list_indices(ed, cls)]

    if not available:
        return 0, [label for label, _, _ in group]

    indices = _shared_indices(available)
    if not indices:
        # Fall back: render whichever indices exist, even if not shared
        indices = _all_indices(available)

    indices = indices[:MAX_EXAMPLES]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    for k in indices:
        # Use the first available model to source the shared clean+mixture
        # references.  These WAVs are produced by the runners with a fixed
        # seed, so they should be identical across models for the same k.
        anchor_label, anchor_ed, anchor_cls = available[0]
        clean_dir, mix_dir = _model_dirs(anchor_ed, anchor_cls)
        clean = clean_dir / f"clean_coi_{k}.wav"
        if not clean.exists():
            clean = mix_dir / f"mixture_coi_clean_{k}.wav"
        mixture = mix_dir / f"mixture_created_{k}.wav"

        if not clean.exists() or not mixture.exists():
            print(f"  · {group_name} idx {k}: missing clean/mixture reference, skip")
            continue

        wav_paths: List[Path] = [clean, mixture]
        titles: List[str] = ["Clean COI (reference)", "Created mixture"]

        for label, ed, cls in available:
            sep = _separated_path(ed, cls, k)
            if sep is not None:
                wav_paths.append(sep)
                titles.append(f"Separated — {label}")
            else:
                print(f"  · {group_name} idx {k}: no separated WAV for {label}")

        if len(wav_paths) <= 2:
            continue  # nothing to compare

        out_path = OUTPUT_DIR / f"gallery_{group_name}_example_{k}.png"
        try:
            plot_combined_spectrograms_from_wavs(
                wav_paths=wav_paths,
                save_path=out_path,
                titles=titles,
                ref_idx=0,                           # clean COI = energy reference
                delta_indices=list(range(2, len(wav_paths))),
            )
            written += 1
        except Exception as e:
            import traceback
            print(f"  · ERROR rendering {out_path.name}: {e}")
            traceback.print_exc()

    return written, missing


def main() -> None:
    print("=" * 72)
    print("Cross-model qualitative gallery")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)

    summary: Dict[str, Tuple[int, List[str]]] = {}
    for group_name, group in GROUPS.items():
        print(f"\n— {group_name} —")
        written, missing = render_group(group_name, group)
        summary[group_name] = (written, missing)
        print(f"  Rendered {written} figure(s) for group '{group_name}'.")

    print("\n" + "=" * 72)
    print("Summary")
    for name, (n, missing) in summary.items():
        print(f"  {name:>10}: {n} figure(s)" +
              (f"  (missing examples: {', '.join(missing)})" if missing else ""))
    print("=" * 72)


if __name__ == "__main__":
    main()

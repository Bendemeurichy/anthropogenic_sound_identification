"""
plot_qualitative_gallery.py — Cross-model spectrogram gallery for the
dissertation Results chapter.

For each curated example index ``k`` shared across the model-specific
``*_examples/`` directories, this script produces one combined figure
that stacks:

  1. Clean COI target  (ground truth — what the separator should recover)
  2. Clean background  (the interferer that was added)
  3. Created mixture   (input the separator actually sees)
  4. Separated COI from each available model

ΔRMS and ΔSEL are reported relative to the clean COI panel (panel 0) so
panels 3+ directly quantify how each model's energy envelope deviates
from the target.  The clean BG / mixture references make it possible to
read off whether residual energy in the separated output is leftover
background or genuine COI content.

Examples are auto-curated to three illustrative cases per group:

  * **best**     — highest mean SI-SNRi across available models
  * **typical**  — median SI-SNRi
  * **worst**    — lowest SI-SNRi

This requires only the WAVs themselves; no extra metric files are read.

Output: ``final_results/chapter_figures/gallery/gallery_<group>_<case>_<k>.png``
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from demo_separation import (  # noqa: E402
    load_wav,
    plot_combined_spectrograms_from_wavs,
)

# ── Configuration ───────────────────────────────────────────────────────────
FINAL_RESULTS_DIR = SCRIPT_DIR / "final_results"
OUTPUT_DIR = FINAL_RESULTS_DIR / "chapter_figures" / "gallery"

GROUPS: Dict[str, List[Tuple[str, Path, str]]] = {
    "airplane": [
        ("SuDoRM-RF",     FINAL_RESULTS_DIR / "sudormrf_airplane_examples",         "pann_finetuned"),
        ("TUSS (single)", FINAL_RESULTS_DIR / "tuss_singleclass_airplane_examples", "pann_finetuned"),
        ("CLAPSep",       FINAL_RESULTS_DIR / "clapsep_airplane_examples",          "pann_finetuned"),
        ("TUSS (multi)",  FINAL_RESULTS_DIR / "tuss_multiclass_airplane_examples",  "pann_finetuned"),
    ],
    "bird": [
        ("TUSS (single)", FINAL_RESULTS_DIR / "tuss_singleclass_bird_examples",     "bird_mae"),
        ("TUSS (multi)",  FINAL_RESULTS_DIR / "tuss_multiclass_bird_examples",      "bird_mae"),
    ],
}


# ── WAV layout helpers ──────────────────────────────────────────────────────

def _model_dirs(examples_dir: Path, classifier: str) -> Tuple[Path, Path]:
    return (examples_dir / classifier / "clean_sep",
            examples_dir / classifier / "mixture_sep")


def _list_indices(examples_dir: Path, classifier: str) -> List[str]:
    _, mix_dir = _model_dirs(examples_dir, classifier)
    if not mix_dir.exists():
        return []
    return [p.stem.rsplit("_", 1)[-1]
            for p in sorted(mix_dir.glob("mixture_created_*.wav"))]


def _separated_path(examples_dir: Path, classifier: str, k: str) -> Optional[Path]:
    _, mix_dir = _model_dirs(examples_dir, classifier)
    for fname in (f"mixture_separated_coi_head_{k}.wav",
                  f"mixture_separated_coi_est_{k}.wav"):
        p = mix_dir / fname
        if p.exists():
            return p
    extras = list(mix_dir.glob(f"*separated*_{k}.wav"))
    return extras[0] if extras else None


def _reference_paths(group: List[Tuple[str, Path, str]], k: str
                     ) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Return (clean_coi, clean_bg, mixture) WAVs sourced from the first
    available model dir.  These are produced with a fixed seed and are
    therefore identical across runs of the same group."""
    for _, ed, cls in group:
        clean_dir, mix_dir = _model_dirs(ed, cls)
        coi = clean_dir / f"clean_coi_{k}.wav"
        if not coi.exists():
            coi = mix_dir / f"mixture_coi_clean_{k}.wav"
        bg  = mix_dir / f"mixture_bg_clean_{k}.wav"
        mix = mix_dir / f"mixture_created_{k}.wav"
        if coi.exists() and mix.exists():
            return (coi, bg if bg.exists() else None, mix)
    return None, None, None


# ── Curation: rank shared indices by mean SI-SNRi across models ─────────────

def _si_snri_db(coi: Path, mix: Path, sep: Path) -> float:
    """Scale-invariant SNR improvement of `sep` over `mix`, both vs. `coi`.

    ``si_snri = si_snr(coi, sep) − si_snr(coi, mix)``
    """
    try:
        wc, sr = load_wav(coi); wc = wc.flatten().numpy()
        wm, _  = load_wav(mix, target_sr=sr); wm = wm.flatten().numpy()
        ws, _  = load_wav(sep, target_sr=sr); ws = ws.flatten().numpy()
    except Exception:
        return float("nan")

    L = min(len(wc), len(wm), len(ws))
    if L < 16:
        return float("nan")
    wc, wm, ws = wc[:L], wm[:L], ws[:L]

    def si_snr(ref: np.ndarray, est: np.ndarray) -> float:
        ref = ref - ref.mean(); est = est - est.mean()
        denom = float(np.dot(ref, ref))
        if denom <= 1e-12:
            return float("nan")
        alpha = float(np.dot(est, ref)) / denom
        target = alpha * ref
        noise = est - target
        n = float(np.dot(noise, noise))
        t = float(np.dot(target, target))
        if n <= 1e-12 or t <= 1e-12:
            return float("nan")
        return 10.0 * np.log10(t / n)

    return si_snr(wc, ws) - si_snr(wc, wm)


def _rank_indices(group: List[Tuple[str, Path, str]]) -> List[Tuple[str, float]]:
    """Return [(index, mean_si_snri_db), …] sorted ascending by SI-SNRi,
    restricted to indices present in every available model dir."""
    available = [(label, ed, cls) for label, ed, cls in group
                 if _list_indices(ed, cls)]
    if not available:
        return []
    sets = [set(_list_indices(ed, cls)) for _, ed, cls in available]
    shared = sorted(set.intersection(*sets),
                    key=lambda s: int(s) if s.isdigit() else s)
    if not shared:
        return []

    scored: List[Tuple[str, float]] = []
    for k in shared:
        coi, _bg, mix = _reference_paths(group, k)
        if coi is None or mix is None:
            continue
        per_model = []
        for _, ed, cls in available:
            sep = _separated_path(ed, cls, k)
            if sep is None:
                continue
            per_model.append(_si_snri_db(coi, mix, sep))
        per_model = [v for v in per_model if not np.isnan(v)]
        if per_model:
            scored.append((k, float(np.mean(per_model))))
    scored.sort(key=lambda t: t[1])
    return scored


def _curated(scored: List[Tuple[str, float]]) -> List[Tuple[str, str, float]]:
    """Pick (case, k, mean_si_snri) for worst / typical / best."""
    if not scored:
        return []
    if len(scored) == 1:
        return [("typical", *scored[0])]
    if len(scored) == 2:
        return [("worst", *scored[0]), ("best", *scored[-1])]
    mid = scored[len(scored) // 2]
    return [
        ("worst",   *scored[0]),
        ("typical", *mid),
        ("best",    *scored[-1]),
    ]


# ── Rendering ───────────────────────────────────────────────────────────────

def render_group(group_name: str, group: List[Tuple[str, Path, str]]
                 ) -> Tuple[int, List[str]]:
    available = [(label, ed, cls) for label, ed, cls in group
                 if _list_indices(ed, cls)]
    missing = [label for label, ed, cls in group if not _list_indices(ed, cls)]
    if not available:
        return 0, [label for label, _, _ in group]

    print(f"  Ranking shared indices by mean SI-SNRi across "
          f"{len(available)} model(s)…")
    scored = _rank_indices(group)
    if not scored:
        print(f"  No shared indices with computable SI-SNRi.")
        return 0, missing
    curated = _curated(scored)
    print("  Selected:")
    for case, k, sni in curated:
        print(f"    {case:>7s}  k={k}  SI-SNRi={sni:+.2f} dB")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for case, k, sni in curated:
        coi, bg, mix = _reference_paths(group, k)
        if coi is None or mix is None:
            continue

        wav_paths: List[Path] = [coi]
        titles: List[str] = ["Clean COI target (reference)"]
        if bg is not None:
            wav_paths.append(bg)
            titles.append("Clean background (added interferer)")
        wav_paths.append(mix)
        titles.append("Created mixture (separator input)")

        for label, ed, cls in available:
            sep = _separated_path(ed, cls, k)
            if sep is None:
                continue
            wav_paths.append(sep)
            titles.append(f"Separated COI — {label}")

        if len(wav_paths) <= 3:
            continue  # nothing to compare

        delta_indices = list(range(3 if bg is not None else 2,
                                   len(wav_paths)))
        out_path = OUTPUT_DIR / f"gallery_{group_name}_{case}_k{k}.png"
        try:
            plot_combined_spectrograms_from_wavs(
                wav_paths=wav_paths,
                save_path=out_path,
                titles=titles,
                ref_idx=0,                 # clean COI = energy reference
                delta_indices=delta_indices,
            )
            written += 1
        except Exception as e:
            import traceback
            print(f"  ERROR rendering {out_path.name}: {e}")
            traceback.print_exc()
    return written, missing


def main() -> None:
    print("=" * 72)
    print("Cross-model qualitative gallery (curated)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)

    summary: Dict[str, Tuple[int, List[str]]] = {}
    for group_name, group in GROUPS.items():
        print(f"\n— {group_name} —")
        written, missing = render_group(group_name, group)
        summary[group_name] = (written, missing)
        print(f"  Rendered {written} figure(s).")

    print("\n" + "=" * 72)
    print("Summary")
    for name, (n, missing) in summary.items():
        print(f"  {name:>10}: {n} figure(s)" +
              (f"  (missing examples for: {', '.join(missing)})" if missing else ""))
    print("=" * 72)


if __name__ == "__main__":
    main()

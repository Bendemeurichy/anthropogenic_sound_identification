"""Harness verification on known-good audio.

Stage A: TUSS demo (multi_coi_14_05) — live vs pre-saved separated WAVs.
Stage B: 3 airplane + 3 background samples from the webdataset test split,
         classified with and without separation.
Stage C: For each Stage B sample, print waveform RMS/peak/length and TUSS
         airplane-head RMS/peak (probe asymmetry without editing harness).

Print-only. No edits to test_pipeline.py. No WAVs written.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _sf_load(path, *args, **kwargs):
    """soundfile-backed replacement for torchaudio.load (env has no FFmpeg)."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    # soundfile returns (T, C); torchaudio expects (C, T).
    wav = torch.from_numpy(data.T.copy())
    return wav, sr


torchaudio.load = _sf_load  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation_functions.test_pipeline import ValidationPipeline  # noqa: E402
from src.label_loading.coi_labels import (  # noqa: E402
    AIRPLANE_SYNONYMS,
    is_coi_label,
)

# ---------- CONFIG ----------
SEP_CHECKPOINT = str(
    PROJECT_ROOT / "src/models/tuss/checkpoints/multi_coi_14_05/best_model.pt"
)
DEMO_DIR = PROJECT_ROOT / "src/validation_functions/demo_output"
WEBDATASET_DIR = Path("/home/bendm/Thesis/project/data/webdataset")
TEST_SHARDS = sorted(WEBDATASET_DIR.glob("test-*.tar"))
N_AIRPLANE = 3
N_BACKGROUND = 3
SNR_DB = 0.0
PRIMARY_CLASSIFIER = "pann_finetuned"
# ---------------------------


def hr(title: str = "") -> None:
    bar = "=" * 78
    if title:
        print(f"\n{bar}\n{title}\n{bar}")
    else:
        print(bar)


def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.float() ** 2)).item())


def peak(x: torch.Tensor) -> float:
    return float(x.abs().max().item())


def load_wav_resampled(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def align_len(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def list_webdataset_samples():
    """Yield (shard_path, basename, json_dict) for each test sample."""
    for shard in TEST_SHARDS:
        with tarfile.open(shard, "r") as tar:
            members = tar.getmembers()
            by_base = {}
            for m in members:
                if not m.isfile():
                    continue
                base, _, ext = m.name.rpartition(".")
                by_base.setdefault(base, {})[ext] = m
            for base, exts in by_base.items():
                if "json" not in exts or ("flac" not in exts and "wav" not in exts):
                    continue
                jf = tar.extractfile(exts["json"])
                if jf is None:
                    continue
                meta = json.load(jf)
                yield shard, base, exts, meta


def pick_samples():
    """Pick N_AIRPLANE ESC50 airplane + N_AIRPLANE aerosonicdb airplane + N_BACKGROUND ESC50 BG."""
    airplane_esc50: List[Tuple[Path, str, dict, dict]] = []
    airplane_aero: List[Tuple[Path, str, dict, dict]] = []
    background: List[Tuple[Path, str, dict, dict]] = []
    seen_bg_labels: set = set()

    for shard, base, exts, meta in list_webdataset_samples():
        label = str(meta.get("label", "")).strip()
        dataset = str(meta.get("dataset", "")).strip()
        is_plane = is_coi_label(label, AIRPLANE_SYNONYMS)

        if is_plane and dataset == "esc50" and len(airplane_esc50) < N_AIRPLANE:
            airplane_esc50.append((shard, base, exts, meta))
        elif is_plane and dataset == "aerosonicdb" and len(airplane_aero) < N_AIRPLANE:
            airplane_aero.append((shard, base, exts, meta))
        elif (
            len(background) < N_BACKGROUND
            and dataset == "esc50"
            and not is_plane
            and label not in seen_bg_labels
        ):
            background.append((shard, base, exts, meta))
            seen_bg_labels.add(label)

        if (
            len(airplane_esc50) >= N_AIRPLANE
            and len(airplane_aero) >= N_AIRPLANE
            and len(background) >= N_BACKGROUND
        ):
            break

    return airplane_esc50 + airplane_aero, background


def extract_audio_to_tmp(shard: Path, exts: dict) -> Path:
    ext_key = "flac" if "flac" in exts else "wav"
    member = exts[ext_key]
    with tarfile.open(shard, "r") as tar:
        f = tar.extractfile(member)
        assert f is not None
        data = f.read()
    suffix = "." + ext_key
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def classify_no_sep(pipeline: ValidationPipeline, waveform: torch.Tensor):
    return pipeline._classify_recording(waveform, pipeline._classify)


def classify_with_sep(pipeline: ValidationPipeline, waveform: torch.Tensor):
    """Split → separate → take COI head → concat → classify."""
    segs = pipeline._split_into_segments(waveform)
    batch = torch.stack(segs).to(pipeline.device)
    sep_out = pipeline._separate_batch(batch)  # (B, n_sources, T)
    coi_idx = pipeline._get_coi_head_index()
    coi_chunks = [sep_out[i, coi_idx].detach().cpu() for i in range(sep_out.shape[0])]
    coi_wave = pipeline._concat_with_crossfade(coi_chunks)
    pred, conf = pipeline._classify_recording(coi_wave, pipeline._classify)
    return pred, conf, coi_wave


def chunk_confidences(pipeline: ValidationPipeline, waveform: torch.Tensor) -> List[float]:
    """Return per-chunk confidences from the primary classifier on the whole recording."""
    # Use _classify_recording internals path: replicate to also return per-chunk confs.
    # We re-run lightweight: resample → chunk → predict_batch → confs.
    target_sr = pipeline.classifier_sample_rate
    cls_seg = pipeline.classifier_segment_samples
    wav = waveform.detach().cpu()
    if pipeline.sample_rate != target_sr:
        wav = torchaudio.functional.resample(wav, pipeline.sample_rate, target_sr)
    T = wav.shape[0]
    n_chunks = max(1, (T + cls_seg - 1) // cls_seg)
    chunks = []
    for c in range(n_chunks):
        start = c * cls_seg
        ch = wav[start : start + cls_seg]
        if ch.shape[0] < cls_seg:
            ch = torch.nn.functional.pad(ch, (0, cls_seg - ch.shape[0]))
        chunks.append(ch)
    batch = torch.stack(chunks).to(pipeline.device)
    if hasattr(pipeline.classifier, "predict_batch"):
        _, confs = pipeline.classifier.predict_batch(batch)
        return [float(c.item()) for c in confs]
    confs = []
    for ch in batch:
        _, c = pipeline.classifier(ch)
        confs.append(float(c))
    return confs


def fmt_conf_list(xs: List[float]) -> str:
    return "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"


# ========================================================================
# Stage A
# ========================================================================
def stage_a(pipeline: ValidationPipeline) -> None:
    hr("STAGE A — TUSS demo (multi_coi_14_05) live vs saved")

    mix_path = DEMO_DIR / "mixture.wav"
    saved_plane_path = DEMO_DIR / "separated_plane.wav"
    saved_bg_path = DEMO_DIR / "separated_background.wav"

    for p in (mix_path, saved_plane_path, saved_bg_path):
        if not p.exists():
            print(f"MISSING: {p}")
            return

    mixture = pipeline._load_full_audio(str(mix_path))
    print(f"mixture            len={mixture.shape[0]:>8}  rms={rms(mixture):.4f}  peak={peak(mixture):.4f}")

    # Live separation through harness
    segs = pipeline._split_into_segments(mixture)
    batch = torch.stack(segs).to(pipeline.device)
    sep_out = pipeline._separate_batch(batch)  # (B, n_sources, T)
    n_sources = sep_out.shape[1]
    coi_idx = pipeline._get_coi_head_index()
    print(f"separator: n_sources={n_sources}  coi_head_index={coi_idx}")

    head_waves = {}
    for h in range(n_sources):
        chunks = [sep_out[i, h].detach().cpu() for i in range(sep_out.shape[0])]
        head_waves[h] = pipeline._concat_with_crossfade(chunks)
        print(
            f"live head {h}        len={head_waves[h].shape[0]:>8}  "
            f"rms={rms(head_waves[h]):.4f}  peak={peak(head_waves[h]):.4f}"
        )

    saved_plane = load_wav_resampled(saved_plane_path, pipeline.sample_rate)
    saved_bg = load_wav_resampled(saved_bg_path, pipeline.sample_rate)
    print(
        f"saved plane WAV     len={saved_plane.shape[0]:>8}  "
        f"rms={rms(saved_plane):.4f}  peak={peak(saved_plane):.4f}"
    )
    print(
        f"saved bg WAV        len={saved_bg.shape[0]:>8}  "
        f"rms={rms(saved_bg):.4f}  peak={peak(saved_bg):.4f}"
    )

    # Live-vs-saved divergence (head 0 = airplane, head 2 = background per demo order).
    # We'll match by name regardless of head index by trying all and reporting best/worst.
    def divergence(live: torch.Tensor, saved: torch.Tensor) -> float:
        a, b = align_len(live, saved)
        return float((a - b).abs().max().item())

    print("\nLive-vs-saved max|diff|:")
    for h in range(n_sources):
        d_plane = divergence(head_waves[h], saved_plane)
        d_bg = divergence(head_waves[h], saved_bg)
        print(f"  head {h}: vs saved_plane={d_plane:.4f}  vs saved_bg={d_bg:.4f}")

    # Per-chunk confidences
    print("\nPer-chunk airplane confidences (PANN @ hardcoded 0.5):")
    print(f"  raw mixture          conf={fmt_conf_list(chunk_confidences(pipeline, mixture))}")
    for h in range(n_sources):
        print(
            f"  live head {h}          conf={fmt_conf_list(chunk_confidences(pipeline, head_waves[h]))}"
        )
    print(f"  saved plane WAV      conf={fmt_conf_list(chunk_confidences(pipeline, saved_plane))}")
    print(f"  saved bg WAV         conf={fmt_conf_list(chunk_confidences(pipeline, saved_bg))}")

    print("\nRecording-level predictions (any-positive aggregation):")
    p, c = classify_no_sep(pipeline, mixture)
    print(f"  raw mixture          pred={p}  max_conf={c:.3f}")
    for h in range(n_sources):
        p, c = classify_no_sep(pipeline, head_waves[h])
        print(f"  live head {h}          pred={p}  max_conf={c:.3f}")
    p, c = classify_no_sep(pipeline, saved_plane)
    print(f"  saved plane WAV      pred={p}  max_conf={c:.3f}")
    p, c = classify_no_sep(pipeline, saved_bg)
    print(f"  saved bg WAV         pred={p}  max_conf={c:.3f}")


# ========================================================================
# Stage B + C
# ========================================================================
def stage_bc(pipeline: ValidationPipeline) -> None:
    hr("STAGE B — Webdataset spot check (3 airplane + 3 background)")

    airplane_samples, bg_samples = pick_samples()
    print(f"Picked {len(airplane_samples)} airplane + {len(bg_samples)} background samples")

    rows = []  # (gt, label, dataset, nosep_pred, nosep_conf, sep_pred, sep_conf)

    def process(meta_tuple, gt: int) -> Tuple[torch.Tensor, torch.Tensor]:
        shard, base, exts, meta = meta_tuple
        label = meta.get("label", "?")
        dataset = meta.get("dataset", "?")
        start_time = meta.get("start_time")
        end_time = meta.get("end_time")
        tmp_path = extract_audio_to_tmp(shard, exts)
        try:
            wav = pipeline._load_labeled_audio(str(tmp_path), start_time, end_time)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        nosep_p, nosep_c = classify_no_sep(pipeline, wav)
        sep_p, sep_c, coi_wave = classify_with_sep(pipeline, wav)

        nosep_chunk = chunk_confidences(pipeline, wav)
        sep_chunk = chunk_confidences(pipeline, coi_wave)

        print(
            f"\n[{('AIRPLANE' if gt == 1 else 'BG     ')}] "
            f"dataset={dataset!r:<20} label={label!r:<30} "
            f"len={wav.shape[0]:>7}  rms_in={rms(wav):.4f}  peak_in={peak(wav):.4f}"
        )
        print(
            f"   no-sep: pred={nosep_p} max_conf={nosep_c:.3f} chunks={fmt_conf_list(nosep_chunk)}"
        )
        print(
            f"   w/-sep: pred={sep_p}  max_conf={sep_c:.3f} chunks={fmt_conf_list(sep_chunk)}"
        )
        # Stage C diagnostics
        ratio = rms(coi_wave) / max(rms(wav), 1e-12)
        print(
            f"   TUSS airplane head: len={coi_wave.shape[0]:>7}  "
            f"rms_out={rms(coi_wave):.4f}  peak_out={peak(coi_wave):.4f}  "
            f"rms_ratio(out/in)={ratio:.3f}"
        )

        rows.append((gt, label, dataset, nosep_p, nosep_c, sep_p, sep_c))
        return wav, coi_wave

    airplane_waves = []
    bg_waves = []
    for s in airplane_samples:
        w, _ = process(s, gt=1)
        airplane_waves.append(w)
    for s in bg_samples:
        w, _ = process(s, gt=0)
        bg_waves.append(w)

    # Synthetic mixture: first airplane + first background
    if airplane_waves and bg_waves:
        hr("STAGE B — Synthetic mixture (first airplane + first BG @ SNR=0 dB)")
        a = airplane_waves[0]
        b = bg_waves[0]
        n = min(a.shape[0], b.shape[0])
        mix, actual = pipeline._create_mixture_rms(a[:n], b[:n], SNR_DB)
        print(
            f"mixture len={mix.shape[0]:>7}  rms={rms(mix):.4f}  peak={peak(mix):.4f}  "
            f"actual_snr={actual:.2f} dB"
        )
        nosep_p, nosep_c = classify_no_sep(pipeline, mix)
        sep_p, sep_c, coi_wave = classify_with_sep(pipeline, mix)
        print(f"   no-sep: pred={nosep_p} max_conf={nosep_c:.3f} "
              f"chunks={fmt_conf_list(chunk_confidences(pipeline, mix))}")
        print(f"   w/-sep: pred={sep_p}  max_conf={sep_c:.3f} "
              f"chunks={fmt_conf_list(chunk_confidences(pipeline, coi_wave))}")
        ratio = rms(coi_wave) / max(rms(mix), 1e-12)
        print(f"   TUSS airplane head: rms_out={rms(coi_wave):.4f}  peak_out={peak(coi_wave):.4f}  "
              f"rms_ratio={ratio:.3f}")
        rows.append((1, "<synthetic_mix>", "synthetic", nosep_p, nosep_c, sep_p, sep_c))

    # Summary table
    hr("STAGE B — Summary table")
    print(f"{'gt':<3} {'pred_nosep':<11} {'pred_sep':<9} {'conf_nosep':<11} {'conf_sep':<9} "
          f"{'dataset':<14} label")
    for gt, label, dataset, np_, nc, sp, sc in rows:
        print(
            f"{gt:<3} {np_:<11} {sp:<9} {nc:<11.3f} {sc:<9.3f} "
            f"{str(dataset)[:14]:<14} {label}"
        )

    # 2x2 agreement matrices
    def cm(rows_subset):
        tp = sum(1 for r in rows_subset if r[0] == 1 and r[3] == 1)  # nosep
        fn = sum(1 for r in rows_subset if r[0] == 1 and r[3] == 0)
        fp = sum(1 for r in rows_subset if r[0] == 0 and r[3] == 1)
        tn = sum(1 for r in rows_subset if r[0] == 0 and r[3] == 0)
        tp_s = sum(1 for r in rows_subset if r[0] == 1 and r[5] == 1)  # sep
        fn_s = sum(1 for r in rows_subset if r[0] == 1 and r[5] == 0)
        fp_s = sum(1 for r in rows_subset if r[0] == 0 and r[5] == 1)
        tn_s = sum(1 for r in rows_subset if r[0] == 0 and r[5] == 0)
        return (tp, fn, fp, tn), (tp_s, fn_s, fp_s, tn_s)

    (n_tp, n_fn, n_fp, n_tn), (s_tp, s_fn, s_fp, s_tn) = cm(rows)
    print("\nConfusion (no-sep vs sep):")
    print(f"  no-sep: TP={n_tp} FN={n_fn} FP={n_fp} TN={n_tn}")
    print(f"  w/-sep: TP={s_tp} FN={s_fn} FP={s_fp} TN={s_tn}")


def main() -> None:
    hr("Loading pipeline")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = "cuda:1"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    pipeline = ValidationPipeline(device=device)
    pipeline.load_models(
        sep_checkpoint=SEP_CHECKPOINT,
        cls_weights=None,
        classifier_type=PRIMARY_CLASSIFIER,
        use_tuss=True,
        tuss_coi_prompt="airplane",
        tuss_bg_prompt="background",
        use_clapsep=False,
        use_ast_finetuned=False,
        use_bird_mae=False,
        use_audioprotopnet=False,
    )
    print(f"COI head index: {pipeline._get_coi_head_index()}")
    print(f"Classifier sample rate: {pipeline.classifier_sample_rate}")
    print(f"Classifier segment samples: {pipeline.classifier_segment_samples}")

    stage_a(pipeline)
    stage_bc(pipeline)
    hr("DONE")


if __name__ == "__main__":
    main()

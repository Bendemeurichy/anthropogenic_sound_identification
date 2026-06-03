"""
Shared training utilities extracted from duplicated code across sudomrmrf/ and tuss/.

Consolidates:
    - Logging/redirection helpers for detached/pythonw runs
    - set_seed for reproducibility
    - robust_load_audio for cross-backend audio I/O
    - get_audio_info for metadata extraction with multiple fallbacks
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


# =============================================================================
# Logging helpers for detached / pythonw runs
# =============================================================================


class _AutoFlushStream:
    """Wraps a file object and flushes after every write."""

    def __init__(self, f):
        self._f = f

    def write(self, text):
        self._f.write(text)
        self._f.flush()

    def flush(self):
        self._f.flush()

    def __getattr__(self, name):
        return getattr(self._f, name)


def _redirect_to_log(log_path: Path) -> None:
    """Redirect sys.stdout and sys.stderr to *log_path* (append mode).

    Only used when stdout is truly absent (pythonw launched without
    -RedirectStandardOutput).  When the caller has already redirected stdout
    to a file we leave that handle in place and just ensure it auto-flushes.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    stream = _AutoFlushStream(fh)
    sys.stdout = stream  # type: ignore[assignment]
    sys.stderr = stream  # type: ignore[assignment]


def _ensure_autoflush() -> None:
    """Wrap sys.stdout/stderr with _AutoFlushStream if they exist but are not
    a TTY.  This prevents Python's default block-buffering from holding output
    in an 8 KB buffer when stdout is redirected to a file (e.g. via
    Start-Process -RedirectStandardOutput)."""
    if sys.stdout is not None and not _is_tty():
        sys.stdout = _AutoFlushStream(sys.stdout)  # type: ignore[assignment]
    if sys.stderr is not None and not _is_tty():
        sys.stderr = _AutoFlushStream(sys.stderr)  # type: ignore[assignment]


def _is_tty() -> bool:
    """Return True only when stdout is an interactive terminal."""
    try:
        return sys.stdout is not None and sys.stdout.isatty()
    except Exception:
        return False


class StepProgress:
    """Minimal tqdm replacement that writes plain-text progress lines.

    Used automatically when stdout is not a TTY (e.g. pythonw or a log file
    redirect).  Prints one line every *log_every_pct* percent of steps so the
    log file stays readable without being flooded.
    """

    def __init__(
        self,
        iterable,
        desc: str = "",
        total: Optional[int] = None,
        log_every_pct: float = 5.0,
    ):
        self._it = iterable
        self._desc = desc
        try:
            self._total = total if total is not None else len(iterable)
        except TypeError:
            self._total = None
        self._n = 0
        self._postfix: dict = {}
        self._start = time.time()
        if self._total:
            self._log_every = max(1, int(self._total * log_every_pct / 100))
        else:
            self._log_every = 10

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __iter__(self):
        for item in self._it:
            yield item
            self._n += 1
            should_log = (self._n % self._log_every == 0) or (
                self._total is not None and self._n == self._total
            )
            if should_log:
                self._print_line()

    def _print_line(self):
        elapsed = time.time() - self._start
        rate = self._n / elapsed if elapsed > 0 else 0.0
        if self._total:
            pct = 100.0 * self._n / self._total
            eta = (self._total - self._n) / rate if rate > 0 else 0.0
            progress = f"{self._n}/{self._total} ({pct:.0f}%) eta {eta:.0f}s"
        else:
            progress = f"{self._n} steps"
        postfix = "  ".join(f"{k}={v}" for k, v in self._postfix.items())
        sep = "  |  " if postfix else ""
        print(f"  [{self._desc}] {progress}  {elapsed:.0f}s elapsed{sep}{postfix}")

    def set_postfix(self, refresh=True, **kwargs):
        self._postfix = {k: v for k, v in kwargs.items()}


def progress_bar(iterable, desc: str = "", total: Optional[int] = None, **tqdm_kwargs):
    """Return a tqdm bar when interactive, StepProgress otherwise."""
    if _is_tty():
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            leave=False,
            ascii=True,
            ncols=100,
            **tqdm_kwargs,
        )
    return StepProgress(iterable, desc=desc, total=total)


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Audio I/O helpers
# =============================================================================


def robust_load_audio(path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    """Load audio robustly: try torchaudio first, then switch backend to
    soundfile, and finally fall back to the soundfile Python API.

    Returns (waveform, sample_rate) where waveform has shape (channels, frames).
    """
    p = str(path)
    try:
        return torchaudio.load(p)
    except Exception as e:
        try:
            torchaudio.set_audio_backend("soundfile")
            return torchaudio.load(p)
        except Exception:
            try:
                import soundfile as sf
            except Exception:
                raise RuntimeError(
                    "Failed to load audio with torchaudio and 'soundfile' is not installed. "
                    "Install pysoundfile (`pip install soundfile`) or ensure FFmpeg/torchcodec compatibility."
                ) from e

            data, sr = sf.read(p, always_2d=True)
            wav = torch.from_numpy(np.asarray(data).T).to(torch.float32)
            return wav, int(sr)


def get_audio_info(filepath: str) -> Tuple[int, int]:
    """Return (sample_rate, num_frames) using the best available backend.

    Attempts in order:
        1. torchaudio.info
        2. soundfile (pysoundfile)
        3. torchaudio.load + file-size heuristic (WAV only)

    Raises RuntimeError if none of the strategies succeed.
    """
    if hasattr(torchaudio, "info"):
        try:
            info = torchaudio.info(filepath)
            return int(info.sample_rate), int(info.num_frames)
        except Exception:
            pass

    try:
        import soundfile as sf

        info = sf.info(filepath)
        return int(info.samplerate), int(info.frames)
    except Exception:
        pass

    try:
        waveform, sr = torchaudio.load(filepath, num_frames=1)
        n_channels = waveform.shape[0]
        file_size = os.path.getsize(filepath)
        bytes_per_sample = 2
        try:
            with open(filepath, "rb") as f:
                header = f.read(44)
            if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                import struct as _struct

                bits_per_sample = _struct.unpack_from("<H", header, 34)[0]
                bytes_per_sample = max(1, bits_per_sample // 8)
        except Exception:
            pass
        data_bytes = max(0, file_size - 44)
        total_frames = data_bytes // (bytes_per_sample * max(1, n_channels))
        return int(sr), max(1, total_frames)
    except Exception:
        pass

    raise RuntimeError(
        f"Cannot determine audio info for {filepath}: "
        "torchaudio.info, soundfile, and torchaudio.load all failed"
    )

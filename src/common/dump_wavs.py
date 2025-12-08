"""
dump_audioset_wavs_streaming.py

- Uses polars.scan_parquet + collect(streaming=True)
- Single-threaded (friendly to WSL1)
- Writes OUT_DIR/{index}.wav at 48000 Hz (resamples if scipy available)
"""

import os
import io
import sys
import argparse
import warnings
from typing import Optional, Union

import numpy as np
import polars as pl
import soundfile as sf
from scipy.signal import resample_poly

# ensure polars uses 1 thread (helps on WSL1)
os.environ.setdefault("POLARS_MAX_THREADS", "1")


# -----------------------
# Your decode function
# -----------------------
def decode_binary_audio(audio_input: Optional[Union[bytes, dict]]) -> np.ndarray:
    """
    Decodes audio from either binary bytes or HuggingFace audio dict format.

    HuggingFace format: {'bytes': b'...', 'path': '...', 'sampling_rate': 48000}
    """
    if audio_input is None:
        return np.array([], dtype=np.float32)

    try:
        # Handle HuggingFace audio dict format
        if isinstance(audio_input, dict):
            if "bytes" in audio_input:
                audio_binary = audio_input["bytes"]
            elif "path" in audio_input and audio_input["path"]:
                audio_data, _ = sf.read(audio_input["path"], dtype="float32")
                return audio_data
            else:
                print(
                    f"⚠️  Audio dict missing 'bytes' or 'path' key: {audio_input.keys()}"
                )
                return np.array([], dtype=np.float32)
        else:
            # Handle raw bytes
            audio_binary = audio_input

        if not audio_binary or len(audio_binary) == 0:
            return np.array([], dtype=np.float32)

        # Decode the audio bytes
        audio_data, _ = sf.read(io.BytesIO(audio_binary), dtype="float32")
        return audio_data

    except Exception as e:
        print(f"❌ Error decoding audio in validation: {e}")
        print(f"   Input type: {type(audio_input)}")
        if isinstance(audio_input, dict):
            print(f"   Dict keys: {audio_input.keys()}")
        return np.array([], dtype=np.float32)


# -----------------------
# Helpers
# -----------------------
def get_samplerate_from_audio_input(
    audio_input: Optional[Union[bytes, dict]],
) -> Optional[int]:
    try:
        if audio_input is None:
            return None
        if isinstance(audio_input, dict):
            if "sampling_rate" in audio_input and audio_input["sampling_rate"]:
                return int(audio_input["sampling_rate"])
            if "path" in audio_input and audio_input["path"]:
                info = sf.info(audio_input["path"])
                return info.samplerate
            if "bytes" in audio_input and audio_input["bytes"]:
                bio = io.BytesIO(audio_input["bytes"])
                info = sf.info(bio)
                return info.samplerate
        else:
            bio = io.BytesIO(audio_input)
            info = sf.info(bio)
            return info.samplerate
    except Exception:
        return None
    return None


def resample_to_target(x: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return x

    squeeze_after = False
    if x.ndim == 1:
        x = x[:, None]
        squeeze_after = True
    gcd = np.gcd(src_sr, tgt_sr)
    up = tgt_sr // gcd
    down = src_sr // gcd
    channels = []
    for ch in range(x.shape[1]):
        y = resample_poly(x[:, ch], up, down)
        channels.append(y)
    y = np.stack(channels, axis=1)
    if squeeze_after:
        y = np.squeeze(y, axis=1)
    return y.astype(np.float32)


def safe_filename(name: str) -> str:
    keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    cleaned = "".join(c for c in name if c in keep).strip().replace(" ", "_")
    if not cleaned:
        cleaned = "audio"

    cut_username = ("_").join(cleaned.split("_")[5:])

    return cut_username


# -----------------------
# Main: use scan_parquet + collect(streaming=True)
# -----------------------
def process_parquet_stream(
    pq_path: str,
    out_dir: str,
    target_sr: int = 48000,
    index_col: str = "index",
    audio_col: str = "audio",
    overwrite: bool = False,
):
    # Lazy scan, select only needed cols, collect in streaming mode (Polars will try to stream)
    try:
        lf = pl.scan_parquet(pq_path).select([pl.col(index_col), pl.col(audio_col)])
    except Exception as e:
        print(f"Failed to scan {pq_path}: {e}")
        return

    try:
        df = lf.collect(streaming=True)  # streaming collect
    except Exception as e:
        print(f"Collect failed for {pq_path}: {e}")
        return

    if df.height == 0:
        print(f"No rows in {pq_path}")
        return

    # iterate rows (dicts)
    for row in df.to_dicts():
        idx_val = row.get(index_col)
        audio_input = row.get(audio_col)
        if idx_val is None:
            print(f"Skipping row with no index in {pq_path}")
            continue

        filename_split = safe_filename(str(idx_val))

        split = filename_split.split("_")[0]

        out_name = ("_").join(filename_split.split("_")[1:]) + ".wav"
        out_path = os.path.join(out_dir + f"/{split}", out_name)

        if os.path.exists(out_path) and not overwrite:
            # skip existing to avoid re-writing large dataset
            print(f"Skipping existing: {out_path}")
            continue

        # decode
        samples = decode_binary_audio(audio_input)
        if samples.size == 0:
            print(f"Empty audio for index {idx_val}; skipping.")
            continue

        # sr detection best-effort
        src_sr = get_samplerate_from_audio_input(audio_input)
        if src_sr is None:
            warnings.warn(
                f"Unknown source sample rate for index {idx_val}. Assuming {target_sr} Hz."
            )
            src_sr = target_sr

        # resample if necessary
        try:
            if src_sr != target_sr:
                samples = resample_to_target(samples, src_sr, target_sr)
                write_sr = target_sr
            else:
                write_sr = target_sr

            # write wav (16-bit PCM)
            sf.write(out_path, samples, samplerate=write_sr, subtype="PCM_16")
            print(f"Wrote: {out_path} (sr={write_sr})")

        except Exception as e:
            print(f"Error writing {out_path}: {e}")


def find_parquet_files(input_dir: str):
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".parquet"):
                yield os.path.join(root, f)


def main():
    parser = argparse.ArgumentParser(
        description="Stream-decoding .parquet audio bytes -> wav files at 48kHz (uses polars.scan_parquet)."
    )
    parser.add_argument("input_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--index-col", default="index")
    parser.add_argument("--audio-col", default="audio")
    parser.add_argument("--target-sr", type=int, default=48000)
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing wavs"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pq_files = list(find_parquet_files(args.input_dir))
    if not pq_files:
        print("No parquet files found.")
        return

    print(f"Found {len(pq_files)} parquet files. scipy resampling available")
    for pq in pq_files:
        print("Processing:", pq)
        process_parquet_stream(
            pq,
            args.out_dir,
            target_sr=args.target_sr,
            index_col=args.index_col,
            audio_col=args.audio_col,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

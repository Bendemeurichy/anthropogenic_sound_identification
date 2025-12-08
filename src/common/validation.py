import glob
import polars as pl
import numpy as np
import io
import soundfile as sf
from typing import Optional, Union
from pathlib import Path
import sounddevice as sd

# --- Configuration ---
PARQUET_PATH = "/mnt/acoustserv/audioset_strong/**/*.parquet"
AUDIO_COLUMN_NAME = "audio"


def check_parquet_path(path_pattern: str) -> bool:
    """
    Checks if the parquet path pattern resolves to any files.
    """
    base_path = path_pattern.split("**")[0].rstrip("/")

    if not Path(base_path).exists():
        print(f"❌ Base directory does not exist: {base_path}")
        return False

    matching_files = glob.glob(path_pattern, recursive=True)
    if not matching_files:
        print(f"❌ No parquet files found matching pattern: {path_pattern}")
        return False

    print(f"✅ Found {len(matching_files)} parquet file(s) matching the pattern")
    return True


def decode_binary_audio(audio_input: Optional[Union[bytes, dict]]) -> np.ndarray:
    """
    Decodes audio from either binary bytes or HuggingFace audio dict format.

    HuggingFace format: {'bytes': b'...', 'path': '...', 'sampling_rate': 16000}
    """
    if audio_input is None:
        return np.array([], dtype=np.float32)

    try:
        # Handle HuggingFace audio dict format
        if isinstance(audio_input, dict):
            if "bytes" in audio_input:
                audio_binary = audio_input["bytes"]
            elif "path" in audio_input and audio_input["path"]:
                # If there's a path but no bytes, try reading from path
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


def play_audio(audio_array: np.ndarray, sample_rate: int):
    """
    Play audio array through system speakers.
    """
    print(f"\n🔊 Playing audio at {sample_rate}Hz...")
    print(f"   Duration: {len(audio_array) / sample_rate:.2f} seconds")
    print("   Press Ctrl+C to stop playback early")

    try:
        sd.play(audio_array, sample_rate)
        sd.wait()  # Wait until playback is finished
        print("✅ Playback complete")
    except KeyboardInterrupt:
        sd.stop()
        print("\n⏹️  Playback stopped by user")
    except Exception as e:
        print(f"❌ Error during playback: {e}")


def validate_first_row_decoding():
    if not check_parquet_path(PARQUET_PATH):
        print("Validation aborted due to invalid path.")
        return

    print(f"Scanning first row from: {PARQUET_PATH}")

    # Lazily scan and take first row
    lazy_df = pl.scan_parquet(PARQUET_PATH).head(1)

    # Apply the UDF and keep the index column (filename)
    validation_df = lazy_df.with_columns(
        pl.col(AUDIO_COLUMN_NAME)
        .map_elements(decode_binary_audio, return_dtype=pl.Object)
        .alias("decoded_audio_array")
    ).select("decoded_audio_array", "index")

    # Collect result
    result = validation_df.collect()

    # Validation Check
    print("\n--- Validation Result ---")
    if result.height == 1:
        decoded_array = result[0, 0]
        filename = result[0, 1] if result.width > 1 else None

        # Print filename and extract video ID if available
        if filename:
            print(f"📁 Filename: {filename}")
            # Extract video ID from filename (format like: Y123abc456_10.000_20.000.flac)
            video_id = (
                filename.split("_")[-1] if "_" in filename else filename.split(".")[0]
            )
            print(f"🎬 Video ID: {video_id}")
            print(f"   YouTube URL: https://www.youtube.com/watch?v={video_id}")

        if isinstance(decoded_array, np.ndarray) and decoded_array.size > 0:
            print(f"✅ SUCCESS! First row decoded correctly.")
            print(f"   Decoded Array Shape: {decoded_array.shape}")
            print(f"   Decoded Array Dtype: {decoded_array.dtype}")
            print(f"   Total Samples: {decoded_array.shape[0]}")

            # Test with different sample rates
            print("\n📊 Duration estimates at different sample rates:")
            sample_rates = [16000, 22050, 32000, 40000, 44100, 48000]

            for sr in sample_rates:
                duration = decoded_array.shape[0] / sr
                marker = "  ⭐" if abs(duration - 10.0) < 0.5 else ""  # Mark ~10s clips
                print(f"   @ {sr:5d}Hz: {duration:6.2f}s{marker}")

            # Ask user which sample rate to test
            print("\n" + "=" * 50)
            print("Would you like to play the audio to validate?")
            print("=" * 50)

            while True:
                user_input = input(
                    "\nEnter sample rate to test (or 'q' to quit): "
                ).strip()

                if user_input.lower() == "q":
                    print("Exiting without playback.")
                    break

                try:
                    test_sr = int(user_input)
                    if test_sr < 8000 or test_sr > 96000:
                        print("⚠️  Sample rate seems unusual. Try between 8000-96000 Hz")
                        continue

                    play_audio(decoded_array, test_sr)

                    # Ask if they want to try another rate
                    again = input("\nTry another sample rate? (y/n): ").strip().lower()
                    if again != "y":
                        break

                except ValueError:
                    print("❌ Invalid input. Please enter a number or 'q' to quit.")
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break

        else:
            print("❌ FAILURE! Decoding returned an empty array or was corrupted.")
            print(f"   Returned type: {type(decoded_array)}")
            if isinstance(decoded_array, np.ndarray):
                print(f"   Array size: {decoded_array.size}")
    else:
        print("❌ FAILURE! Could not read the first row from the Parquet files.")


if __name__ == "__main__":
    validate_first_row_decoding()

import dask.dataframe as dd
import numpy as np
import io
import soundfile as sf
import os
from dask.distributed import Client, LocalCluster  # NEW IMPORTS
from typing import Optional

# --- Configuration ---
PARQUET_PATH = "./audioset_strong/**/*.parquet"
AUDIO_COLUMN_NAME = "audio_bytes"
OUTPUT_PATH = "decoded_audioset_with_arrays/"


# --- UDF (User-Defined Function) for Decoding ---
def decode_binary_audio(audio_binary: Optional[bytes]) -> np.ndarray:
    """
    Decodes a binary audio blob into a float32 NumPy array.
    """
    if audio_binary is None or len(audio_binary) == 0:
        return np.array([], dtype=np.float32)

    try:
        audio_data, _ = sf.read(io.BytesIO(audio_binary), dtype="float32")
        return audio_data
    except Exception:
        # Return an empty array on failure
        return np.array([], dtype=np.float32)


# --- Dask Parallel Execution Pipeline ---
def run_dask_decoding_pipeline():
    # When this function runs, it will automatically connect to the Client
    # created in the main block.

    print(f"Scanning Parquet files from: {PARQUET_PATH}")

    # 1. Lazily read the Parquet files
    ddf = dd.read_parquet(PARQUET_PATH)

    print(f"Initial Partitions: {ddf.npartitions}")

    # 2. Apply the UDF to the binary column in parallel
    decoded_series = ddf[AUDIO_COLUMN_NAME].apply(
        decode_binary_audio, meta=("decoded_audio_array", "object")
    )

    # 3. Add the new decoded column to the DataFrame
    ddf_with_audio = ddf.assign(decoded_audio_array=decoded_series)

    print("\n--- Starting Full Parallel Computation and Save ---")

    # 4. EXECUTE THE SAVE OPERATION
    # This triggers the simultaneous reading, decoding, and writing across all workers.
    ddf_with_audio.to_parquet(OUTPUT_PATH, write_index=False, overwrite=True)

    print(f"\n✅ FULL DATASET DECODED AND SAVED.")
    print(f"Results are saved to the directory: {OUTPUT_PATH}")


if __name__ == "__main__":
    # 1. Determine the optimal number of workers
    # For CPU-bound tasks like decoding, using one thread per worker (and one worker per core)
    # is generally the fastest configuration.
    N_CORES = os.cpu_count() or 4
    N_WORKERS = N_CORES

    print(f"Setting up Dask Local Cluster with {N_WORKERS} workers...")

    # 2. Set up the Dask Client and LocalCluster using a context manager
    with Client(n_workers=N_WORKERS, threads_per_worker=1) as client:
        print(f"Dask Dashboard link: {client.dashboard_link}")

        # 3. Run the main pipeline function
        run_dask_decoding_pipeline()

    print("\nDask Client and Cluster gracefully shut down.")

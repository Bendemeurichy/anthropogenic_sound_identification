"""
Simple utility to download individual AudioSet files using audioset_strong_download.
"""

from audioset_strong_download import Downloader
from pathlib import Path
import pandas as pd


def download_single_file(
    youtube_id: str,
    start_seconds: float,
    end_seconds: float,
    positive_labels: str,
    output_dir: str,
    split_type: str = "train",
    cookies_path: str | None = None,
) -> bool:
    """
    Download a single AudioSet file by YouTube ID with time segment and labels.

    Args:
        youtube_id: YouTube video ID (e.g., "Y1234567890")
        start_seconds: Start time of the segment in seconds
        end_seconds: End time of the segment in seconds
        positive_labels: Comma-separated string of labels for this segment
        output_dir: Directory to save the downloaded file
        split_type: 'train' or 'eval' split
        cookies_path: Path to cookies.txt file for authentication

    Returns:
        True if download succeeded, False otherwise
    """
    try:
        # Initialize downloader
        downloader = Downloader(
            root_path=output_dir,
            labels=["Aircraft"],  # Dummy label, not used for download_file
            n_jobs=1,
            dataset_ver="strong",
            download_type=split_type,
            copy_and_replicate=False,
            cookies=cookies_path,
        )

        # Download the file with segment info
        downloader.download_file(
            ytid=youtube_id,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            positive_labels=positive_labels,
        )
        return True
    except Exception as e:
        print(f"  Failed to download {youtube_id}: {e}")
        return False


def download_missing_files_from_df(
    df_missing,
    output_dir: str,
    split_type: str = "train",
    cookies_path: str | None = None,
) -> tuple[int, int]:
    """
    Download multiple missing files using metadata from dataframe.

    Args:
        df_missing: DataFrame with columns: filename, start_time, end_time, label
        output_dir: Directory to save downloaded files
        split_type: 'train' or 'eval' split
        cookies_path: Path to cookies.txt file

    Returns:
        Tuple of (successful_count, failed_count)
    """
    if df_missing.empty:
        return 0, 0

    print(f"\nDownloading {len(df_missing)} missing files...")
    successful = 0
    failed = 0

    for _, row in df_missing.iterrows():
        # Extract YouTube ID from filename (remove path and .wav extension)
        youtube_id = Path(row["filename"]).stem

        # change label to mid
        display_names_df = pd.read_csv(
            "../../data/metadata/audioset/mid_to_display_name.tsv",
            sep="\t",
            header=None,
            names=["mid", "display_name"],
        )
        label_mapping = display_names_df.set_index("display_name")["mid"].to_dict()
        label = row.get("label", "")
        if isinstance(label, list):
            mids = [label_mapping.get(l, l) for l in label]
            positive_labels = ",".join(mids)
        else:
            positive_labels = label_mapping.get(label, str(label))

        # Get time boundaries and labels
        start_time = float(row.get("start_time", 0))
        end_time = float(row.get("end_time", 10))  # Default 10 seconds if not specified

        # Convert label to string format (handle both single labels and lists)
        label = row.get("label", "")
        if isinstance(label, list):
            positive_labels = ",".join(label)
        else:
            positive_labels = str(label)

        print(f"  Downloading {youtube_id} ({start_time}s-{end_time}s)...")
        if download_single_file(
            youtube_id,
            start_time,
            end_time,
            positive_labels,
            output_dir,
            split_type,
            cookies_path,
        ):
            successful += 1
        else:
            failed += 1

    print(f"\nDownload complete: {successful} succeeded, {failed} failed")
    return successful, failed


def download_missing_files(
    missing_files: list[str],
    output_dir: str,
    split_type: str = "train",
    cookies_path: str | None = None,
    format: str = "wav",
    quality: int = 10,
) -> tuple[int, int]:
    """
    Legacy function for downloading files by path only (without metadata).
    Uses default values for start_time, end_time, and labels.

    For better results, use download_missing_files_from_df() with full metadata.

    Args:
        missing_files: List of file paths to download
        output_dir: Directory to save downloaded files
        split_type: 'train' or 'eval' split
        cookies_path: Path to cookies.txt file
        format: Audio format (unused, kept for compatibility)
        quality: Audio quality (unused, kept for compatibility)

    Returns:
        Tuple of (successful_count, failed_count)
    """
    if not missing_files:
        return 0, 0

    print(f"\nDownloading {len(missing_files)} missing files...")
    print("⚠️  Warning: Downloading without metadata, using default segment (0-10s)")
    successful = 0
    failed = 0

    for filepath in missing_files:
        # Extract YouTube ID from filepath (remove path and .wav extension)
        youtube_id = Path(filepath).stem

        print(f"  Downloading {youtube_id}...")
        # Use default values since we don't have metadata
        if download_single_file(
            youtube_id, 0.0, 10.0, "Unknown", output_dir, split_type, cookies_path
        ):
            successful += 1
        else:
            failed += 1

    print(f"\nDownload complete: {successful} succeeded, {failed} failed")
    return successful, failed

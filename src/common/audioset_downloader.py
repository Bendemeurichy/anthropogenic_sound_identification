"""
Simple utility to download individual AudioSet files using audioset_strong_download.
"""

from audioset_strong_download import Downloader
from pathlib import Path


def download_single_file(
    youtube_id: str,
    output_dir: str,
    split_type: str = "train",
    cookies_path: str | None = None,
    format: str = "wav",
    quality: int = 10,
) -> bool:
    """
    Download a single AudioSet file by YouTube ID.

    Args:
        youtube_id: YouTube video ID (e.g., "Y1234567890")
        output_dir: Directory to save the downloaded file
        split_type: 'train' or 'eval' split
        cookies_path: Path to cookies.txt file for authentication
        format: Audio format ('wav', 'mp3', etc.)
        quality: Audio quality (0-10, where 10 is best)

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

        # Download the file
        downloader.download_file(youtube_id=youtube_id, format=format, quality=quality)
        return True
    except Exception as e:
        print(f"  Failed to download {youtube_id}: {e}")
        return False


def download_missing_files(
    missing_files: list[str],
    output_dir: str,
    split_type: str = "train",
    cookies_path: str | None = None,
    format: str = "wav",
    quality: int = 10,
) -> tuple[int, int]:
    """
    Download multiple missing files.

    Args:
        missing_files: List of file paths or YouTube IDs to download
        output_dir: Directory to save downloaded files
        split_type: 'train' or 'eval' split
        cookies_path: Path to cookies.txt file
        format: Audio format
        quality: Audio quality (0-10)

    Returns:
        Tuple of (successful_count, failed_count)
    """
    if not missing_files:
        return 0, 0

    print(f"\nDownloading {len(missing_files)} missing files...")
    successful = 0
    failed = 0

    for filepath in missing_files:
        # Extract YouTube ID from filepath (remove path and .wav extension)
        youtube_id = Path(filepath).stem

        print(f"  Downloading {youtube_id}...")
        if download_single_file(
            youtube_id, output_dir, split_type, cookies_path, format, quality
        ):
            successful += 1
        else:
            failed += 1

    print(f"\nDownload complete: {successful} succeeded, {failed} failed")
    return successful, failed

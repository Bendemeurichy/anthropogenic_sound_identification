"""
Audio file validation utilities.

Provides functions to validate audio files before training to catch
corrupted files, missing files, and unsupported formats early.

Can be used by any audio model (PANN, YAMNet, AST, etc.).
"""

import pandas as pd
import torchaudio
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm


def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a file is a proper audio file that can be loaded.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        (is_valid, error_message) tuple where:
            - is_valid: True if file can be loaded, False otherwise
            - error_message: Empty string if valid, error description otherwise
    """
    try:
        if not Path(file_path).exists():
            return False, f"File not found: {file_path}"
        
        # Try to load the file
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Check if waveform is valid
        if waveform.numel() == 0:
            return False, "Empty waveform"
        
        # Check for NaN or inf values
        if not waveform.isfinite().all():
            return False, "Waveform contains NaN or inf values"
        
        # Check sample rate is reasonable
        if sample_rate <= 0 or sample_rate > 200000:
            return False, f"Invalid sample rate: {sample_rate}"
        
        return True, ""
        
    except Exception as e:
        return False, str(e)


def validate_dataset_files(
    df: pd.DataFrame,
    filename_column: str = "filename",
    verbose: bool = True,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Validate all audio files in a dataframe and report issues.
    
    Args:
        df: DataFrame with audio file paths
        filename_column: Column name containing file paths
        verbose: Whether to print detailed error messages
        show_progress: Whether to show a progress bar
        
    Returns:
        DataFrame with only valid files (invalid files removed)
    """
    if filename_column not in df.columns:
        raise ValueError(f"Column '{filename_column}' not found in DataFrame")
    
    invalid_files = []
    valid_indices = []
    
    # Create iterator with optional progress bar
    iterator = df.itertuples()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Validating audio files")
    
    for row in iterator:
        file_path = getattr(row, filename_column)
        is_valid, error_msg = validate_audio_file(file_path)
        
        if is_valid:
            valid_indices.append(row.Index)
        else:
            invalid_files.append((file_path, error_msg))
            if verbose:
                print(f"Invalid file: {file_path} - {error_msg}")
    
    if invalid_files:
        print(
            f"\nFound {len(invalid_files)} invalid audio files "
            f"out of {len(df)} total files"
        )
        print(f"Filtering dataset to {len(valid_indices)} valid files")
    else:
        print(f"All {len(df)} audio files are valid")
    
    return df.loc[valid_indices].reset_index(drop=True)


def get_invalid_files(
    df: pd.DataFrame, filename_column: str = "filename"
) -> List[Tuple[str, str]]:
    """
    Get list of invalid files without modifying the dataframe.
    
    Args:
        df: DataFrame with audio file paths
        filename_column: Column name containing file paths
        
    Returns:
        List of (file_path, error_message) tuples for invalid files
    """
    invalid_files = []
    
    for _, row in df.iterrows():
        file_path = row[filename_column]
        is_valid, error_msg = validate_audio_file(file_path)
        
        if not is_valid:
            invalid_files.append((file_path, error_msg))
    
    return invalid_files


def check_file_exists(df: pd.DataFrame, filename_column: str = "filename") -> pd.DataFrame:
    """
    Quick check for file existence (faster than full validation).
    
    Args:
        df: DataFrame with audio file paths
        filename_column: Column name containing file paths
        
    Returns:
        DataFrame with 'file_exists' column added
    """
    df = df.copy()
    df['file_exists'] = df[filename_column].apply(lambda f: Path(f).exists())
    return df

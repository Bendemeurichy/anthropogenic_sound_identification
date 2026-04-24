"""
High-quality audio resampling utilities for multi-rate datasets.

This module provides standardized, high-quality resampling across all models
(TUSS, SuDoRM-RF, CLAPSep) to ensure consistent audio quality when handling
datasets with different native sample rates (ESC-50: 44.1k, FreeSound: 24k,
BirdSet: 48k, AeroSonic: 22k, etc.).

References:
    [1] Smith, J.O. "Digital Audio Resampling Home Page", CCRMA, Stanford
        https://ccrma.stanford.edu/~jos/resample/Kaiser_Window.html
    [2] resampy (bmcfee) - Used by librosa for high-quality resampling
        https://github.com/bmcfee/resampy
"""

import torch
import torch.nn.functional as F
import torchaudio

# =============================================================================
# High-Quality Resampling Parameters
# =============================================================================
# These parameters match the 'kaiser_best' configuration from resampy (used by
# librosa), which is considered the gold standard for high-quality audio
# resampling in machine learning applications.
#
# Why these values?
# -----------------
# resampling_method: 'sinc_interp_kaiser'
#   - Uses windowed-sinc interpolation with Kaiser window
#   - Near-ideal frequency domain reconstruction
#
# beta: 14.769
#   - Standard value cited by J.O. Smith III (CCRMA/Stanford) [1]
#   - Provides ~96 dB of stopband rejection
#   - Used in: resampy, SoX, libsamplerate
#
# lowpass_filter_width: 64
#   - Half-width of the filter kernel in samples
#   - From resampy's kaiser_best default [2]
#   - Higher = sharper frequency cutoff, better anti-aliasing
#
# rolloff: 0.99
#   - Cutoff at 99% of Nyquist frequency
#   - More conservative than resampy's 0.95 (sharper anti-aliasing)
#   - Reduces aliasing when upsampling low-rate audio
# =============================================================================

RESAMPLE_METHOD = "sinc_interp_kaiser"
RESAMPLE_BETA = 14.769
RESAMPLE_LOWPASS_WIDTH = 64
RESAMPLE_ROLLOFF = 0.99


def create_high_quality_resampler(
    orig_sr: int,
    target_sr: int,
    method: str = RESAMPLE_METHOD,
    beta: float = RESAMPLE_BETA,
    lowpass_filter_width: int = RESAMPLE_LOWPASS_WIDTH,
    rolloff: float = RESAMPLE_ROLLOFF,
) -> torchaudio.transforms.Resample:
    """
    Create a high-quality audio resampler.

    Uses Kaiser windowed sinc interpolation for best anti-aliasing performance,
    matching the quality of resampy's kaiser_best mode (used by librosa).

    Note on edge artifacts:
        Windowed sinc interpolation can produce artifacts at signal boundaries
        because the filter needs samples on both sides of each point. To mitigate
        this, consider using resample_with_padding() which adds reflection padding
        before resampling.

    Args:
        orig_sr: Original sample rate (Hz)
        target_sr: Target sample rate (Hz)
        method: Resampling method (default: 'sinc_interp_kaiser')
        beta: Kaiser window beta parameter (default: 14.769 for ~96 dB rejection)
        lowpass_filter_width: Filter kernel half-width (default: 64)
        rolloff: Cutoff frequency as fraction of Nyquist (default: 0.99)

    Returns:
        Configured Resample transform

    Example:
        >>> resampler = create_high_quality_resampler(22050, 48000)
        >>> upsampled = resampler(waveform)  # (channels, samples)

    References:
        [1] Smith, J.O. "Digital Audio Resampling", CCRMA, Stanford
        [2] resampy kaiser_best mode (librosa's resampler)
    """
    return torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
        resampling_method=method,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
        beta=beta,
    )


def resample_with_padding(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """
    Resample audio with padding to eliminate edge artifacts.

    Windowed sinc interpolation produces artifacts at signal boundaries because
    the filter needs samples on both sides of each interpolation point. This
    function adds reflection padding before resampling and trims it afterward,
    ensuring clean edges in the resampled output.

    This is especially important for:
    - Audio separation models (artifacts visible in spectrograms can affect models)
    - Fragment-based processing (edge artifacts at every boundary)
    - High-quality audio output (visible transients in spectrograms)

    Args:
        waveform: Audio tensor (..., time) - can be any shape with time as last dim
        orig_sr: Original sample rate (Hz)
        target_sr: Target sample rate (Hz)
        pad_mode: Padding mode - 'reflect' (default), 'replicate', or 'constant'
                  'reflect': mirrors signal at boundaries (best for most audio)
                  'replicate': extends edge values (good for DC-offset signals)
                  'constant': zero-padding (not recommended, causes discontinuities)

    Returns:
        Resampled waveform with same shape except time dimension scaled by ratio

    Example:
        >>> # Shape: (batch, channels, time)
        >>> waveform = torch.randn(4, 2, 32000)
        >>> resampled = resample_with_padding(waveform, 32000, 48000)
        >>> # Shape: (4, 2, 48000)

    Note:
        Padding length is set to 2x the filter width (128 samples) which is
        sufficient for the Kaiser window with lowpass_filter_width=64.
    """
    if orig_sr == target_sr:
        return waveform

    # Calculate padding needed (2x filter width for safety)
    # lowpass_filter_width=64 means 64 samples on each side
    pad_samples = RESAMPLE_LOWPASS_WIDTH * 2

    # Add padding to avoid edge artifacts
    # F.pad expects (left, right) for 1D or (..., left, right) for last dim
    if pad_mode == "reflect":
        # Reflect mode: mirror the signal at boundaries
        waveform_padded = F.pad(waveform, (pad_samples, pad_samples), mode="reflect")
    elif pad_mode == "replicate":
        # Replicate mode: extend edge values
        waveform_padded = F.pad(waveform, (pad_samples, pad_samples), mode="replicate")
    elif pad_mode == "constant":
        # Zero padding (not recommended but available)
        waveform_padded = F.pad(waveform, (pad_samples, pad_samples), mode="constant", value=0)
    else:
        raise ValueError(f"Unknown pad_mode: {pad_mode}. Use 'reflect', 'replicate', or 'constant'")

    # Create resampler and process
    resampler = create_high_quality_resampler(orig_sr, target_sr)
    resampled_padded = resampler(waveform_padded)

    # Calculate how many samples to trim from resampled output
    ratio = target_sr / orig_sr
    trim_samples = int(pad_samples * ratio)

    # Trim padding from output
    resampled = resampled_padded[..., trim_samples:-trim_samples]

    return resampled


def create_low_quality_resampler(
    orig_sr: int,
    target_sr: int,
    method: str = "sinc_interp_hann",
    lowpass_filter_width: int = 16,
    rolloff: float = 0.8,
) -> torchaudio.transforms.Resample:
    """
    Create a low-quality audio resampler for augmentations.

    Uses simpler methods to avoid memory issues during data augmentation.

    Args:
        orig_sr: Original sample rate (Hz)
        target_sr: Target sample rate (Hz)
        method: Resampling method ('sinc_interp_hann', etc.)
        lowpass_filter_width: Filter kernel half-width (default: 16)
        rolloff: Cutoff frequency as fraction of Nyquist (default: 0.8)

    Returns:
        Configured Resample transform
    """
    return torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
        resampling_method=method,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
    )


class ResamplerCache:
    """
    Cache of high-quality audio resamplers for efficient multi-rate audio loading.

    Maintains a limited-size cache of Resample transforms keyed by
    (orig_sr, target_sr) pairs. This avoids repeatedly creating expensive
    resampler objects during training/inference.

    Args:
        max_size: Maximum number of resamplers to cache (default: 8)

    Example:
        >>> cache = ResamplerCache(max_size=8)
        >>> waveform = cache.resample(waveform, orig_sr=22050, target_sr=48000)
    """

    def __init__(self, max_size: int = 8):
        self._cache: dict[tuple[int, int], torchaudio.transforms.Resample] = {}
        self._max_size = max_size

    def get_resampler(
        self, orig_sr: int, target_sr: int
    ) -> torchaudio.transforms.Resample:
        """
        Get or create a high-quality resampler for the given sample rate pair.

        Args:
            orig_sr: Original sample rate (Hz)
            target_sr: Target sample rate (Hz)

        Returns:
            Cached or newly created Resample transform
        """
        key = (orig_sr, target_sr)

        if key not in self._cache:
            # Evict oldest entry if cache is full (FIFO policy)
            if len(self._cache) >= self._max_size:
                self._cache.pop(next(iter(self._cache)))

            # Create new high-quality resampler
            self._cache[key] = create_high_quality_resampler(orig_sr, target_sr)

        return self._cache[key]

    def resample(self, waveform, orig_sr: int, target_sr: int):
        """
        Resample waveform using cached high-quality resampler.

        Note: This method does NOT use padding. For fragment-based processing
        or when edge artifacts are visible in spectrograms, use resample_with_padding()
        instead (available as a standalone function in this module).

        Args:
            waveform: Audio tensor (any shape, resampling applied to last dimension)
            orig_sr: Original sample rate (Hz)
            target_sr: Target sample rate (Hz)

        Returns:
            Resampled waveform with same shape except last dimension
        """
        if orig_sr == target_sr:
            return waveform

        resampler = self.get_resampler(orig_sr, target_sr)
        return resampler(waveform)

    def resample_padded(
        self, waveform, orig_sr: int, target_sr: int, pad_mode: str = "reflect"
    ):
        """
        Resample waveform with padding to eliminate edge artifacts.

        This method uses reflection padding before resampling to avoid edge
        artifacts that are visible in spectrograms and can affect separation models.

        Args:
            waveform: Audio tensor (any shape, resampling applied to last dimension)
            orig_sr: Original sample rate (Hz)
            target_sr: Target sample rate (Hz)
            pad_mode: Padding mode ('reflect', 'replicate', or 'constant')

        Returns:
            Resampled waveform with clean edges
        """
        return resample_with_padding(waveform, orig_sr, target_sr, pad_mode)

    def clear(self):
        """Clear the resampler cache."""
        self._cache.clear()

    def __len__(self):
        """Return number of cached resamplers."""
        return len(self._cache)

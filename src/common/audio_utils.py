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


def create_low_quality_resampler(
    orig_sr: int,
    target_sr: int,
    method: str = "linear",
    lowpass_filter_width: int = 16,
    rolloff: float = 0.8,
) -> torchaudio.transforms.Resample:
    """
    Create a low-quality audio resampler for augmentations.

    Uses simpler methods to avoid memory issues during data augmentation.

    Args:
        orig_sr: Original sample rate (Hz)
        target_sr: Target sample rate (Hz)
        method: Resampling method ('linear', 'cubic', etc.)
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

    def clear(self):
        """Clear the resampler cache."""
        self._cache.clear()

    def __len__(self):
        """Return number of cached resamplers."""
        return len(self._cache)

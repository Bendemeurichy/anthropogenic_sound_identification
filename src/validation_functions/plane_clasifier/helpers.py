import tensorflow as tf
import resampy
import numpy as np
from data_config import DataLoaderConfig


def _to_mono(waveform: tf.Tensor) -> tf.Tensor:
    # decode_wav yields shape [samples, channels]
    if waveform.shape.ndims == 2:
        return tf.reduce_mean(waveform, axis=-1)
    return waveform


def _resample_audio(waveform_np: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Core resampling logic using resampy."""
    if rate_in == rate_out:
        return waveform_np
    return resampy.resample(waveform_np, rate_in, rate_out, filter="kaiser_best")


def _resample_tensor(
    waveform: tf.Tensor, rate_in: tf.Tensor, rate_out: int
) -> tf.Tensor:
    """Resample using resampy library wrapped in tf.py_function.
    rate_in: scalar int tensor (sample rate of input waveform)
    rate_out: python int (target sample rate)
    """

    def _resample_with_resampy(waveform_tensor, rate_in_tensor):
        # Convert TensorFlow tensors to NumPy arrays
        waveform_np = (
            waveform_tensor.numpy()
            if hasattr(waveform_tensor, "numpy")
            else np.array(waveform_tensor)
        )
        rate_in_np = (
            rate_in_tensor.numpy()
            if hasattr(rate_in_tensor, "numpy")
            else int(rate_in_tensor)
        )
        return _resample_audio(waveform_np, int(rate_in_np), rate_out)

    # Use tf.py_function to wrap the numpy/resampy operation
    resampled = tf.py_function(
        func=_resample_with_resampy, inp=[waveform, rate_in], Tout=waveform.dtype
    )

    # Set shape information since py_function loses it
    resampled.set_shape([None])

    return resampled


def normalize_audio_length(waveform: tf.Tensor, config: DataLoaderConfig) -> tf.Tensor:
    target_length = int(config.sample_rate * config.audio_duration)
    current_length = tf.shape(waveform)[0]

    def _pad():
        padding = target_length - current_length
        return tf.concat([waveform, tf.zeros([padding], dtype=waveform.dtype)], axis=0)

    def _crop():
        start = (current_length - target_length) // 2
        return waveform[start : start + target_length]

    def _ident():
        return waveform

    waveform = tf.case(
        [
            (current_length < target_length, _pad),
            (current_length > target_length, _crop),
        ],
        default=_ident,
    )
    waveform.set_shape([int(config.sample_rate * config.audio_duration)])
    return waveform


def random_crop_audio(waveform: tf.Tensor, config: DataLoaderConfig) -> tf.Tensor:
    target_length = int(config.sample_rate * config.audio_duration)
    current_length = tf.shape(waveform)[0]

    def _crop():
        max_start = current_length - target_length
        start = tf.random.uniform([], 0, max_start + 1, dtype=tf.int32)
        return waveform[start : start + target_length]

    return tf.cond(current_length > target_length, _crop, lambda: waveform)


def _augment_waveform(waveform: tf.Tensor, config: DataLoaderConfig) -> tf.Tensor:
    target_length = int(config.sample_rate * config.audio_duration)

    # Random crop/temporal jitter
    if (
        config.aug_time_stretch_prob > 0
        and tf.random.uniform([]) < config.aug_time_stretch_prob
    ):
        pad = target_length // 4
        extended = tf.concat(
            [
                tf.zeros([pad], dtype=waveform.dtype),
                waveform,
                tf.zeros([pad], dtype=waveform.dtype),
            ],
            axis=0,
        )
        waveform = random_crop_audio(extended, config)

    # Time stretch (via resampling)
    def _apply_time_stretch():
        rate = tf.random.uniform(
            [], config.aug_time_stretch_range[0], config.aug_time_stretch_range[1]
        )
        stretch_length = tf.cast(tf.cast(target_length, tf.float32) * rate, tf.int32)

        # Use resampy for time stretching
        def _stretch_with_resampy(waveform_tensor, stretch_len_tensor):
            # Convert TensorFlow tensors to NumPy arrays
            waveform_np = (
                waveform_tensor.numpy()
                if hasattr(waveform_tensor, "numpy")
                else np.array(waveform_tensor)
            )
            stretch_len_np = (
                stretch_len_tensor.numpy()
                if hasattr(stretch_len_tensor, "numpy")
                else int(stretch_len_tensor)
            )
            return _resample_audio(waveform_np, len(waveform_np), int(stretch_len_np))

        stretched = tf.py_function(
            func=_stretch_with_resampy,
            inp=[waveform, stretch_length],
            Tout=waveform.dtype,
        )
        stretched.set_shape([None])

        def _crop_stretched():
            start = (stretch_length - target_length) // 2
            return stretched[start : start + target_length]

        def _pad_stretched():
            padding = target_length - stretch_length
            pad_before = padding // 2
            pad_after = padding - pad_before
            return tf.concat(
                [
                    tf.zeros([pad_before], dtype=waveform.dtype),
                    stretched,
                    tf.zeros([pad_after], dtype=waveform.dtype),
                ],
                axis=0,
            )

        return tf.cond(stretch_length > target_length, _crop_stretched, _pad_stretched)

    if (
        config.aug_time_stretch_prob > 0
        and tf.random.uniform([]) < config.aug_time_stretch_prob
    ):
        waveform = _apply_time_stretch()

    # Add noise
    if config.aug_noise_prob > 0 and tf.random.uniform([]) < config.aug_noise_prob:
        noise = tf.random.normal(tf.shape(waveform), stddev=config.aug_noise_stddev)
        waveform = waveform + noise

    # Random gain
    if config.aug_gain_prob > 0 and tf.random.uniform([]) < config.aug_gain_prob:
        gain = tf.random.uniform([], config.aug_gain_range[0], config.aug_gain_range[1])
        waveform = waveform * gain

    waveform = tf.clip_by_value(waveform, -1.0, 1.0)
    waveform.set_shape([target_length])
    return waveform

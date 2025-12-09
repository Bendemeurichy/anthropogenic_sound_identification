import tensorflow as tf
import pandas as pd
import numpy as np
from data_config import DataLoaderConfig
from config import TrainingConfig
from helpers import (
    _to_mono,
    _resample_tensor,
    normalize_audio_length,
    _augment_waveform,
)
import typing as t
import math


class KerasAudioDataLoader:
    """Factory that turns a dataframe into tf.data.Datasets for Keras model.fit.

    Usage:
        loader = KerasAudioDataLoader(df, config)
        train_ds = loader.get_dataset("train", batch_size=32, augment=True)
        model.fit(train_ds, validation_data=val_ds, epochs=...)
    """

    def __init__(
        self, df: pd.DataFrame, config: t.Union[DataLoaderConfig, "TrainingConfig"]
    ):
        self.df = df.copy().reset_index(drop=True)
        if isinstance(config, DataLoaderConfig):
            self.config = config
        else:
            self.config = DataLoaderConfig(
                filename_column=getattr(config, "filename_column", "filename"),
                start_time_column=getattr(config, "start_time_column", "start_time"),
                end_time_column=getattr(config, "end_time_column", "end_time"),
                label_column=getattr(config, "label_column", "label"),
                split_column=getattr(config, "split_column", "split"),
                sample_rate=getattr(config, "sample_rate", 16000),
                audio_duration=getattr(config, "audio_duration", 5.0),
                split_long=getattr(config, "split_long", True),
                min_clip_length=getattr(config, "min_clip_length", 0.5),
                batch_size=getattr(config, "batch_size", 32),
                shuffle_buffer=getattr(config, "shuffle_buffer", 10000),
                use_augmentation=getattr(config, "use_augmentation", False),
                aug_time_stretch_prob=getattr(config, "aug_time_stretch_prob", 0.0),
                aug_time_stretch_range=getattr(
                    config, "aug_time_stretch_range", (0.9, 1.1)
                ),
                aug_noise_prob=getattr(config, "aug_noise_prob", 0.0),
                aug_noise_stddev=getattr(config, "aug_noise_stddev", 0.002),
                aug_gain_prob=getattr(config, "aug_gain_prob", 0.0),
                aug_gain_range=getattr(config, "aug_gain_range", (0.8, 1.2)),
            )

        # ensure columns exist
        required = [
            self.config.filename_column,
            self.config.start_time_column,
            self.config.end_time_column,
            self.config.label_column,
            self.config.split_column,
        ]
        for c in required:
            if c not in self.df.columns:
                raise ValueError(f"Dataframe missing required column: {c}")

        # Optionally expand long annotated segments into multiple fixed-length rows
        if self.config.split_long:
            expanded_rows = []
            for _, row in self.df.iterrows():
                fname = row[self.config.filename_column]
                s = row[self.config.start_time_column]
                e = row[self.config.end_time_column]
                lbl = row[self.config.label_column]
                sp = row[self.config.split_column]

                # If start/end are NaN we cannot split reliably; keep as-is
                if pd.isna(s) or pd.isna(e):
                    expanded_rows.append(
                        {
                            self.config.filename_column: str(fname),
                            self.config.start_time_column: s,
                            self.config.end_time_column: e,
                            self.config.label_column: int(lbl),
                            self.config.split_column: sp,
                        }
                    )
                    continue

                duration = float(e) - float(s)
                target = float(self.config.audio_duration)

                if duration <= target:
                    expanded_rows.append(
                        {
                            self.config.filename_column: str(fname),
                            self.config.start_time_column: float(s),
                            self.config.end_time_column: float(e),
                            self.config.label_column: int(lbl),
                            self.config.split_column: sp,
                        }
                    )
                else:
                    # number of full segments
                    n = int(math.ceil(duration / target))
                    for i in range(n):
                        new_s = float(s) + i * target
                        new_e = min(float(s) + (i + 1) * target, float(e))
                        # include even short remainders (they will be padded later)
                        if new_e - new_s >= self.config.min_clip_length or i < n:
                            expanded_rows.append(
                                {
                                    self.config.filename_column: str(fname),
                                    self.config.start_time_column: new_s,
                                    self.config.end_time_column: new_e,
                                    self.config.label_column: int(lbl),
                                    self.config.split_column: sp,
                                }
                            )
            self.df = pd.DataFrame(expanded_rows).reset_index(drop=True)

        # Prepare numpy arrays for tf.data
        self._filenames = self.df[self.config.filename_column].astype(str).values
        # start/end may contain NaN; keep them as np.nan floats
        self._starts = self.df[self.config.start_time_column].astype(np.float32).values
        self._ends = self.df[self.config.end_time_column].astype(np.float32).values
        self._labels = self.df[self.config.label_column].astype(np.int32).values
        self._splits = self.df[self.config.split_column].astype(str).values

    def _make_base_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(
            (self._filenames, self._starts, self._ends, self._labels, self._splits)
        )

    def _load_and_preprocess_tensors(
        self,
        filename: tf.Tensor,
        start_time: tf.Tensor,
        end_time: tf.Tensor,
        label: tf.Tensor,
        split: tf.Tensor,
    ) -> t.Tuple[tf.Tensor, tf.Tensor]:
        """Graph-friendly loader that takes tensor columns (no py_function).

        Extraction policy for fixed-length clips (audio_duration seconds):
        - If start_time/end_time are NaN: take center `audio_duration` seconds of full file (or pad).
        - If annotated segment >= audio_duration: when split_long was used the row already
          corresponds to at most audio_duration length; otherwise center-crop.
        - If annotated segment < audio_duration: take the available samples and pad to target length.

        Note: Invalid WAV files will cause errors. Use _safe_load_and_preprocess for error handling.
        """
        # filename: scalar tf.string
        audio_binary = tf.io.read_file(filename)

        # Try to decode WAV - this will raise an error for invalid files
        waveform, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
        waveform = _to_mono(waveform)
        sample_rate_scalar = tf.cast(tf.squeeze(sample_rate), tf.int32)

        # n_samples in original file
        n_samples = tf.shape(waveform)[0]

        # Resolve NaN starts/ends: if NaN, use full-file times
        start_time = tf.where(
            tf.math.is_nan(start_time),
            tf.constant(0.0, dtype=tf.float32),
            tf.cast(start_time, tf.float32),
        )
        end_time = tf.where(
            tf.math.is_nan(end_time),
            tf.cast(
                tf.cast(n_samples, tf.float32)
                / tf.cast(sample_rate_scalar, tf.float32),
                tf.float32,
            ),
            tf.cast(end_time, tf.float32),
        )

        # Convert to sample indices
        start_sample = tf.cast(
            tf.math.floor(start_time * tf.cast(sample_rate_scalar, tf.float32)),
            tf.int32,
        )
        end_sample = tf.cast(
            tf.math.floor(end_time * tf.cast(sample_rate_scalar, tf.float32)), tf.int32
        )
        length = tf.maximum(0, end_sample - start_sample)

        target_length = tf.cast(
            int(self.config.sample_rate * self.config.audio_duration), tf.int32
        )

        # If annotated segment longer than target and splitting wasn't done, center-crop; otherwise
        # slicing will already be <= target_length when split_long=True because init expanded rows.
        start_offset = tf.where(
            length > target_length,
            start_sample + ((length - target_length) // 2),
            start_sample,
        )
        slice_length = tf.where(length > target_length, target_length, length)

        waveform_segment = tf.slice(waveform, [start_offset], [slice_length])
        waveform_segment.set_shape([None])

        # Resample to target sample rate
        waveform_segment = tf.cond(
            tf.not_equal(
                sample_rate_scalar,
                tf.constant(self.config.sample_rate, dtype=sample_rate_scalar.dtype),
            ),
            lambda: _resample_tensor(
                waveform_segment, sample_rate_scalar, self.config.sample_rate
            ),
            lambda: waveform_segment,
        )

        # Ensure exactly target_length (pad or center-crop)
        waveform_segment = normalize_audio_length(waveform_segment, self.config)
        return waveform_segment, tf.cast(label, tf.int32)

    def get_dataset(
        self,
        split: t.Optional[str] = None,
        batch_size: t.Optional[int] = None,
        shuffle: bool = True,
        augment: bool = False,
        repeat: bool = False,
    ) -> tf.data.Dataset:
        """Return a tf.data.Dataset ready for Keras model.fit.

        Args:
            split: 'train' | 'val' | 'test' or None (None = all rows)
            batch_size: override config.batch_size
            shuffle: whether to shuffle
            augment: whether to apply augmentation (requires config.use_augmentation)
            repeat: whether to repeat dataset indefinitely (useful for fit with steps_per_epoch)
        """
        ds = self._make_base_dataset()

        if split is not None:
            split_str = str(split)
            ds = ds.filter(
                lambda f, s, e, lbl, sp: tf.equal(sp, tf.constant(split_str))
            )

        # Map with explicit tensor args to avoid py_function (graph-friendly)
        ds = ds.map(
            lambda f, st, en, lbl, sp: self._load_and_preprocess_tensors(
                f, st, en, lbl, sp
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Ignore errors from corrupted or invalid audio files
        ds = ds.apply(tf.data.experimental.ignore_errors())

        # If augmentation is enabled, create both original and augmented versions
        if augment and self.config.use_augmentation:
            # Original samples (no augmentation)
            ds_original = ds

            # Augmented samples
            ds_augmented = ds.map(
                lambda x, y: (_augment_waveform(x, self.config), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

            # Concatenate original + augmented to double the dataset
            ds = ds_original.concatenate(ds_augmented)

        # IMPORTANT: Repeat BEFORE shuffle to ensure proper epoch boundaries
        # This way each epoch reshuffles all samples in a new order
        if repeat:
            ds = ds.repeat()

        if shuffle:
            # reshuffle_each_iteration ensures different order each epoch when repeat=True
            ds = ds.shuffle(
                buffer_size=self.config.shuffle_buffer, reshuffle_each_iteration=True
            )

        final_batch = batch_size or self.config.batch_size
        # drop_remainder=False to ensure all samples are included (especially important for validation)
        ds = ds.batch(final_batch, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


def prepare_dataset(
    df: pd.DataFrame,
    config: t.Union[DataLoaderConfig, "TrainingConfig"],
    shuffle: bool = True,
    augment: bool = False,
    repeat: bool = False,
) -> tf.data.Dataset:
    """Compatibility wrapper matching the training code's expected API.

    This function mirrors the signature used in your `train.py` so the
    training pipeline can keep calling `prepare_dataset(train_df, config, ...)`.

    - `df` may contain rows from any split; `train.py` already passes split-specific
      dfs (train/val/test) so no filtering is performed here.
    - start_time/end_time may be NaN to indicate "use full file".
    - `augment` should be True only for training.
    - `repeat` should be True for training to enable proper steps_per_epoch.

    Returns:
        tf.data.Dataset yielding (waveform, label) where waveform has shape
        [batch, config.sample_rate * config.audio_duration] and dtype float32.
    """
    loader = KerasAudioDataLoader(df, config)
    batch_size = getattr(config, "batch_size", None)
    return loader.get_dataset(
        split=None,
        batch_size=batch_size,
        shuffle=shuffle,
        augment=augment,
        repeat=repeat,
    )

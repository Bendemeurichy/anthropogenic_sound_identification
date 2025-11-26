"""
Data loading and augmentation utilities for YAMNet finetuning.

This module provides functions for loading audio data, extracting features
using the pre-trained YAMNet model, and performing data augmentation.
"""

import os
import random
import glob

import numpy as np
import librosa
import resampy
import pandas as pd
from tqdm import tqdm


def random_augment_wav(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply random augmentations to audio.

    Augmentations include:
        - Time stretch (20% probability)
        - Resample (60% probability when no time stretch)
        - Volume change (always applied)
        - Random noise (50% probability)

    Args:
        audio: Audio waveform as numpy array.
        sr: Sample rate.

    Returns:
        Augmented audio waveform.
    """
    audio_aug = audio.copy()

    if np.random.uniform() > 0.8:  # Random stretch
        stretch = np.random.uniform(0.75, 1.5)
        audio_aug = librosa.effects.time_stretch(audio_aug, rate=stretch)
    elif np.random.uniform() > 0.2:  # Random resample
        new_sr = int(sr * np.random.uniform(0.9, 1.1))
        audio_aug = resampy.resample(audio_aug, sr, new_sr)

    # Random volume
    volume = np.random.uniform(0.65, 1.2)
    audio_aug = audio_aug * volume

    # Random noise
    if np.random.uniform() > 0.5:
        noise_ratio = 0.001
        audio_aug += np.random.uniform(
            -noise_ratio, noise_ratio, size=audio_aug.shape
        )

    return audio_aug


def balance_classes(
    features: list, labels: list
) -> tuple[list, list]:
    """Balance classes by removing samples randomly.

    Ensures equal representation of all classes by randomly removing
    samples from over-represented classes.

    Args:
        features: List of feature vectors.
        labels: List of corresponding labels.

    Returns:
        Tuple of balanced (features, labels).
    """
    # Randomize sample/label order
    idxs_random = list(range(len(labels)))
    random.shuffle(idxs_random)
    features = [features[i] for i in idxs_random]
    labels = [labels[i] for i in idxs_random]

    # Find minimum class count
    idx_labels, counts = np.unique(labels, return_counts=True)
    idx_locs_delete = []

    for idx in idx_labels:
        idx_locs = np.array(labels == np.array(idx)).nonzero()[0]

        if len(idx_locs) > counts.min():
            idx_locs_delete.extend(idx_locs[counts.min() :].tolist())

    idx_locs_keep = list(set(range(len(labels))) - set(idx_locs_delete))

    features = [features[i] for i in idx_locs_keep]
    labels = [labels[i] for i in idx_locs_keep]

    return features, labels


def extract_features(audio_list: list, yamnet_features) -> list:
    """Extract features from audio samples using YAMNet.

    Args:
        audio_list: List of audio waveforms.
        yamnet_features: YAMNet feature extraction model.

    Returns:
        List of extracted feature arrays.
    """
    features_extracted = []

    for audio in audio_list:
        _, _, dense_out, _ = yamnet_features.predict(
            np.reshape(audio, [1, -1]), steps=1
        )

        samples = [patch for patch in dense_out]
        features_extracted.append(samples)

    return features_extracted


def load_audio(
    path: str, sr: int, mono: bool = True
) -> tuple[np.ndarray, int]:
    """Load audio file.

    Args:
        path: Path to audio file.
        sr: Target sample rate.
        mono: Whether to convert to mono.

    Returns:
        Tuple of (waveform, sample_rate).
    """
    waveform, _ = librosa.load(path, sr=sr, mono=mono, dtype=np.float32)
    return waveform, sr


def save_features(
    path_audio: str,
    sr: int,
    audio_min_dur: float,
    path_data_train: str,
    patch_hop_seconds_str: str,
    num_augmentations: list,
    class_idx: int,
    yamnet_features,
) -> None:
    """Extract and save features for a single audio file.

    Args:
        path_audio: Path to audio file.
        sr: Sample rate.
        audio_min_dur: Minimum audio duration in seconds.
        path_data_train: Base path for saving features.
        patch_hop_seconds_str: Patch hop seconds as string (for filename).
        num_augmentations: Number of augmentations per class.
        class_idx: Index of the class for this audio.
        yamnet_features: YAMNet feature extraction model.
    """
    # Read audio waveform
    audio, _ = load_audio(path_audio, sr)

    # Check minimum duration
    audio_curr_dur = max(audio.shape) / sr
    if audio_curr_dur <= audio_min_dur:
        print(
            f"Audio duration is < {audio_min_dur} s for "
            f"{os.path.basename(path_audio)}. Continue."
        )
        return

    # Build feature path
    audio_filename = "_".join(path_audio.split(os.path.sep)[-2:]).split(".")[0]
    path_features = os.path.join(
        path_data_train,
        "features",
        "yamnet",
        audio_filename + "_features_" + patch_hop_seconds_str,
    )

    path_features_aug = [path_features + "_00"]

    if os.path.isfile(path_features_aug[-1] + ".npy"):
        audio_list, path_features_save = [], []
    else:
        audio_list = [audio]
        path_features_save = path_features_aug.copy()

    # Perform data augmentation on audio
    for idx_aug in range(num_augmentations[class_idx]):
        path_features_aug.append(path_features + f"_{idx_aug + 1:02d}")

        if not os.path.isfile(path_features_aug[-1] + ".npy"):
            print(f"Augmenting audio: {idx_aug + 1}")
            audio_aug = random_augment_wav(audio, sr)
            audio_list.append(audio_aug)
            path_features_save.append(path_features_aug[-1])

    if not audio_list:
        print(
            f"All features were previously extracted for {audio_filename}. "
            "Continue."
        )
        return

    features_save = extract_features(audio_list, yamnet_features)

    if len(features_save) != len(path_features_save):
        raise ValueError(
            "The number of extracted features is different from the number "
            "of features expected to be saved."
        )

    for features_tmp, path_features_tmp in zip(features_save, path_features_save):
        np.save(path_features_tmp, features_tmp)


def data_augmentation(
    data_path: str,
    classes: list,
    yamnet_features,
    num_augmentations: list | None = None,
    min_sample_seconds: float = 1.0,
    max_sample_seconds: float = 5.0,
    desired_sr: int = 16000,
) -> tuple[list, list]:
    """Load and augment training data.

    Args:
        data_path: Path to training data directory.
        classes: List of class names.
        yamnet_features: YAMNet feature extraction model.
        num_augmentations: Number of augmentations per class.
        min_sample_seconds: Minimum audio duration.
        max_sample_seconds: Maximum audio duration.
        desired_sr: Target sample rate.

    Returns:
        Tuple of (features, labels).
    """
    if num_augmentations is None:
        num_augmentations = [1, 1]

    print(f"Loading training data, number of augmentations = {num_augmentations}\n")
    samples = []
    labels = []

    min_wav_size = int(desired_sr * min_sample_seconds)
    max_wav_size = int(desired_sr * max_sample_seconds)

    for label_idx, class_name in enumerate(classes):
        label_dir = os.path.join(data_path, class_name)
        label_name = os.path.basename(label_dir)
        wavs = glob.glob(os.path.join(label_dir, "*.wav"))
        print(f"Loading {label_idx:<5}-> '{label_name}'")

        for wav_file in tqdm(wavs):
            waveform, _ = load_audio(wav_file, desired_sr)

            if len(waveform) < min_wav_size:
                print(f"\nIgnoring audio shorter than {min_sample_seconds} seconds")
                continue

            if len(waveform) > max_wav_size:
                waveform = waveform[:max_wav_size]
                print(f"\nIgnoring audio data after {max_sample_seconds} seconds")

            for aug_idx in range(1 + num_augmentations[label_idx]):
                aug_wav = waveform.copy()

                if aug_idx > 0:
                    aug_wav = random_augment_wav(aug_wav, desired_sr)

                _, _, dense_out, _ = yamnet_features.predict(
                    np.reshape(aug_wav, [1, -1]), steps=1
                )

                for patch in dense_out:
                    samples.append(patch)
                    labels.append(label_idx)

    return samples, labels


def load_features_from_csv(
    path_data_csv_file: str,
    path_data_train: str,
    patch_hop_seconds_str: str,
    num_augmentations: list,
) -> tuple[list, list]:
    """Load pre-extracted features from CSV file paths.

    Args:
        path_data_csv_file: Path to CSV file with audio paths and labels.
        path_data_train: Base path for feature files.
        patch_hop_seconds_str: Patch hop seconds as string.
        num_augmentations: Number of augmentations per class.

    Returns:
        Tuple of (features, labels).
    """
    print("Loading features and labels.\n")
    features = []
    labels = []

    df_data_csv = pd.read_csv(path_data_csv_file, header=None)
    classes = np.unique(df_data_csv.iloc[:, 1]).tolist()

    for class_idx, class_label in enumerate(classes):
        print(f"\nLoading class {class_label}\n")

        df_data_csv_class = df_data_csv[df_data_csv.iloc[:, 1] == class_label]
        path_audios = df_data_csv_class.iloc[:, 0].tolist()

        for path_audio in tqdm(path_audios):
            audio_filename = "_".join(
                path_audio.split(os.path.sep)[-2:]
            ).split(".")[0]
            path_features = os.path.join(
                path_data_train,
                "features",
                "yamnet",
                audio_filename + "_features_" + patch_hop_seconds_str,
            )

            for idx_aug in range(num_augmentations[class_idx] + 1):
                path_features_tmp = path_features + f"_{idx_aug:02d}.npy"

                if os.path.isfile(path_features_tmp):
                    features_tmp = np.load(path_features_tmp)

                    for feature in features_tmp:
                        features.append(feature)
                        labels.append(class_idx)
                else:
                    print(f"WARNING: Cannot find file \n{path_features_tmp}.")

    return features, labels

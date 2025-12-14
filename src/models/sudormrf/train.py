"""
Training script for sudormrf model with custom serperation head and loss function.
- Uses PITLossWrapper for automatic permutation handling
Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for aircraft (COI), 0 for background (non-COI)
"""

import os
import gc

# Pin to single GPU before importing torch (prevents multi-GPU OOM issues)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import torch

# Limit CPU threads to reduce memory overhead
torch.set_num_threads(4)
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from base.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF
from base.sudo_rm_rf.dnn.losses.sisdr import PITLossWrapper, PairwiseNegSDR
from seperation_head import wrap_model_for_coi
from multi_class_seperation import wrap_model_for_multiclass
from config import Config


from label_loading.sampler import get_coi, sample_non_coi
from label_loading.metadata_loader import (
    load_metadata_datasets,
    split_seperation_classification,
)


class AudioDataset(Dataset):
    """pytorch dataset handler for wav files."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str = "train",
        sample_rate: int = 16000,
        segment_length: float = 5.0,
        snr_range: tuple = (-5, 5),
        n_coi_classes: int = 1,
        augment: bool = True,
    ):
        self.split = split
        # Only keep necessary columns to reduce memory
        split_df = dataframe[dataframe["split"] == split][["filename", "label"]].copy()
        if "coi_class" in dataframe.columns:
            split_df["coi_class"] = dataframe.loc[split_df.index, "coi_class"]
        split_df = split_df.reset_index(drop=True)

        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.snr_range = snr_range
        self.n_coi_classes = n_coi_classes
        self.augment = augment

        # Store only file paths as lists instead of DataFrames (more memory efficient)
        coi_mask = split_df["label"] == 1
        self.coi_files = split_df.loc[coi_mask, "filename"].tolist()
        self.non_coi_files = split_df.loc[~coi_mask, "filename"].tolist()

        if n_coi_classes > 1:
            # Store file lists per class instead of DataFrames
            self.coi_by_class = [
                split_df.loc[
                    (split_df["label"] == 1) & (split_df.get("coi_class", 0) == i),
                    "filename",
                ].tolist()
                for i in range(n_coi_classes)
            ]
            print(
                f"{split} set: {[len(c) for c in self.coi_by_class]} per class, "
                f"{len(self.non_coi_files)} non-COI"
            )
        else:
            print(
                f"{split} set: {len(self.coi_files)} COI, {len(self.non_coi_files)} non-COI"
            )

        # Clear the temporary DataFrame
        del split_df
        gc.collect()

    def __len__(self):
        return len(self.coi_files)

    def load_and_preprocess(self, filepath):
        # Get file info first to determine if we need full load
        info = torchaudio.info(filepath)
        num_frames = info.num_frames
        file_sr = info.sample_rate

        # Calculate how many frames we need at original sample rate
        if file_sr != self.sample_rate:
            frames_needed = (
                int(self.segment_samples * file_sr / self.sample_rate) + 100
            )  # buffer
        else:
            frames_needed = self.segment_samples

        # Random offset for augmentation - calculated BEFORE loading
        if self.augment and num_frames > frames_needed:
            frame_offset = np.random.randint(0, num_frames - frames_needed)
        else:
            frame_offset = 0

        # Load only the frames we need (memory efficient)
        waveform, sr = torchaudio.load(
            filepath,
            frame_offset=frame_offset,
            num_frames=min(frames_needed, num_frames - frame_offset),
        )

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        # Pad if too short
        if waveform.shape[0] < self.segment_samples:
            padding = self.segment_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Trim to exact length
            waveform = waveform[: self.segment_samples]

        return waveform

    def create_mixture(self, source, noise, snr_db):
        """Create a mixture of source and noise at a given SNR."""
        source_power = torch.mean(source**2)
        noise_power = torch.mean(noise**2)

        snr_linear = 10 ** (snr_db / 10)
        scaling_factor = torch.sqrt(source_power / (snr_linear * noise_power + 1e-8))

        scaled_noise = noise * scaling_factor
        mixture = source + scaled_noise

        return mixture

    def normalize(self, waveform):
        """Normalize waveform to have zero mean and unit variance."""
        mean = torch.mean(waveform)
        std = torch.std(waveform) + 1e-8
        normalized_waveform = (waveform - mean) / std
        return normalized_waveform

    def __getitem__(self, idx):
        # Select COI class
        if self.n_coi_classes == 1:
            # Binary case
            coi_idx = idx % len(self.coi_files)
            coi_file = self.coi_files[coi_idx]
            coi_audio = self.load_and_preprocess(coi_file)

            # Create zero tensors for other classes
            sources = [coi_audio]
        else:
            # Multi-class case: randomly select one COI class
            class_idx = np.random.randint(0, self.n_coi_classes)
            coi_idx = np.random.randint(0, len(self.coi_by_class[class_idx]))
            coi_file = self.coi_by_class[class_idx][coi_idx]
            coi_audio = self.load_and_preprocess(coi_file)

            # Create sources list with zeros for other classes
            sources = [torch.zeros_like(coi_audio) for _ in range(self.n_coi_classes)]
            sources[class_idx] = coi_audio

        # Sample background
        noncoi_idx = np.random.randint(0, len(self.non_coi_files))
        noncoi_file = self.non_coi_files[noncoi_idx]
        background = self.load_and_preprocess(noncoi_file)

        # Normalize sources FIRST
        sources = [self.normalize(s) for s in sources]
        background = self.normalize(background)

        # Create mixture from normalized sources
        total_coi = sum(sources)
        snr_db = np.random.uniform(*self.snr_range)

        # Scale background for target SNR
        coi_power = torch.mean(total_coi**2) + 1e-8
        bg_power = torch.mean(background**2) + 1e-8
        snr_linear = 10 ** (snr_db / 10)
        scaling_factor = torch.sqrt(coi_power / (snr_linear * bg_power))
        scaled_background = background * scaling_factor

        # Mixture is now sum of sources
        mixture = total_coi + scaled_background

        # Update background to the scaled version (what's actually in mixture)
        sources.append(scaled_background)
        sources_tensor = torch.stack(sources, dim=0)

        return mixture, sources_tensor


def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad_norm=5.0):
    model.train()
    running_loss = 0.0
    n_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False, ascii=True, ncols=100)
    for mixtures, sources in progress_bar:
        mixtures = mixtures.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
        estimates = model(mixtures.unsqueeze(1))
        loss = criterion(estimates, sources)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        batch_size = mixtures.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size
        progress_bar.set_postfix(loss=loss.item())

        # Free memory explicitly
        del mixtures, sources, estimates, loss

    # Clear GPU cache after epoch
    if device != "cpu":
        torch.cuda.empty_cache()

    epoch_loss = running_loss / n_samples
    return epoch_loss


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    progress_bar = tqdm(
        dataloader, desc="Validation", leave=False, ascii=True, ncols=100
    )
    for mixtures, sources in progress_bar:
        mixtures = mixtures.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)

        estimates = model(mixtures.unsqueeze(1))
        loss = criterion(estimates, sources)

        batch_size = mixtures.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size
        progress_bar.set_postfix(loss=loss.item())

        # Free memory explicitly
        del mixtures, sources, estimates, loss

    # Clear GPU cache after validation
    if device != "cpu":
        torch.cuda.empty_cache()

    epoch_loss = running_loss / n_samples
    return epoch_loss


def create_dataloaders(config: Config):
    """Create train and validation dataloaders."""
    df = pd.read_csv(config.data.df_path)

    if config.data.n_coi_classes > 1 and "coi_class" not in df.columns:
        raise ValueError("Multi-class requires 'coi_class' column in dataframe")

    train_dataset = AudioDataset(
        df,
        split="train",
        sample_rate=config.data.sample_rate,
        segment_length=config.data.segment_length,
        snr_range=tuple(config.data.snr_range),
        n_coi_classes=config.data.n_coi_classes,
        augment=True,
    )
    val_dataset = AudioDataset(
        df,
        split="val",
        sample_rate=config.data.sample_rate,
        segment_length=config.data.segment_length,
        snr_range=tuple(config.data.snr_range),
        n_coi_classes=config.data.n_coi_classes,
        augment=False,
    )

    # Memory-optimized DataLoader settings
    # CRITICAL: num_workers=0 saves ~10-15GB RAM by avoiding worker process memory duplication
    # Each worker copies the dataset and loads audio into its own memory space
    # With num_workers=0, all loading happens in main process (slower but much less RAM)
    num_workers = config.training.num_workers

    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": num_workers,
        "pin_memory": num_workers == 0,  # Only pin if no workers (avoids extra copies)
    }

    # Only add worker-specific options if using workers
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 1

    # Delete dataframe after creating datasets
    del df
    gc.collect()

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_kwargs
    )

    # Force garbage collection after creating datasets
    gc.collect()

    return train_loader, val_loader


def get_model_args(model):
    """Extract model arguments for checkpoint saving."""
    return {
        "in_channels": getattr(model, "in_channels", None),
        "out_channels": getattr(model, "out_channels", None),
        "num_blocks": getattr(model, "num_blocks", None),
    }


def create_model(config: Config, compile_model: bool = False):
    """Create and wrap model with aircraft separation head."""
    if config.model.type == "improved":
        base_model = SuDORMRF(
            out_channels=config.model.out_channels,
            in_channels=config.model.in_channels,
            num_blocks=config.model.num_blocks,
            upsampling_depth=config.model.upsampling_depth,
            enc_kernel_size=config.model.enc_kernel_size,
            enc_num_basis=config.model.enc_num_basis,
            num_sources=2,
        )
    else:
        from base.sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 import (
            GroupCommSudoRmRf,
        )

        base_model = GroupCommSudoRmRf(
            out_channels=config.model.out_channels,
            in_channels=config.model.in_channels,
            num_blocks=config.model.num_blocks,
            upsampling_depth=config.model.upsampling_depth,
            enc_kernel_size=config.model.enc_kernel_size,
            enc_num_basis=config.model.enc_num_basis,
            num_sources=2,
        )

    if config.data.n_coi_classes > 1:
        print(
            f"Wrapping model for Multi-class separation with {config.data.n_coi_classes} classes."
        )
        model = wrap_model_for_multiclass(
            base_model, n_coi_classes=config.data.n_coi_classes
        )
    else:
        print("Wrapping model for Single COI separation.")
        model = wrap_model_for_coi(base_model)

    try:
        model = model.to(config.training.device)
    except Exception as e:
        print(f"Error moving model to device {config.training.device}: {e}")
        print("Moving model to CPU instead.")
        model = model.to("cpu")

    # Compile model for faster training (requires PyTorch 2.0+)
    # Note: torch.compile with inductor backend can be slow on WSL
    # Use 'eager' backend or disable compilation if experiencing slowness
    if hasattr(torch, "compile") and config.training.compile_model:
        backend = getattr(config.training, "compile_backend", "inductor")
        print(f"Compiling model with torch.compile() using '{backend}' backend...")
        model = torch.compile(model, backend=backend)
    elif hasattr(torch, "compile"):
        print("Model compilation disabled (set compile_model=True in config to enable)")

    return model


def train(config: Config):
    """Main training function."""
    # Setup
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Save config
    config.save(checkpoint_dir / "config.yaml")

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    gc.collect()  # Clean up after dataloader creation

    print("Creating model...")
    model = create_model(config)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")

    # Setup training
    pairwise_neg_sisdr = PairwiseNegSDR("sisdr")
    criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.training.num_epochs}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.training.device,
            config.training.clip_grad_norm,
        )
        history["train_loss"].append(train_loss)

        val_loss = validate_epoch(model, val_loader, criterion, config.training.device)
        history["val_loss"].append(val_loss)

        print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        scheduler.step(val_loss)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "model_args": get_model_args(model),
                "config": config.to_dict(),
                "history": history,
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.training.patience:
            print(f"\nEarly stopping after {epoch} epochs")
            break

    # Save training history
    import json

    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train aircraft sound separation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args = parser.parse_args()

    config_path = (
        Path(args.config)
        if isinstance(args.config, str)
        else Path("./training_config.yaml")
    )

    # Load config
    config = Config.from_yaml(config_path)
    print("Configuration:")
    print(f"  Model: {config.model.type} ({config.model.num_blocks} blocks)")
    print(f"  Device: {config.training.device}")

    # 1. Load all dataset metadata (same as plane classifier)
    print("\nLoading dataset metadata...")
    project_root = Path(__file__).parent.parent.parent.parent
    datasets_path = str(project_root / "data")
    audio_base_path = str(project_root.parent / "datasets")

    all_metadata = load_metadata_datasets(datasets_path, audio_base_path)

    # 2. Get the separation half (70%) of the data
    separation_metadata, _ = split_seperation_classification(all_metadata)

    print(f"Loaded {len(all_metadata)} total samples from all datasets")
    print(f"Using {len(separation_metadata)} samples for separation training (70%)")
    print(f"Datasets included: {separation_metadata['dataset'].unique()}")

    # 3. Define target classes (plane-related sounds) - same as plane classifier
    target_classes = [
        "airplane",
        "Aircraft",
        "Fixed-wing aircraft, airplane",
        "Aircraft engine",
        "Fixed-wing_aircraft_and_airplane",
    ]

    print(f"\nTarget classes: {target_classes}")

    # 4. Sample data to get balanced dataset
    print("\nSampling data with class-of-interest ratio...")
    coi_df = get_coi(separation_metadata, target_classes)
    sampled_df = sample_non_coi(
        separation_metadata,
        coi_df,
        coi_ratio=0.25,
    )

    # 5. Create binary labels: 1 for COI (plane), 0 for non-COI (background)
    sampled_df["label"] = sampled_df["label"].apply(
        lambda x: (
            1
            if (isinstance(x, list) and any(label in target_classes for label in x))
            or (isinstance(x, str) and x in target_classes)
            else 0
        )
    )

    # 6. Check for missing files
    print("\nChecking for missing audio files...")
    sampled_df["file_exists"] = sampled_df["filename"].apply(lambda f: Path(f).exists())
    missing_mask = ~sampled_df["file_exists"]

    if missing_mask.any():
        missing_count = missing_mask.sum()
        print(
            f"⚠️  Found {missing_count} missing files out of {len(sampled_df)} total samples"
        )
        print("Dropping samples with missing files...")
        sampled_df = sampled_df[sampled_df["file_exists"]].copy()

    sampled_df = sampled_df.drop(columns=["file_exists"])
    print(f"✅ Final dataset size: {len(sampled_df)} samples")

    # 7. Print dataset statistics
    print("\nDataset splits:")
    for split in ["train", "val", "test"]:
        split_df = sampled_df[sampled_df["split"] == split]
        coi_count = (split_df["label"] == 1).sum()
        non_coi_count = (split_df["label"] == 0).sum()
        print(
            f"  {split}: {len(split_df)} samples (COI: {coi_count}, non-COI: {non_coi_count})"
        )

    # 8. Save the prepared dataframe for reproducibility
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    df_save_path = checkpoint_dir / "separation_dataset.csv"
    sampled_df.to_csv(df_save_path, index=False)
    print(f"\nSaved prepared dataset to: {df_save_path}")

    # 9. Override the config's df_path with our prepared dataframe path
    config.data.df_path = str(df_save_path)
    print(f"  Data: {config.data.df_path}")

    # 10. Clean up large DataFrames before training
    del all_metadata, separation_metadata, coi_df, sampled_df
    gc.collect()
    print("Cleaned up metadata DataFrames from memory.")

    # 11. Train
    train(config)


if __name__ == "__main__":
    main()

"""
Training script for sudormrf model with custom serperation head and loss function.
- Uses asteroid's PITLossWrapper for automatic permutation handling
Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for aircraft (COI), 0 for background (non-COI)
"""

import sys
import torch
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
from src.models.sudormrf.seperation_head import wrap_model_for_coi
from src.models.sudormrf.multi_class_seperation import wrap_model_for_multiclass
from src.models.sudormrf.config import Config
from src.label_loading.sampler import get_coi, sample_non_coi
from src.label_loading.metadata_loader import (
    load_metadata_datasets,
    split_seperation_classification,
)

from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


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
        self.dataframe = dataframe[dataframe["split"] == split].reset_index(drop=True)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.snr_range = snr_range
        self.n_coi_classes = n_coi_classes
        self.augment = augment

        self.coi_df = self.dataframe[self.dataframe["label"] == 1].reset_index(
            drop=True
        )
        self.non_coi_df = self.dataframe[self.dataframe["label"] == 0].reset_index(
            drop=True
        )

        if n_coi_classes > 1:
            self.coi_by_class = [
                self.coi_df[self.coi_df.get("coi_class", 0) == i].reset_index(drop=True)
                for i in range(n_coi_classes)
            ]
            print(
                f"{split} set: {[len(c) for c in self.coi_by_class]} per class, "
                f"{len(self.non_coi_df)} non-COI"
            )
        else:
            print(
                f"{split} set: {len(self.coi_df)} COI, {len(self.non_coi_df)} non-COI"
            )

    def __len__(self):
        return len(self.coi_df)

    def load_and_preprocess(self, filepath):
        waveform, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        if waveform.shape[0] < self.segment_samples:
            padding = self.segment_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        if self.augment and waveform.shape[0] > self.segment_samples:
            start = np.random.randint(0, waveform.shape[0] - self.segment_samples)
            waveform = waveform[start : start + self.segment_samples]
        else:
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
            coi_idx = idx % len(self.coi_df)
            coi_file = self.coi_df.iloc[coi_idx]["filename"]
            coi_audio = self.load_and_preprocess(coi_file)

            # Create zero tensors for other classes
            sources = [coi_audio]
        else:
            # Multi-class case: randomly select one COI class
            class_idx = np.random.randint(0, self.n_coi_classes)
            coi_idx = np.random.randint(0, len(self.coi_by_class[class_idx]))
            coi_file = self.coi_by_class[class_idx].iloc[coi_idx]["filename"]
            coi_audio = self.load_and_preprocess(coi_file)

            # Create sources list with zeros for other classes
            sources = [torch.zeros_like(coi_audio) for _ in range(self.n_coi_classes)]
            sources[class_idx] = coi_audio

        # Sample background
        noncoi_idx = np.random.randint(0, len(self.non_coi_df))
        noncoi_file = self.non_coi_df.iloc[noncoi_idx]["filename"]
        background = self.load_and_preprocess(noncoi_file)

        # Mix all COI classes with background
        total_coi = sum(sources)
        snr_db = np.random.uniform(*self.snr_range)
        mixture = self.create_mixture(total_coi, background, snr_db)

        # Normalize
        mixture = self.normalize(mixture)
        sources = [self.normalize(s) for s in sources]
        background = self.normalize(background)

        # Stack: [coi_class_1, ..., coi_class_n, background]
        sources.append(background)
        sources_tensor = torch.stack(sources, dim=0)

        return mixture, sources_tensor


def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad_norm=5.0):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for mixtures, sources in progress_bar:
        mixtures = mixtures.to(device)
        sources = sources.to(device)

        optimizer.zero_grad()
        estimates = model(mixtures.unsqueeze(1))
        loss = criterion(estimates, sources)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        running_loss += loss.item() * mixtures.size(0)
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    for mixtures, sources in progress_bar:
        mixtures = mixtures.to(device)
        sources = sources.to(device)

        estimates = model(mixtures.unsqueeze(1))
        loss = criterion(estimates, sources)

        running_loss += loss.item() * mixtures.size(0)
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)
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

    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": True,
        "persistent_workers": config.training.num_workers > 0,
        "prefetch_factor": 2 if config.training.num_workers > 0 else None,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


def get_model_args(model):
    """Extract model arguments for checkpoint saving."""
    return {
        "in_channels": getattr(model, "in_channels", None),
        "out_channels": getattr(model, "out_channels", None),
        "num_blocks": getattr(model, "num_blocks", None),
    }


def create_model(config: Config):
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
        from base.sudo_rm_rf.sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 import (
            GroupCommSuDORMRFv2,
        )

        base_model = GroupCommSuDORMRFv2(
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

    return model.to(config.training.device)


def train(config: Config):
    """Main training function."""
    # Setup
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Save config
    config.save(checkpoint_dir / "config.yaml")

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)

    print("Creating model...")
    model = create_model(config)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")

    # Setup training
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
    datasets_path = str(project_root / "data" / "metadata")
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

    # 10. Train
    train(config)


if __name__ == "__main__":
    main()

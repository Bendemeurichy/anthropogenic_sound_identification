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
import json
from datetime import datetime

# Pin to single GPU before importing torch (prevents multi-GPU OOM issues)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from contextlib import nullcontext

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base.sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF
from .base.sudo_rm_rf.dnn.losses.sisdr import PairwiseNegSDR
from .seperation_head import wrap_model_for_coi, COILoss
from .multi_class_seperation import wrap_model_for_multiclass
from .config import Config


from label_loading.sampler import get_coi, sample_non_coi
from label_loading.metadata_loader import (
    load_metadata_datasets,
    split_seperation_classification,
)


class AudioDataset(Dataset):
    """pytorch dataset handler for wav files."""

    def __init__(
        # Select epoch item (one per COI file and one per non-COI file)
        item = self.epoch_index[idx]

        if item["type"] == "coi":
            # COI entry: may contain class_idx for multi-class
            filepath = item["filepath"]
            orig_sr = item["orig_sr"]
            num_frames_orig = item["num_frames_orig"]

            seg_frames_orig = max(
                1, int(self.segment_samples * orig_sr / self.sample_rate)
            )
            max_offset = max(0, num_frames_orig - seg_frames_orig)
            frame_offset = int(np.random.randint(0, max_offset + 1)) if max_offset > 0 else 0

            coi_audio = self.load_and_preprocess(
                filepath, frame_offset=frame_offset, num_frames=seg_frames_orig
            )

            if self.n_coi_classes > 1:
                # multi-class: place coi_audio in correct class slot
                class_idx = item.get("class_idx", 0)
                sources = [torch.zeros_like(coi_audio) for _ in range(self.n_coi_classes)]
                sources[class_idx] = coi_audio
            else:
                sources = [coi_audio]

        else:
            # non-COI entry: use this file as the background, COI targets are zeros
            filepath = item["filepath"]
            orig_sr = item["orig_sr"]
            num_frames_orig = item["num_frames_orig"]

            seg_frames_orig = max(
                1, int(self.segment_samples * orig_sr / self.sample_rate)
            )
            max_offset = max(0, num_frames_orig - seg_frames_orig)
            frame_offset = int(np.random.randint(0, max_offset + 1)) if max_offset > 0 else 0

            background = self.load_and_preprocess(
                filepath, frame_offset=frame_offset, num_frames=seg_frames_orig
            )

            if self.n_coi_classes > 1:
                sources = [torch.zeros_like(background) for _ in range(self.n_coi_classes)]
            else:
                sources = [torch.zeros_like(background)]
            # append background later after scaling
                # If info fails, load whole file to determine length
                waveform, orig_sr = torchaudio.load(filepath)
                num_frames_orig = (
                    waveform.shape[1] if waveform.ndim > 1 else waveform.shape[0]
                )

            self.coi_files_info.append(
                {
                    "filepath": filepath,
                    "orig_sr": orig_sr,
                    "num_frames_orig": num_frames_orig,
                }
            )

        # Build non-COI per-file info as well (sampled backgrounds)
        self.noncoi_files_info: list[dict] = []
        for filepath in self.non_coi_files:
            try:
                info = torchaudio.info(filepath)
                orig_sr = info.sample_rate
                num_frames_orig = int(info.num_frames)
            except Exception:
                waveform, orig_sr = torchaudio.load(filepath)
                num_frames_orig = (
                    waveform.shape[1] if waveform.ndim > 1 else waveform.shape[0]
                )
            self.noncoi_files_info.append(
                {"filepath": filepath, "orig_sr": orig_sr, "num_frames_orig": num_frames_orig}
            )

        if n_coi_classes > 1:
            # Store file lists per class instead of DataFrames
            coi_class_col = (
                split_df["coi_class"]
                if "coi_class" in split_df.columns
                else pd.Series(0, index=split_df.index)
            )
            self.coi_by_class = [
                split_df.loc[
                    (split_df["label"] == 1) & (coi_class_col == i),
                    "filename",
                ].tolist()
                for i in range(n_coi_classes)
            ]
            # Also build per-class file info lists for random sampling
            self.coi_by_class_info = []
            for i in range(n_coi_classes):
                files = self.coi_by_class[i]
                info_list = []
                for fp in files:
                    try:
                        info = torchaudio.info(fp)
                        orig_sr = info.sample_rate
                        num_frames_orig = int(info.num_frames)
                    except Exception:
                        waveform, orig_sr = torchaudio.load(fp)
                        num_frames_orig = (
                            waveform.shape[1]
                            if waveform.ndim > 1
                            else waveform.shape[0]
                        )
                    info_list.append(
                        {
                            "filepath": fp,
                            "orig_sr": orig_sr,
                            "num_frames_orig": num_frames_orig,
                        }
                    )
                self.coi_by_class_info.append(info_list)
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

            # Build epoch index: one entry per COI file and one per non-COI file
            # Each epoch will therefore yield exactly len(coi_files)+len(non_coi_files)
            # random mixtures (with randomized offsets at each __getitem__ call).
            self.epoch_index: list[dict] = []
            if self.n_coi_classes > 1:
                for class_idx, info_list in enumerate(self.coi_by_class_info):
                    for info in info_list:
                        entry = {
                            "type": "coi",
                            "filepath": info["filepath"],
                            "orig_sr": info["orig_sr"],
                            "num_frames_orig": info["num_frames_orig"],
                            "class_idx": class_idx,
                        }
                        self.epoch_index.append(entry)
            else:
                for info in self.coi_files_info:
                    entry = {
                        "type": "coi",
                        "filepath": info["filepath"],
                        "orig_sr": info["orig_sr"],
                        "num_frames_orig": info["num_frames_orig"],
                    }
                    self.epoch_index.append(entry)

            for info in self.noncoi_files_info:
                entry = {
                    "type": "noncoi",
                    "filepath": info["filepath"],
                    "orig_sr": info["orig_sr"],
                    "num_frames_orig": info["num_frames_orig"],
                }
                self.epoch_index.append(entry)

    def __len__(self):
        # One item per entry in epoch_index (COI files + non-COI files)
        return max(1, len(self.epoch_index))

    def regenerate_epoch_index(self):
        """Shuffle / rebuild the epoch index so each epoch yields different mixtures.

        This method keeps the same set of files but shuffles their order. It can
        be extended to sample subsets or alter class balancing per epoch.
        """
        import random

        # For now just shuffle in-place to vary order each epoch
        random.shuffle(self.epoch_index)

    def load_and_preprocess(
        self, filepath: str, frame_offset: int = 0, num_frames: int | None = None
    ) -> torch.Tensor:
        """Load only the needed segment from disk to reduce RAM spikes.

        Args:
            filepath: path to audio file
            frame_offset: offset in original-file frames (used with torchaudio.load)
            num_frames: number of frames to load at original sampling rate. If
                None, computed from target `segment_samples` and file sample rate.
        """

        try:
            info = torchaudio.info(filepath)
            orig_sr = info.sample_rate
            total_frames = int(info.num_frames)
        except Exception:
            # Fallback: torchaudio.info can fail on some backends/codecs
            waveform, orig_sr = torchaudio.load(filepath)
            if orig_sr != self.sample_rate:
                key = (orig_sr, self.sample_rate)
                resampler = self._resamplers.get(key)
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(
                        orig_sr, self.sample_rate
                    )
                    self._resamplers[key] = resampler
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.squeeze(0)
            if waveform.shape[0] < self.segment_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.segment_samples - waveform.shape[0])
                )
            return waveform[: self.segment_samples]

        # Decide how many frames to load at original sample rate
        segment_frames_orig = int(self.segment_samples * orig_sr / self.sample_rate)
        segment_frames_orig = max(segment_frames_orig, 1)

        # If caller provided num_frames use it, otherwise default to segment_frames_orig
        if num_frames is None:
            num_frames_to_load = segment_frames_orig
        else:
            num_frames_to_load = int(num_frames)

        # Load only a segment from disk at the requested offset
        waveform, sr = torchaudio.load(
            filepath, frame_offset=int(frame_offset), num_frames=int(num_frames_to_load)
        )

        if sr != self.sample_rate:
            key = (sr, self.sample_rate)
            resampler = self._resamplers.get(key)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                self._resamplers[key] = resampler
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        # Enforce exact segment length at target SR (pad or trim)
        if waveform.shape[0] < self.segment_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_samples - waveform.shape[0])
            )
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
            # Binary case: one sample per COI file per epoch. Randomize
            # offset within the file to produce varied segments.
            info = self.coi_files_info[idx]
            filepath = info["filepath"]
            orig_sr = info["orig_sr"]
            num_frames_orig = info["num_frames_orig"]

            seg_frames_orig = max(
                1, int(self.segment_samples * orig_sr / self.sample_rate)
            )
            max_offset = max(0, num_frames_orig - seg_frames_orig)
            frame_offset = (
                int(np.random.randint(0, max_offset + 1)) if max_offset > 0 else 0
            )

            coi_audio = self.load_and_preprocess(
                filepath, frame_offset=frame_offset, num_frames=seg_frames_orig
            )

            # Create zero tensors for other classes
            sources = [coi_audio]
        else:
            # Multi-class case: randomly select one COI class and file,
            # then sample a random offset inside that file.
            class_idx = np.random.randint(0, self.n_coi_classes)
            info_list = self.coi_by_class_info[class_idx]
            chosen = info_list[np.random.randint(0, len(info_list))]
            filepath = chosen["filepath"]
            orig_sr = chosen["orig_sr"]
            num_frames_orig = chosen["num_frames_orig"]

            seg_frames_orig = max(
                1, int(self.segment_samples * orig_sr / self.sample_rate)
            )
            max_offset = max(0, num_frames_orig - seg_frames_orig)
            frame_offset = (
                int(np.random.randint(0, max_offset + 1)) if max_offset > 0 else 0
            )

            coi_audio = self.load_and_preprocess(
                filepath, frame_offset=frame_offset, num_frames=seg_frames_orig
            )

            # Create sources list with zeros for other classes
            sources = [torch.zeros_like(coi_audio) for _ in range(self.n_coi_classes)]
            sources[class_idx] = coi_audio

        # Sample background: for COI entries, draw a random non-COI file;
        # for non-COI entries, `background` is already loaded above.
        if item["type"] == "coi":
            noncoi_idx = np.random.randint(0, len(self.non_coi_files))
            noncoi_file = self.non_coi_files[noncoi_idx]
            background = self.load_and_preprocess(noncoi_file)

        # Create mixture from sources and background. Do not pre-normalize
        # individual sources here — instead normalize the whole mixture (and
        # targets) using the mixture mean/std so training matches inference
        # where inputs are normalized per-chunk before passing to the model.
        total_coi = torch.stack(sources).sum(dim=0)
        snr_db = np.random.uniform(*self.snr_range)

        # Scale background for target SNR
        coi_power = torch.mean(total_coi**2) + 1e-8
        bg_power = torch.mean(background**2) + 1e-8
        snr_linear = 10 ** (snr_db / 10)
        if item["type"] == "noncoi":
            # For pure-background samples, keep background as-is
            scaled_background = background
        else:
            scaling_factor = torch.sqrt(coi_power / (snr_linear * bg_power))
            scaled_background = background * scaling_factor

        # Mixture is now sum of sources (in original waveform scale)
        mixture = total_coi + scaled_background

        # Update background to the scaled version (what's actually in mixture)
        sources.append(scaled_background)
        sources_tensor = torch.stack(sources, dim=0)

        # Normalize mixture and targets using mixture statistics to match
        # inference-time normalization (per-chunk mean/std).
        mean = mixture.mean()
        std = mixture.std() + 1e-8
        mixture = (mixture - mean) / std
        sources_tensor = (sources_tensor - mean) / std

        return mixture, sources_tensor


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    clip_grad_norm=5.0,
    *,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
):
    model.train()
    running_loss = 0.0
    n_samples = 0

    grad_accum_steps = max(int(grad_accum_steps), 1)
    use_amp = bool(use_amp) and (str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if use_amp
        else nullcontext()
    )

    progress_bar = tqdm(dataloader, desc="Training", leave=False, ascii=True, ncols=100)
    optimizer.zero_grad(set_to_none=True)
    step_idx = 0
    for step_idx, (mixtures, sources) in enumerate(progress_bar, start=1):
        mixtures = mixtures.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)

        with autocast_ctx:
            estimates = model(mixtures.unsqueeze(1))
            loss = criterion(estimates, sources)

            # Scale loss for gradient accumulation
            loss_to_backprop = loss / grad_accum_steps

        if use_amp:
            assert scaler is not None
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        if (step_idx % grad_accum_steps) == 0:
            # Unscale before clipping
            if use_amp:
                assert scaler is not None
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            if use_amp:
                assert scaler is not None
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = mixtures.size(0)
        running_loss += float(loss.detach().cpu().item()) * batch_size
        n_samples += batch_size
        progress_bar.set_postfix(loss=float(loss.detach().cpu().item()))

        # Free memory explicitly
        del mixtures, sources, estimates, loss, loss_to_backprop

    # Flush any remaining accumulated gradients
    if step_idx != 0 and (step_idx % grad_accum_steps) != 0:
        if use_amp:
            assert scaler is not None
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        if use_amp:
            assert scaler is not None
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Clear GPU cache after epoch
    if device != "cpu":
        torch.cuda.empty_cache()

    epoch_loss = running_loss / n_samples
    return epoch_loss


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, *, use_amp: bool = True):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    use_amp = bool(use_amp) and (str(device).startswith("cuda"))
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if use_amp
        else nullcontext()
    )

    progress_bar = tqdm(
        dataloader, desc="Validation", leave=False, ascii=True, ncols=100
    )
    for mixtures, sources in progress_bar:
        mixtures = mixtures.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)

        with autocast_ctx:
            estimates = model(mixtures.unsqueeze(1))
            loss = criterion(estimates, sources)

        batch_size = mixtures.size(0)
        running_loss += float(loss.detach().cpu().item()) * batch_size
        n_samples += batch_size
        progress_bar.set_postfix(loss=float(loss.detach().cpu().item()))

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
        segment_stride=getattr(config.data, "segment_stride", None),
    )
    val_dataset = AudioDataset(
        df,
        split="val",
        sample_rate=config.data.sample_rate,
        segment_length=config.data.segment_length,
        snr_range=tuple(config.data.snr_range),
        n_coi_classes=config.data.n_coi_classes,
        augment=False,
        segment_stride=getattr(config.data, "segment_stride", None),
    )

    # Memory-optimized DataLoader settings
    # CRITICAL: num_workers=0 saves ~10-15GB RAM by avoiding worker process memory duplication
    # Each worker copies the dataset and loads audio into its own memory space
    # With num_workers=0, all loading happens in main process (slower but much less RAM)
    num_workers = config.training.num_workers

    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": num_workers,
        "pin_memory": True,  # weird fix, works
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


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config: Config, timestamp: str = None):
    """Main training function."""
    # Set seed for reproducibility
    seed = getattr(config.training, "seed", 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")

    # Setup - create timestamped subdirectory
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

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

    # Primary loss: use class-aware COI loss for binary case, or an
    # ordered-pairwise diagonal for multi-class (no PIT).
    if config.data.n_coi_classes == 1:
        primary_loss = COILoss(
            class_weight=getattr(config.training, "class_weight", 1.5)
        )
    else:

        class OrderedPairwiseNegSDR(torch.nn.Module):
            def __init__(self, pw_module):
                super().__init__()
                self.pw = pw_module

            def forward(self, est, tgt):
                # PairwiseNegSDR returns (B, n_src, n_src) pairwise losses.
                pw = self.pw(est, tgt)
                # Use ordered diagonal (assumes head produces fixed ordering).
                diag = torch.diagonal(pw, dim1=1, dim2=2)
                return diag.mean()

        primary_loss = OrderedPairwiseNegSDR(pairwise_neg_sisdr)

    # Optional auxiliary L1 on waveform to stabilize early training
    aux_weight = getattr(config.training, "aux_waveform_weight", 0.0)

    if aux_weight and aux_weight > 0.0:

        class CompositeLoss(torch.nn.Module):
            def __init__(self, primary, aux_w):
                super().__init__()
                self.primary = primary
                self.aux_w = aux_w

            def forward(self, est, tgt):
                loss = self.primary(est, tgt)
                aux = torch.nn.functional.l1_loss(est, tgt)
                return loss + self.aux_w * aux

        criterion = CompositeLoss(primary_loss, aux_weight)
    else:
        criterion = primary_loss
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

        # Regenerate / shuffle epoch index so each epoch produces different
        # random mixtures (one per COI file + one per sampled non-COI file).
        try:
            if hasattr(train_loader.dataset, "regenerate_epoch_index"):
                train_loader.dataset.regenerate_epoch_index()
        except Exception:
            pass

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.training.device,
            config.training.clip_grad_norm,
            grad_accum_steps=getattr(config.training, "grad_accum_steps", 1),
            use_amp=getattr(config.training, "use_amp", True),
        )
        history["train_loss"].append(train_loss)

        # Shuffle / regenerate validation mixtures as well for variability
        try:
            if hasattr(val_loader.dataset, "regenerate_epoch_index"):
                val_loader.dataset.regenerate_epoch_index()
        except Exception:
            pass

        val_loss = validate_epoch(
            model,
            val_loader,
            criterion,
            config.training.device,
            use_amp=getattr(config.training, "use_amp", True),
        )
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
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
    train(config, timestamp=timestamp)


if __name__ == "__main__":
    main()

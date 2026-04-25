"""
Training script for CLAPSep model with text-prompt conditioning for COI separation.

This script fine-tunes the pretrained CLAPSep model using text prompts (like the base
model) instead of learned embeddings. It supports LoRA tuning of the audio encoder
for parameter-efficient adaptation to your COI classes while retaining the flexibility
to change target sounds at inference time by modifying the text prompt.

Key differences from train_coi.py:
  - Uses text prompts → CLAP text encoder (NOT learned embeddings)
  - Retains inference flexibility (can change prompts without retraining)
  - Multiple prompt variations per class for robustness
  - Simpler architecture (uses base CLAPSep decoder)

Expected dataframe structure:
    - 'filename': path to wav file
    - 'split': train/val/test
    - 'label': 1 for COI, 0 for background (non-COI)
    - 'coi_class': integer index (0 ... n_coi_classes-1) for multi-class COI
"""

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchlibrosa import ISTFT, STFT
from torchlibrosa.stft import magphase

# Add parent directories to path for imports
_src_root = Path(__file__).parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

# Import shared COI training utilities
from common.coi_training import (
    COIWeightedLoss,
    create_coi_dataloader,
    prepare_batch,
    sisnr,
)

# Import CLAPSep components and LoRA
try:
    import laion_clap
    from models.clapsep.base.model.CLAPSep_decoder import HTSAT_Decoder
    HAS_CLAP = True
except ImportError:
    HAS_CLAP = False
    print("Warning: laion_clap not available. Install with: pip install laion-clap")

try:
    import loralib as lora
    HAS_LORA = True
except ImportError:
    HAS_LORA = False
    print("Warning: loralib not available. Install with: pip install loralib")


# =============================================================================
# LoRA utilities (from base CLAPSep implementation)
# =============================================================================


def set_module(model, submodule_key, module):
    """Set a submodule in a model by its key path."""
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def apply_lora_to_model(model, rank: int = 8):
    """
    Replace linear layers in WindowAttention modules with LoRA layers.
    
    This follows the original CLAPSep implementation which applies LoRA
    specifically to attention layers in the HTSAT encoder.
    
    Args:
        model: The audio encoder model
        rank: LoRA rank (lower = fewer parameters, typically 4-16)
        
    Returns:
        Model with LoRA layers applied
    """
    if not HAS_LORA:
        raise RuntimeError(
            "loralib is required for LoRA fine-tuning. "
            "Install with: pip install loralib"
        )
    
    for module_name, module in model.named_modules():
        # Apply LoRA only to WindowAttention layers (as in original CLAPSep)
        if 'WindowAttention' in str(type(module)):
            for layer_name, layer in module.named_modules():
                if isinstance(layer, torch.nn.Linear):
                    # Create LoRA layer with same dimensions
                    lora_layer = lora.Linear(
                        layer.in_features,
                        layer.out_features,
                        r=rank,
                        bias=hasattr(layer, 'bias'),
                        merge_weights=False
                    )
                    # Copy pretrained weights
                    lora_layer.weight = layer.weight
                    if hasattr(layer, 'bias'):
                        lora_layer.bias = layer.bias
                    # Replace the layer
                    full_path = f"{module_name}.{layer_name}" if module_name else layer_name
                    set_module(model, full_path, lora_layer)
    
    # Mark only LoRA parameters as trainable
    lora.mark_only_lora_as_trainable(model, bias='lora_only')
    
    return model


# =============================================================================
# Text-Prompt Dataset
# =============================================================================


class TextPromptCOIDataset(torch.utils.data.Dataset):
    """
    COI dataset that provides text prompts for each sample.
    
    Maps COI labels to text prompt variations for robust training.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str,
        sample_rate: int,
        resample_rate: int,
        segment_length: float,
        coi_text_prompts: List[List[str]],
        background_text_prompts: List[str],
        target_classes: Optional[List[List[str]]] = None,
    ):
        """
        Args:
            dataframe: DataFrame with 'filename', 'split', 'label' columns
            split: 'train', 'val', or 'test'
            sample_rate: Target sample rate for audio
            resample_rate: CLAP input sample rate (48kHz)
            segment_length: Audio segment length in seconds
            coi_text_prompts: List of prompt variations per COI class
                             e.g., [["airplane engine", "aircraft noise"], ["bird chirping"]]
            background_text_prompts: List of background prompt variations
            target_classes: List of semantic label groups per COI class (for adding coi_class)
        """
        self.split = split
        self.sample_rate = sample_rate
        self.resample_rate = resample_rate
        self.segment_length = segment_length
        self.coi_text_prompts = coi_text_prompts
        self.background_text_prompts = background_text_prompts
        self.n_coi_classes = len(coi_text_prompts)
        
        # Add coi_class column if needed
        if 'coi_class' not in dataframe.columns and target_classes is not None:
            dataframe = self._add_coi_class_column(dataframe, target_classes)
        
        # Filter by split
        self.df = dataframe[dataframe['split'] == split].reset_index(drop=True)
        
        print(f"[{split}] Loaded {len(self.df)} samples")
        if 'coi_class' in self.df.columns:
            for cls_idx in range(self.n_coi_classes):
                n = ((self.df['label'] == 1) & (self.df['coi_class'] == cls_idx)).sum()
                print(f"  COI class {cls_idx}: {n} samples")
    
    def _add_coi_class_column(self, df: pd.DataFrame, target_classes: List[List[str]]) -> pd.DataFrame:
        """Add coi_class column by mapping semantic labels to class indices."""
        def _coi_class(label_val):
            # Handle string or list labels
            labels = label_val if isinstance(label_val, list) else [label_val]
            for cls_idx, group in enumerate(target_classes):
                if any(lbl in group for lbl in labels):
                    return cls_idx
            return -1  # non-COI
        
        df = df.copy()
        if 'orig_label' in df.columns:
            df['coi_class'] = df['orig_label'].apply(_coi_class)
        else:
            df['coi_class'] = df.get('label', pd.Series([-1] * len(df))).apply(
                lambda x: -1 if x == 0 else 0
            )
        df['coi_class'] = df['coi_class'].astype('int16')
        return df
    
    def __len__(self):
        # Return 0 for test split (not used in training/validation)
        return 0 if self.split == 'test' else len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
                - 'filename': audio file path
                - 'label': 1 for COI, 0 for background
                - 'coi_class': integer class index (for COI samples)
                - 'pos_text': text prompt for target (randomly selected variation)
                - 'neg_text': text prompt for background (randomly selected variation)
        """
        row = self.df.iloc[idx]
        
        is_coi = row['label'] == 1
        
        # Select random text prompts
        if is_coi:
            coi_class = int(row.get('coi_class', 0))
            coi_class = max(0, min(coi_class, self.n_coi_classes - 1))  # clamp
            pos_text = random.choice(self.coi_text_prompts[coi_class])
            neg_text = random.choice(self.background_text_prompts)
        else:
            # For background samples, still need prompts for the model
            # Use a random COI prompt as "positive" and background as "negative"
            # (the model will learn to extract background when it's the target)
            pos_text = random.choice(self.background_text_prompts)
            neg_text = random.choice(self.coi_text_prompts[0])  # any COI class
        
        return {
            'filename': row['filename'],
            'label': row['label'],
            'coi_class': row.get('coi_class', -1),
            'pos_text': pos_text,
            'neg_text': neg_text,
        }


def collate_with_text_prompts(batch):
    """Collate function that preserves text prompts."""
    return {
        'filenames': [item['filename'] for item in batch],
        'labels': torch.tensor([item['label'] for item in batch], dtype=torch.long),
        'coi_classes': torch.tensor([item['coi_class'] for item in batch], dtype=torch.long),
        'pos_texts': [item['pos_text'] for item in batch],
        'neg_texts': [item['neg_text'] for item in batch],
    }


# =============================================================================
# Text-Prompt CLAPSep Model
# =============================================================================


class TextPromptCLAPSep(pl.LightningModule):
    """
    PyTorch Lightning module for COI separation using CLAPSep with text prompts.
    
    This model uses text prompts (not learned embeddings) for conditioning,
    retaining the flexibility to change target sounds at inference time.
    
    Supports three encoder modes:
    1. freeze_encoder=True, use_lora=False: Encoder completely frozen (fastest, least memory)
    2. freeze_encoder=False, use_lora=False: Full encoder fine-tuning (most parameters)
    3. freeze_encoder=False, use_lora=True: LoRA fine-tuning (parameter-efficient, recommended)
    """
    
    def __init__(
        self,
        clap_model,
        decoder_model: nn.Module,
        coi_text_prompts: List[List[str]],
        background_text_prompts: List[str],
        lr: float = 1e-4,
        nfft: int = 1024,
        sample_rate: int = 32000,
        resample_rate: int = 48000,
        class_weight: float = 1.5,
        freeze_encoder: bool = True,
        use_lora: bool = False,
        lora_rank: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["clap_model", "decoder_model"])
        
        self.lr = lr
        self.class_weight = class_weight
        self.sample_rate = sample_rate
        self.resample_rate = resample_rate
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.coi_text_prompts = coi_text_prompts
        self.background_text_prompts = background_text_prompts
        
        # CLAP model (always frozen, only used for text embeddings)
        self.clap_model = clap_model
        for p in self.clap_model.parameters():
            p.requires_grad = False
        
        # Copy audio branch for fine-tuning
        import copy
        self.audio_branch = copy.deepcopy(self.clap_model.model.audio_branch)
        
        if freeze_encoder:
            # Mode 1: Completely frozen encoder
            for p in self.audio_branch.parameters():
                p.requires_grad = False
            print("Audio encoder: FROZEN (decoder-only training)")
        elif use_lora:
            # Mode 2: LoRA fine-tuning (parameter-efficient)
            if not HAS_LORA:
                raise RuntimeError(
                    "loralib is required for LoRA fine-tuning. "
                    "Install with: pip install loralib\n"
                    "Or set use_lora=False to use full fine-tuning or frozen encoder."
                )
            print(f"Applying LoRA (rank={lora_rank}) to audio encoder...")
            self.audio_branch = apply_lora_to_model(self.audio_branch, rank=lora_rank)
            # Count trainable params
            lora_params = sum(p.numel() for p in self.audio_branch.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.audio_branch.parameters())
            print(f"  LoRA trainable: {lora_params/1e6:.2f}M / {total_params/1e6:.2f}M total "
                  f"({100*lora_params/total_params:.2f}%)")
        else:
            # Mode 3: Full fine-tuning (all parameters trainable)
            print("Audio encoder: FULL FINE-TUNING (all parameters trainable)")
        
        # Decoder (always trainable)
        self.decoder = decoder_model
        
        # STFT for time-frequency processing
        self.stft = STFT(
            n_fft=nfft,
            hop_length=320,
            win_length=nfft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.istft = ISTFT(
            n_fft=nfft,
            hop_length=320,
            win_length=nfft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        
        # Resampler for CLAP input (expects 48kHz)
        import torchaudio
        self.resampler = torchaudio.transforms.Resample(sample_rate, resample_rate)
        
        # Install forward hooks for skip connections
        self.features = self._install_forward_hooks()
        
        # Loss function
        self.criterion = COIWeightedLoss(class_weight=class_weight)
        
        # Track best validation metric
        self.best_val_sisnr = -float('inf')
    
    def _install_forward_hooks(self):
        """Install forward hooks to capture intermediate features."""
        features = []
        
        def get_features_list(_, __, output):
            features.append(output)
        
        def get_features_list_basic_layer(_, __, output):
            features.append(output[0])
        
        def spectrogram_padding(_, __, out):
            return torch.nn.functional.pad(out, (0, 0, 0, 1024 - out.size(2)))
        
        self.audio_branch.spectrogram_extractor.register_forward_hook(
            spectrogram_padding
        )
        self.audio_branch.patch_embed.register_forward_hook(get_features_list)
        for module in self.audio_branch.layers:
            module.register_forward_hook(get_features_list_basic_layer)
        
        return features
    
    def wav_reconstruct(self, mask, mag_x, cos_x, sin_x, length):
        """Reconstruct waveform from mask and STFT components."""
        if isinstance(mask, (list, tuple)):
            # Phase-aware mask (not used in current config)
            mag_y = torch.nn.functional.relu_(mag_x * mask[0])
            _, mask_cos, mask_sin = magphase(mask[1], mask[2])
            cos_y = cos_x * mask_cos - sin_x * mask_sin
            sin_y = sin_x * mask_cos + cos_x * mask_sin
        else:
            mag_y = torch.nn.functional.relu_(mag_x * mask)
            cos_y = cos_x
            sin_y = sin_x
        
        pred = self.istft(mag_y * cos_y, mag_y * sin_y, length=length)
        return pred
    
    def forward(self, mixture, pos_texts, neg_texts):
        """
        Separate mixture using text prompts.
        
        Args:
            mixture: (B, T) input mixture waveform at self.sample_rate
            pos_texts: List of positive text prompts (length B)
            neg_texts: List of negative text prompts (length B)
        
        Returns:
            separated: (B, 2, T) with [COI, background]
        """
        B = mixture.shape[0]
        length = mixture.shape[-1]
        
        # STFT
        real, imag = self.stft(mixture)
        mag, cos, sin = magphase(real, imag)
        
        # Get text embeddings
        with torch.no_grad():
            embed_pos = self.clap_model.get_text_embedding(pos_texts, use_tensor=True)
            embed_neg = self.clap_model.get_text_embedding(neg_texts, use_tensor=True)
        
        # Clear features from previous forward pass
        del self.features[:]
        self.features.append(mag)
        
        # Get features from audio encoder (LoRA layers train if enabled)
        mixture_resampled = self.resampler(mixture)
        self.audio_branch({"waveform": mixture_resampled})
        
        # Generate separation masks using text embeddings
        embed = torch.nn.functional.normalize(
            torch.cat([embed_pos, embed_neg], dim=-1), dim=-1
        )
        mask = self.decoder(
            hidden_state=self.features[-1],
            skip_features=self.features[:-1],
            embed=embed
        )
        
        # Reconstruct separated source
        pred = self.wav_reconstruct(mask, mag, cos, sin, length=length)
        
        # Stack as [COI, background]
        background = mixture - pred
        separated = torch.stack([pred, background], dim=1)
        
        return separated
    
    def training_step(self, batch, batch_idx):
        """Training step using text prompts."""
        # Prepare batch (loads audio, creates mixtures)
        mixture, sources = prepare_batch(
            batch,
            sample_rate=self.sample_rate,
            segment_samples=int(self.sample_rate * 5.0),  # 5 second segments
            device=self.device,
        )
        
        # Get text prompts from batch
        pos_texts = batch['pos_texts']
        neg_texts = batch['neg_texts']
        
        # Forward pass
        pred_sources = self.forward(mixture, pos_texts, neg_texts)
        
        # Compute loss
        loss = self.criterion(pred_sources, sources)
        
        # Log metrics
        with torch.no_grad():
            coi_sisnr = sisnr(pred_sources[:, 0, :], sources[:, 0, :]).mean()
            bg_sisnr = sisnr(pred_sources[:, 1, :], sources[:, 1, :]).mean()
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/coi_sisnr', coi_sisnr, on_step=False, on_epoch=True)
        self.log('train/bg_sisnr', bg_sisnr, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step using text prompts."""
        # Prepare batch
        mixture, sources = prepare_batch(
            batch,
            sample_rate=self.sample_rate,
            segment_samples=int(self.sample_rate * 5.0),
            device=self.device,
        )
        
        # Get text prompts
        pos_texts = batch['pos_texts']
        neg_texts = batch['neg_texts']
        
        # Forward pass
        with torch.no_grad():
            pred_sources = self.forward(mixture, pos_texts, neg_texts)
        
        # Compute loss
        loss = self.criterion(pred_sources, sources)
        
        # Compute SI-SNR improvement
        coi_sisnr = sisnr(pred_sources[:, 0, :], sources[:, 0, :]).mean()
        bg_sisnr = sisnr(pred_sources[:, 1, :], sources[:, 1, :]).mean()
        
        # Mixture SI-SNR (baseline)
        mix_sisnr = sisnr(mixture, sources[:, 0, :]).mean()
        sisnr_improvement = coi_sisnr - mix_sisnr
        
        # Log metrics
        self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/coi_sisnr', coi_sisnr, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/bg_sisnr', bg_sisnr, on_epoch=True, sync_dist=True)
        self.log('val/sisnr_improvement', sisnr_improvement, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/coi_sisnr",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# =============================================================================
# Training
# =============================================================================


def train(config_path: Path, args):
    """Main training function."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line args
    if args.df_path:
        config['data']['df_path'] = args.df_path
    if args.clap_checkpoint:
        config['model']['clap_checkpoint'] = args.clap_checkpoint
    if args.freeze_encoder is not None:
        config['model']['freeze_encoder'] = args.freeze_encoder
    if args.use_lora is not None:
        config['model']['use_lora'] = args.use_lora
    if args.lora_rank:
        config['model']['lora_rank'] = args.lora_rank
    if args.device:
        config['training']['device'] = args.device
    
    # Set random seed
    pl.seed_everything(config['training'].get('seed', 42))
    
    # Load dataset
    print(f"\nLoading dataset from: {config['data']['df_path']}")
    df = pd.read_csv(config['data']['df_path'])
    
    # Extract text prompt configuration
    coi_text_prompts = config['model'].get('coi_text_prompts', [["airplane engine", "aircraft noise"]])
    background_text_prompts = config['model'].get('background_text_prompts', ["ambient noise", "background sounds"])
    target_classes = config['data'].get('target_classes', None)
    
    print(f"\nText prompts configuration:")
    print(f"  COI classes: {len(coi_text_prompts)}")
    for i, prompts in enumerate(coi_text_prompts):
        print(f"    [{i}] {prompts}")
    print(f"  Background: {background_text_prompts}")
    
    # Create datasets
    sample_rate = config['data']['sample_rate']
    resample_rate = config['model'].get('resample_rate', 48000)
    segment_length = config['data']['segment_length']
    
    train_dataset = TextPromptCOIDataset(
        df, 'train', sample_rate, resample_rate, segment_length,
        coi_text_prompts, background_text_prompts, target_classes
    )
    
    val_dataset = TextPromptCOIDataset(
        df, 'val', sample_rate, resample_rate, segment_length,
        coi_text_prompts, background_text_prompts, target_classes
    )
    
    # Create data loaders using shared COI utilities
    train_loader = create_coi_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_with_text_prompts,
    )
    
    val_loader = create_coi_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_with_text_prompts,
    )
    
    # Load CLAP model
    print(f"\nLoading CLAP model from: {config['model']['clap_checkpoint']}")
    clap_model = laion_clap.CLAP_Module(
        enable_fusion=False,
        amodel='HTSAT-base',
        device='cpu'
    )
    clap_model.load_ckpt(config['model']['clap_checkpoint'])
    
    # Create decoder
    decoder_config = {
        'lan_embed_dim': config['model']['lan_embed_dim'],
        'depths': config['model']['depths'],
        'embed_dim': config['model']['embed_dim'],
        'encoder_embed_dim': config['model']['encoder_embed_dim'],
        'phase': config['model']['phase'],
        'spec_factor': config['model']['spec_factor'],
        'd_attn': config['model']['d_attn'],
        'n_masker_layer': config['model']['n_masker_layer'],
        'conv': config['model']['conv'],
    }
    decoder = HTSAT_Decoder(**decoder_config)
    
    # Create Lightning module
    model = TextPromptCLAPSep(
        clap_model=clap_model,
        decoder_model=decoder,
        coi_text_prompts=coi_text_prompts,
        background_text_prompts=background_text_prompts,
        lr=config['training']['lr'],
        nfft=config['model']['nfft'],
        sample_rate=sample_rate,
        resample_rate=resample_rate,
        class_weight=config['training']['class_weight'],
        freeze_encoder=config['model']['freeze_encoder'],
        use_lora=config['model']['use_lora'],
        lora_rank=config['model']['lora_rank'],
    )
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / f"text_prompt_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = checkpoint_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\nSaved config to: {config_save_path}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_model',
        monitor='val/coi_sisnr',
        mode='max',
        save_top_k=1,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/coi_sisnr',
        patience=config['training']['patience'],
        mode='max',
        verbose=True,
    )
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=checkpoint_dir,
        accelerator='gpu' if config['training']['device'].startswith('cuda') else 'cpu',
        devices=1,
        max_epochs=config['training']['num_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=config['training'].get('gradient_clip_val', 5.0),
        precision=config['training'].get('precision', 'bf16-mixed'),
        log_every_n_steps=10,
        val_check_interval=1.0,
    )
    
    # Train
    print(f"\nStarting training...")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Max epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['lr']}")
    print(f"  Freeze encoder: {config['model']['freeze_encoder']}")
    print(f"  Use LoRA: {config['model']['use_lora']}")
    if config['model']['use_lora']:
        print(f"  LoRA rank: {config['model']['lora_rank']}")
    
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\n✓ Training complete!")
    print(f"  Best model: {checkpoint_callback.best_model_path}")
    print(f"  Best val SI-SNR: {checkpoint_callback.best_model_score:.2f} dB")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Train CLAPSep with text-prompt conditioning for COI separation'
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default='training_config.yaml',
        help='Path to config YAML file'
    )
    
    # Data
    parser.add_argument('--df-path', type=str, help='Path to dataset CSV')
    
    # Model
    parser.add_argument('--clap-checkpoint', type=str, help='Path to CLAP checkpoint')
    parser.add_argument('--freeze-encoder', action='store_true', help='Freeze audio encoder')
    parser.add_argument('--no-freeze-encoder', dest='freeze_encoder', action='store_false')
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA fine-tuning')
    parser.add_argument('--no-lora', dest='use_lora', action='store_false')
    parser.add_argument('--lora-rank', type=int, help='LoRA rank (4-16 typical)')
    
    # Training
    parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    
    parser.set_defaults(freeze_encoder=None, use_lora=None)
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    train(config_path, args)


if __name__ == '__main__':
    main()

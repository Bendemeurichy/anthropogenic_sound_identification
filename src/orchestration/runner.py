"""
Runner module that bridges orchestration layer to TUSS training and inference.

This module provides the actual execution logic for training and inference,
importing from the existing TUSS implementation.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Ensure src is in path for imports
_SCRIPT_DIR = Path(__file__).parent.resolve()
_SRC_DIR = _SCRIPT_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def run_training(
    config_path: Union[str, Path],
    device: Optional[str] = None,
) -> Path:
    """Run TUSS training with the given config.
    
    Args:
        config_path: Path to training_config.yaml
        device: Optional device override ("cuda", "cuda:0", "cpu")
        
    Returns:
        Path to checkpoint directory containing trained model
    """
    import torch
    from datetime import datetime
    
    # Import TUSS training components
    # We need to temporarily change the config path used by train.py
    config_path = Path(config_path).resolve()
    
    # Import config and train function
    from models.tuss.train import Config, train, resolve_device
    
    print(f"Loading training config from: {config_path}")
    config = Config.from_yaml(str(config_path))
    
    # Apply device override
    if device is not None:
        config.training.device = resolve_device(device)
    else:
        config.training.device = resolve_device(config.training.device)
    
    # Claim the target GPU
    if config.training.device.startswith("cuda:"):
        torch.cuda.set_device(int(config.training.device.split(":")[1]))
    
    print(f"Device:      {config.training.device}")
    print(f"COI prompts: {config.model.coi_prompts}")
    print(f"BG prompt:   {config.model.bg_prompt}")
    
    # Generate timestamp for checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run training
    # The train function will create checkpoints in config.training.checkpoint_dir/timestamp
    train(config, timestamp=timestamp)
    
    # Return the checkpoint directory
    checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
    return checkpoint_dir


def create_pipeline(
    checkpoint_path: Union[str, Path],
    coi_prompts: Optional[List[str]] = None,
    bg_prompt: str = "background",
    device: Optional[str] = None,
    enable_mask_recycling: bool = True,
    cache_size: int = 5,
    similarity_threshold: float = 0.85,
):
    """Create a SeparationPipeline from a trained checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        coi_prompts: List of COI prompt names (uses config defaults if None)
        bg_prompt: Background prompt name
        device: Device to run on (auto-detected if None)
        enable_mask_recycling: Enable mask recycling optimization
        cache_size: Number of segments to cache
        similarity_threshold: Similarity threshold for cache reuse
        
    Returns:
        SeparationPipeline instance ready for inference
    """
    import torch
    from models.tuss.inference import TUSSInference
    from pipeline.separation_pipeline import SeparationPipeline
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load TUSS inference
    tuss = TUSSInference.from_checkpoint(
        str(checkpoint_path),
        device=device,
        coi_prompts=coi_prompts,
        bg_prompt=bg_prompt,
    )
    
    # Wrap with SeparationPipeline
    pipeline = SeparationPipeline(
        tuss_inference=tuss,
        enable_mask_recycling=enable_mask_recycling,
        cache_size=cache_size,
        similarity_threshold=similarity_threshold,
    )
    
    return pipeline


def run_inference(
    checkpoint_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    coi_prompts: Optional[List[str]] = None,
    bg_prompt: str = "background",
    device: Optional[str] = None,
    enable_mask_recycling: bool = True,
    cache_size: int = 5,
    similarity_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Run inference on an audio file.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        audio_path: Path to audio file to process
        output_dir: Optional directory to save separated audio
        coi_prompts: List of COI prompt names (uses config defaults if None)
        bg_prompt: Background prompt name
        device: Device to run on (auto-detected if None)
        enable_mask_recycling: Enable mask recycling optimization
        cache_size: Number of segments to cache
        similarity_threshold: Similarity threshold for cache reuse
        
    Returns:
        Dictionary mapping class names to separated audio tensors
    """
    import torch
    import torchaudio
    
    audio_path = Path(audio_path)
    
    # Create pipeline
    pipeline = create_pipeline(
        checkpoint_path=checkpoint_path,
        coi_prompts=coi_prompts,
        bg_prompt=bg_prompt,
        device=device,
        enable_mask_recycling=enable_mask_recycling,
        cache_size=cache_size,
        similarity_threshold=similarity_threshold,
    )
    
    # Load audio
    waveform, sample_rate = torchaudio.load(str(audio_path))
    
    # Resample if needed
    if sample_rate != pipeline.tuss.sample_rate:
        resampler = torchaudio.transforms.Resample(
            sample_rate, pipeline.tuss.sample_rate
        )
        waveform = resampler(waveform)
    
    # Run separation
    sources = pipeline.separate_waveform(waveform, return_dict=True)
    
    # Save outputs if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stem = audio_path.stem
        for name, audio in sources.items():
            output_path = output_dir / f"{stem}_{name}.wav"
            # Ensure 2D tensor for torchaudio
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            torchaudio.save(str(output_path), audio.cpu(), pipeline.tuss.sample_rate)
            print(f"Saved: {output_path}")
    
    # Print stats if mask recycling was used
    stats = pipeline.get_stats()
    if stats is not None:
        print(f"\nMask recycling stats:")
        print(f"  Cache hits:  {stats['hits']}")
        print(f"  Cache misses: {stats['misses']}")
        print(f"  Hit rate:    {stats['hit_rate']:.1%}")
    
    return sources


def batch_inference(
    checkpoint_path: Union[str, Path],
    audio_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    coi_prompts: Optional[List[str]] = None,
    bg_prompt: str = "background",
    device: Optional[str] = None,
    enable_mask_recycling: bool = True,
    cache_size: int = 5,
    similarity_threshold: float = 0.85,
    clear_cache_between_files: bool = False,
) -> List[Dict[str, Any]]:
    """Run inference on multiple audio files.
    
    Uses a single pipeline instance for efficiency, optionally keeping
    the mask recycling cache across files for additional speedup.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        audio_paths: List of audio file paths to process
        output_dir: Directory to save separated audio
        coi_prompts: List of COI prompt names
        bg_prompt: Background prompt name
        device: Device to run on
        enable_mask_recycling: Enable mask recycling optimization
        cache_size: Number of segments to cache
        similarity_threshold: Similarity threshold for cache reuse
        clear_cache_between_files: If True, clear cache after each file
        
    Returns:
        List of dictionaries mapping class names to separated audio tensors
    """
    import torch
    import torchaudio
    from tqdm import tqdm
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create single pipeline instance
    pipeline = create_pipeline(
        checkpoint_path=checkpoint_path,
        coi_prompts=coi_prompts,
        bg_prompt=bg_prompt,
        device=device,
        enable_mask_recycling=enable_mask_recycling,
        cache_size=cache_size,
        similarity_threshold=similarity_threshold,
    )
    
    results = []
    
    for audio_path in tqdm(audio_paths, desc="Processing audio files"):
        audio_path = Path(audio_path)
        
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Resample if needed
        if sample_rate != pipeline.tuss.sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, pipeline.tuss.sample_rate
            )
            waveform = resampler(waveform)
        
        # Run separation
        sources = pipeline.separate_waveform(waveform, return_dict=True)
        results.append(sources)
        
        # Save outputs
        stem = audio_path.stem
        for name, audio in sources.items():
            output_path = output_dir / f"{stem}_{name}.wav"
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            torchaudio.save(str(output_path), audio.cpu(), pipeline.tuss.sample_rate)
        
        if clear_cache_between_files:
            pipeline.clear_cache()
    
    # Print final stats
    stats = pipeline.get_stats()
    if stats is not None:
        print(f"\nBatch processing complete:")
        print(f"  Files processed: {len(audio_paths)}")
        print(f"  Total cache hits:  {stats['hits']}")
        print(f"  Total cache misses: {stats['misses']}")
        print(f"  Overall hit rate:  {stats['hit_rate']:.1%}")
    
    return results

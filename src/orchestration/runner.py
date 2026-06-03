"""
Runner module that bridges orchestration layer to TUSS training and inference.

This module provides the actual execution logic for training and inference,
importing from the existing TUSS implementation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union


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

    config_path = Path(config_path).resolve()

    from models.tuss.train import Config, train, resolve_device

    print(f"Loading training config from: {config_path}")
    config = Config.from_yaml(str(config_path))

    if device is not None:
        config.training.device = resolve_device(device)
    else:
        config.training.device = resolve_device(config.training.device)

    if config.training.device.startswith("cuda:"):
        torch.cuda.set_device(int(config.training.device.split(":")[1]))

    print(f"Device:      {config.training.device}")
    print(f"COI prompts: {config.model.coi_prompts}")
    print(f"BG prompt:   {config.model.bg_prompt}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train(config, timestamp=timestamp)

    checkpoint_dir = Path(config.training.checkpoint_dir) / timestamp
    return checkpoint_dir


def create_pipeline(
    checkpoint_path: Union[str, Path],
    coi_prompt: Optional[Union[str, List[str]]] = None,
    bg_prompt: str = "background",
    device: Optional[str] = None,
    enable_mask_recycling: bool = True,
    cache_size: int = 5,
    similarity_threshold: float = 0.85,
    target_coi: Optional[str] = None,
):
    """Create a SeparationPipeline from a trained checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        coi_prompt: COI prompt name(s). Uses checkpoint defaults if None.
        bg_prompt: Background prompt name (uses checkpoint default if not set)
        device: Device to run on (auto-detected if None)
        enable_mask_recycling: Enable mask recycling optimization
        cache_size: Number of segments to cache
        similarity_threshold: Similarity threshold for cache reuse
        target_coi: Specific COI name to target at inference time (fuzzy match)

    Returns:
        SeparationPipeline instance ready for inference
    """
    import torch
    from models.tuss.inference import TUSSInference
    from pipeline.separation_pipeline import SeparationPipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(checkpoint_path)

    tuss = TUSSInference.from_checkpoint(
        str(checkpoint_path),
        device=device,
        coi_prompt=coi_prompt,
        bg_prompt=bg_prompt,
    )

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
    coi_prompt: Optional[Union[str, List[str]]] = None,
    bg_prompt: str = "background",
    device: Optional[str] = None,
    enable_mask_recycling: bool = True,
    cache_size: int = 5,
    similarity_threshold: float = 0.85,
    target_coi: Optional[str] = None,
) -> Dict[str, Any]:
    """Run inference on an audio file.

    Args:
        checkpoint_path: Path to checkpoint directory
        audio_path: Path to audio file to process
        output_dir: Optional directory to save separated audio files
        coi_prompt: COI prompt name(s). Uses checkpoint defaults if None.
        bg_prompt: Background prompt name
        device: Device to run on (auto-detected if None)
        enable_mask_recycling: Enable mask recycling optimization
        cache_size: Number of segments to cache
        similarity_threshold: Similarity threshold for cache reuse
        target_coi: Specific COI name to target (fuzzy match)

    Returns:
        Dictionary mapping prompt names to separated audio tensors
    """
    import torchaudio

    audio_path = Path(audio_path)

    pipeline = create_pipeline(
        checkpoint_path=checkpoint_path,
        coi_prompt=coi_prompt,
        bg_prompt=bg_prompt,
        device=device,
        enable_mask_recycling=enable_mask_recycling,
        cache_size=cache_size,
        similarity_threshold=similarity_threshold,
        target_coi=target_coi,
    )

    waveform, sample_rate = torchaudio.load(str(audio_path))

    if sample_rate != pipeline.tuss.sample_rate:
        resampler = torchaudio.transforms.Resample(
            sample_rate, pipeline.tuss.sample_rate
        )
        waveform = resampler(waveform)

    sources = pipeline.separate_waveform(waveform, return_dict=True)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = audio_path.stem
        for name, audio in sources.items():
            output_path = output_dir / f"{stem}_{name}.wav"
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            torchaudio.save(str(output_path), audio.cpu(), pipeline.tuss.sample_rate)
            print(f"Saved: {output_path}")

    stats = pipeline.get_stats()
    if stats is not None:
        print("\nMask recycling stats:")
        print(f"  Cache hits:  {stats['hits']}")
        print(f"  Cache misses: {stats['misses']}")
        print(f"  Hit rate:    {stats['hit_rate']:.1%}")

    return sources

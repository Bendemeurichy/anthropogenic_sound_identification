"""
Example script demonstrating mask-recycling with TUSS separation.

This script shows how to use the SeparationPipeline with mask recycling
to efficiently process audio with repeated or similar segments.
"""

import sys
from pathlib import Path

import torch
import torchaudio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tuss.inference import TUSSInference
from src.pipeline.separation_pipeline import SeparationPipeline


def create_test_audio(duration_sec: float = 60.0, sample_rate: int = 48000) -> torch.Tensor:
    """Create test audio with repeated segments for demonstrating mask recycling.
    
    Creates audio with:
    - 4 seconds of noise
    - 4 seconds of sine wave (repeated 3 times)
    - 4 seconds of different noise
    
    This pattern should show high cache hit rates for the repeated sine segments.
    """
    samples_per_segment = int(4.0 * sample_rate)
    total_samples = int(duration_sec * sample_rate)
    
    audio = torch.zeros(total_samples)
    
    # Fill with pattern
    pos = 0
    pattern_idx = 0
    patterns = []
    
    # Pattern 0: Random noise
    patterns.append(torch.randn(samples_per_segment) * 0.1)
    
    # Pattern 1: 440 Hz sine wave (repeated segment)
    t = torch.arange(samples_per_segment) / sample_rate
    patterns.append(torch.sin(2 * torch.pi * 440 * t) * 0.3)
    
    # Pattern 2: Different noise
    patterns.append(torch.randn(samples_per_segment) * 0.15)
    
    # Pattern 3: 880 Hz sine wave
    patterns.append(torch.sin(2 * torch.pi * 880 * t) * 0.2)
    
    # Fill audio with pattern: [0, 1, 1, 1, 2, 3, 1, 1, 2, ...]
    sequence = [0, 1, 1, 1, 2, 3, 1, 1, 2]
    
    for i in range(0, total_samples, samples_per_segment):
        pattern_num = sequence[pattern_idx % len(sequence)]
        pattern = patterns[pattern_num]
        end = min(i + samples_per_segment, total_samples)
        length = end - i
        audio[i:end] = pattern[:length]
        pattern_idx += 1
    
    return audio


def demo_basic_usage():
    """Demonstrate basic mask-recycling usage."""
    print("=" * 70)
    print("DEMO 1: Basic Mask-Recycling Usage")
    print("=" * 70)
    print()
    
    # Check for checkpoint
    checkpoint_path = Path("checkpoints/tuss/")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please provide a valid TUSS checkpoint path.")
        return
    
    print("Loading TUSS model...")
    tuss = TUSSInference.from_checkpoint(
        checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        coi_prompts=["airplane"],  # Single COI
        bg_prompt="background"
    )
    print(f"  COI prompts: {tuss.coi_prompts}")
    print(f"  Num sources: {tuss.num_sources}")
    print()
    
    print("Creating SeparationPipeline with mask recycling...")
    pipeline = SeparationPipeline(
        tuss_inference=tuss,
        enable_mask_recycling=True,
        cache_size=5,
        similarity_threshold=0.85
    )
    print()
    
    print("Creating test audio (60s with repeated segments)...")
    test_audio = create_test_audio(duration_sec=60.0, sample_rate=tuss.sample_rate)
    print(f"  Audio shape: {test_audio.shape}")
    print(f"  Duration: {len(test_audio) / tuss.sample_rate:.1f}s")
    print()
    
    print("Processing audio with mask recycling...")
    sources_dict = pipeline.separate_waveform(test_audio, return_dict=True)
    print(f"  Separated sources: {list(sources_dict.keys())}")
    print()
    
    print("Mask Recycling Statistics:")
    stats = pipeline.get_stats()
    if stats:
        print(f"  Cache hits: {stats['hits']}")
        print(f"  Cache misses: {stats['misses']}")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
        print(f"  Inference reduction: {stats['inference_reduction']:.1%}")
        print(f"  Cache size: {stats['cache_size']}")
    print()


def demo_multi_coi():
    """Demonstrate multi-COI separation with mask recycling."""
    print("=" * 70)
    print("DEMO 2: Multi-COI Separation with Mask-Recycling")
    print("=" * 70)
    print()
    
    checkpoint_path = Path("checkpoints/tuss/")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    print("Loading TUSS model with multiple COI prompts...")
    tuss = TUSSInference.from_checkpoint(
        checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        coi_prompts=["airplane", "bird", "car"],  # Multiple COIs
        bg_prompt="background"
    )
    print(f"  COI prompts (sorted): {tuss.coi_prompts}")
    print(f"  Num sources: {tuss.num_sources}")
    print()
    
    print("Creating SeparationPipeline...")
    pipeline = SeparationPipeline(
        tuss_inference=tuss,
        enable_mask_recycling=True,
        cache_size=5,
        similarity_threshold=0.85
    )
    print()
    
    print("Creating test audio (30s)...")
    test_audio = create_test_audio(duration_sec=30.0, sample_rate=tuss.sample_rate)
    print()
    
    print("Processing audio...")
    sources_dict = pipeline.separate_waveform(test_audio, return_dict=True)
    print(f"  Separated sources: {list(sources_dict.keys())}")
    for name, audio in sources_dict.items():
        rms = torch.sqrt(torch.mean(audio ** 2)).item()
        print(f"    {name}: shape={audio.shape}, RMS={rms:.6f}")
    print()
    
    print("Mask Recycling Statistics:")
    stats = pipeline.get_stats()
    if stats:
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
        print(f"  Inference reduction: {stats['inference_reduction']:.1%}")
    print()


def demo_comparison():
    """Compare performance with and without mask recycling."""
    print("=" * 70)
    print("DEMO 3: Performance Comparison")
    print("=" * 70)
    print()
    
    checkpoint_path = Path("checkpoints/tuss/")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    print("Loading TUSS model...")
    tuss = TUSSInference.from_checkpoint(
        checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        coi_prompts=["airplane"],
        bg_prompt="background"
    )
    print()
    
    print("Creating test audio with highly repeated segments...")
    # Create audio where every other segment is identical
    test_audio = create_test_audio(duration_sec=40.0, sample_rate=tuss.sample_rate)
    print(f"  Duration: {len(test_audio) / tuss.sample_rate:.1f}s")
    print()
    
    # Test WITHOUT mask recycling
    print("Testing WITHOUT mask recycling...")
    pipeline_no_cache = SeparationPipeline(
        tuss_inference=tuss,
        enable_mask_recycling=False
    )
    
    import time
    start = time.time()
    _ = pipeline_no_cache.separate_waveform(test_audio, return_dict=True)
    time_no_cache = time.time() - start
    print(f"  Time: {time_no_cache:.2f}s")
    print()
    
    # Test WITH mask recycling
    print("Testing WITH mask recycling...")
    pipeline_with_cache = SeparationPipeline(
        tuss_inference=tuss,
        enable_mask_recycling=True,
        cache_size=5,
        similarity_threshold=0.85
    )
    
    start = time.time()
    _ = pipeline_with_cache.separate_waveform(test_audio, return_dict=True)
    time_with_cache = time.time() - start
    print(f"  Time: {time_with_cache:.2f}s")
    
    stats = pipeline_with_cache.get_stats()
    if stats:
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print()
    
    speedup = time_no_cache / time_with_cache
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time saved: {time_no_cache - time_with_cache:.2f}s ({(1 - time_with_cache/time_no_cache)*100:.1f}%)")
    print()


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("*  TUSS Mask-Recycling Demonstration")
    print("*" * 70)
    print("\n")
    
    try:
        demo_basic_usage()
        print("\n")
        
        demo_multi_coi()
        print("\n")
        
        demo_comparison()
        print("\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("*" * 70)
    print("*  All demos completed successfully!")
    print("*" * 70)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

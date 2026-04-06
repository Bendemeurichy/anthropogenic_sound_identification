"""
Separation pipeline that combines TUSS inference with mask recycling.

This module provides a pipeline that processes audio through TUSS separation
with optional mask recycling to reduce redundant inference calls on similar
audio segments.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Union

import torch

# Add parent directory for imports
_SCRIPT_DIR = Path(__file__).parent.resolve()
_SRC_DIR = _SCRIPT_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from models.tuss.inference import TUSSInference
from activity_filter.mask_recycler import MaskRecycler


class SeparationPipeline:
    """Pipeline that combines TUSS separation with mask recycling.
    
    Processes long audio by:
    1. Segmenting into 4s chunks with 50% overlap
    2. Checking cache for similar segments (mask recycling)
    3. Running TUSS inference only on cache misses
    4. Reconstructing full-length separated sources with overlap-add
    
    The mask recycling compares normalized audio segments using cosine similarity.
    When a similar segment is found in the cache (above the threshold), the cached
    separation result is reused instead of running inference.
    
    Example:
        >>> from models.tuss.inference import TUSSInference
        >>> from pipeline.separation_pipeline import SeparationPipeline
        >>> 
        >>> # Load TUSS with multiple COI prompts
        >>> tuss = TUSSInference.from_checkpoint(
        ...     "checkpoints/tuss/",
        ...     device="cuda",
        ...     coi_prompts=["airplane", "bird", "car"]
        ... )
        >>> 
        >>> # Create pipeline with mask recycling
        >>> pipeline = SeparationPipeline(
        ...     tuss_inference=tuss,
        ...     enable_mask_recycling=True,
        ...     cache_size=5,
        ...     similarity_threshold=0.85
        ... )
        >>> 
        >>> # Process audio
        >>> import torchaudio
        >>> waveform, sr = torchaudio.load("recording.wav")
        >>> 
        >>> # Separate into all sources
        >>> sources_dict = pipeline.separate_waveform(waveform)
        >>> # sources_dict = {
        >>> #     "airplane": tensor(T,),
        >>> #     "bird": tensor(T,),
        >>> #     "car": tensor(T,),
        >>> #     "background": tensor(T,),
        >>> # }
        >>> 
        >>> # Check efficiency
        >>> stats = pipeline.get_stats()
        >>> print(f"Cache hit rate: {stats['hit_rate']:.1%}")
    """
    
    def __init__(
        self,
        tuss_inference: TUSSInference,
        enable_mask_recycling: bool = True,
        cache_size: int = 5,
        similarity_threshold: float = 0.85,
    ):
        """Initialize separation pipeline.
        
        Args:
            tuss_inference: TUSSInference instance (already loaded with prompts)
            enable_mask_recycling: Whether to enable mask recycling optimization
            cache_size: Number of segments to cache (default: 5)
            similarity_threshold: Cosine similarity threshold for cache reuse (0-1).
                                Higher values are more conservative. Default: 0.85
        """
        self.tuss = tuss_inference
        self.enable_recycling = enable_mask_recycling
        
        if enable_mask_recycling:
            self.recycler = MaskRecycler(cache_size, similarity_threshold)
        else:
            self.recycler = None
    
    def separate_waveform(
        self,
        waveform: torch.Tensor,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Separate a waveform into COI classes + background.
        
        For long audio (> segment length), processes in overlapping chunks with
        mask recycling optimization.
        
        Args:
            waveform: Input audio (T,) or (C, T) at model's sample rate.
                     Multi-channel audio is averaged to mono.
            return_dict: If True, return dict mapping names to audio.
                        If False, return tensor (n_sources, T). Default: True
        
        Returns:
            If return_dict=True:
                Dictionary mapping prompt names to separated audio:
                {
                    "airplane": tensor(T,),
                    "bird": tensor(T,),
                    "background": tensor(T,),
                }
            
            If return_dict=False:
                Tensor of shape (n_sources, T) where:
                    sources[0:len(coi_prompts)] = COI classes
                    sources[-1] = background
        """
        # Ensure 1D
        if waveform.dim() == 2:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
        
        original_length = waveform.shape[0]
        
        # Process in overlapping chunks for long audio
        if waveform.shape[0] > self.tuss.segment_samples:
            sources = self._separate_long(waveform, original_length)
        else:
            # Single segment
            if waveform.shape[0] < self.tuss.segment_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.tuss.segment_samples - waveform.shape[0])
                )
            sources = self._separate_segment_with_cache(waveform)
            sources = sources[:, :original_length]
        
        # Convert to dictionary if requested
        if return_dict:
            return self.tuss.get_sources_dict(sources)
        else:
            return sources
    
    def _separate_segment_with_cache(
        self,
        segment: torch.Tensor
    ) -> torch.Tensor:
        """Separate a segment with cache check.
        
        Normalizes the segment and checks the cache for similar segments.
        On cache miss, runs TUSS inference and updates the cache.
        
        Args:
            segment: Input segment (segment_samples,)
        
        Returns:
            sources: (n_sources, segment_samples) tensor
        """
        # Normalize for cache comparison (same as inference normalization)
        mean = segment.mean()
        std = segment.std() + 1e-8
        normalized = (segment - mean) / std
        
        # Check cache if enabled
        if self.enable_recycling:
            is_hit, cached_sources = self.recycler.check_cache(normalized)
            if is_hit:
                return cached_sources
        
        # Cache miss - run inference
        sources = self.tuss._separate_segment(segment)
        
        # Update cache
        if self.enable_recycling:
            self.recycler.update_cache(normalized, sources)
        
        return sources
    
    def _separate_long(
        self,
        waveform: torch.Tensor,
        original_length: int
    ) -> torch.Tensor:
        """Process long audio with overlap-add and caching.
        
        Uses Hann windowing with 50% overlap for smooth transitions.
        Each segment is processed through the cache before inference.
        
        Args:
            waveform: Input waveform (T,) where T > segment_samples
            original_length: Original length to trim output to
        
        Returns:
            sources: (n_sources, T) tensor
        """
        hop = self.tuss.segment_samples // 2
        window = torch.hann_window(self.tuss.segment_samples)
        
        # Initialize output buffers
        n_sources = self.tuss.num_sources
        output = torch.zeros(n_sources, original_length)
        weight = torch.zeros(original_length)
        
        # Process overlapping segments
        for start in range(0, waveform.shape[0], hop):
            chunk = waveform[start : start + self.tuss.segment_samples]
            if chunk.shape[0] < self.tuss.segment_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, self.tuss.segment_samples - chunk.shape[0])
                )
            
            # Separate with caching
            sources = self._separate_segment_with_cache(chunk)
            
            end = min(start + self.tuss.segment_samples, original_length)
            length = end - start
            
            # Add windowed segment (broadcast window across all sources)
            output[:, start:end] += sources[:, :length] * window[:length]
            weight[start:end] += window[:length]
        
        # Normalize by overlap weight
        return output / (weight + 1e-8)
    
    def get_stats(self) -> Optional[dict]:
        """Get mask recycling statistics.
        
        Returns:
            Dictionary with cache statistics if mask recycling is enabled:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - total_requests: Total cache queries
                - hit_rate: Proportion of hits (0-1)
                - inference_reduction: Proportion of avoided inferences
                - cache_size: Current number of cached segments
            
            Returns None if mask recycling is disabled.
        """
        if self.recycler:
            return self.recycler.get_stats()
        return None
    
    def reset_stats(self):
        """Reset mask recycling statistics counters."""
        if self.recycler:
            self.recycler.reset_stats()
    
    def clear_cache(self):
        """Clear all cached segments."""
        if self.recycler:
            self.recycler.clear_cache()

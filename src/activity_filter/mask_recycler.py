"""
Mask recycling module for reducing inference runs on similar audio segments.

This module implements a caching mechanism that stores recent separation results
and reuses them when processing similar audio segments. Similarity is determined
using cosine similarity on normalized audio waveforms.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class CachedSegment:
    """Stores a cached segment with its normalized input and separated sources.
    
    Attributes:
        normalized_input: Normalized audio segment (T,) used for similarity comparison
        sources: Separated audio sources (n_sources, T) from TUSS model
    """
    normalized_input: torch.Tensor  # (T,) - normalized audio
    sources: torch.Tensor  # (n_sources, T) - all separated sources


class MaskRecycler:
    """Caches recent segment separation results and reuses them based on similarity.
    
    This works with multi-source separation where each inference call produces
    multiple COI classes + background in a single forward pass.
    
    The cache stores normalized input audio and corresponding separated sources.
    When a new segment arrives, it's compared against cached segments using
    cosine similarity. If a match is found above the threshold, the cached
    separation result is reused instead of running inference.
    
    Example:
        >>> recycler = MaskRecycler(cache_size=5, similarity_threshold=0.85)
        >>> 
        >>> # Check if similar segment exists
        >>> is_hit, cached_sources = recycler.check_cache(normalized_audio)
        >>> 
        >>> if not is_hit:
        >>>     # Run inference
        >>>     sources = model.separate(audio)
        >>>     # Update cache
        >>>     recycler.update_cache(normalized_audio, sources)
        >>> else:
        >>>     sources = cached_sources
        >>> 
        >>> # Check statistics
        >>> stats = recycler.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
    """
    
    def __init__(
        self,
        cache_size: int = 5,
        similarity_threshold: float = 0.85,
    ):
        """Initialize mask recycler.
        
        Args:
            cache_size: Number of recent segments to cache (FIFO eviction)
            similarity_threshold: Cosine similarity threshold for reuse (0-1).
                                Higher values are more conservative (only reuse
                                very similar segments).
        """
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.cache: list[CachedSegment] = []
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def check_cache(
        self,
        normalized_segment: torch.Tensor
    ) -> Tuple[bool, Optional[torch.Tensor]]:
        """Check if a similar segment exists in cache.
        
        Compares the input segment against all cached segments using cosine
        similarity. Returns the first match found above the threshold.
        
        Args:
            normalized_segment: Normalized input segment (T,) with same
                              normalization as used during inference
                              (zero-mean, unit-variance)
        
        Returns:
            Tuple of (is_hit, cached_sources):
                - is_hit: True if similar segment found
                - cached_sources: (n_sources, T) tensor if hit, None if miss
        """
        # Empty cache
        if not self.cache:
            self.misses += 1
            return False, None
        
        # Check if segment is silent (avoid division by zero in cosine sim)
        rms = torch.sqrt(torch.mean(normalized_segment ** 2))
        if rms < 1e-6:
            # Silent segment - always run inference (don't cache silent audio)
            self.misses += 1
            return False, None
        
        # Compare with cached segments (check most recent first)
        for cached in reversed(self.cache):
            similarity = self.compute_similarity(
                normalized_segment,
                cached.normalized_input
            )
            
            if similarity >= self.similarity_threshold:
                self.hits += 1
                return True, cached.sources.clone()
        
        # No match found
        self.misses += 1
        return False, None
    
    def update_cache(
        self,
        normalized_segment: torch.Tensor,
        sources: torch.Tensor
    ):
        """Add new segment to cache.
        
        Stores the normalized input and corresponding separated sources.
        If cache is full, removes the oldest entry (FIFO).
        
        Args:
            normalized_segment: Normalized input (T,)
            sources: Separated sources (n_sources, T)
        """
        cached = CachedSegment(
            normalized_input=normalized_segment.clone(),
            sources=sources.clone()
        )
        
        self.cache.append(cached)
        
        # Maintain cache size limit (FIFO eviction)
        if len(self.cache) > self.cache_size:
            self.cache.pop(0)
    
    def compute_similarity(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor
    ) -> float:
        """Compute cosine similarity between two audio segments.
        
        Both inputs should be normalized (zero-mean, unit-variance) using
        the same normalization as applied during inference.
        
        Args:
            audio1: First audio segment (T,)
            audio2: Second audio segment (T,)
        
        Returns:
            Cosine similarity in [0, 1] (absolute value, phase-independent)
        """
        # Flatten to 1D if needed
        a1 = audio1.flatten()
        a2 = audio2.flatten()
        
        # Handle length mismatch (shouldn't happen but be defensive)
        min_len = min(len(a1), len(a2))
        a1 = a1[:min_len]
        a2 = a2[:min_len]
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(a1.unsqueeze(0), a2.unsqueeze(0))
        
        # Return absolute value (phase doesn't matter for audio similarity)
        return abs(similarity.item())
    
    def get_stats(self) -> dict:
        """Return cache statistics.
        
        Returns:
            Dictionary containing:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - total_requests: Total cache queries
                - hit_rate: Proportion of hits (0-1)
                - inference_reduction: Same as hit_rate (proportion of
                                     avoided inference calls)
                - cache_size: Current number of cached segments
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "inference_reduction": hit_rate,
            "cache_size": len(self.cache),
        }
    
    def reset_stats(self):
        """Reset statistics counters (hits and misses)."""
        self.hits = 0
        self.misses = 0
    
    def clear_cache(self):
        """Clear all cached segments."""
        self.cache.clear()

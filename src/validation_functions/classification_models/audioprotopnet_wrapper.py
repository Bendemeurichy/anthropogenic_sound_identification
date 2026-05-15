import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
from typing import Tuple
import os

class AudioProtoPNetClassifierWrapper:
    """Wrapper for AudioProtoPNet-20-BirdSet-XCL model from HuggingFace."""

    # AudioProtoPNet activates 9736 classes × 20 prototypes per forward pass,
    # which requires ~150-200 MB per sample.  A sub-batch of 4 stays well
    # within a 14 GiB GPU even after the separator has consumed ~10 GiB.
    SUB_BATCH_SIZE = 4

    def __init__(self, model_id="DBD-research-group/AudioProtoPNet-20-BirdSet-XCL", device="cpu", threshold=0.5, **kwargs):
        self.device = device
        self.threshold = threshold
        self.model_id = model_id
        
        print(f"Loading AudioProtoPNet model from {self.model_id}...")
        
        # Windows compatibility: trust_remote_code causes SIGALRM error on Windows
        # Always trust code for HuggingFace models to avoid timeout issues
        # Set env var to disable symlink warnings
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        
        # Load with trust_remote_code=True and local_files_only fallback for faster loading
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                local_files_only=True
            ).to(self.device)
        except Exception as e:
            # Fallback to downloading if not cached
            print(f"  Model not cached locally (error: {e}), downloading from HuggingFace (this may take a while)...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                trust_remote_code=True
            ).to(self.device)
        
        self.model.eval()
        
        self._sample_rate = self.feature_extractor.sampling_rate
        # The HF feature extractor uses padding="longest" with max_length=5*sr
        # and truncation=True (see processing_protonet.py.__call__,
        # clip_duration=5). Anything longer than 5 s is silently truncated, so
        # we must expose segment_samples here to let _classify_recording chunk
        # the waveform at the classifier's native 5 s window before calling us.
        self._segment_length = 5.0
        self._segment_samples = int(self._sample_rate * self._segment_length)
        print(f"  AudioProtoPNet loaded successfully (sample_rate={self._sample_rate} Hz)")
        print(f"  Segment length: {self._segment_length} s ({self._segment_samples} samples)")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def segment_samples(self) -> int:
        """Preferred classifier input length in samples (5 s @ sampling_rate)."""
        return self._segment_samples

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor) -> Tuple[int, float]:
        """Single waveform inference."""
        preds, confs = self.predict_batch(waveform.unsqueeze(0))
        return int(preds[0].item()), float(confs[0].item())

    @torch.inference_mode()
    def predict_batch(self, waveforms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch inference for better performance.

        Internally processes in sub-batches of SUB_BATCH_SIZE to avoid CUDA
        OOM errors: AudioProtoPNet computes prototype activations over ~9 736
        bird classes × 20 prototypes per forward pass, which is very
        memory-intensive.  Splitting into small chunks and clearing the CUDA
        cache between them keeps peak VRAM manageable.

        Args:
            waveforms: (B, T) batch of waveforms (CPU or GPU, any dtype)

        Returns:
            Tuple of (predictions, confidences) both shape (B,)
        """
        all_preds: list[torch.Tensor] = []
        all_max_probs: list[torch.Tensor] = []

        for start in range(0, len(waveforms), self.SUB_BATCH_SIZE):
            sub = waveforms[start : start + self.SUB_BATCH_SIZE]

            # Feature extraction expects a list/array on CPU
            sub_np = sub.cpu().numpy()
            inputs = self.feature_extractor(
                sub_np,
                return_tensors="pt",
                padding=True,
            )

            # The custom feature extractor may return a raw Tensor instead of
            # a dict-like BatchEncoding, so handle both cases.
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)

            logits = outputs.logits

            # Take max probability across all bird classes
            probs = torch.sigmoid(logits)
            max_probs, _ = torch.max(probs, dim=1)
            preds = (max_probs >= self.threshold).long()

            all_preds.append(preds.cpu())
            all_max_probs.append(max_probs.cpu())

            # Release GPU tensors and defragment allocator between sub-batches
            del inputs, outputs, logits, probs, max_probs, preds
            if self.device != "cpu":
                torch.cuda.empty_cache()

        return torch.cat(all_preds), torch.cat(all_max_probs)

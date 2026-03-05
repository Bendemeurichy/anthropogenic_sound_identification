"""Module that detects if segment is different from previous segment based on averaged kernel with running window.
If the distance is above a certain threshold, the segment is considered different."""

import numpy as np


class ChangeDetection:
    def __init__(self, kernel_size: int = 3, threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.kernel = np.ones(kernel_size) / kernel_size
        self.previous_segment = None

    def detect_change(self, segment: np.ndarray) -> bool:
        pass

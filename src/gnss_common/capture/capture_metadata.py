# src/gnss_l5/capture_metadata.py

import numpy as np
from dataclasses import dataclass


@dataclass
class CaptureMetadata:
    sample_rate: float      # Hz
    center_freq: float      # Hz — IF or RF center frequency
    dtype: np.dtype
    is_complex: bool
    num_channels: int = 1

    def ms_to_samples(self, milliseconds: float) -> int:
        """Convert a duration in milliseconds to the nearest sample count."""
        return round(self.sample_rate * milliseconds * 1e-3)
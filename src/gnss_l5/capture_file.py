# src/gnss_l5/capture_file.py

import numpy as np
from pathlib import Path


SUPPORTED_DTYPES = (np.int8, np.int16, np.float32)


# src/gnss_l5/capture_file.py

import numpy as np
from pathlib import Path


SUPPORTED_DTYPES = (np.int8, np.int16, np.float32)


class CaptureFile:

    def __init__(self, path, dtype, is_complex: bool, num_channels: int = 1):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Capture file not found: {path}")
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype {dtype}. Must be one of {SUPPORTED_DTYPES}")

        self._path = path
        self._dtype = np.dtype(dtype)
        self._is_complex = is_complex
        self._num_channels = num_channels
        self._components_per_sample = (2 if is_complex else 1) * num_channels

        total_raw = path.stat().st_size // self._dtype.itemsize
        self._num_samples = total_raw // self._components_per_sample

        self._mmap = np.memmap(path, dtype=self._dtype, mode='r',
                               shape=(total_raw,))

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def read(self, start: int = 0, count: int = None) -> np.ndarray:
        if count is None:
            count = self._num_samples - start

        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if count <= 0:
            raise ValueError(f"count must be > 0, got {count}")
        if start + count > self._num_samples:
            raise ValueError(
                f"Read out of range: start={start}, count={count}, "
                f"num_samples={self._num_samples}"
            )

        raw_start = start * self._components_per_sample
        raw_count = count * self._components_per_sample
        raw = self._mmap[raw_start : raw_start + raw_count]

        if self._is_complex:
            raw = raw.reshape(count, self._num_channels, 2)
            data = raw[..., 0].astype(np.float64) + 1j * raw[..., 1].astype(np.float64)
        else:
            raw = raw.reshape(count, self._num_channels)
            data = raw.astype(np.float64)

        if self._num_channels == 1:
            data = data.squeeze(axis=1)

        return data
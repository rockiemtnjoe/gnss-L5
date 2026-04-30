# tests/test_capture_metadata.py

import numpy as np
import pytest
from gnss_common.capture.capture_metadata import CaptureMetadata


@pytest.fixture
def meta():
    return CaptureMetadata(
        sample_rate=25e6,
        center_freq=1176.45e6,  # GPS L5 center frequency
        dtype=np.int8,
        is_complex=True,
    )


class TestConstruction:

    def test_instantiation_succeeds(self, meta):
        assert meta is not None

    def test_sample_rate_stored(self, meta):
        assert meta.sample_rate == 25e6

    def test_center_freq_stored(self, meta):
        assert meta.center_freq == 1176.45e6

    def test_dtype_stored(self, meta):
        assert meta.dtype == np.int8

    def test_is_complex_stored(self, meta):
        assert meta.is_complex == True

    def test_num_channels_defaults_to_one(self, meta):
        assert meta.num_channels == 1


class TestMsToSamples:

    def test_one_millisecond(self, meta):
        """25 MHz * 1ms = 25,000 samples."""
        assert meta.ms_to_samples(1.0) == 25_000

    def test_ten_milliseconds(self, meta):
        """25 MHz * 10ms = 250,000 samples."""
        assert meta.ms_to_samples(10.0) == 250_000

    def test_fractional_milliseconds(self, meta):
        """25 MHz * 0.5ms = 12,500 samples."""
        assert meta.ms_to_samples(0.5) == 12_500

    def test_rounding_up(self):
        """Verify round() not int() — 999.9 rounds to 1000, int() gives 999."""
        meta = CaptureMetadata(
            sample_rate=999_900,   # contrived rate to force rounding
            center_freq=0,
            dtype=np.int8,
            is_complex=True,
        )
        # 999900 * 1e-3 = 999.9 → round gives 1000, int gives 999
        assert meta.ms_to_samples(1.0) == 1000

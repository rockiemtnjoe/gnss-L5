# tests/test_capture_file.py
"""
Tests for CaptureFile — TDD Milestone 1.

CaptureFile contract:
    CaptureFile(path, dtype, is_complex, num_channels=1)

    .num_samples      -> int   (complex samples, per channel)
    .num_channels     -> int
    .read(start=0, count=None) -> np.ndarray
        single channel:  shape (count,)          dtype complex or real
        multi-channel:   shape (count, channels) dtype complex or real

Binary layout on disk (int8, complex, 2-channel example):
    I0_chA  Q0_chA  I0_chB  Q0_chB  I1_chA  Q1_chA  ...

Design principle: CaptureFile knows ONLY what is needed to correctly
interpret the binary layout.  sample_rate, center_freq, IF, gain, etc.
are signal-processing metadata and live elsewhere.
"""

import numpy as np
import pytest
from pathlib import Path

from gnss_l5.capture_file import CaptureFile


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------
def make_iq_file(path: Path, num_samples: int, dtype, num_channels: int = 1) -> Path:
    """
    Write a synthetic interleaved I/Q file.

    I values: channel index * 20 + sample index  (e.g. ch0 s3 -> 3, ch1 s3 -> 103)
    Q values: I value + 10
    Interleaving order per frame: I_ch0 Q_ch0 I_ch1 Q_ch1 ...
    """
    raw = []
    for s in range(num_samples):
        for ch in range(num_channels):
            i_val = ch * 20 + s      # was ch * 100 + s
            q_val = i_val + 10       # was i_val + 50
            raw.extend([i_val, q_val])
    arr = np.array(raw, dtype=dtype)
    arr.tofile(path)
    return path

@pytest.fixture
def iq_file_10samples(tmp_path: Path) -> Path:
    """Single-channel int8 complex I/Q, 10 samples."""
    return make_iq_file(tmp_path / "single_ch.bin", num_samples=10, dtype=np.int8)


@pytest.fixture
def iq_file_2ch(tmp_path: Path) -> Path:
    """Two-channel int8 complex I/Q, 10 samples per channel."""
    return make_iq_file(tmp_path / "two_ch.bin", num_samples=10,
                        dtype=np.int8, num_channels=2)


@pytest.fixture
def cf_single(iq_file_10samples: Path) -> CaptureFile:
    return CaptureFile(iq_file_10samples, dtype=np.int8, is_complex=True)


@pytest.fixture
def cf_two_ch(iq_file_2ch: Path) -> CaptureFile:
    return CaptureFile(iq_file_2ch, dtype=np.int8, is_complex=True, num_channels=2)


# ---------------------------------------------------------------------------
# Group 1: Construction & metadata
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_instantiation_succeeds(self, cf_single):
        assert cf_single is not None

    def test_num_samples_single_channel(self, cf_single):
        """20 bytes / (1 ch * 2 bytes-per-complex-int8-sample) = 10 samples."""
        assert cf_single.num_samples == 10

    def test_num_channels_defaults_to_one(self, cf_single):
        assert cf_single.num_channels == 1

    def test_num_channels_two(self, cf_two_ch):
        assert cf_two_ch.num_channels == 2

    def test_num_samples_two_channel(self, cf_two_ch):
        """40 bytes / (2 ch * 2 bytes-per-complex-int8-sample) = 10 samples."""
        assert cf_two_ch.num_samples == 10

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CaptureFile(tmp_path / "ghost.bin", dtype=np.int8, is_complex=True)

    def test_unsupported_dtype_raises(self, iq_file_10samples):
        with pytest.raises(ValueError, match="dtype"):
            CaptureFile(iq_file_10samples, dtype=np.float64, is_complex=True)

    def test_no_sample_rate_attribute(self, cf_single):
        """Explicit contract test: sample_rate is NOT CaptureFile's responsibility."""
        assert not hasattr(cf_single, 'sample_rate')


# ---------------------------------------------------------------------------
# Group 2: Single-channel reads
# ---------------------------------------------------------------------------

class TestSingleChannelReading:

    def test_read_all_returns_correct_length(self, cf_single):
        assert len(cf_single.read()) == 10

    def test_read_returns_complex(self, cf_single):
        assert np.iscomplexobj(cf_single.read())

    def test_read_first_sample(self, cf_single):
        """Sample 0: I=0, Q=10 -> 0+10j"""
        data = cf_single.read(start=0, count=1)
        assert data[0] == pytest.approx(0 + 10j)

    def test_read_last_sample(self, cf_single):
        """Sample 9: I=9, Q=19 -> 9+19j"""
        data = cf_single.read(start=9, count=1)
        assert data[0] == pytest.approx(9 + 19j)

    def test_read_middle_slice(self, cf_single):
        """Samples 3-5: I=[3,4,5], Q=[13,14,15]"""
        data = cf_single.read(start=3, count=3)
        expected = np.array([3+13j, 4+14j, 5+15j])
        np.testing.assert_array_almost_equal(data, expected)

    def test_read_default_start_is_zero(self, cf_single):
        assert (cf_single.read(count=3) == cf_single.read(start=0, count=3)).all()

    def test_read_default_count_reads_all(self, cf_single):
        assert len(cf_single.read()) == cf_single.num_samples


# ---------------------------------------------------------------------------
# Group 3: Multi-channel reads
# ---------------------------------------------------------------------------

class TestMultiChannelReading:

    def test_read_returns_2d_array(self, cf_two_ch):
        data = cf_two_ch.read()
        assert data.ndim == 2

    def test_read_shape_is_samples_by_channels(self, cf_two_ch):
        data = cf_two_ch.read()
        assert data.shape == (10, 2)

# TestMultiChannelReading — ch0: offset=0, ch1: offset=20, q_offset=10
    def test_channel_zero_first_sample(self, cf_two_ch):
        """ch0, sample 0: I=0, Q=10 -> 0+10j"""
        data = cf_two_ch.read(start=0, count=1)
        assert data[0, 0] == pytest.approx(0 + 10j)

    def test_channel_one_first_sample(self, cf_two_ch):
        """ch1, sample 0: I=20, Q=30 -> 20+30j"""
        data = cf_two_ch.read(start=0, count=1)
        assert data[0, 1] == pytest.approx(20 + 30j)

    def test_channel_zero_later_sample(self, cf_two_ch):
        """ch0, sample 5: I=5, Q=15 -> 5+15j"""
        data = cf_two_ch.read(start=5, count=1)
        assert data[0, 0] == pytest.approx(5 + 15j)

# ---------------------------------------------------------------------------
# Group 4: Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_read_past_end_raises(self, cf_single):
        with pytest.raises(ValueError, match="out of range"):
            cf_single.read(start=8, count=5)   # 8+5=13 > 10

    def test_start_at_exact_end_raises(self, cf_single):
        with pytest.raises(ValueError):
            cf_single.read(start=10, count=1)  # start == num_samples

    def test_negative_start_raises(self, cf_single):
        with pytest.raises(ValueError):
            cf_single.read(start=-1, count=2)

    def test_zero_count_raises(self, cf_single):
        with pytest.raises(ValueError):
            cf_single.read(start=0, count=0)

@pytest.fixture
def real_file_10samples(tmp_path: Path) -> Path:
    """Single-channel int8 real (non-complex) file, 10 samples."""
    data = np.arange(10, dtype=np.int8)
    path = tmp_path / "real.bin"
    data.tofile(path)
    return path


class TestRealSamples:

    def test_num_samples_real(self, tmp_path):
        path = tmp_path / "real.bin"
        np.arange(10, dtype=np.int8).tofile(path)
        cf = CaptureFile(path, dtype=np.int8, is_complex=False)
        assert cf.num_samples == 10

    def test_read_returns_real(self, tmp_path):
        path = tmp_path / "real.bin"
        np.arange(10, dtype=np.int8).tofile(path)
        cf = CaptureFile(path, dtype=np.int8, is_complex=False)
        data = cf.read()
        assert not np.iscomplexobj(data)

    def test_read_values_correct(self, tmp_path):
        path = tmp_path / "real.bin"
        np.arange(10, dtype=np.int8).tofile(path)
        cf = CaptureFile(path, dtype=np.int8, is_complex=False)
        data = cf.read(start=2, count=3)
        np.testing.assert_array_equal(data, [2.0, 3.0, 4.0])
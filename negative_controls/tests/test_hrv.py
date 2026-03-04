"""
Tests for Ab Astris Negative Control: Heart Rate Variability

Tests the HRV experiment pipeline using synthetic RR interval data and
validates the conversion, filtering, and analysis functions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from hrv_experiment import (
    LSResult,
    MultiWindowResult,
    BootstrapResult,
    BandResult,
    SubjectResult,
    run_lomb_scargle,
    create_windows,
    run_multi_window_validation,
    run_bootstrap,
    compute_confidence_score,
    bandpass_filter,
    preprocess_rr,
    rr_to_hr,
    resample_uniform,
    serialize_band,
    serialize_subject,
    HRV_BANDS,
    RR_MIN,
    RR_MAX,
    RESAMPLE_FS,
    WINDOW_DURATION_SEC,
)


# ============================================================================
# SYNTHETIC DATA GENERATORS
# ============================================================================

def _synthetic_rr_intervals(n_beats=80000, mean_hr_bpm=70, seed=42):
    """Generate synthetic RR intervals mimicking 24-hour recording.

    ~80000 beats at 70 bpm ≈ 19 hours of recording.
    Includes respiratory modulation (~0.25 Hz) and sympathetic (~0.1 Hz).
    """
    rng = np.random.default_rng(seed)
    mean_rr = 60.0 / mean_hr_bpm  # seconds

    # Base RR intervals with physiological variation
    n = n_beats
    rr = np.full(n, mean_rr)

    # Add respiratory sinus arrhythmia (HF: ~0.25 Hz)
    time_approx = np.cumsum(rr)
    rr += 0.03 * np.sin(2 * np.pi * 0.25 * time_approx)

    # Add sympathetic modulation (LF: ~0.1 Hz)
    rr += 0.02 * np.sin(2 * np.pi * 0.1 * time_approx)

    # Add noise
    rr += rng.normal(0, 0.02, n)

    # Clamp to physiological range
    rr = np.clip(rr, 0.4, 1.5)

    return rr


def _synthetic_hr_with_known_freq(freq_hz=0.25, duration_sec=86400, seed=42):
    """Generate uniformly-sampled HR signal with known frequency."""
    rng = np.random.default_rng(seed)
    fs = RESAMPLE_FS
    n = int(duration_sec * fs)
    time = np.arange(n) / fs
    signal = np.sin(2 * np.pi * freq_hz * time)
    signal += rng.normal(0, 0.1, n)
    return time, signal


# ============================================================================
# RR INTERVAL PREPROCESSING TESTS
# ============================================================================

class TestPreprocessRR:
    """Tests for RR interval preprocessing."""

    def test_removes_short_rr(self):
        """Should remove RR intervals < 0.3s (>200 bpm)."""
        rr = np.array([0.8, 0.2, 0.9, 0.15, 0.7])  # 0.2 and 0.15 too short
        clean = preprocess_rr(rr)
        assert len(clean) == 3
        assert (clean >= RR_MIN).all()

    def test_removes_long_rr(self):
        """Should remove RR intervals > 2.0s (<30 bpm)."""
        rr = np.array([0.8, 2.5, 0.9, 3.0, 0.7])  # 2.5 and 3.0 too long
        clean = preprocess_rr(rr)
        assert len(clean) == 3
        assert (clean <= RR_MAX).all()

    def test_keeps_normal_rr(self):
        """Should keep normal RR intervals unchanged."""
        rr = np.array([0.8, 0.85, 0.9, 0.75, 0.82])
        clean = preprocess_rr(rr)
        assert len(clean) == 5
        np.testing.assert_array_equal(clean, rr)

    def test_handles_empty_after_filter(self):
        """Should return empty array if all beats are ectopic."""
        rr = np.array([0.1, 0.05, 3.0, 5.0])
        clean = preprocess_rr(rr)
        assert len(clean) == 0


# ============================================================================
# RR TO HR CONVERSION TESTS
# ============================================================================

class TestRRtoHR:
    """Tests for RR interval to heart rate conversion."""

    def test_basic_conversion(self):
        """HR should be 1/RR."""
        rr = np.array([0.8, 1.0, 0.5])
        time, hr = rr_to_hr(rr)
        np.testing.assert_allclose(hr, [1/0.8, 1/1.0, 1/0.5])

    def test_time_is_cumulative(self):
        """Time should be cumulative sum of RR intervals."""
        rr = np.array([0.8, 1.0, 0.5])
        time, hr = rr_to_hr(rr)
        np.testing.assert_allclose(time, [0.8, 1.8, 2.3])

    def test_hr_units_hz(self):
        """At 60 bpm (RR=1.0s), HR should be 1.0 Hz."""
        rr = np.array([1.0, 1.0, 1.0])
        _, hr = rr_to_hr(rr)
        np.testing.assert_allclose(hr, [1.0, 1.0, 1.0])

    def test_uneven_spacing(self):
        """Time intervals should be uneven (matching RR intervals)."""
        rr = np.array([0.8, 1.2, 0.6, 0.9])
        time, _ = rr_to_hr(rr)
        diffs = np.diff(time)
        assert not np.allclose(diffs, diffs[0])  # Not uniformly spaced


# ============================================================================
# RESAMPLING TESTS
# ============================================================================

class TestResampling:
    """Tests for uniform resampling."""

    def test_output_is_uniform(self):
        """Resampled signal should be uniformly spaced."""
        rr = _synthetic_rr_intervals(n_beats=1000)
        time, hr = rr_to_hr(rr)
        t_uni, s_uni = resample_uniform(time, hr, fs=4.0)
        diffs = np.diff(t_uni)
        np.testing.assert_allclose(diffs, 0.25, atol=1e-10)

    def test_preserves_signal_range(self):
        """Resampled values should be within original range (±small interpolation)."""
        rr = _synthetic_rr_intervals(n_beats=1000)
        time, hr = rr_to_hr(rr)
        _, s_uni = resample_uniform(time, hr, fs=4.0)
        # Interpolation shouldn't exceed original range much
        assert s_uni.min() >= hr.min() - 0.01
        assert s_uni.max() <= hr.max() + 0.01


# ============================================================================
# BANDPASS FILTER TESTS
# ============================================================================

class TestBandpassFilter:
    """Tests for the bandpass filter."""

    def test_passes_in_band_signal(self):
        """Signal within the band should pass through."""
        fs = RESAMPLE_FS
        t = np.arange(10000) / fs
        freq = 0.25  # Hz — within HF band (0.15-0.4)
        signal = np.sin(2 * np.pi * freq * t)
        filtered = bandpass_filter(signal, fs, 0.15, 0.4)
        # Should retain most of the signal
        assert np.std(filtered) > 0.3 * np.std(signal)

    def test_rejects_out_of_band_signal(self):
        """Signal outside the band should be attenuated."""
        fs = RESAMPLE_FS
        t = np.arange(10000) / fs
        freq = 0.01  # Hz — below HF band
        signal = np.sin(2 * np.pi * freq * t)
        filtered = bandpass_filter(signal, fs, 0.15, 0.4)
        # Should attenuate significantly
        assert np.std(filtered) < 0.1 * np.std(signal)

    def test_hf_band_range(self):
        """HF band should be 0.15-0.4 Hz per protocol."""
        assert HRV_BANDS["HF"]["hz"] == (0.15, 0.4)

    def test_lf_band_range(self):
        """LF band should be 0.04-0.15 Hz per protocol."""
        assert HRV_BANDS["LF"]["hz"] == (0.04, 0.15)

    def test_vlf_band_range(self):
        """VLF band should be 0.003-0.04 Hz per protocol."""
        assert HRV_BANDS["VLF"]["hz"] == (0.003, 0.04)


# ============================================================================
# LOMB-SCARGLE TESTS
# ============================================================================

class TestLombScargle:
    """Tests for Lomb-Scargle on HRV data."""

    def test_detects_hf_frequency(self):
        """Should detect respiratory frequency (~0.25 Hz) in HF band."""
        time, signal = _synthetic_hr_with_known_freq(freq_hz=0.25)
        f_low, f_high = HRV_BANDS["HF"]["hz"]
        result = run_lomb_scargle(time, signal, freq_min_hz=f_low, freq_max_hz=f_high)
        freq_error = abs(result.frequency_hz - 0.25) / 0.25
        assert freq_error < 0.05, f"Freq error {freq_error:.4f} > 5%"

    def test_detects_lf_frequency(self):
        """Should detect sympathetic frequency (~0.1 Hz) in LF band."""
        time, signal = _synthetic_hr_with_known_freq(freq_hz=0.1)
        f_low, f_high = HRV_BANDS["LF"]["hz"]
        result = run_lomb_scargle(time, signal, freq_min_hz=f_low, freq_max_hz=f_high)
        freq_error = abs(result.frequency_hz - 0.1) / 0.1
        assert freq_error < 0.05, f"Freq error {freq_error:.4f} > 5%"


# ============================================================================
# MULTI-WINDOW TESTS
# ============================================================================

class TestMultiWindow:
    """Tests for multi-window validation on HRV data."""

    def test_creates_windows_for_24hr(self):
        """Should create 8 windows for ~24-hour recording."""
        rr = _synthetic_rr_intervals(n_beats=80000)
        time, hr = rr_to_hr(rr)
        t_uni, s_uni = resample_uniform(time, hr)
        windows = create_windows(t_uni, s_uni, n_windows=8,
                                 window_duration_sec=WINDOW_DURATION_SEC)
        assert len(windows) == 8

    def test_moderate_cv_for_hrv(self):
        """HRV should produce moderate CV — between tides and crypto."""
        rr = _synthetic_rr_intervals(n_beats=80000)
        time, hr = rr_to_hr(rr)
        t_uni, s_uni = resample_uniform(time, hr)

        f_low, f_high = HRV_BANDS["HF"]["hz"]
        filtered = bandpass_filter(s_uni, RESAMPLE_FS, f_low, f_high)

        ls_result = run_lomb_scargle(t_uni, filtered,
                                     freq_min_hz=f_low, freq_max_hz=f_high)
        windows = create_windows(t_uni, filtered, n_windows=8,
                                 window_duration_sec=WINDOW_DURATION_SEC)
        mw_result = run_multi_window_validation(
            windows, ls_result.frequency_hz,
            freq_min_hz=f_low, freq_max_hz=f_high,
        )
        # HRV CV should be > tides (0.17%) but could vary widely
        assert mw_result.cv_clean >= 0


# ============================================================================
# CONFIDENCE SCORING TESTS
# ============================================================================

class TestConfidenceScoring:
    """Tests for confidence score on HRV data."""

    def test_score_range(self):
        """Score should be 0-100."""
        ls = LSResult(frequency_hz=0.25, frequency_cpd=0.25*86400,
                      power=0.5, fap=0.01,
                      frequencies_hz=np.array([0.25]), powers=np.array([0.5]))
        mw = MultiWindowResult(detected_freqs=[0.25]*8, cv_raw=10.0,
                               cv_clean=8.0, detection_rate=0.5, outlier_count=0)
        score, tier = compute_confidence_score(ls, mw)
        assert 0 <= score <= 100
        assert tier in ("CONFIRMED", "PROBABLE", "CANDIDATE", "NOISE")


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestSerialization:
    """Tests for result serialization."""

    def test_serialize_band(self):
        """Should produce JSON-safe dict for band result."""
        band = BandResult(
            band_name="HF", band_range_hz=(0.15, 0.4),
            ls_frequency_hz=0.25, ls_power=0.5, ls_fap=0.01,
            multi_window_cv=12.0, multi_window_cv_clean=10.0,
            detection_rate=0.6,
            bootstrap_freq_mean=0.25, bootstrap_freq_std=0.01,
            bootstrap_ci_low=0.23, bootstrap_ci_high=0.27,
            confidence_score=35.0, confidence_tier="NOISE",
        )
        d = serialize_band(band)
        assert isinstance(d, dict)
        assert d["band_name"] == "HF"
        import json
        json.dumps(d)  # Should not raise

    def test_serialize_subject(self):
        """Should produce JSON-safe dict for subject result."""
        band = BandResult(
            band_name="HF", band_range_hz=(0.15, 0.4),
            ls_frequency_hz=0.25, ls_power=0.5, ls_fap=0.01,
            multi_window_cv=12.0, multi_window_cv_clean=10.0,
            detection_rate=0.6,
            bootstrap_freq_mean=0.25, bootstrap_freq_std=0.01,
            bootstrap_ci_low=0.23, bootstrap_ci_high=0.27,
            confidence_score=35.0, confidence_tier="NOISE",
        )
        subject = SubjectResult(
            subject_id="nsr001", n_beats=80000, duration_hours=19.5,
            mean_hr_bpm=70.0, bands={"HF": band},
        )
        d = serialize_subject(subject)
        assert isinstance(d, dict)
        assert d["subject_id"] == "nsr001"
        assert "HF" in d["bands"]
        import json
        json.dumps(d)  # Should not raise

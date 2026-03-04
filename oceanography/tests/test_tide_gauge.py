"""Tests for oceanography tide gauge experiment."""

from pathlib import Path

import numpy as np
import pytest

from oceanography.tide_gauge_experiment import (
    BANDPASS_ORDER,
    CV_EXCELLENT,
    CV_GOOD,
    DIURNAL_BAND_HZ,
    FULL_TIDAL_BAND_HZ,
    N_WINDOWS,
    SAMPLING_RATE_HZ,
    SEMIDIURNAL_BAND_HZ,
    TIDAL_CONSTITUENTS,
    WINDOW_DURATION_DAYS,
    BootstrapResult,
    ConstituentResult,
    LSResult,
    MultiWindowResult,
    bandpass_filter,
    compute_confidence_score,
    create_windows,
    extract_envelope,
    find_peak_near,
    prepare_time_series,
    run_bootstrap,
    run_lomb_scargle,
    run_multi_window_validation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _has_noaa_data() -> bool:
    """Check if cached NOAA data exists."""
    return any(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else False


def _synthetic_tidal_signal(
    freq_hz: float,
    duration_days: float = 365,
    noise_level: float = 0.01,
    sampling_interval_sec: float = 3600,
) -> tuple:
    """Generate synthetic tidal signal at hourly sampling.

    Returns (time_seconds, signal).
    """
    n_samples = int(duration_days * 86400 / sampling_interval_sec)
    time = np.arange(n_samples) * sampling_interval_sec
    signal = np.sin(2 * np.pi * freq_hz * time)
    if noise_level > 0:
        rng = np.random.default_rng(42)
        signal += rng.normal(0, noise_level, n_samples)
    return time, signal


def _multi_constituent_signal(
    duration_days: float = 365,
    noise_level: float = 0.05,
) -> tuple:
    """Generate signal with all 4 tidal constituents."""
    n_samples = int(duration_days * 24)
    time = np.arange(n_samples) * 3600.0
    signal = np.zeros(n_samples)

    # Amplitudes roughly matching San Francisco tidal range
    amplitudes = {"M2": 0.55, "S2": 0.13, "K1": 0.37, "O1": 0.22}
    for name, info in TIDAL_CONSTITUENTS.items():
        amp = amplitudes.get(name, 0.1)
        signal += amp * np.sin(2 * np.pi * info["hz"] * time)

    if noise_level > 0:
        rng = np.random.default_rng(42)
        signal += rng.normal(0, noise_level, n_samples)

    return time, signal


# ---------------------------------------------------------------------------
# Test tidal constituent constants
# ---------------------------------------------------------------------------

class TestTidalConstituents:

    def test_four_major_constituents(self):
        assert len(TIDAL_CONSTITUENTS) == 4
        for name in ("M2", "S2", "K1", "O1"):
            assert name in TIDAL_CONSTITUENTS

    def test_m2_frequency_values(self):
        """M2 should be ~1.9323 cpd."""
        m2 = TIDAL_CONSTITUENTS["M2"]
        assert abs(m2["cpd"] - 1.9323) < 0.001

    def test_s2_frequency_values(self):
        """S2 should be exactly 2.0 cpd (solar)."""
        s2 = TIDAL_CONSTITUENTS["S2"]
        assert s2["cpd"] == 2.0

    def test_hz_cpd_consistency(self):
        """Hz should equal cpd / 86400 for all constituents."""
        for name, info in TIDAL_CONSTITUENTS.items():
            expected_hz = info["cpd"] / 86400
            assert abs(info["hz"] - expected_hz) < 1e-15, f"{name}: hz mismatch"

    def test_diurnal_vs_semidiurnal(self):
        """K1 and O1 are diurnal (~1 cpd), M2 and S2 are semidiurnal (~2 cpd)."""
        assert TIDAL_CONSTITUENTS["K1"]["cpd"] < 1.1
        assert TIDAL_CONSTITUENTS["O1"]["cpd"] < 1.1
        assert TIDAL_CONSTITUENTS["M2"]["cpd"] > 1.5
        assert TIDAL_CONSTITUENTS["S2"]["cpd"] > 1.5


# ---------------------------------------------------------------------------
# Test bandpass filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:

    def test_output_shape(self):
        _, signal = _synthetic_tidal_signal(
            TIDAL_CONSTITUENTS["M2"]["hz"], duration_days=60
        )
        filtered = bandpass_filter(
            signal, SAMPLING_RATE_HZ, SEMIDIURNAL_BAND_HZ[0], SEMIDIURNAL_BAND_HZ[1]
        )
        assert filtered.shape == signal.shape

    def test_passes_m2_frequency(self):
        """M2 in the semidiurnal band should pass through."""
        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        time, signal = _synthetic_tidal_signal(m2_hz, duration_days=90, noise_level=0)
        filtered = bandpass_filter(
            signal, SAMPLING_RATE_HZ, SEMIDIURNAL_BAND_HZ[0], SEMIDIURNAL_BAND_HZ[1]
        )
        # Power should be mostly preserved
        assert np.std(filtered) > 0.3 * np.std(signal)

    def test_attenuates_out_of_band(self):
        """A frequency well below the tidal band should be attenuated."""
        # 0.1 cpd = ~1.16e-6 Hz (well below diurnal band)
        low_freq = 0.1 / 86400
        time, signal = _synthetic_tidal_signal(low_freq, duration_days=90, noise_level=0)
        filtered = bandpass_filter(
            signal, SAMPLING_RATE_HZ, DIURNAL_BAND_HZ[0], DIURNAL_BAND_HZ[1]
        )
        assert np.std(filtered) < 0.1 * np.std(signal)


# ---------------------------------------------------------------------------
# Test envelope extraction
# ---------------------------------------------------------------------------

class TestEnvelopeExtraction:

    def test_dc_removed(self):
        """Envelope should have approximately zero mean."""
        _, signal = _synthetic_tidal_signal(
            TIDAL_CONSTITUENTS["M2"]["hz"], duration_days=30
        )
        envelope = extract_envelope(signal)
        assert abs(np.mean(envelope)) < 1e-10

    def test_output_shape(self):
        _, signal = _synthetic_tidal_signal(
            TIDAL_CONSTITUENTS["M2"]["hz"], duration_days=30
        )
        envelope = extract_envelope(signal)
        assert envelope.shape == signal.shape


# ---------------------------------------------------------------------------
# Test Lomb-Scargle
# ---------------------------------------------------------------------------

class TestLombScargle:

    def test_detects_m2_frequency(self):
        """Should detect M2 within 1% error."""
        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        time, signal = _synthetic_tidal_signal(m2_hz, duration_days=365, noise_level=0.05)

        result = run_lomb_scargle(
            time, signal,
            freq_min_hz=SEMIDIURNAL_BAND_HZ[0],
            freq_max_hz=SEMIDIURNAL_BAND_HZ[1],
        )

        freq_error = abs(result.frequency_hz - m2_hz) / m2_hz
        assert freq_error < 0.01, f"Frequency error {freq_error:.4f} exceeds 1%"
        assert result.power > 0.1

    def test_detects_k1_frequency(self):
        """Should detect K1 within 1% error."""
        k1_hz = TIDAL_CONSTITUENTS["K1"]["hz"]
        time, signal = _synthetic_tidal_signal(k1_hz, duration_days=365, noise_level=0.05)

        result = run_lomb_scargle(
            time, signal,
            freq_min_hz=DIURNAL_BAND_HZ[0],
            freq_max_hz=DIURNAL_BAND_HZ[1],
        )

        freq_error = abs(result.frequency_hz - k1_hz) / k1_hz
        assert freq_error < 0.01

    def test_cpd_conversion(self):
        """frequency_cpd should equal frequency_hz * 86400."""
        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        time, signal = _synthetic_tidal_signal(m2_hz, duration_days=365, noise_level=0)

        result = run_lomb_scargle(
            time, signal,
            freq_min_hz=SEMIDIURNAL_BAND_HZ[0],
            freq_max_hz=SEMIDIURNAL_BAND_HZ[1],
        )

        assert abs(result.frequency_cpd - result.frequency_hz * 86400) < 1e-15

    def test_returns_ls_result(self):
        time, signal = _synthetic_tidal_signal(
            TIDAL_CONSTITUENTS["M2"]["hz"], duration_days=60
        )
        result = run_lomb_scargle(
            time, signal,
            freq_min_hz=SEMIDIURNAL_BAND_HZ[0],
            freq_max_hz=SEMIDIURNAL_BAND_HZ[1],
        )
        assert isinstance(result, LSResult)
        assert len(result.frequencies_hz) > 0
        assert len(result.powers) == len(result.frequencies_hz)


# ---------------------------------------------------------------------------
# Test find_peak_near
# ---------------------------------------------------------------------------

class TestFindPeakNear:

    def test_finds_peak_near_target(self):
        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        freqs = np.linspace(SEMIDIURNAL_BAND_HZ[0], SEMIDIURNAL_BAND_HZ[1], 10000)
        powers = np.zeros_like(freqs)
        idx = np.argmin(np.abs(freqs - m2_hz))
        powers[idx] = 1.0

        result = find_peak_near(freqs, powers, m2_hz)
        assert abs(result.frequency_hz - m2_hz) / m2_hz < 0.01

    def test_returns_low_power_when_no_peak(self):
        """If target is outside frequency range, should return 0 power."""
        freqs = np.linspace(1e-6, 5e-6, 1000)  # well below tidal
        powers = np.ones_like(freqs) * 0.1
        target = TIDAL_CONSTITUENTS["M2"]["hz"]  # ~2.24e-5, outside range

        result = find_peak_near(freqs, powers, target)
        assert result.power == 0.0


# ---------------------------------------------------------------------------
# Test multi-window validation
# ---------------------------------------------------------------------------

class TestMultiWindow:

    def test_creates_correct_number_of_windows(self):
        time, signal = _synthetic_tidal_signal(
            TIDAL_CONSTITUENTS["M2"]["hz"], duration_days=365
        )
        windows = create_windows(time, signal, n_windows=8)
        assert len(windows) == 8

    def test_windows_have_time_and_signal(self):
        time, signal = _synthetic_tidal_signal(
            TIDAL_CONSTITUENTS["M2"]["hz"], duration_days=365
        )
        windows = create_windows(time, signal, n_windows=6)
        for w in windows:
            assert "time" in w
            assert "signal" in w
            assert len(w["time"]) == len(w["signal"])
            assert len(w["time"]) > 100

    def test_short_signal_raises(self):
        """Signal shorter than window duration should raise."""
        time, signal = _synthetic_tidal_signal(
            TIDAL_CONSTITUENTS["M2"]["hz"], duration_days=10
        )
        with pytest.raises(ValueError, match="shorter than"):
            create_windows(time, signal, n_windows=4, window_duration_sec=30 * 86400)

    def test_clean_signal_has_low_cv(self):
        """Perfectly periodic signal should have very low CV."""
        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        time, signal = _synthetic_tidal_signal(m2_hz, duration_days=365, noise_level=0)

        windows = create_windows(time, signal)
        mw_result = run_multi_window_validation(
            windows, m2_hz,
            freq_min_hz=SEMIDIURNAL_BAND_HZ[0],
            freq_max_hz=SEMIDIURNAL_BAND_HZ[1],
        )

        assert mw_result.cv_clean < CV_GOOD, f"CV {mw_result.cv_clean:.4f}% exceeds {CV_GOOD}%"
        assert mw_result.detection_rate >= 0.8

    def test_noisy_signal_still_detects(self):
        """Moderately noisy tidal signal should still have reasonable CV."""
        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        time, signal = _synthetic_tidal_signal(m2_hz, duration_days=365, noise_level=0.3)

        windows = create_windows(time, signal)
        mw_result = run_multi_window_validation(
            windows, m2_hz,
            freq_min_hz=SEMIDIURNAL_BAND_HZ[0],
            freq_max_hz=SEMIDIURNAL_BAND_HZ[1],
        )

        assert mw_result.cv_clean < 5.0  # Should still be in reasonable range
        assert mw_result.detection_rate > 0.5


# ---------------------------------------------------------------------------
# Test bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:

    def test_returns_confidence_interval(self):
        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        time, signal = _synthetic_tidal_signal(m2_hz, duration_days=180, noise_level=0.05)

        result = run_bootstrap(
            time, signal, n_bootstrap=20,
            freq_min_hz=SEMIDIURNAL_BAND_HZ[0],
            freq_max_hz=SEMIDIURNAL_BAND_HZ[1],
        )

        assert isinstance(result, BootstrapResult)
        # For very clean signals, CI may collapse to a point (all bootstrap
        # samples detect the same frequency). Use approx for fp tolerance.
        assert result.freq_ci_low <= result.freq_mean or abs(result.freq_ci_low - result.freq_mean) < 1e-18
        assert result.freq_std >= 0

    def test_ci_near_true_frequency(self):
        """Bootstrap mean should be near the true frequency for clean signal."""
        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        time, signal = _synthetic_tidal_signal(m2_hz, duration_days=365, noise_level=0.01)

        result = run_bootstrap(
            time, signal, n_bootstrap=50,
            freq_min_hz=SEMIDIURNAL_BAND_HZ[0],
            freq_max_hz=SEMIDIURNAL_BAND_HZ[1],
        )

        # Mean should be within 1% of true frequency
        error = abs(result.freq_mean - m2_hz) / m2_hz
        assert error < 0.01, f"Bootstrap mean error {error:.4f} exceeds 1%"


# ---------------------------------------------------------------------------
# Test confidence scoring
# ---------------------------------------------------------------------------

class TestConfidenceScoring:

    def test_perfect_detection_high_score(self):
        """Perfect match + low CV + low FAP → score >= 90."""
        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        ls_result = LSResult(
            frequency_hz=m2_hz,
            frequency_cpd=m2_hz * 86400,
            power=0.95,
            fap=1e-20,
            frequencies_hz=np.array([]),
            powers=np.array([]),
        )
        mw_result = MultiWindowResult(
            detected_freqs=[m2_hz] * 8,
            cv_raw=0.001,
            cv_clean=0.001,
            detection_rate=1.0,
            outlier_count=0,
        )

        score, tier = compute_confidence_score(ls_result, mw_result, m2_hz)
        assert score >= 90
        assert tier == "CONFIRMED"

    def test_poor_detection_low_score(self):
        """Large error + high CV + high FAP → score < 60."""
        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        wrong_hz = m2_hz * 1.5  # 50% off
        ls_result = LSResult(
            frequency_hz=wrong_hz,
            frequency_cpd=wrong_hz * 86400,
            power=0.01,
            fap=0.5,
            frequencies_hz=np.array([]),
            powers=np.array([]),
        )
        mw_result = MultiWindowResult(
            detected_freqs=[wrong_hz] * 8,
            cv_raw=50.0,
            cv_clean=50.0,
            detection_rate=0.0,
            outlier_count=0,
        )

        score, tier = compute_confidence_score(ls_result, mw_result, m2_hz)
        assert score < 60
        assert tier in ("CANDIDATE", "NOISE")

    def test_scoring_breakdown_sum(self):
        """Score components should add up: 30 + 25 + 25 + 20 = 100 max."""
        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        ls_result = LSResult(
            frequency_hz=m2_hz,
            frequency_cpd=m2_hz * 86400,
            power=0.95,
            fap=1e-20,
            frequencies_hz=np.array([]),
            powers=np.array([]),
        )
        mw_result = MultiWindowResult(
            detected_freqs=[m2_hz] * 8,
            cv_raw=0.001,
            cv_clean=0.001,
            detection_rate=1.0,
            outlier_count=0,
        )
        score, _ = compute_confidence_score(ls_result, mw_result, m2_hz)
        assert score == 100.0


# ---------------------------------------------------------------------------
# Test prepare_time_series
# ---------------------------------------------------------------------------

class TestPrepareTimeSeries:

    def test_handles_nan_values(self):
        """Should drop NaN values from the time series."""
        import pandas as pd

        idx = pd.date_range("2023-01-01", periods=100, freq="h")
        values = np.random.default_rng(0).normal(0, 1, 100)
        values[10] = np.nan
        values[50] = np.nan
        df = pd.DataFrame({"v": values}, index=idx)

        time, signal = prepare_time_series(df, "v")
        assert len(time) == 98
        assert not np.any(np.isnan(signal))

    def test_time_starts_at_zero(self):
        import pandas as pd

        idx = pd.date_range("2023-06-15", periods=50, freq="h")
        df = pd.DataFrame({"v": np.ones(50)}, index=idx)

        time, _ = prepare_time_series(df, "v")
        assert time[0] == 0.0
        assert time[1] == pytest.approx(3600.0)


# ---------------------------------------------------------------------------
# Integration: multi-constituent detection
# ---------------------------------------------------------------------------

class TestMultiConstituentDetection:

    def test_detects_m2_in_mixed_signal(self):
        """M2 should be detectable in a signal with all 4 constituents."""
        time, signal = _multi_constituent_signal(duration_days=365, noise_level=0.02)

        # Bandpass to semidiurnal band
        filtered = bandpass_filter(
            signal, SAMPLING_RATE_HZ,
            SEMIDIURNAL_BAND_HZ[0], SEMIDIURNAL_BAND_HZ[1],
        )

        result = run_lomb_scargle(
            time, filtered,
            freq_min_hz=SEMIDIURNAL_BAND_HZ[0],
            freq_max_hz=SEMIDIURNAL_BAND_HZ[1],
        )

        m2_hz = TIDAL_CONSTITUENTS["M2"]["hz"]
        freq_error = abs(result.frequency_hz - m2_hz) / m2_hz
        assert freq_error < 0.01, f"M2 detection error {freq_error:.4f}"

    def test_detects_k1_in_mixed_signal(self):
        """K1 should be detectable in the diurnal band."""
        time, signal = _multi_constituent_signal(duration_days=365, noise_level=0.02)

        filtered = bandpass_filter(
            signal, SAMPLING_RATE_HZ,
            DIURNAL_BAND_HZ[0], DIURNAL_BAND_HZ[1],
        )

        result = run_lomb_scargle(
            time, filtered,
            freq_min_hz=DIURNAL_BAND_HZ[0],
            freq_max_hz=DIURNAL_BAND_HZ[1],
        )

        k1_hz = TIDAL_CONSTITUENTS["K1"]["hz"]
        freq_error = abs(result.frequency_hz - k1_hz) / k1_hz
        assert freq_error < 0.01, f"K1 detection error {freq_error:.4f}"


# ---------------------------------------------------------------------------
# Integration test (requires downloaded NOAA data)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_noaa_data(), reason="NOAA data not cached")
class TestWithRealData:

    def test_cached_data_loads(self):
        """Verify cached CSV data loads correctly."""
        import pandas as pd

        csv_files = list(DATA_DIR.glob("*hourly_height*.csv"))
        assert len(csv_files) > 0
        df = pd.read_csv(csv_files[0], index_col=0, parse_dates=True)
        assert "v" in df.columns
        assert len(df) > 1000

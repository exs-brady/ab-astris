"""
Tests for Ab Astris Negative Control: Cryptocurrency

Tests the crypto experiment pipeline using synthetic random walk data
and validates that the Ab Astris core functions work correctly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from crypto_experiment import (
    LSResult,
    MultiWindowResult,
    BootstrapResult,
    SignalResult,
    run_lomb_scargle,
    create_windows,
    run_multi_window_validation,
    run_bootstrap,
    compute_confidence_score,
    prepare_crypto_time_series,
    serialize_result,
    FREQ_MIN_HZ,
    FREQ_MAX_HZ,
)


# ============================================================================
# SYNTHETIC DATA GENERATORS
# ============================================================================

def _synthetic_random_walk(duration_days=365, seed=42):
    """Generate geometric random walk mimicking crypto prices."""
    rng = np.random.default_rng(seed)
    n = duration_days
    time = np.arange(n) * 86400.0  # daily in seconds
    returns = rng.normal(0, 0.02, n)  # 2% daily vol
    prices = 30000 * np.exp(np.cumsum(returns))
    return time, prices


def _synthetic_periodic_signal(freq_hz, duration_days=365, noise_level=0.01, seed=42):
    """Generate clean periodic signal (for baseline comparison)."""
    rng = np.random.default_rng(seed)
    n = duration_days
    time = np.arange(n) * 86400.0
    signal = np.sin(2 * np.pi * freq_hz * time)
    if noise_level > 0:
        signal += rng.normal(0, noise_level, n)
    return time, signal


# ============================================================================
# LOMB-SCARGLE TESTS
# ============================================================================

class TestLombScargle:
    """Tests for the Lomb-Scargle periodogram."""

    def test_detects_known_frequency(self):
        """LS should detect a clean sinusoid at the correct frequency."""
        target_freq = 0.05 / 86400  # 0.05 cpd = 20-day period
        time, signal = _synthetic_periodic_signal(target_freq)
        result = run_lomb_scargle(time, signal,
                                  freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ)
        freq_error = abs(result.frequency_hz - target_freq) / target_freq
        assert freq_error < 0.02, f"Freq error {freq_error:.4f} > 2%"

    def test_returns_lsresult(self):
        """LS should return an LSResult dataclass."""
        time, prices = _synthetic_random_walk()
        result = run_lomb_scargle(time, prices,
                                  freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ)
        assert isinstance(result, LSResult)
        assert result.frequency_hz > 0
        assert result.power > 0
        assert len(result.frequencies_hz) == 10000
        assert len(result.powers) == 10000

    def test_fap_is_float(self):
        """FAP should be a float between 0 and 1."""
        time, prices = _synthetic_random_walk()
        result = run_lomb_scargle(time, prices,
                                  freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ)
        assert isinstance(result.fap, float)
        assert 0 <= result.fap <= 1

    def test_frequency_cpd_conversion(self):
        """frequency_cpd should be frequency_hz * 86400."""
        time, prices = _synthetic_random_walk()
        result = run_lomb_scargle(time, prices,
                                  freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ)
        assert abs(result.frequency_cpd - result.frequency_hz * 86400) < 1e-15


# ============================================================================
# MULTI-WINDOW TESTS
# ============================================================================

class TestMultiWindow:
    """Tests for multi-window validation."""

    def test_creates_correct_number_of_windows(self):
        """Should create the requested number of windows."""
        time, prices = _synthetic_random_walk(duration_days=365)
        windows = create_windows(time, prices, n_windows=8,
                                 window_duration_sec=60 * 86400)
        assert len(windows) == 8

    def test_window_contains_time_and_signal(self):
        """Each window should have time and signal arrays."""
        time, prices = _synthetic_random_walk()
        windows = create_windows(time, prices, n_windows=4,
                                 window_duration_sec=60 * 86400)
        for w in windows:
            assert "time" in w
            assert "signal" in w
            assert len(w["time"]) == len(w["signal"])
            assert len(w["time"]) > 0

    def test_raises_for_short_signal(self):
        """Should raise if signal is shorter than window duration."""
        time, prices = _synthetic_random_walk(duration_days=10)
        with pytest.raises(ValueError, match="shorter than"):
            create_windows(time, prices, window_duration_sec=30 * 86400)

    def test_random_walk_high_cv(self):
        """Random walk should produce high CV (>>1%)."""
        time, prices = _synthetic_random_walk()
        ls_result = run_lomb_scargle(time, prices,
                                     freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ)
        windows = create_windows(time, prices, n_windows=8,
                                 window_duration_sec=60 * 86400)
        mw_result = run_multi_window_validation(
            windows, ls_result.frequency_hz,
            freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ,
        )
        assert isinstance(mw_result, MultiWindowResult)
        # Random walk should have very unstable frequencies
        assert mw_result.cv_clean > 1.0, f"CV {mw_result.cv_clean} too low for random walk"

    def test_periodic_signal_low_cv(self):
        """Clean periodic signal should have low CV."""
        freq = 0.05 / 86400  # 20-day period
        time, signal = _synthetic_periodic_signal(freq, noise_level=0.01)
        windows = create_windows(time, signal, n_windows=8,
                                 window_duration_sec=60 * 86400)
        mw_result = run_multi_window_validation(
            windows, freq,
            freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ,
        )
        assert mw_result.cv_clean < 5.0, f"CV {mw_result.cv_clean} too high for periodic signal"


# ============================================================================
# BOOTSTRAP TESTS
# ============================================================================

class TestBootstrap:
    """Tests for bootstrap error estimation."""

    def test_returns_bootstrap_result(self):
        """Should return BootstrapResult dataclass."""
        time, prices = _synthetic_random_walk(duration_days=100)
        result = run_bootstrap(time, prices, n_bootstrap=10,
                               freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ)
        assert isinstance(result, BootstrapResult)
        assert result.freq_mean > 0
        assert result.freq_std >= 0
        assert result.freq_ci_low <= result.freq_ci_high

    def test_bootstrap_produces_narrow_ci_for_periodic(self):
        """Bootstrap of periodic signal should produce narrow CI relative to random walk."""
        freq = 0.05 / 86400  # 20-day period
        time_p, signal_p = _synthetic_periodic_signal(freq, duration_days=365)
        result_periodic = run_bootstrap(time_p, signal_p, n_bootstrap=20,
                                        freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ)
        time_r, prices_r = _synthetic_random_walk(duration_days=365)
        result_random = run_bootstrap(time_r, prices_r, n_bootstrap=20,
                                      freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ)
        # Periodic signal should have much smaller CI width
        ci_periodic = result_periodic.freq_ci_high - result_periodic.freq_ci_low
        ci_random = result_random.freq_ci_high - result_random.freq_ci_low
        assert ci_periodic <= ci_random or ci_periodic < 1e-7  # periodic is narrower or extremely narrow


# ============================================================================
# CONFIDENCE SCORING TESTS
# ============================================================================

class TestConfidenceScoring:
    """Tests for confidence score computation."""

    def test_noise_tier_for_random_walk(self):
        """Random walk should score NOISE tier."""
        time, prices = _synthetic_random_walk()
        ls_result = run_lomb_scargle(time, prices,
                                     freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ)
        windows = create_windows(time, prices, n_windows=8,
                                 window_duration_sec=60 * 86400)
        mw_result = run_multi_window_validation(
            windows, ls_result.frequency_hz,
            freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ,
        )
        score, tier = compute_confidence_score(ls_result, mw_result)
        assert tier == "NOISE", f"Expected NOISE, got {tier} (score={score})"

    def test_exploratory_no_target(self):
        """With no target freq, accuracy component should be 0."""
        time, prices = _synthetic_random_walk()
        ls_result = run_lomb_scargle(time, prices,
                                     freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ)
        windows = create_windows(time, prices, n_windows=8,
                                 window_duration_sec=60 * 86400)
        mw_result = run_multi_window_validation(
            windows, ls_result.frequency_hz,
            freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ,
        )
        score_no_target, _ = compute_confidence_score(ls_result, mw_result, target_freq_hz=None)
        score_with_target, _ = compute_confidence_score(ls_result, mw_result,
                                                         target_freq_hz=ls_result.frequency_hz)
        # With target = detected freq, should get perfect accuracy (30 pts)
        assert score_with_target >= score_no_target

    def test_score_range(self):
        """Score should be 0-100."""
        time, prices = _synthetic_random_walk()
        ls_result = run_lomb_scargle(time, prices,
                                     freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ)
        windows = create_windows(time, prices, n_windows=8,
                                 window_duration_sec=60 * 86400)
        mw_result = run_multi_window_validation(
            windows, ls_result.frequency_hz,
            freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ,
        )
        score, _ = compute_confidence_score(ls_result, mw_result)
        assert 0 <= score <= 100


# ============================================================================
# DATA PREPARATION TESTS
# ============================================================================

class TestDataPreparation:
    """Tests for data preparation functions."""

    def test_prepare_crypto_time_series(self):
        """Should convert DataFrame to (time, prices) arrays."""
        import pandas as pd
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        rng = np.random.default_rng(42)
        prices = 30000 + rng.normal(0, 1000, 365).cumsum()
        prices = np.abs(prices) + 100  # ensure positive
        df = pd.DataFrame({"Close": prices}, index=dates)

        time, values = prepare_crypto_time_series(df)
        assert len(time) == 365
        assert len(values) == 365
        assert time[0] == 0.0
        assert (values > 0).all()
        assert time[-1] > 0

    def test_prepare_handles_nans(self):
        """Should drop NaN values."""
        import pandas as pd
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        prices = np.array([100, 101, np.nan, 103, 104, np.nan, 106, 107, 108, 109])
        df = pd.DataFrame({"Close": prices}, index=dates)

        time, values = prepare_crypto_time_series(df)
        assert len(values) == 8  # 2 NaNs dropped
        assert not np.any(np.isnan(values))


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestSerialization:
    """Tests for result serialization."""

    def test_serialize_result(self):
        """Should produce JSON-safe dict."""
        result = SignalResult(
            asset_name="BTC", symbol="BTC-USD",
            ls_frequency_hz=1e-6, ls_frequency_cpd=0.0864,
            ls_period_days=11.57, ls_power=0.5, ls_fap=0.01,
            multi_window_cv=45.0, multi_window_cv_clean=42.0,
            detection_rate=0.25,
            bootstrap_freq_mean=1e-6, bootstrap_freq_std=5e-7,
            bootstrap_ci_low=5e-7, bootstrap_ci_high=2e-6,
            confidence_score=15.0, confidence_tier="NOISE",
        )
        d = serialize_result(result)
        assert isinstance(d, dict)
        assert d["asset_name"] == "BTC"
        assert d["confidence_tier"] == "NOISE"
        # All values should be JSON-serializable
        import json
        json.dumps(d)  # Should not raise

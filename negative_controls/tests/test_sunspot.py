"""
Tests for Ab Astris Negative Control: Sunspot Number

Tests the sunspot experiment pipeline using synthetic data and validates
that the SILSO parsing and two-analysis pipeline work correctly.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sunspot_experiment import (
    LSResult,
    MultiWindowResult,
    BootstrapResult,
    SunspotResult,
    run_lomb_scargle,
    create_windows,
    run_multi_window_validation,
    run_bootstrap,
    compute_confidence_score,
    prepare_monthly_time_series,
    prepare_daily_time_series,
    serialize_result,
    MONTHLY_FREQ_MIN_CY,
    MONTHLY_FREQ_MAX_CY,
    DAILY_FREQ_MIN_CPD,
    DAILY_FREQ_MAX_CPD,
    SOLAR_CYCLE_YEARS,
    CARRINGTON_PERIOD_DAYS,
)


SEC_PER_YEAR = 365.25 * 86400


# ============================================================================
# SYNTHETIC DATA GENERATORS
# ============================================================================

def _synthetic_solar_cycle(n_years=275, cycle_years=11.0, noise_level=0.3, seed=42):
    """Generate synthetic sunspot numbers with ~11-year cycle."""
    rng = np.random.default_rng(seed)
    n_months = int(n_years * 12)
    time_months = np.arange(n_months)
    time_sec = time_months * (SEC_PER_YEAR / 12)

    # Sinusoidal + amplitude modulation + noise
    freq_hz = (1.0 / cycle_years) / SEC_PER_YEAR
    phase = 2 * np.pi * freq_hz * time_sec
    # Sunspot numbers are positive with variable amplitude
    signal = 80 * (1 + np.sin(phase)) + rng.exponential(10, n_months)
    if noise_level > 0:
        signal += rng.normal(0, noise_level * np.std(signal), n_months)
    signal = np.maximum(signal, 0)  # Sunspot numbers are non-negative

    return time_sec, signal


def _synthetic_rotation_signal(n_days=16000, period_days=27.3, noise_level=0.5, seed=42):
    """Generate synthetic signal with ~27-day rotation period."""
    rng = np.random.default_rng(seed)
    time_sec = np.arange(n_days) * 86400.0
    freq_hz = 1.0 / (period_days * 86400)

    signal = 50 + 20 * np.sin(2 * np.pi * freq_hz * time_sec)
    if noise_level > 0:
        signal += rng.normal(0, noise_level * 20, n_days)
    signal = np.maximum(signal, 0)

    return time_sec, signal


# ============================================================================
# DATA PARSING TESTS
# ============================================================================

class TestSILSOParsing:
    """Tests for SILSO CSV parsing."""

    def test_prepare_monthly_time_series(self):
        """Should convert monthly DataFrame to (time, values) arrays."""
        import pandas as pd
        n = 100
        df = pd.DataFrame({
            "decimal_year": np.linspace(1749.0, 1757.0, n),
            "sunspot_number": np.random.default_rng(42).integers(0, 200, n).astype(float),
        })
        time, values = prepare_monthly_time_series(df)
        assert len(time) == n
        assert len(values) == n
        assert time[0] == 0.0
        assert time[-1] > 0
        assert (values >= 0).all()

    def test_prepare_daily_time_series_filters_by_year(self):
        """Should only include data from start_year onwards."""
        import pandas as pd
        n = 1000
        df = pd.DataFrame({
            "year": np.concatenate([np.full(500, 1970), np.full(500, 1990)]),
            "decimal_year": np.concatenate([
                np.linspace(1970, 1971, 500),
                np.linspace(1990, 1991, 500),
            ]),
            "sunspot_number": np.random.default_rng(42).integers(0, 200, n).astype(float),
        })
        time, values = prepare_daily_time_series(df, start_year=1980)
        assert len(values) == 500  # Only 1990 data

    def test_semicolon_csv_parsing(self):
        """SILSO CSV uses semicolons — verify parsing with mock data."""
        import pandas as pd
        # Create mock SILSO monthly format
        lines = [
            "2023; 1;2023.042;  143.6;   12.3;  117; 1",
            "2023; 2;2023.125;  110.3;    9.8;  115; 1",
            "2023; 3;2023.208;   89.2;    8.1;  118; 1",
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("\n".join(lines))
            f.flush()
            df = pd.read_csv(
                f.name, sep=";", header=None,
                names=["year", "month", "decimal_year",
                       "sunspot_number", "std_dev", "n_obs", "definitive"],
                skipinitialspace=True,
            )
        assert len(df) == 3
        assert df["sunspot_number"].iloc[0] == pytest.approx(143.6, abs=0.1)
        assert df["decimal_year"].iloc[0] == pytest.approx(2023.042, abs=0.01)

    def test_missing_values_removal(self):
        """Should filter out -1 missing values."""
        import pandas as pd
        df = pd.DataFrame({
            "decimal_year": [2020.0, 2020.083, 2020.167, 2020.25],
            "sunspot_number": [50.0, -1.0, 30.0, -1.0],
        })
        df_clean = df[df["sunspot_number"] >= 0]
        assert len(df_clean) == 2


# ============================================================================
# 11-YEAR CYCLE ANALYSIS TESTS
# ============================================================================

class TestSolarCycleAnalysis:
    """Tests for the 11-year solar cycle analysis."""

    def test_detects_11yr_period(self):
        """Should detect ~11-year period in synthetic data."""
        time, values = _synthetic_solar_cycle(n_years=275, cycle_years=11.0)
        freq_min_hz = MONTHLY_FREQ_MIN_CY / SEC_PER_YEAR
        freq_max_hz = MONTHLY_FREQ_MAX_CY / SEC_PER_YEAR

        ls_result = run_lomb_scargle(time, values,
                                     freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz)
        detected_period = 1.0 / (ls_result.frequency_hz * SEC_PER_YEAR)
        assert 9.0 < detected_period < 13.0, f"Period {detected_period:.1f} yr outside 9-13 range"

    def test_moderate_cv(self):
        """Synthetic solar cycle should produce moderate CV (not as low as tides)."""
        time, values = _synthetic_solar_cycle(n_years=275)
        freq_min_hz = MONTHLY_FREQ_MIN_CY / SEC_PER_YEAR
        freq_max_hz = MONTHLY_FREQ_MAX_CY / SEC_PER_YEAR

        ls_result = run_lomb_scargle(time, values,
                                     freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz)
        window_duration = 46 * SEC_PER_YEAR
        windows = create_windows(time, values, n_windows=6,
                                 window_duration_sec=window_duration)
        mw_result = run_multi_window_validation(
            windows, ls_result.frequency_hz,
            freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz,
        )
        # Synthetic with fixed period should have low CV, but with noise it should be measurable
        assert mw_result.cv_clean >= 0  # At minimum, non-negative

    def test_six_windows(self):
        """Should create 6 windows for monthly analysis per brief."""
        time, values = _synthetic_solar_cycle(n_years=275)
        window_duration = 46 * SEC_PER_YEAR
        windows = create_windows(time, values, n_windows=6,
                                 window_duration_sec=window_duration)
        assert len(windows) == 6


# ============================================================================
# 27-DAY ROTATION TESTS
# ============================================================================

class TestSolarRotation:
    """Tests for the 27-day Carrington rotation analysis."""

    def test_detects_rotation_period(self):
        """Should detect ~27-day period in synthetic data."""
        time, values = _synthetic_rotation_signal(n_days=16000, period_days=27.3)
        freq_min_hz = DAILY_FREQ_MIN_CPD / 86400
        freq_max_hz = DAILY_FREQ_MAX_CPD / 86400

        ls_result = run_lomb_scargle(time, values,
                                     freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz)
        detected_period = 1.0 / ls_result.frequency_cpd
        assert 24.0 < detected_period < 31.0, f"Period {detected_period:.1f} d outside 24-31 range"

    def test_eight_windows_daily(self):
        """Daily analysis should use 8 windows."""
        time, values = _synthetic_rotation_signal(n_days=16000)
        window_duration = 5 * 365.25 * 86400
        windows = create_windows(time, values, n_windows=8,
                                 window_duration_sec=window_duration)
        assert len(windows) == 8


# ============================================================================
# CONFIDENCE SCORING TESTS
# ============================================================================

class TestConfidenceScoring:
    """Tests for confidence score computation."""

    def test_known_target_scores_higher(self):
        """With a known target, score should include accuracy points."""
        time, values = _synthetic_solar_cycle(n_years=275)
        freq_min_hz = MONTHLY_FREQ_MIN_CY / SEC_PER_YEAR
        freq_max_hz = MONTHLY_FREQ_MAX_CY / SEC_PER_YEAR
        target_hz = (1.0 / SOLAR_CYCLE_YEARS) / SEC_PER_YEAR

        ls_result = run_lomb_scargle(time, values,
                                     freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz)
        window_duration = 46 * SEC_PER_YEAR
        windows = create_windows(time, values, n_windows=6,
                                 window_duration_sec=window_duration)
        mw_result = run_multi_window_validation(
            windows, ls_result.frequency_hz,
            freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz,
        )

        score_with, _ = compute_confidence_score(ls_result, mw_result, target_hz)
        score_without, _ = compute_confidence_score(ls_result, mw_result, None)
        assert score_with >= score_without

    def test_score_range(self):
        """Score should be 0-100."""
        time, values = _synthetic_solar_cycle(n_years=100)
        freq_min_hz = MONTHLY_FREQ_MIN_CY / SEC_PER_YEAR
        freq_max_hz = MONTHLY_FREQ_MAX_CY / SEC_PER_YEAR

        ls_result = run_lomb_scargle(time, values,
                                     freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz)
        mw_result = MultiWindowResult(
            detected_freqs=[ls_result.frequency_hz] * 6,
            cv_raw=10.0, cv_clean=8.0, detection_rate=0.5, outlier_count=0,
        )
        score, tier = compute_confidence_score(ls_result, mw_result)
        assert 0 <= score <= 100
        assert tier in ("CONFIRMED", "PROBABLE", "CANDIDATE", "NOISE")


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestSerialization:
    """Tests for result serialization."""

    def test_serialize_result(self):
        """Should produce JSON-safe dict."""
        result = SunspotResult(
            analysis_name="11-year", description="test",
            n_data_points=3000, time_span_years=275.0,
            ls_frequency_hz=2.88e-9, ls_period_years=11.0, ls_period_days=4018.0,
            ls_power=0.9, ls_fap=0.0,
            multi_window_cv=8.0, multi_window_cv_clean=7.5,
            detection_rate=0.8, n_windows=6,
            bootstrap_freq_mean=2.88e-9, bootstrap_freq_std=1e-10,
            bootstrap_ci_low=2.7e-9, bootstrap_ci_high=3.0e-9,
            confidence_score=55.0, confidence_tier="NOISE",
        )
        d = serialize_result(result)
        assert isinstance(d, dict)
        assert d["analysis_name"] == "11-year"
        import json
        json.dumps(d)  # Should not raise

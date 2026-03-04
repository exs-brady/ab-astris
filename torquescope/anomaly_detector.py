"""
TorqueScope Phase 2: Anomaly Detector

Multi-signal anomaly detection combining:
1. Periodic structure degradation (LS-based) - THE CORE AB ASTRIS METHODOLOGY
2. Temperature delta analysis (component - ambient)
3. Trend detection
4. Power curve residual analysis

Updated for v3 brief Fix 2: Implements LS periodic baseline as false-positive filter.
Updated for v4 brief: NBM-Residual approach - LS on NBM residuals instead of raw signals.
Updated for v5 brief: Hybrid approach - v3 heuristic primary, v4 NBM gating.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ruptures as rpt  # For changepoint detection (v8 Session 3)

# NBM imports for residual-based detection (v6: updated predict_nbm with return_stds)
from torquescope_phase2.nbm import (
    NBMModel, PowerCurveModel,
    build_nbm, predict_nbm, compute_residuals,
    build_power_curve, power_curve_residual,
    get_operational_columns
)  # Note: predict_nbm now supports return_stds=True for v6

# Correlation monitoring imports for v7 Phase 3
from torquescope_phase2.correlation_monitor import (
    CorrelationBaseline, DecorrelationResult,
    build_correlation_baseline, compute_decorrelation_score
)


# =============================================================================
# PER-FARM TUNING PARAMETERS (v5 Per-Farm Implementation)
# =============================================================================
# Experimental per-farm tuning was attempted but did not improve on baseline.
# These parameters are preserved for reference but should use baseline values.
# See torquescope_phase2_summary.md for full experiment history.

FARM_THRESHOLDS = {
    'Wind Farm A': 0.45,   # Baseline
    'Wind Farm B': 0.45,   # Baseline (per-farm tuning did not improve)
    'Wind Farm C': 0.45,   # Baseline (per-farm tuning did not improve)
}

FARM_SUPPRESSION = {
    'Wind Farm A': 0.5,    # Baseline
    'Wind Farm B': 0.5,    # Baseline
    'Wind Farm C': 0.5,    # Baseline
}

FARM_NBM_DISABLED = {
    'Wind Farm A': False,
    'Wind Farm B': False,
    'Wind Farm C': False,
}

def get_farm_key(farm: str) -> str:
    """Normalize farm identifier to full form for dict lookup."""
    if farm.startswith('Wind Farm'):
        return farm
    # Single letter form
    return f'Wind Farm {farm}'


@dataclass
class PeriodicBaseline:
    """
    Periodic baseline computed from training data using Lomb-Scargle.
    This is the core Ab Astris methodology - multi-window CV validation.
    """
    signal_name: str
    dominant_period_hours: float  # Primary periodic component (typically 24h diurnal)
    dominant_freq: float          # cycles per hour
    dominant_amplitude: float     # LS power at dominant frequency
    dominant_period_cv: float     # Multi-window CV (percent) - THE KEY METRIC
    residual_fraction: float      # Fraction of variance not explained by periodicity
    n_windows: int = 4
    window_freqs: List[float] = field(default_factory=list)  # Per-window dominant freqs
    is_stable: bool = False       # CV < 5% indicates stable periodic structure


@dataclass
class ResidualBaseline:
    """
    Baseline computed from NBM residuals during training.

    This is the v4 approach: LS on residuals (after NBM removes operational variation)
    instead of raw signals. The residual should have stable periodic structure
    (diurnal thermal lags, cooling cycles) that degrades when component health changes.

    Per Brief v4:
    - Residual mean should be ~0 during training
    - Residual std is the baseline noise level
    - LS power spectrum captures periodic structure
    - Multi-window CV < 20% indicates stable periodicity (usable)
    - Multi-window CV 20-40% is acceptable (marginal)
    - Multi-window CV > 40% means no stable periodicity (use other metrics only)
    """
    signal_name: str
    nbm: NBMModel                   # The trained NBM for this signal
    baseline_mean: float            # Mean residual during training (~0)
    baseline_std: float             # Std of residual during training
    baseline_variance: float        # Variance of residual
    ls_powers: np.ndarray           # LS power spectrum of training residual
    ls_freqs: np.ndarray            # Frequency grid (cycles per hour)
    dominant_freq: float            # Dominant frequency in training residual
    dominant_power: float           # LS power at dominant frequency
    dominant_period_hours: float    # 1/dominant_freq
    multi_window_cv: float          # CV of dominant frequency across windows (%)
    total_spectral_power: float     # Sum of LS powers (for energy comparison)
    is_usable: bool = False         # True if CV < 40% (can use LS scoring)
    is_excellent: bool = False      # True if CV < 20% (strong LS signal)


@dataclass
class PCABaseline:
    """
    PCA baseline for multivariate anomaly detection (v8 Session 2).

    PCA captures joint sensor correlations during training. When a component
    degrades, it breaks the correlation structure, causing higher reconstruction
    error even if individual sensors haven't exceeded thresholds.

    Per Brief v8 Session 2:
    - Fit PCA on standardized temperature sensors during training
    - Keep top k components explaining 95% of variance
    - Reconstruction error = ||x - x_reconstructed||
    - Z-score the reconstruction error against training baseline
    """
    pca_model: object               # sklearn PCA model
    scaler: object                  # sklearn StandardScaler
    n_components: int               # Number of components kept
    explained_variance_ratio: float # Total variance explained
    sensor_names: List[str]         # Sensor names in order
    baseline_mean_error: float      # Mean reconstruction error during training
    baseline_std_error: float       # Std of reconstruction error during training
    n_training_samples: int         # Number of valid training samples used


@dataclass
class ChangepointResult:
    """
    Changepoint detection result for a sensor (v8 Session 3).

    Changepoints indicate sudden level shifts in sensor behavior.
    When a fault begins, temperature or other sensors often show
    a distinct level shift that rolling z-scores may miss if
    the shift happens within the rolling window.
    """
    sensor_name: str
    changepoints: List[int]         # Indices of detected changepoints
    n_changepoints: int             # Total number of changepoints detected
    changepoint_magnitudes: List[float]  # Magnitude of each level shift
    baseline_shift_std: float       # Typical shift magnitude during training


@dataclass
class SignalAnomaly:
    """Anomaly detection result for a single signal."""
    signal_name: str
    anomaly_score: float  # 0-1
    trend_score: float    # Positive = upward trend
    volatility_score: float  # Ratio to baseline
    periodic_score: float  # Periodic degradation (NEW: from LS baseline)
    periodic_degraded: bool = False  # True if periodic structure is disrupted


@dataclass
class WindowResult:
    """Result for a single detection window."""
    window_start: int
    window_end: int
    overall_score: float
    n_signals_anomalous: int
    n_periodic_degraded: int  # NEW: count of signals with degraded periodicity
    signal_results: List[SignalAnomaly]
    periodic_baseline_used: bool = False  # NEW: track if LS baseline was applied


def compute_rolling_z_scores(
    residuals: np.ndarray,
    test_start_idx: int,
    rolling_window: int = 30 * 144,  # 30 days at 10-min resolution
    min_samples: int = 500,
    fallback_std: float = 1.0
) -> np.ndarray:
    """
    Compute z-scores relative to a rolling baseline instead of the full training period.

    This fixes seasonal drift (Farm A inversion, Farm B elevation) by adapting the
    baseline to recent conditions rather than using a static training-period baseline.

    Per v6.2 brief, for each timestamp t in the test period:
    1. Look back `rolling_window` samples
    2. Compute rolling_mean and rolling_std of residuals in that window
    3. z_score[t] = (residual[t] - rolling_mean) / rolling_std

    For the first `rolling_window` samples of the test period, the lookback
    includes training data, which provides a warm start.

    Uses O(N) cumulative sum implementation for efficiency.

    Args:
        residuals: Array of residuals (actual - predicted) or raw z-scores
        test_start_idx: Index where test/prediction period starts
        rolling_window: Number of samples to look back (default 30 days)
        min_samples: Minimum valid samples required in window
        fallback_std: Std to use if rolling_std is near zero

    Returns:
        Array of rolling z-scores (same length as residuals)
    """
    n = len(residuals)
    z_scores = np.full(n, np.nan)

    # Handle NaN values: create a mask and use cumsum on valid values only
    valid_mask = ~np.isnan(residuals)

    # For efficiency, compute cumulative sums
    # Replace NaN with 0 for cumsum, then track count separately
    residuals_clean = np.where(valid_mask, residuals, 0.0)
    residuals_sq = residuals_clean ** 2

    # Cumulative sums
    cumsum = np.cumsum(residuals_clean)
    cumsum_sq = np.cumsum(residuals_sq)
    cumcount = np.cumsum(valid_mask.astype(float))

    # Prepend zero for easier window calculation
    cumsum = np.concatenate([[0], cumsum])
    cumsum_sq = np.concatenate([[0], cumsum_sq])
    cumcount = np.concatenate([[0], cumcount])

    for t in range(test_start_idx, n):
        # Look back rolling_window samples
        window_start = max(0, t - rolling_window)

        # Window statistics from cumulative sums
        window_count = cumcount[t] - cumcount[window_start]

        if window_count < min_samples:
            continue

        window_sum = cumsum[t] - cumsum[window_start]
        window_sum_sq = cumsum_sq[t] - cumsum_sq[window_start]

        rolling_mean = window_sum / window_count
        rolling_var = (window_sum_sq / window_count) - (rolling_mean ** 2)

        # Numerical stability: variance can be slightly negative due to floating point
        rolling_std = np.sqrt(max(0, rolling_var))

        if rolling_std < 0.01:
            rolling_std = fallback_std

        if valid_mask[t]:
            z_scores[t] = (residuals[t] - rolling_mean) / rolling_std

    return z_scores


def compute_changepoint_scores(
    signal: np.ndarray,
    test_start_idx: int,
    samples_per_hour: int = 6,
    lookback_hours: int = 48,      # How far back to look for changepoints (reduced from 72)
    min_segment_hours: int = 24,   # Minimum segment size (increased from 12)
    penalty_factor: float = 10.0,  # PELT penalty multiplier (increased from 3.0)
    min_shift_std: float = 2.0     # Minimum shift in std units to count as significant
) -> Tuple[np.ndarray, List[int]]:
    """
    Detect changepoints (sudden level shifts) in a signal and compute scores.

    This is v8 Session 3: Changepoint Detection. The idea is that fault onsets
    often cause sudden UPWARD level shifts that rolling z-scores may miss because
    the rolling baseline adapts to the new level.

    Uses PELT (Pruned Exact Linear Time) algorithm from ruptures library.

    IMPORTANT: Only UPWARD shifts (positive residual increase) are scored.
    This avoids flagging normal operational changes (cooldowns, shutdowns).

    Args:
        signal: Array of sensor values (can have NaNs) - typically NBM residuals
        test_start_idx: Index where test/prediction period starts
        samples_per_hour: Number of samples per hour (default 6 = 10-min intervals)
        lookback_hours: How far back to look for recent changepoints
        min_segment_hours: Minimum segment size to avoid detecting noise
        penalty_factor: PELT penalty multiplier (higher = fewer changepoints)
        min_shift_std: Minimum shift magnitude (in std units) to consider significant

    Returns:
        Tuple of:
        - changepoint_scores: Array of scores indicating proximity to UPWARD changepoints
        - all_changepoints: List of detected changepoint indices (upward only)
    """
    n = len(signal)
    changepoint_scores = np.zeros(n)
    all_changepoints = []

    # Convert time parameters to samples
    lookback_samples = lookback_hours * samples_per_hour
    min_segment_samples = max(min_segment_hours * samples_per_hour, 72)  # At least 12 hours

    # Handle NaN values: interpolate for changepoint detection
    valid_mask = ~np.isnan(signal)
    if np.sum(valid_mask) < 2 * min_segment_samples:
        return changepoint_scores, all_changepoints

    # Create clean signal with interpolated NaNs
    signal_clean = signal.copy()
    nan_indices = np.where(~valid_mask)[0]
    valid_indices = np.where(valid_mask)[0]

    if len(nan_indices) > 0 and len(valid_indices) > 1:
        # Linear interpolation for NaN values
        signal_clean[nan_indices] = np.interp(
            nan_indices,
            valid_indices,
            signal[valid_indices]
        )

    # Compute robust penalty based on signal variance
    # Use MAD (Median Absolute Deviation) for robustness
    signal_diff = np.diff(signal_clean)
    mad = np.median(np.abs(signal_diff - np.median(signal_diff)))
    robust_std = 1.4826 * mad  # Convert MAD to std estimate

    if robust_std < 0.01:
        robust_std = np.std(signal_clean)

    # PELT penalty: based on BIC, scaled by robust_std and penalty_factor
    # Higher penalty = fewer changepoints (more conservative)
    penalty = penalty_factor * robust_std * np.log(n)

    # Run PELT on the full signal for efficiency
    try:
        # Use L2 cost (mean shift detection)
        algo = rpt.Pelt(model="l2", min_size=min_segment_samples, jump=samples_per_hour)
        algo.fit(signal_clean.reshape(-1, 1))
        changepoints = algo.predict(pen=penalty)

        # Remove last element (which is always n in ruptures output)
        if changepoints and changepoints[-1] == n:
            changepoints = changepoints[:-1]

    except Exception:
        # Fall back to sliding window approach if PELT fails
        return changepoint_scores, all_changepoints

    # Filter changepoints: only keep UPWARD shifts with significant magnitude
    upward_changepoints = []
    for cp in changepoints:
        if cp >= min_segment_samples and cp < n - min_segment_samples:
            before = signal_clean[cp - min_segment_samples:cp]
            after = signal_clean[cp:cp + min_segment_samples]

            mean_before = np.mean(before)
            mean_after = np.mean(after)
            shift = mean_after - mean_before  # Signed shift

            # Only keep UPWARD shifts (positive = temperature/residual increase)
            # AND with significant magnitude
            normalized_shift = shift / robust_std if robust_std > 0 else 0
            if normalized_shift >= min_shift_std:
                upward_changepoints.append(cp)

    all_changepoints = upward_changepoints

    # Compute changepoint-based scores
    # For each point in test period, score based on proximity to recent UPWARD changepoint
    for t in range(test_start_idx, n):
        # Find changepoints in the lookback window
        recent_cps = [cp for cp in all_changepoints
                      if t - lookback_samples <= cp <= t]

        if not recent_cps:
            continue

        # Score based on:
        # 1. Recency of changepoint (more recent = higher score)
        # 2. Magnitude of level shift at changepoint
        max_score = 0.0

        for cp in recent_cps:
            # Compute level shift magnitude at changepoint
            if cp >= min_segment_samples and cp < n - min_segment_samples:
                before = signal_clean[cp - min_segment_samples:cp]
                after = signal_clean[cp:cp + min_segment_samples]

                mean_before = np.mean(before)
                mean_after = np.mean(after)
                shift = mean_after - mean_before  # Already filtered to be positive

                # Normalize shift by robust std
                normalized_shift = shift / robust_std if robust_std > 0 else 0

                # Recency factor: exponential decay from changepoint
                # Half-life of 12 hours (more aggressive decay)
                distance = t - cp
                recency = np.exp(-distance / (12 * samples_per_hour))

                # Combined score (weighted)
                score = normalized_shift * (0.3 + 0.7 * recency)
                max_score = max(max_score, score)

        changepoint_scores[t] = max_score

    return changepoint_scores, all_changepoints


class AnomalyDetector:
    """
    Multi-method anomaly detector for SCADA data.

    Combines multiple detection strategies:
    1. Temperature rise detection (delta from ambient)
    2. Volatility increase detection
    3. Trend detection (slow degradation)
    4. Periodic structure degradation
    """

    def __init__(
        self,
        window_days: int = 7,
        step_days: int = 1,
        samples_per_hour: int = 6,
        temp_rise_threshold: float = 5.0,   # Degrees above normal
        volatility_threshold: float = 2.0,   # X times baseline std
        trend_threshold: float = 0.1,        # Degrees per day
        heuristic_floor: float = 0.10,       # v5.2: configurable heuristic-only score floor
    ):
        self.window_days = window_days
        self.step_days = step_days
        self.samples_per_hour = samples_per_hour
        self.temp_rise_threshold = temp_rise_threshold
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.heuristic_floor = heuristic_floor

        self.window_size = window_days * 24 * samples_per_hour
        self.step_size = step_days * 24 * samples_per_hour

        # LS periodic baseline settings (Fix 2)
        # Note: SCADA signals don't have strong diurnal periodicity like astronomical data.
        # They correlate with wind/power, not time-of-day. Use more permissive settings.
        self.n_baseline_windows = 4  # Number of windows for CV calculation
        self.baseline_cv_threshold = 50.0  # Relaxed from 5% - SCADA signals are less periodic
        self.baseline_amplitude_threshold = 0.01  # Minimum amplitude to consider periodic
        self.periodic_freq_min = 1.0 / (30 * 24)  # cycles per hour (30-day period)
        self.periodic_freq_max = 1.0 / 2.0        # cycles per hour (2-hour period)
        self.n_freq_points = 5000

    def _build_periodic_baseline(
        self,
        df: pd.DataFrame,
        col: str,
        train_mask: np.ndarray,
        status_mask: np.ndarray
    ) -> Optional[PeriodicBaseline]:
        """
        Build LS periodic baseline from training data.
        This is the core Ab Astris methodology - multi-window CV validation.

        Args:
            df: DataFrame with all data
            col: Column name to analyze
            train_mask: Boolean mask for training rows
            status_mask: Boolean mask for normal status rows

        Returns:
            PeriodicBaseline or None if insufficient data
        """
        # Get training data with normal status
        combined_mask = train_mask & status_mask
        values, valid = self._get_valid_data(df, col, combined_mask)

        # Get timestamps in hours
        timestamps = pd.to_datetime(df['time_stamp'])
        # Convert to hours since start
        t0 = timestamps.iloc[0]
        hours = (timestamps - t0).dt.total_seconds().values / 3600.0

        # Apply mask
        hours_masked = hours[valid]
        values_masked = values[valid]

        if len(values_masked) < 500:  # Need sufficient data for LS
            return None

        # Remove mean for LS analysis
        signal_mean = np.mean(values_masked)
        signal_centered = values_masked - signal_mean

        # Frequency grid
        freqs = np.linspace(self.periodic_freq_min, self.periodic_freq_max, self.n_freq_points)

        # Full-signal LS
        ls = LombScargle(hours_masked, signal_centered)
        powers = ls.power(freqs)

        # Find dominant peak
        peak_indices, _ = find_peaks(powers, height=0.01)
        if len(peak_indices) == 0:
            # No significant peaks found - signal may not be periodic
            dominant_idx = np.argmax(powers)
        else:
            # Use highest peak
            dominant_idx = peak_indices[np.argmax(powers[peak_indices])]

        dominant_freq = freqs[dominant_idx]
        dominant_power = powers[dominant_idx]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else np.inf

        # Multi-window CV calculation (THE KEY AB ASTRIS METRIC)
        n_samples = len(hours_masked)
        window_size = n_samples // self.n_baseline_windows
        window_freqs = []

        for w in range(self.n_baseline_windows):
            start = w * window_size
            end = min(start + window_size, n_samples)

            t_win = hours_masked[start:end]
            s_win = signal_centered[start:end]

            if len(t_win) < 100:
                continue

            ls_win = LombScargle(t_win, s_win)
            p_win = ls_win.power(freqs)
            peak_freq = freqs[np.argmax(p_win)]
            window_freqs.append(peak_freq)

        # Compute CV
        if len(window_freqs) >= 2:
            freq_mean = np.mean(window_freqs)
            freq_std = np.std(window_freqs)
            cv = (freq_std / freq_mean * 100) if freq_mean > 0 else 100.0
        else:
            cv = 100.0  # Insufficient windows

        # Estimate residual fraction (variance not explained by dominant periodic component)
        total_variance = np.var(signal_centered)
        # Approximate periodic variance from LS power
        # This is a rough estimate; LS power doesn't directly map to variance
        periodic_variance = dominant_power * total_variance  # Approximate scaling
        residual_fraction = max(0, 1 - (periodic_variance / total_variance)) if total_variance > 0 else 1.0

        # A signal is considered to have usable periodicity if:
        # 1. CV is below threshold (some consistency across windows)
        # 2. Amplitude is above threshold (periodic component is meaningful)
        is_stable = (cv < self.baseline_cv_threshold and
                     dominant_power > self.baseline_amplitude_threshold)

        return PeriodicBaseline(
            signal_name=col,
            dominant_period_hours=dominant_period,
            dominant_freq=dominant_freq,
            dominant_amplitude=dominant_power,
            dominant_period_cv=cv,
            residual_fraction=residual_fraction,
            n_windows=len(window_freqs),
            window_freqs=window_freqs,
            is_stable=is_stable
        )

    def _compute_periodic_anomaly_score(
        self,
        window_values: np.ndarray,
        window_hours: np.ndarray,
        baseline: PeriodicBaseline
    ) -> Tuple[float, bool]:
        """
        Compare window's periodic structure to training baseline.

        Returns:
            (anomaly_score, is_degraded)
            anomaly_score: 0 (normal) to 1 (highly anomalous)
            is_degraded: True if periodic structure is significantly disrupted
        """
        if baseline is None or not baseline.is_stable:
            # No stable baseline - can't use periodic filtering
            return 0.0, False

        # Get valid data in window
        valid = ~np.isnan(window_values)
        if np.sum(valid) < 100:
            return 0.0, False

        hours_valid = window_hours[valid]
        values_valid = window_values[valid] - np.mean(window_values[valid])

        # Run LS on this window
        try:
            ls = LombScargle(hours_valid, values_valid)
            window_power = ls.power(baseline.dominant_freq)
        except Exception:
            return 0.0, False

        # 1. Amplitude ratio: compare window's periodic amplitude to baseline
        if baseline.dominant_amplitude > 0:
            amp_ratio = window_power / baseline.dominant_amplitude
        else:
            amp_ratio = 1.0

        # Normal range: amp_ratio ≈ 1.0 (seasonal variation might give 0.5–2.0)
        # Anomalous: amp_ratio < 0.3 (periodicity lost) or > 3.0 (amplified oscillation)
        if amp_ratio < 0.3:
            amp_anomaly = 1.0 - (amp_ratio / 0.3)  # Higher anomaly as ratio drops
        elif amp_ratio > 2.0:
            amp_anomaly = min(1.0, (amp_ratio - 2.0) / 3.0)  # Higher anomaly as ratio increases
        else:
            amp_anomaly = 0.0

        # 2. Residual increase: is there more non-periodic energy?
        total_variance = np.var(values_valid)
        periodic_variance = window_power * total_variance if total_variance > 0 else 0
        residual_fraction = 1 - (periodic_variance / total_variance) if total_variance > 0 else 1.0

        # Compare to baseline residual
        residual_increase = residual_fraction - baseline.residual_fraction
        residual_anomaly = max(0, residual_increase / 0.5)  # Normalize

        # Combined periodic anomaly score
        periodic_score = max(amp_anomaly, residual_anomaly * 0.5)

        # Determine if periodic structure is degraded
        is_degraded = (amp_ratio < 0.5 or amp_ratio > 2.5) or (residual_anomaly > 0.3)

        return periodic_score, is_degraded

    def _apply_periodic_filter(
        self,
        heuristic_score: float,
        periodic_score: float,
        periodic_degraded: bool,
        baseline_is_stable: bool
    ) -> float:
        """
        Combine heuristic anomaly score with periodic baseline check.

        If periodic structure is intact → reduce anomaly score (suppress false positives)
        If periodic structure is degraded → amplify anomaly score (confirm anomaly)
        """
        if not baseline_is_stable:
            # No stable baseline - use heuristic score as-is
            return heuristic_score

        if periodic_degraded:
            # Periodic structure disrupted - this confirms the anomaly
            return min(1.0, heuristic_score * 1.3 + periodic_score * 0.3)
        else:
            # Periodic structure intact - this is likely normal operation
            # Suppress the heuristic score (reduce false positives)
            return heuristic_score * 0.4

    # =========================================================================
    # NBM-RESIDUAL METHODS (v4 Brief)
    # =========================================================================

    def _build_residual_baseline(
        self,
        df: pd.DataFrame,
        signal_col: str,
        power_col: str,
        ambient_col: str,
        train_mask: np.ndarray,
        status_mask: np.ndarray
    ) -> Optional[ResidualBaseline]:
        """
        Build LS baseline from NBM residuals during training.

        This is the v4 approach: instead of running LS on raw signals (which have
        CV 53-71% because they track wind/power), we:
        1. Build an NBM that predicts temperature from operational conditions
        2. Compute residual = actual - predicted
        3. Run LS on the residual, which should have stable periodic structure

        Args:
            df: Full DataFrame
            signal_col: Temperature signal column to model
            power_col: Power column for NBM
            ambient_col: Ambient temperature column for NBM
            train_mask: Boolean mask for training rows
            status_mask: Boolean mask for normal status rows

        Returns:
            ResidualBaseline if successful, None if insufficient data
        """
        # Get training data with normal status
        combined_mask = train_mask & status_mask
        train_df = df[combined_mask].copy()

        if len(train_df) < 500:
            return None

        # Build NBM from training data
        nbm = build_nbm(
            train_df,
            signal_col,
            power_col,
            ambient_col,
            n_power_bins=10,
            n_ambient_bins=5
        )

        if nbm is None:
            return None

        # Compute residuals for training period
        residuals = compute_residuals(train_df, nbm, signal_col, power_col, ambient_col)

        # Filter to valid residuals
        valid_mask = ~np.isnan(residuals)
        residuals_valid = residuals[valid_mask]

        if len(residuals_valid) < 500:
            return None

        # Compute baseline statistics
        baseline_mean = np.mean(residuals_valid)
        baseline_std = np.std(residuals_valid)
        baseline_variance = baseline_std ** 2

        if baseline_std == 0:
            baseline_std = 1.0
            baseline_variance = 1.0

        # Create time axis in hours (assume 10-min intervals = 6 samples/hour)
        samples_per_hour = self.samples_per_hour
        hours = np.arange(len(residuals_valid)) / samples_per_hour

        # LS frequency grid (30 days to 2 hours)
        freqs = np.linspace(self.periodic_freq_min, self.periodic_freq_max, self.n_freq_points)

        # Run LS on training residuals
        residuals_centered = residuals_valid - baseline_mean
        ls = LombScargle(hours, residuals_centered)
        powers = ls.power(freqs)

        # Find dominant frequency
        dominant_idx = np.argmax(powers)
        dominant_freq = freqs[dominant_idx]
        dominant_power = powers[dominant_idx]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else np.inf

        # Total spectral power for energy comparison
        total_spectral_power = np.sum(powers)

        # Multi-window CV calculation
        n_samples = len(residuals_valid)
        window_size = n_samples // self.n_baseline_windows
        window_freqs = []

        for w in range(self.n_baseline_windows):
            start = w * window_size
            end = min(start + window_size, n_samples)

            if end - start < 100:
                continue

            t_win = hours[start:end]
            r_win = residuals_centered[start:end]

            ls_win = LombScargle(t_win, r_win)
            p_win = ls_win.power(freqs)
            peak_freq = freqs[np.argmax(p_win)]
            window_freqs.append(peak_freq)

        # Compute CV
        if len(window_freqs) >= 2:
            freq_mean = np.mean(window_freqs)
            freq_std = np.std(window_freqs)
            cv = (freq_std / freq_mean * 100) if freq_mean > 0 else 100.0
        else:
            cv = 100.0

        # Determine usability based on CV thresholds from brief v4
        is_excellent = cv < 20.0  # Strong periodic signal
        is_usable = cv < 40.0     # Acceptable for LS scoring

        return ResidualBaseline(
            signal_name=signal_col,
            nbm=nbm,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            baseline_variance=baseline_variance,
            ls_powers=powers,
            ls_freqs=freqs,
            dominant_freq=dominant_freq,
            dominant_power=dominant_power,
            dominant_period_hours=dominant_period,
            multi_window_cv=cv,
            total_spectral_power=total_spectral_power,
            is_usable=is_usable,
            is_excellent=is_excellent
        )

    def _compute_residual_anomaly_score(
        self,
        window_residuals: np.ndarray,
        window_hours: np.ndarray,
        baseline: ResidualBaseline
    ) -> Dict[str, float]:
        """
        Score a sliding window of residuals against the training baseline.

        Per Brief v4, uses weighted combination:
        - mean_shift: 35% - Direct thermal excess (bearing degradation → positive shift)
        - ls_score: 30% - Periodic structure change (THE TORQUESCOPE COMPONENT)
        - trend: 20% - Progressive degradation trend
        - variance: 15% - Increased variability

        Args:
            window_residuals: Residual values for this window
            window_hours: Time in hours for each sample
            baseline: ResidualBaseline from training

        Returns:
            Dict with:
            - anomaly_score: Combined score (0-1)
            - mean_shift_score: Mean shift component (0-1)
            - ls_score: LS spectral component (0-1)
            - trend_score: Trend component (0-1)
            - variance_score: Variance component (0-1)
            - mean_shift: Raw z-score of mean shift
            - var_ratio: Current/baseline variance ratio
            - ls_power_ratio: Window/baseline power at dominant frequency
            - spectral_energy_ratio: Window/baseline total spectral power
            - trend_slope: Degrees per day
        """
        # Filter to valid values
        valid = ~np.isnan(window_residuals)
        if np.sum(valid) < 100:
            return {
                'anomaly_score': 0.0,
                'mean_shift_score': 0.0,
                'ls_score': 0.0,
                'trend_score': 0.0,
                'variance_score': 0.0,
                'mean_shift': 0.0,
                'var_ratio': 1.0,
                'ls_power_ratio': 1.0,
                'spectral_energy_ratio': 1.0,
                'trend_slope': 0.0,
                'insufficient_data': True
            }

        wr = window_residuals[valid]
        wt = window_hours[valid]

        # === 1. MEAN SHIFT (35%) ===
        # Z-score relative to training baseline
        # Positive shift = bearing running hotter than expected
        window_mean = np.mean(wr)
        mean_shift = (window_mean - baseline.baseline_mean) / baseline.baseline_std
        # Normalize to 0-1 (saturates at 3 std)
        mean_shift_score = min(1.0, max(0.0, abs(mean_shift) / 3.0))

        # === 2. VARIANCE (15%) ===
        # Ratio of window variance to baseline variance
        window_var = np.var(wr)
        var_ratio = window_var / baseline.baseline_variance if baseline.baseline_variance > 0 else 1.0
        # Flag if variance exceeds 1.5x baseline
        if var_ratio <= 1.5:
            variance_score = 0.0
        else:
            variance_score = min(1.0, (var_ratio - 1.5) / 3.0)

        # === 3. LS SPECTRAL CHANGE (30%) - THE TORQUESCOPE COMPONENT ===
        ls_score = 0.0
        ls_power_ratio = 1.0
        spectral_energy_ratio = 1.0

        if baseline.is_usable:
            # Run LS on window residuals
            try:
                wr_centered = wr - np.mean(wr)
                ls = LombScargle(wt, wr_centered)
                window_powers = ls.power(baseline.ls_freqs)

                # 3a. Power ratio at dominant baseline frequency
                # How much has the main periodic component changed?
                dominant_idx = np.argmax(baseline.ls_powers)
                window_dominant_power = window_powers[dominant_idx]
                if baseline.dominant_power > 0:
                    ls_power_ratio = window_dominant_power / baseline.dominant_power
                else:
                    ls_power_ratio = 1.0

                # 3b. Total spectral energy ratio
                # Has new periodic energy appeared (e.g., fault cycling)?
                window_total_power = np.sum(window_powers)
                if baseline.total_spectral_power > 0:
                    spectral_energy_ratio = window_total_power / baseline.total_spectral_power
                else:
                    spectral_energy_ratio = 1.0

                # Score: penalize if periodic structure significantly changed
                # Normal range: 0.5 - 2.0 (seasonal variation)
                # Anomalous: < 0.3 (periodicity lost) or > 3.0 (amplified oscillation)
                if ls_power_ratio < 0.3:
                    amp_component = 0.5  # Periodicity significantly reduced
                elif ls_power_ratio > 3.0:
                    amp_component = min(0.5, (ls_power_ratio - 3.0) / 4.0)  # Amplified
                else:
                    amp_component = 0.0

                # Penalize if new spectral energy appeared
                if spectral_energy_ratio > 2.0:
                    energy_component = min(0.5, (spectral_energy_ratio - 2.0) / 4.0)
                else:
                    energy_component = 0.0

                ls_score = min(1.0, amp_component + energy_component)

            except Exception:
                ls_score = 0.0

        # === 4. TREND (20%) ===
        # Linear slope of residual over window (degrees per day)
        trend_slope = 0.0
        trend_score = 0.0

        if len(wr) >= 50:
            try:
                # Convert hours to days for interpretability
                days = (wt - wt[0]) / 24.0
                slope, intercept, r_value, p_value, std_err = stats.linregress(days, wr)
                trend_slope = slope  # degrees per day

                # Only count significant upward trends (p < 0.05)
                if slope > 0 and p_value < 0.05:
                    # Normalize by baseline std (how many stds per day?)
                    # Flag if rising > 0.5 std per day
                    normalized_slope = slope / baseline.baseline_std
                    trend_score = min(1.0, normalized_slope / 0.5)
            except Exception:
                pass

        # === COMBINED SCORE ===
        # Weights from Brief v4:
        # mean_shift: 35%, variance: 15%, ls_score: 30%, trend: 20%
        anomaly_score = (
            0.35 * mean_shift_score +
            0.15 * variance_score +
            0.30 * ls_score +
            0.20 * trend_score
        )

        return {
            'anomaly_score': float(anomaly_score),
            'mean_shift_score': float(mean_shift_score),
            'ls_score': float(ls_score),
            'trend_score': float(trend_score),
            'variance_score': float(variance_score),
            'mean_shift': float(mean_shift),
            'var_ratio': float(var_ratio),
            'ls_power_ratio': float(ls_power_ratio),
            'spectral_energy_ratio': float(spectral_energy_ratio),
            'trend_slope': float(trend_slope),
            'insufficient_data': False
        }

    def _get_valid_data(
        self,
        df: pd.DataFrame,
        col: str,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get valid (non-NaN, normal status) data from column."""
        values = df[col].values.astype(float)

        # Handle zeros as missing (common in Farms B/C)
        if not col.startswith('power'):
            zero_fraction = np.sum(values == 0) / len(values)
            if zero_fraction > 0.1:
                values = np.where(values == 0, np.nan, values)

        # Apply status mask if provided
        if mask is not None:
            values = np.where(mask, values, np.nan)

        # Create valid mask
        valid = ~np.isnan(values)

        return values, valid

    def _compute_baseline_stats(
        self,
        df: pd.DataFrame,
        col: str,
        train_mask: np.ndarray,
        status_mask: np.ndarray
    ) -> Dict[str, float]:
        """Compute baseline statistics from training period."""
        combined_mask = train_mask & status_mask
        values, valid = self._get_valid_data(df, col, combined_mask)

        valid_values = values[valid]

        if len(valid_values) < 100:
            return {'mean': np.nan, 'std': np.nan, 'median': np.nan}

        return {
            'mean': np.nanmean(valid_values),
            'std': np.nanstd(valid_values),
            'median': np.nanmedian(valid_values),
            'q25': np.nanpercentile(valid_values, 25),
            'q75': np.nanpercentile(valid_values, 75),
        }

    def _detect_temp_rise(
        self,
        window_values: np.ndarray,
        baseline_stats: Dict[str, float]
    ) -> float:
        """Detect temperature rise above baseline."""
        if np.isnan(baseline_stats['mean']) or len(window_values) < 10:
            return 0.0

        valid = ~np.isnan(window_values)
        if np.sum(valid) < 10:
            return 0.0

        current_mean = np.nanmean(window_values)
        baseline_mean = baseline_stats['mean']
        baseline_std = baseline_stats['std']

        if baseline_std == 0:
            return 0.0

        # Z-score of mean shift
        z_score = (current_mean - baseline_mean) / baseline_std

        # Convert to 0-1 score (saturates at 3 std)
        score = min(1.0, max(0.0, z_score / 3.0))

        return score

    def _detect_volatility_increase(
        self,
        window_values: np.ndarray,
        baseline_stats: Dict[str, float]
    ) -> float:
        """Detect increase in signal volatility."""
        if np.isnan(baseline_stats['std']) or baseline_stats['std'] == 0:
            return 0.0

        valid = ~np.isnan(window_values)
        if np.sum(valid) < 10:
            return 0.0

        current_std = np.nanstd(window_values)
        ratio = current_std / baseline_stats['std']

        # Score: 0 if ratio <= 1, increases to 1 at threshold
        if ratio <= 1.0:
            return 0.0

        score = min(1.0, (ratio - 1.0) / (self.volatility_threshold - 1.0))
        return score

    def _detect_trend(
        self,
        window_values: np.ndarray,
        hours_per_sample: float = 1/6
    ) -> float:
        """Detect upward trend in window."""
        valid = ~np.isnan(window_values)
        if np.sum(valid) < 50:
            return 0.0

        # Create time array in days
        times = np.arange(len(window_values)) * hours_per_sample / 24

        # Fit linear trend
        valid_times = times[valid]
        valid_values = window_values[valid]

        slope, intercept, r_value, p_value, std_err = stats.linregress(valid_times, valid_values)

        # Only count significant upward trends
        if slope <= 0 or p_value > 0.05:
            return 0.0

        # Score based on slope magnitude (degrees per day)
        score = min(1.0, slope / self.trend_threshold)
        return score

    def _compute_temp_delta(
        self,
        df: pd.DataFrame,
        temp_col: str,
        ambient_col: str,
        mask: np.ndarray
    ) -> np.ndarray:
        """Compute temperature delta (component - ambient)."""
        temp_values, temp_valid = self._get_valid_data(df, temp_col, mask)
        ambient_values, ambient_valid = self._get_valid_data(df, ambient_col, mask)

        both_valid = temp_valid & ambient_valid
        delta = np.full(len(df), np.nan)
        delta[both_valid] = temp_values[both_valid] - ambient_values[both_valid]

        return delta

    def detect_single_signal(
        self,
        df: pd.DataFrame,
        signal_col: str,
        train_mask: np.ndarray,
        status_mask: np.ndarray,
        test_start_idx: int,
        periodic_baseline: Optional[PeriodicBaseline] = None
    ) -> List[WindowResult]:
        """
        Run detection on a single signal with LS periodic baseline filtering.

        Updated for v3 brief Fix 2: Uses periodic baseline to suppress false positives.
        """
        # Compute baseline stats
        baseline_stats = self._compute_baseline_stats(df, signal_col, train_mask, status_mask)

        if np.isnan(baseline_stats['mean']):
            return []

        results = []
        n_rows = len(df)

        # Get full signal values
        values, valid = self._get_valid_data(df, signal_col, status_mask)

        # Get timestamps in hours for periodic analysis
        timestamps = pd.to_datetime(df['time_stamp'])
        t0 = timestamps.iloc[0]
        hours = (timestamps - t0).dt.total_seconds().values / 3600.0

        # Sliding window
        window_start = test_start_idx

        while window_start + self.window_size <= n_rows:
            window_end = window_start + self.window_size
            window_values = values[window_start:window_end]
            window_hours = hours[window_start:window_end]

            # Run heuristic detectors
            temp_rise = self._detect_temp_rise(window_values, baseline_stats)
            volatility = self._detect_volatility_increase(window_values, baseline_stats)
            trend = self._detect_trend(window_values)

            # Compute heuristic score (max of indicators)
            heuristic_score = max(temp_rise, volatility, trend)

            # Compute periodic anomaly score (Fix 2)
            periodic_score = 0.0
            periodic_degraded = False
            baseline_is_stable = periodic_baseline is not None and periodic_baseline.is_stable

            if periodic_baseline is not None:
                periodic_score, periodic_degraded = self._compute_periodic_anomaly_score(
                    window_values, window_hours, periodic_baseline
                )

            # Apply periodic filter to get final score (Fix 2)
            overall = self._apply_periodic_filter(
                heuristic_score, periodic_score, periodic_degraded, baseline_is_stable
            )

            signal_result = SignalAnomaly(
                signal_name=signal_col,
                anomaly_score=overall,
                trend_score=trend,
                volatility_score=volatility,
                periodic_score=periodic_score,
                periodic_degraded=periodic_degraded
            )

            results.append(WindowResult(
                window_start=window_start,
                window_end=window_end,
                overall_score=overall,
                n_signals_anomalous=1 if overall > 0.5 else 0,
                n_periodic_degraded=1 if periodic_degraded else 0,
                signal_results=[signal_result],
                periodic_baseline_used=baseline_is_stable
            ))

            window_start += self.step_size

        return results

    def detect_multi_signal(
        self,
        df: pd.DataFrame,
        signal_cols: List[str],
        train_mask: np.ndarray,
        test_start_idx: int,
        ambient_col: Optional[str] = None
    ) -> List[WindowResult]:
        """
        Run detection across multiple signals with consensus voting.

        Updated for v3 brief Fix 2: Builds LS periodic baselines and uses them
        to filter false positives and confirm genuine anomalies.

        Args:
            df: DataFrame with all data
            signal_cols: List of signal columns to analyze
            train_mask: Boolean mask for training rows
            test_start_idx: Index where test period starts
            ambient_col: Optional ambient temperature column for delta calculation

        Returns:
            List of WindowResult with multi-signal consensus scores
        """
        # Status mask (only normal operation)
        status_mask = df['status_type_id'].isin([0, 2]).values

        # Build periodic baselines for each signal (Fix 2)
        periodic_baselines: Dict[str, PeriodicBaseline] = {}
        for col in signal_cols:
            if col not in df.columns:
                continue
            baseline = self._build_periodic_baseline(df, col, train_mask, status_mask)
            if baseline is not None:
                periodic_baselines[col] = baseline

        # Collect per-signal results
        all_signal_results: Dict[str, List[WindowResult]] = {}

        for col in signal_cols:
            if col not in df.columns:
                continue

            # Pass periodic baseline if available
            periodic_baseline = periodic_baselines.get(col)

            results = self.detect_single_signal(
                df, col, train_mask, status_mask, test_start_idx,
                periodic_baseline=periodic_baseline
            )

            if results:
                all_signal_results[col] = results

        if not all_signal_results:
            return []

        # Also try temperature deltas if ambient available
        if ambient_col and ambient_col in df.columns:
            for col in signal_cols:
                if 'temp' in col.lower() and col in df.columns:
                    delta_col = f"{col}_delta"
                    df[delta_col] = self._compute_temp_delta(df, col, ambient_col, status_mask)

                    # Build baseline for delta signal too
                    delta_baseline = self._build_periodic_baseline(
                        df, delta_col, train_mask, status_mask
                    )

                    delta_results = self.detect_single_signal(
                        df, delta_col, train_mask, status_mask, test_start_idx,
                        periodic_baseline=delta_baseline
                    )

                    if delta_results:
                        all_signal_results[delta_col] = delta_results

        # Merge results across signals
        reference_signal = list(all_signal_results.keys())[0]
        reference_results = all_signal_results[reference_signal]

        merged_results = []

        for i, ref_result in enumerate(reference_results):
            # Collect all signal scores for this window
            window_signals = []
            scores = []
            n_periodic_degraded = 0
            any_baseline_used = False

            for signal_name, signal_results in all_signal_results.items():
                if i < len(signal_results):
                    result = signal_results[i]
                    window_signals.extend(result.signal_results)
                    scores.append(result.overall_score)
                    n_periodic_degraded += result.n_periodic_degraded
                    if result.periodic_baseline_used:
                        any_baseline_used = True

            # Consensus: signals above threshold, weighted by count
            n_anomalous = sum(1 for s in scores if s > 0.3)

            # Multi-signal consensus score with periodic degradation boost (Fix 2)
            # If multiple signals show periodic degradation, this is strong evidence
            if n_periodic_degraded >= 3:
                # Strong periodic degradation across signals - high confidence anomaly
                overall = 0.95
            elif n_periodic_degraded >= 2:
                overall = 0.85
            elif n_anomalous >= 3:
                overall = 0.9
            elif n_anomalous >= 2:
                overall = 0.7
            elif n_anomalous >= 1:
                overall = 0.5
            else:
                overall = np.mean(scores) if scores else 0.0

            merged_results.append(WindowResult(
                window_start=ref_result.window_start,
                window_end=ref_result.window_end,
                overall_score=overall,
                n_signals_anomalous=n_anomalous,
                n_periodic_degraded=n_periodic_degraded,
                signal_results=window_signals,
                periodic_baseline_used=any_baseline_used
            ))

        return merged_results

    def results_to_predictions(
        self,
        results: List[WindowResult],
        n_total: int,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Convert window results to per-timestamp binary predictions."""
        predictions = np.zeros(n_total, dtype=int)

        for result in results:
            if result.overall_score >= threshold:
                # Mark the step interval as anomaly
                step_end = min(result.window_start + self.step_size, n_total)
                predictions[result.window_start:step_end] = 1

        return predictions

    # =========================================================================
    # NBM-RESIDUAL DETECTION (v4 Brief) - MAIN ENTRY POINT
    # =========================================================================

    def detect_multi_signal_residual(
        self,
        df: pd.DataFrame,
        signal_cols: List[str],
        farm: str,
        train_mask: np.ndarray,
        test_start_idx: int,
        power_col: Optional[str] = None,
        ambient_col: Optional[str] = None,
        wind_col: Optional[str] = None
    ) -> Tuple[List[WindowResult], Dict[str, ResidualBaseline]]:
        """
        NBM-Residual based multi-signal anomaly detection.

        This is the v4 approach: instead of running LS on raw signals (which have
        CV 53-71% because they correlate with wind/power, not time-of-day), we:

        1. Build NBMs that predict temperature from operational conditions
        2. Compute residuals (actual - predicted) - removes operational variation
        3. Build LS baselines on training residuals
        4. Run sliding window detection on prediction residuals
        5. Multi-signal consensus with v4 scoring weights

        Scoring weights (Brief v4):
        - mean_shift: 35% - Direct thermal excess
        - ls_score: 30% - Periodic structure change (THE TORQUESCOPE COMPONENT)
        - trend: 20% - Progressive degradation trend
        - variance: 15% - Increased variability

        Args:
            df: Full DataFrame with all data
            signal_cols: Temperature signal columns to analyze
            farm: Farm identifier ('A', 'B', 'C' or 'Wind Farm A', etc.)
            train_mask: Boolean mask for training rows
            test_start_idx: Index where test/prediction period starts
            power_col: Power column (auto-detected if None)
            ambient_col: Ambient temperature column (auto-detected if None)
            wind_col: Wind speed column (auto-detected if None)

        Returns:
            Tuple of:
            - List of WindowResult with multi-signal consensus scores
            - Dict mapping signal names to their ResidualBaseline (for diagnostics)
        """
        # Status mask (only normal operation)
        status_mask = df['status_type_id'].isin([0, 2]).values

        # Get operational columns
        op_cols = get_operational_columns(farm, df)
        power_col = power_col or op_cols.get('power')
        ambient_col = ambient_col or op_cols.get('ambient')
        wind_col = wind_col or op_cols.get('wind')

        if power_col is None or ambient_col is None:
            # Cannot build NBMs without operational columns
            return [], {}

        # Build residual baselines for each signal
        residual_baselines: Dict[str, ResidualBaseline] = {}

        for col in signal_cols:
            if col not in df.columns:
                continue
            if col == ambient_col:
                # Don't model ambient as function of itself
                continue

            baseline = self._build_residual_baseline(
                df, col, power_col, ambient_col, train_mask, status_mask
            )
            if baseline is not None:
                residual_baselines[col] = baseline

        if not residual_baselines:
            return [], {}

        # Get timestamps in hours
        timestamps = pd.to_datetime(df['time_stamp'])
        t0 = timestamps.iloc[0]
        hours = (timestamps - t0).dt.total_seconds().values / 3600.0

        n_rows = len(df)

        # Compute residuals for entire prediction period for each signal
        all_residuals: Dict[str, np.ndarray] = {}
        for col, baseline in residual_baselines.items():
            residuals = compute_residuals(
                df, baseline.nbm, col, power_col, ambient_col
            )
            all_residuals[col] = residuals

        # Collect per-window, per-signal scores
        # Structure: window_idx -> signal_name -> score_dict
        all_window_scores: Dict[int, Dict[str, Dict[str, float]]] = {}

        window_start = test_start_idx
        window_idx = 0

        while window_start + self.window_size <= n_rows:
            window_end = window_start + self.window_size
            window_hours = hours[window_start:window_end]

            all_window_scores[window_idx] = {}

            for col, baseline in residual_baselines.items():
                window_residuals = all_residuals[col][window_start:window_end]

                score_dict = self._compute_residual_anomaly_score(
                    window_residuals, window_hours, baseline
                )
                all_window_scores[window_idx][col] = score_dict

            window_start += self.step_size
            window_idx += 1

        # Merge into WindowResults with multi-signal consensus
        merged_results = []
        window_start = test_start_idx

        for w_idx in range(window_idx):
            window_end = window_start + self.window_size
            signal_scores = all_window_scores[w_idx]

            # Build signal anomaly list
            signal_results = []
            scores = []
            n_ls_usable = 0

            for col, score_dict in signal_scores.items():
                if score_dict.get('insufficient_data', False):
                    continue

                score = score_dict['anomaly_score']
                scores.append(score)

                # Count signals with usable LS baselines
                if residual_baselines[col].is_usable:
                    n_ls_usable += 1

                signal_results.append(SignalAnomaly(
                    signal_name=col,
                    anomaly_score=score,
                    trend_score=score_dict['trend_score'],
                    volatility_score=score_dict['variance_score'],
                    periodic_score=score_dict['ls_score'],
                    periodic_degraded=(score_dict['ls_score'] > 0.3)
                ))

            # Multi-signal consensus
            if not scores:
                overall = 0.0
                n_anomalous = 0
            else:
                n_anomalous = sum(1 for s in scores if s > 0.3)

                # Consensus logic (similar to original but using residual scores)
                if n_anomalous >= 3:
                    overall = 0.9
                elif n_anomalous >= 2:
                    overall = 0.7
                elif n_anomalous >= 1:
                    overall = max(0.5, max(scores))
                else:
                    overall = np.mean(scores)

            merged_results.append(WindowResult(
                window_start=window_start,
                window_end=window_end,
                overall_score=overall,
                n_signals_anomalous=n_anomalous,
                n_periodic_degraded=sum(
                    1 for sr in signal_results if sr.periodic_degraded
                ),
                signal_results=signal_results,
                periodic_baseline_used=(n_ls_usable > 0)
            ))

            window_start += self.step_size

        return merged_results, residual_baselines

    # =========================================================================
    # V5 HYBRID DETECTION (v5 Brief)
    # =========================================================================

    def _compute_nbm_gating_score(
        self,
        window_data: pd.DataFrame,
        signal_cols: List[str],
        residual_baselines: Dict[str, 'ResidualBaseline'],
        power_col: str,
        ambient_col: str
    ) -> float:
        """
        Compute NBM-based gating score for a window.

        This checks if the NBM residuals are elevated (bearing running hotter
        than expected for operational conditions). Returns 0-1 score.
        """
        if not residual_baselines:
            return 0.0

        scores = []
        for col, baseline in residual_baselines.items():
            if col not in signal_cols:
                continue

            # Compute residuals for this window
            residuals = compute_residuals(
                window_data, baseline.nbm, col, power_col, ambient_col
            )

            valid = ~np.isnan(residuals)
            if np.sum(valid) < 50:
                continue

            wr = residuals[valid]

            # Mean shift z-score
            mean_shift = (np.mean(wr) - baseline.baseline_mean) / baseline.baseline_std
            mean_shift_score = min(1.0, max(0.0, abs(mean_shift) / 3.0))

            # Variance ratio
            var_ratio = np.var(wr) / baseline.baseline_variance if baseline.baseline_variance > 0 else 1.0
            if var_ratio > 1.5:
                var_score = min(1.0, (var_ratio - 1.5) / 3.0)
            else:
                var_score = 0.0

            # Combined: weight mean shift higher (it's the key thermal signature)
            signal_score = 0.7 * mean_shift_score + 0.3 * var_score
            scores.append(signal_score)

        if not scores:
            return 0.0

        # Use max of signal scores (any signal with elevated residual is suspicious)
        return max(scores)

    def _detect_non_temperature_signals(
        self,
        window_data: pd.DataFrame,
        train_stats: Dict[str, Dict[str, float]],
        wind_farm: str
    ) -> Dict[str, float]:
        """
        Compute anomaly scores for non-temperature signals (vibration, pressure, position).

        These are farm-specific and complement the temperature-based detection.
        Per Brief v5: Add vibration, hydraulic pressure, pitch position for Farm C.

        Returns: dict of {signal_name: anomaly_score}
        """
        scores = {}

        # Farm C specific signals
        if 'C' in wind_farm:
            # Vibration: rolling variance increase
            vib_cols = ['sensor_90_avg', 'sensor_91_avg', 'sensor_92_avg', 'sensor_93_avg']
            for col in vib_cols:
                if col in window_data.columns and col in train_stats:
                    window_var = window_data[col].var()
                    train_var = train_stats[col].get('variance', train_stats[col].get('std', 1.0) ** 2)
                    if train_var > 0:
                        ratio = window_var / train_var
                        # Flag if variance exceeds 2x baseline
                        if ratio > 2.0:
                            scores[col] = min(1.0, (ratio - 2.0) / 3.0)

            # Hydraulic pressure: mean shift
            hyd_cols = ['sensor_48_avg', 'sensor_49_avg', 'sensor_50_avg',
                        'sensor_51_avg', 'sensor_52_avg', 'sensor_54_avg', 'sensor_55_avg']
            for col in hyd_cols:
                if col in window_data.columns and col in train_stats:
                    shift = abs(window_data[col].mean() - train_stats[col]['mean'])
                    if train_stats[col]['std'] > 0:
                        z_score = shift / train_stats[col]['std']
                        if z_score > 2.0:
                            scores[col] = min(1.0, (z_score - 2.0) / 3.0)

            # Pitch axis divergence: all three blades should behave similarly
            pitch_cols = ['sensor_100_avg', 'sensor_101_avg', 'sensor_102_avg']
            available = [c for c in pitch_cols if c in window_data.columns]
            if len(available) >= 2:
                # Std across axes indicates divergence
                axis_std = window_data[available].std(axis=1).mean()
                # Compare to training axis std if available
                train_axis_std = train_stats.get('_pitch_axis_std', {}).get('mean', axis_std)
                if train_axis_std > 0 and axis_std > 2 * train_axis_std:
                    scores['pitch_divergence'] = min(1.0, (axis_std / train_axis_std - 2.0) / 3.0)

            # Gearbox oil level and pressure
            oil_cols = ['sensor_94_avg', 'sensor_117_avg', 'sensor_118_avg']
            for col in oil_cols:
                if col in window_data.columns and col in train_stats:
                    shift = abs(window_data[col].mean() - train_stats[col]['mean'])
                    if train_stats[col]['std'] > 0:
                        z_score = shift / train_stats[col]['std']
                        if z_score > 2.0:
                            scores[col] = min(1.0, (z_score - 2.0) / 3.0)

        # Farm B specific signals
        if 'B' in wind_farm:
            # Drive train and tower vibration
            vib_cols = ['sensor_54_avg', 'sensor_55_avg', 'sensor_56_avg']
            for col in vib_cols:
                if col in window_data.columns and col in train_stats:
                    window_var = window_data[col].var()
                    train_var = train_stats[col].get('variance', train_stats[col].get('std', 1.0) ** 2)
                    if train_var > 0:
                        ratio = window_var / train_var
                        if ratio > 2.0:
                            scores[col] = min(1.0, (ratio - 2.0) / 3.0)

            # Motor current (pitch motors)
            motor_cols = ['sensor_28_avg', 'sensor_29_avg', 'sensor_30_avg']
            for col in motor_cols:
                if col in window_data.columns and col in train_stats:
                    shift = abs(window_data[col].mean() - train_stats[col]['mean'])
                    if train_stats[col]['std'] > 0:
                        z_score = shift / train_stats[col]['std']
                        if z_score > 2.0:
                            scores[col] = min(1.0, (z_score - 2.0) / 3.0)

        return scores

    def _build_non_temp_train_stats(
        self,
        df: pd.DataFrame,
        train_mask: np.ndarray,
        status_mask: np.ndarray,
        wind_farm: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Build training statistics for non-temperature signals.
        """
        combined_mask = train_mask & status_mask
        train_df = df[combined_mask]

        stats = {}

        # Determine which columns to compute stats for based on farm
        if 'C' in wind_farm:
            cols = (
                ['sensor_90_avg', 'sensor_91_avg', 'sensor_92_avg', 'sensor_93_avg'] +
                ['sensor_48_avg', 'sensor_49_avg', 'sensor_50_avg', 'sensor_51_avg',
                 'sensor_52_avg', 'sensor_54_avg', 'sensor_55_avg'] +
                ['sensor_94_avg', 'sensor_117_avg', 'sensor_118_avg'] +
                ['sensor_100_avg', 'sensor_101_avg', 'sensor_102_avg']
            )

            # Also compute pitch axis divergence baseline
            pitch_cols = ['sensor_100_avg', 'sensor_101_avg', 'sensor_102_avg']
            available = [c for c in pitch_cols if c in train_df.columns]
            if len(available) >= 2:
                axis_std = train_df[available].std(axis=1).mean()
                stats['_pitch_axis_std'] = {'mean': axis_std, 'std': axis_std * 0.5}

        elif 'B' in wind_farm:
            cols = (
                ['sensor_54_avg', 'sensor_55_avg', 'sensor_56_avg'] +
                ['sensor_28_avg', 'sensor_29_avg', 'sensor_30_avg']
            )
        else:
            cols = []

        for col in cols:
            if col in train_df.columns:
                values = train_df[col].values
                valid = ~np.isnan(values)
                if np.sum(valid) > 100:
                    stats[col] = {
                        'mean': float(np.nanmean(values)),
                        'std': float(np.nanstd(values)),
                        'variance': float(np.nanvar(values))
                    }

        return stats

    def detect_multi_signal_hybrid(
        self,
        df: pd.DataFrame,
        signal_cols: List[str],
        farm: str,
        train_mask: np.ndarray,
        test_start_idx: int,
        power_col: Optional[str] = None,
        ambient_col: Optional[str] = None,
        wind_col: Optional[str] = None,
        use_per_farm_tuning: bool = False,
        ablation_mode: str = "hybrid"
    ) -> Tuple[List['WindowResult'], Dict[str, 'ResidualBaseline'], List[float]]:
        """
        V5 Hybrid detection: v3 heuristic primary, v4 NBM gating.

        Per Brief v5:
        - Stage 1: v3 heuristic detector produces raw anomaly score
        - Stage 2: NBM residual check modulates the score

        Combination logic:
        - Both agree (heuristic >= 0.3 AND nbm >= 0.2): boost score (max * 1.1)
        - Heuristic only (heuristic >= 0.3 AND nbm < 0.2): halve score (heuristic * 0.5)
        - NBM only (heuristic < 0.3 AND nbm >= 0.3): moderate (nbm * 0.7)
        - Neither: low score (max * 0.3)

        Also includes non-temperature signals for Farm B/C per Brief v5.

        Per-farm tuning (when use_per_farm_tuning=True):
        - Farm A: threshold=0.475, suppression=0.5 (baseline)
        - Farm B: threshold=0.65, suppression=0.3 (reduce 89% FP rate)
        - Farm C: threshold=0.38, suppression=0.7 (improve 41% detection)

        Args:
            df: Full DataFrame with all data
            signal_cols: Temperature signal columns to analyze
            farm: Farm identifier ('A', 'B', 'C' or 'Wind Farm A', etc.)
            train_mask: Boolean mask for training rows
            test_start_idx: Index where test/prediction period starts
            power_col: Power column (auto-detected if None)
            ambient_col: Ambient temperature column (auto-detected if None)
            wind_col: Wind speed column (auto-detected if None)
            use_per_farm_tuning: Use farm-specific thresholds/suppression (default False)
            ablation_mode: Ablation mode for isolating component contributions.
                "hybrid" (default): Full v5 logic unchanged.
                "periodic_only": Score from LS periodic baseline only (no NBM, no heuristics).
                "nbm_only": Score from NBM gating only (heuristic scores ignored).

        Returns:
            Tuple of:
            - List of WindowResult with hybrid consensus scores
            - Dict mapping signal names to their ResidualBaseline
            - List of continuous scores per window (for threshold sweep)
        """
        # Status mask (only normal operation)
        status_mask = df['status_type_id'].isin([0, 2]).values

        # Get operational columns
        op_cols = get_operational_columns(farm, df)
        power_col = power_col or op_cols.get('power')
        ambient_col = ambient_col or op_cols.get('ambient')
        wind_col = wind_col or op_cols.get('wind')

        # =====================================================================
        # STAGE 1: Run v3 heuristic detector (primary)
        # =====================================================================
        heuristic_results = self.detect_multi_signal(
            df, signal_cols, train_mask, test_start_idx, ambient_col=ambient_col
        )

        if not heuristic_results:
            return [], {}, []

        # =====================================================================
        # STAGE 2: Build NBM residual baselines for gating
        # =====================================================================
        residual_baselines: Dict[str, ResidualBaseline] = {}

        if ablation_mode != "periodic_only" and power_col and ambient_col:
            for col in signal_cols:
                if col not in df.columns or col == ambient_col:
                    continue

                baseline = self._build_residual_baseline(
                    df, col, power_col, ambient_col, train_mask, status_mask
                )
                if baseline is not None:
                    residual_baselines[col] = baseline

        # =====================================================================
        # STAGE 3: Build non-temperature signal stats (for Farm B/C)
        # =====================================================================
        if ablation_mode == "hybrid":
            non_temp_stats = self._build_non_temp_train_stats(df, train_mask, status_mask, farm)
        else:
            non_temp_stats = {}

        # =====================================================================
        # STAGE 4: Apply hybrid scoring to each window
        # =====================================================================
        n_rows = len(df)
        hybrid_results = []
        continuous_scores = []

        for heur_result in heuristic_results:
            window_start = heur_result.window_start
            window_end = heur_result.window_end
            heuristic_score = heur_result.overall_score

            # Get window data for NBM gating
            window_df = df.iloc[window_start:window_end].copy()

            # =================================================================
            # ABLATION BRANCHING
            # =================================================================

            if ablation_mode == "periodic_only":
                # ---------------------------------------------------------
                # PERIODIC ONLY: Use raw LS periodic scores, ignore
                # heuristic combination and NBM entirely.
                # ---------------------------------------------------------
                periodic_scores = []
                n_degraded = 0
                for sr in heur_result.signal_results:
                    periodic_scores.append(sr.periodic_score)
                    if sr.periodic_degraded:
                        n_degraded += 1

                # Aggregate across signals (mirrors multi-signal consensus
                # logic at lines 1307-1319 but using periodic scores only)
                if n_degraded >= 3:
                    hybrid_score = 0.95
                elif n_degraded >= 2:
                    hybrid_score = 0.85
                elif periodic_scores:
                    hybrid_score = max(periodic_scores)
                else:
                    hybrid_score = 0.0

            elif ablation_mode == "nbm_only":
                # ---------------------------------------------------------
                # NBM ONLY: Use NBM gating score directly, ignore
                # heuristic and periodic scores.
                # ---------------------------------------------------------
                farm_key = get_farm_key(farm)
                nbm_score = 0.0
                nbm_disabled = FARM_NBM_DISABLED.get(farm_key, False)

                if not nbm_disabled and residual_baselines and power_col and ambient_col:
                    nbm_score = self._compute_nbm_gating_score(
                        window_df, signal_cols, residual_baselines, power_col, ambient_col
                    )

                hybrid_score = nbm_score

            else:
                # ---------------------------------------------------------
                # HYBRID (DEFAULT): Full v5 combination logic, unchanged.
                # ---------------------------------------------------------

                # Compute NBM gating score
                farm_key = get_farm_key(farm)
                nbm_score = 0.0
                nbm_disabled = FARM_NBM_DISABLED.get(farm_key, False)

                if not nbm_disabled and residual_baselines and power_col and ambient_col:
                    nbm_score = self._compute_nbm_gating_score(
                        window_df, signal_cols, residual_baselines, power_col, ambient_col
                    )
                # If NBM disabled for this farm, nbm_score stays 0.0

                # Compute non-temperature signal scores
                non_temp_scores = self._detect_non_temperature_signals(
                    window_df, non_temp_stats, farm
                )

                # Include non-temp in heuristic assessment
                if non_temp_scores:
                    max_non_temp = max(non_temp_scores.values())
                    # Boost heuristic score if non-temp signals are anomalous
                    if max_non_temp > 0.3:
                        heuristic_score = max(heuristic_score, max_non_temp)

                # =====================================================================
                # V5.1 Hybrid Combination Logic - CLEAR SCORE SEPARATION
                # =====================================================================
                # Key principle: when heuristic and NBM disagree, the score should be
                # LOW (0.1-0.25), not MEDIUM (0.45-0.50). Medium scores create a dead
                # zone that no threshold can resolve.
                #
                # Score bands:
                #   0.7-1.0: Both agree anomalous → high confidence
                #   0.4-0.6: NBM says anomalous, heuristic moderate → moderate confidence
                #   0.1-0.25: Heuristic only (disagreement) → low confidence
                #   0.0-0.1: Neither fires → normal
                #
                # Per-farm tuning (when use_per_farm_tuning=True):
                #   - Farm B: more aggressive suppression (0.3) to reduce 89% FP rate
                #   - Farm C: less suppression (0.7) to improve 41% detection rate
                # =====================================================================

                # Get farm-specific suppression factor if per-farm tuning enabled
                if use_per_farm_tuning:
                    farm_key = get_farm_key(farm)
                    suppression_factor = FARM_SUPPRESSION.get(farm_key, 0.5)
                else:
                    suppression_factor = 0.5  # Default v5 behavior

                if heuristic_score >= 0.3 and nbm_score >= 0.2:
                    # BOTH AGREE: genuine anomaly. Score in 0.7-1.0 range.
                    hybrid_score = min(1.0, 0.7 + 0.3 * max(heuristic_score, nbm_score))
                elif heuristic_score >= 0.3 and nbm_score < 0.2:
                    # HEURISTIC ONLY: Apply suppression factor.
                    # Per-farm tuning uses FARM_SUPPRESSION values:
                    #   - Farm B (0.3): more aggressive suppression → scores ~0.3 * heuristic
                    #   - Farm C (0.7): trust heuristics more → scores ~0.7 * heuristic
                    # Original v5 behavior (suppression_factor=0.5): scores ~0.5 * heuristic
                    if use_per_farm_tuning:
                        # Per-farm: suppressed_score = suppression_factor * heuristic_score
                        hybrid_score = suppression_factor * heuristic_score
                    else:
                        # Original v5 behavior: 0.5x suppression factor
                        # This produces scores of 0.15-0.50 for heuristic-only signals
                        # (v5.1 used a floor formula that regressed to CARE=0.442)
                        hybrid_score = 0.5 * heuristic_score
                elif heuristic_score < 0.3 and nbm_score >= 0.3:
                    # NBM ONLY: moderate confidence (0.4-0.6)
                    hybrid_score = 0.4 + 0.2 * nbm_score  # Maps 0.3-1.0 → 0.46-0.60
                else:
                    # NEITHER: normal (0.0-0.1)
                    hybrid_score = max(heuristic_score, nbm_score) * 0.1

            continuous_scores.append(hybrid_score)

            # Update the result with hybrid score
            hybrid_results.append(WindowResult(
                window_start=window_start,
                window_end=window_end,
                overall_score=hybrid_score,
                n_signals_anomalous=heur_result.n_signals_anomalous,
                n_periodic_degraded=heur_result.n_periodic_degraded,
                signal_results=heur_result.signal_results,
                periodic_baseline_used=heur_result.periodic_baseline_used
            ))

        return hybrid_results, residual_baselines, continuous_scores


    # =========================================================================
    # V6 DETECTION (v6 Brief) - SIMPLE NBM Z-SCORE APPROACH
    # =========================================================================

    def detect_v6(
        self,
        df: pd.DataFrame,
        signal_cols: List[str],
        farm: str,
        train_mask: np.ndarray,
        test_start_idx: int,
        power_col: Optional[str] = None,
        ambient_col: Optional[str] = None,
        z_threshold: float = 2.0
    ) -> Tuple[List['WindowResult'], Dict[str, 'NBMModel'], Dict[str, Dict]]:
        """
        v6 Detection: Simple NBM z-score approach.

        This is a clean break from v1-v5. No Lomb-Scargle, no periodograms.
        Just simple binned-average NBM with z-score thresholding.

        Per Brief v6:
        - 3-day windows, 6-hour steps (vs. 7-day/1-day in v1-v5)
        - Z-score = (actual - expected) / training_std
        - Multi-sensor consensus: 0.7 * max_score + 0.3 * top3_mean
        - Only positive z-scores matter (running hotter than expected)

        Args:
            df: Full DataFrame with all data
            signal_cols: Temperature signal columns to analyze
            farm: Farm identifier ('A', 'B', 'C' or 'Wind Farm A', etc.)
            train_mask: Boolean mask for training rows
            test_start_idx: Index where test/prediction period starts
            power_col: Power column (auto-detected if None)
            ambient_col: Ambient temperature column (auto-detected if None)
            z_threshold: Z-score threshold for anomaly detection (default 2.0)

        Returns:
            Tuple of:
            - List of WindowResult with z-score based consensus scores
            - Dict mapping signal names to their NBMModel
            - Dict with per-sensor z-score statistics for diagnostics
        """
        # v6 window parameters (3 days, 6 hours step)
        v6_window_hours = 3 * 24  # 3 days
        v6_step_hours = 6  # 6 hours
        samples_per_hour = self.samples_per_hour

        v6_window_size = v6_window_hours * samples_per_hour
        v6_step_size = v6_step_hours * samples_per_hour

        # Status mask (only normal operation)
        status_mask = df['status_type_id'].isin([0, 2]).values

        # Get operational columns
        op_cols = get_operational_columns(farm, df)
        power_col = power_col or op_cols.get('power')
        ambient_col = ambient_col or op_cols.get('ambient')

        if power_col is None or ambient_col is None:
            return [], {}, {}

        # =====================================================================
        # STAGE 1: Build NBMs for each sensor (v6 enhanced: 20x10 bins)
        # =====================================================================
        combined_mask = train_mask & status_mask
        train_df = df[combined_mask].copy()

        nbm_models: Dict[str, NBMModel] = {}
        sensor_stats: Dict[str, Dict] = {}

        for col in signal_cols:
            if col not in df.columns or col == ambient_col:
                continue

            # Build NBM with v6 settings (20x10 bins)
            nbm = build_nbm(
                train_df,
                col,
                power_col,
                ambient_col,
                n_power_bins=20,
                n_ambient_bins=10
            )

            if nbm is not None:
                nbm_models[col] = nbm
                sensor_stats[col] = {
                    'n_bins_populated': len(nbm.bin_medians),
                    'global_median': nbm.global_median,
                    'global_std': nbm.global_std,
                    'n_training_samples': nbm.n_training_samples
                }

        if not nbm_models:
            return [], {}, {}

        # =====================================================================
        # STAGE 2: Compute z-scores for entire dataset
        # =====================================================================
        all_z_scores: Dict[str, np.ndarray] = {}

        for col, nbm in nbm_models.items():
            # Predict and get stds
            predictions, stds = predict_nbm(df, nbm, power_col, ambient_col, return_stds=True)

            # Get actual values
            actual = df[col].values.astype(float)

            # Handle zeros as missing (common in Farms B/C)
            zero_fraction = np.sum(actual == 0) / len(actual)
            if zero_fraction > 0.1:
                actual = np.where(actual == 0, np.nan, actual)

            # Compute z-scores
            residuals = actual - predictions
            z_scores = np.where(stds > 0, residuals / stds, 0.0)

            # Mask non-operating periods (set to NaN)
            max_power = df[power_col].max()
            power_threshold = max_power * 0.05
            non_operating = df[power_col].values < power_threshold
            z_scores[non_operating] = np.nan

            # Also mask where status is not normal
            z_scores[~status_mask] = np.nan

            all_z_scores[col] = z_scores

            # Update sensor stats
            valid_z = z_scores[~np.isnan(z_scores)]
            if len(valid_z) > 0:
                sensor_stats[col].update({
                    'z_mean': float(np.mean(valid_z)),
                    'z_std': float(np.std(valid_z)),
                    'z_max': float(np.max(valid_z)),
                    'z_min': float(np.min(valid_z)),
                    'n_valid': len(valid_z)
                })

        # =====================================================================
        # STAGE 3: Sliding window detection with z-score consensus
        # =====================================================================
        n_rows = len(df)
        results = []

        window_start = test_start_idx

        while window_start + v6_window_size <= n_rows:
            window_end = window_start + v6_window_size

            # Get z-scores for all sensors in this window
            sensor_scores = {}

            for col, z_scores in all_z_scores.items():
                window_z = z_scores[window_start:window_end]
                valid = ~np.isnan(window_z)

                if np.sum(valid) < 50:  # Need meaningful data
                    continue

                # Mean z-score in window (only positive matters - running hotter)
                mean_z = np.mean(window_z[valid])
                sensor_scores[col] = max(0.0, mean_z)

            # =====================================================================
            # v6.1: Enhanced scoring with trend and variance components
            # Pure mean_z isn't enough - many CARE faults don't show thermal shift
            # =====================================================================
            trend_scores = {}
            var_scores = {}

            for col, z_scores_arr in all_z_scores.items():
                window_z = z_scores_arr[window_start:window_end]
                valid = ~np.isnan(window_z)

                if np.sum(valid) < 50:
                    continue

                wz = window_z[valid]

                # Trend: are z-scores increasing over the window?
                if len(wz) >= 100:
                    try:
                        x = np.arange(len(wz))
                        slope, _, _, p_value, _ = stats.linregress(x, wz)
                        # Normalize: slope is z-units per sample
                        # Convert to z-units per window (meaningful magnitude)
                        slope_per_window = slope * len(wz)
                        if slope > 0 and p_value < 0.05:
                            trend_scores[col] = max(0, slope_per_window)
                    except Exception:
                        pass

                # Variance: compare to training variance (should be ~1 for z-scores)
                # If variance >> 1, the signal is more variable than training
                window_var = np.var(wz)
                if window_var > 1.5:
                    var_scores[col] = (window_var - 1.5) / 2.0  # Normalize

            # Multi-sensor consensus score (v6.1 enhanced formula)
            if not sensor_scores:
                window_score = 0.0
                n_anomalous = 0
            else:
                scores = list(sensor_scores.values())

                # Weighted combination:
                # - max sensor score (catches single-component faults)
                # - mean of top-3 scores (catches systemic degradation)
                max_score = max(scores)
                sorted_scores = sorted(scores, reverse=True)
                top3_mean = np.mean(sorted_scores[:3]) if len(scores) >= 3 else max_score

                # v6 base: 0.7 * max + 0.3 * top3_mean
                base_score = 0.7 * max_score + 0.3 * top3_mean

                # v6.1: Add trend and variance boosts
                trend_boost = 0.0
                var_boost = 0.0

                if trend_scores:
                    max_trend = max(trend_scores.values())
                    trend_boost = min(0.5, max_trend * 0.5)  # Cap at 0.5 z-units

                if var_scores:
                    max_var = max(var_scores.values())
                    var_boost = min(0.3, max_var * 0.3)  # Cap at 0.3 z-units

                window_score = base_score + trend_boost + var_boost

                # Count sensors above threshold
                n_anomalous = sum(1 for s in scores if s >= z_threshold)

            # Build signal results
            signal_results = []
            for col, score in sensor_scores.items():
                signal_results.append(SignalAnomaly(
                    signal_name=col,
                    anomaly_score=score / z_threshold if z_threshold > 0 else score,  # Normalize to 0-1 scale
                    trend_score=0.0,  # v6 doesn't compute trend per-window
                    volatility_score=0.0,
                    periodic_score=0.0,
                    periodic_degraded=False
                ))

            results.append(WindowResult(
                window_start=window_start,
                window_end=window_end,
                overall_score=window_score,
                n_signals_anomalous=n_anomalous,
                n_periodic_degraded=0,  # v6 doesn't use periodic analysis
                signal_results=signal_results,
                periodic_baseline_used=False
            ))

            window_start += v6_step_size

        return results, nbm_models, sensor_stats

    def results_to_predictions_v6(
        self,
        results: List['WindowResult'],
        n_total: int,
        z_threshold: float = 2.0
    ) -> np.ndarray:
        """
        Convert v6 window results to per-timestamp binary predictions.

        Args:
            results: List of WindowResult from detect_v6
            n_total: Total number of timestamps
            z_threshold: Z-score threshold for flagging (default 2.0)

        Returns:
            Binary array of predictions
        """
        # v6 step size: 6 hours = 36 samples at 10-min intervals
        v6_step_size = 6 * self.samples_per_hour

        predictions = np.zeros(n_total, dtype=int)

        for result in results:
            if result.overall_score >= z_threshold:
                # Mark the step interval as anomaly
                step_end = min(result.window_start + v6_step_size, n_total)
                predictions[result.window_start:step_end] = 1

        return predictions

    # =========================================================================
    # V7 DETECTION (Phase 1: Non-Temperature Signals)
    # =========================================================================

    def compute_non_temp_z_scores(
        self,
        df: pd.DataFrame,
        train_mask: np.ndarray,
        status_mask: np.ndarray,
        non_temp_sensors: Dict[str, List[str]],
        farm: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """
        Compute z-scores for non-temperature signals.

        Unlike temperature signals, these don't need NBM (no power/ambient
        correction needed). Just z-score against training mean/std.

        For vibration signals, use variance z-score (fault → increased
        variability, not necessarily mean shift).

        Args:
            df: Full DataFrame with all data
            train_mask: Boolean mask for training rows
            status_mask: Boolean mask for normal status rows
            non_temp_sensors: Dict mapping category to list of sensor columns
            farm: Farm identifier

        Returns:
            Tuple of:
            - Dict mapping column name to z-score array
            - Dict mapping column name to training stats
        """
        combined_mask = train_mask & status_mask

        z_scores = {}
        train_stats = {}

        # Categories that use variance ratio instead of mean shift
        variance_categories = {'vibration'}

        for category, cols in non_temp_sensors.items():
            for col in cols:
                if col not in df.columns:
                    continue

                values = df[col].values.astype(float)

                # Handle zeros as missing (common in Farms B/C)
                zero_frac = np.sum(values == 0) / len(values)
                if zero_frac > 0.1:
                    values = np.where(values == 0, np.nan, values)

                # Compute training statistics
                train_vals = values[combined_mask]
                train_valid = train_vals[~np.isnan(train_vals)]

                if len(train_valid) < 100:
                    continue

                train_mean = float(np.mean(train_valid))
                train_std = float(np.std(train_valid))
                train_var = float(np.var(train_valid))

                if train_std < 1e-6:
                    continue

                train_stats[col] = {
                    'mean': train_mean,
                    'std': train_std,
                    'variance': train_var,
                    'category': category,
                    'n_samples': len(train_valid)
                }

                # Compute z-scores for entire dataset
                z = (values - train_mean) / train_std
                z[np.isnan(values)] = np.nan
                z[~status_mask] = np.nan

                z_scores[col] = z

        # Special case: pitch divergence (inter-blade variation)
        pitch_cols = non_temp_sensors.get('pitch_position', [])
        available_pitch = [c for c in pitch_cols if c in df.columns]

        if len(available_pitch) >= 2:
            # Compute per-timestamp std across blades (measures divergence)
            blade_std = df[available_pitch].std(axis=1).values

            # Training statistics for blade divergence
            train_div = blade_std[combined_mask]
            train_div_valid = train_div[~np.isnan(train_div)]

            if len(train_div_valid) > 100:
                div_mean = float(np.mean(train_div_valid))
                div_std = float(np.std(train_div_valid))

                if div_std > 1e-6:
                    train_stats['pitch_divergence'] = {
                        'mean': div_mean,
                        'std': div_std,
                        'variance': div_std ** 2,
                        'category': 'pitch_divergence',
                        'n_samples': len(train_div_valid)
                    }

                    # Z-score for divergence
                    div_z = (blade_std - div_mean) / div_std
                    div_z[~status_mask] = np.nan
                    z_scores['pitch_divergence'] = div_z

        return z_scores, train_stats

    def detect_v7(
        self,
        df: pd.DataFrame,
        temp_signal_cols: List[str],
        non_temp_sensors: Dict[str, List[str]],
        farm: str,
        train_mask: np.ndarray,
        test_start_idx: int,
        power_col: Optional[str] = None,
        ambient_col: Optional[str] = None,
        z_threshold: float = 2.0,
        phase: int = 1
    ) -> Tuple[List['WindowResult'], Dict[str, 'NBMModel'], Dict[str, Dict]]:
        """
        v7 Detection: Phase 1 - Non-temperature signals added to v6 NBM approach.

        This extends v6 by adding non-temperature signals (vibration, hydraulic,
        pitch, gearbox oil, motor current) to the consensus scoring.

        Per Brief v7 Phase 1:
        - Keep v6's temperature NBM z-scores
        - Add non-temperature z-scores (simple z vs training)
        - Vibration: variance ratio > 2.0x instead of mean shift
        - Pitch: inter-blade divergence score
        - Same consensus: 0.7 * max + 0.3 * top3_mean across ALL sensors

        Args:
            df: Full DataFrame with all data
            temp_signal_cols: Temperature signal columns for NBM analysis
            non_temp_sensors: Dict mapping category to list of non-temp sensor columns
            farm: Farm identifier ('A', 'B', 'C' or 'Wind Farm A', etc.)
            train_mask: Boolean mask for training rows
            test_start_idx: Index where test/prediction period starts
            power_col: Power column (auto-detected if None)
            ambient_col: Ambient temperature column (auto-detected if None)
            z_threshold: Z-score threshold for anomaly detection (default 2.0)
            phase: Implementation phase (1-4, controls which features are enabled)

        Returns:
            Tuple of:
            - List of WindowResult with combined consensus scores
            - Dict mapping signal names to their NBMModel (temp sensors only)
            - Dict with per-sensor statistics for diagnostics
        """
        # v7 window parameters (same as v6: 3 days, 6 hours step)
        v7_window_hours = 3 * 24
        v7_step_hours = 6
        samples_per_hour = self.samples_per_hour

        v7_window_size = v7_window_hours * samples_per_hour
        v7_step_size = v7_step_hours * samples_per_hour

        # Status mask (only normal operation)
        status_mask = df['status_type_id'].isin([0, 2]).values

        # Get operational columns
        op_cols = get_operational_columns(farm, df)
        power_col = power_col or op_cols.get('power')
        ambient_col = ambient_col or op_cols.get('ambient')

        if power_col is None or ambient_col is None:
            return [], {}, {}

        # =====================================================================
        # STAGE 1: Build temperature NBMs (same as v6)
        # =====================================================================
        combined_mask = train_mask & status_mask
        train_df = df[combined_mask].copy()

        nbm_models: Dict[str, NBMModel] = {}
        sensor_stats: Dict[str, Dict] = {}

        for col in temp_signal_cols:
            if col not in df.columns or col == ambient_col:
                continue

            nbm = build_nbm(
                train_df, col, power_col, ambient_col,
                n_power_bins=20, n_ambient_bins=10
            )

            if nbm is not None:
                nbm_models[col] = nbm
                sensor_stats[col] = {
                    'type': 'temperature',
                    'n_bins_populated': len(nbm.bin_medians),
                    'global_median': nbm.global_median,
                    'global_std': nbm.global_std,
                    'n_training_samples': nbm.n_training_samples
                }

        # =====================================================================
        # STAGE 2a: Compute temperature z-scores (same as v6)
        # =====================================================================
        temp_z_scores: Dict[str, np.ndarray] = {}

        for col, nbm in nbm_models.items():
            predictions, stds = predict_nbm(df, nbm, power_col, ambient_col, return_stds=True)
            actual = df[col].values.astype(float)

            # Handle zeros as missing
            zero_fraction = np.sum(actual == 0) / len(actual)
            if zero_fraction > 0.1:
                actual = np.where(actual == 0, np.nan, actual)

            # Compute z-scores
            residuals = actual - predictions
            z = np.where(stds > 0, residuals / stds, 0.0)

            # Mask non-operating periods
            max_power = df[power_col].max()
            power_threshold = max_power * 0.05
            non_operating = df[power_col].values < power_threshold
            z[non_operating] = np.nan
            z[~status_mask] = np.nan

            temp_z_scores[col] = z

            # Update stats
            valid_z = z[~np.isnan(z)]
            if len(valid_z) > 0:
                sensor_stats[col].update({
                    'z_mean': float(np.mean(valid_z)),
                    'z_std': float(np.std(valid_z)),
                    'z_max': float(np.max(valid_z)),
                    'n_valid': len(valid_z)
                })

        # =====================================================================
        # STAGE 2b: Compute non-temperature z-scores (NEW in v7 Phase 1)
        # =====================================================================
        non_temp_z_scores, non_temp_stats = self.compute_non_temp_z_scores(
            df, train_mask, status_mask, non_temp_sensors, farm
        )

        # Merge stats
        for col, stats in non_temp_stats.items():
            sensor_stats[col] = {
                'type': 'non_temperature',
                **stats
            }

        # =====================================================================
        # STAGE 2c: Apply rolling z-scores (NEW in v7 Phase 2)
        # =====================================================================
        # Phase 2 fix: Replace static z-scores with rolling z-scores to
        # address seasonal drift (Farm A inversion, Farm B elevation).
        # The rolling baseline adapts to recent conditions, removing
        # systematic offsets that affect both normal and anomaly classes.
        if phase >= 2:
            # Rolling window: 30 days at 10-min resolution
            # samples_per_hour = 6 (10-min samples)
            rolling_window = 30 * 24 * samples_per_hour  # 4320 samples

            # Apply to temperature z-scores
            # We use the raw residuals (actual - predicted) for rolling z-score computation
            for col, nbm in nbm_models.items():
                if col not in temp_z_scores:
                    continue

                # Get the residuals (need to recompute since we only stored z-scores)
                predictions, stds = predict_nbm(df, nbm, power_col, ambient_col, return_stds=True)
                actual = df[col].values.astype(float)

                zero_fraction = np.sum(actual == 0) / len(actual)
                if zero_fraction > 0.1:
                    actual = np.where(actual == 0, np.nan, actual)

                residuals = actual - predictions

                # Mask non-operating periods before computing rolling z-scores
                max_power = df[power_col].max()
                power_threshold = max_power * 0.05
                non_operating = df[power_col].values < power_threshold
                residuals[non_operating] = np.nan
                residuals[~status_mask] = np.nan

                # Compute rolling z-scores
                fallback_std = nbm.global_std if nbm.global_std > 0 else 1.0
                rolling_z = compute_rolling_z_scores(
                    residuals,
                    test_start_idx,
                    rolling_window=rolling_window,
                    min_samples=500,
                    fallback_std=fallback_std
                )

                temp_z_scores[col] = rolling_z

                # Update stats for rolling z-scores
                valid_z = rolling_z[~np.isnan(rolling_z)]
                if len(valid_z) > 0:
                    sensor_stats[col].update({
                        'z_mean': float(np.mean(valid_z)),
                        'z_std': float(np.std(valid_z)),
                        'z_max': float(np.max(valid_z)),
                        'n_valid': len(valid_z),
                        'rolling_baseline': True
                    })

            # Apply to non-temperature z-scores
            # For non-temp, the "residual" is just (value - training_mean)
            for col, z_arr in non_temp_z_scores.items():
                if col not in non_temp_stats:
                    continue

                stats = non_temp_stats[col]
                train_mean = stats.get('mean', 0.0)
                train_std = stats.get('std', 1.0)

                # Get the centered values (before z-scoring)
                if col == 'pitch_divergence':
                    # Special case: pitch divergence was computed differently
                    pitch_cols = non_temp_sensors.get('pitch_position', [])
                    available_pitch = [c for c in pitch_cols if c in df.columns]
                    if len(available_pitch) >= 2:
                        centered = df[available_pitch].std(axis=1).values - train_mean
                    else:
                        continue
                else:
                    if col not in df.columns:
                        continue
                    values = df[col].values.astype(float)
                    zero_frac = np.sum(values == 0) / len(values)
                    if zero_frac > 0.1:
                        values = np.where(values == 0, np.nan, values)
                    centered = values - train_mean

                # Mask non-operating periods
                centered[~status_mask] = np.nan

                # Compute rolling z-scores
                rolling_z = compute_rolling_z_scores(
                    centered,
                    test_start_idx,
                    rolling_window=rolling_window,
                    min_samples=500,
                    fallback_std=train_std if train_std > 0 else 1.0
                )

                non_temp_z_scores[col] = rolling_z

                # Update stats
                valid_z = rolling_z[~np.isnan(rolling_z)]
                if len(valid_z) > 0:
                    sensor_stats[col].update({
                        'z_mean': float(np.mean(valid_z)),
                        'z_std': float(np.std(valid_z)),
                        'z_max': float(np.max(valid_z)),
                        'n_valid': len(valid_z),
                        'rolling_baseline': True
                    })

        # =====================================================================
        # STAGE 2d: Build correlation baseline (NEW in v7 Phase 3)
        # =====================================================================
        # Phase 3 fix: Add correlation monitoring to catch multivariate anomalies.
        # A degrading component changes its relationship to other components
        # BEFORE it shows absolute temperature rise.
        corr_baseline: Optional[CorrelationBaseline] = None
        if phase >= 3:
            # Get all sensors for correlation analysis
            all_sensor_cols = list(temp_z_scores.keys()) + list(non_temp_z_scores.keys())

            # Build baseline from training data
            corr_baseline = build_correlation_baseline(
                train_df,
                all_sensor_cols,
                min_correlation=0.5,
                rolling_window_samples=432,  # 3 days
                rolling_step_samples=72       # 12 hours
            )

            if corr_baseline is not None:
                sensor_stats['_correlation'] = {
                    'n_sensors': len(corr_baseline.sensors),
                    'n_stable_pairs': corr_baseline.n_stable_pairs,
                    'min_correlation': corr_baseline.min_correlation
                }

        # =====================================================================
        # STAGE 3: Sliding window detection with tiered consensus
        # =====================================================================
        # v7 Phase 1 approach: Temperature-first with non-temp boost
        # - Temperature signals are primary (NBM-calibrated, more reliable)
        # - Non-temp signals are secondary (can boost borderline cases)
        # - This prevents non-temp noise from dominating the score
        n_rows = len(df)
        results = []

        # Categories that use variance ratio instead of mean shift
        variance_categories = {'vibration'}

        # Borderline threshold: below z_threshold but showing some elevation
        borderline_threshold = z_threshold * 0.75  # e.g., 1.5 if z_threshold=2.0

        window_start = test_start_idx

        while window_start + v7_window_size <= n_rows:
            window_end = window_start + v7_window_size

            signal_results = []

            # --- Temperature sensor scores (positive z only - running hotter) ---
            temp_scores = []
            for col, z_arr in temp_z_scores.items():
                window_z = z_arr[window_start:window_end]
                valid = ~np.isnan(window_z)

                if np.sum(valid) < 50:
                    continue

                mean_z = np.mean(window_z[valid])
                score = max(0.0, mean_z)  # Only positive (hotter than expected)
                temp_scores.append(score)

                signal_results.append(SignalAnomaly(
                    signal_name=col,
                    anomaly_score=score / z_threshold if z_threshold > 0 else score,
                    trend_score=0.0,
                    volatility_score=0.0,
                    periodic_score=0.0,
                    periodic_degraded=False
                ))

            # --- Non-temperature sensor scores ---
            non_temp_scores = []
            for col, z_arr in non_temp_z_scores.items():
                window_z = z_arr[window_start:window_end]
                valid = ~np.isnan(window_z)

                if np.sum(valid) < 50:
                    continue

                # Get category for this sensor
                category = non_temp_stats.get(col, {}).get('category', '')

                if category in variance_categories:
                    # Vibration: use variance ratio
                    window_var = np.var(window_z[valid])
                    if window_var > 2.0:
                        score = (window_var - 1.0)
                    else:
                        score = 0.0
                else:
                    # Mean shift (abs value - both directions matter)
                    mean_z = np.mean(window_z[valid])
                    score = max(0.0, abs(mean_z))

                non_temp_scores.append(score)

                signal_results.append(SignalAnomaly(
                    signal_name=col,
                    anomaly_score=score / z_threshold if z_threshold > 0 else score,
                    trend_score=0.0,
                    volatility_score=score if category in variance_categories else 0.0,
                    periodic_score=0.0,
                    periodic_degraded=False
                ))

            # ===== MULTI-MODAL OR SCORING =====
            # Three independent detection channels:
            # 1. Temperature channel: standard NBM z-score consensus
            # 2. Non-temp channel: quorum-based (multiple sensors must agree)
            # 3. Correlation channel (Phase 3): decorrelation score

            # Temperature consensus (v6 approach)
            if temp_scores:
                temp_max = max(temp_scores)
                temp_sorted = sorted(temp_scores, reverse=True)
                temp_top3_mean = np.mean(temp_sorted[:3]) if len(temp_scores) >= 3 else temp_max
                temp_consensus = 0.7 * temp_max + 0.3 * temp_top3_mean
            else:
                temp_consensus = 0.0

            # Non-temp quorum scoring
            # Require at least 2 non-temp sensors above a moderate threshold
            # to reduce false positives from single noisy sensors
            non_temp_quorum_threshold = z_threshold * 0.8  # Slightly lower for individual sensors
            non_temp_above_threshold = sum(1 for s in non_temp_scores if s >= non_temp_quorum_threshold)
            non_temp_quorum_met = non_temp_above_threshold >= 2 and len(non_temp_scores) >= 3

            if non_temp_scores:
                non_temp_max = max(non_temp_scores)
                non_temp_sorted = sorted(non_temp_scores, reverse=True)
                non_temp_top3_mean = np.mean(non_temp_sorted[:3]) if len(non_temp_scores) >= 3 else non_temp_max
                non_temp_consensus = 0.7 * non_temp_max + 0.3 * non_temp_top3_mean
            else:
                non_temp_consensus = 0.0
                non_temp_quorum_met = False

            # --- Correlation channel (Phase 3) ---
            decorr_score = 0.0
            decorr_normalized = 0.0  # Initialize before conditionals
            decorr_result: Optional[DecorrelationResult] = None
            if phase >= 3 and corr_baseline is not None and corr_baseline.n_stable_pairs > 0:
                window_df = df.iloc[window_start:window_end]
                decorr_result = compute_decorrelation_score(
                    window_df,
                    corr_baseline,
                    z_threshold=1.5  # Decorrelation z-threshold
                )
                decorr_score = decorr_result.score

                # Normalize decorr_score to same scale as z-scores
                # decorr_score of 3.0 (max_decorr + 2*breadth) is "significant"
                # We want this to map to roughly z=2.0
                decorr_normalized = decorr_score / 1.5  # So 3.0 -> 2.0

            # Multi-modal OR: use the higher of the channels
            # BUT only use non-temp if quorum is met (reduces false positives)
            # Phase 3: Add correlation channel as third path
            if temp_consensus >= z_threshold:
                # Temperature detected → flag
                window_score = temp_consensus
            elif non_temp_quorum_met and non_temp_consensus >= z_threshold:
                # Non-temp quorum met and consensus high → flag
                window_score = non_temp_consensus
            elif phase >= 3 and decorr_normalized >= z_threshold:
                # Correlation breakdown detected → flag (Phase 3)
                window_score = decorr_normalized
            elif temp_consensus >= borderline_threshold and non_temp_consensus >= borderline_threshold:
                # Both borderline → combine for boost
                window_score = 0.5 * (temp_consensus + non_temp_consensus)
            elif phase >= 3 and temp_consensus >= borderline_threshold and decorr_normalized >= borderline_threshold:
                # Temperature + correlation both borderline → combine (Phase 3)
                window_score = 0.5 * (temp_consensus + decorr_normalized)
            else:
                # Use the higher score (but won't flag unless one is high)
                base_score = max(temp_consensus, non_temp_consensus * 0.8)
                if phase >= 3:
                    # Consider decorrelation as well, with slight discount
                    base_score = max(base_score, decorr_normalized * 0.8)
                window_score = base_score

            n_anomalous = sum(1 for s in temp_scores if s >= z_threshold)

            # Track correlation metrics for diagnostics
            n_broken = decorr_result.n_broken if decorr_result else 0
            broken_pairs_info = decorr_result.broken_pairs if decorr_result else []

            results.append(WindowResult(
                window_start=window_start,
                window_end=window_end,
                overall_score=window_score,
                n_signals_anomalous=n_anomalous,
                n_periodic_degraded=n_broken,  # Repurposing this field for broken correlations
                signal_results=signal_results,
                periodic_baseline_used=(phase >= 3 and corr_baseline is not None)
            ))

            window_start += v7_step_size

        return results, nbm_models, sensor_stats

    def results_to_predictions_v7(
        self,
        results: List['WindowResult'],
        n_total: int,
        z_threshold: float = 2.0
    ) -> np.ndarray:
        """
        Convert v7 window results to per-timestamp binary predictions.

        Same as v6 - uses 6-hour step size.
        """
        v7_step_size = 6 * self.samples_per_hour
        predictions = np.zeros(n_total, dtype=int)

        for result in results:
            if result.overall_score >= z_threshold:
                step_end = min(result.window_start + v7_step_size, n_total)
                predictions[result.window_start:step_end] = 1

        return predictions

    def detect_v8(
        self,
        df: pd.DataFrame,
        temp_signal_cols: List[str],
        farm: str,
        train_mask: np.ndarray,
        test_start_idx: int,
        power_col: Optional[str] = None,
        ambient_col: Optional[str] = None,
        wind_col: Optional[str] = None,
        z_threshold: float = 2.0
    ) -> Tuple[List['WindowResult'], Dict[str, 'NBMModel'], Dict[str, Dict]]:
        """
        v8 Detection: Ensemble with Power Curve Residual.

        This extends v7 Phase 2 (rolling baseline) by adding power curve residual
        as a second detection channel. Power curve deviations are a universal fault
        indicator - almost any drivetrain fault eventually affects power output.

        Per Brief v8 Session 1:
        - Keep v7 Phase 2's rolling temperature z-scores
        - Add power curve residual z-score as second channel
        - Fusion: max(temp_z, power_z) - anomaly if either channel high

        Args:
            df: Full DataFrame with all data
            temp_signal_cols: Temperature signal columns for NBM analysis
            farm: Farm identifier ('A', 'B', 'C' or 'Wind Farm A', etc.)
            train_mask: Boolean mask for training rows
            test_start_idx: Index where test/prediction period starts
            power_col: Power column (auto-detected if None)
            ambient_col: Ambient temperature column (auto-detected if None)
            wind_col: Wind speed column (auto-detected if None)
            z_threshold: Z-score threshold for anomaly detection (default 2.0)

        Returns:
            Tuple of:
            - List of WindowResult with fused ensemble scores
            - Dict mapping signal names to their NBMModel (temp sensors only)
            - Dict with per-sensor statistics for diagnostics
        """
        # v8 window parameters (same as v7: 3 days, 6 hours step)
        v8_window_hours = 3 * 24
        v8_step_hours = 6
        samples_per_hour = self.samples_per_hour

        v8_window_size = v8_window_hours * samples_per_hour
        v8_step_size = v8_step_hours * samples_per_hour

        # Rolling window for z-score baseline (30 days)
        rolling_window = 30 * 24 * samples_per_hour  # 30 days in samples

        # Status mask (only normal operation)
        status_mask = df['status_type_id'].isin([0, 2]).values

        # Get operational columns
        op_cols = get_operational_columns(farm, df)
        power_col = power_col or op_cols.get('power')
        ambient_col = ambient_col or op_cols.get('ambient')
        wind_col = wind_col or op_cols.get('wind')

        if power_col is None or ambient_col is None:
            return [], {}, {}

        # =====================================================================
        # STAGE 1: Build temperature NBMs (same as v7)
        # =====================================================================
        combined_mask = train_mask & status_mask
        train_df = df[combined_mask].copy()

        nbm_models: Dict[str, NBMModel] = {}
        sensor_stats: Dict[str, Dict] = {}

        for col in temp_signal_cols:
            if col not in df.columns or col == ambient_col:
                continue

            nbm = build_nbm(
                train_df, col, power_col, ambient_col,
                n_power_bins=20, n_ambient_bins=10
            )

            if nbm is not None:
                nbm_models[col] = nbm
                sensor_stats[col] = {
                    'type': 'temperature',
                    'n_bins_populated': len(nbm.bin_medians),
                    'global_median': nbm.global_median,
                    'global_std': nbm.global_std,
                    'n_training_samples': nbm.n_training_samples
                }

        # =====================================================================
        # STAGE 2a: Compute temperature z-scores with rolling baseline
        # =====================================================================
        temp_z_scores: Dict[str, np.ndarray] = {}

        for col, nbm in nbm_models.items():
            predictions, stds = predict_nbm(df, nbm, power_col, ambient_col, return_stds=True)
            actual = df[col].values.astype(float)

            # Handle zeros as missing
            zero_frac = np.sum(actual == 0) / len(actual)
            if zero_frac > 0.1:
                actual = np.where(actual == 0, np.nan, actual)

            # Compute residuals
            residuals = actual - predictions

            # Mask non-operating and non-normal periods
            residuals[~status_mask] = np.nan

            # Compute rolling z-scores (v7 Phase 2)
            fallback_std = nbm.global_std if nbm.global_std > 0 else 1.0
            rolling_z = compute_rolling_z_scores(
                residuals,
                test_start_idx,
                rolling_window=rolling_window,
                min_samples=500,
                fallback_std=fallback_std
            )

            temp_z_scores[col] = rolling_z

            # Update stats
            valid_z = rolling_z[~np.isnan(rolling_z)]
            if len(valid_z) > 0:
                sensor_stats[col].update({
                    'z_mean': float(np.mean(valid_z)),
                    'z_std': float(np.std(valid_z)),
                    'z_max': float(np.max(valid_z)),
                    'n_valid': len(valid_z),
                    'rolling_baseline': True
                })

        # =====================================================================
        # STAGE 2b: Build power curve and compute residual z-scores (NEW in v8)
        # =====================================================================
        power_curve_z_scores: Optional[np.ndarray] = None
        power_curve_model: Optional[PowerCurveModel] = None

        if wind_col is not None and wind_col in df.columns:
            # Build power curve from training data
            power_curve_model = build_power_curve(
                train_df, power_col, wind_col, n_bins=20
            )

            if power_curve_model is not None:
                # Compute power curve residuals
                pc_residuals = power_curve_residual(df, power_curve_model, power_col, wind_col)

                # Mask non-operating periods
                pc_residuals[~status_mask] = np.nan

                # Compute baseline stats from training period
                train_residuals = pc_residuals[train_mask & status_mask]
                valid_train = train_residuals[~np.isnan(train_residuals)]

                if len(valid_train) > 100:
                    pc_baseline_mean = np.mean(valid_train)
                    pc_baseline_std = np.std(valid_train)

                    if pc_baseline_std > 0:
                        # Compute rolling z-scores for power curve residual
                        # For power curve, negative residual = underperformance = anomaly
                        # So we take abs() or use signed z-score and check for negative
                        # Actually for faults, power typically DROPS, so residual is negative
                        # We care about the magnitude of deviation

                        # Use rolling z-score same as temperature
                        power_curve_z_scores = compute_rolling_z_scores(
                            pc_residuals,
                            test_start_idx,
                            rolling_window=rolling_window,
                            min_samples=500,
                            fallback_std=pc_baseline_std
                        )

                        # For power curve, negative z-score (underperformance) is anomaly
                        # Convert to absolute for fusion (both directions matter)
                        # But underperformance is more indicative, so weight it
                        if power_curve_z_scores is not None:
                            # Take absolute value - both under and over are anomalous
                            # But weight underperformance (negative) more
                            power_curve_z_scores = np.abs(power_curve_z_scores)

                        sensor_stats['_power_curve'] = {
                            'type': 'power_curve',
                            'baseline_mean': float(pc_baseline_mean),
                            'baseline_std': float(pc_baseline_std),
                            'rated_power': float(power_curve_model.rated_power),
                            'n_wind_bins': power_curve_model.n_bins,
                            'rolling_baseline': True
                        }

                        # Add z-score stats
                        valid_pc_z = power_curve_z_scores[~np.isnan(power_curve_z_scores)]
                        if len(valid_pc_z) > 0:
                            sensor_stats['_power_curve'].update({
                                'z_mean': float(np.mean(valid_pc_z)),
                                'z_std': float(np.std(valid_pc_z)),
                                'z_max': float(np.max(valid_pc_z)),
                                'n_valid': len(valid_pc_z)
                            })

        # =====================================================================
        # STAGE 2c: PCA Reconstruction Error (NEW in v8 Session 2)
        # =====================================================================
        # PCA captures multivariate sensor correlations. When a component degrades,
        # it breaks the correlation structure, causing higher reconstruction error.
        pca_z_scores: Optional[np.ndarray] = None
        pca_baseline: Optional[PCABaseline] = None

        # Need at least 3 sensors to do meaningful PCA
        if len(nbm_models) >= 3:
            # Collect all temperature sensor columns that have valid NBMs
            pca_sensor_cols = list(nbm_models.keys())

            # Build sensor matrix from training data (standardized)
            train_idx = np.where(train_mask & status_mask)[0]

            if len(train_idx) >= 500:
                # Build training matrix: rows = samples, cols = sensors
                train_matrix = []
                for col in pca_sensor_cols:
                    col_data = df[col].values[train_idx].astype(float)
                    # Replace zeros with NaN if too many zeros
                    zero_frac = np.sum(col_data == 0) / len(col_data)
                    if zero_frac > 0.1:
                        col_data = np.where(col_data == 0, np.nan, col_data)
                    train_matrix.append(col_data)

                train_matrix = np.array(train_matrix).T  # (n_samples, n_sensors)

                # Find rows with all valid values
                valid_rows = ~np.any(np.isnan(train_matrix), axis=1)
                train_matrix_clean = train_matrix[valid_rows]

                if len(train_matrix_clean) >= 500:
                    # Standardize the training data
                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(train_matrix_clean)

                    # Fit PCA keeping components that explain 95% variance
                    pca = PCA(n_components=0.95, svd_solver='full')
                    pca.fit(train_scaled)

                    # Compute reconstruction error on training data
                    train_transformed = pca.transform(train_scaled)
                    train_reconstructed = pca.inverse_transform(train_transformed)
                    train_errors = np.sqrt(np.sum((train_scaled - train_reconstructed) ** 2, axis=1))

                    pca_baseline_mean = np.mean(train_errors)
                    pca_baseline_std = np.std(train_errors)

                    if pca_baseline_std > 1e-6:
                        # Store PCA baseline
                        pca_baseline = PCABaseline(
                            pca_model=pca,
                            scaler=scaler,
                            n_components=pca.n_components_,
                            explained_variance_ratio=float(np.sum(pca.explained_variance_ratio_)),
                            sensor_names=pca_sensor_cols,
                            baseline_mean_error=float(pca_baseline_mean),
                            baseline_std_error=float(pca_baseline_std),
                            n_training_samples=len(train_matrix_clean)
                        )

                        # Compute reconstruction error for all data (not just test)
                        # Build full matrix
                        full_matrix = []
                        for col in pca_sensor_cols:
                            col_data = df[col].values.astype(float)
                            zero_frac = np.sum(col_data == 0) / len(col_data)
                            if zero_frac > 0.1:
                                col_data = np.where(col_data == 0, np.nan, col_data)
                            full_matrix.append(col_data)
                        full_matrix = np.array(full_matrix).T

                        # Compute reconstruction error per sample
                        n_total = len(df)
                        recon_errors = np.full(n_total, np.nan)

                        # Process in chunks for efficiency
                        valid_full_rows = ~np.any(np.isnan(full_matrix), axis=1)
                        valid_full_idx = np.where(valid_full_rows & status_mask)[0]

                        if len(valid_full_idx) > 0:
                            full_scaled = scaler.transform(full_matrix[valid_full_idx])
                            full_transformed = pca.transform(full_scaled)
                            full_reconstructed = pca.inverse_transform(full_transformed)
                            errors = np.sqrt(np.sum((full_scaled - full_reconstructed) ** 2, axis=1))
                            recon_errors[valid_full_idx] = errors

                        # Compute rolling z-scores on reconstruction error
                        pca_z_scores = compute_rolling_z_scores(
                            recon_errors,
                            test_start_idx,
                            rolling_window=rolling_window,
                            min_samples=500,
                            fallback_std=pca_baseline_std
                        )

                        # Store stats
                        sensor_stats['_pca'] = {
                            'type': 'pca',
                            'n_components': pca.n_components_,
                            'explained_variance_ratio': float(np.sum(pca.explained_variance_ratio_)),
                            'n_sensors': len(pca_sensor_cols),
                            'baseline_mean_error': float(pca_baseline_mean),
                            'baseline_std_error': float(pca_baseline_std),
                            'n_training_samples': len(train_matrix_clean),
                            'rolling_baseline': True
                        }

                        # Add z-score stats
                        valid_pca_z = pca_z_scores[~np.isnan(pca_z_scores)]
                        if len(valid_pca_z) > 0:
                            sensor_stats['_pca'].update({
                                'z_mean': float(np.mean(valid_pca_z)),
                                'z_std': float(np.std(valid_pca_z)),
                                'z_max': float(np.max(valid_pca_z)),
                                'n_valid': len(valid_pca_z)
                            })

        # =====================================================================
        # STAGE 2d: Changepoint Detection (NEW in v8 Session 3)
        # =====================================================================
        # Detect sudden level shifts in temperature sensors that rolling z-scores
        # may miss because the rolling baseline adapts to the new level.
        # A recent changepoint indicates a potential fault onset.
        changepoint_scores_dict: Dict[str, np.ndarray] = {}
        all_sensor_changepoints: Dict[str, List[int]] = {}

        for col, nbm in nbm_models.items():
            # Get the NBM residual (actual - predicted)
            predictions, _ = predict_nbm(df, nbm, power_col, ambient_col, return_stds=True)
            actual = df[col].values.astype(float)

            # Handle zeros as missing
            zero_frac = np.sum(actual == 0) / len(actual)
            if zero_frac > 0.1:
                actual = np.where(actual == 0, np.nan, actual)

            residuals = actual - predictions

            # Mask non-operating periods
            residuals_masked = residuals.copy()
            residuals_masked[~status_mask] = np.nan

            # Run changepoint detection on residuals
            # Only detect significant UPWARD shifts (temperature increases)
            cp_scores, cp_indices = compute_changepoint_scores(
                residuals_masked,
                test_start_idx,
                samples_per_hour=samples_per_hour,
                lookback_hours=48,        # Look back 2 days for recent changepoints
                min_segment_hours=24,     # Minimum 24-hour segments (conservative)
                penalty_factor=10.0,      # High penalty for fewer false positives
                min_shift_std=2.0         # Only count shifts > 2 std
            )

            changepoint_scores_dict[col] = cp_scores
            all_sensor_changepoints[col] = cp_indices

            # Update sensor stats
            if col in sensor_stats:
                n_changepoints = len([cp for cp in cp_indices if cp >= test_start_idx])
                valid_cp_scores = cp_scores[cp_scores > 0]
                sensor_stats[col].update({
                    'n_changepoints': n_changepoints,
                    'changepoint_max': float(np.max(cp_scores)) if len(valid_cp_scores) > 0 else 0.0,
                    'changepoint_mean': float(np.mean(valid_cp_scores)) if len(valid_cp_scores) > 0 else 0.0
                })

        # Aggregate changepoint statistics
        total_changepoints = sum(len([cp for cp in cps if cp >= test_start_idx])
                                  for cps in all_sensor_changepoints.values())
        sensor_stats['_changepoint'] = {
            'type': 'changepoint',
            'n_sensors_analyzed': len(changepoint_scores_dict),
            'total_changepoints_in_test': total_changepoints,
            'lookback_hours': 72,
            'min_segment_hours': 12
        }

        # =====================================================================
        # STAGE 3: Sliding window detection with ensemble fusion
        # =====================================================================
        n_rows = len(df)
        results = []

        window_start = test_start_idx

        while window_start + v8_window_size <= n_rows:
            window_end = window_start + v8_window_size

            signal_results = []

            # --- Temperature sensor scores (positive z only - running hotter) ---
            temp_scores = []
            for col, z_arr in temp_z_scores.items():
                window_z = z_arr[window_start:window_end]
                valid = ~np.isnan(window_z)

                if np.sum(valid) < 50:
                    continue

                mean_z = np.mean(window_z[valid])
                score = max(0.0, mean_z)  # Only positive (hotter than expected)
                temp_scores.append(score)

                signal_results.append(SignalAnomaly(
                    signal_name=col,
                    anomaly_score=score / z_threshold if z_threshold > 0 else score,
                    trend_score=0.0,
                    volatility_score=0.0,
                    periodic_score=0.0,
                    periodic_degraded=False
                ))

            # Temperature consensus
            if temp_scores:
                temp_max = max(temp_scores)
                temp_sorted = sorted(temp_scores, reverse=True)
                temp_top3_mean = np.mean(temp_sorted[:3]) if len(temp_scores) >= 3 else temp_max
                temp_consensus = 0.7 * temp_max + 0.3 * temp_top3_mean
            else:
                temp_consensus = 0.0

            # --- PCA reconstruction error score (v8 Session 2) ---
            # Replaced power curve (Session 1) with PCA - power curve added too many FPs
            pca_score = 0.0
            if pca_z_scores is not None:
                window_pca_z = pca_z_scores[window_start:window_end]
                valid_pca = ~np.isnan(window_pca_z)

                if np.sum(valid_pca) >= 50:
                    # Use mean z-score in window (reconstruction error is already aggregated)
                    mean_pca_z = np.mean(window_pca_z[valid_pca])
                    max_pca_z = np.max(window_pca_z[valid_pca])

                    # Weight mean more than max since reconstruction error is smoother
                    pca_score = 0.7 * mean_pca_z + 0.3 * max_pca_z

                    # Only count positive (higher reconstruction error = anomaly)
                    pca_score = max(0.0, pca_score)

                    signal_results.append(SignalAnomaly(
                        signal_name='_pca',
                        anomaly_score=pca_score / z_threshold if z_threshold > 0 else pca_score,
                        trend_score=0.0,
                        volatility_score=0.0,
                        periodic_score=0.0,
                        periodic_degraded=False
                    ))

            # --- Changepoint score (v8 Session 3) ---
            # Aggregate changepoint scores across sensors for this window
            changepoint_score = 0.0
            if changepoint_scores_dict:
                cp_scores_in_window = []
                for col, cp_arr in changepoint_scores_dict.items():
                    window_cp = cp_arr[window_start:window_end]
                    # Use max changepoint score in window (any recent shift matters)
                    max_cp_score = np.max(window_cp) if len(window_cp) > 0 else 0.0
                    cp_scores_in_window.append(max_cp_score)

                if cp_scores_in_window:
                    # Use max across sensors (any sensor showing changepoint is suspicious)
                    changepoint_score = max(cp_scores_in_window)

                    # Also track in signal_results
                    if changepoint_score > 0:
                        signal_results.append(SignalAnomaly(
                            signal_name='_changepoint',
                            anomaly_score=changepoint_score / z_threshold if z_threshold > 0 else changepoint_score,
                            trend_score=0.0,
                            volatility_score=0.0,
                            periodic_score=0.0,
                            periodic_degraded=False
                        ))

            # ===== ENSEMBLE FUSION: temp + changepoint boost =====
            # v8 Session 3: Temperature is primary, changepoint provides boost
            # Dropped max fusion (Session 3 v1) - too many false positives
            # Changepoint BOOSTS temperature score when there's a recent upward shift
            # This catches fault onsets that rolling z-scores miss
            cp_boost_factor = 0.3  # Changepoint contribution weight
            window_score = temp_consensus + cp_boost_factor * changepoint_score

            n_anomalous = sum(1 for s in temp_scores if s >= z_threshold)

            results.append(WindowResult(
                window_start=window_start,
                window_end=window_end,
                overall_score=window_score,
                n_signals_anomalous=n_anomalous,
                n_periodic_degraded=0,
                signal_results=signal_results,
                periodic_baseline_used=False
            ))

            window_start += v8_step_size

        return results, nbm_models, sensor_stats

    def results_to_predictions_v8(
        self,
        results: List['WindowResult'],
        n_total: int,
        z_threshold: float = 2.0
    ) -> np.ndarray:
        """
        Convert v8 window results to per-timestamp binary predictions.

        Same as v7 - uses 6-hour step size.
        """
        v8_step_size = 6 * self.samples_per_hour
        predictions = np.zeros(n_total, dtype=int)

        for result in results:
            if result.overall_score >= z_threshold:
                step_end = min(result.window_start + v8_step_size, n_total)
                predictions[result.window_start:step_end] = 1

        return predictions


def find_ambient_sensor(df: pd.DataFrame, farm: str) -> Optional[str]:
    """Find the ambient temperature sensor column for a wind farm."""
    # Common patterns for ambient temperature
    patterns = [
        'sensor_0_avg',      # Farm A: Ambient temperature
        'sensor_8_avg',      # Farm B: Outside temperature
        'sensor_7_avg',      # Farm C: Ambient temperature
        'sensor_41_avg',     # Farm C: Nacelle outside temperature
    ]

    for pattern in patterns:
        if pattern in df.columns:
            return pattern

    # Fallback: search for columns with 'ambient' or 'outside' in description
    for col in df.columns:
        if 'ambient' in col.lower() or 'outside' in col.lower():
            return col

    return None


def demo_detector():
    """Demo the anomaly detector."""
    print("=" * 60)
    print("Anomaly Detector Demo")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    n_days = 365
    n_samples = n_days * 144  # 10-min intervals

    # Normal temperature signal with diurnal pattern
    times = np.arange(n_samples) / 144  # Days
    diurnal = 10 * np.sin(2 * np.pi * times)
    noise = np.random.normal(0, 2, n_samples)
    temp_normal = 50 + diurnal + noise

    # Add anomaly: gradual rise in last 30 days
    anomaly_start = int(n_samples * 0.9)
    temp_anomaly = temp_normal.copy()
    temp_anomaly[anomaly_start:] += np.linspace(0, 15, n_samples - anomaly_start)

    # Create DataFrame
    df = pd.DataFrame({
        'time_stamp': pd.date_range('2022-01-01', periods=n_samples, freq='10min'),
        'status_type_id': np.zeros(n_samples, dtype=int),
        'train_test': ['train'] * int(n_samples * 0.7) + ['test'] * (n_samples - int(n_samples * 0.7)),
        'bearing_temp_avg': temp_anomaly,
        'ambient_temp_avg': 20 + diurnal * 0.5 + np.random.normal(0, 1, n_samples)
    })

    train_mask = (df['train_test'] == 'train').values
    test_start = int(n_samples * 0.7)

    # Run detector
    detector = AnomalyDetector()

    print("\nRunning multi-signal detection...")
    results = detector.detect_multi_signal(
        df,
        ['bearing_temp_avg'],
        train_mask,
        test_start,
        ambient_col='ambient_temp_avg'
    )

    print(f"Analyzed {len(results)} windows")

    # Convert to predictions
    predictions = detector.results_to_predictions(results, n_samples, threshold=0.3)

    n_flagged = np.sum(predictions)
    print(f"Flagged {n_flagged} timestamps as anomaly ({100*n_flagged/n_samples:.1f}%)")

    # Check if we caught the anomaly period
    anomaly_region_preds = predictions[anomaly_start:]
    print(f"Anomaly region detection rate: {100*np.mean(anomaly_region_preds):.1f}%")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    demo_detector()

"""
TorqueScope Phase 2: Periodic Baseline Detector

Uses Lomb-Scargle periodogram to characterize normal periodic structure in SCADA data.
Detects anomalies when the periodic structure degrades from the training baseline.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy import signal
from astropy.timeseries import LombScargle


@dataclass
class PeriodicComponent:
    """A detected periodic component in a signal."""
    frequency: float  # cycles per hour
    period_hours: float
    amplitude: float
    power: float
    phase: float
    cv_percent: float  # Multi-window CV


@dataclass
class PeriodicBaseline:
    """Baseline periodic profile for a signal."""
    signal_name: str
    components: List[PeriodicComponent]
    residual_power_fraction: float  # Fraction of variance not explained by periodics
    total_variance: float
    n_windows: int


@dataclass
class AnomalyScore:
    """Anomaly score for a single window."""
    window_start_idx: int
    window_end_idx: int
    timestamp_start: str
    timestamp_end: str
    score: float  # 0-1 anomaly score
    amplitude_ratio: float  # Ratio of current amplitude to baseline
    cv_change: float  # Ratio of current CV to baseline CV
    residual_ratio: float  # Ratio of current residual to baseline
    signals_anomalous: List[str]  # Which signals flagged anomaly


class PeriodicBaselineDetector:
    """
    Detector for periodic structure degradation in SCADA data.

    Uses Lomb-Scargle periodogram to:
    1. Build baseline periodic profile from training data
    2. Detect anomalies when periodic structure degrades
    """

    # Target periods for wind turbine SCADA (in hours)
    TARGET_PERIODS = {
        'diurnal': 24.0,        # Daily cycle
        'semi_diurnal': 12.0,   # Half-day (tidal for offshore)
        'synoptic_low': 72.0,   # Weather front (3 days)
        'synoptic_high': 168.0, # Weather system (7 days)
    }

    def __init__(
        self,
        window_size_hours: float = 168.0,  # 7 days
        step_size_hours: float = 24.0,     # 1 day
        n_baseline_windows: int = 4,       # For CV calculation
        min_frequency: float = 1/720,      # 30 day period
        max_frequency: float = 0.5,        # 2 hour period
        n_freq_points: int = 5000,
        amplitude_threshold: float = 0.5,  # Flag if amplitude < 50% or > 200%
        cv_threshold: float = 2.0,         # Flag if CV doubles
        residual_threshold: float = 1.5,   # Flag if residual increases 50%
    ):
        self.window_size_hours = window_size_hours
        self.step_size_hours = step_size_hours
        self.n_baseline_windows = n_baseline_windows
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.n_freq_points = n_freq_points
        self.amplitude_threshold = amplitude_threshold
        self.cv_threshold = cv_threshold
        self.residual_threshold = residual_threshold

        # Convert to timestamps (10-min intervals)
        self.samples_per_hour = 6
        self.window_size = int(window_size_hours * self.samples_per_hour)
        self.step_size = int(step_size_hours * self.samples_per_hour)

    def _prepare_signal(
        self,
        df: pd.DataFrame,
        signal_col: str,
        status_col: str = 'status_type_id'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare signal for analysis: handle missing values and create time array.

        Args:
            df: DataFrame with signal data
            signal_col: Column name for signal
            status_col: Column name for status

        Returns:
            times: Time array in hours from start
            values: Signal values (NaN for missing/abnormal status)
            valid_mask: Boolean mask for valid data points
        """
        values = df[signal_col].values.copy().astype(float)

        # Mask abnormal status (keep only 0 and 2)
        status = df[status_col].values
        abnormal_mask = ~np.isin(status, [0, 2])
        values[abnormal_mask] = np.nan

        # Handle zeros as missing for non-power signals (common in Farms B/C)
        if not signal_col.startswith('power'):
            # Check if signal has large sequences of zeros
            zero_mask = values == 0
            if np.sum(zero_mask) > len(values) * 0.1:  # More than 10% zeros
                # Only replace zeros if surrounded by non-zeros (likely missing)
                values[zero_mask] = np.nan

        # Create time array in hours
        times = np.arange(len(values)) / self.samples_per_hour

        # Valid mask
        valid_mask = ~np.isnan(values)

        return times, values, valid_mask

    def _run_lomb_scargle(
        self,
        times: np.ndarray,
        values: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run Lomb-Scargle periodogram on signal.

        Returns:
            frequencies: Frequency array (cycles/hour)
            power: Normalized power spectrum
            peak_freq: Frequency of maximum power
        """
        if np.sum(valid_mask) < 100:  # Not enough data
            return None, None, None

        t_valid = times[valid_mask]
        v_valid = values[valid_mask]

        # Remove mean
        v_valid = v_valid - np.nanmean(v_valid)

        # Create frequency grid
        frequencies = np.linspace(self.min_frequency, self.max_frequency, self.n_freq_points)

        # Run Lomb-Scargle
        ls = LombScargle(t_valid, v_valid)
        power = ls.power(frequencies)

        # Find peak
        peak_idx = np.argmax(power)
        peak_freq = frequencies[peak_idx]

        return frequencies, power, peak_freq

    def _extract_periodic_components(
        self,
        frequencies: np.ndarray,
        power: np.ndarray,
        n_components: int = 5
    ) -> List[Tuple[float, float]]:
        """
        Extract top N periodic components from power spectrum.

        Returns:
            List of (frequency, power) tuples
        """
        if frequencies is None:
            return []

        # Find peaks in power spectrum
        peaks, properties = signal.find_peaks(power, height=0.01, distance=50)

        if len(peaks) == 0:
            return []

        # Sort by power
        peak_powers = power[peaks]
        sorted_idx = np.argsort(peak_powers)[::-1]

        components = []
        for i in sorted_idx[:n_components]:
            idx = peaks[i]
            components.append((frequencies[idx], power[idx]))

        return components

    def _compute_multi_window_cv(
        self,
        times: np.ndarray,
        values: np.ndarray,
        valid_mask: np.ndarray,
        target_freq: float
    ) -> float:
        """
        Compute CV of peak frequency across multiple windows.
        """
        window_freqs = []

        n_samples = len(times)
        window_size = n_samples // self.n_baseline_windows

        for i in range(self.n_baseline_windows):
            start = i * window_size
            end = min((i + 1) * window_size, n_samples)

            t_win = times[start:end]
            v_win = values[start:end]
            m_win = valid_mask[start:end]

            if np.sum(m_win) < 50:
                continue

            freqs, power, peak_freq = self._run_lomb_scargle(t_win, v_win, m_win)
            if peak_freq is not None:
                window_freqs.append(peak_freq)

        if len(window_freqs) < 2:
            return 100.0  # High CV = unstable

        cv = np.std(window_freqs) / np.mean(window_freqs) * 100
        return cv

    def build_baseline(
        self,
        df: pd.DataFrame,
        signal_cols: List[str],
        train_mask: np.ndarray
    ) -> Dict[str, PeriodicBaseline]:
        """
        Build periodic baseline from training data.

        Args:
            df: Full DataFrame
            signal_cols: List of signal column names to analyze
            train_mask: Boolean mask for training rows

        Returns:
            Dict mapping signal name to PeriodicBaseline
        """
        baselines = {}

        train_df = df[train_mask].copy()

        for col in signal_cols:
            if col not in train_df.columns:
                continue

            times, values, valid_mask = self._prepare_signal(train_df, col)

            if np.sum(valid_mask) < 200:
                continue

            # Run LS on full training period
            freqs, power, peak_freq = self._run_lomb_scargle(times, values, valid_mask)

            if freqs is None:
                continue

            # Extract top components
            raw_components = self._extract_periodic_components(freqs, power)

            # Convert to PeriodicComponent objects with CV
            components = []
            for freq, pwr in raw_components:
                cv = self._compute_multi_window_cv(times, values, valid_mask, freq)
                period = 1.0 / freq if freq > 0 else float('inf')

                # Estimate amplitude (rough approximation)
                amplitude = np.sqrt(2 * pwr) * np.nanstd(values[valid_mask])

                components.append(PeriodicComponent(
                    frequency=freq,
                    period_hours=period,
                    amplitude=amplitude,
                    power=pwr,
                    phase=0.0,  # Not computed
                    cv_percent=cv
                ))

            # Compute residual power fraction
            total_power = np.sum(power)
            explained_power = sum(c.power for c in components)
            residual_fraction = 1.0 - (explained_power / total_power) if total_power > 0 else 1.0

            baselines[col] = PeriodicBaseline(
                signal_name=col,
                components=components,
                residual_power_fraction=residual_fraction,
                total_variance=np.nanvar(values[valid_mask]),
                n_windows=self.n_baseline_windows
            )

        return baselines

    def _score_window(
        self,
        window_df: pd.DataFrame,
        signal_cols: List[str],
        baselines: Dict[str, PeriodicBaseline]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score a single window for anomalies.

        Returns:
            (overall_score, per_signal_scores)
        """
        signal_scores = {}

        for col in signal_cols:
            if col not in window_df.columns or col not in baselines:
                continue

            baseline = baselines[col]
            if not baseline.components:
                continue

            times, values, valid_mask = self._prepare_signal(window_df, col)

            if np.sum(valid_mask) < 50:
                continue

            # Run LS on window
            freqs, power, peak_freq = self._run_lomb_scargle(times, values, valid_mask)

            if freqs is None:
                signal_scores[col] = 0.0
                continue

            # Compare to baseline
            baseline_peak = baseline.components[0] if baseline.components else None

            anomaly_indicators = []

            if baseline_peak:
                # 1. Amplitude change
                current_amplitude = np.sqrt(2 * np.max(power)) * np.nanstd(values[valid_mask])
                if baseline_peak.amplitude > 0:
                    amp_ratio = current_amplitude / baseline_peak.amplitude
                    if amp_ratio < self.amplitude_threshold or amp_ratio > 1/self.amplitude_threshold:
                        anomaly_indicators.append(1.0)
                    else:
                        anomaly_indicators.append(0.0)

                # 2. CV change
                current_cv = self._compute_multi_window_cv(times, values, valid_mask, peak_freq)
                if baseline_peak.cv_percent > 0:
                    cv_ratio = current_cv / baseline_peak.cv_percent
                    if cv_ratio > self.cv_threshold:
                        anomaly_indicators.append(1.0)
                    else:
                        anomaly_indicators.append(0.0)

            # 3. Residual power change
            components = self._extract_periodic_components(freqs, power)
            total_power = np.sum(power)
            explained = sum(p for _, p in components)
            current_residual = 1.0 - (explained / total_power) if total_power > 0 else 1.0

            if baseline.residual_power_fraction > 0:
                res_ratio = current_residual / baseline.residual_power_fraction
                if res_ratio > self.residual_threshold:
                    anomaly_indicators.append(1.0)
                else:
                    anomaly_indicators.append(0.0)

            # Combine indicators
            if anomaly_indicators:
                signal_scores[col] = np.mean(anomaly_indicators)
            else:
                signal_scores[col] = 0.0

        # Overall score: weighted average across signals
        if signal_scores:
            overall = np.mean(list(signal_scores.values()))
        else:
            overall = 0.0

        return overall, signal_scores

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        signal_cols: List[str],
        baselines: Dict[str, PeriodicBaseline],
        test_start_idx: int
    ) -> List[AnomalyScore]:
        """
        Detect anomalies in prediction period using sliding window.

        Args:
            df: Full DataFrame
            signal_cols: Columns to analyze
            baselines: Pre-computed baselines
            test_start_idx: Where prediction period starts

        Returns:
            List of AnomalyScore for each window
        """
        scores = []

        n_rows = len(df)

        # Sliding window over prediction period
        window_start = test_start_idx

        while window_start + self.window_size <= n_rows:
            window_end = window_start + self.window_size
            window_df = df.iloc[window_start:window_end].copy()

            # Score this window
            overall_score, signal_scores = self._score_window(window_df, signal_cols, baselines)

            # Identify which signals are anomalous
            signals_anomalous = [s for s, score in signal_scores.items() if score > 0.5]

            # Get timestamps
            ts_start = str(df.iloc[window_start]['time_stamp']) if 'time_stamp' in df.columns else ""
            ts_end = str(df.iloc[window_end-1]['time_stamp']) if 'time_stamp' in df.columns else ""

            scores.append(AnomalyScore(
                window_start_idx=window_start,
                window_end_idx=window_end,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
                score=overall_score,
                amplitude_ratio=0.0,  # Simplified
                cv_change=0.0,
                residual_ratio=0.0,
                signals_anomalous=signals_anomalous
            ))

            window_start += self.step_size

        return scores

    def scores_to_predictions(
        self,
        scores: List[AnomalyScore],
        n_total_rows: int,
        test_start_idx: int,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Convert window-level scores to per-timestamp binary predictions.

        Args:
            scores: List of AnomalyScore
            n_total_rows: Total rows in dataset
            test_start_idx: Where prediction period starts
            threshold: Score threshold for anomaly prediction

        Returns:
            Binary prediction array (0 or 1 for each row)
        """
        predictions = np.zeros(n_total_rows, dtype=int)

        for score in scores:
            if score.score >= threshold:
                # Mark all timestamps in this window's step as anomaly
                step_start = score.window_start_idx
                step_end = min(step_start + self.step_size, n_total_rows)
                predictions[step_start:step_end] = 1

        return predictions


def run_periodic_baseline_demo():
    """Demo the periodic baseline detector on synthetic data."""
    print("=" * 60)
    print("Periodic Baseline Detector Demo")
    print("=" * 60)

    # Create synthetic SCADA data
    np.random.seed(42)
    n_hours = 24 * 365  # 1 year
    n_samples = n_hours * 6  # 10-min intervals

    times = np.arange(n_samples) / 6.0  # Hours

    # Create signal with diurnal pattern + noise
    diurnal = 10 * np.sin(2 * np.pi * times / 24)  # 24-hour period
    noise = np.random.normal(0, 2, n_samples)
    signal_healthy = diurnal + noise + 50  # Baseline temp around 50

    # Create anomalous signal (degraded periodic structure)
    anomaly_start = int(n_samples * 0.8)
    signal_anomaly = signal_healthy.copy()
    signal_anomaly[anomaly_start:] += np.random.normal(0, 5, n_samples - anomaly_start)  # More noise
    signal_anomaly[anomaly_start:] += times[anomaly_start:] * 0.001  # Upward trend

    # Create DataFrame
    df = pd.DataFrame({
        'time_stamp': pd.date_range('2022-01-01', periods=n_samples, freq='10min'),
        'status_type_id': np.zeros(n_samples, dtype=int),
        'train_test': ['train'] * int(n_samples * 0.7) + ['test'] * (n_samples - int(n_samples * 0.7)),
        'bearing_temp_avg': signal_anomaly
    })

    train_mask = df['train_test'] == 'train'
    test_start_idx = int(n_samples * 0.7)

    # Build detector
    detector = PeriodicBaselineDetector()

    # Build baseline
    print("\n1. Building baseline from training data...")
    baselines = detector.build_baseline(df, ['bearing_temp_avg'], train_mask)

    if 'bearing_temp_avg' in baselines:
        baseline = baselines['bearing_temp_avg']
        print(f"   Found {len(baseline.components)} periodic components")
        for i, comp in enumerate(baseline.components[:3]):
            print(f"   Component {i+1}: period={comp.period_hours:.1f}h, CV={comp.cv_percent:.2f}%")
        print(f"   Residual power fraction: {baseline.residual_power_fraction:.3f}")

    # Detect anomalies
    print("\n2. Detecting anomalies in prediction period...")
    scores = detector.detect_anomalies(df, ['bearing_temp_avg'], baselines, test_start_idx)

    print(f"   Analyzed {len(scores)} windows")

    # Summary
    anomaly_scores = [s.score for s in scores]
    print(f"   Score range: {min(anomaly_scores):.3f} - {max(anomaly_scores):.3f}")
    print(f"   Windows above 0.5: {sum(1 for s in anomaly_scores if s > 0.5)}")

    # Convert to predictions
    predictions = detector.scores_to_predictions(scores, n_samples, test_start_idx, threshold=0.5)
    print(f"\n3. Predictions: {np.sum(predictions)} timestamps flagged as anomaly")
    print(f"   ({100*np.sum(predictions)/len(predictions):.1f}% of all timestamps)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    run_periodic_baseline_demo()

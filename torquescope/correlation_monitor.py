"""
TorqueScope Phase 2: Correlation Stability Monitoring

This module implements Phase 3 of v7 detection - correlation monitoring
to catch multivariate anomalies that single-sensor detection misses.

Key insight: The autoencoder baseline (CARE = 0.66) works because it learns
joint distributions - the relationships between sensors. This module captures
that structure explicitly using correlation matrices.

Per Brief v7:
- Build correlation baseline from training data
- Monitor for decorrelation during test period
- A degrading component changes its relationship to other components
  BEFORE it shows absolute temperature rise
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class CorrelationBaseline:
    """
    Baseline correlation structure computed from training data.

    A "stable pair" is two sensors that are strongly correlated during
    training and should remain correlated during normal operation.
    Decorrelation of stable pairs is an anomaly indicator.
    """
    sensors: List[str]                    # Sensor column names
    matrix: np.ndarray                    # n_sensors x n_sensors correlation matrix
    stable_pairs: List[Dict]              # Pairs with |rho| > min_correlation
    correlation_stds: Dict[Tuple[int, int], float]  # Per-pair training std
    n_stable_pairs: int
    min_correlation: float = 0.5          # Threshold used to identify stable pairs


@dataclass
class DecorrelationResult:
    """
    Result of correlation stability check for a single window.

    A high score indicates the correlation structure has changed
    from what was learned during training.
    """
    score: float                          # Overall decorrelation score (higher = more anomalous)
    broken_pairs: List[Dict]              # Which specific pairs decorrelated
    n_broken: int                         # Count of significantly broken pairs
    n_stable_pairs: int                   # Total stable pairs monitored
    pair_details: List[Dict] = field(default_factory=list)  # All pair z-scores for diagnostics


def build_correlation_baseline(
    training_data: pd.DataFrame,
    sensors: List[str],
    min_correlation: float = 0.5,
    rolling_window_samples: int = 432,  # 3 days at 10-min intervals
    rolling_step_samples: int = 72       # 12-hour steps
) -> Optional[CorrelationBaseline]:
    """
    Build correlation baseline from training data.

    Identifies stable sensor pairs (correlation > threshold) and computes
    the natural variance of each pairwise correlation using rolling windows.

    Args:
        training_data: DataFrame with sensor columns (training period only)
        sensors: List of sensor column names to include
        min_correlation: Minimum |correlation| to consider a pair "stable"
        rolling_window_samples: Window size for computing correlation variance
        rolling_step_samples: Step size for rolling windows

    Returns:
        CorrelationBaseline if successful, None if insufficient data
    """
    # Filter to available sensors
    available_sensors = [s for s in sensors if s in training_data.columns]

    if len(available_sensors) < 2:
        return None

    # Get sensor data, dropping rows with any NaN
    sensor_data = training_data[available_sensors].copy()

    # Handle zeros as NaN (common in CARE dataset)
    for col in available_sensors:
        zero_frac = (sensor_data[col] == 0).sum() / len(sensor_data)
        if zero_frac > 0.1:  # More than 10% zeros
            sensor_data[col] = sensor_data[col].replace(0, np.nan)

    # Drop rows with any NaN for correlation computation
    clean_data = sensor_data.dropna()

    if len(clean_data) < 100:
        return None

    # Compute baseline correlation matrix
    corr_matrix = clean_data.corr().values
    n_sensors = len(available_sensors)

    # Identify stable pairs
    stable_pairs = []
    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            rho = corr_matrix[i, j]
            if not np.isnan(rho) and abs(rho) >= min_correlation:
                stable_pairs.append({
                    'sensor_i': available_sensors[i],
                    'sensor_j': available_sensors[j],
                    'index_i': i,
                    'index_j': j,
                    'baseline_rho': float(rho),
                    'expected_sign': int(np.sign(rho))
                })

    if len(stable_pairs) == 0:
        # No stable pairs found - return baseline with empty pairs
        return CorrelationBaseline(
            sensors=available_sensors,
            matrix=corr_matrix,
            stable_pairs=[],
            correlation_stds={},
            n_stable_pairs=0,
            min_correlation=min_correlation
        )

    # Compute correlation stability (std) during training using rolling windows
    correlation_stds = compute_rolling_correlation_std(
        clean_data,
        available_sensors,
        stable_pairs,
        window_size=rolling_window_samples,
        step_size=rolling_step_samples
    )

    return CorrelationBaseline(
        sensors=available_sensors,
        matrix=corr_matrix,
        stable_pairs=stable_pairs,
        correlation_stds=correlation_stds,
        n_stable_pairs=len(stable_pairs),
        min_correlation=min_correlation
    )


def compute_rolling_correlation_std(
    data: pd.DataFrame,
    sensors: List[str],
    stable_pairs: List[Dict],
    window_size: int = 432,
    step_size: int = 72
) -> Dict[Tuple[int, int], float]:
    """
    Compute the natural variance of each pairwise correlation using rolling windows.

    This tells us how much correlation naturally varies during training,
    so we can properly z-score deviations during test.

    Args:
        data: Clean DataFrame (no NaN) with sensor columns
        sensors: List of sensor names
        stable_pairs: List of stable pair dicts from baseline building
        window_size: Samples per window
        step_size: Samples between windows

    Returns:
        Dict mapping (i, j) index pairs to std of correlation
    """
    n_samples = len(data)

    if n_samples < window_size:
        # Not enough data for rolling windows
        return {(p['index_i'], p['index_j']): 0.1 for p in stable_pairs}

    # Collect correlations per window for each pair
    pair_correlations: Dict[Tuple[int, int], List[float]] = {
        (p['index_i'], p['index_j']): [] for p in stable_pairs
    }

    for start in range(0, n_samples - window_size, step_size):
        end = start + window_size
        window_data = data.iloc[start:end]

        # Skip windows with too few valid samples
        if len(window_data.dropna()) < window_size * 0.5:
            continue

        window_corr = window_data.corr().values

        for pair in stable_pairs:
            i, j = pair['index_i'], pair['index_j']
            rho = window_corr[i, j]
            if not np.isnan(rho):
                pair_correlations[(i, j)].append(rho)

    # Compute std for each pair
    correlation_stds = {}
    for (i, j), rhos in pair_correlations.items():
        if len(rhos) >= 3:
            correlation_stds[(i, j)] = float(np.std(rhos))
        else:
            # Default std if not enough windows
            correlation_stds[(i, j)] = 0.1

    return correlation_stds


def compute_decorrelation_score(
    window_data: pd.DataFrame,
    baseline: CorrelationBaseline,
    z_threshold: float = 1.5
) -> DecorrelationResult:
    """
    Compute how much the correlation structure has changed from baseline.

    For each stable pair, compute the z-score of correlation change.
    A pair is "broken" if it decorrelates significantly (z > threshold).

    Special attention to sign flips: a pair that was rho=0.8 and is now
    rho=-0.2 has REALLY decorrelated.

    Args:
        window_data: DataFrame for current window
        baseline: CorrelationBaseline from training
        z_threshold: Z-score threshold to consider a pair "broken"

    Returns:
        DecorrelationResult with score and broken pair details
    """
    if baseline.n_stable_pairs == 0:
        return DecorrelationResult(
            score=0.0,
            broken_pairs=[],
            n_broken=0,
            n_stable_pairs=0,
            pair_details=[]
        )

    # Filter to available sensors
    available = [s for s in baseline.sensors if s in window_data.columns]

    if len(available) < 2:
        return DecorrelationResult(
            score=0.0,
            broken_pairs=[],
            n_broken=0,
            n_stable_pairs=baseline.n_stable_pairs,
            pair_details=[]
        )

    # Get window data and handle zeros/NaN
    sensor_data = window_data[available].copy()
    for col in available:
        zero_frac = (sensor_data[col] == 0).sum() / len(sensor_data)
        if zero_frac > 0.1:
            sensor_data[col] = sensor_data[col].replace(0, np.nan)

    clean_data = sensor_data.dropna()

    if len(clean_data) < 50:
        # Not enough valid data in window
        return DecorrelationResult(
            score=0.0,
            broken_pairs=[],
            n_broken=0,
            n_stable_pairs=baseline.n_stable_pairs,
            pair_details=[]
        )

    # Create sensor name to index mapping for current data
    sensor_to_idx = {s: i for i, s in enumerate(available)}

    # Compute current correlation matrix
    current_corr = clean_data.corr().values

    broken_pairs = []
    pair_scores = []
    pair_details = []

    for pair in baseline.stable_pairs:
        sensor_i = pair['sensor_i']
        sensor_j = pair['sensor_j']

        # Check if both sensors are available
        if sensor_i not in sensor_to_idx or sensor_j not in sensor_to_idx:
            continue

        curr_i = sensor_to_idx[sensor_i]
        curr_j = sensor_to_idx[sensor_j]

        baseline_rho = pair['baseline_rho']
        current_rho = current_corr[curr_i, curr_j]

        if np.isnan(current_rho):
            continue

        # Correlation change (positive = decorrelation for positive baseline pairs)
        # A pair that was rho=0.8 and is now rho=0.3 has decorrelated (delta=0.5)
        delta_rho = baseline_rho - current_rho

        # Get the training std for this pair
        pair_key = (pair['index_i'], pair['index_j'])
        pair_std = baseline.correlation_stds.get(pair_key, 0.1)

        # Compute z-score of decorrelation
        # Avoid division by tiny std
        z_decorr = delta_rho / max(pair_std, 0.05)

        # Sign flip is especially anomalous
        expected_sign = pair['expected_sign']
        if expected_sign > 0 and current_rho < 0:
            # Was positive, now negative - strong anomaly
            z_decorr = abs(z_decorr) + 2.0
        elif expected_sign < 0 and current_rho > 0:
            # Was negative, now positive - strong anomaly
            z_decorr = abs(z_decorr) + 2.0

        # Store details for diagnostics
        pair_details.append({
            'sensor_i': sensor_i,
            'sensor_j': sensor_j,
            'baseline_rho': baseline_rho,
            'current_rho': float(current_rho),
            'delta_rho': float(delta_rho),
            'z_decorr': float(z_decorr)
        })

        # Check if this pair is significantly broken
        if z_decorr > z_threshold:
            broken_pairs.append({
                'sensor_i': sensor_i,
                'sensor_j': sensor_j,
                'baseline_rho': baseline_rho,
                'current_rho': float(current_rho),
                'z_decorr': float(z_decorr)
            })

        # Only count decorrelation (positive z for positive baseline pairs)
        pair_scores.append(max(0.0, z_decorr))

    if not pair_scores:
        return DecorrelationResult(
            score=0.0,
            broken_pairs=[],
            n_broken=0,
            n_stable_pairs=baseline.n_stable_pairs,
            pair_details=pair_details
        )

    # Aggregate score: combination of max and breadth
    max_decorr = max(pair_scores)

    # Count how many pairs are significantly decorrelated
    n_broken = sum(1 for z in pair_scores if z > z_threshold)
    breadth_score = n_broken / len(pair_scores) if pair_scores else 0

    # Score combines: worst pair + breadth of decorrelation
    # A single badly broken pair is anomalous
    # Many mildly broken pairs is also anomalous
    decorrelation_score = max_decorr + 2.0 * breadth_score

    return DecorrelationResult(
        score=float(decorrelation_score),
        broken_pairs=broken_pairs,
        n_broken=n_broken,
        n_stable_pairs=baseline.n_stable_pairs,
        pair_details=pair_details
    )

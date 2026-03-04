"""
Normal Behaviour Model (NBM) for TorqueScope Phase 2.

This module implements binned regression models that predict component temperatures
from operational conditions (power, ambient temperature). The residuals (actual - predicted)
remove operational variation, making periodic structure detectable by Lomb-Scargle analysis.

The NBM is the Phase 2 analogue of envelope extraction in Phase 1 - a preprocessing step
that makes the signal amenable to LS analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class NBMModel:
    """
    Normal Behaviour Model for a single temperature signal.

    Uses binned regression: operational space is divided into (power x ambient_temp) cells,
    and each cell stores the median temperature observed during normal training operation.

    This is NOT a machine learning model - it's a physics-informed lookup table.

    v6: Enhanced with 20x10 bins, percentiles, and nearest-neighbor fallback.
    """
    target_col: str
    power_edges: np.ndarray              # Bin edges for power (n_power_bins + 1 values)
    ambient_edges: np.ndarray            # Bin edges for ambient temp (n_ambient_bins + 1 values)
    bin_medians: Dict[Tuple[int, int], float] = field(default_factory=dict)
    bin_stds: Dict[Tuple[int, int], float] = field(default_factory=dict)
    bin_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    bin_q10: Dict[Tuple[int, int], float] = field(default_factory=dict)  # v6: 10th percentile
    bin_q90: Dict[Tuple[int, int], float] = field(default_factory=dict)  # v6: 90th percentile
    global_median: float = 0.0
    global_std: float = 1.0
    n_training_samples: int = 0
    n_power_bins: int = 20   # v6: increased from 10
    n_ambient_bins: int = 10  # v6: increased from 5


@dataclass
class PowerCurveModel:
    """
    Binned power curve model (wind speed -> power).

    The power curve residual (actual - expected power) is a universal fault indicator.
    Almost any drivetrain fault eventually affects power output.
    """
    wind_edges: np.ndarray               # Bin edges for wind speed (n_bins + 1 values)
    bin_medians: np.ndarray              # Median power per wind bin
    bin_counts: np.ndarray               # Sample count per bin
    n_bins: int = 20
    rated_power: float = 1.0             # For normalisation


def build_nbm(
    train_df: pd.DataFrame,
    target_col: str,
    power_col: str,
    ambient_col: str,
    wind_col: Optional[str] = None,
    n_power_bins: int = 20,  # v6: increased from 10
    n_ambient_bins: int = 10,  # v6: increased from 5
    n_wind_bins: int = 5,
    min_power_fraction: float = 0.05,
    min_samples_per_bin: int = 30  # v6: for nearest-neighbor fallback
) -> Optional[NBMModel]:
    """
    Build a binned Normal Behaviour Model.

    Bins operational space into (power x ambient_temp) cells, optionally with wind speed.
    Each cell stores the median target temperature.

    Args:
        train_df: Training data (should be pre-filtered to normal status)
        target_col: Temperature signal to model (e.g. 'sensor_11_avg')
        power_col: Active power column
        ambient_col: Ambient temperature column
        wind_col: Optional wind speed column (adds 3rd dimension to binning)
        n_power_bins: Number of power bins (default 10)
        n_ambient_bins: Number of ambient temperature bins (default 5)
        n_wind_bins: Number of wind speed bins (default 5, only used if wind_col provided)
        min_power_fraction: Minimum power as fraction of max (default 0.05)
            Rows below this are considered non-operating and excluded

    Returns:
        NBMModel if successful, None if insufficient data
    """
    # Validate columns exist
    for col in [target_col, power_col, ambient_col]:
        if col not in train_df.columns:
            return None

    # Get valid rows (no NaN in key columns)
    valid_mask = (
        train_df[target_col].notna() &
        train_df[power_col].notna() &
        train_df[ambient_col].notna()
    )
    df = train_df[valid_mask].copy()

    if len(df) < 100:
        return None

    # Determine power threshold (5% of max observed)
    max_power = df[power_col].max()
    power_threshold = max_power * min_power_fraction

    # Filter to operating conditions only
    operating = df[df[power_col] > power_threshold].copy()

    if len(operating) < 100:
        return None

    # Compute global statistics
    global_median = operating[target_col].median()
    global_std = operating[target_col].std()

    if global_std == 0 or np.isnan(global_std):
        global_std = 1.0

    # Create bin edges using quantiles (handles non-uniform distributions)
    try:
        _, power_edges = pd.qcut(operating[power_col], q=n_power_bins, retbins=True, duplicates='drop')
        _, ambient_edges = pd.qcut(operating[ambient_col], q=n_ambient_bins, retbins=True, duplicates='drop')
    except ValueError:
        # Not enough unique values for binning
        return None

    # Assign bins
    power_bins = pd.cut(operating[power_col], bins=power_edges, labels=False, include_lowest=True)
    ambient_bins = pd.cut(operating[ambient_col], bins=ambient_edges, labels=False, include_lowest=True)

    # Compute statistics per bin (v6: add percentiles)
    bin_medians = {}
    bin_stds = {}
    bin_counts = {}
    bin_q10 = {}
    bin_q90 = {}

    # Track bin centers for nearest-neighbor lookup
    bin_centers = []

    for pb in range(len(power_edges) - 1):
        for ab in range(len(ambient_edges) - 1):
            mask = (power_bins == pb) & (ambient_bins == ab)
            values = operating.loc[mask, target_col]

            if len(values) >= 5:  # Require at least 5 samples per bin
                bin_medians[(pb, ab)] = values.median()
                bin_stds[(pb, ab)] = values.std() if values.std() > 0 else global_std * 0.1
                bin_counts[(pb, ab)] = len(values)
                bin_q10[(pb, ab)] = values.quantile(0.10)
                bin_q90[(pb, ab)] = values.quantile(0.90)

                # Track bin center for nearest-neighbor
                power_center = (power_edges[pb] + power_edges[pb + 1]) / 2
                ambient_center = (ambient_edges[ab] + ambient_edges[ab + 1]) / 2
                bin_centers.append((pb, ab, power_center, ambient_center))

    if len(bin_medians) < 5:
        # Too few populated bins
        return None

    return NBMModel(
        target_col=target_col,
        power_edges=power_edges,
        ambient_edges=ambient_edges,
        bin_medians=bin_medians,
        bin_stds=bin_stds,
        bin_counts=bin_counts,
        bin_q10=bin_q10,
        bin_q90=bin_q90,
        global_median=global_median,
        global_std=global_std,
        n_training_samples=len(operating),
        n_power_bins=len(power_edges) - 1,
        n_ambient_bins=len(ambient_edges) - 1
    )


def _find_nearest_bin(
    pb: int,
    ab: int,
    nbm: NBMModel,
    power_val: float,
    ambient_val: float
) -> Tuple[int, int]:
    """
    Find the nearest populated bin using Euclidean distance in normalized bin space.

    v6: Nearest-neighbor fallback for bins with insufficient training samples.
    """
    if not nbm.bin_medians:
        return (pb, ab)

    # Normalize distances by number of bins
    power_range = nbm.power_edges[-1] - nbm.power_edges[0]
    ambient_range = nbm.ambient_edges[-1] - nbm.ambient_edges[0]

    if power_range == 0:
        power_range = 1.0
    if ambient_range == 0:
        ambient_range = 1.0

    best_key = None
    best_dist = float('inf')

    for key in nbm.bin_medians.keys():
        kpb, kab = key
        # Compute bin center for this key
        power_center = (nbm.power_edges[kpb] + nbm.power_edges[kpb + 1]) / 2
        ambient_center = (nbm.ambient_edges[kab] + nbm.ambient_edges[kab + 1]) / 2

        # Normalized Euclidean distance
        dp = (power_val - power_center) / power_range
        da = (ambient_val - ambient_center) / ambient_range
        dist = np.sqrt(dp**2 + da**2)

        if dist < best_dist:
            best_dist = dist
            best_key = key

    return best_key if best_key else (pb, ab)


def predict_nbm(
    df: pd.DataFrame,
    nbm: NBMModel,
    power_col: str,
    ambient_col: str,
    return_stds: bool = False
) -> np.ndarray:
    """
    Predict expected temperature for each row using the NBM.

    v6: Added nearest-neighbor fallback for unseen bins and optional std return.

    Args:
        df: DataFrame with operational columns
        nbm: Trained NBMModel
        power_col: Power column name
        ambient_col: Ambient temperature column name
        return_stds: If True, return (predictions, stds) tuple

    Returns:
        Array of predicted temperatures. NaN for rows where:
        - Power or ambient is NaN
        If return_stds=True, returns (predictions, stds) tuple
    """
    n_rows = len(df)
    predictions = np.full(n_rows, np.nan)
    stds = np.full(n_rows, np.nan)

    # Get column values
    power_vals = df[power_col].values
    ambient_vals = df[ambient_col].values

    # Assign to bins
    power_bins = np.digitize(power_vals, nbm.power_edges[1:-1])  # Returns 0 to n_bins-1
    ambient_bins = np.digitize(ambient_vals, nbm.ambient_edges[1:-1])

    # Clip to valid range
    power_bins = np.clip(power_bins, 0, nbm.n_power_bins - 1)
    ambient_bins = np.clip(ambient_bins, 0, nbm.n_ambient_bins - 1)

    # Look up predictions
    for i in range(n_rows):
        if np.isnan(power_vals[i]) or np.isnan(ambient_vals[i]):
            continue

        key = (int(power_bins[i]), int(ambient_bins[i]))
        if key in nbm.bin_medians:
            predictions[i] = nbm.bin_medians[key]
            stds[i] = nbm.bin_stds.get(key, nbm.global_std)
        else:
            # v6: Nearest-neighbor fallback for unseen bins
            nearest_key = _find_nearest_bin(
                int(power_bins[i]), int(ambient_bins[i]),
                nbm, power_vals[i], ambient_vals[i]
            )
            if nearest_key in nbm.bin_medians:
                predictions[i] = nbm.bin_medians[nearest_key]
                stds[i] = nbm.bin_stds.get(nearest_key, nbm.global_std)
            else:
                predictions[i] = nbm.global_median
                stds[i] = nbm.global_std

    if return_stds:
        return predictions, stds
    return predictions


def compute_residuals(
    df: pd.DataFrame,
    nbm: NBMModel,
    target_col: str,
    power_col: str,
    ambient_col: str,
    min_power_fraction: float = 0.05
) -> np.ndarray:
    """
    Compute residuals: T_actual - T_predicted.

    For a healthy turbine: residual ~ 0 with random noise
    For a degrading bearing: residual drifts positive (increased friction)

    Args:
        df: DataFrame with all columns
        nbm: Trained NBMModel
        target_col: Temperature column to compute residual for
        power_col: Power column name
        ambient_col: Ambient temperature column name
        min_power_fraction: Minimum power fraction for valid residual

    Returns:
        Array of residuals. NaN when:
        - Turbine not operating (power < threshold)
        - Prediction is NaN
        - Actual temperature is NaN
    """
    # Get predictions
    predictions = predict_nbm(df, nbm, power_col, ambient_col)

    # Get actual values
    actual = df[target_col].values

    # Compute residuals
    residuals = actual - predictions

    # Mask non-operating periods
    max_power = df[power_col].max()
    power_threshold = max_power * min_power_fraction
    non_operating = df[power_col].values < power_threshold
    residuals[non_operating] = np.nan

    return residuals


def build_power_curve(
    train_df: pd.DataFrame,
    power_col: str,
    wind_col: str,
    n_bins: int = 20
) -> Optional[PowerCurveModel]:
    """
    Build binned power curve model.

    Args:
        train_df: Training data (pre-filtered to normal status)
        power_col: Active power column
        wind_col: Wind speed column
        n_bins: Number of wind speed bins

    Returns:
        PowerCurveModel if successful, None if insufficient data
    """
    # Validate columns
    if power_col not in train_df.columns or wind_col not in train_df.columns:
        return None

    # Get valid rows
    valid_mask = train_df[power_col].notna() & train_df[wind_col].notna()
    df = train_df[valid_mask].copy()

    if len(df) < 100:
        return None

    # Filter to positive power (operating)
    operating = df[df[power_col] > 0].copy()

    if len(operating) < 100:
        return None

    # Create wind speed bins
    try:
        _, wind_edges = pd.qcut(operating[wind_col], q=n_bins, retbins=True, duplicates='drop')
    except ValueError:
        # Try with fewer bins
        try:
            _, wind_edges = pd.qcut(operating[wind_col], q=10, retbins=True, duplicates='drop')
            n_bins = 10
        except ValueError:
            return None

    # Assign bins and compute statistics
    wind_bins = pd.cut(operating[wind_col], bins=wind_edges, labels=False, include_lowest=True)

    bin_medians = np.zeros(len(wind_edges) - 1)
    bin_counts = np.zeros(len(wind_edges) - 1, dtype=int)

    for wb in range(len(wind_edges) - 1):
        mask = wind_bins == wb
        values = operating.loc[mask, power_col]
        if len(values) > 0:
            bin_medians[wb] = values.median()
            bin_counts[wb] = len(values)

    return PowerCurveModel(
        wind_edges=wind_edges,
        bin_medians=bin_medians,
        bin_counts=bin_counts,
        n_bins=len(wind_edges) - 1,
        rated_power=operating[power_col].max()
    )


def power_curve_residual(
    df: pd.DataFrame,
    power_curve: PowerCurveModel,
    power_col: str,
    wind_col: str
) -> np.ndarray:
    """
    Compute power curve residual: actual - expected power.

    Args:
        df: DataFrame with power and wind columns
        power_curve: Trained PowerCurveModel
        power_col: Power column name
        wind_col: Wind speed column name

    Returns:
        Array of power residuals. NaN when prediction unavailable.
    """
    n_rows = len(df)
    residuals = np.full(n_rows, np.nan)

    wind_vals = df[wind_col].values
    power_vals = df[power_col].values

    # Assign wind speed to bins
    wind_bins = np.digitize(wind_vals, power_curve.wind_edges[1:-1])
    wind_bins = np.clip(wind_bins, 0, power_curve.n_bins - 1)

    for i in range(n_rows):
        if np.isnan(wind_vals[i]) or np.isnan(power_vals[i]):
            continue

        expected = power_curve.bin_medians[wind_bins[i]]
        if expected > 0:  # Only compute residual where we have a valid expectation
            residuals[i] = power_vals[i] - expected

    return residuals


def get_operational_columns(farm: str, df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    Return the power, wind, and ambient columns for each farm.

    These mappings are based on the CARE dataset feature analysis.
    Column names include the _avg suffix as that's how they appear in the dataset.

    If df is provided, validates columns exist and falls back to alternatives.

    Args:
        farm: Farm identifier ('Wind Farm A', 'Wind Farm B', 'Wind Farm C')
              or short form ('A', 'B', 'C')
        df: Optional DataFrame to validate columns exist

    Returns:
        Dict with keys 'power', 'wind', 'ambient' mapping to column names
    """
    # Define preferred and alternative columns per farm
    farm_key = farm.upper()
    if 'A' in farm_key:
        candidates = {
            'power': ['power_30_avg', 'power_29_avg'],
            'wind': ['wind_speed_3_avg', 'wind_speed_4_avg'],
            'ambient': ['sensor_0_avg']
        }
    elif 'B' in farm_key:
        candidates = {
            'power': ['power_62_avg', 'power_58_avg'],
            'wind': ['wind_speed_59_avg', 'wind_speed_60_avg'],
            'ambient': ['sensor_0_avg', 'sensor_7_avg', 'sensor_8_avg']
        }
    elif 'C' in farm_key:
        candidates = {
            'power': ['power_6_avg', 'power_2_avg', 'power_5_avg'],
            'wind': ['wind_speed_235_avg', 'wind_speed_236_avg'],
            'ambient': ['sensor_7_avg', 'sensor_0_avg', 'sensor_177_avg']
        }
    else:
        raise ValueError(f"Unknown farm: {farm}")

    # If no df provided, return first candidate
    if df is None:
        return {k: v[0] for k, v in candidates.items()}

    # Otherwise, find first existing column for each
    result = {}
    for key, cols in candidates.items():
        for col in cols:
            if col in df.columns:
                result[key] = col
                break
        if key not in result:
            # Try to find any matching column
            if key == 'power':
                matches = [c for c in df.columns if 'power_' in c.lower()
                          and 'reactive' not in c.lower() and '_avg' in c]
            elif key == 'wind':
                matches = [c for c in df.columns if 'wind_speed' in c.lower() and '_avg' in c]
            else:  # ambient
                matches = [c for c in df.columns if 'sensor_0' in c or 'sensor_7' in c]
            if matches:
                result[key] = matches[0]

    return result


def validate_nbm_residuals(
    df: pd.DataFrame,
    nbm: NBMModel,
    target_col: str,
    power_col: str,
    ambient_col: str,
    n_windows: int = 4
) -> Dict[str, float]:
    """
    Validate that NBM residuals have lower variance and better periodicity than raw signal.

    This is the CRITICAL test from Brief v4:
    "If the NBM residual for training data has a CV < 20% for the dominant LS frequency,
    the approach will work. If CV is still > 40%, we need a better NBM."

    Args:
        df: DataFrame with all columns
        nbm: Trained NBMModel
        target_col: Temperature column
        power_col: Power column
        ambient_col: Ambient temperature column
        n_windows: Number of windows for multi-window CV calculation

    Returns:
        Dict with validation metrics:
        - raw_variance: Variance of raw signal
        - residual_variance: Variance of residuals
        - variance_reduction: raw_variance / residual_variance
        - raw_cv: Multi-window CV of raw signal (%)
        - residual_cv: Multi-window CV of residuals (%)
        - cv_improvement: raw_cv / residual_cv
        - dominant_period_hours: Dominant period of residual
    """
    from astropy.timeseries import LombScargle
    from scipy.signal import find_peaks

    # Compute residuals
    residuals = compute_residuals(df, nbm, target_col, power_col, ambient_col)
    raw_signal = df[target_col].values

    # Filter to valid (non-NaN) values
    valid_mask = ~np.isnan(residuals) & ~np.isnan(raw_signal)
    residuals_valid = residuals[valid_mask]
    raw_valid = raw_signal[valid_mask]

    if len(residuals_valid) < 500:
        return {'error': 'Insufficient valid samples', 'n_valid': len(residuals_valid)}

    # Compute variance
    raw_variance = np.var(raw_valid)
    residual_variance = np.var(residuals_valid)
    variance_reduction = raw_variance / residual_variance if residual_variance > 0 else 0

    # Create time axis (assume 10-minute intervals = 6 samples per hour)
    samples_per_hour = 6
    hours = np.arange(len(residuals_valid)) / samples_per_hour

    # LS frequency grid (30 days to 2 hours)
    freqs = np.linspace(1.0 / (30 * 24), 0.5, 5000)

    def compute_multi_window_cv(signal, hours, freqs, n_windows):
        """Compute multi-window CV for dominant frequency."""
        n = len(signal)
        window_size = n // n_windows
        window_freqs = []

        for w in range(n_windows):
            start = w * window_size
            end = (w + 1) * window_size if w < n_windows - 1 else n

            t_win = hours[start:end]
            s_win = signal[start:end] - np.mean(signal[start:end])

            if len(t_win) >= 100:
                ls = LombScargle(t_win, s_win)
                powers = ls.power(freqs)
                peak_freq = freqs[np.argmax(powers)]
                window_freqs.append(peak_freq)

        if len(window_freqs) >= 2:
            mean_freq = np.mean(window_freqs)
            std_freq = np.std(window_freqs)
            cv = (std_freq / mean_freq) * 100 if mean_freq > 0 else 100
            return cv, window_freqs
        return 100.0, []

    # Compute CV for raw signal
    raw_cv, raw_freqs = compute_multi_window_cv(raw_valid, hours, freqs, n_windows)

    # Compute CV for residuals
    residual_cv, residual_freqs = compute_multi_window_cv(residuals_valid, hours, freqs, n_windows)

    # Find dominant period of residual
    residuals_centered = residuals_valid - np.mean(residuals_valid)
    ls = LombScargle(hours, residuals_centered)
    powers = ls.power(freqs)
    dominant_freq = freqs[np.argmax(powers)]
    dominant_period_hours = 1.0 / dominant_freq if dominant_freq > 0 else 0

    return {
        'raw_variance': float(raw_variance),
        'residual_variance': float(residual_variance),
        'variance_reduction': float(variance_reduction),
        'raw_cv': float(raw_cv),
        'residual_cv': float(residual_cv),
        'cv_improvement': float(raw_cv / residual_cv) if residual_cv > 0 else 0,
        'dominant_period_hours': float(dominant_period_hours),
        'n_valid_samples': len(residuals_valid),
        'cv_threshold_met': residual_cv < 20.0,
        'cv_acceptable': residual_cv < 40.0
    }


if __name__ == '__main__':
    """
    Validation test: Run on one Farm A dataset to verify:
    1. NBM can be built successfully
    2. Residual variance < raw signal variance
    3. Residual CV < 20% (critical success criterion)
    """
    import sys
    sys.path.insert(0, '/Users/tombrady/Documents/GitHub/tremorscope')

    from torquescope_phase2.data_loader import CAREDataLoader

    # Load a Farm A dataset
    data_dir = '/Users/tombrady/Documents/GitHub/tremorscope/data/care/CARE_To_Compare'
    loader = CAREDataLoader(data_dir)

    # Get list of datasets
    profiles_dict = loader.profile_all_datasets()

    # Find a Farm A dataset with good data
    farm_a_profiles = [p for p in profiles_dict.values() if 'Farm A' in p.wind_farm]

    if not farm_a_profiles:
        print("ERROR: No Farm A datasets found")
        sys.exit(1)

    # Use first Farm A dataset
    profile = farm_a_profiles[0]
    print(f"\nValidation Test: {profile.wind_farm} - {profile.event_id}")
    print("=" * 60)

    # Load dataset
    df = loader.load_dataset(profile.wind_farm, profile.event_id)

    # Get operational columns
    op_cols = get_operational_columns(profile.wind_farm)
    print(f"\nOperational columns: {op_cols}")

    # Check columns exist
    for key, col in op_cols.items():
        if col in df.columns:
            print(f"  {key}: {col} - OK")
        else:
            print(f"  {key}: {col} - MISSING")

    # Filter to training data with normal status
    train_mask = df['train_test'] == 'train'
    status_mask = df['status_type_id'].isin([0, 2])
    train_df = df[train_mask & status_mask]

    print(f"\nTraining samples: {len(train_df)}")

    # Find a temperature signal (not the ambient one we're using as predictor)
    temp_cols = [c for c in df.columns if 'sensor_' in c and '_avg' in c
                 and c != op_cols['ambient']]

    if not temp_cols:
        print("ERROR: No temperature columns found")
        sys.exit(1)

    # Use sensor_11_avg (typically a bearing temp) if available, else first temp
    target_col = 'sensor_11_avg' if 'sensor_11_avg' in temp_cols else temp_cols[0]
    print(f"\nTarget signal: {target_col}")

    # Build NBM
    print("\nBuilding NBM...")
    nbm = build_nbm(
        train_df,
        target_col,
        op_cols['power'],
        op_cols['ambient'],
        n_power_bins=10,
        n_ambient_bins=5
    )

    if nbm is None:
        print("ERROR: Failed to build NBM")
        sys.exit(1)

    print(f"  Bins populated: {len(nbm.bin_medians)}")
    print(f"  Training samples used: {nbm.n_training_samples}")
    print(f"  Global median: {nbm.global_median:.2f}")
    print(f"  Global std: {nbm.global_std:.2f}")

    # Validate residuals
    print("\nValidating residuals...")
    validation = validate_nbm_residuals(
        train_df,
        nbm,
        target_col,
        op_cols['power'],
        op_cols['ambient']
    )

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Raw signal variance:     {validation['raw_variance']:.4f}")
    print(f"Residual variance:       {validation['residual_variance']:.4f}")
    print(f"Variance reduction:      {validation['variance_reduction']:.2f}x")
    print(f"\nRaw signal CV:           {validation['raw_cv']:.1f}%")
    print(f"Residual CV:             {validation['residual_cv']:.1f}%")
    print(f"CV improvement:          {validation['cv_improvement']:.2f}x")
    print(f"\nDominant period:         {validation['dominant_period_hours']:.1f} hours")
    print(f"Valid samples:           {validation['n_valid_samples']}")

    print(f"\n{'='*60}")
    if validation['cv_threshold_met']:
        print("SUCCESS: Residual CV < 20% - approach should work!")
    elif validation['cv_acceptable']:
        print("WARNING: Residual CV 20-40% - approach may work, monitor results")
    else:
        print("FAILURE: Residual CV > 40% - NBM needs improvement")
    print(f"{'='*60}")

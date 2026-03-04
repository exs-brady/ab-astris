"""
Ab Astris Oceanography Experiment: Tidal Signal Detection

Validates the Ab Astris cross-domain signal detection methodology on
oceanographic tide gauge data from NOAA CO-OPS. Runs the full pipeline:
1. Bandpass filter (tidal bands)
2. Lomb-Scargle periodogram analysis
3. Multi-window validation (CV computation)
4. Bootstrap error estimation
5. Confidence scoring (0-100)

Phases:
- Phase 1: Tidal Validation (detect M2, S2, K1, O1 constituents)
- Phase 2: Residual Analysis (subtract predictions, find anomalies)

Note: Unlike bearing/seismic domains, tidal signals ARE the direct periodic
signal (no carrier modulation), so Hilbert envelope extraction is skipped.

Usage:
    python oceanography/tide_gauge_experiment.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import butter, filtfilt, hilbert

try:
    from noaa_coops import Station
except ImportError:
    print("ERROR: noaa-coops not installed. Run: pip install noaa-coops")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Output paths
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULTS_JSON = OUTPUT_DIR / "tide_gauge_results.json"

# Tidal constituent reference frequencies
# cpd = cycles per day; hz = cpd / 86400
TIDAL_CONSTITUENTS = {
    "M2": {"cpd": 1.9322736, "hz": 1.9322736 / 86400, "name": "Principal lunar semidiurnal"},
    "S2": {"cpd": 2.0000000, "hz": 2.0000000 / 86400, "name": "Principal solar semidiurnal"},
    "K1": {"cpd": 1.0027379, "hz": 1.0027379 / 86400, "name": "Lunisolar diurnal"},
    "O1": {"cpd": 0.9295357, "hz": 0.9295357 / 86400, "name": "Principal lunar diurnal"},
}

# Station configuration
DEFAULT_STATION_ID = "9414290"  # San Francisco, CA — longest continuous US record
DEFAULT_YEAR = 2023

# Sampling
SAMPLING_INTERVAL_SEC = 3600  # Hourly data
SAMPLING_RATE_HZ = 1 / SAMPLING_INTERVAL_SEC  # ~2.778e-4 Hz

# Bandpass ranges (in Hz) — two bands for diurnal and semidiurnal groups
DIURNAL_BAND_HZ = (0.4 / 86400, 1.2 / 86400)      # 0.4–1.2 cpd
SEMIDIURNAL_BAND_HZ = (1.5 / 86400, 2.5 / 86400)   # 1.5–2.5 cpd
FULL_TIDAL_BAND_HZ = (0.4 / 86400, 2.5 / 86400)    # 0.4–2.5 cpd (combined)
BANDPASS_ORDER = 4

# Multi-window parameters
N_WINDOWS = 8
WINDOW_DURATION_DAYS = 30
FREQ_TOLERANCE = 0.05  # 5% tolerance for detection rate

# Bootstrap
N_BOOTSTRAP = 100

# LS grid resolution
N_FREQ_POINTS = 10000

# Confidence scoring thresholds
CV_EXCELLENT = 0.01    # < 0.01% for tidal constituents
CV_GOOD = 0.1          # < 0.1%
CV_FAIR = 1.0           # < 1.0%

# Plot styling (dark theme matching other Ab Astris modules)
COLORS = {
    "background": "#0d1117",
    "panel": "#161b22",
    "accent": "#00b4d8",
    "accent2": "#ff6b6b",
    "accent3": "#4ecdc4",
    "text": "#c9d1d9",
    "grid": "#21262d",
    "tidal": "#4ecdc4",
    "residual": "#ff6b6b",
    "warning": "#f0ad4e",
}

# Cross-domain reference values (from other Ab Astris experiments)
CROSS_DOMAIN_REFS = {
    "variable_stars": {"cv_mean": 0.005, "detection_rate": 1.00, "domain": "Astronomy"},
    "bearings": {"cv_mean": 0.008, "detection_rate": 1.00, "domain": "Industrial"},
    "volcanoes": {"cv_mean": 3.96, "detection_rate": 0.997, "domain": "Geophysics"},
    "crypto": {"cv_mean": 68.0, "detection_rate": 0.30, "domain": "Financial"},
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LSResult:
    """Results from Lomb-Scargle periodogram analysis."""
    frequency_hz: float
    frequency_cpd: float
    power: float
    fap: float
    frequencies_hz: np.ndarray = field(repr=False)
    powers: np.ndarray = field(repr=False)


@dataclass
class MultiWindowResult:
    """Results from multi-window validation."""
    detected_freqs: List[float]
    cv_raw: float
    cv_clean: float
    detection_rate: float
    outlier_count: int


@dataclass
class BootstrapResult:
    """Results from bootstrap error estimation."""
    freq_mean: float
    freq_std: float
    freq_ci_low: float
    freq_ci_high: float


@dataclass
class ConstituentResult:
    """Complete analysis results for one tidal constituent."""
    constituent_name: str
    target_freq_hz: float
    target_freq_cpd: float

    # Lomb-Scargle results
    ls_frequency_hz: float
    ls_frequency_cpd: float
    ls_power: float
    ls_fap: float

    # Multi-window validation
    multi_window_cv: float
    multi_window_cv_clean: float
    detection_rate: float

    # Bootstrap
    bootstrap_freq_mean: float
    bootstrap_freq_std: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    # Derived metrics
    freq_error_percent: float
    confidence_score: float
    confidence_tier: str


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_tide_data(
    station_id: str,
    start_year: int,
    end_year: int,
    product: str = "hourly_height",
    data_dir: Path = None,
) -> pd.DataFrame:
    """Download tide gauge data from NOAA CO-OPS with caching.

    API limit: 1 year per request for hourly data.
    Downloads year-by-year and caches locally as CSV.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    cache_file = data_dir / f"{station_id}_{product}_{start_year}_{end_year}.csv"
    if cache_file.exists():
        print(f"  Loading cached: {cache_file.name}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    print(f"  Downloading {product} for station {station_id} ({start_year}-{end_year})...")
    station = Station(id=station_id)
    frames = []

    for year in range(start_year, end_year + 1):
        print(f"    {year}...", end=" ", flush=True)
        try:
            kwargs = dict(
                begin_date=f"{year}0101",
                end_date=f"{year}1231",
                product=product,
                datum="MSL",
                units="metric",
                time_zone="gmt",
            )
            if product == "predictions":
                kwargs["interval"] = "h"
            df = station.get_data(**kwargs)
            frames.append(df)
            print(f"{len(df)} records")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    if not frames:
        raise RuntimeError(f"No data downloaded for station {station_id}")

    data = pd.concat(frames)
    data.to_csv(cache_file)
    print(f"  Cached to: {cache_file.name}")
    return data


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = fs / 2
    low = f_low / nyq
    high = f_high / nyq
    # Clamp to valid range
    low = max(low, 1e-10)
    high = min(high, 0.9999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def extract_envelope(signal: np.ndarray) -> np.ndarray:
    """Compute Hilbert envelope with DC removed.

    Included for completeness — not used in the tidal pipeline since
    water level is directly periodic (no carrier to demodulate).
    """
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    envelope = envelope - np.mean(envelope)
    return envelope


def prepare_time_series(
    df: pd.DataFrame, value_column: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare time series for Lomb-Scargle analysis.

    Handles gaps (Lomb-Scargle advantage over FFT).
    Returns (time_seconds, values) where time is seconds since first obs.
    """
    df_clean = df[[value_column]].dropna()
    time_sec = (df_clean.index - df_clean.index[0]).total_seconds().values
    values = df_clean[value_column].values.astype(np.float64)
    return time_sec, values


# ============================================================================
# LOMB-SCARGLE ANALYSIS
# ============================================================================

def run_lomb_scargle(
    time: np.ndarray,
    signal: np.ndarray,
    freq_min_hz: float = None,
    freq_max_hz: float = None,
    n_points: int = N_FREQ_POINTS,
) -> LSResult:
    """Run Lomb-Scargle periodogram following Ab Astris methodology."""
    if freq_min_hz is None:
        freq_min_hz = FULL_TIDAL_BAND_HZ[0]
    if freq_max_hz is None:
        freq_max_hz = FULL_TIDAL_BAND_HZ[1]

    # Normalize
    data_mean = np.mean(signal)
    data_std = np.std(signal)
    if data_std > 1e-15:
        data_norm = (signal - data_mean) / data_std
    else:
        data_norm = signal - data_mean

    # Frequency grid
    frequencies_hz = np.linspace(freq_min_hz, freq_max_hz, n_points)

    # Compute Lomb-Scargle
    ls = LombScargle(time, data_norm)
    powers = ls.power(frequencies_hz)

    # Find best peak
    best_idx = np.argmax(powers)
    best_freq_hz = frequencies_hz[best_idx]
    best_power = float(powers[best_idx])

    # False alarm probability
    try:
        fap = float(ls.false_alarm_probability(best_power))
    except Exception:
        fap = 1.0

    return LSResult(
        frequency_hz=best_freq_hz,
        frequency_cpd=best_freq_hz * 86400,
        power=best_power,
        fap=fap,
        frequencies_hz=frequencies_hz,
        powers=powers,
    )


def find_peak_near(
    frequencies_hz: np.ndarray,
    powers: np.ndarray,
    target_hz: float,
    tolerance: float = FREQ_TOLERANCE,
) -> LSResult:
    """Find the strongest peak within ±tolerance of target frequency."""
    low = target_hz * (1 - tolerance)
    high = target_hz * (1 + tolerance)
    mask = (frequencies_hz >= low) & (frequencies_hz <= high)

    if not np.any(mask):
        return LSResult(
            frequency_hz=target_hz,
            frequency_cpd=target_hz * 86400,
            power=0.0,
            fap=1.0,
            frequencies_hz=frequencies_hz,
            powers=powers,
        )

    local_idx = np.argmax(powers[mask])
    peak_freq = frequencies_hz[mask][local_idx]
    peak_power = float(powers[mask][local_idx])

    # Recompute FAP for this peak would require the LombScargle object,
    # so we estimate from the global LS run
    return LSResult(
        frequency_hz=peak_freq,
        frequency_cpd=peak_freq * 86400,
        power=peak_power,
        fap=0.0,  # Will be set by the caller if needed
        frequencies_hz=frequencies_hz,
        powers=powers,
    )


# ============================================================================
# MULTI-WINDOW VALIDATION
# ============================================================================

def create_windows(
    time: np.ndarray,
    signal: np.ndarray,
    n_windows: int = N_WINDOWS,
    window_duration_sec: float = WINDOW_DURATION_DAYS * 86400,
) -> List[Dict]:
    """Create overlapping windows for multi-window validation."""
    total_duration = time[-1] - time[0]
    if total_duration < window_duration_sec:
        raise ValueError(
            f"Signal duration ({total_duration/86400:.0f} days) shorter than "
            f"window duration ({window_duration_sec/86400:.0f} days)"
        )

    step = (total_duration - window_duration_sec) / max(n_windows - 1, 1)

    windows = []
    for i in range(n_windows):
        start = time[0] + i * step
        end = start + window_duration_sec
        mask = (time >= start) & (time <= end)
        if np.sum(mask) < 100:
            continue
        windows.append({
            "time": time[mask] - time[mask][0],
            "signal": signal[mask],
        })

    return windows


def run_multi_window_validation(
    windows: List[Dict],
    target_freq_hz: float,
    freq_min_hz: float = None,
    freq_max_hz: float = None,
) -> MultiWindowResult:
    """Run Lomb-Scargle on each window. Compute CV across windows."""
    detected_freqs = []

    for window in windows:
        result = run_lomb_scargle(
            window["time"], window["signal"],
            freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz,
        )
        detected_freqs.append(result.frequency_hz)

    detected_freqs = np.array(detected_freqs)

    # Raw CV
    freq_mean = np.mean(detected_freqs)
    freq_std = np.std(detected_freqs)
    cv_raw = (freq_std / freq_mean * 100) if freq_mean > 0 else 0.0

    # Clean CV (remove 2-sigma outliers)
    outlier_count = 0
    if len(detected_freqs) > 2:
        z_scores = np.abs((detected_freqs - freq_mean) / (freq_std + 1e-15))
        clean_freqs = detected_freqs[z_scores < 2]
        outlier_count = len(detected_freqs) - len(clean_freqs)
        if len(clean_freqs) > 1:
            cv_clean = np.std(clean_freqs) / np.mean(clean_freqs) * 100
        else:
            cv_clean = cv_raw
    else:
        cv_clean = cv_raw

    # Detection rate (fraction within tolerance of target)
    detections = np.abs(detected_freqs - target_freq_hz) / target_freq_hz < FREQ_TOLERANCE
    detection_rate = float(np.mean(detections))

    return MultiWindowResult(
        detected_freqs=detected_freqs.tolist(),
        cv_raw=float(cv_raw),
        cv_clean=float(cv_clean),
        detection_rate=detection_rate,
        outlier_count=outlier_count,
    )


# ============================================================================
# BOOTSTRAP ERROR ESTIMATION
# ============================================================================

def run_bootstrap(
    time: np.ndarray,
    signal: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    freq_min_hz: float = None,
    freq_max_hz: float = None,
) -> BootstrapResult:
    """Bootstrap resampling for frequency uncertainty estimation."""
    n_samples = len(time)
    bootstrap_freqs = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        indices = np.sort(indices)  # Maintain temporal order

        t_boot = time[indices]
        s_boot = signal[indices]

        result = run_lomb_scargle(
            t_boot, s_boot,
            freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz,
        )
        bootstrap_freqs.append(result.frequency_hz)

    bootstrap_freqs = np.array(bootstrap_freqs)

    return BootstrapResult(
        freq_mean=float(np.mean(bootstrap_freqs)),
        freq_std=float(np.std(bootstrap_freqs)),
        freq_ci_low=float(np.percentile(bootstrap_freqs, 2.5)),
        freq_ci_high=float(np.percentile(bootstrap_freqs, 97.5)),
    )


# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

def compute_confidence_score(
    ls_result: LSResult,
    mw_result: MultiWindowResult,
    target_freq_hz: float,
) -> Tuple[float, str]:
    """Compute confidence score (0-100) following Ab Astris framework.

    Scoring:
    - 30 pts: Frequency accuracy (error from target)
    - 25 pts: Signal significance (FAP)
    - 25 pts: Multi-window stability (CV)
    - 20 pts: Detection rate across windows
    """
    score = 0.0

    # 1. Frequency accuracy (30 pts)
    freq_error = abs(ls_result.frequency_hz - target_freq_hz) / target_freq_hz
    if freq_error < 0.001:
        score += 30
    elif freq_error < 0.005:
        score += 25
    elif freq_error < 0.01:
        score += 20
    elif freq_error < 0.05:
        score += 15
    elif freq_error < 0.10:
        score += 5

    # 2. FAP significance (25 pts)
    fap = ls_result.fap
    if fap < 1e-10:
        score += 25
    elif fap < 1e-5:
        score += 20
    elif fap < 1e-3:
        score += 15
    elif fap < 0.01:
        score += 10
    elif fap < 0.05:
        score += 5

    # 3. CV stability (25 pts)
    cv = mw_result.cv_clean
    if cv < CV_EXCELLENT:
        score += 25
    elif cv < CV_GOOD:
        score += 20
    elif cv < CV_FAIR:
        score += 15
    elif cv < 5.0:
        score += 10
    elif cv < 10.0:
        score += 5

    # 4. Detection rate (20 pts)
    score += 20.0 * mw_result.detection_rate

    # Tier
    if score >= 90:
        tier = "CONFIRMED"
    elif score >= 75:
        tier = "PROBABLE"
    elif score >= 60:
        tier = "CANDIDATE"
    else:
        tier = "NOISE"

    return round(score, 1), tier


# ============================================================================
# PHASE 1: TIDAL VALIDATION
# ============================================================================

def _band_for_constituent(name: str) -> Tuple[float, float]:
    """Return the appropriate bandpass range (Hz) for a tidal constituent."""
    if name in ("K1", "O1"):
        return DIURNAL_BAND_HZ
    return SEMIDIURNAL_BAND_HZ


def analyze_constituent(
    time: np.ndarray,
    raw_signal: np.ndarray,
    constituent_name: str,
) -> ConstituentResult:
    """Run the full Ab Astris pipeline for one tidal constituent.

    Uses a narrow search band (±3%) around the target frequency so that
    each constituent is found independently rather than being dominated
    by the strongest signal in the diurnal/semidiurnal group.
    """
    info = TIDAL_CONSTITUENTS[constituent_name]
    target_hz = info["hz"]
    band_low, band_high = _band_for_constituent(constituent_name)

    # Bandpass filter to the broad tidal band (diurnal or semidiurnal)
    filtered = bandpass_filter(raw_signal, SAMPLING_RATE_HZ, band_low, band_high)

    # Narrow LS search window: ±3% around target frequency
    # This ensures we find this constituent specifically, not the dominant one
    narrow_low = target_hz * 0.97
    narrow_high = target_hz * 1.03

    # Lomb-Scargle on filtered signal (no envelope — tidal is direct periodic)
    ls_result = run_lomb_scargle(
        time, filtered, freq_min_hz=narrow_low, freq_max_hz=narrow_high,
    )

    # Multi-window validation with the same narrow band
    windows = create_windows(time, filtered)
    mw_result = run_multi_window_validation(
        windows, target_hz, freq_min_hz=narrow_low, freq_max_hz=narrow_high,
    )

    # Bootstrap error estimation
    boot_result = run_bootstrap(
        time, filtered, freq_min_hz=narrow_low, freq_max_hz=narrow_high,
    )

    # Confidence scoring
    score, tier = compute_confidence_score(ls_result, mw_result, target_hz)

    # Frequency error
    freq_error_pct = abs(ls_result.frequency_hz - target_hz) / target_hz * 100

    return ConstituentResult(
        constituent_name=constituent_name,
        target_freq_hz=target_hz,
        target_freq_cpd=info["cpd"],
        ls_frequency_hz=ls_result.frequency_hz,
        ls_frequency_cpd=ls_result.frequency_cpd,
        ls_power=ls_result.power,
        ls_fap=ls_result.fap,
        multi_window_cv=mw_result.cv_raw,
        multi_window_cv_clean=mw_result.cv_clean,
        detection_rate=mw_result.detection_rate,
        bootstrap_freq_mean=boot_result.freq_mean,
        bootstrap_freq_std=boot_result.freq_std,
        bootstrap_ci_low=boot_result.freq_ci_low,
        bootstrap_ci_high=boot_result.freq_ci_high,
        freq_error_percent=freq_error_pct,
        confidence_score=score,
        confidence_tier=tier,
    )


def run_phase1_tidal_validation(
    station_id: str = DEFAULT_STATION_ID,
    year: int = DEFAULT_YEAR,
    data_dir: Path = None,
) -> Dict:
    """Phase 1: Detect known tidal constituents with Ab Astris pipeline."""
    if data_dir is None:
        data_dir = DATA_DIR

    print("\n" + "=" * 60)
    print("PHASE 1: TIDAL VALIDATION")
    print("=" * 60)

    # Download observed data
    print(f"\n[1/3] Downloading observed water levels ({year})...")
    df_obs = download_tide_data(station_id, year, year, "hourly_height", data_dir)

    # Prepare time series
    print("[2/3] Preparing time series...")
    time, signal = prepare_time_series(df_obs, "v")
    duration_days = (time[-1] - time[0]) / 86400
    print(f"  {len(time)} observations, {duration_days:.1f} days")

    # Analyze each constituent
    print(f"[3/3] Analyzing {len(TIDAL_CONSTITUENTS)} tidal constituents...\n")
    results = []

    for name in TIDAL_CONSTITUENTS:
        info = TIDAL_CONSTITUENTS[name]
        print(f"  {name} ({info['name']}):")

        result = analyze_constituent(time, signal, name)
        results.append(result)

        print(f"    Detected: {result.ls_frequency_cpd:.6f} cpd "
              f"(target: {result.target_freq_cpd:.6f} cpd)")
        print(f"    Error:    {result.freq_error_percent:.4f}%")
        print(f"    CV:       {result.multi_window_cv_clean:.6f}%")
        print(f"    Score:    {result.confidence_score:.0f} ({result.confidence_tier})")
        print()

    return {
        "station_id": station_id,
        "year": year,
        "n_observations": len(time),
        "duration_days": round(duration_days, 1),
        "constituents": results,
    }


# ============================================================================
# PHASE 2: RESIDUAL ANALYSIS
# ============================================================================

def run_phase2_residual_analysis(
    station_id: str = DEFAULT_STATION_ID,
    year: int = DEFAULT_YEAR,
    data_dir: Path = None,
) -> Dict:
    """Phase 2: Analyze tidal residual for anomalous periodicities."""
    if data_dir is None:
        data_dir = DATA_DIR

    print("\n" + "=" * 60)
    print("PHASE 2: RESIDUAL ANALYSIS")
    print("=" * 60)

    # Download observed and predicted
    print(f"\n[1/4] Downloading observed data ({year})...")
    df_obs = download_tide_data(station_id, year, year, "hourly_height", data_dir)

    print(f"[2/4] Downloading tidal predictions ({year})...")
    df_pred = download_tide_data(station_id, year, year, "predictions", data_dir)

    # Compute residual
    print("[3/4] Computing residual (observed - predicted)...")
    # Both DataFrames may have column 'v'; rename to avoid collision
    col_obs = "v" if "v" in df_obs.columns else df_obs.columns[0]
    col_pred = "v" if "v" in df_pred.columns else (
        "predicted_wl" if "predicted_wl" in df_pred.columns else df_pred.columns[0]
    )

    df_obs_renamed = df_obs[[col_obs]].rename(columns={col_obs: "observed"})
    df_pred_renamed = df_pred[[col_pred]].rename(columns={col_pred: "predicted"})

    df_merged = pd.merge(
        df_obs_renamed,
        df_pred_renamed,
        left_index=True,
        right_index=True,
        how="inner",
    )
    df_merged["residual"] = df_merged["observed"] - df_merged["predicted"]
    residual_std = float(df_merged["residual"].std())
    print(f"  Merged: {len(df_merged)} points, residual std = {residual_std:.4f} m")

    # Prepare time series
    time, residual = prepare_time_series(df_merged, "residual")

    # Broad-spectrum LS on residual (sub-tidal through supra-tidal)
    # Search from ~12-day period to ~3-hour period
    freq_min = 0.08 / 86400   # 0.08 cpd = ~12.5 day period
    freq_max = 8.0 / 86400    # 8 cpd = ~3 hour period

    print("[4/4] Running Lomb-Scargle on residual...")
    ls_result = run_lomb_scargle(time, residual, freq_min_hz=freq_min, freq_max_hz=freq_max)

    period_days = 1.0 / ls_result.frequency_cpd if ls_result.frequency_cpd > 0 else float("inf")
    print(f"  Peak: {ls_result.frequency_cpd:.4f} cpd (period = {period_days:.2f} days)")
    print(f"  Power: {ls_result.power:.4f}, FAP: {ls_result.fap:.2e}")

    # Multi-window on residual
    windows = create_windows(time, residual)
    mw_result = run_multi_window_validation(
        windows, ls_result.frequency_hz,
        freq_min_hz=freq_min, freq_max_hz=freq_max,
    )
    print(f"  CV: {mw_result.cv_clean:.3f}%, detection rate: {mw_result.detection_rate:.1%}")

    # Find additional peaks (top 5)
    top_peaks = []
    freqs = ls_result.frequencies_hz
    powers = ls_result.powers.copy()
    for _ in range(5):
        idx = np.argmax(powers)
        peak_hz = freqs[idx]
        peak_cpd = peak_hz * 86400
        peak_power = float(powers[idx])
        peak_period = 1.0 / peak_cpd if peak_cpd > 0 else float("inf")
        top_peaks.append({
            "frequency_cpd": round(peak_cpd, 4),
            "period_days": round(peak_period, 2),
            "power": round(peak_power, 4),
        })
        # Zero out region around this peak to find next
        mask = np.abs(freqs - peak_hz) < (peak_hz * 0.02)
        powers[mask] = 0

    return {
        "station_id": station_id,
        "year": year,
        "n_points": len(time),
        "residual_std_m": round(residual_std, 4),
        "peak_frequency_cpd": round(ls_result.frequency_cpd, 4),
        "peak_period_days": round(period_days, 2),
        "peak_power": round(ls_result.power, 4),
        "peak_fap": float(ls_result.fap),
        "multi_window_cv": round(mw_result.cv_clean, 3),
        "detection_rate": round(mw_result.detection_rate, 3),
        "top_5_peaks": top_peaks,
    }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_tidal_validation(phase1_results: Dict, plot_dir: Path = None):
    """Generate dark-themed Phase 1 visualization (4-panel)."""
    if plot_dir is None:
        plot_dir = PLOT_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = phase1_results["constituents"]
    names = [r.constituent_name for r in results]
    n = len(names)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=COLORS["background"])
    fig.suptitle(
        f"Ab Astris Oceanography — Tidal Validation\n"
        f"Station {phase1_results['station_id']} ({phase1_results['year']})",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )

    # Panel 1: Frequency error
    ax = axes[0, 0]
    ax.set_facecolor(COLORS["panel"])
    errors = [r.freq_error_percent for r in results]
    bars = ax.bar(names, errors, color=COLORS["tidal"], alpha=0.8, edgecolor=COLORS["accent"], linewidth=0.5)
    ax.set_ylabel("Frequency Error (%)", color=COLORS["text"])
    ax.set_title("Detection Accuracy", color=COLORS["text"], fontweight="bold")
    ax.tick_params(colors=COLORS["text"])
    ax.grid(True, alpha=0.2, color=COLORS["grid"], axis="y")
    for i, (bar, err) in enumerate(zip(bars, errors)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{err:.4f}%", ha="center", va="bottom", fontsize=9, color=COLORS["text"])

    # Panel 2: CV comparison
    ax = axes[0, 1]
    ax.set_facecolor(COLORS["panel"])
    cvs = [r.multi_window_cv_clean for r in results]
    bars = ax.bar(names, cvs, color=COLORS["tidal"], alpha=0.8, edgecolor=COLORS["accent"], linewidth=0.5)
    ax.axhline(CV_EXCELLENT, color=COLORS["accent2"], linestyle="--", alpha=0.7,
               label=f"Excellent (<{CV_EXCELLENT}%)")
    ax.axhline(CV_GOOD, color=COLORS["warning"], linestyle="--", alpha=0.5,
               label=f"Good (<{CV_GOOD}%)")
    ax.set_ylabel("CV (%)", color=COLORS["text"])
    ax.set_title("Multi-Window Stability", color=COLORS["text"], fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(colors=COLORS["text"])
    ax.grid(True, alpha=0.2, color=COLORS["grid"], axis="y")
    for i, (bar, cv) in enumerate(zip(bars, cvs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{cv:.4f}%", ha="center", va="bottom", fontsize=9, color=COLORS["text"])

    # Panel 3: Confidence scores
    ax = axes[1, 0]
    ax.set_facecolor(COLORS["panel"])
    scores = [r.confidence_score for r in results]
    tiers = [r.confidence_tier for r in results]
    bar_colors = [COLORS["tidal"] if s >= 90 else COLORS["warning"] if s >= 75 else COLORS["accent2"]
                  for s in scores]
    bars = ax.barh(names, scores, color=bar_colors, alpha=0.8, edgecolor=COLORS["accent"], linewidth=0.5)
    ax.axvline(90, color=COLORS["accent"], linestyle="--", alpha=0.5, label="CONFIRMED (90)")
    ax.axvline(75, color=COLORS["warning"], linestyle="--", alpha=0.3, label="PROBABLE (75)")
    ax.set_xlabel("Confidence Score", color=COLORS["text"])
    ax.set_title("Confidence Scores", color=COLORS["text"], fontweight="bold")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(colors=COLORS["text"])
    ax.grid(True, alpha=0.2, color=COLORS["grid"], axis="x")
    for bar, score, tier in zip(bars, scores, tiers):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.0f} ({tier})", va="center", fontsize=9, color=COLORS["text"])

    # Panel 4: Cross-domain CV comparison
    ax = axes[1, 1]
    ax.set_facecolor(COLORS["panel"])
    ocean_cv = np.mean(cvs)
    domains = ["Variable\nStars", "Bearings", "Tidal\n(this work)", "Volcanoes"]
    cv_values = [
        CROSS_DOMAIN_REFS["variable_stars"]["cv_mean"],
        CROSS_DOMAIN_REFS["bearings"]["cv_mean"],
        ocean_cv,
        CROSS_DOMAIN_REFS["volcanoes"]["cv_mean"],
    ]
    domain_colors = [COLORS["accent3"], COLORS["accent3"], COLORS["tidal"], COLORS["accent3"]]
    bars = ax.bar(domains, cv_values, color=domain_colors, alpha=0.8,
                  edgecolor=COLORS["accent"], linewidth=0.5)
    ax.set_ylabel("CV (%)", color=COLORS["text"])
    ax.set_title("Cross-Domain CV Comparison", color=COLORS["text"], fontweight="bold")
    ax.set_yscale("log")
    ax.tick_params(colors=COLORS["text"])
    ax.grid(True, alpha=0.2, color=COLORS["grid"], axis="y")
    for bar, cv_val in zip(bars, cv_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                f"{cv_val:.4f}%", ha="center", fontsize=9, color=COLORS["text"])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = plot_dir / f"tidal_validation_{phase1_results['station_id']}_{phase1_results['year']}.png"
    fig.savefig(out_path, dpi=150, facecolor=COLORS["background"], bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")


# ============================================================================
# RESULTS OUTPUT
# ============================================================================

def _serialize_constituent(r: ConstituentResult) -> Dict:
    """Serialize a ConstituentResult, converting numpy types."""
    d = asdict(r)
    for k, v in d.items():
        if isinstance(v, (np.floating, np.float64)):
            d[k] = float(v)
        elif isinstance(v, (np.integer, np.int64)):
            d[k] = int(v)
    return d


def save_results(
    phase1_results: Dict,
    phase2_results: Dict,
    output_path: Path = None,
) -> Dict:
    """Generate comprehensive JSON results file."""
    if output_path is None:
        output_path = RESULTS_JSON

    constituents = phase1_results["constituents"]
    cvs = [r.multi_window_cv_clean for r in constituents]
    errors = [r.freq_error_percent for r in constituents]
    scores = [r.confidence_score for r in constituents]
    cv_mean = float(np.mean(cvs))

    output = {
        "experiment": "oceanography_tidal_validation",
        "methodology": "Ab Astris (Lomb-Scargle + Multi-window + Bootstrap)",
        "data_source": "NOAA CO-OPS Tide Gauges",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "sampling_interval_sec": SAMPLING_INTERVAL_SEC,
            "n_windows": N_WINDOWS,
            "window_duration_days": WINDOW_DURATION_DAYS,
            "n_bootstrap": N_BOOTSTRAP,
            "n_freq_points": N_FREQ_POINTS,
            "diurnal_band_cpd": [DIURNAL_BAND_HZ[0] * 86400, DIURNAL_BAND_HZ[1] * 86400],
            "semidiurnal_band_cpd": [SEMIDIURNAL_BAND_HZ[0] * 86400, SEMIDIURNAL_BAND_HZ[1] * 86400],
        },
        "phase1_tidal_validation": {
            "station_id": phase1_results["station_id"],
            "year": phase1_results["year"],
            "n_observations": phase1_results["n_observations"],
            "duration_days": phase1_results["duration_days"],
            "constituents": [_serialize_constituent(r) for r in constituents],
            "summary": {
                "mean_cv": round(cv_mean, 6),
                "max_cv": round(float(np.max(cvs)), 6),
                "min_cv": round(float(np.min(cvs)), 6),
                "mean_freq_error_pct": round(float(np.mean(errors)), 4),
                "max_freq_error_pct": round(float(np.max(errors)), 4),
                "mean_confidence_score": round(float(np.mean(scores)), 1),
                "all_confirmed": all(s >= 90 for s in scores),
            },
        },
        "phase2_residual_analysis": phase2_results,
        "cross_domain_comparison": {
            "variable_stars": {
                "domain": "Astronomy",
                "system": "Variable stars",
                "cv_mean": CROSS_DOMAIN_REFS["variable_stars"]["cv_mean"],
                "detection_rate": CROSS_DOMAIN_REFS["variable_stars"]["detection_rate"],
                "constraint": "Stellar physics",
            },
            "bearings": {
                "domain": "Industrial",
                "system": "Bearing faults (CWRU)",
                "cv_mean": CROSS_DOMAIN_REFS["bearings"]["cv_mean"],
                "detection_rate": CROSS_DOMAIN_REFS["bearings"]["detection_rate"],
                "constraint": "Mechanical resonance",
            },
            "oceanography_tidal": {
                "domain": "Oceanography",
                "system": "Tidal constituents (NOAA)",
                "cv_mean": round(cv_mean, 6),
                "detection_rate": round(float(np.mean([r.detection_rate for r in constituents])), 3),
                "constraint": "Gravitational (orbital mechanics)",
            },
            "volcanoes": {
                "domain": "Geophysics",
                "system": "Volcanic tremor",
                "cv_mean": CROSS_DOMAIN_REFS["volcanoes"]["cv_mean"],
                "detection_rate": CROSS_DOMAIN_REFS["volcanoes"]["detection_rate"],
                "constraint": "Conduit resonance",
            },
            "crypto": {
                "domain": "Financial",
                "system": "BTC cryptocurrency",
                "cv_mean": CROSS_DOMAIN_REFS["crypto"]["cv_mean"],
                "detection_rate": CROSS_DOMAIN_REFS["crypto"]["detection_rate"],
                "constraint": "Behavioural (correctly rejected)",
            },
        },
        "validation_status": "VALIDATED" if all(s >= 75 for s in scores) else "REVIEW",
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the complete oceanography validation experiment."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Ab Astris Oceanography Validation Experiment")
    print("=" * 60)
    print(f"  Station:  {DEFAULT_STATION_ID} (San Francisco)")
    print(f"  Year:     {DEFAULT_YEAR}")
    print(f"  Pipeline: Bandpass → Lomb-Scargle → Multi-window → Bootstrap")
    print(f"  Note:     No Hilbert envelope (tidal = direct periodic signal)")

    # Phase 1: Tidal validation
    phase1 = run_phase1_tidal_validation()

    # Phase 2: Residual analysis
    phase2 = run_phase2_residual_analysis()

    # Outputs
    plot_tidal_validation(phase1)
    results = save_results(phase1, phase2)

    # Summary table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Constituent':<6} {'Target (cpd)':<14} {'Detected (cpd)':<16} "
          f"{'Error %':<10} {'CV %':<12} {'Score':<8} {'Tier'}")
    print("-" * 80)
    for r in phase1["constituents"]:
        print(f"{r.constituent_name:<6} {r.target_freq_cpd:<14.6f} {r.ls_frequency_cpd:<16.6f} "
              f"{r.freq_error_percent:<10.4f} {r.multi_window_cv_clean:<12.6f} "
              f"{r.confidence_score:<8.0f} {r.confidence_tier}")

    summary = results["phase1_tidal_validation"]["summary"]
    print(f"\nMean CV:              {summary['mean_cv']:.6f}%")
    print(f"Mean Frequency Error: {summary['mean_freq_error_pct']:.4f}%")
    print(f"All Confirmed:        {summary['all_confirmed']}")

    print(f"\nPhase 2 Peak Residual: {phase2['peak_frequency_cpd']:.4f} cpd "
          f"(period = {phase2['peak_period_days']:.2f} days)")
    print(f"Residual CV:           {phase2['multi_window_cv']:.3f}%")

    print(f"\nValidation Status: {results['validation_status']}")

    return results


if __name__ == "__main__":
    main()

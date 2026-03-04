"""
Ab Astris Negative Control: Sunspot Number (Solar Cycle)

The most rhetorically powerful negative control — astronomical, like Ab Astris's
origin story. The ~11-year solar cycle is real physics (MHD dynamo), but chaotic
rather than deterministic. Cycle lengths vary 9–14 years, amplitudes 2–3x.

Two analyses:
  A. Monthly 1749–2024: ~25 complete cycles, targeting ~11-year period
  B. Daily 1980–2024: higher resolution, targeting ~27-day Carrington rotation

Expected: CV 5–25% (detectable but unstable), contrasting with stellar pulsation
CV < 0.01%.

Data source: SILSO (Royal Observatory of Belgium), CC BY-NC 4.0.

Usage:
    python negative_controls/sunspot_experiment.py
"""

import argparse
import json
import ssl
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULTS_JSON = OUTPUT_DIR / "sunspot_results.json"

# SILSO data URLs
SILSO_MONTHLY_URL = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php"
SILSO_DAILY_URL = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"

# Analysis A: Monthly, 11-year cycle
MONTHLY_FREQ_MIN_CY = 0.05   # cycles/year (20-year period)
MONTHLY_FREQ_MAX_CY = 0.20   # cycles/year (5-year period)
MONTHLY_N_WINDOWS = 6         # Per brief: 6 windows, ~46 years each
SOLAR_CYCLE_YEARS = 11.0      # Expected ~11-year period

# Analysis B: Daily, 27-day rotation (1980–present)
DAILY_START_YEAR = 1980
DAILY_FREQ_MIN_CPD = 0.01    # 0.01 cpd (100-day period)
DAILY_FREQ_MAX_CPD = 0.10    # 0.10 cpd (10-day period)
DAILY_N_WINDOWS = 8
CARRINGTON_PERIOD_DAYS = 27.2753  # Carrington rotation period

# Shared parameters
N_BOOTSTRAP = 100
N_FREQ_POINTS = 10000
FREQ_TOLERANCE = 0.05

# Confidence scoring thresholds
CV_EXCELLENT = 0.01
CV_GOOD = 0.1
CV_FAIR = 1.0

# Dark theme
COLORS = {
    "background": "#0d1117",
    "panel": "#161b22",
    "accent": "#00b4d8",
    "accent2": "#ff6b6b",
    "accent3": "#4ecdc4",
    "text": "#c9d1d9",
    "grid": "#21262d",
    "sunspot": "#f0ad4e",
    "sunspot2": "#9b59b6",
}

# Cross-domain references
CROSS_DOMAIN_REFS = {
    "variable_stars": {"cv_mean": 0.005, "detection_rate": 1.00, "domain": "Astronomy (stellar)"},
    "bearings": {"cv_mean": 0.008, "detection_rate": 1.00, "domain": "Industrial"},
    "tides": {"cv_mean": 0.171, "detection_rate": 1.00, "domain": "Oceanography"},
    "volcanoes": {"cv_mean": 3.96, "detection_rate": 0.997, "domain": "Geophysics"},
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
class SunspotResult:
    """Complete analysis results for one sunspot analysis."""
    analysis_name: str
    description: str

    # Data
    n_data_points: int
    time_span_years: float

    # Lomb-Scargle
    ls_frequency_hz: float
    ls_period_years: float
    ls_period_days: float
    ls_power: float
    ls_fap: float

    # Multi-window
    multi_window_cv: float
    multi_window_cv_clean: float
    detection_rate: float
    n_windows: int

    # Bootstrap
    bootstrap_freq_mean: float
    bootstrap_freq_std: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    # Scoring
    confidence_score: float
    confidence_tier: str


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_sunspot_data(
    data_type: str = "monthly",
    data_dir: Path = None,
) -> pd.DataFrame:
    """Download SILSO sunspot data with caching.

    Args:
        data_type: "monthly" or "daily"
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    if data_type == "monthly":
        url = SILSO_MONTHLY_URL
        cache_file = data_dir / "SN_m_tot_V2.0.csv"
        col_names = ["year", "month", "decimal_year",
                     "sunspot_number", "std_dev", "n_obs", "definitive"]
    elif data_type == "daily":
        url = SILSO_DAILY_URL
        cache_file = data_dir / "SN_d_tot_V2.0.csv"
        col_names = ["year", "month", "day", "decimal_year",
                     "sunspot_number", "std_dev", "n_obs", "definitive"]
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    if cache_file.exists():
        print(f"  Loading cached: {cache_file.name}")
    else:
        print(f"  Downloading SILSO {data_type} data...")
        # Handle macOS SSL certificate verification issues
        ctx = ssl.create_default_context()
        try:
            urllib.request.urlretrieve(url, cache_file)
        except urllib.error.URLError:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ctx) as response:
                cache_file.write_bytes(response.read())
        print(f"  Cached to: {cache_file.name}")

    df = pd.read_csv(
        cache_file, sep=";", header=None, names=col_names,
        skipinitialspace=True,
    )

    # Remove missing values (coded as -1)
    n_before = len(df)
    df = df[df["sunspot_number"] >= 0].copy()
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"  Removed {n_removed} missing values (-1)")

    print(f"  {data_type.capitalize()} data: {len(df)} records, "
          f"{df['decimal_year'].min():.1f}–{df['decimal_year'].max():.1f}")

    return df


def prepare_monthly_time_series(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare monthly sunspot data for analysis.

    Returns (time_seconds, sunspot_number) with time in seconds since first obs.
    """
    time_years = df["decimal_year"].values.astype(np.float64)
    time_sec = (time_years - time_years[0]) * 365.25 * 86400
    values = df["sunspot_number"].values.astype(np.float64)
    return time_sec, values


def prepare_daily_time_series(
    df: pd.DataFrame,
    start_year: int = DAILY_START_YEAR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare daily sunspot data (1980+) for analysis."""
    df_recent = df[df["year"] >= start_year].copy()
    if df_recent.empty:
        raise ValueError(f"No data after {start_year}")

    time_years = df_recent["decimal_year"].values.astype(np.float64)
    time_sec = (time_years - time_years[0]) * 365.25 * 86400
    values = df_recent["sunspot_number"].values.astype(np.float64)
    print(f"  Daily subset ({start_year}+): {len(values)} records, "
          f"{(time_sec[-1] - time_sec[0]) / (365.25 * 86400):.1f} years")
    return time_sec, values


# ============================================================================
# AB ASTRIS CORE PIPELINE
# ============================================================================

def run_lomb_scargle(
    time: np.ndarray,
    signal: np.ndarray,
    freq_min_hz: float = None,
    freq_max_hz: float = None,
    n_points: int = N_FREQ_POINTS,
) -> LSResult:
    """Run Lomb-Scargle periodogram following Ab Astris methodology."""
    # Normalize
    data_mean = np.mean(signal)
    data_std = np.std(signal)
    if data_std > 1e-15:
        data_norm = (signal - data_mean) / data_std
    else:
        data_norm = signal - data_mean

    frequencies_hz = np.linspace(freq_min_hz, freq_max_hz, n_points)

    ls = LombScargle(time, data_norm)
    powers = ls.power(frequencies_hz)

    best_idx = np.argmax(powers)
    best_freq_hz = frequencies_hz[best_idx]
    best_power = float(powers[best_idx])

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


def create_windows(
    time: np.ndarray,
    signal: np.ndarray,
    n_windows: int = 8,
    window_duration_sec: float = None,
) -> List[Dict]:
    """Create overlapping windows for multi-window validation."""
    total_duration = time[-1] - time[0]
    if window_duration_sec is None:
        # Default: windows cover ~50% of total, with 50% overlap
        window_duration_sec = total_duration * 0.3
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
        if np.sum(mask) < 10:
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

    freq_mean = np.mean(detected_freqs)
    freq_std = np.std(detected_freqs)
    cv_raw = (freq_std / freq_mean * 100) if freq_mean > 0 else 0.0

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

    detections = np.abs(detected_freqs - target_freq_hz) / target_freq_hz < FREQ_TOLERANCE
    detection_rate = float(np.mean(detections))

    return MultiWindowResult(
        detected_freqs=detected_freqs.tolist(),
        cv_raw=float(cv_raw),
        cv_clean=float(cv_clean),
        detection_rate=detection_rate,
        outlier_count=outlier_count,
    )


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
        indices = np.sort(indices)
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


def compute_confidence_score(
    ls_result: LSResult,
    mw_result: MultiWindowResult,
    target_freq_hz: float = None,
) -> Tuple[float, str]:
    """Compute confidence score (0-100) following Ab Astris framework."""
    score = 0.0

    # 1. Frequency accuracy (30 pts)
    if target_freq_hz is not None:
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
    # No target → no accuracy points (exploratory)

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
# ANALYSIS
# ============================================================================

def analyze_monthly_11yr(
    time: np.ndarray,
    values: np.ndarray,
) -> Tuple[SunspotResult, LSResult]:
    """Analysis A: ~11-year solar cycle from monthly data."""
    print("\n  === Analysis A: 11-Year Solar Cycle (Monthly) ===")

    # Convert frequency range to Hz
    sec_per_year = 365.25 * 86400
    freq_min_hz = MONTHLY_FREQ_MIN_CY / sec_per_year
    freq_max_hz = MONTHLY_FREQ_MAX_CY / sec_per_year
    target_freq_hz = (1.0 / SOLAR_CYCLE_YEARS) / sec_per_year

    time_span_years = (time[-1] - time[0]) / sec_per_year

    # Lomb-Scargle
    print("  Running Lomb-Scargle...")
    ls_result = run_lomb_scargle(time, values,
                                 freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz)
    period_years = 1.0 / (ls_result.frequency_hz * sec_per_year)
    print(f"    Peak: {ls_result.frequency_hz * sec_per_year:.4f} cy/yr "
          f"({period_years:.2f} years)")
    print(f"    Power: {ls_result.power:.4f}, FAP: {ls_result.fap:.2e}")

    # Multi-window: 6 windows, ~46 years each
    window_duration = 46 * sec_per_year
    print(f"  Running multi-window ({MONTHLY_N_WINDOWS} windows, ~46 yr each)...")
    windows = create_windows(time, values,
                             n_windows=MONTHLY_N_WINDOWS,
                             window_duration_sec=window_duration)
    mw_result = run_multi_window_validation(
        windows, ls_result.frequency_hz,
        freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz,
    )
    print(f"    CV (clean): {mw_result.cv_clean:.2f}%")
    print(f"    Detection rate: {mw_result.detection_rate:.2f}")

    # Per-window periods for insight
    per_window_periods = [1.0 / (f * sec_per_year) for f in mw_result.detected_freqs]
    print(f"    Per-window periods: {[f'{p:.1f}' for p in per_window_periods]} years")

    # Bootstrap
    print("  Running bootstrap (n=100)...")
    boot_result = run_bootstrap(time, values, freq_min_hz=freq_min_hz,
                                freq_max_hz=freq_max_hz)
    boot_period = 1.0 / (boot_result.freq_mean * sec_per_year)
    print(f"    Mean period: {boot_period:.2f} years")

    # Confidence (use known ~11yr as target)
    score, tier = compute_confidence_score(ls_result, mw_result, target_freq_hz)
    print(f"    Confidence: {score}/100 -> {tier}")

    result = SunspotResult(
        analysis_name="11-year solar cycle",
        description="Monthly sunspot numbers 1749-present, targeting ~11-year MHD dynamo cycle",
        n_data_points=len(values),
        time_span_years=time_span_years,
        ls_frequency_hz=ls_result.frequency_hz,
        ls_period_years=period_years,
        ls_period_days=period_years * 365.25,
        ls_power=ls_result.power,
        ls_fap=ls_result.fap,
        multi_window_cv=mw_result.cv_raw,
        multi_window_cv_clean=mw_result.cv_clean,
        detection_rate=mw_result.detection_rate,
        n_windows=len(windows),
        bootstrap_freq_mean=boot_result.freq_mean,
        bootstrap_freq_std=boot_result.freq_std,
        bootstrap_ci_low=boot_result.freq_ci_low,
        bootstrap_ci_high=boot_result.freq_ci_high,
        confidence_score=score,
        confidence_tier=tier,
    )

    return result, ls_result


def analyze_daily_27day(
    time: np.ndarray,
    values: np.ndarray,
) -> Tuple[SunspotResult, LSResult]:
    """Analysis B: ~27-day Carrington rotation from daily data (1980+)."""
    print("\n  === Analysis B: 27-Day Solar Rotation (Daily, 1980+) ===")

    freq_min_hz = DAILY_FREQ_MIN_CPD / 86400
    freq_max_hz = DAILY_FREQ_MAX_CPD / 86400
    target_freq_hz = (1.0 / CARRINGTON_PERIOD_DAYS) / 86400

    time_span_years = (time[-1] - time[0]) / (365.25 * 86400)

    # Lomb-Scargle
    print("  Running Lomb-Scargle...")
    ls_result = run_lomb_scargle(time, values,
                                 freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz)
    period_days = 1.0 / ls_result.frequency_cpd if ls_result.frequency_cpd > 0 else float("inf")
    print(f"    Peak: {ls_result.frequency_cpd:.4f} cpd ({period_days:.1f} days)")
    print(f"    Power: {ls_result.power:.4f}, FAP: {ls_result.fap:.2e}")

    # Multi-window
    # Window: ~5 years each (covers solar max and min periods)
    window_duration = 5 * 365.25 * 86400
    print(f"  Running multi-window ({DAILY_N_WINDOWS} windows, ~5 yr each)...")
    windows = create_windows(time, values,
                             n_windows=DAILY_N_WINDOWS,
                             window_duration_sec=window_duration)
    mw_result = run_multi_window_validation(
        windows, ls_result.frequency_hz,
        freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz,
    )
    print(f"    CV (clean): {mw_result.cv_clean:.2f}%")
    print(f"    Detection rate: {mw_result.detection_rate:.2f}")

    per_window_periods = [1.0 / (f * 86400) if f > 0 else float("inf")
                          for f in mw_result.detected_freqs]
    print(f"    Per-window periods: {[f'{p:.1f}' for p in per_window_periods]} days")

    # Bootstrap
    print("  Running bootstrap (n=100)...")
    boot_result = run_bootstrap(time, values, freq_min_hz=freq_min_hz,
                                freq_max_hz=freq_max_hz)

    # Confidence (use Carrington period as target)
    score, tier = compute_confidence_score(ls_result, mw_result, target_freq_hz)
    print(f"    Confidence: {score}/100 -> {tier}")

    period_years = period_days / 365.25

    result = SunspotResult(
        analysis_name="27-day solar rotation",
        description="Daily sunspot numbers 1980-present, targeting ~27.3-day Carrington rotation",
        n_data_points=len(values),
        time_span_years=time_span_years,
        ls_frequency_hz=ls_result.frequency_hz,
        ls_period_years=period_years,
        ls_period_days=period_days,
        ls_power=ls_result.power,
        ls_fap=ls_result.fap,
        multi_window_cv=mw_result.cv_raw,
        multi_window_cv_clean=mw_result.cv_clean,
        detection_rate=mw_result.detection_rate,
        n_windows=len(windows),
        bootstrap_freq_mean=boot_result.freq_mean,
        bootstrap_freq_std=boot_result.freq_std,
        bootstrap_ci_low=boot_result.freq_ci_low,
        bootstrap_ci_high=boot_result.freq_ci_high,
        confidence_score=score,
        confidence_tier=tier,
    )

    return result, ls_result


# ============================================================================
# PLOTTING
# ============================================================================

def plot_sunspot_results(
    monthly_result: SunspotResult,
    daily_result: SunspotResult,
    monthly_ls: LSResult,
    daily_ls: LSResult,
    monthly_time: np.ndarray,
    monthly_values: np.ndarray,
    daily_time: np.ndarray,
    daily_values: np.ndarray,
):
    """Create 6-panel dark-themed sunspot validation figure."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.patch.set_facecolor(COLORS["background"])

    for ax in axes.flat:
        ax.set_facecolor(COLORS["panel"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])

    sec_per_year = 365.25 * 86400

    # Panel 1: Monthly time series 1749–present
    ax = axes[0, 0]
    years = monthly_time / sec_per_year + 1749  # approximate
    ax.plot(years, monthly_values, color=COLORS["sunspot"], linewidth=0.5)
    ax.set_xlabel("Year", color=COLORS["text"])
    ax.set_ylabel("Monthly Sunspot Number", color=COLORS["text"])
    ax.set_title("Sunspot Number 1749–Present", color=COLORS["text"], fontweight="bold")
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 2: 11-year periodogram
    ax = axes[0, 1]
    freqs_cy = monthly_ls.frequencies_hz * sec_per_year  # cycles/year
    periods_yr = 1.0 / freqs_cy
    ax.plot(periods_yr, monthly_ls.powers, color=COLORS["sunspot"], linewidth=1)
    detected_period = monthly_result.ls_period_years
    ax.axvline(detected_period, color=COLORS["accent"], linestyle="--",
               label=f"Peak: {detected_period:.2f} yr")
    ax.axvline(11.0, color=COLORS["accent3"], linestyle=":", alpha=0.7,
               label="Expected: 11.0 yr")
    ax.set_xlabel("Period (years)", color=COLORS["text"])
    ax.set_ylabel("Lomb-Scargle Power", color=COLORS["text"])
    ax.set_title("11-Year Cycle Periodogram", color=COLORS["text"], fontweight="bold")
    ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 3: Multi-window stability for 11-year
    ax = axes[1, 0]
    labels = ["CV (%)", "Detection Rate", "Score"]
    values = [monthly_result.multi_window_cv_clean,
              monthly_result.detection_rate * 100,
              monthly_result.confidence_score]
    bar_colors = [COLORS["sunspot"], COLORS["accent"], COLORS["accent3"]]
    bars = ax.bar(labels, values, color=bar_colors, edgecolor=COLORS["grid"], alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", color=COLORS["text"], fontsize=11)
    ax.set_ylabel("Value", color=COLORS["text"])
    ax.set_title(f"11-Year Analysis Metrics (Tier: {monthly_result.confidence_tier})",
                 color=COLORS["text"], fontweight="bold")
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 4: Daily time series 1980+
    ax = axes[1, 1]
    days = daily_time / 86400
    years_daily = days / 365.25 + DAILY_START_YEAR
    ax.plot(years_daily, daily_values, color=COLORS["sunspot2"], linewidth=0.3, alpha=0.6)
    # Add smoothed (90-day running average)
    if len(daily_values) > 90:
        kernel = np.ones(90) / 90
        smoothed = np.convolve(daily_values, kernel, mode="valid")
        ax.plot(years_daily[:len(smoothed)], smoothed, color=COLORS["sunspot"],
                linewidth=1.5, label="90-day avg")
        ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
                  labelcolor=COLORS["text"], fontsize=9)
    ax.set_xlabel("Year", color=COLORS["text"])
    ax.set_ylabel("Daily Sunspot Number", color=COLORS["text"])
    ax.set_title(f"Daily Sunspot Number {DAILY_START_YEAR}–Present",
                 color=COLORS["text"], fontweight="bold")
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 5: 27-day periodogram
    ax = axes[2, 0]
    freqs_cpd = daily_ls.frequencies_hz * 86400
    periods_days = 1.0 / freqs_cpd
    ax.plot(periods_days, daily_ls.powers, color=COLORS["sunspot2"], linewidth=1)
    detected_period_d = daily_result.ls_period_days
    ax.axvline(detected_period_d, color=COLORS["accent"], linestyle="--",
               label=f"Peak: {detected_period_d:.1f} d")
    ax.axvline(CARRINGTON_PERIOD_DAYS, color=COLORS["accent3"], linestyle=":", alpha=0.7,
               label=f"Carrington: {CARRINGTON_PERIOD_DAYS:.1f} d")
    ax.set_xlabel("Period (days)", color=COLORS["text"])
    ax.set_ylabel("Lomb-Scargle Power", color=COLORS["text"])
    ax.set_title("27-Day Rotation Periodogram", color=COLORS["text"], fontweight="bold")
    ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 6: Within-astronomy contrast
    ax = axes[2, 1]
    domains = ["Stellar\npulsation", "Sunspot\n11-yr", "Sunspot\n27-day"]
    cv_vals = [0.005, monthly_result.multi_window_cv_clean, daily_result.multi_window_cv_clean]
    bar_colors = [COLORS["accent3"], COLORS["sunspot"], COLORS["sunspot2"]]
    bars = ax.bar(domains, cv_vals, color=bar_colors, edgecolor=COLORS["grid"], alpha=0.8)
    for bar, cv in zip(bars, cv_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{cv:.2f}%", ha="center", color=COLORS["text"], fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("CV (%) — log scale", color=COLORS["text"])
    ax.set_title("Within-Astronomy CV Contrast", color=COLORS["text"], fontweight="bold")
    ax.axhline(15, color=COLORS["accent2"], linestyle=":", linewidth=1.5,
               label="15% threshold")
    ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    plt.tight_layout()
    plot_path = PLOT_DIR / "sunspot_validation.png"
    fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")


# ============================================================================
# RESULTS SERIALIZATION
# ============================================================================

def serialize_result(r: SunspotResult) -> Dict:
    """Convert SunspotResult to JSON-safe dict."""
    sec_per_year = 365.25 * 86400
    return {
        "analysis_name": r.analysis_name,
        "description": r.description,
        "n_data_points": int(r.n_data_points),
        "time_span_years": float(r.time_span_years),
        "ls_frequency_hz": float(r.ls_frequency_hz),
        "ls_frequency_cy_per_year": float(r.ls_frequency_hz * sec_per_year),
        "ls_period_years": float(r.ls_period_years),
        "ls_period_days": float(r.ls_period_days),
        "ls_power": float(r.ls_power),
        "ls_fap": float(r.ls_fap),
        "multi_window_cv": float(r.multi_window_cv),
        "multi_window_cv_clean": float(r.multi_window_cv_clean),
        "detection_rate": float(r.detection_rate),
        "n_windows": int(r.n_windows),
        "bootstrap_freq_mean_hz": float(r.bootstrap_freq_mean),
        "bootstrap_freq_std_hz": float(r.bootstrap_freq_std),
        "bootstrap_ci_low_hz": float(r.bootstrap_ci_low),
        "bootstrap_ci_high_hz": float(r.bootstrap_ci_high),
        "confidence_score": float(r.confidence_score),
        "confidence_tier": r.confidence_tier,
    }


def save_results(
    monthly_result: SunspotResult,
    daily_result: SunspotResult,
):
    """Save results to JSON."""
    output = {
        "experiment": "Ab Astris Negative Control: Sunspot Number",
        "methodology": "Ab Astris (Lomb-Scargle + Multi-window + Bootstrap)",
        "data_source": "SILSO (Royal Observatory of Belgium), CC BY-NC 4.0",
        "timestamp": datetime.now().isoformat(),
        "analyses": {
            "monthly_11yr": serialize_result(monthly_result),
            "daily_27day": serialize_result(daily_result),
        },
        "summary": {
            "11yr_cycle_cv": float(monthly_result.multi_window_cv_clean),
            "11yr_cycle_period": float(monthly_result.ls_period_years),
            "11yr_cycle_tier": monthly_result.confidence_tier,
            "27day_rotation_cv": float(daily_result.multi_window_cv_clean),
            "27day_rotation_period": float(daily_result.ls_period_days),
            "27day_rotation_tier": daily_result.confidence_tier,
            "interpretation": (
                f"The solar dynamo produces CV of {monthly_result.multi_window_cv_clean:.1f}% "
                f"for the ~11-year cycle and {daily_result.multi_window_cv_clean:.1f}% for the "
                f"~27-day rotation, placing them orders of magnitude above stellar pulsation "
                f"CV of <0.01% — despite both being astronomical periodic signals. "
                f"This within-domain contrast demonstrates that the CV metric discriminates "
                f"between deterministic and chaotic periodic forcing, not merely between domains."
            ),
        },
        "cross_domain_comparison": {
            name: ref for name, ref in CROSS_DOMAIN_REFS.items()
        },
    }

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {RESULTS_JSON}")


# ============================================================================
# MAIN
# ============================================================================

def run_sunspot_experiment() -> Dict[str, SunspotResult]:
    """Run the full sunspot negative control experiment."""
    print(f"\n{'='*70}")
    print(f"  AB ASTRIS NEGATIVE CONTROL: SUNSPOT NUMBER")
    print(f"{'='*70}")

    # Download data
    print("\n--- Downloading data ---")
    df_monthly = download_sunspot_data("monthly")
    df_daily = download_sunspot_data("daily")

    # Prepare time series
    monthly_time, monthly_values = prepare_monthly_time_series(df_monthly)
    daily_time, daily_values = prepare_daily_time_series(df_daily)

    # Analysis A: 11-year cycle
    monthly_result, monthly_ls = analyze_monthly_11yr(monthly_time, monthly_values)

    # Analysis B: 27-day rotation
    daily_result, daily_ls = analyze_daily_27day(daily_time, daily_values)

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  11-year cycle: Period={monthly_result.ls_period_years:.2f} yr, "
          f"CV={monthly_result.multi_window_cv_clean:.1f}%, "
          f"Tier={monthly_result.confidence_tier}")
    print(f"  27-day rotation: Period={daily_result.ls_period_days:.1f} days, "
          f"CV={daily_result.multi_window_cv_clean:.1f}%, "
          f"Tier={daily_result.confidence_tier}")
    print(f"\n  Stellar pulsation reference: CV < 0.01%")
    print(f"  Within-astronomy contrast: {monthly_result.multi_window_cv_clean / 0.005:.0f}× "
          f"higher CV than stellar pulsation")

    # Save
    save_results(monthly_result, daily_result)
    plot_sunspot_results(
        monthly_result, daily_result,
        monthly_ls, daily_ls,
        monthly_time, monthly_values,
        daily_time, daily_values,
    )

    return {
        "monthly_11yr": monthly_result,
        "daily_27day": daily_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ab Astris Negative Control: Sunspot Number"
    )
    args = parser.parse_args()
    run_sunspot_experiment()


if __name__ == "__main__":
    main()

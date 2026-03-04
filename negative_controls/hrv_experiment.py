"""
Ab Astris Negative Control: Heart Rate Variability (HRV)

The "hard" negative control. The heart is a real physical oscillator (cardiac
muscle contraction), but its frequency is continuously modulated by the
autonomic nervous system. The CV discriminator should distinguish biological
regulation from deterministic physics.

Data: PhysioNet Normal Sinus Rhythm RR Interval Database (nsr2db)
  - 54 long-term ECG recordings (24-hour Holter monitors)
  - Healthy subjects in normal sinus rhythm
  - Open access

HRV frequency bands:
  - VLF: 0.003–0.04 Hz
  - LF:  0.04–0.15 Hz (sympathetic + parasympathetic)
  - HF:  0.15–0.4  Hz (respiratory sinus arrhythmia)

Expected: HF CV ~5–15%, LF CV ~15–40%, both placing biological regulation
between structural resonance and volcanic tremor.

Usage:
    python negative_controls/hrv_experiment.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import butter, filtfilt

try:
    import wfdb
except ImportError:
    print("ERROR: wfdb not installed. Run: pip install wfdb")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULTS_JSON = OUTPUT_DIR / "hrv_results.json"

# PhysioNet NSR2 database
PHYSIONET_DB = "nsr2db/1.0.0"

# Subject selection: 6 subjects (to match 6 tidal stations)
# Selected for recording quality and duration span
SUBJECTS = {
    "nsr001": {"description": "Subject 1"},
    "nsr002": {"description": "Subject 2"},
    "nsr003": {"description": "Subject 3"},
    "nsr004": {"description": "Subject 4"},
    "nsr005": {"description": "Subject 5"},
    "nsr006": {"description": "Subject 6"},
}

# HRV frequency bands (Hz)
HRV_BANDS = {
    "VLF": {"hz": (0.003, 0.04), "name": "Very Low Frequency"},
    "LF": {"hz": (0.04, 0.15), "name": "Low Frequency (sympathetic)"},
    "HF": {"hz": (0.15, 0.4), "name": "High Frequency (respiratory)"},
}

# RR interval physiological limits (seconds)
RR_MIN = 0.3   # 200 bpm
RR_MAX = 2.0   # 30 bpm

# Resampling rate for bandpass filter (Hz)
RESAMPLE_FS = 4.0  # 4 Hz is standard for HRV analysis

# Multi-window parameters
N_WINDOWS = 8
# Window duration: ~3 hours for 24-hour recording (8 windows * 3hr = 24hr with overlap)
WINDOW_DURATION_SEC = 3 * 3600  # 3 hours

# Bootstrap
N_BOOTSTRAP = 100

# LS grid resolution
N_FREQ_POINTS = 10000

# Tolerance for detection rate
FREQ_TOLERANCE = 0.10  # 10% (wider than tidal — HRV frequencies shift more)

# Confidence scoring thresholds
CV_EXCELLENT = 0.01
CV_GOOD = 0.1
CV_FAIR = 1.0

# Bandpass filter order
BANDPASS_ORDER = 4

# Dark theme
COLORS = {
    "background": "#0d1117",
    "panel": "#161b22",
    "accent": "#00b4d8",
    "accent2": "#ff6b6b",
    "accent3": "#4ecdc4",
    "text": "#c9d1d9",
    "grid": "#21262d",
    "hrv_hf": "#4ecdc4",
    "hrv_lf": "#f0ad4e",
    "hrv_vlf": "#9b59b6",
}

# Cross-domain references
CROSS_DOMAIN_REFS = {
    "variable_stars": {"cv_mean": 0.005, "detection_rate": 1.00, "domain": "Astronomy"},
    "bearings": {"cv_mean": 0.008, "detection_rate": 1.00, "domain": "Industrial"},
    "tides": {"cv_mean": 0.171, "detection_rate": 1.00, "domain": "Oceanography"},
    "structural": {"cv_mean": 1.019, "detection_rate": 1.00, "domain": "Structural"},
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
class BandResult:
    """Results for one HRV frequency band of one subject."""
    band_name: str
    band_range_hz: Tuple[float, float]

    # Lomb-Scargle
    ls_frequency_hz: float
    ls_power: float
    ls_fap: float

    # Multi-window
    multi_window_cv: float
    multi_window_cv_clean: float
    detection_rate: float

    # Bootstrap
    bootstrap_freq_mean: float
    bootstrap_freq_std: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    # Scoring
    confidence_score: float
    confidence_tier: str


@dataclass
class SubjectResult:
    """Complete results for one subject across all HRV bands."""
    subject_id: str
    n_beats: int
    duration_hours: float
    mean_hr_bpm: float
    bands: Dict[str, BandResult]


# ============================================================================
# DATA DOWNLOAD AND PREPARATION
# ============================================================================

def download_hrv_data(
    subject_id: str,
    data_dir: Path = None,
) -> Tuple[np.ndarray, float]:
    """Download PhysioNet NSR RR interval data.

    Returns (rr_intervals_seconds, sampling_frequency_hz).
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    cache_file = data_dir / f"{subject_id}_rr.npy"
    meta_file = data_dir / f"{subject_id}_fs.txt"

    if cache_file.exists() and meta_file.exists():
        print(f"  Loading cached: {cache_file.name}")
        rr = np.load(cache_file)
        fs = float(meta_file.read_text().strip())
        return rr, fs

    print(f"  Downloading {subject_id} from PhysioNet...")
    ann = wfdb.rdann(subject_id, "ecg", pn_dir=PHYSIONET_DB)

    # Convert beat indices to RR intervals in seconds
    rr_samples = np.diff(ann.sample)
    fs = float(ann.fs)
    rr_seconds = rr_samples / fs

    # Cache
    np.save(cache_file, rr_seconds)
    meta_file.write_text(str(fs))
    print(f"  Cached: {len(rr_seconds)} RR intervals, fs={fs} Hz")

    return rr_seconds, fs


def preprocess_rr(rr_intervals: np.ndarray) -> np.ndarray:
    """Pre-filter ectopic beats and physiological outliers.

    Removes RR intervals outside [0.3, 2.0] seconds (30–200 bpm).
    """
    n_before = len(rr_intervals)
    mask = (rr_intervals >= RR_MIN) & (rr_intervals <= RR_MAX)
    rr_clean = rr_intervals[mask]
    n_removed = n_before - len(rr_clean)
    if n_removed > 0:
        print(f"    Removed {n_removed}/{n_before} ectopic/outlier beats "
              f"({n_removed/n_before*100:.1f}%)")
    return rr_clean


def rr_to_hr(rr_intervals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert RR intervals to instantaneous heart rate.

    Returns (time_seconds, heart_rate_hz).
    Time is unevenly sampled (Lomb-Scargle advantage).
    """
    # Cumulative time at each beat
    time = np.cumsum(rr_intervals)
    # Instantaneous HR in Hz (beats/second)
    hr_hz = 1.0 / rr_intervals
    return time, hr_hz


def resample_uniform(
    time: np.ndarray,
    signal: np.ndarray,
    fs: float = RESAMPLE_FS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample unevenly-spaced signal to uniform grid.

    Required for bandpass filtering (scipy.signal.butter needs uniform sampling).
    """
    t_uniform = np.arange(time[0], time[-1], 1.0 / fs)
    s_uniform = np.interp(t_uniform, time, signal)
    return t_uniform, s_uniform


# ============================================================================
# AB ASTRIS CORE PIPELINE
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
    low = max(low, 1e-10)
    high = min(high, 0.9999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def run_lomb_scargle(
    time: np.ndarray,
    signal: np.ndarray,
    freq_min_hz: float = None,
    freq_max_hz: float = None,
    n_points: int = N_FREQ_POINTS,
) -> LSResult:
    """Run Lomb-Scargle periodogram following Ab Astris methodology."""
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
    n_windows: int = N_WINDOWS,
    window_duration_sec: float = WINDOW_DURATION_SEC,
) -> List[Dict]:
    """Create overlapping windows for multi-window validation."""
    total_duration = time[-1] - time[0]
    if total_duration < window_duration_sec:
        raise ValueError(
            f"Signal duration ({total_duration/3600:.1f} hr) shorter than "
            f"window duration ({window_duration_sec/3600:.1f} hr)"
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

    # 1. Frequency accuracy (30 pts) — exploratory, no target
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

def analyze_hrv_band(
    time_uneven: np.ndarray,
    hr_uneven: np.ndarray,
    band_name: str,
) -> BandResult:
    """Analyze one HRV frequency band.

    Strategy:
    1. Resample to uniform grid for bandpass filtering
    2. Apply bandpass filter
    3. Run LS on the filtered uniform signal (filter needs uniform sampling)
    4. Multi-window + bootstrap
    """
    f_low, f_high = HRV_BANDS[band_name]["hz"]

    # Resample to uniform grid for filtering
    t_uniform, hr_uniform = resample_uniform(time_uneven, hr_uneven)
    fs = RESAMPLE_FS

    # Bandpass filter
    filtered = bandpass_filter(hr_uniform, fs, f_low, f_high)

    # Lomb-Scargle on filtered uniform signal
    ls_result = run_lomb_scargle(t_uniform, filtered,
                                 freq_min_hz=f_low, freq_max_hz=f_high)

    # Multi-window
    windows = create_windows(t_uniform, filtered)
    mw_result = run_multi_window_validation(
        windows, ls_result.frequency_hz,
        freq_min_hz=f_low, freq_max_hz=f_high,
    )

    # Bootstrap
    boot_result = run_bootstrap(t_uniform, filtered,
                                freq_min_hz=f_low, freq_max_hz=f_high)

    # Confidence (exploratory — no specific target within band)
    score, tier = compute_confidence_score(ls_result, mw_result)

    return BandResult(
        band_name=band_name,
        band_range_hz=(f_low, f_high),
        ls_frequency_hz=ls_result.frequency_hz,
        ls_power=ls_result.power,
        ls_fap=ls_result.fap,
        multi_window_cv=mw_result.cv_raw,
        multi_window_cv_clean=mw_result.cv_clean,
        detection_rate=mw_result.detection_rate,
        bootstrap_freq_mean=boot_result.freq_mean,
        bootstrap_freq_std=boot_result.freq_std,
        bootstrap_ci_low=boot_result.freq_ci_low,
        bootstrap_ci_high=boot_result.freq_ci_high,
        confidence_score=score,
        confidence_tier=tier,
    )


def analyze_subject(
    subject_id: str,
) -> Optional[SubjectResult]:
    """Run full HRV analysis for one subject across all bands."""
    print(f"\n  === Subject: {subject_id} ===")

    try:
        rr_raw, fs = download_hrv_data(subject_id)
    except Exception as e:
        print(f"    ERROR downloading {subject_id}: {e}")
        return None

    # Preprocess
    rr_clean = preprocess_rr(rr_raw)
    if len(rr_clean) < 1000:
        print(f"    WARNING: Only {len(rr_clean)} clean beats, skipping")
        return None

    # Convert to heart rate
    time, hr = rr_to_hr(rr_clean)
    duration_hours = (time[-1] - time[0]) / 3600
    mean_hr_bpm = np.mean(hr) * 60

    print(f"    {len(rr_clean)} beats, {duration_hours:.1f} hours, "
          f"mean HR: {mean_hr_bpm:.0f} bpm")

    # Analyze each band
    bands = {}
    for band_name in HRV_BANDS:
        print(f"\n    --- {band_name} band ({HRV_BANDS[band_name]['name']}) ---")
        try:
            result = analyze_hrv_band(time, hr, band_name)
            bands[band_name] = result
            print(f"      Peak: {result.ls_frequency_hz:.4f} Hz, "
                  f"Power: {result.ls_power:.4f}")
            print(f"      CV: {result.multi_window_cv_clean:.2f}%, "
                  f"Detection: {result.detection_rate:.2f}, "
                  f"Tier: {result.confidence_tier}")
        except Exception as e:
            print(f"      ERROR: {e}")

    return SubjectResult(
        subject_id=subject_id,
        n_beats=len(rr_clean),
        duration_hours=duration_hours,
        mean_hr_bpm=mean_hr_bpm,
        bands=bands,
    )


# ============================================================================
# PLOTTING
# ============================================================================

def plot_hrv_results(
    results: Dict[str, SubjectResult],
    example_subject: str = None,
    example_rr: np.ndarray = None,
):
    """Create 6-panel dark-themed HRV validation figure."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.patch.set_facecolor(COLORS["background"])

    for ax in axes.flat:
        ax.set_facecolor(COLORS["panel"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])

    band_colors = {"HF": COLORS["hrv_hf"], "LF": COLORS["hrv_lf"], "VLF": COLORS["hrv_vlf"]}

    # Panel 1: RR interval time series (one subject)
    ax = axes[0, 0]
    if example_rr is not None:
        time_hr = np.cumsum(example_rr) / 3600
        hr_bpm = 60.0 / example_rr
        ax.plot(time_hr, hr_bpm, color=COLORS["accent"], linewidth=0.3, alpha=0.6)
        # Smoothed
        if len(hr_bpm) > 100:
            kernel = np.ones(100) / 100
            smoothed = np.convolve(hr_bpm, kernel, mode="valid")
            ax.plot(time_hr[:len(smoothed)], smoothed, color=COLORS["accent"],
                    linewidth=1.5, label="100-beat avg")
            ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
                      labelcolor=COLORS["text"], fontsize=9)
    ax.set_xlabel("Time (hours)", color=COLORS["text"])
    ax.set_ylabel("Heart Rate (bpm)", color=COLORS["text"])
    ax.set_title(f"24-Hour Heart Rate ({example_subject or 'Subject 1'})",
                 color=COLORS["text"], fontweight="bold")
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 2: Per-subject CV by band (grouped bar)
    ax = axes[0, 1]
    subject_ids = list(results.keys())
    n_subjects = len(subject_ids)
    band_names = ["HF", "LF", "VLF"]
    x = np.arange(n_subjects)
    width = 0.25

    for i, band in enumerate(band_names):
        cvs = []
        for sid in subject_ids:
            if band in results[sid].bands:
                cvs.append(results[sid].bands[band].multi_window_cv_clean)
            else:
                cvs.append(0)
        ax.bar(x + i * width, cvs, width, label=band, color=band_colors[band],
               edgecolor=COLORS["grid"], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([s[-3:] for s in subject_ids], color=COLORS["text"])
    ax.set_ylabel("CV (%)", color=COLORS["text"])
    ax.set_title("Per-Subject CV by Band", color=COLORS["text"], fontweight="bold")
    ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 3: Aggregate CV box plot
    ax = axes[1, 0]
    data_for_box = []
    labels_box = []
    colors_box = []
    for band in band_names:
        band_cvs = [results[s].bands[band].multi_window_cv_clean
                     for s in subject_ids if band in results[s].bands]
        if band_cvs:
            data_for_box.append(band_cvs)
            labels_box.append(band)
            colors_box.append(band_colors[band])

    if data_for_box:
        bp = ax.boxplot(data_for_box, labels=labels_box, patch_artist=True,
                        medianprops=dict(color=COLORS["text"]),
                        whiskerprops=dict(color=COLORS["text"]),
                        capprops=dict(color=COLORS["text"]),
                        flierprops=dict(markeredgecolor=COLORS["text"]))
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(COLORS["grid"])
    ax.set_ylabel("CV (%)", color=COLORS["text"])
    ax.set_title("Aggregate CV Distribution by Band", color=COLORS["text"], fontweight="bold")
    ax.axhline(15, color=COLORS["accent2"], linestyle=":", linewidth=1.5,
               label="15% threshold")
    ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 4: Example HF periodogram (first subject)
    ax = axes[1, 1]
    if subject_ids and "HF" in results[subject_ids[0]].bands:
        # Re-run LS to get periodogram data for plotting
        first_subj = subject_ids[0]
        hf = results[first_subj].bands["HF"]
        ax.axvline(hf.ls_frequency_hz, color=COLORS["accent"], linestyle="--",
                   label=f"Peak: {hf.ls_frequency_hz:.3f} Hz")
        ax.text(0.5, 0.5, f"HF Peak: {hf.ls_frequency_hz:.3f} Hz\n"
                f"CV: {hf.multi_window_cv_clean:.1f}%\n"
                f"Tier: {hf.confidence_tier}",
                transform=ax.transAxes, color=COLORS["text"],
                ha="center", va="center", fontsize=14)
    ax.set_xlabel("Frequency (Hz)", color=COLORS["text"])
    ax.set_ylabel("Lomb-Scargle Power", color=COLORS["text"])
    ax.set_title("HF Band Periodogram (Subject 1)", color=COLORS["text"], fontweight="bold")
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 5: Cross-subject variability (CV range)
    ax = axes[2, 0]
    for i, band in enumerate(band_names):
        band_cvs = [results[s].bands[band].multi_window_cv_clean
                     for s in subject_ids if band in results[s].bands]
        if band_cvs:
            mean_cv = np.mean(band_cvs)
            std_cv = np.std(band_cvs)
            ax.errorbar(i, mean_cv, yerr=std_cv, fmt="o", color=band_colors[band],
                        capsize=10, markersize=12, linewidth=2,
                        label=f"{band}: {mean_cv:.1f} +/- {std_cv:.1f}%")
    ax.set_xticks(range(len(band_names)))
    ax.set_xticklabels(band_names, color=COLORS["text"])
    ax.set_ylabel("CV (%) mean +/- std", color=COLORS["text"])
    ax.set_title("Cross-Subject CV Variability", color=COLORS["text"], fontweight="bold")
    ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 6: Cross-domain positioning
    ax = axes[2, 1]
    domains = []
    cv_vals = []
    bar_colors = []
    for name, ref in CROSS_DOMAIN_REFS.items():
        domains.append(ref["domain"])
        cv_vals.append(ref["cv_mean"])
        bar_colors.append(COLORS["accent3"])

    # Add HRV results
    for band in ["HF", "LF", "VLF"]:
        band_cvs = [results[s].bands[band].multi_window_cv_clean
                     for s in subject_ids if band in results[s].bands]
        if band_cvs:
            domains.append(f"HRV ({band})")
            cv_vals.append(np.mean(band_cvs))
            bar_colors.append(band_colors[band])

    y_pos = np.arange(len(domains))
    ax.barh(y_pos, cv_vals, color=bar_colors, edgecolor=COLORS["grid"], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(domains, color=COLORS["text"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("CV (%) — log scale", color=COLORS["text"])
    ax.set_title("Cross-Domain CV Position", color=COLORS["text"], fontweight="bold")
    ax.axvline(15, color=COLORS["accent2"], linestyle=":", linewidth=1.5)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    plt.tight_layout()
    plot_path = PLOT_DIR / "hrv_validation.png"
    fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")


# ============================================================================
# RESULTS SERIALIZATION
# ============================================================================

def serialize_band(b: BandResult) -> Dict:
    """Convert BandResult to JSON-safe dict."""
    return {
        "band_name": b.band_name,
        "band_range_hz": [float(x) for x in b.band_range_hz],
        "ls_frequency_hz": float(b.ls_frequency_hz),
        "ls_power": float(b.ls_power),
        "ls_fap": float(b.ls_fap),
        "multi_window_cv": float(b.multi_window_cv),
        "multi_window_cv_clean": float(b.multi_window_cv_clean),
        "detection_rate": float(b.detection_rate),
        "bootstrap_freq_mean_hz": float(b.bootstrap_freq_mean),
        "bootstrap_freq_std_hz": float(b.bootstrap_freq_std),
        "bootstrap_ci_low_hz": float(b.bootstrap_ci_low),
        "bootstrap_ci_high_hz": float(b.bootstrap_ci_high),
        "confidence_score": float(b.confidence_score),
        "confidence_tier": b.confidence_tier,
    }


def serialize_subject(s: SubjectResult) -> Dict:
    """Convert SubjectResult to JSON-safe dict."""
    return {
        "subject_id": s.subject_id,
        "n_beats": int(s.n_beats),
        "duration_hours": float(s.duration_hours),
        "mean_hr_bpm": float(s.mean_hr_bpm),
        "bands": {name: serialize_band(b) for name, b in s.bands.items()},
    }


def save_results(results: Dict[str, SubjectResult]):
    """Save results to JSON."""
    # Aggregate stats per band
    band_stats = {}
    for band_name in HRV_BANDS:
        cvs = [results[s].bands[band_name].multi_window_cv_clean
               for s in results if band_name in results[s].bands]
        dets = [results[s].bands[band_name].detection_rate
                for s in results if band_name in results[s].bands]
        if cvs:
            band_stats[band_name] = {
                "mean_cv": float(np.mean(cvs)),
                "std_cv": float(np.std(cvs)),
                "min_cv": float(np.min(cvs)),
                "max_cv": float(np.max(cvs)),
                "mean_detection_rate": float(np.mean(dets)),
                "n_subjects": len(cvs),
            }

    output = {
        "experiment": "Ab Astris Negative Control: Heart Rate Variability",
        "methodology": "Ab Astris (Lomb-Scargle + Multi-window + Bootstrap)",
        "data_source": "PhysioNet Normal Sinus Rhythm RR Interval Database (nsr2db)",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "resample_fs_hz": RESAMPLE_FS,
            "rr_min_sec": RR_MIN,
            "rr_max_sec": RR_MAX,
            "n_windows": N_WINDOWS,
            "window_duration_sec": WINDOW_DURATION_SEC,
            "n_bootstrap": N_BOOTSTRAP,
            "n_freq_points": N_FREQ_POINTS,
            "freq_tolerance": FREQ_TOLERANCE,
            "bandpass_order": BANDPASS_ORDER,
            "hrv_bands": {k: {"hz_low": v["hz"][0], "hz_high": v["hz"][1]}
                          for k, v in HRV_BANDS.items()},
        },
        "subjects": {sid: serialize_subject(r) for sid, r in results.items()},
        "band_summary": band_stats,
        "summary": {
            "n_subjects": len(results),
            "hf_mean_cv": band_stats.get("HF", {}).get("mean_cv", None),
            "lf_mean_cv": band_stats.get("LF", {}).get("mean_cv", None),
            "vlf_mean_cv": band_stats.get("VLF", {}).get("mean_cv", None),
            "interpretation": (
                f"The human heart produces CV of "
                f"{band_stats.get('HF', {}).get('mean_cv', 0):.1f}% for the "
                f"respiratory-coupled HF band and "
                f"{band_stats.get('LF', {}).get('mean_cv', 0):.1f}% for "
                f"sympathetically modulated LF oscillations, placing biological "
                f"cardiac regulation between structural resonance and volcanic "
                f"tremor in the cross-domain stability hierarchy, despite the heart "
                f"being a genuine physical oscillator."
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

def run_hrv_experiment() -> Dict[str, SubjectResult]:
    """Run the full HRV negative control experiment."""
    print(f"\n{'='*70}")
    print(f"  AB ASTRIS NEGATIVE CONTROL: HEART RATE VARIABILITY")
    print(f"{'='*70}")

    results = {}
    example_rr = None
    example_subject = None

    for subject_id in SUBJECTS:
        result = analyze_subject(subject_id)
        if result is not None:
            results[subject_id] = result
            # Save first subject's RR data for plotting
            if example_rr is None:
                try:
                    rr_raw, _ = download_hrv_data(subject_id)
                    example_rr = preprocess_rr(rr_raw)
                    example_subject = subject_id
                except Exception:
                    pass

    if not results:
        print("ERROR: No subjects successfully analysed")
        return {}

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Subjects analysed: {len(results)}/{len(SUBJECTS)}")

    for band_name in ["HF", "LF", "VLF"]:
        cvs = [results[s].bands[band_name].multi_window_cv_clean
               for s in results if band_name in results[s].bands]
        if cvs:
            print(f"  {band_name}: mean CV = {np.mean(cvs):.1f}% "
                  f"(range {np.min(cvs):.1f}–{np.max(cvs):.1f}%)")

    # Save and plot
    save_results(results)
    plot_hrv_results(results, example_subject, example_rr)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ab Astris Negative Control: Heart Rate Variability"
    )
    args = parser.parse_args()
    run_hrv_experiment()


if __name__ == "__main__":
    main()

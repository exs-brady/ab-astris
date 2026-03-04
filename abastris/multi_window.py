"""
Stage 3: Multi-window validation and CV stability metric.

The signal is divided into overlapping windows and the periodogram is
computed independently for each.  The coefficient of variation (CV) of
detected peak frequencies across windows quantifies stability:

    CV = (sigma_f / f_bar) * 100%

An outlier-removal step excludes frequencies more than 2 sigma from the
mean before computing the final CV.  The detection rate is the fraction
of windows in which the detected peak falls within 15% of the global
peak frequency.

Reference: Section 2.1 (Stage 3) of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from .periodogram import run_lomb_scargle


@dataclass
class MultiWindowResult:
    """Results from multi-window validation."""
    detected_freqs: List[float]
    cv_raw: float           # CV before outlier removal (%)
    cv_clean: float         # CV after 2-sigma outlier removal (%)
    detection_rate: float   # Fraction of windows detecting the signal
    outlier_count: int


def create_windows(
    time: np.ndarray,
    signal: np.ndarray,
    n_windows: int = 8,
    overlap: float = 0.5,
) -> List[Dict]:
    """Divide a signal into overlapping analysis windows.

    Parameters
    ----------
    time : array
        Observation times.
    signal : array
        Signal values.
    n_windows : int
        Number of windows (4-8 depending on domain; see Table 2).
    overlap : float
        Fractional overlap between adjacent windows (default 0.5 = 50%).

    Returns
    -------
    list of dict
        Each dict has 'time' and 'signal' arrays for one window.
    """
    total_duration = time[-1] - time[0]
    # Window duration: total / (1 + (n-1)*(1-overlap))
    window_duration = total_duration / (1 + (n_windows - 1) * (1 - overlap))
    step = window_duration * (1 - overlap)

    windows = []
    for i in range(n_windows):
        start = time[0] + i * step
        end = start + window_duration
        mask = (time >= start) & (time <= end)
        if np.sum(mask) < 50:
            continue
        windows.append({
            "time": time[mask],
            "signal": signal[mask],
        })
    return windows


def run_multi_window_validation(
    windows: List[Dict],
    freq_min: float,
    freq_max: float,
    n_points: int = 5_000,
    tolerance: float = 0.15,
) -> MultiWindowResult:
    """Run Lomb-Scargle on each window and compute CV across windows.

    Parameters
    ----------
    windows : list of dict
        Output of :func:`create_windows`.
    freq_min, freq_max : float
        Frequency range for Lomb-Scargle analysis.
    n_points : int
        Frequency grid points per window (default 5,000).
    tolerance : float
        Fractional tolerance for detection rate (default 0.15 = 15%).

    Returns
    -------
    MultiWindowResult
        CV, detection rate, and per-window frequencies.
    """
    detected_freqs = []
    for w in windows:
        result = run_lomb_scargle(
            w["time"], w["signal"],
            freq_min=freq_min, freq_max=freq_max,
            n_points=n_points,
        )
        detected_freqs.append(result.frequency)

    freqs = np.array(detected_freqs)
    freq_mean = np.mean(freqs)
    freq_std = np.std(freqs)
    cv_raw = float(freq_std / freq_mean * 100) if freq_mean > 0 else 0.0

    # 2-sigma outlier removal
    outlier_count = 0
    if len(freqs) > 2 and freq_std > 0:
        z = np.abs((freqs - freq_mean) / freq_std)
        clean = freqs[z < 2]
        outlier_count = len(freqs) - len(clean)
        cv_clean = float(np.std(clean) / np.mean(clean) * 100) if len(clean) > 1 else cv_raw
    else:
        cv_clean = cv_raw

    # Detection rate: fraction within ±tolerance of the global peak
    global_peak = freq_mean
    detections = np.abs(freqs - global_peak) / global_peak < tolerance
    detection_rate = float(np.mean(detections))

    return MultiWindowResult(
        detected_freqs=freqs.tolist(),
        cv_raw=cv_raw,
        cv_clean=cv_clean,
        detection_rate=detection_rate,
        outlier_count=outlier_count,
    )

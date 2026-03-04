#!/usr/bin/env python3
"""
Enhanced statistical analysis for period detection

Provides:
- Alias detection (harmonics, sub-harmonics, daily aliases)
- Window function analysis
- Peak significance testing
- Multiple period detection
"""

import numpy as np
from astropy.timeseries import LombScargle
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def detect_aliases(
    time: np.ndarray,
    flux: np.ndarray,
    primary_period: float,
    primary_power: float,
    min_power_ratio: float = 0.3
) -> Dict[str, Any]:
    """
    Detect potential period aliases including harmonics, sub-harmonics,
    and daily aliases from sampling patterns.

    Args:
        time: Time array (days)
        flux: Flux array
        primary_period: Primary detected period (days)
        primary_power: Power of primary peak
        min_power_ratio: Minimum power ratio to consider as alias

    Returns:
        Dictionary with alias information
    """
    # Ensure time is plain numpy array
    if hasattr(time, 'value'):
        time = time.value
    time = np.asarray(time, dtype=float)

    # Ensure flux is plain numpy array
    if hasattr(flux, 'value'):
        flux = flux.value
    flux = np.asarray(flux, dtype=float)

    aliases = []

    # Create Lomb-Scargle periodogram
    ls = LombScargle(time, flux)

    # Check harmonics (2P, 3P, 4P)
    harmonics_to_check = [2, 3, 4]
    for n in harmonics_to_check:
        harmonic_period = primary_period * n
        harmonic_freq = 1.0 / harmonic_period
        harmonic_power = ls.power(harmonic_freq)

        if harmonic_power > min_power_ratio * primary_power:
            aliases.append({
                'type': 'harmonic',
                'period': harmonic_period,
                'multiplier': n,
                'power': float(harmonic_power),
                'power_ratio': float(harmonic_power / primary_power),
                'description': f'{n}P harmonic'
            })
            logger.info(f"    Detected {n}P harmonic: {harmonic_period:.5f} days (power ratio: {harmonic_power/primary_power:.2f})")

    # Check sub-harmonics (P/2, P/3, P/4)
    for n in harmonics_to_check:
        subharmonic_period = primary_period / n
        subharmonic_freq = 1.0 / subharmonic_period
        subharmonic_power = ls.power(subharmonic_freq)

        if subharmonic_power > min_power_ratio * primary_power:
            aliases.append({
                'type': 'sub-harmonic',
                'period': subharmonic_period,
                'multiplier': 1.0 / n,
                'power': float(subharmonic_power),
                'power_ratio': float(subharmonic_power / primary_power),
                'description': f'P/{n} sub-harmonic'
            })
            logger.info(f"    Detected P/{n} sub-harmonic: {subharmonic_period:.5f} days (power ratio: {subharmonic_power/primary_power:.2f})")

    # Check 1-day aliases (common for ground-based but can affect TESS too)
    # P ± 1 day, P ± 1 sidereal day
    one_day_aliases = [
        (primary_period + 1.0, '+1 day'),
        (primary_period - 1.0, '-1 day'),
        (primary_period + 0.9973, '+1 sidereal day'),
        (primary_period - 0.9973, '-1 sidereal day')
    ]

    for alias_period, description in one_day_aliases:
        if alias_period > 0:  # Must be positive
            alias_freq = 1.0 / alias_period
            alias_power = ls.power(alias_freq)

            if alias_power > min_power_ratio * primary_power:
                aliases.append({
                    'type': 'daily',
                    'period': alias_period,
                    'multiplier': alias_period / primary_period,
                    'power': float(alias_power),
                    'power_ratio': float(alias_power / primary_power),
                    'description': description
                })
                logger.info(f"    Detected daily alias {description}: {alias_period:.5f} days (power ratio: {alias_power/primary_power:.2f})")

    # Calculate baseline and sampling statistics
    baseline = np.max(time) - np.min(time)
    n_points = len(time)
    median_cadence = np.median(np.diff(np.sort(time)))

    return {
        'aliases_detected': aliases,
        'n_aliases': len(aliases),
        'has_harmonics': any(a['type'] == 'harmonic' for a in aliases),
        'has_subharmonics': any(a['type'] == 'sub-harmonic' for a in aliases),
        'has_daily_aliases': any(a['type'] == 'daily' for a in aliases),
        'baseline_days': float(baseline),
        'n_points': n_points,
        'median_cadence_days': float(median_cadence)
    }


def calculate_window_function(
    time: np.ndarray,
    primary_period: float,
    oversampling: int = 5
) -> Dict[str, Any]:
    """
    Calculate window function (spectral window) to identify aliases
    from sampling pattern.

    The window function is the Fourier transform of the sampling pattern,
    showing which frequencies are aliased due to gaps in observations.

    Args:
        time: Time array (days)
        primary_period: Primary detected period (days)
        oversampling: Oversampling factor for frequency grid

    Returns:
        Dictionary with window function information
    """
    # Ensure time is plain numpy array
    if hasattr(time, 'value'):
        time = time.value
    time = np.asarray(time, dtype=float)

    # Create a constant "flux" array (all ones) to analyze just the sampling
    ones = np.ones_like(time)

    # Calculate Lomb-Scargle of the window (sampling pattern)
    ls_window = LombScargle(time, ones, center_data=False, fit_mean=False)

    # Frequency range around primary period
    primary_freq = 1.0 / primary_period
    freq_range = np.linspace(
        primary_freq * 0.5,
        primary_freq * 2.0,
        1000 * oversampling
    )

    # Window function power
    window_power = ls_window.power(freq_range)

    # Find peaks in window function
    window_peaks = []
    peak_threshold = 0.1  # 10% of maximum

    for i in range(1, len(window_power) - 1):
        if (window_power[i] > window_power[i-1] and
            window_power[i] > window_power[i+1] and
            window_power[i] > peak_threshold):

            peak_freq = freq_range[i]
            peak_period = 1.0 / peak_freq

            window_peaks.append({
                'period': float(peak_period),
                'frequency': float(peak_freq),
                'power': float(window_power[i]),
                'offset_from_primary': float(peak_period - primary_period)
            })

    # Sampling statistics
    baseline = np.max(time) - np.min(time)
    n_points = len(time)
    gaps = np.diff(np.sort(time))
    median_cadence = np.median(gaps)
    max_gap = np.max(gaps)

    return {
        'window_peaks': window_peaks[:5],  # Top 5 peaks
        'n_window_peaks': len(window_peaks),
        'baseline_days': float(baseline),
        'n_observations': n_points,
        'median_cadence_days': float(median_cadence),
        'max_gap_days': float(max_gap),
        'duty_cycle': float(n_points * median_cadence / baseline) if baseline > 0 else 0.0
    }


def find_secondary_periods(
    time: np.ndarray,
    flux: np.ndarray,
    primary_period: float,
    min_power_ratio: float = 0.5,
    max_secondary: int = 3
) -> List[Dict[str, Any]]:
    """
    Search for additional periods beyond the primary.
    Useful for multi-periodic variables (e.g., pulsators with multiple modes).

    Args:
        time: Time array (days)
        flux: Flux array
        primary_period: Primary detected period (days)
        min_power_ratio: Minimum power ratio to consider significant
        max_secondary: Maximum number of secondary periods to return

    Returns:
        List of secondary period detections
    """
    # Ensure time is plain numpy array
    if hasattr(time, 'value'):
        time = time.value
    time = np.asarray(time, dtype=float)

    # Ensure flux is plain numpy array
    if hasattr(flux, 'value'):
        flux = flux.value
    flux = np.asarray(flux, dtype=float)

    # Create Lomb-Scargle periodogram
    ls = LombScargle(time, flux)

    # Broad frequency search
    frequencies = np.linspace(0.1, 50, 10000)
    powers = ls.power(frequencies)

    # Find primary peak
    primary_freq = 1.0 / primary_period
    primary_idx = np.argmin(np.abs(frequencies - primary_freq))
    primary_power = powers[primary_idx]

    # Find other peaks
    secondary_periods = []

    # Local maxima detection
    for i in range(1, len(powers) - 1):
        if powers[i] > powers[i-1] and powers[i] > powers[i+1]:
            # Check if significant and not near primary
            freq_ratio = frequencies[i] / primary_freq

            # Skip if too close to primary or its harmonics
            if 0.9 < freq_ratio < 1.1:
                continue
            if 1.9 < freq_ratio < 2.1:
                continue
            if 2.9 < freq_ratio < 3.1:
                continue

            # Check power threshold
            if powers[i] > min_power_ratio * primary_power:
                period = 1.0 / frequencies[i]
                fap = ls.false_alarm_probability(powers[i])

                secondary_periods.append({
                    'period': float(period),
                    'period_hours': float(period * 24),
                    'power': float(powers[i]),
                    'power_ratio': float(powers[i] / primary_power),
                    'fap': float(fap),
                    'frequency': float(frequencies[i])
                })

    # Sort by power and return top N
    secondary_periods.sort(key=lambda x: x['power'], reverse=True)

    if secondary_periods:
        logger.info(f"  Found {len(secondary_periods)} potential secondary periods")
        for i, sp in enumerate(secondary_periods[:max_secondary]):
            logger.info(f"    Secondary {i+1}: {sp['period']:.5f} days ({sp['period_hours']:.2f} hrs), power ratio: {sp['power_ratio']:.2f}")

    return secondary_periods[:max_secondary]


def comprehensive_period_analysis(
    time: np.ndarray,
    flux: np.ndarray,
    primary_period: float,
    primary_power: float
) -> Dict[str, Any]:
    """
    Run comprehensive statistical analysis on detected period.

    Combines:
    - Alias detection
    - Window function analysis
    - Secondary period search

    Args:
        time: Time array (days)
        flux: Flux array
        primary_period: Primary detected period (days)
        primary_power: Power of primary peak

    Returns:
        Dictionary with all statistical information
    """
    logger.info(f"  Running comprehensive period analysis...")

    # Alias detection
    alias_results = detect_aliases(time, flux, primary_period, primary_power)

    # Window function
    window_results = calculate_window_function(time, primary_period)

    # Secondary periods
    secondary_periods = find_secondary_periods(time, flux, primary_period)

    # Combine results
    analysis = {
        **alias_results,
        'window_function': window_results,
        'secondary_periods': secondary_periods,
        'n_secondary_periods': len(secondary_periods)
    }

    # Summary flags
    analysis['has_strong_aliases'] = alias_results['n_aliases'] > 0
    analysis['has_secondary_periods'] = len(secondary_periods) > 0
    analysis['period_confidence'] = 'high' if alias_results['n_aliases'] == 0 else 'medium'

    logger.info(f"  Analysis complete: {alias_results['n_aliases']} aliases, {len(secondary_periods)} secondary periods")

    return analysis

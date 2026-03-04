"""
Transit Detector - Box Least Squares (BLS) transit search for TESS light curves

Complements Lomb-Scargle periodogram for detecting:
- Planetary transits (box-shaped flux dips)
- Eclipsing binaries with narrow eclipses
- Ultra-short period (USP) transiting systems

Uses astropy.timeseries.BoxLeastSquares for efficient transit search.

Usage:
    from abastris.core.transit_detector import TransitDetector, detect_transits

    detector = TransitDetector()
    result = detector.search(time, flux)

    if result.is_transit_candidate:
        print(f"Transit detected: P={result.bls_period:.6f}d, depth={result.bls_depth:.4f}")

Updated: 2025-12-16 - Two-stage optimization for performance
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging
import time as timer

from astropy.timeseries import BoxLeastSquares

logger = logging.getLogger(__name__)

# =============================================================================
# BLS PERFORMANCE OPTIMIZATION CONSTANTS
# =============================================================================

# Two-stage detection grid sizes
BLS_COARSE_PERIODS = 500       # Number of periods in coarse grid
BLS_FINE_PERIODS = 500         # Number of periods in fine grid
BLS_FINE_DURATIONS = 15        # Number of durations in fine grid
BLS_FINE_WINDOW = 0.05         # +/- 5% around peak for refinement

# Search range (USP-focused)
BLS_MIN_PERIOD = 0.2           # days (4.8 hours)
BLS_MAX_PERIOD = 1.0           # days (USP cutoff)
BLS_COARSE_DURATION = 0.05     # days (1.2 hours) - legacy single duration

# Duration grid for USP detection (covers 14 min to 2 hours)
# USP transits are typically 10-90 minutes; using single duration missed most signals
# Duration grid for USP regime (P < 1d) - extended to 4h for grazing transits
BLS_USP_DURATIONS = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.17])

# Auto-optimization thresholds
# Binning destroys USP transit signals even with small bins
# With current performance (~100ms for 200k points), binning provides minimal benefit
BLS_MAX_POINTS_UNBINNED = 500000  # Effectively disable binning for most real data
BLS_BIN_SIZE = 0.007              # days (~10 minutes) if binning is ever used

# Detection thresholds
# Stage 1 now uses multi-duration grid for better USP detection
# Multi-duration with SNR objective produces lower power values than single-duration
BLS_COARSE_POWER_THRESHOLD = 0.001  # Minimum power to proceed to Stage 2 (very permissive)
BLS_DEFAULT_SNR_THRESHOLD = 7.0     # Final candidate SNR threshold

# Harmonic detection constants
HARMONIC_RATIOS = [0.5, 1.0, 1.5, 2.0, 2.2, 2.5, 3.0]  # Common BLS harmonics
HARMONIC_TOLERANCE = 0.05  # 5% tolerance for matching


# =============================================================================
# HARMONIC DETECTION FUNCTIONS
# =============================================================================

def check_period_harmonic(detected_period: float, reference_period: float,
                          tolerance: float = HARMONIC_TOLERANCE) -> Dict[str, Any]:
    """
    Check if detected period is a harmonic of reference period.

    BLS commonly detects:
    - P/2 (half): Symmetric transits can appear as two events per orbit
    - 2P (double): Deep transits can mask every other detection
    - 3P (triple): Sparse data can cause period tripling
    - 1.5P, 2.5P: Aliasing with observation cadence

    Parameters
    ----------
    detected_period : float
        Period detected by BLS
    reference_period : float
        Known/expected period to compare against
    tolerance : float
        Fractional tolerance for matching (default 3%)

    Returns
    -------
    dict with keys:
        is_harmonic : bool - True if detected is a harmonic of reference
        harmonic_ratio : float - The ratio (0.5, 1.0, 2.0, etc.) or None
        harmonic_type : str - 'direct', 'half', 'double', 'triple', 'alias', or None
        period_error_pct : float - Error from nearest harmonic match
    """
    if detected_period is None or reference_period is None:
        return {
            'is_harmonic': False,
            'harmonic_ratio': None,
            'harmonic_type': None,
            'period_error_pct': None
        }

    best_match = None
    best_error = float('inf')

    for ratio in HARMONIC_RATIOS:
        expected = reference_period * ratio
        error_pct = abs(detected_period - expected) / expected * 100

        if error_pct < best_error:
            best_error = error_pct
            best_match = ratio

    # Check if within tolerance
    if best_error <= tolerance * 100:
        # Determine harmonic type name
        type_names = {
            0.5: 'half',
            1.0: 'direct',
            1.5: 'sesqui',  # 3:2 resonance
            2.0: 'double',
            2.5: 'alias',
            3.0: 'triple'
        }
        harmonic_type = type_names.get(best_match, f'x{best_match}')

        return {
            'is_harmonic': True,
            'harmonic_ratio': best_match,
            'harmonic_type': harmonic_type,
            'period_error_pct': best_error
        }
    else:
        return {
            'is_harmonic': False,
            'harmonic_ratio': None,
            'harmonic_type': None,
            'period_error_pct': best_error
        }


def resolve_harmonic(detected_period: float, time: np.ndarray, flux: np.ndarray,
                     flux_err: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Given a detected period, check if P/2 or P*2 produces a better fit.

    This helps resolve harmonic ambiguity when we don't know the true period.

    Strategy:
    1. Compute BLS power at P, P/2, and 2P
    2. Compare transit depths and SNR
    3. Prefer shorter period if powers are similar (Occam's razor)

    Parameters
    ----------
    detected_period : float
        Initial period from BLS
    time, flux : arrays
        Lightcurve data
    flux_err : array, optional
        Flux uncertainties

    Returns
    -------
    dict with:
        best_period : float - Most likely true period
        original_period : float - Input period
        resolution : str - 'original', 'half', or 'double'
        confidence : float - Relative confidence in resolution
    """
    if detected_period is None:
        return {
            'best_period': None,
            'original_period': None,
            'resolution': None,
            'confidence': 0
        }

    # Check harmonics: P/2, P, P*2, P*3
    # P*3 added to catch cases where BLS detects P/3 harmonic (like TIC 172933154)
    candidates = [
        ('half', detected_period / 2),
        ('original', detected_period),
        ('double', detected_period * 2),
        ('triple', detected_period * 3)
    ]

    # Filter to valid period range (extended to 15 days for triple harmonics)
    candidates = [(name, p) for name, p in candidates if 0.1 < p < 15.0]

    if flux_err is not None:
        bls = BoxLeastSquares(time, flux, dy=flux_err)
    else:
        bls = BoxLeastSquares(time, flux)

    results = []
    for name, period in candidates:
        try:
            # Test this specific period with multiple durations
            # Extended to 0.25 days (6h) to catch longer transits around larger stars
            durations = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.17, 0.25])
            durations = durations[durations < period * 0.25]  # Duration < 25% of period

            if len(durations) == 0:
                durations = np.array([period * 0.1])

            # Get power at this exact period
            test_periods = np.array([period])
            result = bls.power(test_periods, duration=durations)

            power = result.power[0]
            best_duration = result.duration[0]

            # Get depth
            stats = bls.compute_stats(period, best_duration, result.transit_time[0])
            depth_val = stats['depth']
            depth = abs(float(depth_val[0]) if hasattr(depth_val, '__len__') else float(depth_val))

            results.append({
                'name': name,
                'period': period,
                'power': power,
                'depth': depth,
                'duration': best_duration
            })
        except Exception:
            continue

    if not results:
        return {
            'best_period': detected_period,
            'original_period': detected_period,
            'resolution': 'original',
            'confidence': 0
        }

    # Sort by power (highest first)
    results.sort(key=lambda x: x['power'], reverse=True)

    best = results[0]

    # If powers are within 10%, prefer shorter period
    if len(results) > 1:
        for r in results[1:]:
            if r['power'] > best['power'] * 0.9 and r['period'] < best['period']:
                best = r
                break

    # Calculate confidence as ratio of best to second-best power
    if len(results) > 1:
        confidence = best['power'] / results[1]['power']
    else:
        confidence = 1.0

    return {
        'best_period': best['period'],
        'original_period': detected_period,
        'resolution': best['name'],
        'confidence': confidence,
        'all_candidates': results
    }


# =============================================================================
# SNR CALCULATION FUNCTIONS
# =============================================================================

def calculate_transit_snr(time: np.ndarray, flux: np.ndarray,
                          period: float, duration: float, t0: float,
                          flux_err: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate transit SNR using multiple methods.

    Methods:
    1. Formal BLS SNR (from compute_stats)
    2. Empirical SNR (depth / scatter in phase-folded data)
    3. Bootstrap SNR (resample and measure depth variance)

    Parameters
    ----------
    time, flux : arrays
        Lightcurve data
    period, duration, t0 : float
        Transit parameters from BLS
    flux_err : array, optional
        Flux uncertainties

    Returns
    -------
    dict with:
        snr_formal : float - From BLS stats
        snr_empirical : float - Depth / out-of-transit scatter
        snr_bootstrap : float - From bootstrap resampling
        snr_best : float - Recommended SNR to use
        depth : float - Transit depth
        depth_err : float - Depth uncertainty
    """
    if period is None or duration is None or t0 is None:
        return {
            'snr_formal': 0,
            'snr_empirical': 0,
            'snr_bootstrap': 0,
            'snr_best': 0,
            'depth': 0,
            'depth_err': 0,
            'depth_ppm': 0,
            'depth_err_ppm': 0,
            'n_in_transit': 0,
            'n_out_transit': 0,
            'out_of_transit_scatter_ppm': 0
        }

    # Phase-fold the data
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1  # Center on transit at phase 0

    # Define in-transit and out-of-transit regions
    transit_half_phase = (duration / period) / 2
    in_transit = np.abs(phase) < transit_half_phase
    out_of_transit = np.abs(phase) > transit_half_phase * 3  # Well outside transit

    n_in = int(np.sum(in_transit))
    n_out = int(np.sum(out_of_transit))

    if n_in < 3 or n_out < 10:
        return {
            'snr_formal': 0,
            'snr_empirical': 0,
            'snr_bootstrap': 0,
            'snr_best': 0,
            'depth': 0,
            'depth_err': 0,
            'depth_ppm': 0,
            'depth_err_ppm': 0,
            'n_in_transit': n_in,
            'n_out_transit': n_out,
            'out_of_transit_scatter_ppm': 0
        }

    # Calculate depth
    baseline = np.median(flux[out_of_transit])
    transit_flux = np.median(flux[in_transit])
    depth = baseline - transit_flux

    # Method 1: Formal error (if flux_err provided)
    snr_formal = 0
    depth_err_formal = 0
    if flux_err is not None and len(flux_err) == len(flux):
        # Propagate errors
        baseline_err = np.median(flux_err[out_of_transit]) / np.sqrt(n_out)
        transit_err = np.median(flux_err[in_transit]) / np.sqrt(n_in)
        depth_err_formal = np.sqrt(baseline_err**2 + transit_err**2)
        if depth_err_formal > 0:
            snr_formal = abs(depth) / depth_err_formal

    # Method 2: Empirical (depth / scatter)
    out_scatter = np.std(flux[out_of_transit])
    # Account for number of in-transit points
    depth_err_empirical = out_scatter / np.sqrt(n_in)
    snr_empirical = abs(depth) / depth_err_empirical if depth_err_empirical > 0 else 0

    # Method 3: Bootstrap (slower but robust)
    # Reduced from 100 to 30 samples to save memory while remaining statistically valid
    n_bootstrap = 30
    bootstrap_depths = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(len(time), size=len(time), replace=True)
        boot_phase = phase[idx]
        boot_flux = flux[idx]

        boot_in = np.abs(boot_phase) < transit_half_phase
        boot_out = np.abs(boot_phase) > transit_half_phase * 3

        if np.sum(boot_in) > 0 and np.sum(boot_out) > 0:
            boot_baseline = np.median(boot_flux[boot_out])
            boot_transit = np.median(boot_flux[boot_in])
            bootstrap_depths.append(boot_baseline - boot_transit)

    snr_bootstrap = 0
    depth_err_bootstrap = 0
    if bootstrap_depths:
        depth_err_bootstrap = np.std(bootstrap_depths)
        if depth_err_bootstrap > 0:
            snr_bootstrap = abs(depth) / depth_err_bootstrap

    # Best SNR: use empirical as primary, bootstrap as validation
    snr_best = snr_empirical
    depth_err_best = depth_err_empirical

    # If bootstrap disagrees significantly, use the more conservative
    if snr_bootstrap > 0 and snr_empirical > 0:
        ratio = snr_empirical / snr_bootstrap
        if ratio > 1.5 or ratio < 0.67:  # Disagree by >50%
            snr_best = min(snr_empirical, snr_bootstrap)
            depth_err_best = max(depth_err_empirical, depth_err_bootstrap)

    return {
        'snr_formal': float(snr_formal),
        'snr_empirical': float(snr_empirical),
        'snr_bootstrap': float(snr_bootstrap),
        'snr_best': float(snr_best),
        'depth': float(depth),
        'depth_ppm': float(depth * 1e6),
        'depth_err': float(depth_err_best),
        'depth_err_ppm': float(depth_err_best * 1e6),
        'n_in_transit': n_in,
        'n_out_transit': n_out,
        'out_of_transit_scatter_ppm': float(out_scatter * 1e6)
    }


def check_depth_consistency(time: np.ndarray, flux: np.ndarray,
                            period: float, duration: float, t0: float) -> Dict[str, Any]:
    """
    Check consistency of transit depths across individual events.

    High depth variation (>20%) is a red flag for eclipsing binaries,
    which often have different primary and secondary eclipse depths.

    Parameters
    ----------
    time, flux : arrays
        Lightcurve data
    period, duration, t0 : float
        Transit parameters from BLS

    Returns
    -------
    dict with:
        n_transits : int - Number of individual transits analyzed
        depth_mean_ppm : float - Mean transit depth
        depth_std_ppm : float - Standard deviation of depths
        depth_variation : float - Coefficient of variation (std/mean)
        is_consistent : bool - True if depths are consistent (variation < 20%)
        depths_ppm : list - Individual transit depths
        odd_even_diff_pct : float - Percent difference between odd/even depths
    """
    if period is None or duration is None or t0 is None:
        return {
            'n_transits': 0,
            'depth_mean_ppm': 0,
            'depth_std_ppm': 0,
            'depth_variation': 0,
            'is_consistent': True,
            'depths_ppm': [],
            'odd_even_diff_pct': 0
        }

    # Find all transit mid-times in the data
    t_start = time.min()
    t_end = time.max()

    transit_times = []
    t = t0
    while t > t_start:
        t -= period
    while t < t_end:
        if t >= t_start:
            transit_times.append(t)
        t += period

    # Measure depth at each transit
    window = duration * 3  # Look at 3x duration around transit
    depths = []
    transit_numbers = []

    for i, tc in enumerate(transit_times):
        mask = np.abs(time - tc) < window
        if np.sum(mask) < 10:
            continue

        t_transit = time[mask]
        f_transit = flux[mask]

        # In-transit and out-of-transit masks
        in_transit = np.abs(t_transit - tc) < duration / 2
        out_transit = np.abs(t_transit - tc) > duration

        if np.sum(in_transit) < 3 or np.sum(out_transit) < 3:
            continue

        baseline = np.median(f_transit[out_transit])
        transit_flux = np.median(f_transit[in_transit])
        depth = (baseline - transit_flux) * 1e6  # ppm

        if depth > 0:  # Only count actual dips
            depths.append(depth)
            transit_numbers.append(i)

    if len(depths) < 3:
        return {
            'n_transits': len(depths),
            'depth_mean_ppm': np.mean(depths) if depths else 0,
            'depth_std_ppm': 0,
            'depth_variation': 0,
            'is_consistent': True,
            'depths_ppm': depths,
            'odd_even_diff_pct': 0
        }

    depth_mean = np.mean(depths)
    depth_std = np.std(depths)
    depth_variation = depth_std / depth_mean if depth_mean > 0 else 0

    # Check odd/even difference (EB signature)
    odd_depths = [d for i, d in zip(transit_numbers, depths) if i % 2 == 1]
    even_depths = [d for i, d in zip(transit_numbers, depths) if i % 2 == 0]

    odd_even_diff = 0
    odd_mean = None
    even_mean = None
    if odd_depths and even_depths:
        odd_mean = np.mean(odd_depths)
        even_mean = np.mean(even_depths)
        avg = (odd_mean + even_mean) / 2
        if avg > 0:
            odd_even_diff = abs(odd_mean - even_mean) / avg * 100

    # Consistent if variation < 20% AND odd/even diff < 30%
    is_consistent = depth_variation < 0.20 and odd_even_diff < 30

    # NEW: Strict odd/even validity check for FP detection
    # A real transit should have:
    # 1. Both odd and even depths positive and >50 ppm (above noise)
    # 2. Odd/even mismatch <50%
    odd_even_valid = True
    odd_even_rejection_reason = None

    if odd_mean is not None and even_mean is not None:
        min_depth_threshold = 50  # ppm - below this is noise

        if odd_mean < min_depth_threshold or even_mean < min_depth_threshold:
            odd_even_valid = False
            odd_even_rejection_reason = 'odd_even_depth_below_noise'
            logger.debug(f"Odd/even depths below noise threshold: odd={odd_mean:.0f}, even={even_mean:.0f} ppm")
        elif odd_mean < 0 or even_mean < 0:
            odd_even_valid = False
            odd_even_rejection_reason = 'odd_even_negative_depth'
            logger.debug(f"Odd/even has negative depth: odd={odd_mean:.0f}, even={even_mean:.0f} ppm")
        elif odd_even_diff > 50:
            odd_even_valid = False
            odd_even_rejection_reason = 'odd_even_mismatch'
            logger.debug(f"Odd/even mismatch too high: {odd_even_diff:.0f}%")
    elif len(depths) >= 3:
        # We have transits but couldn't separate odd/even - not enough coverage
        # This is okay, don't reject
        pass

    return {
        'n_transits': len(depths),
        'depth_mean_ppm': float(depth_mean),
        'depth_std_ppm': float(depth_std),
        'depth_variation': float(depth_variation),
        'is_consistent': bool(is_consistent),  # Ensure native Python bool for JSON
        'depths_ppm': depths,
        'odd_even_diff_pct': float(odd_even_diff),
        'odd_depth_ppm': float(odd_mean) if odd_mean is not None else None,
        'even_depth_ppm': float(even_mean) if even_mean is not None else None,
        'odd_even_valid': bool(odd_even_valid),
        'odd_even_rejection_reason': odd_even_rejection_reason
    }


def detect_secondary_eclipse(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    primary_depth_ppm: float,
    duration: Optional[float] = None
) -> Dict[str, Any]:
    """
    Detect secondary eclipse at phase 0.5 and compute depth ratio.

    A secondary eclipse (occultation) occurs when the companion passes behind
    the primary. For planets, this is typically very shallow (thermal emission
    or reflection). For eclipsing binaries, the secondary is often comparable
    to the primary.

    Classification based on depth ratio:
    - ratio > 0.5: 'eb' (eclipsing binary - secondary too deep for planet)
    - ratio 0.1-0.5: 'uncertain' (needs investigation)
    - ratio < 0.1: 'planet' (consistent with planetary thermal emission)

    Parameters
    ----------
    time : np.ndarray
        Time array (BTJD or similar)
    flux : np.ndarray
        Normalized flux array (~1.0 baseline)
    period : float
        Orbital period in days
    t0 : float
        Transit mid-time (primary eclipse)
    primary_depth_ppm : float
        Depth of primary transit in ppm
    duration : float, optional
        Transit duration in days (default: 2% of period)

    Returns
    -------
    dict with:
        has_secondary : bool - True if secondary eclipse detected (SNR > 3, depth > 100 ppm)
        secondary_depth_ppm : float - Depth of secondary eclipse
        secondary_snr : float - Signal-to-noise ratio of detection
        depth_ratio : float - Secondary/Primary depth ratio
        classification : str - 'planet', 'eb', or 'uncertain'
    """
    # Estimate duration from primary if not provided
    if duration is None:
        duration = period * 0.02  # Default 2% of period

    # Phase-fold centered on secondary eclipse position (phase = 0.5 from primary)
    phase = ((time - t0) % period) / period

    # Shift phase so secondary (at phase 0.5) is centered at 0
    phase = phase - 0.5
    phase[phase > 0.5] -= 1.0
    phase[phase < -0.5] += 1.0

    # Phase duration for the eclipse
    phase_duration = duration / period
    transit_half = phase_duration / 2

    # Define out-of-eclipse regions (well away from both primary and secondary)
    # Primary is at phase -0.5 (or +0.5 after shift)
    out_of_eclipse = (np.abs(phase) > phase_duration * 3) & (np.abs(phase) < 0.4)

    if np.sum(out_of_eclipse) < 10:
        # Not enough out-of-eclipse points
        return {
            'has_secondary': False,
            'secondary_depth_ppm': 0.0,
            'secondary_snr': 0.0,
            'depth_ratio': 0.0,
            'classification': 'planet'
        }

    # Calculate baseline from out-of-eclipse points
    baseline = np.nanmedian(flux[out_of_eclipse])
    out_scatter = np.nanstd(flux[out_of_eclipse])

    # In-eclipse measurement (centered on secondary)
    in_eclipse = np.abs(phase) < transit_half

    if np.sum(in_eclipse) < 3:
        # Not enough in-eclipse points
        return {
            'has_secondary': False,
            'secondary_depth_ppm': 0.0,
            'secondary_snr': 0.0,
            'depth_ratio': 0.0,
            'classification': 'planet'
        }

    eclipse_flux = np.nanmedian(flux[in_eclipse])
    n_in = np.sum(in_eclipse)

    # Secondary depth (positive = flux dip)
    secondary_depth = baseline - eclipse_flux
    secondary_depth_ppm = float(secondary_depth * 1e6)

    # SNR calculation
    if out_scatter > 0 and n_in > 0:
        secondary_snr = float(abs(secondary_depth) / (out_scatter / np.sqrt(n_in)))
    else:
        secondary_snr = 0.0

    # Detection threshold: SNR > 3 and depth > 100 ppm
    has_secondary = secondary_snr > 3.0 and abs(secondary_depth_ppm) > 100

    # Depth ratio (secondary/primary)
    if primary_depth_ppm > 0:
        depth_ratio = float(abs(secondary_depth_ppm) / primary_depth_ppm)
    else:
        depth_ratio = 0.0

    # Classification based on depth ratio
    if not has_secondary:
        classification = 'planet'  # No secondary = consistent with planet
    elif depth_ratio > 0.5:
        classification = 'eb'  # High ratio = eclipsing binary
    elif depth_ratio > 0.1:
        classification = 'uncertain'  # Medium ratio = needs investigation
    else:
        classification = 'planet'  # Low ratio = could be hot Jupiter thermal emission

    return {
        'has_secondary': bool(has_secondary),
        'secondary_depth_ppm': secondary_depth_ppm,
        'secondary_snr': secondary_snr,
        'depth_ratio': depth_ratio,
        'classification': classification
    }


def check_transit_shape(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
    t0: float,
    n_bins: int = 50,
    bls_reported_depth_ppm: Optional[float] = None
) -> Dict[str, Any]:
    """
    Check if the phase-folded light curve has a transit-like shape.

    Detects false positive patterns:
    1. FLAT: No significant dip at phase 0 (BLS triggered on noise)
    2. SINUSOIDAL: Smooth wave pattern (ellipsoidal variable or EB)
    3. BLS_ARTIFACT: High BLS SNR but binned depth doesn't match BLS depth
    4. HIGHSNR_ARTIFACT: High BLS depth claimed but no visible dip in phase curve

    Parameters
    ----------
    time, flux : arrays
        Lightcurve data (normalized to ~1.0)
    period, duration, t0 : float
        Transit parameters from BLS
    n_bins : int
        Number of phase bins for analysis
    bls_reported_depth_ppm : float, optional
        Depth reported by BLS in ppm (for consistency check)

    Returns
    -------
    dict with:
        shape_classification : str - 'transit', 'flat', 'sinusoidal', 'flat_bls_artifact',
                               'flat_highsnr_artifact', or 'unknown'
        in_transit_dip_sigma : float - Significance of dip at phase 0
        sinusoidal_amplitude : float - Amplitude of best-fit sine wave
        sinusoidal_fit_quality : float - R-squared of sine fit
        is_valid_transit : bool - True if shape is consistent with transit
        bls_depth_mismatch : bool - True if binned depth << BLS depth (artifact)
        bls_depth_ratio : float - Ratio of binned to BLS depth
        binned_depth_ppm : float - Measured depth from phase-binned data
    """
    if period is None or duration is None or t0 is None:
        return {
            'shape_classification': 'unknown',
            'in_transit_dip_sigma': 0,
            'sinusoidal_amplitude': 0,
            'sinusoidal_fit_quality': 0,
            'is_valid_transit': False,  # FAIL SECURE: reject if can't validate
            'bls_depth_mismatch': None,
            'bls_depth_ratio': None,
            'binned_depth_ppm': None
        }

    # Phase-fold the lightcurve
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0  # Center at 0

    # Bin the phase-folded data
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    binned_flux = np.zeros(n_bins)
    binned_std = np.zeros(n_bins)
    binned_count = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.sum(mask) >= 3:
            binned_flux[i] = np.median(flux[mask])
            binned_std[i] = np.std(flux[mask]) / np.sqrt(np.sum(mask))
            binned_count[i] = np.sum(mask)
        else:
            binned_flux[i] = np.nan
            binned_std[i] = np.nan

    # Remove NaN bins
    valid = ~np.isnan(binned_flux)
    if np.sum(valid) < 10:
        return {
            'shape_classification': 'unknown',
            'in_transit_dip_sigma': 0,
            'sinusoidal_amplitude': 0,
            'sinusoidal_fit_quality': 0,
            'is_valid_transit': False,  # FAIL SECURE: reject if insufficient bins
            'bls_depth_mismatch': None,
            'bls_depth_ratio': None,
            'binned_depth_ppm': None
        }

    bin_centers = bin_centers[valid]
    binned_flux = binned_flux[valid]
    binned_std = binned_std[valid]

    # 1. Check for significant dip at phase 0 (transit window)
    transit_phase_width = (duration / period) / 2  # Half-width in phase
    transit_phase_width = min(transit_phase_width, 0.15)  # Cap at ±0.15

    in_transit_mask = np.abs(bin_centers) < transit_phase_width
    out_of_transit_mask = np.abs(bin_centers) > 0.25  # Well away from transit

    if np.sum(in_transit_mask) < 3 or np.sum(out_of_transit_mask) < 10:
        return {
            'shape_classification': 'unknown',
            'in_transit_dip_sigma': 0,
            'sinusoidal_amplitude': 0,
            'sinusoidal_fit_quality': 0,
            'is_valid_transit': False,  # FAIL SECURE: reject if insufficient phase coverage
            'bls_depth_mismatch': None,
            'bls_depth_ratio': None,
            'binned_depth_ppm': None
        }

    baseline = np.median(binned_flux[out_of_transit_mask])
    baseline_std = np.std(binned_flux[out_of_transit_mask])
    in_transit_flux = np.median(binned_flux[in_transit_mask])

    # Dip significance: how many sigma below baseline is the transit?
    dip_sigma = (baseline - in_transit_flux) / baseline_std if baseline_std > 0 else 0

    # Physical constraint: transit must cause dimming (flux below baseline)
    # If in-transit flux >= baseline, this is not a real transit
    causes_dimming = in_transit_flux < baseline

    # ========== NEW: BLS-to-Binned Depth Consistency Check ==========
    # Calculate binned depth from phase-folded data
    binned_depth_ppm = (baseline - in_transit_flux) * 1e6  # Convert to ppm

    # Check if BLS-reported depth matches binned depth
    bls_depth_mismatch = False
    bls_depth_ratio = None

    if bls_reported_depth_ppm is not None and bls_reported_depth_ppm > 100:
        # Only check if BLS claimed significant depth (>100 ppm)
        # Use absolute value for binned depth since it could be negative if no real dip
        bls_depth_ratio = abs(binned_depth_ppm) / bls_reported_depth_ppm

        # If binned depth is <20% of BLS depth, this is a BLS artifact
        if bls_depth_ratio < 0.20:
            bls_depth_mismatch = True
            logger.debug(f"BLS-to-binned depth mismatch: BLS={bls_reported_depth_ppm:.0f} ppm, "
                        f"binned={binned_depth_ppm:.0f} ppm (ratio={bls_depth_ratio:.2f})")

    # ========== NEW: No Visible Transit Detection ==========
    # Catches cases where BLS reports high SNR but phase curve shows no dip
    no_visible_transit = False
    if dip_sigma < 2.0 and bls_reported_depth_ppm is not None and bls_reported_depth_ppm > 1000:
        # BLS claimed >0.1% depth but we see <2σ dip in phase curve
        no_visible_transit = True
        logger.debug(f"No visible transit despite BLS depth={bls_reported_depth_ppm:.0f} ppm "
                    f"(dip_sigma={dip_sigma:.2f} < 2.0)")

    # 2. Check for sinusoidal pattern
    # Fit a simple sine wave: f(phase) = A * sin(2*pi*phase) + B * cos(2*pi*phase) + C
    # This is linear in A, B, C so we can use least squares
    X = np.column_stack([
        np.sin(2 * np.pi * bin_centers),
        np.cos(2 * np.pi * bin_centers),
        np.ones_like(bin_centers)
    ])

    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, binned_flux, rcond=None)
        A, B, C = coeffs

        # Amplitude of the sinusoidal component
        sine_amplitude = np.sqrt(A**2 + B**2)

        # Fit quality (R-squared)
        fitted = X @ coeffs
        ss_res = np.sum((binned_flux - fitted) ** 2)
        ss_tot = np.sum((binned_flux - np.mean(binned_flux)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    except Exception:
        sine_amplitude = 0
        r_squared = 0

    # Classification logic
    # Priority order matters (updated 2025-12-22 for enhanced FP filtering):
    # 1. BLS_ARTIFACT: BLS depth doesn't match binned depth (BLS locked onto periodogram artifact)
    # 2. HIGHSNR_ARTIFACT: High BLS depth claimed but <2σ dip visible (BLS hallucinating)
    # 3. NO DIMMING: In-transit flux >= baseline means not a real transit
    # 4. SINUSOIDAL: Strong sine fit (R² > 0.5) - ellipsoidal variable or EB
    # 5. FLAT: No significant dip at phase 0 (< 4 sigma) - BLS triggered on noise
    # 6. TRANSIT: Significant dip, causes dimming, not sinusoidal - valid candidate
    #
    # Thresholds tuned based on visual inspection of false positives:
    # - dip_sigma threshold: 4.0σ for flat detection (binning artifacts produce 2-3σ)
    # - sine R²: 0.5 to catch weaker sinusoidal patterns
    # - depth ratio: 0.20 for BLS artifact detection (binned <20% of BLS = artifact)

    is_sinusoidal = r_squared > 0.5 and sine_amplitude > baseline_std
    is_flat = dip_sigma < 4.0 and not is_sinusoidal
    no_dimming = not causes_dimming

    # Determine classification with new artifact checks first
    if bls_depth_mismatch:
        # BLS reported deep transit but phase curve shows much shallower/no dip
        classification = 'flat_bls_artifact'
        is_valid = False
    elif no_visible_transit:
        # BLS reported high depth but <2σ dip in phase curve
        classification = 'flat_highsnr_artifact'
        is_valid = False
    elif no_dimming:
        # Flux goes UP at phase 0, not down - definitely not a transit
        classification = 'flat'
        is_valid = False
    elif is_sinusoidal:
        classification = 'sinusoidal'
        is_valid = False
    elif is_flat:
        classification = 'flat'
        is_valid = False
    else:
        classification = 'transit'
        is_valid = True

    return {
        'shape_classification': classification,
        'in_transit_dip_sigma': float(dip_sigma),
        'sinusoidal_amplitude': float(sine_amplitude),
        'sinusoidal_fit_quality': float(r_squared),
        'is_valid_transit': bool(is_valid),
        'bls_depth_mismatch': bool(bls_depth_mismatch),
        'bls_depth_ratio': float(bls_depth_ratio) if bls_depth_ratio is not None else None,
        'binned_depth_ppm': float(binned_depth_ppm)
    }


@dataclass
class TransitSearchResult:
    """Result of BLS transit search."""

    # Detection outcome
    is_transit_candidate: bool = False

    # Best-fit transit parameters
    bls_period: Optional[float] = None          # days
    bls_period_hours: Optional[float] = None    # hours (convenience)
    bls_period_error: Optional[float] = None    # days (from grid resolution)
    bls_t0: Optional[float] = None              # transit mid-time (BTJD)
    bls_duration: Optional[float] = None        # transit duration (days)
    bls_duration_hours: Optional[float] = None  # transit duration (hours)
    bls_depth: Optional[float] = None           # transit depth (fractional flux)
    bls_depth_ppm: Optional[float] = None       # transit depth (parts per million)

    # Detection statistics
    bls_snr: Optional[float] = None             # Signal-to-noise ratio (best)
    bls_snr_empirical: Optional[float] = None   # Empirical SNR
    bls_snr_bootstrap: Optional[float] = None   # Bootstrap SNR
    bls_power: Optional[float] = None           # BLS power at best period
    bls_depth_err: Optional[float] = None       # Depth uncertainty
    n_in_transit: Optional[int] = None          # Number of in-transit points
    out_of_transit_scatter_ppm: Optional[float] = None  # Out-of-transit scatter

    # Search parameters (for reproducibility)
    period_min: float = 0.2                     # days
    period_max: float = 2.0                     # days
    duration_min: float = 0.01                  # days (0.24 hours)
    duration_max: float = 0.1                   # days (2.4 hours)
    snr_threshold: float = 7.0

    # Data quality
    n_points: Optional[int] = None
    baseline_days: Optional[float] = None
    n_transits_expected: Optional[int] = None   # How many transits in baseline

    # Performance metrics (new for optimization)
    computation_time_ms: Optional[float] = None  # Total computation time
    stage1_period: Optional[float] = None        # Period from coarse search
    stage1_power: Optional[float] = None         # Power from coarse search
    binned_detection: bool = False               # Whether binning was used

    # Harmonic resolution
    original_detected_period: Optional[float] = None  # Period before harmonic resolution
    harmonic_resolution: Optional[str] = None         # 'original', 'half', 'double', or 'triple'
    harmonic_confidence: Optional[float] = None       # Confidence in resolution

    # Depth consistency (EB detection)
    depth_consistent: Optional[bool] = None           # True if depths are consistent across transits
    depth_variation: Optional[float] = None           # Coefficient of variation (std/mean)
    odd_even_diff_pct: Optional[float] = None         # Percent difference in odd/even depths
    n_transits_measured: Optional[int] = None         # Number of individual transits analyzed

    # Secondary eclipse detection (distinguishes planets from EBs)
    has_secondary_eclipse: Optional[bool] = None      # True if secondary eclipse detected at phase 0.5
    secondary_eclipse_depth_ppm: Optional[float] = None  # Depth of secondary eclipse
    secondary_to_primary_ratio: Optional[float] = None   # Ratio of secondary/primary depth
    secondary_eclipse_snr: Optional[float] = None        # SNR of secondary eclipse detection
    secondary_classification: Optional[str] = None       # 'planet', 'eb', 'uncertain'

    # Error information
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'is_transit_candidate': self.is_transit_candidate,
            'bls_period': self.bls_period,
            'bls_period_hours': self.bls_period_hours,
            'bls_period_error': self.bls_period_error,
            'bls_t0': self.bls_t0,
            'bls_duration': self.bls_duration,
            'bls_duration_hours': self.bls_duration_hours,
            'bls_depth': self.bls_depth,
            'bls_depth_ppm': self.bls_depth_ppm,
            'bls_snr': self.bls_snr,
            'bls_snr_empirical': self.bls_snr_empirical,
            'bls_snr_bootstrap': self.bls_snr_bootstrap,
            'bls_power': self.bls_power,
            'bls_depth_err': self.bls_depth_err,
            'n_in_transit': self.n_in_transit,
            'out_of_transit_scatter_ppm': self.out_of_transit_scatter_ppm,
            'period_min': self.period_min,
            'period_max': self.period_max,
            'duration_min': self.duration_min,
            'duration_max': self.duration_max,
            'snr_threshold': self.snr_threshold,
            'n_points': self.n_points,
            'baseline_days': self.baseline_days,
            'n_transits_expected': self.n_transits_expected,
            'computation_time_ms': self.computation_time_ms,
            'stage1_period': self.stage1_period,
            'stage1_power': self.stage1_power,
            'binned_detection': self.binned_detection,
            'original_detected_period': self.original_detected_period,
            'harmonic_resolution': self.harmonic_resolution,
            'harmonic_confidence': self.harmonic_confidence,
            'depth_consistent': self.depth_consistent,
            'depth_variation': self.depth_variation,
            'odd_even_diff_pct': self.odd_even_diff_pct,
            'n_transits_measured': self.n_transits_measured,
            'has_secondary_eclipse': self.has_secondary_eclipse,
            'secondary_eclipse_depth_ppm': self.secondary_eclipse_depth_ppm,
            'secondary_to_primary_ratio': self.secondary_to_primary_ratio,
            'secondary_eclipse_snr': self.secondary_eclipse_snr,
            'secondary_classification': self.secondary_classification,
            'error_message': self.error_message
        }


class TransitDetector:
    """
    Box Least Squares transit detector for TESS light curves.

    Optimized for Ultra-Short Period (USP) planet detection:
    - Period range: 0.2-2.0 days (default)
    - Duration range: 0.01-0.1 days (0.24-2.4 hours)
    - SNR threshold: 7.0 (conservative for validation)

    The BLS algorithm fits a periodic box model to the light curve,
    searching for flux decrements (transits) at a grid of periods
    and durations.
    """

    # Default USP-focused search parameters
    DEFAULT_PERIOD_MIN = 0.2     # days (4.8 hours)
    DEFAULT_PERIOD_MAX = 2.0     # days (48 hours)
    DEFAULT_DURATION_MIN = 0.01  # days (0.24 hours = 14.4 min)
    DEFAULT_DURATION_MAX = 0.1   # days (2.4 hours)
    DEFAULT_SNR_THRESHOLD = 7.0  # Conservative threshold

    # BLS computation parameters
    DEFAULT_OVERSAMPLE = 10      # Frequency oversampling factor
    DEFAULT_N_DURATIONS = 20     # Number of duration grid points

    def __init__(
        self,
        period_min: float = DEFAULT_PERIOD_MIN,
        period_max: float = DEFAULT_PERIOD_MAX,
        duration_min: float = DEFAULT_DURATION_MIN,
        duration_max: float = DEFAULT_DURATION_MAX,
        snr_threshold: float = DEFAULT_SNR_THRESHOLD,
        oversample: int = DEFAULT_OVERSAMPLE,
        n_durations: int = DEFAULT_N_DURATIONS
    ):
        """
        Initialize transit detector with search parameters.

        Args:
            period_min: Minimum period to search (days)
            period_max: Maximum period to search (days)
            duration_min: Minimum transit duration (days)
            duration_max: Maximum transit duration (days)
            snr_threshold: SNR threshold for candidate detection
            oversample: BLS frequency oversampling factor
            n_durations: Number of duration grid points
        """
        self.period_min = period_min
        self.period_max = period_max
        self.duration_min = duration_min
        self.duration_max = duration_max
        self.snr_threshold = snr_threshold
        self.oversample = oversample
        self.n_durations = n_durations

    def search(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        use_optimized: bool = True
    ) -> TransitSearchResult:
        """
        Search for periodic transits using Box Least Squares.

        Args:
            time: Array of observation times (days, e.g., BTJD)
            flux: Array of normalized flux values (fractional, centered at 0 or 1)
            flux_err: Optional array of flux uncertainties
            use_optimized: If True, use two-stage optimized search (faster)

        Returns:
            TransitSearchResult with detection outcome and parameters
        """
        # Initialize result with search parameters
        result = TransitSearchResult(
            period_min=self.period_min,
            period_max=self.period_max,
            duration_min=self.duration_min,
            duration_max=self.duration_max,
            snr_threshold=self.snr_threshold
        )

        # Use optimized two-stage search
        if use_optimized:
            # Call the optimized function
            opt_result = bls_transit_search(
                time, flux, flux_err,
                min_period=self.period_min,
                max_period=self.period_max,
                snr_threshold=self.snr_threshold,
                auto_optimize=True
            )

            # Map optimized result to TransitSearchResult
            result.is_transit_candidate = opt_result.get('is_candidate', False)
            result.bls_period = opt_result.get('period')
            result.bls_period_hours = opt_result['period'] * 24.0 if opt_result.get('period') else None
            result.bls_t0 = opt_result.get('t0')
            result.bls_duration = opt_result.get('duration')
            result.bls_duration_hours = opt_result.get('duration_hours')
            result.bls_depth = opt_result.get('depth')
            result.bls_depth_ppm = opt_result.get('depth_ppm')
            result.bls_snr = opt_result.get('snr')
            result.bls_power = opt_result.get('power')
            result.n_points = opt_result.get('n_points')
            result.baseline_days = opt_result.get('baseline_days')
            result.computation_time_ms = opt_result.get('computation_time_ms')
            result.stage1_period = opt_result.get('stage1_period')
            result.stage1_power = opt_result.get('stage1_power')
            result.binned_detection = opt_result.get('binned_detection', False)
            result.error_message = opt_result.get('error')

            # Calculate expected transits and period error if period found
            if result.bls_period and result.baseline_days:
                result.n_transits_expected = int(result.baseline_days / result.bls_period)
                # Estimate period error from grid resolution
                period_resolution = (self.period_max - self.period_min) / BLS_FINE_PERIODS
                result.bls_period_error = period_resolution

            return result

        # Legacy path: original autopower-based search (slower but thorough)
        # Keeping for backwards compatibility and edge cases
        start_time = timer.perf_counter()

        # Validate input
        if len(time) < 100:
            result.error_message = f"Insufficient data points: {len(time)} < 100"
            logger.warning(f"BLS search failed: {result.error_message}")
            return result

        # Clean data
        mask = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            mask &= np.isfinite(flux_err)

        time_clean = time[mask]
        flux_clean = flux[mask]
        flux_err_clean = flux_err[mask] if flux_err is not None else None

        if len(time_clean) < 100:
            result.error_message = f"Insufficient valid data after cleaning: {len(time_clean)}"
            logger.warning(f"BLS search failed: {result.error_message}")
            return result

        # Record data quality
        result.n_points = len(time_clean)
        result.baseline_days = float(np.ptp(time_clean))

        # Ensure flux is properly normalized
        flux_median = np.median(flux_clean)
        if abs(flux_median) < 0.5:
            flux_normalized = flux_clean + 1.0
        else:
            flux_normalized = flux_clean

        try:
            # Create BLS model
            if flux_err_clean is not None:
                model = BoxLeastSquares(time_clean, flux_normalized, dy=flux_err_clean)
            else:
                model = BoxLeastSquares(time_clean, flux_normalized)

            # Create duration grid
            durations = np.geomspace(
                self.duration_min,
                self.duration_max,
                self.n_durations
            )

            # Run BLS periodogram (legacy autopower - slower)
            logger.debug(f"Running BLS (legacy): P=[{self.period_min:.2f}, {self.period_max:.2f}] days")

            periodogram = model.autopower(
                duration=durations,
                minimum_period=self.period_min,
                maximum_period=self.period_max,
                minimum_n_transit=2,
                frequency_factor=1.0,
                oversample=self.oversample
            )

            # Find best period
            best_idx = np.argmax(periodogram.power)
            best_period = float(periodogram.period[best_idx])
            best_power = float(periodogram.power[best_idx])
            best_duration = float(periodogram.duration[best_idx])
            best_t0 = float(periodogram.transit_time[best_idx])

            # Get transit parameters
            stats = model.compute_stats(best_period, best_duration, best_t0)
            depth_arr = stats['depth']
            depth = float(depth_arr[0]) if hasattr(depth_arr, '__len__') else float(depth_arr)
            depth_err_arr = stats['depth_err']
            depth_err = float(depth_err_arr[0]) if hasattr(depth_err_arr, '__len__') else float(depth_err_arr)

            # Calculate SNR
            if depth_err > 0 and depth_err < depth:
                snr = depth / depth_err
            else:
                rms = np.std(flux_normalized)
                snr = depth / rms if rms > 0 else 0.0

            # Record results
            result.bls_period = best_period
            result.bls_period_hours = best_period * 24.0
            result.bls_t0 = best_t0
            result.bls_duration = best_duration
            result.bls_duration_hours = best_duration * 24.0
            result.bls_depth = depth
            result.bls_depth_ppm = depth * 1e6
            result.bls_snr = float(snr)
            result.bls_power = best_power
            result.computation_time_ms = (timer.perf_counter() - start_time) * 1000

            # Period error and transits
            freq_resolution = 1.0 / (result.baseline_days * self.oversample)
            result.bls_period_error = (best_period ** 2) * freq_resolution
            result.n_transits_expected = int(result.baseline_days / best_period)

            result.is_transit_candidate = (snr >= self.snr_threshold)

            if result.is_transit_candidate:
                logger.info(f"Transit candidate: P={best_period:.6f}d, SNR={snr:.1f}")

        except Exception as e:
            result.error_message = f"BLS computation failed: {str(e)}"
            logger.error(f"BLS search error: {e}", exc_info=True)

        return result


# ============================================================================
# OPTIMIZED TWO-STAGE BLS FUNCTIONS
# ============================================================================

def bin_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    bin_size: float = BLS_BIN_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin lightcurve to reduce data volume for initial detection.

    Parameters
    ----------
    time : array-like
        Time values in days
    flux : array-like
        Flux values
    bin_size : float
        Bin size in days (default 0.02 = ~29 minutes)

    Returns
    -------
    tuple of (binned_time, binned_flux, binned_err)
    """
    time = np.asarray(time)
    flux = np.asarray(flux)

    # Create bin edges
    t_min, t_max = time.min(), time.max()
    bin_edges = np.arange(t_min, t_max + bin_size, bin_size)

    # Compute bin indices
    bin_indices = np.digitize(time, bin_edges)

    # Aggregate
    binned_time = []
    binned_flux = []
    binned_err = []

    for i in range(1, len(bin_edges)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            binned_time.append(np.mean(time[mask]))
            binned_flux.append(np.mean(flux[mask]))
            # Error is standard error of mean
            binned_err.append(np.std(flux[mask]) / np.sqrt(np.sum(mask)))

    return np.array(binned_time), np.array(binned_flux), np.array(binned_err)


def get_duration_grid(min_period: float, max_period: float) -> np.ndarray:
    """
    Select appropriate duration grid based on period range.

    USP planets (P < 1d) have shorter transits: 10-90 minutes typical.
    Transit duration scales approximately as P^(1/3) for circular orbits.

    Parameters
    ----------
    min_period : float
        Minimum search period in days
    max_period : float
        Maximum search period in days

    Returns
    -------
    np.ndarray
        Array of transit durations to search (in days)
    """
    # Maximum duration must be shorter than minimum period (BLS requirement)
    # Use 80% of min_period as safety margin
    max_allowed_duration = min_period * 0.8

    if max_period <= 1.0:
        # USP regime: use fine duration grid covering 14 min to 4 hours
        durations = BLS_USP_DURATIONS
    else:
        # Standard regime: scale with period, 30 min to 6 hours
        # Extended from 3.6h to 6h to catch longer-duration transits around larger stars
        durations = np.geomspace(0.02, 0.25, 12)

    # Filter durations to be less than max_allowed_duration
    valid_durations = durations[durations < max_allowed_duration]

    # Ensure we have at least a few durations to search
    if len(valid_durations) < 3:
        # Create a minimal grid that fits within the constraint
        valid_durations = np.geomspace(0.01, max_allowed_duration * 0.9, 5)

    return valid_durations


def _bls_two_stage_search(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    min_period: float = BLS_MIN_PERIOD,
    max_period: float = BLS_MAX_PERIOD,
    snr_threshold: float = BLS_DEFAULT_SNR_THRESHOLD
) -> Dict[str, Any]:
    """
    Two-stage BLS transit search optimized for USP planets.

    Stage 1: Coarse grid detection (fast)
    Stage 2: Fine grid refinement around peak (accurate)

    Parameters
    ----------
    time : array-like
        Time values in days (BJD - 2457000 or similar)
    flux : array-like
        Normalized flux values (centered at 1.0)
    flux_err : array-like, optional
        Flux uncertainties
    min_period : float
        Minimum search period in days
    max_period : float
        Maximum search period in days
    snr_threshold : float
        SNR threshold for candidate flagging

    Returns
    -------
    dict with transit detection results
    """
    start_time = timer.perf_counter()

    result = {
        'period': None,
        't0': None,
        'duration': None,
        'duration_hours': None,
        'depth': None,
        'depth_ppm': None,
        'snr': None,
        'power': None,
        'is_candidate': False,
        'stage1_period': None,
        'stage1_power': None,
        'computation_time_ms': 0,
        'n_points': len(time),
        'baseline_days': float(np.ptp(time)),
        'error': None
    }

    # Input validation
    if len(time) < 100:
        result['error'] = f"Insufficient data points: {len(time)} < 100"
        result['computation_time_ms'] = (timer.perf_counter() - start_time) * 1000
        return result

    # Create BLS model
    try:
        if flux_err is not None:
            bls = BoxLeastSquares(time, flux, dy=flux_err)
        else:
            bls = BoxLeastSquares(time, flux)
    except Exception as e:
        result['error'] = f"BLS model creation failed: {str(e)}"
        result['computation_time_ms'] = (timer.perf_counter() - start_time) * 1000
        return result

    # ========== STAGE 1: Coarse Detection ==========
    coarse_periods = np.linspace(min_period, max_period, BLS_COARSE_PERIODS)

    # Use multi-duration grid for better USP transit detection
    # Single duration (1.2 hr) missed 10-90 minute transits entirely
    coarse_durations = get_duration_grid(min_period, max_period)

    try:
        coarse_result = bls.power(
            coarse_periods,
            duration=coarse_durations,
            objective='snr'
        )
    except Exception as e:
        result['error'] = f"Stage 1 failed: {str(e)}"
        result['computation_time_ms'] = (timer.perf_counter() - start_time) * 1000
        return result

    # Find peak in coarse grid
    best_idx = np.argmax(coarse_result.power)
    stage1_period = float(coarse_periods[best_idx])
    stage1_power = float(coarse_result.power[best_idx])

    result['stage1_period'] = stage1_period
    result['stage1_power'] = stage1_power

    # Early exit if no significant signal
    # With multi-duration, power values should be higher than single-duration
    if stage1_power < BLS_COARSE_POWER_THRESHOLD:
        result['error'] = f"No significant signal in coarse search (power={stage1_power:.3f})"
        result['computation_time_ms'] = (timer.perf_counter() - start_time) * 1000
        logger.debug(f"Stage 1: No signal above threshold (power={stage1_power:.3f})")
        return result

    # ========== STAGE 2: Fine Refinement ==========
    # Narrow period range around peak
    fine_min = max(min_period, stage1_period * (1 - BLS_FINE_WINDOW))
    fine_max = min(max_period, stage1_period * (1 + BLS_FINE_WINDOW))
    fine_periods = np.linspace(fine_min, fine_max, BLS_FINE_PERIODS)

    # Duration grid scaled to period (transit ~2-20% of period for USP)
    duration_min = max(0.01, stage1_period * 0.02)  # Minimum 15 min or 2% of period
    duration_max = min(0.15, stage1_period * 0.20)  # Maximum 3.6 hours or 20% of period
    fine_durations = np.linspace(duration_min, duration_max, BLS_FINE_DURATIONS)

    try:
        fine_result = bls.power(
            fine_periods,
            duration=fine_durations,
            objective='snr'
        )
    except Exception as e:
        result['error'] = f"Stage 2 failed: {str(e)}"
        result['computation_time_ms'] = (timer.perf_counter() - start_time) * 1000
        return result

    # Extract best parameters
    best_idx = np.argmax(fine_result.power)
    best_period = float(fine_periods[best_idx])
    best_power = float(fine_result.power[best_idx])
    best_duration = float(fine_result.duration[best_idx])
    best_t0 = float(fine_result.transit_time[best_idx])

    # Get transit parameters at best period
    depth = None
    depth_err = None
    snr = None

    try:
        stats = bls.compute_stats(best_period, best_duration, best_t0)

        # Extract depth - handle different return formats
        if 'depth' in stats:
            depth_val = stats['depth']
            depth = float(depth_val[0]) if hasattr(depth_val, '__len__') else float(depth_val)

        # Extract depth error if available
        if 'depth_err' in stats:
            depth_err_val = stats['depth_err']
            if depth_err_val is not None:
                depth_err = float(depth_err_val[0]) if hasattr(depth_err_val, '__len__') else float(depth_err_val)

    except Exception as e:
        logger.debug(f"Stats computation issue: {e}")

    # ========== HARMONIC RESOLUTION ==========
    # Check if P/2 or 2P produces a better fit
    harmonic_result = resolve_harmonic(best_period, time, flux, flux_err)
    original_period = best_period

    if harmonic_result['resolution'] != 'original' and harmonic_result['best_period'] is not None:
        resolved_period = harmonic_result['best_period']
        logger.debug(f"Harmonic resolution: {best_period:.6f}d -> {resolved_period:.6f}d "
                    f"({harmonic_result['resolution']})")

        # Re-run BLS at resolved period to get updated parameters
        try:
            test_durations = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
            test_durations = test_durations[test_durations < resolved_period * 0.2]
            if len(test_durations) == 0:
                test_durations = np.array([resolved_period * 0.1])

            resolved_result = bls.power(np.array([resolved_period]), duration=test_durations)
            best_period = resolved_period
            best_power = float(resolved_result.power[0])
            best_duration = float(resolved_result.duration[0])
            best_t0 = float(resolved_result.transit_time[0])

            # Re-compute stats at resolved period
            stats = bls.compute_stats(best_period, best_duration, best_t0)
            if 'depth' in stats:
                depth_val = stats['depth']
                depth = float(depth_val[0]) if hasattr(depth_val, '__len__') else float(depth_val)
        except Exception as e:
            logger.debug(f"Harmonic re-computation issue: {e}")

    # ========== ROBUST SNR CALCULATION ==========
    snr_result = calculate_transit_snr(time, flux, best_period, best_duration, best_t0, flux_err)
    snr = snr_result['snr_best']

    # If robust SNR is 0 (not enough points), fall back to BLS power
    if snr == 0:
        snr = best_power

    # ========== DEPTH CONSISTENCY CHECK (EB DETECTION) ==========
    depth_check = check_depth_consistency(time, flux, best_period, best_duration, best_t0)

    # ========== TRANSIT SHAPE CHECK (FLAT/SINUSOIDAL DETECTION) ==========
    # Pass BLS-reported depth for consistency check
    bls_depth_ppm = snr_result['depth_ppm'] if snr_result['depth_ppm'] != 0 else (depth * 1e6 if depth else None)
    shape_check = check_transit_shape(
        time, flux, best_period, best_duration, best_t0,
        bls_reported_depth_ppm=bls_depth_ppm
    )

    # ========== HARD GATE: Reject candidates with unknown shape classification ==========
    # If shape validation couldn't compute metrics, reject even if BLS SNR is high
    is_candidate = snr >= snr_threshold
    rejection_reason = None

    if is_candidate and shape_check['shape_classification'] == 'unknown':
        logger.warning(f"Shape validation failed (classification=unknown) - "
                      f"rejecting candidate despite SNR={snr:.1f}")
        is_candidate = False
        rejection_reason = 'shape_validation_failed'
    elif is_candidate and not shape_check['is_valid_transit']:
        # Shape check identified a false positive pattern
        rejection_reason = shape_check['shape_classification']
        # Note: we keep is_candidate=True but mark the rejection reason
        # The auto-vetting in transit_tasks.py will use this to set vetting_status

    # ========== ODD/EVEN CONSISTENCY GATE ==========
    # Reject candidates where odd/even depths are inconsistent (noise/artifacts)
    # Criteria: >50% mismatch, depths <50 ppm (noise), or negative depths
    if is_candidate and not depth_check.get('odd_even_valid', True):
        odd_reason = depth_check.get('odd_even_rejection_reason', 'odd_even_invalid')
        logger.warning(f"Odd/even consistency check failed ({odd_reason}) - "
                      f"odd={depth_check.get('odd_depth_ppm'):.0f} ppm, "
                      f"even={depth_check.get('even_depth_ppm'):.0f} ppm, "
                      f"diff={depth_check.get('odd_even_diff_pct', 0):.0f}%")
        is_candidate = False
        rejection_reason = odd_reason

    elapsed_ms = (timer.perf_counter() - start_time) * 1000

    result.update({
        'period': best_period,
        't0': best_t0,
        'duration': best_duration,
        'duration_hours': best_duration * 24.0,
        'depth': snr_result['depth'] if snr_result['depth'] != 0 else depth,
        'depth_ppm': bls_depth_ppm,
        'depth_err': snr_result['depth_err'],
        'snr': float(snr),
        'snr_empirical': snr_result['snr_empirical'],
        'snr_bootstrap': snr_result['snr_bootstrap'],
        'power': best_power,
        'is_candidate': is_candidate,
        'n_in_transit': snr_result['n_in_transit'],
        'out_of_transit_scatter_ppm': snr_result['out_of_transit_scatter_ppm'],
        'original_detected_period': original_period if original_period != best_period else None,
        'harmonic_resolution': harmonic_result['resolution'] if original_period != best_period else None,
        'harmonic_confidence': harmonic_result['confidence'] if original_period != best_period else None,
        'depth_consistent': depth_check['is_consistent'],
        'depth_variation': depth_check['depth_variation'],
        'odd_even_diff_pct': depth_check['odd_even_diff_pct'],
        'odd_depth_ppm': depth_check.get('odd_depth_ppm'),
        'even_depth_ppm': depth_check.get('even_depth_ppm'),
        'odd_even_valid': depth_check.get('odd_even_valid', True),
        'n_transits_measured': depth_check['n_transits'],
        # Shape check results
        'shape_classification': shape_check['shape_classification'],
        'in_transit_dip_sigma': shape_check['in_transit_dip_sigma'],
        'sinusoidal_amplitude': shape_check['sinusoidal_amplitude'],
        'sinusoidal_fit_quality': shape_check['sinusoidal_fit_quality'],
        'is_valid_transit_shape': shape_check['is_valid_transit'],
        # New FP filtering fields
        'bls_depth_mismatch': shape_check.get('bls_depth_mismatch'),
        'bls_depth_ratio': shape_check.get('bls_depth_ratio'),
        'binned_depth_ppm': shape_check.get('binned_depth_ppm'),
        'rejection_reason': rejection_reason,
        'computation_time_ms': elapsed_ms
    })

    if result['is_candidate']:
        logger.info(f"Transit candidate: P={best_period:.6f}d ({best_period*24:.2f}h), "
                   f"SNR={snr:.1f}, time={elapsed_ms:.0f}ms")
        # Warn if depth consistency check fails (possible EB)
        if not depth_check['is_consistent']:
            logger.warning(f"  DEPTH INCONSISTENT: variation={depth_check['depth_variation']:.1%}, "
                          f"odd/even diff={depth_check['odd_even_diff_pct']:.0f}% - possible EB")
        # Warn if shape check fails (flat or sinusoidal)
        if not shape_check['is_valid_transit']:
            logger.warning(f"  INVALID SHAPE: {shape_check['shape_classification']} "
                          f"(dip_sigma={shape_check['in_transit_dip_sigma']:.1f}, "
                          f"sine_r2={shape_check['sinusoidal_fit_quality']:.2f})")
    else:
        logger.debug(f"No significant transit: P={best_period:.6f}d, SNR={snr:.1f}")

    return result


def bls_transit_search(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    min_period: float = BLS_MIN_PERIOD,
    max_period: float = BLS_MAX_PERIOD,
    snr_threshold: float = BLS_DEFAULT_SNR_THRESHOLD,
    auto_optimize: bool = True,
    max_points_unbinned: int = BLS_MAX_POINTS_UNBINNED
) -> Dict[str, Any]:
    """
    Main BLS transit search entry point with automatic optimization.

    For long baselines (>20,000 points), automatically bins data
    for Stage 1 coarse detection, then uses full resolution for
    Stage 2 refinement.

    Parameters
    ----------
    time, flux, flux_err : array-like
        Lightcurve data
    min_period, max_period : float
        Search range in days
    snr_threshold : float
        SNR threshold for candidate flagging
    auto_optimize : bool
        If True, automatically bin long baselines
    max_points_unbinned : int
        Threshold for automatic binning

    Returns
    -------
    dict with transit parameters and performance metrics
    """
    time = np.asarray(time)
    flux = np.asarray(flux)

    # Clean data
    mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        flux_err = np.asarray(flux_err)
        mask &= np.isfinite(flux_err)
        flux_err_clean = flux_err[mask]
    else:
        flux_err_clean = None

    time_clean = time[mask]
    flux_clean = flux[mask]

    # Ensure flux is centered at 1.0
    flux_median = np.median(flux_clean)
    if abs(flux_median) < 0.5:
        flux_normalized = flux_clean + 1.0
    else:
        flux_normalized = flux_clean

    # Auto-bin if needed
    binned_detection = False
    if auto_optimize and len(time_clean) > max_points_unbinned:
        logger.debug(f"Auto-binning: {len(time_clean)} points -> binning")

        # Use binned data for coarse detection
        binned_time, binned_flux, binned_err = bin_lightcurve(time_clean, flux_normalized)
        binned_detection = True

        # Run two-stage on binned data
        result = _bls_two_stage_search(
            binned_time, binned_flux, binned_err,
            min_period, max_period, snr_threshold
        )

        # If candidate found, refine with full-resolution data
        if result['is_candidate'] and result['period'] is not None:
            logger.debug(f"Refining with full-resolution data around P={result['period']:.6f}d")
            refined = _refine_with_full_data(
                time_clean, flux_normalized, flux_err_clean,
                result['period'], snr_threshold
            )
            # Update with refined values
            result.update(refined)
            result['n_points'] = len(time_clean)  # Report full data count

        result['binned_detection'] = True
    else:
        result = _bls_two_stage_search(
            time_clean, flux_normalized, flux_err_clean,
            min_period, max_period, snr_threshold
        )
        result['binned_detection'] = False

    return result


def _refine_with_full_data(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray],
    initial_period: float,
    snr_threshold: float
) -> Dict[str, Any]:
    """Refine transit parameters using full-resolution data."""
    start_time = timer.perf_counter()

    if flux_err is not None:
        bls = BoxLeastSquares(time, flux, dy=flux_err)
    else:
        bls = BoxLeastSquares(time, flux)

    # Fine grid around initial period
    fine_min = initial_period * (1 - BLS_FINE_WINDOW)
    fine_max = initial_period * (1 + BLS_FINE_WINDOW)
    fine_periods = np.linspace(fine_min, fine_max, BLS_FINE_PERIODS)

    # Duration grid
    duration_min = max(0.01, initial_period * 0.02)
    duration_max = min(0.15, initial_period * 0.20)
    fine_durations = np.linspace(duration_min, duration_max, BLS_FINE_DURATIONS)

    try:
        fine_result = bls.power(fine_periods, duration=fine_durations, objective='snr')

        best_idx = np.argmax(fine_result.power)
        best_period = float(fine_periods[best_idx])
        best_duration = float(fine_result.duration[best_idx])
        best_t0 = float(fine_result.transit_time[best_idx])
        best_power = float(fine_result.power[best_idx])

        stats = bls.compute_stats(best_period, best_duration, best_t0)

        depth_arr = stats['depth']
        depth = float(depth_arr[0]) if hasattr(depth_arr, '__len__') else float(depth_arr)

        depth_err_arr = stats['depth_err']
        depth_err = float(depth_err_arr[0]) if hasattr(depth_err_arr, '__len__') else float(depth_err_arr)

        if depth_err > 0 and depth_err < depth:
            snr = depth / depth_err
        else:
            snr = best_power

        elapsed_ms = (timer.perf_counter() - start_time) * 1000

        return {
            'period': best_period,
            't0': best_t0,
            'duration': best_duration,
            'duration_hours': best_duration * 24.0,
            'depth': depth,
            'depth_ppm': depth * 1e6,
            'snr': float(snr),
            'power': best_power,
            'is_candidate': snr >= snr_threshold,
            'refined_with_full_data': True,
            'refinement_time_ms': elapsed_ms
        }
    except Exception as e:
        logger.warning(f"Refinement failed: {e}")
        return {'refined_with_full_data': False}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def detect_transits(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    period_min: float = 0.2,
    period_max: float = 2.0,
    snr_threshold: float = 7.0
) -> TransitSearchResult:
    """
    Convenience function for transit detection.

    Args:
        time: Observation times (days)
        flux: Normalized flux
        flux_err: Optional flux uncertainties
        period_min: Minimum period (days)
        period_max: Maximum period (days)
        snr_threshold: SNR threshold for candidates

    Returns:
        TransitSearchResult
    """
    detector = TransitDetector(
        period_min=period_min,
        period_max=period_max,
        snr_threshold=snr_threshold
    )
    return detector.search(time, flux, flux_err)


def compare_bls_with_known_period(
    time: np.ndarray,
    flux: np.ndarray,
    known_period: float,
    tolerance_percent: float = 5.0,
    flux_err: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Run BLS and compare detected period with known value.

    Used for validation against known USP planets.

    Args:
        time: Observation times
        flux: Normalized flux
        known_period: Known orbital period (days)
        tolerance_percent: Match tolerance (default 5%)
        flux_err: Optional flux uncertainties

    Returns:
        Dictionary with:
        - detected_period: BLS-detected period
        - known_period: Input known period
        - period_diff_percent: Percentage difference (direct comparison)
        - is_match: True if within tolerance (direct, half, or double)
        - match_type: 'direct', 'half_harmonic', 'double_harmonic', or 'no_match'
        - bls_snr: Detection SNR
        - computation_time_ms: Time taken for BLS search
        - binned_detection: Whether binning was used
    """
    detector = TransitDetector()
    result = detector.search(time, flux, flux_err)

    if result.bls_period is None:
        return {
            'detected_period': None,
            'detected_period_hours': None,
            'known_period': known_period,
            'known_period_hours': known_period * 24.0,
            'period_diff_percent': None,
            'is_match': False,
            'match_type': 'detection_failed',
            'bls_snr': None,
            'bls_depth_ppm': None,
            'n_transits_expected': None,
            'computation_time_ms': result.computation_time_ms,
            'binned_detection': result.binned_detection,
            'stage1_period': result.stage1_period,
            'stage1_power': result.stage1_power,
            'error': result.error_message
        }

    detected = result.bls_period

    # Use comprehensive harmonic checking
    harmonic_check = check_period_harmonic(
        detected, known_period, tolerance=tolerance_percent / 100
    )

    # Direct comparison for reporting
    diff_direct = abs(detected - known_period) / known_period * 100

    # Map harmonic types to match types for backward compatibility
    harmonic_to_match = {
        'direct': 'direct',
        'half': 'half_harmonic',
        'double': 'double_harmonic',
        'triple': 'triple_harmonic',
        'sesqui': 'sesqui_harmonic',
        'alias': 'alias_harmonic'
    }

    if harmonic_check['is_harmonic']:
        match_type = harmonic_to_match.get(
            harmonic_check['harmonic_type'],
            f"{harmonic_check['harmonic_type']}_harmonic"
        )
    else:
        match_type = 'no_match'

    return {
        'detected_period': detected,
        'detected_period_hours': detected * 24.0,
        'known_period': known_period,
        'known_period_hours': known_period * 24.0,
        'period_diff_percent': diff_direct,  # Always report direct comparison
        'is_match': harmonic_check['is_harmonic'],
        'match_type': match_type,
        'harmonic_ratio': harmonic_check['harmonic_ratio'],
        'harmonic_error_pct': harmonic_check['period_error_pct'],
        'bls_snr': result.bls_snr,
        'bls_snr_empirical': result.bls_snr_empirical,
        'bls_snr_bootstrap': result.bls_snr_bootstrap,
        'bls_depth_ppm': result.bls_depth_ppm,
        'bls_depth_err': result.bls_depth_err,
        'bls_power': result.bls_power,
        'n_transits_expected': result.n_transits_expected,
        'n_points': result.n_points,
        'n_in_transit': result.n_in_transit,
        'out_of_transit_scatter_ppm': result.out_of_transit_scatter_ppm,
        'baseline_days': result.baseline_days,
        'computation_time_ms': result.computation_time_ms,
        'binned_detection': result.binned_detection,
        'stage1_period': result.stage1_period,
        'stage1_power': result.stage1_power,
        'original_detected_period': result.original_detected_period,
        'harmonic_resolution': result.harmonic_resolution,
        'error': result.error_message
    }

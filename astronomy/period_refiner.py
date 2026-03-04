"""
Period Refinement for VSX Submissions

This module provides high-precision period refinement using:
1. PDM (Phase Dispersion Minimization) - coarse refinement
2. String Length Method - fine refinement for sinusoidal variables
3. Phase scatter calculation - quality metric
4. Spline-based epoch determination - precise maximum/minimum

The goal is to achieve publication-quality phase folding with phase scatter < 0.05.

References:
- Stellingwerf (1978) - Phase Dispersion Minimization
- Lafler & Kinman (1965) - String Length Method
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PDM (PHASE DISPERSION MINIMIZATION)
# =============================================================================

def pdm_refine(
    time: np.ndarray,
    flux: np.ndarray,
    initial_period: float,
    search_fraction: float = 0.001,
    n_trials: int = 1000,
    n_bins: int = 20
) -> Tuple[float, float, float]:
    """
    Refine period using Phase Dispersion Minimization.

    PDM minimizes the variance within phase bins compared to total variance.
    Lower theta values indicate better period solutions.

    Args:
        time: Observation times (days)
        flux: Normalized flux values
        initial_period: Initial period from Lomb-Scargle (days)
        search_fraction: Search range as fraction of period (+/- this amount)
        n_trials: Number of trial periods to test
        n_bins: Number of phase bins

    Returns:
        Tuple of (refined_period, theta_min, period_error)
    """
    # Search range
    p_min = initial_period * (1 - search_fraction)
    p_max = initial_period * (1 + search_fraction)
    trial_periods = np.linspace(p_min, p_max, n_trials)

    # Calculate total variance (constant for all trials)
    total_variance = np.var(flux)
    if total_variance == 0:
        logger.warning("Zero variance in flux - cannot compute PDM")
        return initial_period, 1.0, 0.0

    # Calculate PDM theta for each trial period
    theta_values = np.zeros(n_trials)

    for i, p in enumerate(trial_periods):
        theta_values[i] = _calculate_pdm_theta(time, flux, p, n_bins, total_variance)

    # Find minimum
    best_idx = np.argmin(theta_values)
    refined_period = trial_periods[best_idx]
    theta_min = theta_values[best_idx]

    # Estimate error from width of minimum
    # Period error is approximately where theta increases by sqrt(2)
    threshold = theta_min * np.sqrt(2)
    error_mask = theta_values < threshold

    if np.sum(error_mask) > 1:
        error_range = trial_periods[error_mask]
        period_error = (error_range[-1] - error_range[0]) / 2
    else:
        # Fallback: use trial period spacing
        period_error = (p_max - p_min) / n_trials

    logger.info(f"PDM refinement: {initial_period:.8f} -> {refined_period:.8f} (theta={theta_min:.4f})")

    return refined_period, theta_min, period_error


def _calculate_pdm_theta(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    n_bins: int,
    total_variance: float
) -> float:
    """
    Calculate PDM theta statistic for a given period.

    Theta = (variance within bins) / (total variance)
    Lower theta = better period
    """
    # Fold data on trial period
    phases = (time % period) / period

    # Bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Calculate weighted variance within each bin
    variance_sum = 0.0
    total_points = 0

    for i in range(n_bins):
        mask = (phases >= bin_edges[i]) & (phases < bin_edges[i + 1])
        bin_flux = flux[mask]

        if len(bin_flux) > 1:
            variance_sum += np.var(bin_flux) * len(bin_flux)
            total_points += len(bin_flux)

    if total_points == 0:
        return np.inf

    variance_within = variance_sum / total_points
    theta = variance_within / total_variance

    return theta


# =============================================================================
# STRING LENGTH METHOD
# =============================================================================

def string_length_refine(
    time: np.ndarray,
    flux: np.ndarray,
    initial_period: float,
    search_fraction: float = 0.0001,
    n_trials: int = 500
) -> Tuple[float, float, float]:
    """
    Refine period using String Length Method.

    The "string" is the path connecting points in phase order.
    Shorter string length = better period (points align more smoothly).

    Best for sinusoidal variables (DSCT, RR Lyrae, etc.)

    Args:
        time: Observation times (days)
        flux: Normalized flux values
        initial_period: Initial period from PDM refinement (days)
        search_fraction: Search range as fraction of period (+/- this amount)
        n_trials: Number of trial periods to test

    Returns:
        Tuple of (refined_period, min_string_length, period_error)
    """
    # Normalize flux to [0, 1] range for consistent string length calculation
    flux_min = np.min(flux)
    flux_max = np.max(flux)
    flux_range = flux_max - flux_min

    if flux_range == 0:
        logger.warning("Zero flux range - cannot compute string length")
        return initial_period, 0.0, 0.0

    flux_norm = (flux - flux_min) / flux_range

    # Search range
    p_min = initial_period * (1 - search_fraction)
    p_max = initial_period * (1 + search_fraction)
    trial_periods = np.linspace(p_min, p_max, n_trials)

    # Calculate string length for each trial period
    string_lengths = np.zeros(n_trials)

    for i, p in enumerate(trial_periods):
        string_lengths[i] = _calculate_string_length(time, flux_norm, p)

    # Find minimum
    best_idx = np.argmin(string_lengths)
    refined_period = trial_periods[best_idx]
    min_length = string_lengths[best_idx]

    # Estimate error from width of minimum (5% increase threshold)
    threshold = min_length * 1.05
    error_mask = string_lengths < threshold

    if np.sum(error_mask) > 1:
        error_range = trial_periods[error_mask]
        period_error = (error_range[-1] - error_range[0]) / 2
    else:
        period_error = (p_max - p_min) / n_trials

    logger.info(f"String length refinement: {initial_period:.8f} -> {refined_period:.8f}")

    return refined_period, min_length, period_error


def _calculate_string_length(
    time: np.ndarray,
    flux_norm: np.ndarray,
    period: float
) -> float:
    """
    Calculate string length for a given period.

    String length = sum of distances between consecutive points in phase order.
    """
    # Fold on trial period
    phases = (time % period) / period

    # Sort by phase
    sorted_idx = np.argsort(phases)
    phases_sorted = phases[sorted_idx]
    flux_sorted = flux_norm[sorted_idx]

    # Calculate segment lengths
    dphase = np.diff(phases_sorted)
    dflux = np.diff(flux_sorted)
    segment_lengths = np.sqrt(dphase**2 + dflux**2)
    total_length = np.sum(segment_lengths)

    # Add wrap-around segment (last point to first point)
    wrap_dphase = phases_sorted[0] - phases_sorted[-1] + 1
    wrap_dflux = flux_sorted[0] - flux_sorted[-1]
    wrap_length = np.sqrt(wrap_dphase**2 + wrap_dflux**2)
    total_length += wrap_length

    return total_length


# =============================================================================
# PHASE SCATTER CALCULATION
# =============================================================================

def calculate_phase_scatter(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    n_bins: int = 50
) -> float:
    """
    Calculate phase scatter metric for quality assessment.

    This measures how tightly points align in the phase diagram.
    Lower values = better period refinement.

    Args:
        time: Observation times (days)
        flux: Normalized flux values
        period: Period to evaluate (days)
        n_bins: Number of bins for smoothed curve fitting

    Returns:
        Phase scatter (0-1 scale, where 0 = perfect, 1 = random)
        Values < 0.05 are acceptable for VSX submission
    """
    # Subsample if too many points (spline fitting is slow for large arrays)
    max_points = 5000
    if len(time) > max_points:
        subsample_rate = len(time) // max_points
        time = time[::subsample_rate]
        flux = flux[::subsample_rate]

    # Fold data
    phases = (time % period) / period

    # Sort by phase
    sorted_idx = np.argsort(phases)
    phases_sorted = phases[sorted_idx]
    flux_sorted = flux[sorted_idx]

    # Calculate amplitude
    amplitude = np.max(flux) - np.min(flux)
    if amplitude == 0:
        return 1.0

    # Method 1: Fit spline and calculate residuals
    try:
        # Extend for periodic boundary
        phases_ext = np.concatenate([phases_sorted - 1, phases_sorted, phases_sorted + 1])
        flux_ext = np.concatenate([flux_sorted, flux_sorted, flux_sorted])

        # Fit smooth spline (s controls smoothing)
        # Use larger s for more smoothing
        spline = UnivariateSpline(phases_ext, flux_ext, s=len(flux) * 0.01, k=3)

        # Calculate residuals from smooth curve
        model_flux = spline(phases)
        residuals = flux - model_flux

        # Phase scatter = RMS residual / amplitude
        rms_residual = np.sqrt(np.mean(residuals**2))
        phase_scatter = rms_residual / amplitude

    except Exception as e:
        logger.warning(f"Spline fitting failed: {e}, using binned method")

        # Fallback: binned standard deviation method
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_stds = []

        for i in range(n_bins):
            mask = (phases >= bin_edges[i]) & (phases < bin_edges[i + 1])
            if np.sum(mask) > 2:
                bin_stds.append(np.std(flux[mask]))

        if len(bin_stds) > 0:
            phase_scatter = np.median(bin_stds) / amplitude
        else:
            phase_scatter = 1.0

    return min(phase_scatter, 1.0)


# =============================================================================
# EPOCH DETERMINATION
# =============================================================================

def refine_epoch(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch_type: str = 'max'
) -> float:
    """
    Find time of maximum (or minimum) brightness using spline fitting.

    Args:
        time: Observation times (days, typically BTJD)
        flux: Normalized flux values
        period: Refined period (days)
        epoch_type: 'max' for maximum brightness (DSCT, etc.)
                    'min' for minimum brightness (EBs)

    Returns:
        Epoch (time of extremum in same units as input time)
    """
    # Keep original for epoch calculation at the end
    time_orig = time
    phases_orig = (time_orig % period) / period

    # Subsample if too many points (spline fitting is slow for large arrays)
    max_points = 5000
    if len(time) > max_points:
        subsample_rate = len(time) // max_points
        time = time[::subsample_rate]
        flux = flux[::subsample_rate]

    # Fold data
    phases = (time % period) / period

    # Sort by phase
    sorted_idx = np.argsort(phases)
    phases_sorted = phases[sorted_idx]
    flux_sorted = flux[sorted_idx]

    try:
        # Extend for periodic boundary
        phases_ext = np.concatenate([phases_sorted - 1, phases_sorted, phases_sorted + 1])
        flux_ext = np.concatenate([flux_sorted, flux_sorted, flux_sorted])

        # Fit smooth spline
        spline = UnivariateSpline(phases_ext, flux_ext, s=len(flux) * 0.01, k=3)

        # Find extremum on fine grid
        phase_grid = np.linspace(-0.5, 1.5, 10000)
        flux_grid = spline(phase_grid)

        if epoch_type == 'max':
            # Maximum brightness = maximum flux (assuming normalized flux)
            best_phase = phase_grid[np.argmax(flux_grid)]
        else:
            # Minimum brightness = minimum flux
            best_phase = phase_grid[np.argmin(flux_grid)]

        # Normalize phase to [0, 1)
        best_phase = best_phase % 1.0

    except Exception as e:
        logger.warning(f"Spline epoch fitting failed: {e}, using binned method")

        # Fallback: binned median approach
        n_bins = 50
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_medians = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (phases >= bin_edges[i]) & (phases < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_medians[i] = np.median(flux[mask])
            else:
                bin_medians[i] = np.nan

        valid_mask = ~np.isnan(bin_medians)
        if np.sum(valid_mask) > 0:
            if epoch_type == 'max':
                best_phase = bin_centers[valid_mask][np.argmax(bin_medians[valid_mask])]
            else:
                best_phase = bin_centers[valid_mask][np.argmin(bin_medians[valid_mask])]
        else:
            best_phase = 0.0

    # Find closest observation to this phase and calculate epoch (using original data)
    phase_diffs = np.abs(phases_orig - best_phase)
    # Also check wrap-around
    phase_diffs_wrap = np.minimum(phase_diffs, 1 - phase_diffs)
    closest_idx = np.argmin(phase_diffs_wrap)

    # Epoch is time adjusted to exact phase
    phase_offset = (phases_orig[closest_idx] - best_phase) % 1.0
    if phase_offset > 0.5:
        phase_offset -= 1.0
    epoch = time_orig[closest_idx] - phase_offset * period

    return epoch


# =============================================================================
# FULL REFINEMENT PIPELINE
# =============================================================================

def refine_period_full(
    time: np.ndarray,
    flux: np.ndarray,
    ls_period: float,
    classification: str = 'MISC'
) -> Dict[str, Any]:
    """
    Full period refinement pipeline with multi-stage cascading refinement.

    For long baselines (many cycles), period precision is critical.
    Required precision: dP/P < 1/(2 * N_cycles) for phase alignment.

    Steps:
    1. PDM coarse refinement (+/-0.1% around LS period)
    2. PDM medium refinement (+/-0.01% around stage 1 result)
    3. String Length fine refinement (+/-0.001% around stage 2 result)
       - Only for sinusoidal types: DSCT, HADS, RRC, RRAB, GDOR, SPB
    4. String Length ultra-fine refinement (+/-0.0001% if still poor)
    5. Calculate phase scatter quality metric
    6. Determine epoch via spline fitting

    Args:
        time: Observation times (days)
        flux: Normalized flux values
        ls_period: Initial period from Lomb-Scargle (days)
        classification: Variable type (affects which refinement methods to use)

    Returns:
        Dict with keys:
        - refined_period: Best period (days)
        - period_error: Estimated uncertainty (days)
        - epoch: Time of maximum brightness
        - phase_scatter: Quality metric (< 0.05 for VSX)
        - method: Refinement method used
        - details: Additional diagnostic info
    """
    # Validate inputs
    if time is None or flux is None:
        logger.warning("Missing time or flux data for period refinement")
        return {
            'refined_period': ls_period,
            'period_error': None,
            'epoch': None,
            'phase_scatter': None,
            'method': 'none',
            'details': {'error': 'missing_data'}
        }

    # Convert to numpy arrays if needed
    time = np.asarray(time)
    flux = np.asarray(flux)

    # Remove NaN/inf values
    valid_mask = np.isfinite(time) & np.isfinite(flux)
    time = time[valid_mask]
    flux = flux[valid_mask]

    if len(time) < 100:
        logger.warning(f"Insufficient data points ({len(time)}) for period refinement")
        return {
            'refined_period': ls_period,
            'period_error': None,
            'epoch': None,
            'phase_scatter': None,
            'method': 'none',
            'details': {'error': 'insufficient_data', 'n_points': len(time)}
        }

    # Subsample if too many points (memory/performance optimization)
    if len(time) > 100000:
        subsample_rate = len(time) // 50000
        time = time[::subsample_rate]
        flux = flux[::subsample_rate]
        logger.info(f"Subsampled to {len(time)} points for refinement")

    # Sinusoidal variable types that benefit from string length refinement
    sinusoidal_types = {'DSCT', 'HADS', 'RRC', 'RRAB', 'GDOR', 'SPB', 'BCEP', 'SXPHE'}

    # Calculate baseline and cycles
    baseline = float(np.max(time) - np.min(time))
    n_cycles = baseline / ls_period

    details = {
        'initial_period': ls_period,
        'n_points': len(time),
        'baseline': baseline,
        'n_cycles': n_cycles
    }

    # Determine required precision based on number of cycles
    # For N cycles, need dP/P < 0.5/N for <0.5 phase drift
    required_precision = 0.5 / n_cycles if n_cycles > 0 else 0.001
    logger.info(f"Baseline: {baseline:.1f}d, N_cycles: {n_cycles:.0f}, required dP/P: {required_precision:.2e}")

    try:
        # Stage 1: PDM coarse refinement (+/-0.1%)
        logger.info(f"Stage 1: PDM coarse refinement (search +/-0.1%)")
        pdm1_period, pdm1_theta, pdm1_error = pdm_refine(
            time, flux, ls_period,
            search_fraction=0.001,
            n_trials=2000,
            n_bins=25
        )
        details['pdm1_period'] = pdm1_period
        details['pdm1_theta'] = pdm1_theta

        # Stage 2: PDM medium refinement (+/-0.01%)
        logger.info(f"Stage 2: PDM medium refinement (search +/-0.01%)")
        pdm2_period, pdm2_theta, pdm2_error = pdm_refine(
            time, flux, pdm1_period,
            search_fraction=0.0001,
            n_trials=2000,
            n_bins=30
        )
        details['pdm2_period'] = pdm2_period
        details['pdm2_theta'] = pdm2_theta

        current_period = pdm2_period
        period_error = pdm2_error
        method = 'PDM'

        # Stage 3: String Length fine refinement (for sinusoidal types)
        if classification.upper() in sinusoidal_types:
            logger.info(f"Stage 3: String length fine refinement (search +/-0.001%)")
            sl1_period, sl1_length, sl1_error = string_length_refine(
                time, flux, pdm2_period,
                search_fraction=0.00001,
                n_trials=1000
            )
            details['sl1_period'] = sl1_period
            details['sl1_length'] = sl1_length
            current_period = sl1_period
            period_error = sl1_error
            method = 'PDM+StringLength'

            # Stage 4: String Length ultra-fine refinement if many cycles
            if n_cycles > 10000:
                logger.info(f"Stage 4: String length ultra-fine (search +/-0.0001%)")
                sl2_period, sl2_length, sl2_error = string_length_refine(
                    time, flux, sl1_period,
                    search_fraction=0.000001,
                    n_trials=1000
                )
                details['sl2_period'] = sl2_period
                details['sl2_length'] = sl2_length
                current_period = sl2_period
                period_error = sl2_error
                method = 'PDM+StringLength(ultra)'
        else:
            logger.info(f"Skipping string length for type {classification}")

        refined_period = current_period

        # Stage 5: Calculate phase scatter
        logger.info("Stage 5: Calculating phase scatter")
        phase_scatter = calculate_phase_scatter(time, flux, refined_period)
        details['phase_scatter'] = phase_scatter

        quality = 'excellent' if phase_scatter < 0.02 else \
                  'good' if phase_scatter < 0.05 else \
                  'marginal' if phase_scatter < 0.10 else 'poor'
        details['quality'] = quality

        # Stage 6: Epoch determination
        logger.info("Stage 6: Determining epoch")
        # Use 'max' for pulsators, 'min' for eclipsing binaries
        epoch_type = 'min' if classification.upper() in {'EA', 'EB', 'EW'} else 'max'
        epoch = refine_epoch(time, flux, refined_period, epoch_type)
        details['epoch_type'] = epoch_type

        # Log improvement
        period_change = abs(refined_period - ls_period) / ls_period
        logger.info(f"Refinement complete: P={refined_period:.10f}d (dP/P={period_change:.2e}), scatter={phase_scatter:.4f}, quality={quality}")

        return {
            'refined_period': float(refined_period),
            'period_error': float(period_error) if period_error else None,
            'epoch': float(epoch) if epoch else None,
            'phase_scatter': float(phase_scatter),
            'method': method,
            'details': details
        }

    except Exception as e:
        logger.error(f"Period refinement failed: {e}", exc_info=True)
        return {
            'refined_period': ls_period,
            'period_error': None,
            'epoch': None,
            'phase_scatter': None,
            'method': 'failed',
            'details': {'error': str(e)}
        }

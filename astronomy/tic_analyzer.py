"""
TIC Analyzer - Core analysis engine for TESS Input Catalog targets
Refactored from practical_hunter.py with Supabase integration

Updated: 2025-11-04 - Fixed f-string formatting in plot generation
"""

import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from typing import Optional, Dict, Any, List
import warnings
import logging
from io import BytesIO

warnings.filterwarnings('ignore')

from abastris.utils.config import settings
from abastris.core.discovery_scorer import DiscoveryScorer
from abastris.core.catalog_checker import CatalogChecker
from abastris.core.multi_sector_stitcher import stitch_multi_sector_lightcurve
from abastris.core.period_statistics import comprehensive_period_analysis

logger = logging.getLogger(__name__)


def bootstrap_period_error(
    time: np.ndarray,
    flux: np.ndarray,
    best_period: float,
    n_bootstrap: int = 100
) -> Dict[str, Any]:
    """
    Calculate period uncertainty using bootstrap resampling.

    This provides publication-grade error estimates by:
    1. Resampling the light curve with replacement
    2. Recalculating the period on each resample
    3. Computing statistics on the distribution of periods

    Args:
        time: Array of observation times (days)
        flux: Array of flux measurements
        best_period: The detected best period (days)
        n_bootstrap: Number of bootstrap iterations (default 100)

    Returns:
        Dictionary containing:
        - period: Original best period
        - period_error: Standard deviation (1-sigma uncertainty)
        - confidence_68: 68% confidence interval (1-sigma)
        - confidence_95: 95% confidence interval (2-sigma)
        - confidence_997: 99.7% confidence interval (3-sigma)
        - method: 'bootstrap'
        - iterations: Number of bootstrap iterations

    Reference:
        Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
    """
    logger.info(f"  Calculating bootstrap error estimate (N={n_bootstrap})...")

    periods = []
    n_points = len(time)

    # Create a fine frequency grid centered on the best period
    # This gives better precision for measuring period uncertainty
    best_freq = 1 / best_period
    freq_range = best_freq * 0.2  # Search ±20% of the best frequency
    frequencies = np.linspace(
        max(0.1, best_freq - freq_range),
        best_freq + freq_range,
        50000  # Very fine grid for precision (increased from 10000 for ultra-short periods)
    )

    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n_points, n_points, replace=True)
        boot_time = time[idx]
        boot_flux = flux[idx]

        try:
            # Recalculate period on bootstrap sample
            ls = LombScargle(boot_time, boot_flux)
            powers = ls.power(frequencies)

            # Lower threshold for bootstrap on known periods (was 0.1 for initial detection)
            # For backfilled data, signal is weaker but still valid at 0.03
            # This is acceptable because we already KNOW the period exists
            if np.max(powers) > 0.03:
                best_freq_boot = frequencies[np.argmax(powers)]
                boot_period = 1 / best_freq_boot
                periods.append(boot_period)
        except Exception as e:
            # Skip failed bootstrap iterations
            logger.debug(f"    Bootstrap iteration {i} failed: {e}")
            continue

    # For backfill, accept 30% success rate (was 50%) since we already know the period exists
    # This is more forgiving for weak signals in resampled data
    if len(periods) < n_bootstrap // 3:
        # Not enough successful iterations
        logger.warning(f"  Only {len(periods)}/{n_bootstrap} bootstrap iterations succeeded")
        return {
            'period': best_period,
            'period_error': None,
            'confidence_68': (None, None),
            'confidence_95': (None, None),
            'confidence_997': (None, None),
            'method': 'bootstrap',
            'iterations': len(periods)
        }

    periods = np.array(periods)

    # Calculate statistics
    period_error = np.std(periods)
    conf_68 = np.percentile(periods, [16, 84])  # 1-sigma
    conf_95 = np.percentile(periods, [2.5, 97.5])  # 2-sigma
    conf_997 = np.percentile(periods, [0.15, 99.85])  # 3-sigma

    # Debug: Show period distribution
    logger.debug(f"  Bootstrap periods: min={np.min(periods):.6f}, max={np.max(periods):.6f}, median={np.median(periods):.6f}")
    logger.debug(f"  Bootstrap iterations succeeded: {len(periods)}/{n_bootstrap}")

    # If bootstrap std is zero (all periods identical), use frequency resolution as error estimate
    # This happens for very strong signals where bootstrap can't detect variation
    if period_error == 0 or period_error < 1e-10:
        # Frequency resolution: Δf = (f_max - f_min) / N_freq
        freq_resolution = (frequencies[-1] - frequencies[0]) / len(frequencies)
        # Convert to period uncertainty: Δ P = P^2 * Δf
        period_error = (best_period ** 2) * freq_resolution
        conf_95 = (best_period - 2*period_error, best_period + 2*period_error)
        logger.debug(f"  Bootstrap found no variation - using frequency resolution: Δf={freq_resolution:.6f}")
        logger.info(f"  Period error: ±{period_error:.6f} days (from frequency resolution)")
    else:
        logger.info(f"  Period error: ±{period_error:.6f} days ({period_error*24:.2f} hours)")

    logger.info(f"  95% CI: [{conf_95[0]:.6f}, {conf_95[1]:.6f}] days")

    return {
        'period': best_period,
        'period_error': float(period_error),
        'confidence_68': (float(conf_68[0]), float(conf_68[1])),
        'confidence_95': (float(conf_95[0]), float(conf_95[1])),
        'confidence_997': (float(conf_997[0]), float(conf_997[1])),
        'method': 'bootstrap',
        'iterations': len(periods)
    }


def calculate_amplitude(flux: np.ndarray) -> Dict[str, float]:
    """
    Calculate photometric amplitude in magnitudes.

    Uses robust percentile method to handle outliers and provides
    multiple amplitude metrics for publication.

    Args:
        flux: Normalized flux array (fractional variation, median-centered)

    Returns:
        Dictionary containing:
        - amplitude_mag: Magnitude amplitude (95th - 5th percentile)
        - amplitude_ptp: Peak-to-peak amplitude (fraction of median)
        - amplitude_percent: Peak-to-peak as percentage

    Notes:
        - For normalized flux centered at 0, amplitude measures deviation from median
        - Percentiles are more robust than min/max for noisy data
        - Magnitude amplitude uses Pogson's ratio: Δm = 2.5 * log10(flux_ratio)

    Reference:
        Warner (2006) "A Practical Guide to Lightcurve Photometry and Analysis"
    """
    from scipy.stats import sigmaclip

    # Remove extreme outliers using sigma clipping
    clipped_flux, lower, upper = sigmaclip(flux, low=3, high=3)

    # Calculate percentile-based amplitude (robust to outliers)
    flux_95 = np.percentile(clipped_flux, 95)
    flux_05 = np.percentile(clipped_flux, 5)

    # For normalized flux centered at 0, convert to magnitude amplitude
    # We need to convert fractional variation to magnitude scale
    # Δm = 2.5 * log10((1 + f_high) / (1 + f_low))
    # For small variations: Δm ≈ 2.5 / ln(10) * (f_high - f_low) ≈ 1.086 * Δf
    delta_flux = flux_95 - flux_05

    # More accurate conversion for larger amplitudes
    if abs(flux_95) > 0.1 or abs(flux_05) > 0.1:
        # Use full magnitude formula for large variations
        amplitude_mag = abs(2.5 * np.log10((1 + flux_95) / (1 + flux_05)))
    else:
        # Linear approximation for small variations (< 10%)
        amplitude_mag = abs(1.086 * delta_flux)

    # Peak-to-peak amplitude (max - min of clipped data)
    amplitude_ptp = np.ptp(clipped_flux)
    amplitude_percent = amplitude_ptp * 100

    logger.debug(f"  Amplitude: {amplitude_mag:.4f} mag ({amplitude_percent:.2f}%)")

    return {
        'amplitude_mag': float(amplitude_mag),
        'amplitude_ptp': float(amplitude_ptp),
        'amplitude_percent': float(amplitude_percent)
    }


def classify_period(period_days: Optional[float]) -> Optional[str]:
    """
    Classify variable star period according to scientific standards.

    Classifications:
    - ANOMALOUS: < 0.0104 days (< 15 minutes) - likely instrumental artifact
    - ULTRA_SHORT: 0.0104 - 0.1 days (15 min - 2.4 hours) - Extremely rare, AM CVn candidates
    - SHORT: 0.1 - 1.0 days (2.4 - 24 hours) - Common pulsators, eclipsing binaries
    - MEDIUM: 1.0 - 10.0 days - Cepheids, RR Lyrae, long-period eclipsing binaries
    - LONG: > 10.0 days - Mira variables, long-period variables

    Ultra-short periods are scientifically significant:
    - Less than 0.1% of known variables
    - Potential AM CVn systems (helium mass transfer binaries)
    - Systems approaching stellar contact/merger
    - High priority for spectroscopic follow-up
    - Publishable discoveries

    Args:
        period_days: Period in days

    Returns:
        Classification string or None if period is None
    """
    if period_days is None:
        return None
    elif period_days < 0.0104:  # Less than 15 minutes
        return "ANOMALOUS"
    elif 0.0104 <= period_days < 0.1:  # 15 min to 2.4 hours
        return "ULTRA_SHORT"
    elif 0.1 <= period_days < 1.0:  # 2.4 to 24 hours
        return "SHORT"
    elif 1.0 <= period_days < 10.0:  # 1 to 10 days
        return "MEDIUM"
    else:  # > 10 days
        return "LONG"


class TICAnalyzer:
    """
    Analyzes TESS Input Catalog (TIC) targets for variability
    Based on proven discovery method that found VSX J105515.3-735609
    """

    def __init__(self, check_catalogs: bool = True, use_multi_sector: bool = True,
                 skip_storage: bool = False, max_sectors: int = None,
                 jd_min: float = None, jd_max: float = None,
                 enable_transit_search: bool = False):
        """
        Initialize TIC Analyzer

        Args:
            check_catalogs: Whether to check VSX/SIMBAD for known variables
            use_multi_sector: Whether to use multi-sector stitching (default: True)
                              Combines all available TESS sectors for improved period detection
                              and longer time baselines (typically 3-60x more data points).
            skip_storage: Whether to skip lightcurve/plot storage (default: False)
                          Set to True for enrichment tasks where data is already stored.
            max_sectors: Maximum number of sectors to use (default: None = unlimited)
                         Useful for limiting memory usage during enrichment.
            jd_min: Minimum Julian Date to include (optional). Use to exclude early artifacts.
                    Example: 2459000 to start from late 2020
            jd_max: Maximum Julian Date to include (optional). Use to exclude late artifacts.
                    Example: 2459500 to end in early 2022
            enable_transit_search: Run BLS transit search alongside Lomb-Scargle (default: False).
                                   Phase 1 - validation only. Adds 'transit_search' field to result.
        """
        self.scorer = DiscoveryScorer()
        self.catalog_checker = CatalogChecker() if check_catalogs else None
        self.check_catalogs = check_catalogs
        self.use_multi_sector = use_multi_sector
        self.skip_storage = skip_storage
        self.max_sectors = max_sectors
        self.jd_min = jd_min
        self.jd_max = jd_max
        self.enable_transit_search = enable_transit_search
        self.results = []

    def get_stellar_parameters(self, tic_number: int) -> dict:
        """
        Fetch stellar parameters and verification fields from TIC catalog.

        Retrieves 31 total fields for publication-ready discovery characterization:
        - 11 existing fields: stellar params (Teff, logg, radius, mass), magnitudes, lumclass
        - 20 NEW fields: Gaia/2MASS cross-refs, proper motion, parallax, photometry, etc.

        Args:
            tic_number: TESS Input Catalog identifier

        Returns:
            Dictionary with stellar parameters and verification fields.
            All fields may be None if not available in TIC catalog.

        Reference:
            THE_DATA_BIBLE - Priority 1-6 verification fields
        """
        try:
            from astroquery.mast import Catalogs
        except ImportError:
            logger.error("astroquery required: pip install astroquery")
            return {}

        try:
            # Query MAST TIC catalog
            catalog_data = Catalogs.query_object(
                f"TIC {tic_number}",
                radius=0.001,  # 0.001 degrees = 3.6 arcsec
                catalog="TIC"
            )

            if len(catalog_data) == 0:
                logger.warning(f"No TIC catalog data found for TIC {tic_number}")
                return {}

            # Convert astroquery Row to dict for easier access
            # astroquery Row objects don't have .get() method
            row = catalog_data[0]
            row_dict = {col: row[col] for col in catalog_data.colnames}

            # Helper function to safely get values with None fallback
            def safe_get(key):
                val = row_dict.get(key)
                # Convert masked values, None, and NaN to None
                if val is None or (hasattr(val, 'mask') and val.mask):
                    return None
                # Check for NaN (numpy or Python float NaN)
                try:
                    if np.isnan(val):
                        return None
                except (TypeError, ValueError):
                    # Not a numeric type, return as-is
                    pass
                return val

            # ==================================================================
            # EXISTING FIELDS (11 fields) - Keep these for backwards compatibility
            # ==================================================================
            stellar_params = {
                'teff': safe_get('Teff'),
                'teff_err': safe_get('e_Teff'),
                'logg': safe_get('logg'),
                'logg_err': safe_get('e_logg'),
                'stellar_radius': safe_get('rad'),
                'stellar_radius_err': safe_get('e_rad'),
                'stellar_mass': safe_get('mass'),
                'stellar_mass_err': safe_get('e_mass'),
                'tmag': safe_get('Tmag'),
                'vmag': safe_get('Vmag'),
                'lumclass': safe_get('lumclass'),
            }

            # ==================================================================
            # NEW FIELDS START HERE (20 fields from THE_DATA_BIBLE)
            # ==================================================================

            # PRIORITY 1: Critical cross-references (5 fields)
            stellar_params['gaia_dr3_source_id'] = safe_get('GAIA')
            stellar_params['twomass_id'] = safe_get('TwoMASS')

            # Proper motion (WE SELECT ON THIS: μ > 50 mas/yr)
            pmra = safe_get('pmRA')
            pmdec = safe_get('pmDEC')
            stellar_params['pmra'] = pmra
            stellar_params['pmdec'] = pmdec

            # Compute total proper motion: sqrt(pmRA^2 + pmDEC^2)
            if pmra is not None and pmdec is not None:
                stellar_params['pm_total'] = float(np.sqrt(pmra**2 + pmdec**2))
            else:
                stellar_params['pm_total'] = None

            # PRIORITY 2: Parallax and distance (3 fields)
            plx = safe_get('plx')
            stellar_params['parallax'] = plx
            stellar_params['parallax_error'] = safe_get('e_plx')

            # Compute distance in parsecs: d = 1000 / parallax (mas)
            if plx and plx > 0:
                stellar_params['distance'] = float(1000.0 / plx)
            else:
                stellar_params['distance'] = None

            # PRIORITY 3: Gaia photometry (4 of 8 photometry fields)
            bpmag = safe_get('BPmag')
            rpmag = safe_get('RPmag')
            stellar_params['gmag'] = safe_get('GAIAmag')
            stellar_params['bpmag'] = bpmag
            stellar_params['rpmag'] = rpmag

            # Compute BP-RP color index
            if bpmag is not None and rpmag is not None:
                stellar_params['bp_minus_rp'] = float(bpmag - rpmag)
            else:
                stellar_params['bp_minus_rp'] = None

            # 2MASS photometry (3 more photometry fields)
            stellar_params['jmag'] = safe_get('Jmag')
            stellar_params['hmag'] = safe_get('Hmag')
            stellar_params['kmag'] = safe_get('Kmag')

            # Johnson B magnitude (last photometry field)
            bmag = safe_get('Bmag')
            stellar_params['bmag'] = bmag

            # PRIORITY 4: B-V color index (1 field)
            vmag = safe_get('Vmag')
            if bmag is not None and vmag is not None:
                stellar_params['b_minus_v'] = float(bmag - vmag)
            else:
                stellar_params['b_minus_v'] = None

            # PRIORITY 5: Coordinate errors (2 fields)
            # Convert from milliarcseconds to arcseconds
            ra_err_mas = safe_get('e_RAJ2000')
            dec_err_mas = safe_get('e_DECJ2000')
            stellar_params['ra_error'] = float(ra_err_mas / 1000.0) if ra_err_mas else None
            stellar_params['dec_error'] = float(dec_err_mas / 1000.0) if dec_err_mas else None

            # PRIORITY 6: Data quality (1 field)
            stellar_params['contamination_ratio'] = safe_get('contratio')

            # Note: PRIORITY 7 (epoch) and PRIORITY 8 (ra_hms, dec_dms)
            # are computed in analyze() method and router code, not here

            # Convert NaN values to None for JSON serialization
            for key, value in stellar_params.items():
                if value is not None:
                    try:
                        import math
                        if isinstance(value, float) and math.isnan(value):
                            stellar_params[key] = None
                    except (TypeError, ValueError):
                        pass  # Not a numeric type, leave as is

            logger.debug(f"Fetched stellar parameters for TIC {tic_number}")
            return stellar_params

        except Exception as e:
            logger.error(f"Error fetching stellar parameters for TIC {tic_number}: {e}")
            return {}

    def analyze_single_tic(
        self,
        tic_number: int,
        cadence: str = "short"
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a single TIC for variability

        Args:
            tic_number: TESS Input Catalog identifier
            cadence: Observation cadence ('short' for 2-min, 'long' for 30-min)

        Returns:
            Dictionary with analysis results, or None if no data available
        """
        logger.info(f"🔭 Analyzing TIC {tic_number}")

        try:
            # Use multi-sector stitching if enabled
            if self.use_multi_sector:
                logger.info(f"  Using multi-sector stitching...")
                stitch_result = stitch_multi_sector_lightcurve(
                    tic_number,
                    max_sectors=self.max_sectors,
                    jd_min=self.jd_min,
                    jd_max=self.jd_max
                )

                if stitch_result is None:
                    logger.warning(f"  No data found for TIC {tic_number}")
                    return None

                lc_clean = stitch_result['stitched_lc']
                sectors = stitch_result['sectors_used']
                n_sectors = stitch_result['n_sectors']
                total_baseline = stitch_result['total_baseline_days']
                n_gaps = stitch_result['n_gaps']

                # Quality filtering metadata
                sectors_rejected = stitch_result.get('sectors_rejected', [])
                sectors_rejected_count = stitch_result.get('sectors_rejected_count', 0)
                quality_report = stitch_result.get('quality_report', {})
                has_contaminated_sectors = stitch_result.get('has_contaminated_sectors', False)

                logger.info(f"  ✅ Using {n_sectors} sector(s): {', '.join(map(str, sectors))}")
                logger.info(f"  ✅ Total baseline: {total_baseline:.1f} days")
                if sectors_rejected:
                    logger.info(f"  🚫 Rejected {len(sectors_rejected)} sector(s): {sectors_rejected}")

            else:
                # Single-sector mode (original behavior)
                search = lk.search_lightcurve(
                    f"TIC {tic_number}",
                    mission='TESS',
                    cadence=cadence
                )

                if len(search) == 0:
                    logger.warning(f"  No data found for TIC {tic_number}")
                    return None

                # Download the first available sector
                lc = search[0].download()
                if lc is None:
                    logger.warning(f"  Failed to download data for TIC {tic_number}")
                    return None

                # Clean the light curve
                lc_clean = lc.remove_nans().remove_outliers(sigma=5)

                # Get sector info
                sectors = [str(search[0].mission[0]) if search[0].mission else 'Unknown']
                n_sectors = 1
                baseline = lc_clean.time.ptp()
                # Convert TimeDelta to float (days)
                total_baseline = float(baseline.value) if hasattr(baseline, 'value') else float(baseline)
                n_gaps = 0

                # No quality filtering in single-sector mode
                sectors_rejected = []
                sectors_rejected_count = 0
                quality_report = {}
                has_contaminated_sectors = False

            if len(lc_clean.time) < 100:
                logger.warning(f"  Insufficient data points for TIC {tic_number}")
                return None

            # Extract time and flux
            time = lc_clean.time.value
            # Handle both regular arrays and MaskedNDArrays
            if hasattr(lc_clean.flux, 'value'):
                flux = lc_clean.flux.value
            else:
                flux = np.asarray(lc_clean.flux)

            # Remove any remaining NaNs or masked values
            mask = np.isfinite(flux) & np.isfinite(time)
            time = time[mask]
            flux = flux[mask]

            # CRITICAL: Always normalize the final light curve to fractional flux
            # This must happen AFTER stitching and offset correction (if multi-sector)
            # Normalization centers flux at 0 with units of fractional variation
            flux_median = np.median(flux)
            flux = (flux - flux_median) / flux_median

            # Calculate variability (standard deviation of fractional flux)
            # Use sigma clipping to remove outliers before calculating variability
            # This matches the approach used in calculate_amplitude()
            from scipy.stats import sigmaclip
            clipped_flux, _, _ = sigmaclip(flux, low=3, high=3)
            variability = np.std(clipped_flux) if len(clipped_flux) > 0 else np.std(flux)

            # Sanity check: variability should be between 0 and ~2.0 (0-200%)
            # Values > 5.0 indicate corrupted data or calculation error
            if variability > 5.0:
                logger.warning(f"  ⚠️  EXTREME variability detected: {variability:.2e} ({variability*100:.1f}%)")
                logger.warning(f"     This likely indicates corrupted light curve data")
                logger.warning(f"     Setting variability to 0.0 to prevent database corruption")
                variability = 0.0

            # Calculate amplitude metrics for publication readiness
            amplitude_results = calculate_amplitude(flux)

            # Period search if variable enough
            period = None
            fap = 1.0
            error_results = {}  # Initialize error results

            if variability > settings.variability_threshold:
                logger.info(f"  Variable detected: {variability:.6f}")

                # Lomb-Scargle periodogram
                ls = LombScargle(time, flux)

                # Search for short periods (like the proven discovery)
                frequencies = np.linspace(2, 50, 5000)
                powers = ls.power(frequencies)

                if np.max(powers) > 0.1:
                    best_freq = frequencies[np.argmax(powers)]
                    period = 1 / best_freq
                    fap = ls.false_alarm_probability(np.max(powers))

                    logger.info(f"  Period found: {period:.5f} days ({period*24:.2f} hours)")
                    logger.info(f"  FAP: {fap:.2e}")

                    # Calculate period uncertainty using bootstrap
                    logger.info(f"  Calculating period uncertainty via bootstrap (N=100)...")
                    error_results = bootstrap_period_error(time, flux, period, n_bootstrap=100)

                    if error_results['period_error'] is not None:
                        logger.info(f"  Period error: ±{error_results['period_error']:.5f} days")
                        logger.info(f"  95% CI: [{error_results['confidence_95'][0]:.5f}, {error_results['confidence_95'][1]:.5f}] days")
                    else:
                        logger.warning(f"  Bootstrap failed - could not estimate period uncertainty")

                    # Comprehensive statistical analysis (aliases, window function, secondary periods)
                    stats_results = comprehensive_period_analysis(time, flux, period, np.max(powers))
                else:
                    stats_results = None

            # ==================================================================
            # OPTIONAL: BLS Transit Search (Phase 1 - Validation Only)
            # ==================================================================
            transit_result = None
            if self.enable_transit_search:
                try:
                    from abastris.core.transit_detector import TransitDetector

                    logger.info("  Running BLS transit search (experimental)...")
                    detector = TransitDetector()
                    transit_result = detector.search(time, flux)

                    if transit_result.is_transit_candidate:
                        logger.info(f"  Transit candidate: P={transit_result.bls_period:.6f}d, "
                                   f"SNR={transit_result.bls_snr:.1f}")
                except Exception as e:
                    logger.warning(f"  BLS search failed: {e}")
                    transit_result = None

            # Extract coordinates from light curve metadata (always)
            ra = lc_clean.meta.get('RA_OBJ') or lc_clean.meta.get('RA')
            dec = lc_clean.meta.get('DEC_OBJ') or lc_clean.meta.get('DEC')

            # ==================================================================
            # PRIORITY 7: Calculate epoch of maximum light (BTJD)
            # ==================================================================
            epoch = None
            if period is not None and period > 0:
                try:
                    # Fold light curve at detected period
                    phase_folded = lc_clean.fold(period=period)

                    # Find time of maximum flux
                    if hasattr(phase_folded.flux, 'value'):
                        flux_values = phase_folded.flux.value
                    else:
                        flux_values = phase_folded.flux

                    max_flux_idx = np.argmax(flux_values)

                    # Get corresponding time (BTJD)
                    if hasattr(lc_clean.time, 'value'):
                        epoch = float(lc_clean.time[max_flux_idx].value)
                    else:
                        epoch = float(lc_clean.time[max_flux_idx])

                    logger.debug(f"  Epoch of maximum: {epoch:.6f} BTJD")

                except Exception as e:
                    logger.warning(f"  Could not calculate epoch: {e}")
                    epoch = None

            # ==================================================================
            # Fetch stellar parameters + verification fields from TIC catalog
            # ==================================================================
            stellar_params = self.get_stellar_parameters(tic_number)

            # ==================================================================
            # PRIORITY 8: Convert coordinates to sexagesimal format
            # ==================================================================
            from api.utils.coordinate_formatter import decimal_to_hms, decimal_to_dms
            ra_hms = decimal_to_hms(ra) if ra is not None else None
            dec_dms = decimal_to_dms(dec) if dec is not None else None

            # Classify period
            period_classification = classify_period(period)

            # Prepare result
            result = {
                'tic': tic_number,
                'variability': float(variability),
                'period': float(period) if period else None,
                'period_hours': float(period * 24) if period else None,
                'period_classification': period_classification,
                'fap': float(fap),
                'n_points': len(time),
                'sector': ', '.join([f'TESS Sector {s}' for s in sectors]) if self.use_multi_sector else f'TESS Sector {sectors[0]}',
                'sectors': sectors,  # List of sector numbers
                'n_sectors': n_sectors,
                'total_baseline_days': float(total_baseline),
                'n_gaps': n_gaps,
                # Quality filtering metadata
                'sectors_rejected': sectors_rejected,
                'sectors_rejected_count': sectors_rejected_count,
                'quality_report': quality_report,
                'has_contaminated_sectors': has_contaminated_sectors,
                'ra': float(ra) if ra is not None else None,
                'dec': float(dec) if dec is not None else None,
                # Amplitude measurements for publication
                'amplitude_mag': amplitude_results['amplitude_mag'],
                'amplitude_ptp': amplitude_results['amplitude_ptp'],
                'amplitude_percent': amplitude_results['amplitude_percent'],
                # Error estimation fields
                'period_error': error_results.get('period_error') if period else None,
                'confidence_95_lower': error_results.get('confidence_95', (None, None))[0] if period else None,
                'confidence_95_upper': error_results.get('confidence_95', (None, None))[1] if period else None,
                'error_method': error_results.get('method') if period else None,
                'error_iterations': error_results.get('iterations') if period else None,
                # Statistical analysis fields
                'n_aliases': stats_results['n_aliases'] if (period and stats_results) else None,
                'has_harmonics': stats_results['has_harmonics'] if (period and stats_results) else None,
                'has_subharmonics': stats_results['has_subharmonics'] if (period and stats_results) else None,
                'n_secondary_periods': stats_results['n_secondary_periods'] if (period and stats_results) else None,
                'period_confidence': stats_results['period_confidence'] if (period and stats_results) else None,
                'aliases': stats_results['aliases_detected'] if (period and stats_results) else None,
                'secondary_periods': stats_results['secondary_periods'] if (period and stats_results) else None,
                'window_function': stats_results['window_function'] if (period and stats_results) else None,
                # BLS Transit Search (Phase 1 - validation only, opt-in)
                'transit_search': transit_result.to_dict() if transit_result else None,
                # Epoch of maximum light (PRIORITY 7)
                'epoch': epoch,
                # Stellar parameters + verification fields (PRIORITY 1-6)
                **stellar_params,  # Unpacks all 31 fields from get_stellar_parameters()
                # PRIORITY 8: Sexagesimal coordinates
                'ra_hms': ra_hms,
                'dec_dms': dec_dms,
                # Light curve data (internal use)
                'lc': lc_clean,
                'time': time,
                'flux': flux
            }

            # Highlight ultra-short period discoveries
            if period_classification == 'ULTRA_SHORT':
                logger.warning(f"  🌟⚡ ULTRA-SHORT PERIOD DISCOVERY! 🌟⚡")
                logger.warning(f"     Period: {period*24:.2f} hours ({period:.5f} days)")
                logger.warning(f"     This is EXTREMELY RARE and scientifically valuable!")
                logger.warning(f"     Potential AM CVn candidate - HIGH PRIORITY for follow-up!")
            elif period_classification == 'ANOMALOUS':
                logger.warning(f"  ⚠️  ANOMALOUS period < 15 minutes - likely instrumental artifact")

            # Check catalogs for known variables (using coordinate-based search)
            if self.check_catalogs and self.catalog_checker:

                # Use coordinate-based checking for more accurate results
                if ra is not None and dec is not None:
                    logger.info(f"  Using coordinate-based catalog checking (RA={ra:.6f}°, Dec={dec:.6f}°)")
                    catalog_result = self.catalog_checker.check_tic_with_coordinates(
                        tic_number,
                        ra=float(ra),
                        dec=float(dec),
                        search_radius=5.0  # 5 arcseconds
                    )
                else:
                    logger.warning(f"  Coordinates not available, using name-only search")
                    catalog_result = self.catalog_checker.check_tic(tic_number)

                result['catalog_check'] = catalog_result

                # ==============================================================
                # TWO-TIER DISCOVERY CLASSIFICATION (v2.0)
                # ==============================================================
                from abastris.core.discovery_scorer import classify_discovery_tier, DiscoveryScorerV2

                # Determine discovery tier
                is_known_variable = catalog_result['is_known_variable']
                discovery_tier = classify_discovery_tier(
                    period=period,
                    period_fap=fap,
                    variability=variability,
                    is_known_variable=is_known_variable
                )

                # Set new classification fields
                # IMPORTANT: Periodic discoveries are ALSO variable detections
                # A star that is periodic is by definition variable
                result['discovery_tier'] = discovery_tier
                result['is_periodic_discovery'] = (discovery_tier == 'periodic_discovery')
                result['is_variable_detection'] = (
                    discovery_tier == 'periodic_discovery' or
                    discovery_tier == 'variable_detection'
                )

                # Calculate score using new rubric v2.0
                scorer = DiscoveryScorerV2()
                score_result = scorer.score_discovery(result)
                result['score'] = score_result['score']
                result['score_breakdown'] = score_result['breakdown']
                result['tier_label'] = score_result['tier_label']

                logger.info(f"  Discovery Classification: {discovery_tier}")
                logger.info(f"  Score: {score_result['score']}/100 ({score_result['tier_label']})")

                # Auto-trigger Vizier validation for periodic discoveries
                if result['is_periodic_discovery'] and ra is not None and dec is not None:
                    try:
                        from abastris.core.vizier_checker import validate_tic_for_vsx
                        from datetime import datetime

                        logger.info(f"  Running Vizier validation for periodic discovery...")

                        vizier_result = validate_tic_for_vsx(
                            tic_number,
                            float(ra),
                            float(dec),
                            search_radius_arcsec=30.0
                        )

                        # Populate Vizier fields
                        result['vizier_recommendation'] = vizier_result.recommendation
                        result['vizier_vsx_match'] = vizier_result.vsx_match
                        result['vizier_total_matches'] = vizier_result.total_matches
                        result['vizier_catalog_matches'] = list(vizier_result.matches.keys())
                        result['vizier_validated_at'] = datetime.utcnow().isoformat()
                        result['discovery_category'] = vizier_result.discovery_category
                        result['simbad_coord_match'] = vizier_result.simbad_coord_match

                        logger.info(f"  ✅ Vizier recommendation: {vizier_result.recommendation}")

                        # Log True Discovery status
                        if (result['is_periodic_discovery'] and
                            result['is_variable_detection'] and
                            vizier_result.recommendation == 'new_submission'):
                            logger.info(f"  🌟 TRUE DISCOVERY - New variable star for VSX submission!")

                    except Exception as e:
                        logger.error(f"  ⚠️  Vizier validation failed: {e}")
                        # Don't fail the analysis - just skip Vizier validation
                        result['vizier_recommendation'] = None
                        result['vizier_vsx_match'] = None
                        result['vizier_total_matches'] = None
                        result['vizier_catalog_matches'] = None
                        result['vizier_validated_at'] = None
                        result['discovery_category'] = None
                        result['simbad_coord_match'] = None

                if catalog_result['is_known_variable']:
                    result['known_as'] = catalog_result['known_names']
                    result['known_type'] = catalog_result['variable_type']
            else:
                result['catalog_check'] = None
                # No catalog check - cannot classify discovery tier
                result['discovery_tier'] = None
                result['is_periodic_discovery'] = None
                result['is_variable_detection'] = None
                result['score'] = None
                result['score_breakdown'] = None
                result['tier_label'] = None

            # ================================================================
            # STORE LIGHTCURVES FOR PERIODIC DISCOVERIES
            # ================================================================
            # If this is a periodic discovery (is_periodic_discovery=True) and we have
            # multi-sector data, store individual sector lightcurves and stitched
            # lightcurve to Supabase Storage for:
            #   1. Contamination analysis (VSX Feature #2)
            #   2. Downloadable FITS files for publications
            #   3. Offline analysis
            # This applies to both Pristine (new to VSX) and Cataloged (already in VSX)
            # discoveries for data quality and credibility purposes.
            # Skip storage if skip_storage=True (e.g., during enrichment when data already exists)
            if (result.get('is_periodic_discovery') and
                self.use_multi_sector and
                'stitch_result' in locals() and
                stitch_result is not None and
                not self.skip_storage):

                try:
                    from abastris.db.lightcurve_storage import store_all_lightcurves_for_tic

                    logger.info(f"  💾 Storing lightcurves for discovery...")

                    storage_result = store_all_lightcurves_for_tic(
                        tic_number=tic_number,
                        sector_lightcurves=stitch_result.get('sector_lightcurves', []),
                        stitched_lightcurve=stitch_result.get('stitched_lc'),
                        sectors_used=stitch_result.get('sectors_used', []),
                        user_id=None  # System upload
                    )

                    if storage_result.get('success'):
                        result['lightcurves_stored'] = True
                        result['lightcurves_storage_mb'] = storage_result.get('total_bytes', 0) / 1024 / 1024
                        logger.info(f"  ✅ Stored {len(storage_result['sector_files'])} sector files")

                        # Run contamination analysis
                        try:
                            from abastris.db.queries import analyze_sector_contamination
                            contamination = analyze_sector_contamination(tic_number)

                            if contamination.get('has_data'):
                                result['has_contaminated_sectors'] = contamination.get('has_contamination')
                                result['recommended_sectors'] = contamination.get('recommendation')
                                logger.info(f"  📊 Contamination analysis: {contamination.get('recommendation')}")
                        except Exception as e:
                            logger.warning(f"  ⚠️  Contamination analysis failed: {e}")
                    else:
                        logger.warning(f"  ⚠️  Lightcurve storage had issues")

                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to store lightcurves: {e}")
                    # Don't fail the analysis, just skip storage

            # ================================================================
            # AUTO-GENERATE PLOTS FOR PERIODIC DISCOVERIES
            # ================================================================
            # Automatically generate and upload discovery plot for periodic discoveries
            # (both Pristine and Cataloged) for VSX submission and public credibility.
            # Skip plot generation if skip_storage=True (plot already exists)
            if result.get('is_periodic_discovery') and not self.skip_storage:
                try:
                    from abastris.core.plot_generator import generate_plot_for_tic

                    logger.info(f"  📊 Generating discovery plot...")

                    plot_result = generate_plot_for_tic(
                        tic_number=tic_number,
                        user_id='public',
                        upload_to_storage=True,
                        save_local=False
                    )

                    if plot_result and plot_result.get('plot_url'):
                        result['plot_url'] = plot_result['plot_url']
                        logger.info(f"  ✅ Plot generated: {plot_result['plot_url']}")
                    else:
                        logger.warning(f"  ⚠️  Plot generation returned no URL")

                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to generate plot: {e}")
                    # Don't fail the analysis, just skip plot generation

            return result

        except Exception as e:
            import traceback
            logger.error(f"  Error analyzing TIC {tic_number}: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return None

    def analyze_batch(
        self,
        tic_list: List[int],
        max_targets: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze multiple TICs in batch

        Args:
            tic_list: List of TIC numbers to analyze
            max_targets: Maximum number to process (None for all)
            progress_callback: Function to call with progress updates

        Returns:
            Dictionary with 'discoveries', 'candidates', and 'all_results'
        """
        if max_targets:
            tic_list = tic_list[:max_targets]

        logger.info(f"🚀 Starting batch analysis of {len(tic_list)} TICs")

        discoveries = []
        candidates = []
        all_results = []

        for i, tic in enumerate(tic_list):
            # Progress update
            if progress_callback:
                progress_callback(i + 1, len(tic_list), tic)

            result = self.analyze_single_tic(tic)

            if result:
                all_results.append(result)

                # Classify based on score (handle None for non-catalog-checked analyses)
                score = result.get('score')
                if score is not None and score >= 50:
                    discoveries.append(result)
                    logger.info(f"🌟 DISCOVERY! TIC {tic}: Score={score}")
                elif score is not None and score >= 30:
                    candidates.append(result)
                    logger.info(f"⭐ Candidate TIC {tic}: Score={score}")
                elif result['variability'] > 0.002:
                    logger.info(f"✓ Variable TIC {tic}: Var={result['variability']:.5f}")

        logger.info(f"\n📊 Batch analysis complete:")
        logger.info(f"  Analyzed: {len(all_results)} TICs with data")
        logger.info(f"  Discoveries: {len(discoveries)}")
        logger.info(f"  Candidates: {len(candidates)}")

        return {
            'discoveries': discoveries,
            'candidates': candidates,
            'all_results': all_results,
            'total_processed': len(tic_list),
            'total_with_data': len(all_results)
        }

    def generate_discovery_plot(
        self,
        result: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> bytes:
        """
        Generate a 4-panel discovery plot

        Args:
            result: Analysis result dictionary
            save_path: Optional path to save plot

        Returns:
            PNG image binary data
        """
        lc = result['lc']
        tic = result['tic']
        score = result['score']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Full light curve
        axes[0, 0].scatter(lc.time.value, lc.flux.value, s=0.5, alpha=0.5, c='black')
        axes[0, 0].set_xlabel('Time (BTJD)', fontsize=10)
        axes[0, 0].set_ylabel('Flux', fontsize=10)
        axes[0, 0].set_title(f"TIC {tic} - Score: {score}/100", fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # Panel 2: Periodogram
        time = result['time']
        flux = result['flux']
        ls = LombScargle(time, flux)

        frequencies = np.linspace(0.5, 50, 5000)
        powers = ls.power(frequencies)

        axes[0, 1].plot(1/frequencies, powers, 'k-', linewidth=0.5)
        if result['period']:
            axes[0, 1].axvline(
                result['period'],
                color='red',
                linestyle='--',
                linewidth=2,
                label=f"P={result['period']:.5f}d"
            )
            axes[0, 1].legend(fontsize=9)
        axes[0, 1].set_xlabel('Period (days)', fontsize=10)
        axes[0, 1].set_ylabel('Power', fontsize=10)
        axes[0, 1].set_xlim(0, 2)
        axes[0, 1].set_title('Lomb-Scargle Periodogram', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)

        # Panel 3: Phase-folded light curve
        if result['period']:
            folded = lc.fold(period=result['period'])
            axes[1, 0].scatter(folded.time.value, folded.flux.value, s=0.5, alpha=0.3, c='blue')
            axes[1, 0].set_xlabel('Phase', fontsize=10)
            axes[1, 0].set_ylabel('Flux', fontsize=10)
            axes[1, 0].set_title(f"Folded at P={result['period']:.5f} days", fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No significant period', ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')

        # Panel 4: Info panel
        axes[1, 1].axis('off')

        # Format period info
        period_str = f"{result['period']:.5f}" if result['period'] else 'None'
        hours_str = f"Hours: {result['period']*24:.2f}" if result['period'] else ""
        fap_str = f"{result['fap']:.2e}" if (result.get('fap') is not None and result['period']) else 'N/A'

        info_text = f"""
POTENTIAL DISCOVERY

TIC: {tic}
Variability: {result['variability']:.6f}
Period: {period_str} days
{hours_str}
FAP: {fap_str}
Data Points: {result['n_points']}
Sector: {result['sector']}

Score: {score}/100

Similar to discovery:
VSX J105515.3-735609

NEXT STEPS:
1. Check VSX database
2. Check SIMBAD for V* flag
3. If not listed → SUBMIT!
        """
        axes[1, 1].text(0.05, 0.5, info_text.strip(), fontsize=10, fontfamily='monospace',
                       transform=axes[1, 1].transAxes, va='center')

        plt.suptitle(f'AbAstris Discovery: TIC {tic}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save to bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plot_data = buf.read()
        plt.close()

        # Optionally save to file
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(plot_data)
            logger.info(f"✅ Plot saved to {save_path}")

        return plot_data

    def get_strategic_tics(self, limit: int = 100) -> List[int]:
        """
        Get strategic TIC targets near the original discovery
        Based on TIC 277539431 (VSX J105515.3-735609)
        """
        # TICs near the original discovery
        base_tic = 277529431
        tics = [base_tic + (i * 100) for i in range(limit)]

        return tics


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_analyze(tic_number: int) -> Optional[Dict[str, Any]]:
    """Quick analysis of a single TIC"""
    analyzer = TICAnalyzer()
    return analyzer.analyze_single_tic(tic_number)


def batch_analyze(tic_list: List[int], max_targets: Optional[int] = None):
    """Batch analysis with progress reporting"""
    analyzer = TICAnalyzer()

    def progress_callback(current, total, tic):
        print(f"\r  Progress: {current}/{total} - Analyzing TIC {tic}...", end='')

    results = analyzer.analyze_batch(tic_list, max_targets, progress_callback)
    print()  # New line after progress
    return results

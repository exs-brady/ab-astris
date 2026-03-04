"""
Multi-Sector Light Curve Stitching

Combines data from multiple observing periods (sectors/quarters/campaigns) to:
1. Increase time baseline for better period precision
2. Improve phase coverage
3. Enable detection of long-period variables
4. Reduce period aliases

Supports missions:
- TESS: Sectors (27-day observing periods)
- Kepler: Quarters (90-day observing periods)
- K2: Campaigns (80-day observing periods)

Key Challenges Addressed:
- Different flux normalizations between observation units
- Systematic offsets and zero-point differences
- Gaps between observations
- Quality flag differences
"""

import numpy as np
import gc
import lightkurve as lk
from typing import List, Dict, Any, Optional, Tuple
import logging
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from astropy.time import Time

# Import mission configuration for multi-mission support
from abastris.core.mission_config import (
    get_mission_config,
    format_target_name,
    MISSION_CONFIGS
)

# Import sector quality metrics for adaptive filtering
from abastris.core.sector_metrics import get_adaptive_sigma

logger = logging.getLogger(__name__)


def calculate_sector_amplitude(lc) -> Dict[str, float]:
    """
    Calculate amplitude metrics for a single sector lightcurve.

    Used for quality filtering before multi-sector combination.

    Args:
        lc: lightkurve.LightCurve object (single sector)

    Returns:
        Dictionary with amplitude metrics:
        - amplitude_ptp: Peak-to-peak amplitude (flux units)
        - amplitude_percent: Amplitude as percentage of median
        - std_flux: Standard deviation of flux
        - median_flux: Median flux value
        - n_points: Number of data points
    """
    from scipy.stats import sigmaclip

    # Get flux values
    if hasattr(lc.flux, 'value'):
        flux = lc.flux.value.copy()
    else:
        flux = np.asarray(lc.flux).copy()

    # Remove extreme outliers for robust amplitude calculation
    clipped_flux, lower, upper = sigmaclip(flux, low=3, high=3)

    # Calculate metrics
    median_flux = np.median(clipped_flux)
    std_flux = np.std(clipped_flux)

    # Peak-to-peak amplitude (robust: 95th - 5th percentile)
    flux_95 = np.percentile(clipped_flux, 95)
    flux_05 = np.percentile(clipped_flux, 5)
    amplitude_ptp = flux_95 - flux_05

    # Amplitude as percentage of median
    amplitude_percent = (amplitude_ptp / median_flux) * 100 if median_flux > 0 else 0

    return {
        'amplitude_ptp': float(amplitude_ptp),
        'amplitude_percent': float(amplitude_percent),
        'std_flux': float(std_flux),
        'median_flux': float(median_flux),
        'n_points': len(lc.time)
    }


def filter_contaminated_sectors(
    processed_sectors: List[Dict],
    amplitude_threshold: float = 2.5,
    min_points: int = 100
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Filter contaminated sectors based on amplitude analysis.

    A sector is considered contaminated if:
    1. Its amplitude is < median_amplitude / amplitude_threshold (default: < 40% of median)
    2. It has fewer than min_points data points

    Args:
        processed_sectors: List of sector metadata dicts with 'amplitude' key
        amplitude_threshold: Reject sectors with amplitude < median/threshold
        min_points: Minimum data points required

    Returns:
        Tuple of (clean_sectors, rejected_sectors, quality_report)
    """
    if len(processed_sectors) <= 1:
        # Single sector - no filtering possible
        return processed_sectors, [], {
            'filtering_applied': False,
            'reason': 'single_sector'
        }

    # Extract amplitudes
    amplitudes = [s['amplitude']['amplitude_ptp'] for s in processed_sectors]
    median_amplitude = np.median(amplitudes)

    # Calculate threshold
    rejection_threshold = median_amplitude / amplitude_threshold

    clean_sectors = []
    rejected_sectors = []
    rejection_reasons = {}

    for sector_data in processed_sectors:
        sector_num = sector_data['sector']
        amp = sector_data['amplitude']['amplitude_ptp']
        n_points = sector_data['amplitude']['n_points']

        reasons = []

        # Check amplitude
        if amp < rejection_threshold:
            reasons.append(f"amplitude_too_low ({amp:.6f} < threshold {rejection_threshold:.6f})")

        # Check data points
        if n_points < min_points:
            reasons.append(f"insufficient_points ({n_points} < {min_points})")

        if reasons:
            rejected_sectors.append(sector_data)
            rejection_reasons[sector_num] = '; '.join(reasons)
        else:
            clean_sectors.append(sector_data)

    # Safety check: don't reject ALL sectors
    if len(clean_sectors) == 0 and len(rejected_sectors) > 0:
        logger.warning("  ⚠️  All sectors would be rejected, keeping sector with highest amplitude")
        best_sector = max(rejected_sectors, key=lambda s: s['amplitude']['amplitude_ptp'])
        clean_sectors.append(best_sector)
        rejected_sectors.remove(best_sector)
        del rejection_reasons[best_sector['sector']]

    # Calculate amplitude ratio for quality report
    if len(amplitudes) > 1:
        amplitude_ratio = max(amplitudes) / min(amplitudes) if min(amplitudes) > 0 else float('inf')
    else:
        amplitude_ratio = 1.0

    quality_report = {
        'filtering_applied': True,
        'total_sectors': len(processed_sectors),
        'clean_sectors_count': len(clean_sectors),
        'rejected_sectors_count': len(rejected_sectors),
        'clean_sector_numbers': [s['sector'] for s in clean_sectors],
        'rejected_sector_numbers': [s['sector'] for s in rejected_sectors],
        'rejection_reasons': rejection_reasons,
        'median_amplitude': float(median_amplitude),
        'amplitude_threshold': float(rejection_threshold),
        'amplitude_ratio': float(amplitude_ratio),
        'has_contamination': len(rejected_sectors) > 0
    }

    if len(rejected_sectors) > 0:
        logger.info(f"  🔍 Quality filtering: {len(rejected_sectors)} contaminated sector(s) rejected")
        for sector_num, reason in rejection_reasons.items():
            logger.info(f"      Sector {sector_num}: {reason}")

    return clean_sectors, rejected_sectors, quality_report


def detect_time_format(time_values: np.ndarray, mission: str = 'TESS') -> str:
    """
    Detect the time format based on the magnitude of time values.

    Each mission uses different time formats:
    - TESS: BTJD (BJD - 2457000.0), values ~1000-3000
    - Kepler/K2: BKJD (BJD - 2454833.0), values ~100-1600
    - BJD: Full Barycentric Julian Date, values ~2458000-2460000

    Args:
        time_values: Array of time values
        mission: Mission name ('TESS', 'Kepler', 'K2')

    Returns:
        Detected format: 'btjd', 'bkjd', 'bjd', or 'jd'
    """
    median_time = np.median(time_values)

    if median_time < 10000:
        # Mission-specific offset format
        if mission == 'TESS':
            return 'btjd'  # BTJD (TJD offset from 2457000)
        else:
            return 'bkjd'  # BKJD (Kepler offset from 2454833)
    elif median_time > 2400000:
        return 'bjd'  # Full BJD or JD
    else:
        logger.warning(f"Unusual time format detected (median={median_time}), assuming mission-specific")
        return 'btjd' if mission == 'TESS' else 'bkjd'


def convert_to_mission_time(time_values: np.ndarray, current_format: str, mission: str = 'TESS') -> np.ndarray:
    """
    Convert time values to mission-specific format (BTJD or BKJD).

    Time epochs:
    - TESS: BTJD = BJD - 2457000.0
    - Kepler/K2: BKJD = BJD - 2454833.0

    Args:
        time_values: Array of time values
        current_format: Current time format ('btjd', 'bkjd', 'bjd', 'jd')
        mission: Target mission ('TESS', 'Kepler', 'K2')

    Returns:
        Time values in mission-specific format
    """
    config = get_mission_config(mission)
    target_epoch = config.time_epoch

    # Determine source epoch
    if current_format == 'btjd':
        source_epoch = 2457000.0  # TESS epoch
    elif current_format == 'bkjd':
        source_epoch = 2454833.0  # Kepler epoch
    elif current_format in ('bjd', 'jd'):
        source_epoch = 0.0  # Full BJD
    else:
        logger.warning(f"Unknown time format '{current_format}', returning unchanged")
        return time_values

    # Convert: subtract source epoch, add target epoch offset
    if source_epoch == target_epoch:
        return time_values  # Already in correct format
    elif source_epoch == 0.0:
        # Converting from full BJD to mission time
        return time_values - target_epoch
    else:
        # Converting between mission formats (rare but possible)
        bjd_values = time_values + source_epoch  # Convert to full BJD
        return bjd_values - target_epoch  # Convert to target format


# Backwards compatibility alias
def convert_to_btjd(time_values: np.ndarray, current_format: str) -> np.ndarray:
    """
    Convert time values to BTJD (Barycentric TESS Julian Date).

    BTJD is the standard for TESS: TJD = BJD - 2457000.0

    This is a backwards-compatible wrapper around convert_to_mission_time().

    Args:
        time_values: Array of time values
        current_format: Current time format ('btjd', 'bjd', 'jd')

    Returns:
        Time values in BTJD format
    """
    return convert_to_mission_time(time_values, current_format, mission='TESS')


def identify_gaps(lc, min_gap_days: float = 1.0) -> List[Dict[str, float]]:
    """
    Identify gaps in the light curve.

    Args:
        lc: Lightkurve LightCurve object
        min_gap_days: Minimum gap size to report (days)

    Returns:
        List of gap dictionaries with start, end, and duration
    """
    time = lc.time.value
    time_diffs = np.diff(time)

    gaps = []
    for i, dt in enumerate(time_diffs):
        if dt >= min_gap_days:
            gaps.append({
                'start_time': time[i],
                'end_time': time[i + 1],
                'duration_days': dt,
                'gap_index': i
            })

    return gaps


def calculate_sector_offset(flux1: np.ndarray, flux2: np.ndarray,
                            overlap_indices: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Calculate the median offset between overlapping flux values from two sectors.

    Args:
        flux1: Flux from first sector
        flux2: Flux from second sector
        overlap_indices: Tuple of (indices_in_flux1, indices_in_flux2)

    Returns:
        Median flux offset (flux2 - flux1)
    """
    if len(overlap_indices[0]) == 0:
        return 0.0

    flux1_overlap = flux1[overlap_indices[0]]
    flux2_overlap = flux2[overlap_indices[1]]

    # Calculate median difference
    offset = np.median(flux2_overlap - flux1_overlap)
    return offset


def handle_sector_offsets(stitched_lc, processed_sectors: List[Dict]) -> Any:
    """
    Correct for systematic offsets between sectors.

    This is crucial because each TESS sector may have different:
    - Zero-point calibration
    - Systematic trends
    - Aperture photometry settings

    Strategy:
    1. Use first sector as reference (offset = 0)
    2. For each subsequent sector, calculate median offset
    3. Apply corrections to normalize all sectors

    Args:
        stitched_lc: Combined light curve
        processed_sectors: List of sector metadata dicts

    Returns:
        Offset-corrected light curve
    """
    if len(processed_sectors) <= 1:
        return stitched_lc

    logger.info(f"  Correcting offsets between {len(processed_sectors)} sectors...")

    # Extract time and flux arrays
    time = stitched_lc.time.value

    # Handle MaskedArray in the stitched light curve
    if hasattr(stitched_lc.flux, 'value'):
        flux = stitched_lc.flux.value.copy()
    else:
        # It's a regular array
        flux = np.asarray(stitched_lc.flux).copy()

    # Track cumulative lengths using FILTERED lengths (after masking)
    cumulative_lengths = [0]
    for sector_data in processed_sectors:
        # Use the filtered length (after removing masked values)
        # Ensure lc_len is always an integer (critical for indexing)
        lc_len = sector_data.get('filtered_length', len(sector_data['lightcurve'].time))
        lc_len = int(lc_len)  # Convert to int to prevent indexing errors
        cumulative_lengths.append(cumulative_lengths[-1] + lc_len)

    # Debug logging
    logger.debug(f"  Cumulative lengths: {cumulative_lengths}")
    logger.debug(f"  Total flux array length: {len(flux)}")
    logger.debug(f"  Expected total from cumulative: {cumulative_lengths[-1]}")

    # Calculate offsets relative to first sector
    offsets = [0.0]  # First sector is reference

    for i in range(1, len(processed_sectors)):
        start_idx = cumulative_lengths[i]
        end_idx = cumulative_lengths[i + 1]

        # Get flux for this sector
        sector_flux = flux[start_idx:end_idx]

        # Calculate offset relative to previous sector's median
        prev_sector_median = np.median(flux[cumulative_lengths[i-1]:cumulative_lengths[i]])
        current_sector_median = np.median(sector_flux)

        offset = current_sector_median - prev_sector_median
        offsets.append(offset)

        # Apply correction (subtract offset to align with reference)
        flux[start_idx:end_idx] -= offset

        logger.info(f"    Sector {processed_sectors[i]['sector']}: offset = {offset:.6f}")

    # Create new light curve with corrected flux
    corrected_lc = stitched_lc.copy()
    corrected_lc.flux = flux

    return corrected_lc


def stitch_multi_sector_lightcurve(target_id: int,
                                   mission: str = 'TESS',
                                   quality_bitmask: str = 'default',
                                   sigma_clip: float = 5.0,
                                   max_sectors: int = None,
                                   jd_min: float = None,
                                   jd_max: float = None,
                                   filter_contaminated: bool = True,
                                   amplitude_threshold: float = 2.5,
                                   flux_column: str = None) -> Optional[Dict[str, Any]]:
    """
    Combine all available observation units for a target with quality filtering.

    Supports multiple space telescope missions:
    - TESS: Combines sectors (27-day periods)
    - Kepler: Combines quarters (90-day periods)
    - K2: Combines campaigns (80-day periods)

    This function:
    1. Downloads all available observation units for the mission
    2. Processes each unit (outlier removal, normalization)
    3. Calculates amplitude metrics for each unit
    4. Filters out contaminated units (pre-analysis quality control)
    5. Stitches clean units together
    6. Corrects for systematic offsets
    7. Returns metadata about gaps and coverage

    Args:
        target_id: Target identifier (TIC for TESS, KIC for Kepler, EPIC for K2)
        mission: Space telescope mission ('TESS', 'Kepler', 'K2')
        quality_bitmask: Quality flags for data filtering
        sigma_clip: Sigma threshold for outlier removal
        max_sectors: Maximum observation units to download (default: 8 for memory safety)
        jd_min: Minimum Julian Date to include (optional)
        jd_max: Maximum Julian Date to include (optional)
        filter_contaminated: If True, exclude units with anomalous amplitude (default: True)
        amplitude_threshold: Reject units with amplitude < median/threshold (default: 2.5)
        flux_column: Which flux column to use. Options:
            - None (default): Use mission default (PDCSAP for TESS, SAP for Kepler/K2)
            - 'sap': Use SAP_FLUX (Simple Aperture Photometry, less processed)
            - 'pdcsap': Use PDCSAP_FLUX (Pre-search Data Conditioning, more processed)
            For Kepler/K2, SAP_FLUX preserves more astrophysical signal.

    Returns:
        Dictionary containing:
        - stitched_lc: Combined light curve
        - sectors_used: List of sector/quarter/campaign numbers (only clean units)
        - total_baseline_days: Total time coverage
        - gaps: List of gap information
        - n_sectors: Number of clean units combined
        - metadata: Per-unit processing info
        - time_filter_applied: Whether time filtering was used
        - sectors_rejected: List of rejected unit numbers
        - sectors_rejected_count: Number of rejected units
        - quality_report: Detailed quality filtering report
        - has_contaminated_sectors: Whether any contamination was detected
        - mission: The mission name for reference

        Returns None if no data available
    """
    # Get mission configuration
    config = get_mission_config(mission)
    target_name = format_target_name(mission, target_id)
    obs_unit = config.observation_unit.lower()  # 'sector', 'quarter', or 'campaign'

    # Determine which flux column to use
    # For Kepler/K2, default to SAP_FLUX to preserve astrophysical signals
    # PDCSAP over-processes and removes variability in many cases
    if flux_column is None:
        if mission in ('Kepler', 'K2'):
            use_flux_column = 'sap'  # SAP preserves more signal for Kepler
        else:
            use_flux_column = 'pdcsap'  # TESS PDCSAP is usually fine
    else:
        use_flux_column = flux_column.lower()

    logger.info(f"🔗 Multi-{obs_unit} stitching for {target_name}")
    logger.info(f"  Using flux column: {use_flux_column.upper()}_FLUX")

    # Step 0: Check for cached stitched lightcurve in Supabase storage
    # This allows enrichment to work even when MAST is unavailable
    if mission == 'TESS':
        try:
            from abastris.db.lightcurve_storage import get_lightcurve_data
            cached_data = get_lightcurve_data(target_id, max_sectors=max_sectors or 8)
            if cached_data and cached_data.get('source') == 'cache':
                logger.info(f"  ✅ Using cached lightcurve from Supabase storage")
                lc = cached_data['lc']
                sectors = cached_data.get('sectors', [])

                # Build a result compatible with the standard return format
                time = lc.time.value
                baseline = time.max() - time.min()

                return {
                    'stitched_lc': lc,
                    'sectors_used': sectors,
                    'total_baseline_days': baseline,
                    'gaps': [],
                    'n_sectors': len(sectors) if sectors else 1,
                    'n_gaps': 0,
                    'metadata': {'source': 'supabase_cache'},
                    'time_filter_applied': False,
                    'sectors_rejected': [],
                    'sectors_rejected_count': 0,
                    'quality_report': {},
                    'has_contaminated_sectors': False,
                    'mission': mission
                }
        except Exception as e:
            logger.warning(f"  Cache lookup failed, falling back to MAST: {e}")

    # Step 1: Search for all available observation units
    try:
        search_result = lk.search_lightcurve(target_name, mission=config.name)
    except Exception as e:
        logger.error(f"  ❌ Search failed: {e}")
        return None

    if len(search_result) == 0:
        logger.warning(f"  ⚠️  No {mission} data found")
        return None

    logger.info(f"  Found {len(search_result)} {obs_unit}(s)")

    # Memory safety: Limit sectors based on parameter or default
    # Default 4 for Kepler/K2 (many quarters = timeout risk on Render)
    # Default 8 for TESS (shorter sectors, less data)
    # Each sector ~3-4 MB, so 4 sectors = ~16 MB (safer for Celery workers)
    if max_sectors is not None:
        MAX_SECTORS_FOR_MEMORY = max_sectors
    elif mission in ('Kepler', 'K2'):
        MAX_SECTORS_FOR_MEMORY = 4  # Reduced to avoid Render timeouts
    else:
        MAX_SECTORS_FOR_MEMORY = 8
    if len(search_result) > MAX_SECTORS_FOR_MEMORY:
        logger.warning(f"  ⚠️  Limiting to {MAX_SECTORS_FOR_MEMORY} sectors (memory constraint). Found {len(search_result)} total.")
        search_result = search_result[:MAX_SECTORS_FOR_MEMORY]

    # Step 2: Download sectors individually to handle corrupted files gracefully
    # Instead of download_all(), download one at a time so corrupted files don't fail entire job
    downloaded_sectors = []
    failed_sectors = []

    for i, search_entry in enumerate(search_result):
        try:
            # Extract observation unit number from mission string
            obs_num = search_entry.mission[0].split(' ')[-1] if hasattr(search_entry, 'mission') else f"{i+1}"
            logger.info(f"  Downloading {config.observation_unit} {obs_num}...")

            lc = search_entry.download(quality_bitmask=quality_bitmask)

            if lc is not None:
                downloaded_sectors.append(lc)
                logger.info(f"    ✅ Downloaded ({len(lc.time)} points)")
            else:
                failed_sectors.append(obs_num)
                logger.warning(f"    ⚠️  Download returned None, skipping")

            # Force garbage collection after each download to prevent memory buildup
            gc.collect()

        except Exception as e:
            obs_num = f"{i+1}"
            failed_sectors.append(obs_num)
            logger.warning(f"    ⚠️  Download failed for {obs_unit} {obs_num}: {str(e)[:100]}")
            # Continue with next sector instead of failing completely
            continue

    if len(failed_sectors) > 0:
        logger.warning(f"  ⚠️  Failed to download {len(failed_sectors)} {obs_unit}(s): {', '.join(map(str, failed_sectors))}")

    if len(downloaded_sectors) == 0:
        logger.error(f"  ❌ No {obs_unit}s successfully downloaded")
        return None

    logger.info(f"  ✅ Successfully downloaded {len(downloaded_sectors)}/{len(search_result)} {obs_unit}(s)")

    # Step 3: Process each sector individually
    processed_sectors = []

    for i, lc in enumerate(downloaded_sectors):
        try:
            # Get observation unit number using mission-specific metadata key
            sector = lc.meta.get(config.observation_key, i + 1)

            logger.info(f"  Processing {config.observation_unit} {sector}...")

            # Remove NaNs
            lc = lc.remove_nans()

            if len(lc.time) == 0:
                logger.warning(f"    ⚠️  Empty after removing NaNs, skipping")
                continue

            # Remove outliers
            lc = lc.remove_outliers(sigma=sigma_clip)

            if len(lc.time) == 0:
                logger.warning(f"    ⚠️  Empty after outlier removal, skipping")
                continue

            # DO NOT normalize here - keep raw flux for proper offset correction
            # Normalization will happen once after stitching and offset correction

            # Calculate amplitude metrics for quality filtering
            amplitude_metrics = calculate_sector_amplitude(lc)

            # Store metadata
            processed_sectors.append({
                'lightcurve': lc,
                'sector': sector,
                'time_coverage': [lc.time.min().value, lc.time.max().value],
                'n_points': len(lc.time),
                'median_flux': np.median(lc.flux.value),
                'amplitude': amplitude_metrics  # NEW: Amplitude metrics for filtering
            })

            logger.info(f"    ✅ {len(lc.time)} points, amplitude: {amplitude_metrics['amplitude_percent']:.3f}%")

        except Exception as e:
            logger.warning(f"    ⚠️  Failed to process sector: {e}")
            continue

    if len(processed_sectors) == 0:
        logger.error(f"  ❌ No sectors successfully processed")
        return None

    # Step 4: Filter contaminated sectors BEFORE combining
    quality_report = None
    rejected_sectors = []

    if filter_contaminated and len(processed_sectors) > 1:
        sectors_to_stitch, rejected_sectors, quality_report = filter_contaminated_sectors(
            processed_sectors,
            amplitude_threshold=amplitude_threshold
        )
    else:
        sectors_to_stitch = processed_sectors
        quality_report = {'filtering_applied': False, 'reason': 'disabled_or_single_sector'}

    # Step 5: Stitch sectors together (ONLY clean sectors)
    logger.info(f"  Stitching {len(sectors_to_stitch)} sector(s)...")

    if len(sectors_to_stitch) == 1:
        # Only one sector (or one clean sector), no stitching needed
        lc = sectors_to_stitch[0]['lightcurve']

        # For single sector, also apply SAP_FLUX preference for Kepler/K2
        if use_flux_column == 'sap' and hasattr(lc, 'sap_flux') and lc.sap_flux is not None:
            # Create a copy with SAP_FLUX as the main flux
            stitched = lc.copy()
            stitched.flux = lc.sap_flux
            logger.info(f"  ✅ Single sector with SAP_FLUX, no stitching required")
        else:
            stitched = lc
            logger.info(f"  ✅ Single sector, no stitching required")
    else:
        # Combine multiple sectors manually to avoid metadata conflicts
        # Extract just time, flux, and flux_err arrays
        all_time = []
        all_flux = []
        all_flux_err = []

        for sector_data in sectors_to_stitch:  # Use clean sectors only
            lc = sector_data['lightcurve']
            time_vals = lc.time.value

            # Detect and convert time format to mission-specific format
            time_format = detect_time_format(time_vals, mission)
            time_vals = convert_to_mission_time(time_vals, time_format, mission)

            expected_format = config.time_format_name.lower()
            if time_format != expected_format:
                logger.info(f"    Converted time from {time_format.upper()} to {config.time_format_name}")

            # Select flux column based on mission preference
            # For Kepler/K2, SAP_FLUX preserves more astrophysical signal
            if use_flux_column == 'sap' and hasattr(lc, 'sap_flux') and lc.sap_flux is not None:
                flux_source = lc.sap_flux
                flux_source_name = 'SAP_FLUX'
            else:
                flux_source = lc.flux  # Default (usually PDCSAP_FLUX)
                flux_source_name = 'PDCSAP_FLUX'

            # CRITICAL FIX: Handle MaskedArray properly to avoid 1e20 fill values
            # Check if flux is a masked array (astropy.table.column.MaskedColumn)
            if hasattr(flux_source, 'mask'):
                # Get the explicit mask and convert to boolean array
                flux_mask = np.asarray(flux_source.mask, dtype=bool)
                # Get the actual data (including values that will be masked)
                flux_vals = np.asarray(flux_source.data)
            else:
                # It's a regular array or Quantity
                flux_mask = np.zeros(len(flux_source), dtype=bool)
                if hasattr(flux_source, 'value'):
                    flux_vals = np.asarray(flux_source.value)
                else:
                    flux_vals = np.asarray(lc.flux)

            # Ensure time_vals is also a plain numpy array
            time_vals = np.asarray(time_vals)

            # Create combined filter mask
            # Filter NaNs, Infs, AND explicitly masked data points
            finite_mask = np.isfinite(time_vals) & np.isfinite(flux_vals)
            combined_mask = finite_mask & (~flux_mask)  # Invert flux_mask

            # Append clean, RAW flux data
            clean_time = time_vals[combined_mask]
            clean_flux = flux_vals[combined_mask]

            all_time.append(clean_time)
            all_flux.append(clean_flux)

            # Update sector_data with actual length after filtering
            # Store as int explicitly to prevent indexing errors later
            sector_data['filtered_length'] = int(len(clean_flux))

            if hasattr(lc, 'flux_err') and lc.flux_err is not None:
                if hasattr(lc.flux_err, 'mask'):
                    flux_err_vals = np.asarray(lc.flux_err.data)
                else:
                    if hasattr(lc.flux_err, 'value'):
                        flux_err_vals = np.asarray(lc.flux_err.value)
                    else:
                        flux_err_vals = np.asarray(lc.flux_err)
                all_flux_err.append(flux_err_vals[combined_mask])

        # Concatenate arrays
        time_combined = np.concatenate(all_time)
        flux_combined = np.concatenate(all_flux)

        # Create new light curve from combined arrays
        from lightkurve import LightCurve
        stitched = LightCurve(
            time=time_combined,
            flux=flux_combined,
            meta=sectors_to_stitch[0]['lightcurve'].meta
        )

        # Step 6: Handle systematic offsets between sectors
        stitched = handle_sector_offsets(stitched, sectors_to_stitch)

        logger.info(f"  ✅ Stitched {len(sectors_to_stitch)} clean sectors")

    # Step 6: Apply time-range filtering if specified
    # This allows excluding data with known artifacts (per VSX moderator feedback)
    time_filter_applied = False
    points_before_filter = len(stitched.time)

    if jd_min is not None or jd_max is not None:
        # Convert JD limits to mission-specific time format
        mission_epoch = config.time_epoch

        time_vals = stitched.time.value
        flux_vals = stitched.flux.value if hasattr(stitched.flux, 'value') else np.asarray(stitched.flux)

        # Create time mask
        time_mask = np.ones(len(time_vals), dtype=bool)

        if jd_min is not None:
            mission_time_min = jd_min - mission_epoch
            time_mask &= (time_vals >= mission_time_min)
            logger.info(f"  🔍 Applying time filter: JD >= {jd_min} ({config.time_format_name} >= {mission_time_min:.2f})")

        if jd_max is not None:
            mission_time_max = jd_max - mission_epoch
            time_mask &= (time_vals <= mission_time_max)
            logger.info(f"  🔍 Applying time filter: JD <= {jd_max} ({config.time_format_name} <= {mission_time_max:.2f})")

        # Apply mask
        filtered_time = time_vals[time_mask]
        filtered_flux = flux_vals[time_mask]

        if len(filtered_time) < 100:
            logger.warning(f"  ⚠️  Time filter would leave only {len(filtered_time)} points, keeping original data")
        else:
            # Create filtered light curve
            from lightkurve import LightCurve
            stitched = LightCurve(
                time=filtered_time,
                flux=filtered_flux,
                meta=stitched.meta
            )
            time_filter_applied = True
            logger.info(f"  ✅ Time filter applied: {points_before_filter} -> {len(stitched.time)} points")

    # Step 7: Calculate gaps
    gaps = identify_gaps(stitched, min_gap_days=1.0)

    # Step 8: Calculate total baseline
    total_baseline = np.ptp(stitched.time).value  # Convert TimeDelta to float

    # Prepare result with quality metadata
    result = {
        'stitched_lc': stitched,
        'mission': mission,  # Include mission for reference
        'observation_unit': config.observation_unit,  # 'SECTOR', 'QUARTER', 'CAMPAIGN'
        'sectors_used': [s['sector'] for s in sectors_to_stitch],  # Only clean units
        'sector_lightcurves': [s['lightcurve'] for s in sectors_to_stitch],  # Individual LCs
        'n_sectors': len(sectors_to_stitch),  # Only clean units count
        'total_baseline_days': float(total_baseline),
        'total_points': len(stitched.time),
        'gaps': gaps,
        'n_gaps': len(gaps),
        'metadata': sectors_to_stitch,  # Only clean unit metadata
        'sectors_attempted': len(search_result),
        'sectors_failed': len(failed_sectors),
        'failed_sector_list': failed_sectors,
        'time_filter_applied': time_filter_applied,
        'jd_min': jd_min,
        'jd_max': jd_max,
        'points_before_filter': points_before_filter if time_filter_applied else None,

        # Quality filtering metadata
        'sectors_rejected': [s['sector'] for s in rejected_sectors],
        'sectors_rejected_count': len(rejected_sectors),
        'rejected_sector_metadata': rejected_sectors,
        'quality_report': quality_report,
        'has_contaminated_sectors': len(rejected_sectors) > 0,
        'all_sectors_processed': [s['sector'] for s in processed_sectors],  # All units before filtering
    }

    # Enhanced logging
    logger.info(f"  📊 Total: {len(stitched.time)} points over {total_baseline:.1f} days")
    logger.info(f"  📊 Clean sectors: {', '.join(map(str, result['sectors_used']))}")
    logger.info(f"  📊 Gaps: {len(gaps)} gaps identified")

    if len(rejected_sectors) > 0:
        logger.info(f"  🚫 Rejected sectors: {', '.join(map(str, result['sectors_rejected']))}")

    if len(failed_sectors) > 0:
        logger.info(f"  ⚠️  Download failures: {', '.join(map(str, failed_sectors))}")

    return result


def compare_single_vs_multi_sector(target_id: int, mission: str = 'TESS') -> Dict[str, Any]:
    """
    Compare period detection using single observation unit vs multi-unit stitching.

    Useful for validation and understanding improvement from stitching.

    Args:
        target_id: Target identifier (TIC, KIC, or EPIC)
        mission: Space telescope mission ('TESS', 'Kepler', 'K2')

    Returns:
        Dictionary comparing results
    """
    from scipy.signal import lombscargle

    config = get_mission_config(mission)
    target_name = format_target_name(mission, target_id)
    logger.info(f"📊 Comparing single vs multi-{config.observation_unit.lower()} for {target_name}")

    # Get multi-sector result
    multi_result = stitch_multi_sector_lightcurve(target_id, mission=mission)

    if multi_result is None:
        return {'error': 'No data available'}

    # If only one sector, comparison is trivial
    if multi_result['n_sectors'] == 1:
        return {
            'n_sectors': 1,
            'note': 'Only one sector available, no comparison needed'
        }

    # Analyze with multi-sector
    multi_lc = multi_result['stitched_lc']
    time_multi = multi_lc.time.value
    flux_multi = multi_lc.flux.value

    # Analyze with single sector (first one)
    single_lc = multi_result['metadata'][0]['lightcurve']
    time_single = single_lc.time.value
    flux_single = single_lc.flux.value

    # Simple period detection for comparison
    from astropy.timeseries import LombScargle

    # Multi-sector
    ls_multi = LombScargle(time_multi, flux_multi)
    freq_multi = np.linspace(0.1, 50, 10000)
    power_multi = ls_multi.power(freq_multi)
    best_period_multi = 1 / freq_multi[np.argmax(power_multi)]

    # Single sector
    ls_single = LombScargle(time_single, flux_single)
    freq_single = np.linspace(0.1, 50, 10000)
    power_single = ls_single.power(freq_single)
    best_period_single = 1 / freq_single[np.argmax(power_single)]

    return {
        'n_sectors': multi_result['n_sectors'],
        'sectors': multi_result['sectors_used'],
        'single_sector': {
            'period': best_period_single,
            'n_points': len(time_single),
            'baseline_days': np.ptp(time_single),
            'max_power': float(np.max(power_single))
        },
        'multi_sector': {
            'period': best_period_multi,
            'n_points': len(time_multi),
            'baseline_days': np.ptp(time_multi),
            'max_power': float(np.max(power_multi))
        },
        'improvement': {
            'period_change_pct': 100 * abs(best_period_multi - best_period_single) / best_period_single,
            'baseline_increase': np.ptp(time_multi) / np.ptp(time_single),
            'points_increase': len(time_multi) / len(time_single),
            'power_increase_pct': 100 * (np.max(power_multi) - np.max(power_single)) / np.max(power_single)
        }
    }

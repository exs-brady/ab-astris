"""
Z24 Bridge Frequency Extraction using Ab Astris Pipeline

Extracts structural natural frequencies from Z24 bridge acceleration time
histories using the Ab Astris cross-domain signal detection methodology.

Pipeline:
1. Load multi-channel acceleration data (100 Hz, ~600s duration)
2. Select best channel (SNR-based)
3. Bandpass filter (0.5-25 Hz structural domain)
4. Lomb-Scargle periodogram analysis
5. Multi-window validation (CV computation)
6. Bootstrap error estimation
7. Confidence scoring (0-100)

Adapted from oceanography/tide_gauge_experiment.py for structural vibration domain.

Usage:
    python shm/z24/z24_frequency_extraction.py --phase 01 --max 10
"""

import json
import sys
import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# Import Phase 1 data loader
sys.path.append(str(Path(__file__).parent.parent.parent))
from shm.z24.z24_data_loader import (
    load_z24_measurement,
    inventory_z24_dataset,
    identify_damage_states,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Output paths
OUTPUT_DIR = Path(__file__).parent
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_CSV = RESULTS_DIR / "z24_frequencies.csv"

# Frequency range (Hz) - structural domain
FREQ_MIN = 0.05          # Lowest expected mode (conservative)
FREQ_MAX = 25.0          # Upper bound for structural modes
N_FREQ_POINTS = 10000    # LS grid resolution

# Bandpass filter
BANDPASS_LOW = 0.5       # Hz - structural domain lower bound
BANDPASS_HIGH = 25.0     # Hz - upper bound
FILTER_ORDER = 4         # Butterworth order

# Multi-window validation
N_WINDOWS = 6            # For ~600s recordings: ~100s per window
WINDOW_DURATION_SEC = 100.0  # Seconds per window
FREQ_TOLERANCE = 0.15    # 15% tolerance for detection rate

# Bootstrap
N_BOOTSTRAP = 100        # Resamples for uncertainty estimation

# Expected ranges (for validation)
EXPECTED_CV = 1.0        # Target CV ~1% for structural modes
CV_GOOD = 5.0            # <5% is acceptable
EXPECTED_FREQ_RANGE = (0.5, 10.0)  # Hz - typical for 30m bridges

# Confidence scoring thresholds (adapted from oceanography)
CV_EXCELLENT = 1.0       # <1% for structural
CV_GOOD = 5.0            # <5%
CV_FAIR = 10.0           # <10%
FAP_EXCELLENT = 1e-10
FAP_GOOD = 1e-5


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LSResult:
    """Results from Lomb-Scargle periodogram analysis."""
    frequency_hz: float
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
class Z24FrequencyResult:
    """Complete frequency extraction results for one Z24 measurement."""
    measurement_id: str              # e.g., "01C14"
    phase: str                       # Damage phase
    config: str                      # Measurement configuration
    seq: str                         # Sequence number
    best_channel: str                # Selected channel name
    best_channel_snr_db: float       # SNR of selected channel

    # Fundamental mode (most important)
    freq_1_hz: float
    freq_1_std_hz: float             # Bootstrap uncertainty
    freq_1_cv_pct: float             # Multi-window CV
    freq_1_confidence: float         # 0-100 score
    freq_1_tier: str                 # CONFIRMED/PROBABLE/etc.

    # Higher modes (if detected with confidence >70)
    freq_2_hz: Optional[float] = None
    freq_3_hz: Optional[float] = None
    freq_4_hz: Optional[float] = None
    freq_5_hz: Optional[float] = None

    # Validation metrics
    detection_rate: float = 0.0      # Fraction of windows detecting f1
    fap: float = 1.0                 # False alarm probability
    n_modes_detected: int = 1        # Total modes with confidence >70


# ============================================================================
# SIGNAL PROCESSING (ADAPTED FROM OCEANOGRAPHY)
# ============================================================================

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    f_low: float = BANDPASS_LOW,
    f_high: float = BANDPASS_HIGH,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.

    Adapted from oceanography/tide_gauge_experiment.py for structural domain.

    Args:
        signal: Input time series
        fs: Sampling frequency (Hz)
        f_low: Low cutoff frequency (Hz)
        f_high: High cutoff frequency (Hz)
        order: Filter order

    Returns:
        Filtered signal (same length as input)
    """
    nyq = fs / 2
    low = f_low / nyq
    high = f_high / nyq

    # Clamp to valid range
    low = max(low, 1e-10)
    high = min(high, 0.9999)

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def run_lomb_scargle(
    time: np.ndarray,
    signal: np.ndarray,
    freq_min_hz: float = FREQ_MIN,
    freq_max_hz: float = FREQ_MAX,
    n_points: int = N_FREQ_POINTS,
) -> LSResult:
    """Run Lomb-Scargle periodogram following Ab Astris methodology.

    Copied directly from oceanography module - domain-agnostic.

    Args:
        time: Time array (seconds)
        signal: Signal values
        freq_min_hz: Minimum frequency (Hz)
        freq_max_hz: Maximum frequency (Hz)
        n_points: Number of frequency points

    Returns:
        LSResult with detected frequency and full spectrum
    """
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
    """Create overlapping windows for multi-window validation.

    Adapted from oceanography module - changed from days to seconds.

    Args:
        time: Time array (seconds)
        signal: Signal values
        n_windows: Number of windows
        window_duration_sec: Window duration in seconds (was days in oceanography)

    Returns:
        List of dictionaries with 'time' and 'signal' for each window
    """
    total_duration = time[-1] - time[0]
    if total_duration < window_duration_sec:
        raise ValueError(
            f"Signal duration ({total_duration:.0f}s) shorter than "
            f"window duration ({window_duration_sec:.0f}s)"
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
    freq_min_hz: float = FREQ_MIN,
    freq_max_hz: float = FREQ_MAX,
) -> MultiWindowResult:
    """Run Lomb-Scargle on each window. Compute CV across windows.

    Copied directly from oceanography module - domain-agnostic.

    Args:
        windows: List of window dictionaries
        target_freq_hz: Target frequency for detection rate
        freq_min_hz: Minimum search frequency
        freq_max_hz: Maximum search frequency

    Returns:
        MultiWindowResult with CV and detection rate
    """
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


def run_bootstrap(
    time: np.ndarray,
    signal: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    freq_min_hz: float = FREQ_MIN,
    freq_max_hz: float = FREQ_MAX,
) -> BootstrapResult:
    """Bootstrap resampling for frequency uncertainty estimation.

    Copied directly from oceanography module - domain-agnostic.

    Args:
        time: Time array (seconds)
        signal: Signal values
        n_bootstrap: Number of bootstrap resamples
        freq_min_hz: Minimum search frequency
        freq_max_hz: Maximum search frequency

    Returns:
        BootstrapResult with mean, std, and confidence intervals
    """
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


def compute_confidence_score(
    ls_result: LSResult,
    mw_result: MultiWindowResult,
    target_freq_hz: float,
) -> Tuple[float, str]:
    """Compute confidence score (0-100) following Ab Astris framework.

    Adapted from oceanography module for structural domain CV thresholds.

    Scoring:
    - 30 pts: Frequency accuracy (error from target)
    - 25 pts: Signal significance (FAP)
    - 25 pts: Multi-window stability (CV)
    - 20 pts: Detection rate across windows

    Args:
        ls_result: Lomb-Scargle results
        mw_result: Multi-window validation results
        target_freq_hz: Target frequency

    Returns:
        (score, tier) where score is 0-100 and tier is CONFIRMED/PROBABLE/CANDIDATE/NOISE
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
    if fap < FAP_EXCELLENT:
        score += 25
    elif fap < FAP_GOOD:
        score += 20
    elif fap < 1e-3:
        score += 15
    elif fap < 0.01:
        score += 10
    elif fap < 0.05:
        score += 5

    # 3. CV stability (25 pts) - ADAPTED for structural domain
    cv = mw_result.cv_clean
    if cv < CV_EXCELLENT:
        score += 25
    elif cv < CV_GOOD:
        score += 20
    elif cv < CV_FAIR:
        score += 15
    elif cv < 20.0:
        score += 10
    elif cv < 30.0:
        score += 5

    # 4. Detection rate (20 pts)
    score += 20.0 * mw_result.detection_rate

    # Tier
    if score >= 85:
        tier = "CONFIRMED"
    elif score >= 70:
        tier = "PROBABLE"
    elif score >= 50:
        tier = "CANDIDATE"
    else:
        tier = "NOISE"

    return round(score, 1), tier


# ============================================================================
# MULTI-CHANNEL SELECTION (NEW FOR Z24)
# ============================================================================

def select_best_channel(
    acceleration_matrix: np.ndarray,
    fs: float,
    channel_names: List[str],
    target_freq_range: Tuple[float, float] = EXPECTED_FREQ_RANGE,
) -> Tuple[int, str, float]:
    """Select the best channel from multi-channel Z24 data using SNR.

    For each channel:
    1. Apply bandpass filter
    2. Run quick FFT to find peak in target range
    3. Compute SNR = peak / median(noise floor)
    4. Return channel with highest SNR

    Args:
        acceleration_matrix: (n_samples, n_channels)
        fs: Sampling rate (Hz)
        channel_names: List of channel identifiers
        target_freq_range: Expected fundamental frequency range (Hz)

    Returns:
        (best_channel_idx, best_channel_name, snr_db)
    """
    n_channels = acceleration_matrix.shape[1]
    channel_scores = []

    for ch_idx in range(n_channels):
        # Extract and filter channel
        accel = acceleration_matrix[:, ch_idx]
        filtered = bandpass_filter(accel, fs)

        # Compute FFT
        n = len(filtered)
        freqs = fftfreq(n, 1/fs)[:n//2]
        amplitudes = 2.0/n * np.abs(fft(filtered))[:n//2]

        # Find peak in target range
        target_mask = (freqs >= target_freq_range[0]) & (freqs <= target_freq_range[1])
        if not np.any(target_mask):
            channel_scores.append((ch_idx, -np.inf))
            continue

        peak_amp = np.max(amplitudes[target_mask])

        # Compute noise floor (median of 10-25 Hz band, excluding target range)
        noise_mask = (freqs >= 10) & (freqs <= 25) & ~target_mask
        if np.any(noise_mask):
            noise_floor = np.median(amplitudes[noise_mask])
        else:
            noise_floor = np.median(amplitudes[freqs > 5])

        # SNR in dB
        if noise_floor > 1e-15:
            snr_db = 20 * np.log10((peak_amp + 1e-15) / noise_floor)
        else:
            snr_db = 0.0

        channel_scores.append((ch_idx, snr_db))

    # Select channel with highest SNR
    best_idx, best_snr = max(channel_scores, key=lambda x: x[1])
    best_name = channel_names[best_idx]

    return best_idx, best_name, best_snr


# ============================================================================
# HIGHER MODE DETECTION (NEW FOR Z24)
# ============================================================================

def detect_higher_modes(
    frequencies_hz: np.ndarray,
    powers: np.ndarray,
    fundamental_hz: float,
    n_modes: int = 5,
    min_separation_factor: float = 1.5,
) -> List[float]:
    """Find higher modes (2nd, 3rd, 4th, 5th) in Lomb-Scargle spectrum.

    Higher modes should be:
    - Above fundamental frequency
    - Separated by at least 1.5× fundamental (rough heuristic)
    - Have significant power

    Args:
        frequencies_hz: Frequency array from LS
        powers: Power array from LS
        fundamental_hz: Detected fundamental frequency
        n_modes: Number of higher modes to search for (default 5 total = 1 fundamental + 4 higher)
        min_separation_factor: Minimum frequency ratio between modes

    Returns:
        List of higher mode frequencies (up to n_modes-1)
    """
    # Only search above fundamental
    search_mask = frequencies_hz > fundamental_hz * min_separation_factor

    if not np.any(search_mask):
        return []

    # Find peaks above fundamental
    from scipy.signal import find_peaks
    peak_indices, properties = find_peaks(
        powers,
        prominence=0.01,  # Require some minimum prominence
        distance=int(len(frequencies_hz) * 0.02),  # Peaks must be separated
    )

    # Filter to peaks in search range
    valid_peaks = [idx for idx in peak_indices if search_mask[idx]]

    if len(valid_peaks) == 0:
        return []

    # Sort by power (strongest first)
    valid_peaks = sorted(valid_peaks, key=lambda idx: powers[idx], reverse=True)

    # Return up to n_modes-1 higher modes
    higher_modes = [frequencies_hz[idx] for idx in valid_peaks[:n_modes-1]]

    return higher_modes


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_single_measurement(zip_path: str) -> Z24FrequencyResult:
    """Process one Z24 measurement through full Ab Astris pipeline.

    Args:
        zip_path: Path to .zip file

    Returns:
        Z24FrequencyResult with frequency extraction results
    """
    # Step 1: Load data
    measurement = load_z24_measurement(zip_path)

    # Step 2: Select best channel
    best_ch_idx, best_ch_name, snr_db = select_best_channel(
        measurement['acceleration'],
        measurement['sampling_rate'],
        measurement['channels']
    )

    # Step 3: Extract selected channel
    acceleration = measurement['acceleration'][:, best_ch_idx]
    fs = measurement['sampling_rate']

    # Step 4: Bandpass filter (0.5-25 Hz)
    filtered = bandpass_filter(acceleration, fs)

    # Step 5: Prepare time array
    time = np.arange(len(filtered)) / fs

    # Step 6: Global Lomb-Scargle (find fundamental mode)
    ls_result = run_lomb_scargle(time, filtered)

    # Step 7: Multi-window validation
    windows = create_windows(time, filtered)
    mw_result = run_multi_window_validation(
        windows,
        target_freq_hz=ls_result.frequency_hz
    )

    # Step 8: Bootstrap uncertainty
    bootstrap_result = run_bootstrap(time, filtered)

    # Step 9: Confidence scoring
    confidence, tier = compute_confidence_score(
        ls_result, mw_result, ls_result.frequency_hz
    )

    # Step 10: Higher mode detection (optional)
    higher_modes = detect_higher_modes(
        ls_result.frequencies_hz,
        ls_result.powers,
        fundamental_hz=ls_result.frequency_hz
    )

    # Step 11: Package results
    return Z24FrequencyResult(
        measurement_id=f"{measurement['phase']}{measurement['config']}{measurement['seq']}",
        phase=measurement['phase'],
        config=measurement['config'],
        seq=measurement['seq'],
        best_channel=best_ch_name,
        best_channel_snr_db=snr_db,
        freq_1_hz=ls_result.frequency_hz,
        freq_1_std_hz=bootstrap_result.freq_std,
        freq_1_cv_pct=mw_result.cv_clean,
        freq_1_confidence=confidence,
        freq_1_tier=tier,
        freq_2_hz=higher_modes[0] if len(higher_modes) > 0 else None,
        freq_3_hz=higher_modes[1] if len(higher_modes) > 1 else None,
        freq_4_hz=higher_modes[2] if len(higher_modes) > 2 else None,
        freq_5_hz=higher_modes[3] if len(higher_modes) > 3 else None,
        detection_rate=mw_result.detection_rate,
        fap=ls_result.fap,
        n_modes_detected=1 + len(higher_modes)
    )


def extract_z24_frequencies(
    data_dir: str,
    output_csv: str,
    phase_filter: Optional[List[str]] = None,
    max_measurements: Optional[int] = None,
) -> pd.DataFrame:
    """Extract natural frequencies from Z24 measurements.

    Args:
        data_dir: Path to Z24 data root
        output_csv: Output CSV path
        phase_filter: Optional list of phases to process (e.g., ['01', '02'])
        max_measurements: Optional limit (for testing)

    Returns:
        DataFrame of frequency results
    """
    # Step 1: Load inventory
    print("Loading Z24 dataset inventory...")
    inventory = inventory_z24_dataset(data_dir)
    print(f"Found {len(inventory)} measurements")

    # Step 2: Filter to requested phases
    if phase_filter:
        inventory = inventory[inventory['phase'].isin(phase_filter)]
        print(f"Filtered to phases {phase_filter}: {len(inventory)} measurements")

    # Step 3: Limit for testing
    if max_measurements:
        inventory = inventory.head(max_measurements)
        print(f"Limited to first {max_measurements} measurements")

    # Step 4: Process each measurement
    results = []
    for idx, row in inventory.iterrows():
        print(f"\n[{idx+1}/{len(inventory)}] Processing {Path(row['file_path']).name}...")
        try:
            result = process_single_measurement(row['file_path'])
            results.append(result)
            print(f"  f1={result.freq_1_hz:.3f} Hz, CV={result.freq_1_cv_pct:.2f}%, tier={result.freq_1_tier}")
            if result.n_modes_detected > 1:
                print(f"  Detected {result.n_modes_detected} modes")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Step 5: Save to CSV
    if len(results) > 0:
        df = pd.DataFrame([asdict(r) for r in results])
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Saved {len(results)} results to {output_csv}")

        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Mean f1:       {df['freq_1_hz'].mean():.3f} Hz")
        print(f"Std f1:        {df['freq_1_hz'].std():.3f} Hz")
        print(f"Mean CV:       {df['freq_1_cv_pct'].mean():.2f}%")
        print(f"Mean confidence: {df['freq_1_confidence'].mean():.1f}")
        print(f"Tier distribution:")
        print(df['freq_1_tier'].value_counts())
        print(f"Mean modes detected: {df['n_modes_detected'].mean():.1f}")

        return df
    else:
        print("\nNo successful extractions.")
        return pd.DataFrame()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract natural frequencies from Z24 bridge measurements'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='shm/data/Z24',
        help='Path to Z24 data directory (default: shm/data/Z24)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(RESULTS_CSV),
        help=f'Output CSV path (default: {RESULTS_CSV})'
    )
    parser.add_argument(
        '--phase',
        type=str,
        nargs='+',
        default=None,
        help='Filter to specific phases (e.g., --phase 01 02)'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Limit number of measurements (for testing)'
    )

    args = parser.parse_args()

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run extraction
    df = extract_z24_frequencies(
        data_dir=args.data_dir,
        output_csv=args.output,
        phase_filter=args.phase,
        max_measurements=args.max,
    )

    print("\nDone!")

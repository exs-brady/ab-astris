"""
CESMD Structural Data Analysis with Ab Astris Pipeline

Validates the Ab Astris cross-domain signal detection methodology on
real-world structural response data from CESMD (Center for Engineering
Strong Motion Data).

Produces:
- cesmd_structural_results.json: Results for each structure analyzed
- cesmd_summary.png: Visualization of results
"""

import json
import numpy as np
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, asdict

# Scientific computing
from astropy.timeseries import LombScargle
from scipy import signal

# Visualization
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent
CESMD_DATA_DIR = OUTPUT_DIR / 'cesmd_data'
RESULTS_JSON = OUTPUT_DIR / 'cesmd_structural_results.json'
RESULTS_PLOT = OUTPUT_DIR / 'cesmd_summary.png'

# Analysis parameters (tuned for structural response)
FREQ_MIN = 0.05   # Hz - buildings have low fundamental frequencies
FREQ_MAX = 25.0   # Hz - upper bound for structural modes
N_FREQ_POINTS = 10000
N_BOOTSTRAP = 100
N_WINDOWS = 6

# Plot styling (dark theme)
COLORS = {
    'background': '#0d1117',
    'panel': '#161b22',
    'accent': '#00b4d8',
    'accent2': '#ff6b6b',
    'accent3': '#4ecdc4',
    'text': '#c9d1d9',
    'grid': '#21262d',
}

# Published reference frequencies for target structures
# Source: Brief document citations
PUBLISHED_FREQUENCIES = {
    'CE23287': {
        'name': 'San Bernardino 6-story Hotel',
        'published_freq_hz': 1.1,
        'source': 'CESMD dataset',
        'direction': 'transverse'
    },
    'CE23285': {
        'name': 'San Bernardino CSUSB Library (5-story)',
        'published_freq_hz': None,  # To be determined from analysis
        'source': 'CESMD dataset',
        'direction': 'transverse'
    },
    'CE23555': {
        'name': 'San Bernardino 12-story Govt Bldg',
        'published_freq_hz': None,  # Multi-channel - will analyze roof
        'source': 'CESMD dataset',
        'direction': 'transverse'
    },
    'CE24517': {
        'name': 'Lancaster 3-story Office Building',
        'published_freq_hz': None,
        'source': 'CESMD dataset',
        'direction': 'transverse'
    },
}


# ============================================================================
# V2 FORMAT PARSER
# ============================================================================

@dataclass
class V2Channel:
    """Parsed data from one channel of a V2 file."""
    station_code: str
    station_name: str
    channel_number: int
    channel_direction: str
    channel_location: str
    earthquake_date: str
    sample_rate_hz: float
    duration_s: float
    n_points: int
    peak_accel_cm_s2: float
    peak_velocity_cm_s: float
    peak_displacement_cm: float
    latitude: float
    longitude: float
    acceleration: np.ndarray  # in cm/s^2
    time: np.ndarray


def parse_v2_file(filepath: Path) -> List[V2Channel]:
    """
    Parse a CESMD/CSMIP V2 format file.

    V2 files contain corrected acceleration, velocity, and displacement data.
    Each file may contain multiple channels (typically 3 per sensor location,
    or multiple sensors throughout a building).

    Format structure:
    - Lines 1-45: Header with metadata
    - Line 46: Data format descriptor
    - Lines 47+: Fixed-width data values (8 per line, 10 chars each)
    """
    channels = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Split into channel blocks - each starts with "Corrected accelerogram"
    # Case-insensitive to handle both Ridgecrest (mixed case) and Van Nuys (uppercase)
    # Use \b word boundary to avoid matching "UNCORRECTED ACCELEROGRAM"
    channel_blocks = re.split(r'(?=\bCorrected accelerogram)', content, flags=re.IGNORECASE)
    channel_blocks = [b for b in channel_blocks if b.strip()]

    for block in channel_blocks:
        lines = block.split('\n')
        if len(lines) < 50:
            continue

        try:
            channel = parse_channel_block(lines)
            if channel is not None:
                channels.append(channel)
        except Exception as e:
            print(f"Warning: Failed to parse channel block: {e}")
            continue

    return channels


def parse_channel_block(lines: List[str]) -> Optional[V2Channel]:
    """Parse a single channel block from V2 file."""

    # Extract station code and channel from line 1
    # Format: "Corrected accelerogram   23287-K1974-19185.03       Chan  1:  90 Deg"
    line1 = lines[0] if len(lines) > 0 else ""

    # Extract channel number and direction
    # Try "Chan N: M Deg" format first (horizontal channels)
    chan_match = re.search(r'Chan\s+(\d+):\s+(\d+)\s*Deg', line1, re.IGNORECASE)
    if chan_match:
        channel_number = int(chan_match.group(1))
        channel_direction = f"{chan_match.group(2)} Deg"
    else:
        # Try "Chan N: UP" format (vertical channels)
        chan_up_match = re.search(r'Chan\s+(\d+):\s*UP', line1, re.IGNORECASE)
        if chan_up_match:
            channel_number = int(chan_up_match.group(1))
            channel_direction = "Up"
        else:
            channel_number = 1
            channel_direction = "Unknown"

    # Line 6: Station info
    # "Station No. 23287   34.065N, 117.280W    K2      s/n 1974  (9 Chns of  9 at Sta)"
    line6 = lines[5] if len(lines) > 5 else ""
    station_match = re.search(r'Station No\.\s+(\d+)\s+([\d.]+)([NS]),\s*([\d.]+)([EW])', line6, re.IGNORECASE)
    if station_match:
        station_code = f"CE{station_match.group(1)}"
        lat = float(station_match.group(2))
        if station_match.group(3) == 'S':
            lat = -lat
        lon = float(station_match.group(4))
        if station_match.group(5) == 'W':
            lon = -lon
    else:
        station_code = "Unknown"
        lat, lon = 0.0, 0.0

    # Line 7: Building name
    line7 = lines[6] if len(lines) > 6 else ""
    # Building name: strip trailing "CGS" marker and whitespace
    station_name = re.split(r'CGS', line7, flags=re.IGNORECASE)[0].strip() if re.search(r'CGS', line7, re.IGNORECASE) else line7.strip()

    # Line 8: Channel location
    line8 = lines[7] if len(lines) > 7 else ""
    location_match = re.search(r'Location:\s*(.+)', line8, re.IGNORECASE)
    channel_location = location_match.group(1).strip() if location_match else ""

    # Line 9: Date
    line9 = lines[8] if len(lines) > 8 else ""
    date_match = re.search(r'(\w+\s+\w+\s+\d+,\s+\d+)', line9)
    earthquake_date = date_match.group(1) if date_match else ""

    # Line 16-17: Number of points and sample interval
    # "  9300 points of instrument- and baseline-corrected accel..."
    # "At equally-spaced intervals of   0.010  sec."
    line16 = lines[15] if len(lines) > 15 else ""
    line17 = lines[16] if len(lines) > 16 else ""

    n_points_match = re.search(r'(\d+)\s+points', line16, re.IGNORECASE)
    n_points = int(n_points_match.group(1)) if n_points_match else 0

    dt_match = re.search(r'intervals of\s+([\d.]+)\s+sec', line17, re.IGNORECASE)
    dt = float(dt_match.group(1)) if dt_match else 0.01
    sample_rate_hz = 1.0 / dt if dt > 0 else 100.0
    duration_s = n_points * dt

    # Line 18: Peak acceleration
    line18 = lines[17] if len(lines) > 17 else ""
    peak_accel_match = re.search(r'Peak acceleration\s*=\s*([-\d.]+)', line18, re.IGNORECASE)
    peak_accel = float(peak_accel_match.group(1)) if peak_accel_match else 0.0

    # Line 19: Peak velocity
    line19 = lines[18] if len(lines) > 18 else ""
    peak_vel_match = re.search(r'Peak\s+velocity\s*=\s*([-\d.]+)', line19, re.IGNORECASE)
    peak_velocity = float(peak_vel_match.group(1)) if peak_vel_match else 0.0

    # Line 20: Peak displacement
    line20 = lines[19] if len(lines) > 19 else ""
    peak_disp_match = re.search(r'Peak displacement\s*=\s*([-\d.]+)', line20, re.IGNORECASE)
    peak_displacement = float(peak_disp_match.group(1)) if peak_disp_match else 0.0

    # Find data start line (contains "points of accel data")
    data_start_idx = None
    for i, line in enumerate(lines):
        if 'points of accel data' in line.lower():
            data_start_idx = i + 1
            break

    if data_start_idx is None or n_points == 0:
        return None

    # Parse acceleration data
    # Format: 8 values per line, fixed-width 10 characters each
    acceleration = []
    for line in lines[data_start_idx:]:
        if not line.strip():
            continue
        # Check if this line is a new header (start of next channel)
        if 'corrected' in line.lower() or 'velocity' in line.lower() or 'displacement' in line.lower():
            break

        # Parse fixed-width values
        for i in range(0, len(line), 10):
            val_str = line[i:i+10].strip()
            if val_str:
                try:
                    acceleration.append(float(val_str))
                except ValueError:
                    continue

        if len(acceleration) >= n_points:
            break

    acceleration = np.array(acceleration[:n_points])
    time = np.arange(len(acceleration)) * dt

    return V2Channel(
        station_code=station_code,
        station_name=station_name,
        channel_number=channel_number,
        channel_direction=channel_direction,
        channel_location=channel_location,
        earthquake_date=earthquake_date,
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        n_points=len(acceleration),
        peak_accel_cm_s2=peak_accel,
        peak_velocity_cm_s=peak_velocity,
        peak_displacement_cm=peak_displacement,
        latitude=lat,
        longitude=lon,
        acceleration=acceleration,
        time=time
    )


# ============================================================================
# AB ASTRIS ANALYSIS PIPELINE
# ============================================================================

@dataclass
class LSResult:
    """Results from Lomb-Scargle analysis."""
    frequency: float
    power: float
    fap: float
    frequencies: np.ndarray
    powers: np.ndarray


@dataclass
class AnalysisResult:
    """Complete analysis results for one channel."""
    station_code: str
    station_name: str
    channel_number: int
    channel_direction: str
    channel_location: str
    earthquake_date: str
    sample_rate_hz: float
    duration_s: float
    n_points: int
    pga_cm_s2: float

    # Lomb-Scargle results
    ls_frequency_hz: float
    ls_power: float
    ls_fap: float

    # Multi-window validation
    multi_window_cv: float
    detection_rate: float
    window_frequencies: List[float]

    # Bootstrap
    bootstrap_freq_mean: float
    bootstrap_freq_std: float

    # Comparison with published values
    published_frequency_hz: Optional[float]
    published_source: Optional[str]
    deviation_percent: Optional[float]

    # Confidence
    confidence_score: float
    confidence_tier: str


def run_lomb_scargle(
    time: np.ndarray,
    acceleration: np.ndarray,
    freq_min: float = FREQ_MIN,
    freq_max: float = FREQ_MAX,
    n_points: int = N_FREQ_POINTS
) -> LSResult:
    """Run Lomb-Scargle periodogram on acceleration data."""

    # Normalize
    data_mean = np.mean(acceleration)
    data_norm = acceleration - data_mean

    # Frequency grid
    frequencies = np.linspace(freq_min, freq_max, n_points)

    # Compute Lomb-Scargle
    ls = LombScargle(time, data_norm)
    powers = ls.power(frequencies)

    # Find best peak
    best_idx = np.argmax(powers)
    best_freq = frequencies[best_idx]
    best_power = powers[best_idx]

    # Compute FAP
    try:
        fap = ls.false_alarm_probability(best_power)
    except:
        fap = 1.0

    return LSResult(
        frequency=best_freq,
        power=best_power,
        fap=fap,
        frequencies=frequencies,
        powers=powers
    )


def run_multi_window_validation(
    time: np.ndarray,
    acceleration: np.ndarray,
    global_freq: float,
    n_windows: int = N_WINDOWS
) -> Tuple[float, float, List[float]]:
    """
    Run Lomb-Scargle on multiple overlapping windows.
    Returns CV, detection rate, and window frequencies.
    """
    n_samples = len(time)
    window_len = n_samples // (n_windows // 2 + 1)
    step = window_len // 2

    window_freqs = []

    for i in range(n_windows):
        start = i * step
        end = start + window_len
        if end > n_samples:
            break

        t_win = time[start:end]
        a_win = acceleration[start:end]

        result = run_lomb_scargle(t_win, a_win)
        window_freqs.append(result.frequency)

    window_freqs = np.array(window_freqs)

    # Compute CV
    freq_mean = np.mean(window_freqs)
    freq_std = np.std(window_freqs)
    cv = (freq_std / freq_mean * 100) if freq_mean > 0 else 0

    # Detection rate (within 15% of global)
    detections = np.abs(window_freqs - global_freq) / global_freq < 0.15
    detection_rate = np.mean(detections)

    return cv, detection_rate, window_freqs.tolist()


def run_bootstrap(
    time: np.ndarray,
    acceleration: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP
) -> Tuple[float, float]:
    """Bootstrap resampling for frequency uncertainty."""
    n_samples = len(time)
    bootstrap_freqs = []

    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        indices = np.sort(indices)

        t_boot = time[indices]
        a_boot = acceleration[indices]

        result = run_lomb_scargle(t_boot, a_boot)
        bootstrap_freqs.append(result.frequency)

    bootstrap_freqs = np.array(bootstrap_freqs)
    return np.mean(bootstrap_freqs), np.std(bootstrap_freqs)


def compute_confidence_score(
    ls_fap: float,
    cv: float,
    detection_rate: float,
    freq: float
) -> Tuple[float, str]:
    """
    Compute confidence score (0-100) following Ab Astris methodology.

    Components:
    - FAP score (25 points): Lower is better
    - CV score (25 points): Lower is better
    - Detection rate (25 points): Higher is better
    - Frequency range (25 points): Is it in expected structural range?
    """
    # FAP score (25 pts)
    if ls_fap < 1e-10:
        fap_score = 25
    elif ls_fap < 1e-5:
        fap_score = 20
    elif ls_fap < 0.01:
        fap_score = 15
    elif ls_fap < 0.05:
        fap_score = 10
    else:
        fap_score = 0

    # CV score (25 pts) - structural expectation: 0.5-5%
    if cv < 1:
        cv_score = 25
    elif cv < 2:
        cv_score = 22
    elif cv < 5:
        cv_score = 18
    elif cv < 10:
        cv_score = 12
    else:
        cv_score = 0

    # Detection rate score (25 pts)
    dr_score = detection_rate * 25

    # Frequency range score (25 pts)
    # Buildings: 0.1-10 Hz typical
    if 0.1 <= freq <= 10:
        range_score = 25
    elif 10 < freq <= 25:
        range_score = 15
    else:
        range_score = 5

    total = fap_score + cv_score + dr_score + range_score

    # Tier assignment
    if total >= 85:
        tier = "CONFIRMED RESONANCE"
    elif total >= 70:
        tier = "HIGH CONFIDENCE"
    elif total >= 50:
        tier = "MODERATE"
    elif total >= 30:
        tier = "LOW CONFIDENCE"
    else:
        tier = "REJECTED"

    return total, tier


def analyze_channel(channel: V2Channel) -> AnalysisResult:
    """Run full Ab Astris analysis on a single channel."""

    # 1. Lomb-Scargle
    ls_result = run_lomb_scargle(channel.time, channel.acceleration)

    # 2. Multi-window validation
    cv, detection_rate, window_freqs = run_multi_window_validation(
        channel.time, channel.acceleration, ls_result.frequency
    )

    # 3. Bootstrap
    bootstrap_mean, bootstrap_std = run_bootstrap(
        channel.time, channel.acceleration
    )

    # 4. Compute confidence
    score, tier = compute_confidence_score(
        ls_result.fap, cv, detection_rate, ls_result.frequency
    )

    # 5. Compare with published value if available
    pub_info = PUBLISHED_FREQUENCIES.get(channel.station_code, {})
    pub_freq = pub_info.get('published_freq_hz')
    pub_source = pub_info.get('source')

    if pub_freq:
        deviation = abs(ls_result.frequency - pub_freq) / pub_freq * 100
    else:
        deviation = None

    return AnalysisResult(
        station_code=channel.station_code,
        station_name=channel.station_name,
        channel_number=channel.channel_number,
        channel_direction=channel.channel_direction,
        channel_location=channel.channel_location,
        earthquake_date=channel.earthquake_date,
        sample_rate_hz=channel.sample_rate_hz,
        duration_s=channel.duration_s,
        n_points=channel.n_points,
        pga_cm_s2=abs(channel.peak_accel_cm_s2),
        ls_frequency_hz=ls_result.frequency,
        ls_power=ls_result.power,
        ls_fap=ls_result.fap,
        multi_window_cv=cv,
        detection_rate=detection_rate,
        window_frequencies=window_freqs,
        bootstrap_freq_mean=bootstrap_mean,
        bootstrap_freq_std=bootstrap_std,
        published_frequency_hz=pub_freq,
        published_source=pub_source,
        deviation_percent=deviation,
        confidence_score=score,
        confidence_tier=tier
    )


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def find_v2_files() -> List[Path]:
    """Find all V2 files in the CESMD data directory."""
    v2_files = []
    for pattern in ['*.v2', '*.V2']:
        v2_files.extend(CESMD_DATA_DIR.rglob(pattern))
    return sorted(set(v2_files))


def analyze_building(v2_file: Path) -> List[AnalysisResult]:
    """Analyze all channels from a V2 file."""
    print(f"\nAnalyzing: {v2_file.name}")

    channels = parse_v2_file(v2_file)
    print(f"  Found {len(channels)} channels")

    results = []
    for channel in channels:
        print(f"  Channel {channel.channel_number}: {channel.channel_direction} at {channel.channel_location}")
        result = analyze_channel(channel)
        results.append(result)
        print(f"    → f = {result.ls_frequency_hz:.3f} Hz, CV = {result.multi_window_cv:.2f}%, {result.confidence_tier}")

    return results


def select_best_result(results: List[AnalysisResult]) -> Optional[AnalysisResult]:
    """
    Select the best result from multiple channels.

    For buildings, prefer:
    1. Roof channels (highest response)
    2. Horizontal directions (E-W or N-S)
    3. Highest confidence score
    """
    if not results:
        return None

    # Score each result
    scored = []
    for r in results:
        # Prefer roof locations
        loc_score = 0
        loc_lower = r.channel_location.lower()
        if 'roof' in loc_lower:
            loc_score = 100
        elif 'top' in loc_lower or '12th' in loc_lower or '6th' in loc_lower or '5th' in loc_lower or '3rd' in loc_lower:
            loc_score = 80
        elif 'center' in loc_lower:
            loc_score = 50
        elif '1st' in loc_lower or 'basement' in loc_lower:
            loc_score = 10

        # Prefer horizontal (avoid vertical/Up direction)
        dir_lower = r.channel_direction.lower()
        if 'up' in dir_lower or 'vert' in dir_lower:
            dir_score = 0
        else:
            dir_score = 50

        total = loc_score + dir_score + r.confidence_score
        scored.append((total, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def create_summary_plot(results: List[AnalysisResult]):
    """Create a dark-themed summary visualization."""

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['background'])

    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Panel 1: Detected frequencies comparison
    ax1 = fig.add_subplot(gs[0, 0], facecolor=COLORS['panel'])

    structures = [r.station_name[:20] for r in results]
    detected_freqs = [r.ls_frequency_hz for r in results]
    published_freqs = [r.published_frequency_hz for r in results]

    x = np.arange(len(structures))
    width = 0.35

    bars1 = ax1.bar(x - width/2, detected_freqs, width, label='Detected', color=COLORS['accent'])
    bars2 = ax1.bar(x + width/2, [f if f else 0 for f in published_freqs], width,
                    label='Published', color=COLORS['accent2'], alpha=0.7)

    ax1.set_ylabel('Frequency (Hz)', color=COLORS['text'])
    ax1.set_title('Detected vs Published Frequencies', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(structures, rotation=45, ha='right', fontsize=8)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2, color=COLORS['grid'])

    # Panel 2: CV comparison
    ax2 = fig.add_subplot(gs[0, 1], facecolor=COLORS['panel'])

    cvs = [r.multi_window_cv for r in results]
    colors_cv = [COLORS['accent3'] if cv < 5 else COLORS['accent2'] for cv in cvs]

    ax2.bar(structures, cvs, color=colors_cv)
    ax2.axhline(y=5, color=COLORS['accent2'], linestyle='--', alpha=0.5, label='5% threshold')
    ax2.set_ylabel('CV (%)', color=COLORS['text'])
    ax2.set_title('Coefficient of Variation (Stability)', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax2.set_xticklabels(structures, rotation=45, ha='right', fontsize=8)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2, color=COLORS['grid'])

    # Panel 3: Confidence scores
    ax3 = fig.add_subplot(gs[1, 0], facecolor=COLORS['panel'])

    scores = [r.confidence_score for r in results]
    colors_score = []
    for s in scores:
        if s >= 85:
            colors_score.append(COLORS['accent3'])
        elif s >= 70:
            colors_score.append(COLORS['accent'])
        else:
            colors_score.append(COLORS['accent2'])

    ax3.barh(structures, scores, color=colors_score)
    ax3.axvline(x=85, color=COLORS['accent3'], linestyle='--', alpha=0.5, label='Confirmed (85)')
    ax3.axvline(x=70, color=COLORS['accent'], linestyle='--', alpha=0.5, label='High Conf (70)')
    ax3.set_xlabel('Confidence Score', color=COLORS['text'])
    ax3.set_title('Ab Astris Confidence Scores', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 100)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.2, color=COLORS['grid'])

    # Panel 4: Cross-domain context
    ax4 = fig.add_subplot(gs[1, 1], facecolor=COLORS['panel'])

    # Cross-domain CV reference
    domains = ['Stars', 'Bearings', 'CESMD\n(This)', 'Volcanoes', 'Crypto']
    domain_cvs = [0.005, 0.008, np.mean(cvs), 3.96, 68.0]
    domain_colors = [COLORS['accent3']] * 3 + [COLORS['accent']] + [COLORS['accent2']]

    ax4.bar(domains, domain_cvs, color=domain_colors)
    ax4.set_ylabel('CV (%)', color=COLORS['text'])
    ax4.set_title('Cross-Domain CV Comparison', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.2, color=COLORS['grid'])

    # Add annotation for CESMD result
    cesmd_cv = np.mean(cvs)
    ax4.annotate(f'{cesmd_cv:.2f}%', xy=(2, cesmd_cv), xytext=(2.3, cesmd_cv*2),
                 color=COLORS['text'], fontsize=10,
                 arrowprops=dict(arrowstyle='->', color=COLORS['text']))

    # Title
    fig.suptitle('Ab Astris CESMD Real-World Structural Validation\nRidgecrest M6.4 Earthquake (July 4, 2019)',
                 color=COLORS['text'], fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(RESULTS_PLOT, dpi=150, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()

    print(f"\nSaved plot to: {RESULTS_PLOT}")


def main():
    """Main analysis function."""
    print("=" * 60)
    print("Ab Astris CESMD Structural Data Validation")
    print("=" * 60)

    # Find all V2 files
    v2_files = find_v2_files()
    print(f"\nFound {len(v2_files)} V2 files")

    if not v2_files:
        print("No V2 files found. Please download CESMD data first.")
        return

    # Analyze each building
    all_results = []
    best_results = []

    for v2_file in v2_files:
        results = analyze_building(v2_file)
        all_results.extend(results)

        # Select best result for this building
        best = select_best_result(results)
        if best:
            best_results.append(best)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Best Result Per Building")
    print("=" * 60)

    for r in best_results:
        print(f"\n{r.station_name}")
        print(f"  Channel: {r.channel_number} ({r.channel_direction}) at {r.channel_location}")
        print(f"  Detected frequency: {r.ls_frequency_hz:.3f} ± {r.bootstrap_freq_std:.3f} Hz")
        if r.published_frequency_hz:
            print(f"  Published frequency: {r.published_frequency_hz:.2f} Hz")
            print(f"  Deviation: {r.deviation_percent:.1f}%")
        print(f"  CV: {r.multi_window_cv:.2f}%")
        print(f"  Detection rate: {r.detection_rate:.1%}")
        print(f"  Confidence: {r.confidence_score:.0f}/100 - {r.confidence_tier}")

    # Compute overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    cvs = [r.multi_window_cv for r in best_results]
    scores = [r.confidence_score for r in best_results]

    print(f"  Mean CV: {np.mean(cvs):.2f}%")
    print(f"  Mean confidence: {np.mean(scores):.1f}/100")
    print(f"  Confirmed resonance: {sum(1 for r in best_results if r.confidence_tier == 'CONFIRMED RESONANCE')}/{len(best_results)}")

    # Save results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'earthquake': 'Ridgecrest M6.4 Foreshock',
        'earthquake_date': '2019-07-04',
        'n_buildings_analyzed': len(best_results),
        'mean_cv_percent': float(np.mean(cvs)),
        'mean_confidence_score': float(np.mean(scores)),
        'buildings': []
    }

    for r in best_results:
        output['buildings'].append({
            'station_code': r.station_code,
            'station_name': r.station_name,
            'channel': f"{r.channel_number} ({r.channel_direction})",
            'location': r.channel_location,
            'sample_rate_hz': r.sample_rate_hz,
            'duration_s': r.duration_s,
            'pga_cm_s2': r.pga_cm_s2,
            'ls_frequency_hz': round(r.ls_frequency_hz, 4),
            'ls_frequency_error_hz': round(r.bootstrap_freq_std, 4),
            'ls_power': round(r.ls_power, 6),
            'ls_fap': float(r.ls_fap),
            'multi_window_cv': round(r.multi_window_cv, 3),
            'detection_rate': round(r.detection_rate, 3),
            'published_frequency_hz': r.published_frequency_hz,
            'published_source': r.published_source,
            'deviation_percent': round(r.deviation_percent, 2) if r.deviation_percent else None,
            'confidence_score': round(r.confidence_score, 1),
            'confidence_tier': r.confidence_tier
        })

    with open(RESULTS_JSON, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to: {RESULTS_JSON}")

    # Create visualization
    if best_results:
        create_summary_plot(best_results)

    return best_results


if __name__ == '__main__':
    main()

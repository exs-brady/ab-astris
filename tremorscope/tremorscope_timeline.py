"""
Tremorscope Timeline Analysis — Local Execution
================================================
Run this locally against the downloaded miniSEED day-files.
Produces JSON results + PNG plots for each volcano.

Usage:
    python tremorscope_timeline.py                    # All volcanoes
    python tremorscope_timeline.py kilauea            # One volcano
    python tremorscope_timeline.py --data-dir /path   # Custom data location

Expects data in:
    tremorscope_data/
    ├── kilauea_2018/   (HV.UWE.HHZ.YYYY-MM-DD.mseed)
    ├── msh_2004/       (UW.HSR.EHZ.YYYY-MM-DD.mseed)
    ├── pavlof_2016/    (AV.PS4A.EHZ.YYYY-MM-DD.mseed)
    └── augustine_2006/ (AV.AUH.EHZ.YYYY-MM-DD.mseed)

Output:
    tremorscope_results/
    ├── kilauea_2018_timeline.json
    ├── kilauea_2018_timeline.png
    ├── msh_2004_timeline.json
    ├── msh_2004_timeline.png
    ├── pavlof_2016_timeline.json
    ├── pavlof_2016_timeline.png
    ├── augustine_2006_timeline.json
    ├── augustine_2006_timeline.png
    └── cross_volcano_summary.json

Requires: pip install obspy astropy scipy numpy matplotlib
"""

import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import butter, filtfilt, hilbert
from obspy import read, UTCDateTime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import sys
import time as timer_module
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

TREMOR_FREQ_MIN = 0.5
TREMOR_FREQ_MAX = 10.0
EXPECTED_TREMOR_LOW = 1.0
EXPECTED_TREMOR_HIGH = 5.0
ENVELOPE_BANDPASS = (0.8, 12.0)
N_WINDOWS = 6          # Fewer windows per snapshot (60s is short)
WINDOW_OVERLAP = 0.5
SNAPSHOT_DURATION = 60  # seconds per snapshot
RSAM_THRESHOLD_MULTIPLIER = 1.5


# ═════════════════════════════════════════════════════════════════════════════
# VOLCANO CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

VOLCANOES = {
    'kilauea': {
        'name': 'Kīlauea 2018',
        'folder': 'kilauea_2018',
        'network': 'HV', 'station': 'UWE', 'channel': 'HHZ',
        'start': '2018-04-01', 'end': '2018-08-15',
        'snapshot_interval_hours': 12,
        'reference_event': '2018-05-04T22:32:00',
        'reference_event_name': 'M6.9 earthquake & fissure eruption onset',
        'key_events': {
            '2018-04-30T00:00:00': "Pu'u O'o collapse",
            '2018-05-04T22:32:00': 'M6.9 / Fissure onset',
            '2018-05-27T00:00:00': 'Fissure 8 dominant',
            '2018-06-03T00:00:00': 'First summit collapse',
        },
        'eruption_type': 'Effusive + summit collapse',
    },
    'msh': {
        'name': 'Mt St Helens 2004',
        'folder': 'msh_2004',
        'network': 'UW', 'station': 'HSR', 'channel': 'EHZ',
        'start': '2004-09-01', 'end': '2005-04-01',
        'snapshot_interval_hours': 20,
        'reference_event': '2004-10-01T12:00:00',
        'reference_event_name': 'First phreatic explosion',
        'key_events': {
            '2004-09-23T00:00:00': 'Seismic swarm onset',
            '2004-10-01T12:00:00': 'First phreatic explosion',
            '2004-10-05T00:00:00': 'Dome extrusion begins',
            '2004-10-15T00:00:00': 'Drumbeat earthquakes peak',
        },
        'eruption_type': 'Dome-building',
    },
    'pavlof': {
        'name': 'Pavlof 2016',
        'folder': 'pavlof_2016',
        'network': 'AV', 'station': 'PS4A', 'channel': 'EHZ',
        'start': '2016-02-15', 'end': '2016-04-30',
        'snapshot_interval_hours': 10,
        'reference_event': '2016-03-27T19:18:00',
        'reference_event_name': 'Eruption onset',
        'key_events': {
            '2016-03-27T19:18:00': 'Eruption onset',
            '2016-03-29T00:00:00': 'Peak explosive',
            '2016-04-15T00:00:00': 'Eruption end',
        },
        'eruption_type': 'Vulcanian',
    },
    'augustine': {
        'name': 'Augustine 2006',
        'folder': 'augustine_2006',
        'network': 'AV', 'station': 'AUH', 'channel': 'EHZ',
        'start': '2005-11-15', 'end': '2006-04-01',
        'snapshot_interval_hours': 16,
        'reference_event': '2006-01-11T06:00:00',
        'reference_event_name': 'First explosive event',
        'key_events': {
            '2006-01-02T00:00:00': 'Seismicity escalating',
            '2006-01-11T06:00:00': 'First explosions',
            '2006-01-28T00:00:00': 'Explosive → effusive',
            '2006-02-10T00:00:00': 'Dome extrusion',
        },
        'eruption_type': 'Explosive → effusive',
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def compute_envelope(signal, sr, bandpass_low=0.8, bandpass_high=12.0):
    """Bandpass filter + Hilbert envelope."""
    nyq = sr / 2.0
    low = bandpass_low / nyq
    high = min(bandpass_high / nyq, 0.99)
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return np.abs(hilbert(filtered))


def compute_rsam(signal, sr, window_seconds=10.0):
    """Real-time Seismic Amplitude Measurement."""
    ws = int(window_seconds * sr)
    n = len(signal) // ws
    return float(np.mean([np.mean(np.abs(signal[i*ws:(i+1)*ws])) for i in range(n)]))


# ═════════════════════════════════════════════════════════════════════════════
# LOMB-SCARGLE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def lomb_scargle_envelope(time_arr, envelope, freq_min=TREMOR_FREQ_MIN,
                          freq_max=TREMOR_FREQ_MAX, n_frequencies=10000):
    """LS periodogram on seismic envelope."""
    env_mean = np.mean(envelope)
    env_norm = (envelope - env_mean) / np.abs(env_mean) if env_mean != 0 else envelope - env_mean

    ls = LombScargle(time_arr, env_norm)
    frequencies = np.linspace(freq_min, freq_max, n_frequencies)
    powers = ls.power(frequencies)

    best_idx = np.argmax(powers)
    best_power = float(powers[best_idx])
    best_freq = float(frequencies[best_idx])

    # FAP
    try:
        fap = float(ls.false_alarm_probability(best_power))
    except:
        fap = 1.0

    return {
        'best_frequency': best_freq,
        'best_power': best_power,
        'best_fap': fap,
    }


def multi_window_cv(time_arr, envelope, target_frequency, n_windows=N_WINDOWS,
                    overlap=WINDOW_OVERLAP, freq_tolerance=0.30):
    """Quick multi-window CV computation."""
    total = len(envelope)
    ws = int(total / (n_windows * (1 - overlap) + overlap))
    step = int(ws * (1 - overlap))

    detected = []
    for i in range(n_windows):
        s = i * step
        e = s + ws
        if e > total:
            break
        result = lomb_scargle_envelope(
            time_arr[s:e], envelope[s:e],
            freq_min=max(TREMOR_FREQ_MIN, target_frequency * 0.3),
            freq_max=min(TREMOR_FREQ_MAX, target_frequency * 3.0),
            n_frequencies=5000
        )
        # Check if near target
        if abs(result['best_frequency'] - target_frequency) / target_frequency < freq_tolerance:
            detected.append(result['best_frequency'])

    if len(detected) >= 3:
        freqs = np.array(detected)
        return float((np.std(freqs) / np.mean(freqs)) * 100), len(detected) / n_windows
    return None, len(detected) / max(1, n_windows)


# ═════════════════════════════════════════════════════════════════════════════
# SNAPSHOT EXTRACTION FROM DAY-FILES
# ═════════════════════════════════════════════════════════════════════════════

def load_day_file(data_dir, config, date_str):
    """Load a single day-file miniSEED."""
    net, sta, cha = config['network'], config['station'], config['channel']
    filename = f"{net}.{sta}.{cha}.{date_str}.mseed"
    filepath = data_dir / config['folder'] / filename
    if not filepath.exists():
        return None
    try:
        st = read(str(filepath))
        st.merge(method=1, fill_value='interpolate')
        return st
    except Exception as e:
        print(f"    Warning: Could not read {filename}: {e}")
        return None


def extract_snapshot(stream, snapshot_time, duration=SNAPSHOT_DURATION):
    """Extract a snapshot from a stream at a given time."""
    t1 = UTCDateTime(snapshot_time)
    t2 = t1 + duration
    try:
        st = stream.copy().trim(t1, t2)
        if len(st) == 0 or st[0].stats.npts < duration * 10:
            return None, None, None
        tr = st[0]
        signal = tr.data.astype(np.float64)
        sr = tr.stats.sampling_rate
        time_arr = np.arange(len(signal)) / sr
        return time_arr, signal, sr
    except:
        return None, None, None


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TIMELINE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def analyse_volcano_timeline(key, config, data_dir, output_dir):
    """Run full timeline analysis for one volcano."""
    print(f"\n{'='*70}")
    print(f"  TIMELINE ANALYSIS: {config['name']}")
    print(f"  {config['eruption_type']}")
    print(f"  Ref event: {config['reference_event_name']}")
    print(f"{'='*70}")

    start = datetime.strptime(config['start'], '%Y-%m-%d')
    end = datetime.strptime(config['end'], '%Y-%m-%d')
    interval_hours = config['snapshot_interval_hours']
    ref_event = datetime.strptime(config['reference_event'], '%Y-%m-%d %H:%M:%S'
                                   if ' ' in config['reference_event']
                                   else '%Y-%m-%dT%H:%M:%S')

    # Build snapshot schedule
    snapshots = []
    t = start
    while t < end:
        snapshots.append(t)
        t += timedelta(hours=interval_hours)

    print(f"  Snapshots: {len(snapshots)} (every {interval_hours}h)")
    print(f"  Period: {config['start']} → {config['end']}")

    # Pre-load day files into a cache (one per day)
    day_cache = {}
    vol_dir = data_dir / config['folder']
    if not vol_dir.exists():
        print(f"  ERROR: Data directory not found: {vol_dir}")
        return None

    available_files = list(vol_dir.glob('*.mseed'))
    print(f"  Day-files available: {len(available_files)}")

    # Process snapshots
    results = []
    n_success = 0
    n_fail = 0
    t_start = timer_module.time()

    for i, snap_time in enumerate(snapshots):
        date_str = snap_time.strftime('%Y-%m-%d')
        snap_iso = snap_time.strftime('%Y-%m-%dT%H:%M:%S')

        # Progress
        if i % 25 == 0:
            elapsed = timer_module.time() - t_start
            rate = (i + 1) / max(elapsed, 0.1)
            eta = (len(snapshots) - i) / max(rate, 0.01)
            print(f"  [{i+1}/{len(snapshots)}] {snap_iso} "
                  f"({n_success} OK, {n_fail} fail, ETA {eta:.0f}s)")

        # Load day-file (cache it)
        if date_str not in day_cache:
            day_cache[date_str] = load_day_file(data_dir, config, date_str)
            # Evict old cache entries to save memory (keep last 3 days)
            if len(day_cache) > 5:
                oldest = sorted(day_cache.keys())[0]
                del day_cache[oldest]

        stream = day_cache.get(date_str)
        if stream is None:
            n_fail += 1
            results.append({
                'timestamp': snap_iso,
                'error': 'no_data',
            })
            continue

        # Extract snapshot
        time_arr, signal, sr = extract_snapshot(stream, snap_iso)
        if time_arr is None:
            n_fail += 1
            results.append({
                'timestamp': snap_iso,
                'error': 'extract_failed',
            })
            continue

        try:
            # Envelope
            envelope = compute_envelope(signal, sr,
                                        ENVELOPE_BANDPASS[0], ENVELOPE_BANDPASS[1])

            # RSAM
            rsam = compute_rsam(signal, sr)

            # Lomb-Scargle
            ls = lomb_scargle_envelope(time_arr, envelope)

            # Multi-window CV
            cv, det_rate = multi_window_cv(time_arr, envelope, ls['best_frequency'])

            # Is frequency in tremor band?
            in_band = EXPECTED_TREMOR_LOW <= ls['best_frequency'] <= EXPECTED_TREMOR_HIGH

            results.append({
                'timestamp': snap_iso,
                'ls_frequency': ls['best_frequency'],
                'ls_power': ls['best_power'],
                'ls_fap': ls['best_fap'],
                'multi_window_cv': cv,
                'detection_rate': det_rate,
                'rsam': rsam,
                'in_tremor_band': in_band,
                'signal_std': float(np.std(signal)),
            })
            n_success += 1

        except Exception as e:
            n_fail += 1
            results.append({
                'timestamp': snap_iso,
                'error': str(e)[:100],
            })

    elapsed = timer_module.time() - t_start
    print(f"\n  Complete: {n_success}/{len(snapshots)} snapshots in {elapsed:.1f}s")

    # ── DETECTION ANALYSIS ──────────────────────────────────────────────
    valid = [r for r in results if 'error' not in r]
    if not valid:
        print("  ERROR: No valid snapshots!")
        return None

    # Compute RSAM baseline (first 10% of valid snapshots)
    n_baseline = max(5, len(valid) // 10)
    baseline_rsam = np.mean([r['rsam'] for r in valid[:n_baseline]])
    rsam_threshold = baseline_rsam * RSAM_THRESHOLD_MULTIPLIER

    # Find first LS detection: FAP < 0.01, in tremor band, CV < 10%
    ls_first = None
    for r in valid:
        if (r['ls_fap'] < 0.01 and r['in_tremor_band'] and
                r.get('multi_window_cv') is not None and r['multi_window_cv'] < 10):
            ls_first = r['timestamp']
            break

    # Find first RSAM detection
    rsam_first = None
    for r in valid:
        if r['rsam'] > rsam_threshold:
            rsam_first = r['timestamp']
            break

    # Compute early warning
    early_warning_hours = None
    if ls_first and rsam_first:
        ls_dt = datetime.strptime(ls_first, '%Y-%m-%dT%H:%M:%S')
        rsam_dt = datetime.strptime(rsam_first, '%Y-%m-%dT%H:%M:%S')
        early_warning_hours = (rsam_dt - ls_dt).total_seconds() / 3600

    # Hours before reference event
    ls_hours_before = None
    rsam_hours_before = None
    if ls_first:
        ls_hours_before = (ref_event - datetime.strptime(ls_first, '%Y-%m-%dT%H:%M:%S')).total_seconds() / 3600
    if rsam_first:
        rsam_hours_before = (ref_event - datetime.strptime(rsam_first, '%Y-%m-%dT%H:%M:%S')).total_seconds() / 3600

    detection = {
        'reference_event': config['reference_event'],
        'reference_event_name': config['reference_event_name'],
        'rsam_baseline': float(baseline_rsam),
        'rsam_threshold': float(rsam_threshold),
        'ls_first_detection': ls_first,
        'ls_hours_before_event': ls_hours_before,
        'rsam_first_detection': rsam_first,
        'rsam_hours_before_event': rsam_hours_before,
        'early_warning_hours': early_warning_hours,
    }

    print(f"\n  DETECTION RESULTS:")
    print(f"  RSAM baseline: {baseline_rsam:.1f}, threshold: {rsam_threshold:.1f}")
    if ls_first:
        print(f"  LS first detection:   {ls_first} ({ls_hours_before:.1f}h before event)")
    else:
        print(f"  LS first detection:   None")
    if rsam_first:
        print(f"  RSAM first detection: {rsam_first} ({rsam_hours_before:.1f}h before event)")
    else:
        print(f"  RSAM first detection: None")
    if early_warning_hours is not None:
        if early_warning_hours > 0:
            print(f"  LS EARLY WARNING:     {early_warning_hours:.1f} hours "
                  f"({early_warning_hours/24:.1f} days) before RSAM")
        else:
            print(f"  RSAM detected first by {-early_warning_hours:.1f} hours")

    # ── SAVE JSON ───────────────────────────────────────────────────────
    json_path = output_dir / f"{config['folder']}_timeline.json"
    json_data = {
        'volcano': config['name'],
        'eruption_type': config['eruption_type'],
        'station': f"{config['network']}.{config['station']}.{config['channel']}",
        'period': {'start': config['start'], 'end': config['end']},
        'snapshot_interval_hours': interval_hours,
        'n_snapshots': len(snapshots),
        'n_valid': n_success,
        'n_failed': n_fail,
        'detection': detection,
        'snapshots': results,
        'analysis_seconds': elapsed,
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # ── PLOT ────────────────────────────────────────────────────────────
    plot_path = output_dir / f"{config['folder']}_timeline.png"
    plot_timeline(valid, detection, config, plot_path)

    return json_data


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def plot_timeline(snapshots, detection, config, output_path):
    """Generate the 4-panel timeline plot."""
    # Parse timestamps
    dts = [datetime.strptime(s['timestamp'], '%Y-%m-%dT%H:%M:%S') for s in snapshots]
    rsam = np.array([s['rsam'] for s in snapshots])
    freqs = np.array([s['ls_frequency'] for s in snapshots])
    powers = np.array([s['ls_power'] for s in snapshots])
    faps = np.array([s['ls_fap'] for s in snapshots])
    in_band = np.array([s['in_tremor_band'] for s in snapshots])
    cvs = np.array([s.get('multi_window_cv') if s.get('multi_window_cv') is not None else np.nan
                     for s in snapshots])

    # Colours
    bg = '#0d1117'
    text_c = '#c9d1d9'
    grid_c = '#21262d'
    accent = '#00b4d8'
    warn = '#ff6b35'

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), facecolor=bg,
                              gridspec_kw={'height_ratios': [1, 1, 1, 0.8]})
    fig.subplots_adjust(hspace=0.3)

    def style_ax(ax, title, ylabel):
        ax.set_facecolor(bg)
        ax.set_title(title, color=text_c, fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel(ylabel, color=text_c, fontsize=10)
        ax.tick_params(colors=text_c, labelsize=9)
        ax.grid(True, alpha=0.15, color=grid_c)
        for spine in ax.spines.values():
            spine.set_color(grid_c)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    def add_events(ax):
        for evt_time, evt_name in config.get('key_events', {}).items():
            evt_dt = datetime.strptime(evt_time, '%Y-%m-%dT%H:%M:%S')
            ax.axvline(evt_dt, color=warn, alpha=0.6, linestyle='--', linewidth=0.8)

    # Panel 1: RSAM
    ax1 = axes[0]
    ax1.fill_between(dts, rsam, alpha=0.3, color=accent)
    ax1.plot(dts, rsam, color=accent, linewidth=0.8, alpha=0.8)
    ax1.axhline(detection['rsam_threshold'], color=warn, linestyle='--',
                linewidth=1, alpha=0.7, label=f"Threshold ({RSAM_THRESHOLD_MULTIPLIER}× baseline)")
    add_events(ax1)
    style_ax(ax1, f"RSAM (Seismic Amplitude)", 'RSAM')
    ax1.legend(fontsize=8, facecolor=bg, edgecolor=grid_c, labelcolor=text_c)

    # Panel 2: LS Frequency (colour by in-band)
    ax2 = axes[1]
    colours = [accent if ib else '#666666' for ib in in_band]
    ax2.scatter(dts, freqs, c=colours, s=8, alpha=0.7)
    ax2.axhspan(EXPECTED_TREMOR_LOW, EXPECTED_TREMOR_HIGH, alpha=0.08, color=accent)
    ax2.axhline(EXPECTED_TREMOR_LOW, color=accent, alpha=0.3, linestyle=':')
    ax2.axhline(EXPECTED_TREMOR_HIGH, color=accent, alpha=0.3, linestyle=':')
    add_events(ax2)
    style_ax(ax2, 'Lomb-Scargle Dominant Frequency', 'Frequency (Hz)')

    # Panel 3: LS Power
    ax3 = axes[2]
    ax3.scatter(dts, powers, c=colours, s=8, alpha=0.7)
    add_events(ax3)
    style_ax(ax3, 'Lomb-Scargle Power (Signal Strength)', 'LS Power')

    # Panel 4: Detection summary
    ax4 = axes[3]
    ax4.set_facecolor(bg)
    ax4.axis('off')

    lines = [
        f"{'='*80}",
        f"  EARLY DETECTION COMPARISON: Lomb-Scargle vs RSAM",
        f"  Volcano: {config['name']} | Type: {config['eruption_type']}",
        f"{'='*80}",
    ]
    if detection.get('ls_first_detection'):
        lines.append(f"  LS first detection:   {detection['ls_first_detection'][:19]}"
                     f"  ({detection.get('ls_hours_before_event', 0):.1f}h before event)")
    else:
        lines.append(f"  LS first detection:   Not detected")
    if detection.get('rsam_first_detection'):
        lines.append(f"  RSAM first detection: {detection['rsam_first_detection'][:19]}"
                     f"  ({detection.get('rsam_hours_before_event', 0):.1f}h before event)")
    else:
        lines.append(f"  RSAM first detection: Not detected")
    if detection.get('early_warning_hours') is not None:
        h = detection['early_warning_hours']
        if h > 0:
            lines.append(f"  ★ LS EARLY WARNING:   {h:.1f} hours ({h/24:.1f} days) before RSAM")
        else:
            lines.append(f"  RSAM detected first by {-h:.1f} hours")
    lines.append(f"{'='*80}")
    lines.append(f"  Reference event: {detection['reference_event_name']} "
                 f"({detection['reference_event'][:19]})")
    lines.append(f"  RSAM baseline: {detection['rsam_baseline']:.1f} | "
                 f"Threshold: {detection['rsam_threshold']:.1f} "
                 f"({RSAM_THRESHOLD_MULTIPLIER}× baseline)")

    summary_text = '\n'.join(lines)
    ax4.text(0.02, 0.95, summary_text, transform=ax4.transAxes, va='top',
             color=text_c, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#0a0a0a',
                      edgecolor=grid_c))

    fig.suptitle(f'Tremorscope → {config["name"]}: LS Frequency Tracking vs RSAM',
                 color=text_c, fontsize=16, fontweight='bold', y=0.99)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=bg)
    plt.close()
    print(f"  Plot saved: {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-VOLCANO SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def generate_cross_volcano_summary(all_results, output_dir):
    """Generate summary JSON comparing all volcanoes."""
    summary = {}
    for key, data in all_results.items():
        if data is None:
            continue
        d = data['detection']
        valid = [s for s in data['snapshots'] if 'error' not in s]
        cvs = [s['multi_window_cv'] for s in valid
               if s.get('multi_window_cv') is not None]

        summary[key] = {
            'volcano': data['volcano'],
            'eruption_type': data['eruption_type'],
            'n_snapshots': data['n_valid'],
            'ls_first_detection': d.get('ls_first_detection'),
            'ls_hours_before_event': d.get('ls_hours_before_event'),
            'rsam_first_detection': d.get('rsam_first_detection'),
            'rsam_hours_before_event': d.get('rsam_hours_before_event'),
            'early_warning_hours': d.get('early_warning_hours'),
            'cv_mean': float(np.mean(cvs)) if cvs else None,
            'cv_range': (float(min(cvs)), float(max(cvs))) if cvs else None,
            'mean_detection_rate': float(np.mean([
                s['detection_rate'] for s in valid if s.get('detection_rate') is not None
            ])) if valid else None,
        }

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  CROSS-VOLCANO EARLY DETECTION COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Volcano':<22} {'LS Detect':>15} {'RSAM Detect':>15} "
          f"{'Early Warning':>15} {'CV Mean':>10}")
    print(f"  {'-'*22} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")

    for key, s in summary.items():
        ls_h = f"{s['ls_hours_before_event']:.0f}h" if s.get('ls_hours_before_event') else "N/A"
        rsam_h = f"{s['rsam_hours_before_event']:.0f}h" if s.get('rsam_hours_before_event') else "N/A"
        ew = f"{s['early_warning_hours']:.0f}h" if s.get('early_warning_hours') else "N/A"
        cv = f"{s['cv_mean']:.2f}%" if s.get('cv_mean') else "N/A"
        print(f"  {s['volcano']:<22} {ls_h:>15} {rsam_h:>15} {ew:>15} {cv:>10}")

    json_path = output_dir / 'cross_volcano_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {json_path}")

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]

    # Parse data dir
    data_dir = Path('tremorscope_data')
    for i, a in enumerate(args):
        if a == '--data-dir' and i + 1 < len(args):
            data_dir = Path(args[i + 1])
            args = args[:i] + args[i+2:]
            break

    output_dir = Path('tremorscope_results')
    output_dir.mkdir(exist_ok=True)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print(f"Run tremorscope_download.py first, or use --data-dir /path/to/data")
        sys.exit(1)

    # Which volcanoes?
    if args:
        targets = {}
        for arg in args:
            arg = arg.lower()
            for key in VOLCANOES:
                if arg in key or key in arg:
                    targets[key] = VOLCANOES[key]
                    break
    else:
        targets = VOLCANOES

    print(f"{'='*70}")
    print(f"  TREMORSCOPE TIMELINE ANALYSIS")
    print(f"  Data: {data_dir}")
    print(f"  Volcanoes: {', '.join(v['name'] for v in targets.values())}")
    print(f"{'='*70}")

    all_results = {}
    for key, config in targets.items():
        result = analyse_volcano_timeline(key, config, data_dir, output_dir)
        all_results[key] = result

    # Cross-volcano summary
    if len(all_results) > 1:
        generate_cross_volcano_summary(all_results, output_dir)

    print(f"\n  All outputs in: {output_dir}/")


if __name__ == '__main__':
    main()

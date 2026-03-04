"""
Conventional envelope analysis on CWRU bearing data.
Compares against Ab Astris generic pipeline results.

Three-way comparison:
  1. Ab Astris (existing results) — generic bandpass (50-500 Hz) + Lomb-Scargle
  2. Envelope + FFT — resonance bandpass (1-5 kHz) + FFT
  3. Envelope + LS  — resonance bandpass (1-5 kHz) + Lomb-Scargle

Usage:
    python bearing/envelope_comparison.py
"""

import json
import os
import urllib.request
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from scipy.fft import fft, fftfreq
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLING_RATE = 12_000  # Hz

# CWRU 6205-2RS JEM SKF bearing geometry
BEARING_GEOMETRY = {
    "ball_diameter_mm": 7.94,
    "pitch_diameter_mm": 39.04,
    "n_balls": 9,
    "contact_angle_deg": 0,
}

# Characteristic defect frequency multipliers (per 1 Hz shaft speed)
d, D, n, alpha = 7.94, 39.04, 9, 0.0
BPFI_MULT = n / 2 * (1 + d / D * np.cos(alpha))  # 5.4152
BPFO_MULT = n / 2 * (1 - d / D * np.cos(alpha))  # 3.5848
BSF_MULT = D / (2 * d) * (1 - (d / D * np.cos(alpha)) ** 2)  # 2.3573
FTF_MULT = 0.5 * (1 - d / D * np.cos(alpha))  # 0.3983

# Conditions: (label, fault_type, target_multiplier, rpm, file_number)
CONDITIONS = [
    ("Normal",     "Shaft", None,      1797, 97),
    ("IR 0.007\"", "BPFI",  BPFI_MULT, 1797, 105),
    ("IR 0.014\"", "BPFI",  BPFI_MULT, 1796, 169),
    ("IR 0.021\"", "BPFI",  BPFI_MULT, 1797, 209),
    ("OR 0.007\"", "BPFO",  BPFO_MULT, 1796, 130),
    ("OR 0.014\"", "BPFO",  BPFO_MULT, 1796, 197),
    ("OR 0.021\"", "BPFO",  BPFO_MULT, 1796, 234),
    ("Ball 0.007\"", "BSF", BSF_MULT,  1796, 118),
]

BASE_URL = "https://engineering.case.edu/sites/default/files"

# Envelope analysis parameters
BANDPASS_LOW = 1000   # Hz — lower bound of structural resonance band
BANDPASS_HIGH = 5000  # Hz — upper bound
BANDPASS_ORDER = 4
PEAK_TOLERANCE_PCT = 5.0   # ±5% search window around target
SNR_THRESHOLD_DB = 10.0    # ~3.16× amplitude ratio

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_cwru_data(data_dir: str | Path) -> dict[str, Path]:
    """Download CWRU .mat files if not already present.

    Returns dict mapping file_number -> local path.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for _, _, _, _, file_num in CONDITIONS:
        local_path = data_dir / f"{file_num}.mat"
        if not local_path.exists():
            url = f"{BASE_URL}/{file_num}.mat"
            print(f"  Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                print(f"  Please download manually from the CWRU Bearing Data Center")
                continue
        paths[file_num] = local_path
    return paths


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_condition(mat_path: str | Path) -> np.ndarray:
    """Load drive-end accelerometer signal from a CWRU .mat file.

    Auto-detects the key containing '_DE_time'.
    """
    data = loadmat(str(mat_path))
    de_keys = [k for k in data.keys() if "_DE_time" in k]
    if not de_keys:
        raise KeyError(
            f"No '*_DE_time' key found in {mat_path}. "
            f"Available keys: {[k for k in data.keys() if not k.startswith('__')]}"
        )
    signal = data[de_keys[0]].flatten()
    return signal.astype(np.float64)


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = fs / 2
    b, a = butter(order, [f_low / nyq, f_high / nyq], btype="band")
    return filtfilt(b, a, signal)


def extract_envelope(signal: np.ndarray) -> np.ndarray:
    """Compute Hilbert envelope with DC removed."""
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    envelope = envelope - np.mean(envelope)
    return envelope


def compute_fft_spectrum(
    signal: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided FFT amplitude spectrum.

    Returns (frequencies, amplitudes).
    """
    n = len(signal)
    freqs = fftfreq(n, 1 / fs)[: n // 2]
    amplitudes = 2.0 / n * np.abs(fft(signal))[: n // 2]
    return freqs, amplitudes


def compute_ls_spectrum(
    signal: np.ndarray, fs: float, max_freq: float = 500.0, oversample: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lomb-Scargle periodogram of a uniformly sampled signal.

    Returns (frequencies, power).
    """
    t = np.arange(len(signal)) / fs
    freq_grid = np.linspace(1.0, max_freq, int(max_freq * oversample))
    ls = LombScargle(t, signal)
    power = ls.power(freq_grid)
    return freq_grid, power


# ---------------------------------------------------------------------------
# Peak identification
# ---------------------------------------------------------------------------

def find_peak_near(
    freqs: np.ndarray,
    power: np.ndarray,
    target_freq: float,
    tolerance_pct: float = PEAK_TOLERANCE_PCT,
) -> dict:
    """Search for spectral peak within ±tolerance_pct of target_freq.

    Returns dict with detected_freq_hz, freq_error_pct, peak_snr_db, detected.
    """
    low = target_freq * (1 - tolerance_pct / 100)
    high = target_freq * (1 + tolerance_pct / 100)
    mask = (freqs >= low) & (freqs <= high)

    if not np.any(mask):
        return {
            "detected_freq_hz": None,
            "freq_error_pct": None,
            "peak_snr_db": None,
            "detected": False,
        }

    peak_idx_local = np.argmax(power[mask])
    peak_freq = freqs[mask][peak_idx_local]
    peak_amp = power[mask][peak_idx_local]

    # SNR: peak vs median in 10–500 Hz band
    noise_mask = (freqs >= 10) & (freqs <= 500)
    if np.any(noise_mask):
        noise_floor = np.median(power[noise_mask])
    else:
        noise_floor = np.median(power)

    if noise_floor > 0:
        snr_db = 20 * np.log10(peak_amp / noise_floor)
    else:
        snr_db = float("inf")

    freq_error_pct = abs(peak_freq - target_freq) / target_freq * 100

    detected = snr_db >= SNR_THRESHOLD_DB and freq_error_pct <= tolerance_pct

    return {
        "detected_freq_hz": round(float(peak_freq), 3),
        "freq_error_pct": round(float(freq_error_pct), 3),
        "peak_snr_db": round(float(snr_db), 1),
        "detected": bool(detected),
    }


# ---------------------------------------------------------------------------
# Per-condition analysis
# ---------------------------------------------------------------------------

def analyse_condition(
    signal: np.ndarray,
    fs: float,
    target_freq: float,
) -> dict:
    """Run envelope FFT and envelope LS on one condition.

    Returns dict with envelope_fft and envelope_ls results.
    """
    # Bandpass filter in structural resonance band
    filtered = bandpass_filter(signal, fs, BANDPASS_LOW, BANDPASS_HIGH)

    # Envelope
    envelope = extract_envelope(filtered)

    # FFT of envelope
    fft_freqs, fft_amps = compute_fft_spectrum(envelope, fs)
    fft_result = find_peak_near(fft_freqs, fft_amps, target_freq)

    # Lomb-Scargle of envelope
    ls_freqs, ls_power = compute_ls_spectrum(envelope, fs, max_freq=500.0)
    ls_result = find_peak_near(ls_freqs, ls_power, target_freq)

    return {
        "envelope_fft": fft_result,
        "envelope_ls": ls_result,
        # Keep raw spectra for plotting
        "_fft_freqs": fft_freqs,
        "_fft_amps": fft_amps,
        "_ls_freqs": ls_freqs,
        "_ls_power": ls_power,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_condition(
    condition_label: str,
    target_freq: float,
    ab_astris: dict,
    fft_freqs: np.ndarray,
    fft_amps: np.ndarray,
    ls_freqs: np.ndarray,
    ls_power: np.ndarray,
    plot_dir: Path,
):
    """Generate 2-panel comparison figure for one condition."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"Envelope Analysis — {condition_label}", fontsize=14)

    xlim = (0, min(500, target_freq * 4))

    # Panel A: Envelope FFT
    ax = axes[0]
    fft_mask = (fft_freqs >= xlim[0]) & (fft_freqs <= xlim[1])
    ax.plot(fft_freqs[fft_mask], fft_amps[fft_mask], linewidth=0.5, color="steelblue")
    ax.axvline(target_freq, color="red", linestyle="--", linewidth=1, label=f"Target: {target_freq:.1f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Envelope + FFT (1–5 kHz bandpass)")
    ax.legend(fontsize=9)
    ax.set_xlim(xlim)

    # Panel B: Envelope LS
    ax = axes[1]
    ls_mask = (ls_freqs >= xlim[0]) & (ls_freqs <= xlim[1])
    ax.plot(ls_freqs[ls_mask], ls_power[ls_mask], linewidth=0.5, color="darkorange")
    ax.axvline(target_freq, color="red", linestyle="--", linewidth=1, label=f"Target: {target_freq:.1f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("LS Power")
    ax.set_title("Envelope + Lomb-Scargle (1–5 kHz bandpass)")
    ax.legend(fontsize=9)
    ax.set_xlim(xlim)

    # Safe filename
    safe_name = condition_label.replace('"', '').replace(' ', '_').replace('.', '')
    out_path = plot_dir / f"envelope_comparison_{safe_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_summary(results: dict, plot_dir: Path):
    """Generate summary comparison figure: detection rate bar chart + error scatter."""
    conditions = results["conditions"]

    methods = ["ab_astris", "envelope_fft", "envelope_ls"]
    method_labels = ["Ab Astris\n(generic)", "Envelope + FFT\n(domain-specific)", "Envelope + LS\n(hybrid)"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # Detection counts
    counts = []
    for m in methods:
        counts.append(sum(1 for c in conditions if c.get(m, {}).get("detected", False)))

    # Frequency errors (only where detected)
    errors = {m: [] for m in methods}
    labels_for_scatter = []
    for c in conditions:
        labels_for_scatter.append(c["condition"])
        for m in methods:
            entry = c.get(m, {})
            if entry.get("detected") and entry.get("freq_error_pct") is not None:
                errors[m].append(entry["freq_error_pct"])
            else:
                errors[m].append(None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Left: detection rate bar chart
    ax = axes[0]
    x = np.arange(len(methods))
    bars = ax.bar(x, counts, color=colors, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel("Conditions Detected (out of 8)")
    ax.set_ylim(0, 9)
    ax.set_title("Detection Rate by Method")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(count), ha="center", fontsize=12, fontweight="bold")

    # Right: frequency error scatter
    ax = axes[1]
    x_positions = np.arange(len(conditions))
    width = 0.25
    for i, (m, label, color) in enumerate(zip(methods, ["Ab Astris", "Env+FFT", "Env+LS"], colors)):
        vals = errors[m]
        plot_x = []
        plot_y = []
        for j, v in enumerate(vals):
            if v is not None:
                plot_x.append(x_positions[j] + (i - 1) * width)
                plot_y.append(v)
        ax.scatter(plot_x, plot_y, color=color, label=label, s=50, zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([c["condition"] for c in conditions], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frequency Error (%)")
    ax.set_title("Frequency Error per Condition")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    out_path = plot_dir / "envelope_comparison_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    data_dir: str | Path | None = None,
    plot_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Run the full three-way comparison on all 8 CWRU conditions."""
    base_dir = Path(__file__).resolve().parent
    if data_dir is None:
        data_dir = base_dir / "data"
    if plot_dir is None:
        plot_dir = base_dir / "plots"
    if output_path is None:
        output_path = base_dir / "envelope_comparison_results.json"

    data_dir = Path(data_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load Ab Astris results
    ab_astris_path = base_dir / "bearing_experiment_results.json"
    with open(ab_astris_path) as f:
        ab_astris_all = json.load(f)

    # Build lookup by condition name (JSON is a plain list)
    ab_lookup = {}
    for entry in ab_astris_all:
        freq_err = entry.get("freq_error_pct")
        det_rate = entry.get("detection_rate", 0)
        # Normal condition has freq_error_pct=null but detection_rate=1.0
        is_detected = det_rate >= 0.75 and (
            freq_err is None or freq_err < 5.0
        )
        # If freq_error_pct is huge (>5%), it detected the wrong frequency
        if freq_err is not None and freq_err >= 5.0:
            is_detected = False
        ab_lookup[entry["condition"]] = {
            "detected_freq_hz": entry.get("detected_freq"),
            "freq_error_pct": freq_err,
            "cv_pct": entry.get("multi_window_cv_pct"),
            "detected": is_detected,
        }

    # Download data
    print("Checking / downloading CWRU data ...")
    file_paths = download_cwru_data(data_dir)

    # Process each condition
    print("\nRunning envelope analysis ...")
    condition_results = []

    for label, fault_type, multiplier, rpm, file_num in CONDITIONS:
        print(f"\n--- {label} ({fault_type}) ---")

        # Target frequency
        shaft_freq = rpm / 60.0
        if multiplier is not None:
            target_freq = shaft_freq * multiplier
        else:
            target_freq = shaft_freq  # Normal → shaft frequency

        # Load signal
        mat_path = file_paths.get(file_num)
        if mat_path is None or not Path(mat_path).exists():
            print(f"  SKIP: data file {file_num}.mat not available")
            condition_results.append({
                "condition": label,
                "fault_type": fault_type,
                "target_freq_hz": round(target_freq, 3),
                "ab_astris": ab_lookup.get(label, {}),
                "envelope_fft": {"detected": None, "error": "data not available"},
                "envelope_ls": {"detected": None, "error": "data not available"},
            })
            continue

        signal = load_condition(mat_path)
        print(f"  Signal: {len(signal)} samples, {len(signal)/SAMPLING_RATE:.1f}s")
        print(f"  Target: {target_freq:.3f} Hz ({fault_type})")

        # Run analysis
        result = analyse_condition(signal, SAMPLING_RATE, target_freq)

        # Extract Ab Astris result
        ab_result = ab_lookup.get(label, {})

        # Print comparison
        print(f"  Ab Astris:    detected={ab_result.get('detected')}, "
              f"freq={ab_result.get('detected_freq_hz')}, "
              f"error={ab_result.get('freq_error_pct')}%")
        print(f"  Envelope FFT: detected={result['envelope_fft']['detected']}, "
              f"freq={result['envelope_fft']['detected_freq_hz']}, "
              f"error={result['envelope_fft']['freq_error_pct']}%, "
              f"SNR={result['envelope_fft']['peak_snr_db']} dB")
        print(f"  Envelope LS:  detected={result['envelope_ls']['detected']}, "
              f"freq={result['envelope_ls']['detected_freq_hz']}, "
              f"error={result['envelope_ls']['freq_error_pct']}%, "
              f"SNR={result['envelope_ls']['peak_snr_db']} dB")

        # Plot
        plot_condition(
            label, target_freq, ab_result,
            result["_fft_freqs"], result["_fft_amps"],
            result["_ls_freqs"], result["_ls_power"],
            plot_dir,
        )

        # Store (without raw spectra)
        condition_results.append({
            "condition": label,
            "fault_type": fault_type,
            "target_freq_hz": round(target_freq, 3),
            "ab_astris": ab_result,
            "envelope_fft": result["envelope_fft"],
            "envelope_ls": result["envelope_ls"],
        })

    # Summary statistics
    n_ab = sum(1 for c in condition_results if c["ab_astris"].get("detected"))
    n_fft = sum(1 for c in condition_results if c["envelope_fft"].get("detected"))
    n_ls = sum(1 for c in condition_results if c["envelope_ls"].get("detected"))

    def mean_error(method_key):
        errs = [
            c[method_key]["freq_error_pct"]
            for c in condition_results
            if c[method_key].get("detected") and c[method_key].get("freq_error_pct") is not None
        ]
        return round(float(np.mean(errs)), 3) if errs else None

    output = {
        "comparison_date": str(date.today()),
        "purpose": "Head-to-head: Ab Astris generic vs conventional envelope analysis on CWRU bearing data",
        "cwru_bearing_geometry": BEARING_GEOMETRY,
        "envelope_parameters": {
            "bandpass_hz": [BANDPASS_LOW, BANDPASS_HIGH],
            "bandpass_order": BANDPASS_ORDER,
            "peak_tolerance_pct": PEAK_TOLERANCE_PCT,
            "snr_threshold_db": SNR_THRESHOLD_DB,
            "sampling_rate_hz": SAMPLING_RATE,
        },
        "conditions": condition_results,
        "summary": {
            "ab_astris_detected": n_ab,
            "envelope_fft_detected": n_fft,
            "envelope_ls_detected": n_ls,
            "ab_astris_mean_error_pct": mean_error("ab_astris"),
            "envelope_fft_mean_error_pct": mean_error("envelope_fft"),
            "envelope_ls_mean_error_pct": mean_error("envelope_ls"),
        },
    }

    # Summary plot
    plot_summary(output, plot_dir)

    # Save results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Ab Astris detected:    {n_ab}/8")
    print(f"  Envelope + FFT:        {n_fft}/8")
    print(f"  Envelope + LS:         {n_ls}/8")
    print(f"  Ab Astris mean error:  {mean_error('ab_astris')}%")
    print(f"  Envelope FFT mean err: {mean_error('envelope_fft')}%")
    print(f"  Envelope LS mean err:  {mean_error('envelope_ls')}%")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    run_comparison()


if __name__ == "__main__":
    main()

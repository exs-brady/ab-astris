"""
Ab Astris Negative Control: Cryptocurrency Market Data

Demonstrates that the CV discriminator correctly rejects signals from a domain
without physics-constrained periodic forcing. Crypto prices are pure market
sentiment — no deterministic physics.

Pipeline:
1. No bandpass filter (raw daily closing prices)
2. No Hilbert envelope (prices are direct signal)
3. Lomb-Scargle periodogram
4. Multi-window validation (CV)
5. Bootstrap error estimation
6. Confidence scoring (0-100)

Expected outcome: CV >> 30%, detection rate < 0.5, tier = NOISE.

Usage:
    python negative_controls/crypto_experiment.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULTS_JSON = OUTPUT_DIR / "crypto_results.json"

# Assets to analyse
CRYPTO_ASSETS = {
    "BTC": {"symbol": "BTC-USD", "name": "Bitcoin"},
    "ETH": {"symbol": "ETH-USD", "name": "Ethereum"},
}

DEFAULT_YEAR = 2023

# Sampling: daily closing prices
SAMPLING_INTERVAL_SEC = 86400  # 1 day

# Frequency range: 1/365 cpd to 0.5 cpd (Nyquist for daily)
FREQ_MIN_CPD = 1.0 / 365.0
FREQ_MAX_CPD = 0.5
FREQ_MIN_HZ = FREQ_MIN_CPD / 86400
FREQ_MAX_HZ = FREQ_MAX_CPD / 86400

# Multi-window parameters
N_WINDOWS = 8
WINDOW_DURATION_DAYS = 60  # ~2 months per window (need enough data per window)
FREQ_TOLERANCE = 0.05  # 5% tolerance for detection rate

# Bootstrap
N_BOOTSTRAP = 100

# LS grid resolution
N_FREQ_POINTS = 10000

# Confidence scoring thresholds
CV_EXCELLENT = 0.01
CV_GOOD = 0.1
CV_FAIR = 1.0

# Dark theme plot styling (matching other Ab Astris modules)
COLORS = {
    "background": "#0d1117",
    "panel": "#161b22",
    "accent": "#00b4d8",
    "accent2": "#ff6b6b",
    "accent3": "#4ecdc4",
    "text": "#c9d1d9",
    "grid": "#21262d",
    "crypto": "#ff6b6b",
}

# Cross-domain reference values
CROSS_DOMAIN_REFS = {
    "variable_stars": {"cv_mean": 0.005, "detection_rate": 1.00, "domain": "Astronomy"},
    "bearings": {"cv_mean": 0.008, "detection_rate": 1.00, "domain": "Industrial"},
    "tides": {"cv_mean": 0.171, "detection_rate": 1.00, "domain": "Oceanography"},
    "volcanoes": {"cv_mean": 3.96, "detection_rate": 0.997, "domain": "Geophysics"},
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LSResult:
    """Results from Lomb-Scargle periodogram analysis."""
    frequency_hz: float
    frequency_cpd: float
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
class SignalResult:
    """Complete analysis results for one crypto asset."""
    asset_name: str
    symbol: str

    # Lomb-Scargle
    ls_frequency_hz: float
    ls_frequency_cpd: float
    ls_period_days: float
    ls_power: float
    ls_fap: float

    # Multi-window
    multi_window_cv: float
    multi_window_cv_clean: float
    detection_rate: float

    # Bootstrap
    bootstrap_freq_mean: float
    bootstrap_freq_std: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    # Scoring
    confidence_score: float
    confidence_tier: str


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_crypto_data(
    symbol: str,
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    data_dir: Path = None,
) -> pd.DataFrame:
    """Download crypto data using yfinance with caching."""
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    safe_name = symbol.lower().replace("-", "_")
    year = start_date[:4]
    cache_file = data_dir / f"{safe_name}_{year}.csv"

    if cache_file.exists():
        print(f"  Loading cached: {cache_file.name}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    print(f"  Downloading {symbol} ({start_date} to {end_date})...")
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {symbol}")

    # Flatten MultiIndex columns if present (yfinance 0.2.x+ returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_file)
    print(f"  Cached to: {cache_file.name} ({len(df)} records)")
    return df


def prepare_crypto_time_series(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert yfinance DataFrame to (time_seconds, prices)."""
    df_clean = df[["Close"]].dropna()
    if df_clean.empty:
        raise ValueError("No valid closing prices found")

    time_sec = (df_clean.index - df_clean.index[0]).total_seconds().values.astype(np.float64)
    prices = df_clean["Close"].values.astype(np.float64)

    # Validate
    assert (prices > 0).all(), "Found non-positive prices"
    return time_sec, prices


# ============================================================================
# AB ASTRIS CORE PIPELINE (copied from oceanography/tide_gauge_experiment.py)
# ============================================================================

def run_lomb_scargle(
    time: np.ndarray,
    signal: np.ndarray,
    freq_min_hz: float = FREQ_MIN_HZ,
    freq_max_hz: float = FREQ_MAX_HZ,
    n_points: int = N_FREQ_POINTS,
) -> LSResult:
    """Run Lomb-Scargle periodogram following Ab Astris methodology."""
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
        frequency_cpd=best_freq_hz * 86400,
        power=best_power,
        fap=fap,
        frequencies_hz=frequencies_hz,
        powers=powers,
    )


def create_windows(
    time: np.ndarray,
    signal: np.ndarray,
    n_windows: int = N_WINDOWS,
    window_duration_sec: float = WINDOW_DURATION_DAYS * 86400,
) -> List[Dict]:
    """Create overlapping windows for multi-window validation."""
    total_duration = time[-1] - time[0]
    if total_duration < window_duration_sec:
        raise ValueError(
            f"Signal duration ({total_duration/86400:.0f} days) shorter than "
            f"window duration ({window_duration_sec/86400:.0f} days)"
        )

    step = (total_duration - window_duration_sec) / max(n_windows - 1, 1)

    windows = []
    for i in range(n_windows):
        start = time[0] + i * step
        end = start + window_duration_sec
        mask = (time >= start) & (time <= end)
        if np.sum(mask) < 10:  # Need at least 10 data points
            continue
        windows.append({
            "time": time[mask] - time[mask][0],
            "signal": signal[mask],
        })

    return windows


def run_multi_window_validation(
    windows: List[Dict],
    target_freq_hz: float,
    freq_min_hz: float = FREQ_MIN_HZ,
    freq_max_hz: float = FREQ_MAX_HZ,
) -> MultiWindowResult:
    """Run Lomb-Scargle on each window. Compute CV across windows."""
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
    freq_min_hz: float = FREQ_MIN_HZ,
    freq_max_hz: float = FREQ_MAX_HZ,
) -> BootstrapResult:
    """Bootstrap resampling for frequency uncertainty estimation."""
    n_samples = len(time)
    bootstrap_freqs = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        indices = np.sort(indices)

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
    target_freq_hz: float = None,
) -> Tuple[float, str]:
    """Compute confidence score (0-100) following Ab Astris framework.

    For exploratory analysis (no target), frequency accuracy uses self-consistency.
    """
    score = 0.0

    # 1. Frequency accuracy (30 pts)
    if target_freq_hz is not None:
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
    else:
        # Exploratory: award based on bootstrap self-consistency
        # (no target = no accuracy points, rely on other metrics)
        score += 0

    # 2. FAP significance (25 pts)
    fap = ls_result.fap
    if fap < 1e-10:
        score += 25
    elif fap < 1e-5:
        score += 20
    elif fap < 1e-3:
        score += 15
    elif fap < 0.01:
        score += 10
    elif fap < 0.05:
        score += 5

    # 3. CV stability (25 pts)
    cv = mw_result.cv_clean
    if cv < CV_EXCELLENT:
        score += 25
    elif cv < CV_GOOD:
        score += 20
    elif cv < CV_FAIR:
        score += 15
    elif cv < 5.0:
        score += 10
    elif cv < 10.0:
        score += 5

    # 4. Detection rate (20 pts)
    score += 20.0 * mw_result.detection_rate

    # Tier
    if score >= 90:
        tier = "CONFIRMED"
    elif score >= 75:
        tier = "PROBABLE"
    elif score >= 60:
        tier = "CANDIDATE"
    else:
        tier = "NOISE"

    return round(score, 1), tier


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_crypto_asset(
    time: np.ndarray,
    prices: np.ndarray,
    asset_name: str,
    symbol: str,
) -> SignalResult:
    """Run full Ab Astris pipeline on one crypto asset."""
    print(f"\n  === {asset_name} ({symbol}) ===")

    # 1. Lomb-Scargle (no filtering — raw prices)
    print("  Running Lomb-Scargle...")
    ls_result = run_lomb_scargle(time, prices)
    period_days = 1.0 / ls_result.frequency_cpd if ls_result.frequency_cpd > 0 else float("inf")
    print(f"    Peak: {ls_result.frequency_cpd:.4f} cpd ({period_days:.1f} days)")
    print(f"    Power: {ls_result.power:.4f}, FAP: {ls_result.fap:.2e}")

    # 2. Multi-window validation
    print("  Running multi-window validation...")
    windows = create_windows(time, prices)
    mw_result = run_multi_window_validation(
        windows, ls_result.frequency_hz,
    )
    print(f"    CV (clean): {mw_result.cv_clean:.2f}%")
    print(f"    Detection rate: {mw_result.detection_rate:.2f}")

    # 3. Bootstrap
    print("  Running bootstrap (n=100)...")
    boot_result = run_bootstrap(time, prices)
    print(f"    Freq mean: {boot_result.freq_mean*86400:.4f} cpd")
    print(f"    95% CI: [{boot_result.freq_ci_low*86400:.4f}, {boot_result.freq_ci_high*86400:.4f}] cpd")

    # 4. Confidence scoring (exploratory — no target)
    score, tier = compute_confidence_score(ls_result, mw_result, target_freq_hz=None)
    print(f"    Confidence: {score}/100 → {tier}")

    return SignalResult(
        asset_name=asset_name,
        symbol=symbol,
        ls_frequency_hz=ls_result.frequency_hz,
        ls_frequency_cpd=ls_result.frequency_cpd,
        ls_period_days=period_days,
        ls_power=ls_result.power,
        ls_fap=ls_result.fap,
        multi_window_cv=mw_result.cv_raw,
        multi_window_cv_clean=mw_result.cv_clean,
        detection_rate=mw_result.detection_rate,
        bootstrap_freq_mean=boot_result.freq_mean,
        bootstrap_freq_std=boot_result.freq_std,
        bootstrap_ci_low=boot_result.freq_ci_low,
        bootstrap_ci_high=boot_result.freq_ci_high,
        confidence_score=score,
        confidence_tier=tier,
    )


# ============================================================================
# PLOTTING
# ============================================================================

def plot_crypto_results(
    results: Dict[str, SignalResult],
    ls_data: Dict[str, LSResult],
    time_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
):
    """Create 4-panel dark-themed validation figure."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(COLORS["background"])

    for ax in axes.flat:
        ax.set_facecolor(COLORS["panel"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])

    # Panel 1: BTC periodogram
    ax = axes[0, 0]
    if "BTC" in ls_data:
        ls = ls_data["BTC"]
        freqs_cpd = ls.frequencies_hz * 86400
        ax.plot(freqs_cpd, ls.powers, color=COLORS["crypto"], linewidth=0.8)
        ax.axvline(ls.frequency_cpd, color=COLORS["accent"], linestyle="--",
                   label=f"Peak: {ls.frequency_cpd:.3f} cpd ({1/ls.frequency_cpd:.1f}d)")
        ax.set_xlabel("Frequency (cpd)", color=COLORS["text"])
        ax.set_ylabel("Lomb-Scargle Power", color=COLORS["text"])
        ax.set_title("BTC-USD Periodogram (2023)", color=COLORS["text"], fontweight="bold")
        ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
                  labelcolor=COLORS["text"], fontsize=9)
        ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 2: Multi-window CV comparison
    ax = axes[0, 1]
    assets = list(results.keys())
    cvs = [results[a].multi_window_cv_clean for a in assets]
    bars = ax.bar(assets, cvs, color=[COLORS["crypto"], COLORS["accent"]],
                  edgecolor=COLORS["grid"], alpha=0.8)
    for bar, cv in zip(bars, cvs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{cv:.1f}%", ha="center", color=COLORS["text"], fontsize=11)
    ax.axhline(15, color=COLORS["accent3"], linestyle=":", linewidth=1.5,
               label="Physics threshold (15%)")
    ax.set_ylabel("CV (%)", color=COLORS["text"])
    ax.set_title("Multi-Window CV", color=COLORS["text"], fontweight="bold")
    ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 3: Price time series
    ax = axes[1, 0]
    for asset_name, (t, p) in time_data.items():
        days = t / 86400
        color = COLORS["crypto"] if asset_name == "BTC" else COLORS["accent"]
        # Normalise to percent change from start for comparability
        pct = (p / p[0] - 1) * 100
        ax.plot(days, pct, color=color, linewidth=1, label=asset_name)
    ax.set_xlabel("Day of Year", color=COLORS["text"])
    ax.set_ylabel("Price Change (%)", color=COLORS["text"])
    ax.set_title("2023 Price Series (Normalised)", color=COLORS["text"], fontweight="bold")
    ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 4: Cross-domain CV comparison (log scale)
    ax = axes[1, 1]
    domains = []
    cv_vals = []
    colors_list = []
    for name, ref in CROSS_DOMAIN_REFS.items():
        domains.append(ref["domain"])
        cv_vals.append(ref["cv_mean"])
        colors_list.append(COLORS["accent3"])
    # Add crypto results
    for asset in assets:
        domains.append(f"Crypto ({asset})")
        cv_vals.append(results[asset].multi_window_cv_clean)
        colors_list.append(COLORS["crypto"])

    y_pos = np.arange(len(domains))
    bars = ax.barh(y_pos, cv_vals, color=colors_list, edgecolor=COLORS["grid"], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(domains, color=COLORS["text"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("CV (%) — log scale", color=COLORS["text"])
    ax.set_title("Cross-Domain CV Comparison", color=COLORS["text"], fontweight="bold")
    ax.axvline(15, color=COLORS["accent2"], linestyle=":", linewidth=1.5, label="15% threshold")
    ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    plt.tight_layout()
    plot_path = PLOT_DIR / "crypto_validation.png"
    fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")


# ============================================================================
# RESULTS SERIALIZATION
# ============================================================================

def serialize_result(r: SignalResult) -> Dict:
    """Convert SignalResult to JSON-safe dict."""
    return {
        "asset_name": r.asset_name,
        "symbol": r.symbol,
        "ls_frequency_hz": float(r.ls_frequency_hz),
        "ls_frequency_cpd": float(r.ls_frequency_cpd),
        "ls_period_days": float(r.ls_period_days),
        "ls_power": float(r.ls_power),
        "ls_fap": float(r.ls_fap),
        "multi_window_cv": float(r.multi_window_cv),
        "multi_window_cv_clean": float(r.multi_window_cv_clean),
        "detection_rate": float(r.detection_rate),
        "bootstrap_freq_mean_cpd": float(r.bootstrap_freq_mean * 86400),
        "bootstrap_freq_std_cpd": float(r.bootstrap_freq_std * 86400),
        "bootstrap_ci_low_cpd": float(r.bootstrap_ci_low * 86400),
        "bootstrap_ci_high_cpd": float(r.bootstrap_ci_high * 86400),
        "confidence_score": float(r.confidence_score),
        "confidence_tier": r.confidence_tier,
    }


def save_results(results: Dict[str, SignalResult], year: int):
    """Save results to JSON."""
    cv_values = [r.multi_window_cv_clean for r in results.values()]
    det_rates = [r.detection_rate for r in results.values()]

    output = {
        "experiment": "Ab Astris Negative Control: Cryptocurrency",
        "methodology": "Ab Astris (Lomb-Scargle + Multi-window + Bootstrap)",
        "data_source": f"Yahoo Finance (yfinance), {year} daily closing prices",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "sampling_interval_sec": SAMPLING_INTERVAL_SEC,
            "freq_min_cpd": FREQ_MIN_CPD,
            "freq_max_cpd": FREQ_MAX_CPD,
            "n_windows": N_WINDOWS,
            "window_duration_days": WINDOW_DURATION_DAYS,
            "n_bootstrap": N_BOOTSTRAP,
            "n_freq_points": N_FREQ_POINTS,
            "bandpass_filter": "none",
            "hilbert_envelope": "none",
        },
        "assets": {name: serialize_result(r) for name, r in results.items()},
        "summary": {
            "mean_cv": float(np.mean(cv_values)),
            "max_cv": float(np.max(cv_values)),
            "min_cv": float(np.min(cv_values)),
            "mean_detection_rate": float(np.mean(det_rates)),
            "all_noise": all(r.confidence_tier == "NOISE" for r in results.values()),
            "interpretation": (
                f"Cryptocurrency prices produce mean CV of {np.mean(cv_values):.1f}%, "
                f"consistent with the absence of physics-constrained periodic forcing, "
                f"compared to 0.17% for tidal constituents measured over the same "
                f"calendar year."
            ),
        },
        "cross_domain_comparison": {
            name: ref for name, ref in CROSS_DOMAIN_REFS.items()
        },
    }

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {RESULTS_JSON}")


# ============================================================================
# MAIN
# ============================================================================

def run_crypto_experiment(year: int = DEFAULT_YEAR) -> Dict[str, SignalResult]:
    """Run the full crypto negative control experiment."""
    print(f"\n{'='*70}")
    print(f"  AB ASTRIS NEGATIVE CONTROL: CRYPTOCURRENCY ({year})")
    print(f"{'='*70}")

    results = {}
    ls_data = {}
    time_data = {}

    for asset_name, info in CRYPTO_ASSETS.items():
        # Download
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01"
        df = download_crypto_data(info["symbol"], start, end)

        # Prepare time series
        time, prices = prepare_crypto_time_series(df)
        time_data[asset_name] = (time, prices)
        print(f"  {asset_name}: {len(prices)} data points, "
              f"{time[-1]/86400:.0f} days")

        # Run pipeline
        result = analyze_crypto_asset(time, prices, asset_name, info["symbol"])
        results[asset_name] = result

        # Save LS data for plotting
        ls_result = run_lomb_scargle(time, prices)
        ls_data[asset_name] = ls_result

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for name, r in results.items():
        print(f"  {name}: CV={r.multi_window_cv_clean:.1f}%, "
              f"Detection={r.detection_rate:.2f}, "
              f"Tier={r.confidence_tier}")
    mean_cv = np.mean([r.multi_window_cv_clean for r in results.values()])
    print(f"\n  Mean CV: {mean_cv:.1f}%")
    print(f"  Expected: >>30% (stochastic, no physics)")
    print(f"  Tidal reference: 0.17% (physics-constrained)")

    # Save
    save_results(results, year)
    plot_crypto_results(results, ls_data, time_data)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ab Astris Negative Control: Cryptocurrency"
    )
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    args = parser.parse_args()

    run_crypto_experiment(year=args.year)


if __name__ == "__main__":
    main()

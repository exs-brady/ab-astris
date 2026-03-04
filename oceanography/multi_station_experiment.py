"""
Ab Astris Oceanography — Multi-Station Tidal Validation

Extends the single-station tidal validation (San Francisco) to 5 additional
NOAA CO-OPS tide gauge stations across diverse geographic and tidal regimes.

Reuses all core functions from tide_gauge_experiment.py. Produces:
- oceanography_multi_station_results.json (cross-station + cross-domain comparison)
- plots/oceanography_multi_station_summary.png (2x2 dark-themed summary)

Usage:
    python oceanography/multi_station_experiment.py
    python oceanography/multi_station_experiment.py --stations 9414290 8518750
    python oceanography/multi_station_experiment.py --year 2022
"""

import argparse
import json
import time as time_module
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from tide_gauge_experiment import (
    # Core functions
    download_tide_data,
    prepare_time_series,
    analyze_constituent,
    run_lomb_scargle,
    create_windows,
    run_multi_window_validation,
    # Data classes
    ConstituentResult,
    # Constants
    TIDAL_CONSTITUENTS,
    SAMPLING_INTERVAL_SEC,
    N_WINDOWS,
    WINDOW_DURATION_DAYS,
    N_BOOTSTRAP,
    N_FREQ_POINTS,
    DIURNAL_BAND_HZ,
    SEMIDIURNAL_BAND_HZ,
    CROSS_DOMAIN_REFS,
    COLORS,
    _serialize_constituent,
)


# ============================================================================
# STATION CONFIGURATION
# ============================================================================

STATIONS = {
    "9414290": {
        "name": "San Francisco",
        "location": "CA",
        "tidal_type": "Mixed, mainly semidiurnal",
    },
    "8518750": {
        "name": "The Battery",
        "location": "New York, NY",
        "tidal_type": "Semidiurnal",
    },
    "8724580": {
        "name": "Key West",
        "location": "FL",
        "tidal_type": "Mixed, mainly diurnal",
    },
    "1612340": {
        "name": "Honolulu",
        "location": "HI",
        "tidal_type": "Mixed",
    },
    "9447130": {
        "name": "Seattle",
        "location": "WA",
        "tidal_type": "Mixed, mainly semidiurnal",
    },
    "8658120": {
        "name": "Wilmington",
        "location": "NC",
        "tidal_type": "Semidiurnal",
    },
}

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULTS_JSON = OUTPUT_DIR / "oceanography_multi_station_results.json"


# ============================================================================
# PER-STATION ANALYSIS
# ============================================================================

def run_station_analysis(
    station_id: str,
    station_info: Dict,
    year: int = 2023,
) -> Dict:
    """Run full Ab Astris tidal validation + residual analysis for one station.

    Returns structured results dict with Phase 1 (tidal) and Phase 2 (residual).
    If 2023 data fails, falls back to 2022.
    """
    name = station_info["name"]
    print(f"\n{'=' * 60}")
    print(f"  Station: {station_id} — {name} ({station_info['location']})")
    print(f"  Tidal type: {station_info['tidal_type']}")
    print(f"{'=' * 60}")

    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    actual_year = year

    # Download observed data (try requested year, fall back to year-1)
    print(f"\n[1/5] Downloading observed water levels ({year})...")
    try:
        df_obs = download_tide_data(station_id, year, year, "hourly_height", data_dir)
    except Exception as e:
        print(f"  {year} failed ({e}), trying {year - 1}...")
        actual_year = year - 1
        df_obs = download_tide_data(station_id, actual_year, actual_year, "hourly_height", data_dir)

    time_module.sleep(1)  # Courteous API delay

    # Prepare time series
    print(f"[2/5] Preparing time series...")
    time_sec, signal = prepare_time_series(df_obs, "v")
    duration_days = (time_sec[-1] - time_sec[0]) / 86400
    n_obs = len(time_sec)
    print(f"  {n_obs} observations, {duration_days:.1f} days")

    # Phase 1: Tidal constituent detection
    print(f"[3/5] Analyzing {len(TIDAL_CONSTITUENTS)} tidal constituents...\n")
    constituents = []
    for cname in TIDAL_CONSTITUENTS:
        info = TIDAL_CONSTITUENTS[cname]
        print(f"  {cname} ({info['name']}):")
        result = analyze_constituent(time_sec, signal, cname)
        constituents.append(result)
        print(f"    Detected: {result.ls_frequency_cpd:.6f} cpd "
              f"(target: {result.target_freq_cpd:.6f} cpd)")
        print(f"    Error:    {result.freq_error_percent:.4f}%")
        print(f"    CV:       {result.multi_window_cv_clean:.6f}%")
        print(f"    Score:    {result.confidence_score:.0f} ({result.confidence_tier})")
        print()

    # Phase 2: Residual analysis
    print(f"[4/5] Downloading tidal predictions ({actual_year})...")
    try:
        df_pred = download_tide_data(station_id, actual_year, actual_year, "predictions", data_dir)
    except Exception as e:
        print(f"  Predictions download failed ({e}), skipping residual analysis")
        df_pred = None

    time_module.sleep(1)  # Courteous API delay

    residual_results = None
    if df_pred is not None:
        print(f"[5/5] Running residual analysis...")
        residual_results = _run_residual(df_obs, df_pred, station_id, actual_year)

    # Compile station results
    cvs = [r.multi_window_cv_clean for r in constituents]
    errors = [r.freq_error_percent for r in constituents]
    scores = [r.confidence_score for r in constituents]

    station_result = {
        "name": station_info["name"],
        "location": station_info["location"],
        "tidal_type": station_info["tidal_type"],
        "year": actual_year,
        "n_observations": n_obs,
        "duration_days": round(duration_days, 1),
        "constituents": [_serialize_constituent(r) for r in constituents],
        "summary": {
            "mean_cv": round(float(np.mean(cvs)), 6),
            "max_cv": round(float(np.max(cvs)), 6),
            "min_cv": round(float(np.min(cvs)), 6),
            "mean_freq_error_pct": round(float(np.mean(errors)), 4),
            "max_freq_error_pct": round(float(np.max(errors)), 4),
            "mean_confidence_score": round(float(np.mean(scores)), 1),
            "all_confirmed": all(s >= 90 for s in scores),
            "n_confirmed": sum(1 for r in constituents if r.confidence_tier == "CONFIRMED"),
            "n_probable": sum(1 for r in constituents if r.confidence_tier == "PROBABLE"),
            "n_candidate": sum(1 for r in constituents if r.confidence_tier == "CANDIDATE"),
            "n_noise": sum(1 for r in constituents if r.confidence_tier == "NOISE"),
        },
        "residual": residual_results,
    }

    # Print station summary
    print(f"\n  --- {name} Summary ---")
    print(f"  Mean CV: {np.mean(cvs):.6f}%, Mean Error: {np.mean(errors):.4f}%")
    tiers = [r.confidence_tier for r in constituents]
    print(f"  Tiers: {', '.join(f'{r.constituent_name}={r.confidence_tier}' for r in constituents)}")

    return station_result


def _run_residual(df_obs, df_pred, station_id, year) -> Dict:
    """Run residual analysis (observed - predicted) for one station."""
    import pandas as pd

    col_obs = "v" if "v" in df_obs.columns else df_obs.columns[0]
    col_pred = "v" if "v" in df_pred.columns else (
        "predicted_wl" if "predicted_wl" in df_pred.columns else df_pred.columns[0]
    )

    df_obs_renamed = df_obs[[col_obs]].rename(columns={col_obs: "observed"})
    df_pred_renamed = df_pred[[col_pred]].rename(columns={col_pred: "predicted"})

    df_merged = pd.merge(
        df_obs_renamed, df_pred_renamed,
        left_index=True, right_index=True, how="inner",
    )
    df_merged["residual"] = df_merged["observed"] - df_merged["predicted"]
    residual_std = float(df_merged["residual"].std())
    print(f"  Merged: {len(df_merged)} points, residual std = {residual_std:.4f} m")

    time_sec, residual = prepare_time_series(df_merged, "residual")

    # Broad-spectrum LS on residual
    freq_min = 0.08 / 86400   # 0.08 cpd
    freq_max = 8.0 / 86400    # 8 cpd

    ls_result = run_lomb_scargle(time_sec, residual, freq_min_hz=freq_min, freq_max_hz=freq_max)
    period_days = 1.0 / ls_result.frequency_cpd if ls_result.frequency_cpd > 0 else float("inf")

    # Multi-window on residual
    windows = create_windows(time_sec, residual)
    mw_result = run_multi_window_validation(
        windows, ls_result.frequency_hz,
        freq_min_hz=freq_min, freq_max_hz=freq_max,
    )

    print(f"  Residual peak: {ls_result.frequency_cpd:.4f} cpd "
          f"(period = {period_days:.2f} days), CV: {mw_result.cv_clean:.3f}%")

    # Top 5 peaks
    top_peaks = []
    freqs = ls_result.frequencies_hz
    powers = ls_result.powers.copy()
    for _ in range(5):
        idx = np.argmax(powers)
        peak_hz = freqs[idx]
        peak_cpd = peak_hz * 86400
        peak_power = float(powers[idx])
        peak_period = 1.0 / peak_cpd if peak_cpd > 0 else float("inf")
        top_peaks.append({
            "frequency_cpd": round(peak_cpd, 4),
            "period_days": round(peak_period, 2),
            "power": round(peak_power, 4),
        })
        mask = np.abs(freqs - peak_hz) < (peak_hz * 0.02)
        powers[mask] = 0

    return {
        "n_points": len(time_sec),
        "residual_std_m": round(residual_std, 4),
        "peak_frequency_cpd": round(ls_result.frequency_cpd, 4),
        "peak_period_days": round(period_days, 2),
        "peak_power": round(ls_result.power, 4),
        "peak_fap": float(ls_result.fap),
        "multi_window_cv": round(mw_result.cv_clean, 3),
        "detection_rate": round(mw_result.detection_rate, 3),
        "top_5_peaks": top_peaks,
    }


# ============================================================================
# CROSS-STATION SUMMARY
# ============================================================================

def compute_cross_station_summary(station_results: Dict) -> Dict:
    """Aggregate metrics across all stations."""
    n_stations = len(station_results)
    total_tested = 0
    total_confirmed = 0
    total_probable = 0
    total_candidate = 0
    total_noise = 0

    all_cvs = []
    all_errors = []
    all_detection_rates = []

    # Per-constituent aggregation
    constituent_cvs = {name: [] for name in TIDAL_CONSTITUENTS}
    constituent_errors = {name: [] for name in TIDAL_CONSTITUENTS}

    for sid, sdata in station_results.items():
        summary = sdata["summary"]
        total_tested += 4
        total_confirmed += summary["n_confirmed"]
        total_probable += summary["n_probable"]
        total_candidate += summary["n_candidate"]
        total_noise += summary["n_noise"]

        for c in sdata["constituents"]:
            cname = c["constituent_name"]
            constituent_cvs[cname].append(c["multi_window_cv_clean"])
            constituent_errors[cname].append(c["freq_error_percent"])
            all_cvs.append(c["multi_window_cv_clean"])
            all_errors.append(c["freq_error_percent"])
            all_detection_rates.append(c["detection_rate"])

    result = {
        "n_stations": n_stations,
        "total_constituents_tested": total_tested,
        "total_confirmed": total_confirmed,
        "total_probable": total_probable,
        "total_candidate": total_candidate,
        "total_noise": total_noise,
        "overall_mean_cv": round(float(np.mean(all_cvs)), 6),
        "overall_mean_freq_error_pct": round(float(np.mean(all_errors)), 4),
        "overall_detection_rate": round(float(np.mean(all_detection_rates)), 3),
    }

    # Per-constituent stats
    for cname in TIDAL_CONSTITUENTS:
        cvs = constituent_cvs[cname]
        errs = constituent_errors[cname]
        key = cname.lower()
        result[f"{key}_cv_range"] = [round(min(cvs), 6), round(max(cvs), 6)]
        result[f"{key}_cv_mean"] = round(float(np.mean(cvs)), 6)
        result[f"{key}_error_range"] = [round(min(errs), 4), round(max(errs), 4)]
        result[f"{key}_error_mean"] = round(float(np.mean(errs)), 4)

    return result


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_multi_station_summary(station_results: Dict, cross_summary: Dict):
    """Generate 2x2 dark-themed multi-station summary figure."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    station_ids = list(station_results.keys())
    station_labels = [f"{station_results[s]['name']}" for s in station_ids]
    constituent_names = list(TIDAL_CONSTITUENTS.keys())
    n_stations = len(station_ids)
    n_constituents = len(constituent_names)

    # Constituent colours
    c_colors = ["#00b4d8", "#4ecdc4", "#ff6b6b", "#f0ad4e"]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=COLORS["background"])
    fig.suptitle(
        "Ab Astris Oceanography — Multi-Station Tidal Validation\n"
        f"{n_stations} NOAA CO-OPS Stations, 2023",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )

    # ---- Panel 1: Per-station CV comparison (grouped bars) ----
    ax = axes[0, 0]
    ax.set_facecolor(COLORS["panel"])
    bar_width = 0.18
    x = np.arange(n_stations)

    for i, cname in enumerate(constituent_names):
        cvs = []
        for sid in station_ids:
            c_data = [c for c in station_results[sid]["constituents"]
                      if c["constituent_name"] == cname][0]
            cvs.append(c_data["multi_window_cv_clean"])
        bars = ax.bar(x + i * bar_width, cvs, bar_width,
                      label=cname, color=c_colors[i], alpha=0.85)

    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(station_labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("CV (%)", color=COLORS["text"])
    ax.set_title("Multi-Window CV by Station", color=COLORS["text"], fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(colors=COLORS["text"])
    ax.grid(True, alpha=0.2, color=COLORS["grid"], axis="y")

    # ---- Panel 2: Frequency error comparison ----
    ax = axes[0, 1]
    ax.set_facecolor(COLORS["panel"])

    for i, cname in enumerate(constituent_names):
        errs = []
        for sid in station_ids:
            c_data = [c for c in station_results[sid]["constituents"]
                      if c["constituent_name"] == cname][0]
            errs.append(c_data["freq_error_percent"])
        bars = ax.bar(x + i * bar_width, errs, bar_width,
                      label=cname, color=c_colors[i], alpha=0.85)

    ax.axhline(0.21, color=COLORS["accent2"], linestyle="--", alpha=0.7,
               label="Bearing mean (0.21%)")
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(station_labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Frequency Error (%)", color=COLORS["text"])
    ax.set_title("Detection Accuracy by Station", color=COLORS["text"], fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(colors=COLORS["text"])
    ax.grid(True, alpha=0.2, color=COLORS["grid"], axis="y")

    # ---- Panel 3: Cross-domain CV ladder (log scale) ----
    ax = axes[1, 0]
    ax.set_facecolor(COLORS["panel"])

    ocean_cv = cross_summary["overall_mean_cv"]
    domains = ["Variable\nStars", "Bearings", "Tidal\n(this work)", "Structural", "Volcanoes", "Crypto"]
    cv_values = [
        CROSS_DOMAIN_REFS["variable_stars"]["cv_mean"],
        CROSS_DOMAIN_REFS["bearings"]["cv_mean"],
        ocean_cv,
        1.0,  # structural reference
        CROSS_DOMAIN_REFS["volcanoes"]["cv_mean"],
        CROSS_DOMAIN_REFS["crypto"]["cv_mean"],
    ]
    domain_colors = [
        COLORS["accent3"], COLORS["accent3"], COLORS["tidal"],
        COLORS["accent3"], COLORS["accent3"], COLORS["accent2"],
    ]

    bars = ax.bar(domains, cv_values, color=domain_colors, alpha=0.85,
                  edgecolor=COLORS["accent"], linewidth=0.5)
    ax.set_ylabel("Mean CV (%)", color=COLORS["text"])
    ax.set_title("Cross-Domain CV Comparison", color=COLORS["text"], fontweight="bold")
    ax.set_yscale("log")
    ax.tick_params(colors=COLORS["text"])
    ax.grid(True, alpha=0.2, color=COLORS["grid"], axis="y")
    for bar, cv_val in zip(bars, cv_values):
        y_pos = bar.get_height() * 1.5
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{cv_val:.3f}%", ha="center", fontsize=8, color=COLORS["text"])

    # ---- Panel 4: Residual dominant periods ----
    ax = axes[1, 1]
    ax.set_facecolor(COLORS["panel"])

    periods = []
    res_cvs = []
    labels_with_data = []
    for sid in station_ids:
        res = station_results[sid].get("residual")
        if res:
            periods.append(res["peak_period_days"])
            res_cvs.append(res["multi_window_cv"])
            labels_with_data.append(station_results[sid]["name"])

    if periods:
        x_res = np.arange(len(labels_with_data))
        bars = ax.bar(x_res, periods, color=COLORS["residual"], alpha=0.85,
                      edgecolor=COLORS["accent"], linewidth=0.5)
        ax.set_xticks(x_res)
        ax.set_xticklabels(labels_with_data, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Dominant Period (days)", color=COLORS["text"])
        ax.set_title("Residual Dominant Periods", color=COLORS["text"], fontweight="bold")
        ax.tick_params(colors=COLORS["text"])
        ax.grid(True, alpha=0.2, color=COLORS["grid"], axis="y")
        for bar, period, cv in zip(bars, periods, res_cvs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"CV={cv:.1f}%", ha="center", va="bottom", fontsize=8,
                    color=COLORS["text"])

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = PLOT_DIR / "oceanography_multi_station_summary.png"
    fig.savefig(out_path, dpi=150, facecolor=COLORS["background"], bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")


# ============================================================================
# RESULTS OUTPUT
# ============================================================================

def save_multi_station_results(
    station_results: Dict,
    cross_summary: Dict,
) -> Dict:
    """Save comprehensive multi-station results as JSON."""
    ocean_cv = cross_summary["overall_mean_cv"]
    ocean_dr = cross_summary["overall_detection_rate"]

    output = {
        "experiment": "oceanography_multi_station_validation",
        "methodology": "Ab Astris (Lomb-Scargle + Multi-window + Bootstrap)",
        "data_source": "NOAA CO-OPS Tide Gauges",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "sampling_interval_sec": SAMPLING_INTERVAL_SEC,
            "n_windows": N_WINDOWS,
            "window_duration_days": WINDOW_DURATION_DAYS,
            "n_bootstrap": N_BOOTSTRAP,
            "n_freq_points": N_FREQ_POINTS,
            "diurnal_band_cpd": [DIURNAL_BAND_HZ[0] * 86400, DIURNAL_BAND_HZ[1] * 86400],
            "semidiurnal_band_cpd": [SEMIDIURNAL_BAND_HZ[0] * 86400, SEMIDIURNAL_BAND_HZ[1] * 86400],
        },
        "stations": station_results,
        "cross_station_summary": cross_summary,
        "cross_domain_comparison": {
            "variable_stars": {"cv_mean": 0.005, "detection_rate": 1.0},
            "bearings": {"cv_mean": 0.008, "detection_rate": 1.0},
            "oceanography_tidal": {"cv_mean": round(ocean_cv, 6), "detection_rate": round(ocean_dr, 3)},
            "structural": {"cv_mean": 1.0, "detection_rate": 1.0},
            "volcanoes": {"cv_mean": 3.96, "detection_rate": 0.997},
            "crypto": {"cv_mean": 68.0, "detection_rate": 0.3},
        },
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_JSON}")
    return output


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ab Astris Multi-Station Tidal Validation")
    parser.add_argument("--stations", nargs="+", default=list(STATIONS.keys()),
                        help="Station IDs to process (default: all 6)")
    parser.add_argument("--year", type=int, default=2023,
                        help="Data year (default: 2023)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Ab Astris Oceanography — Multi-Station Validation")
    print("=" * 60)
    print(f"  Stations: {len(args.stations)}")
    print(f"  Year:     {args.year}")
    print(f"  Pipeline: Bandpass -> Lomb-Scargle -> Multi-window -> Bootstrap")
    print(f"  Note:     No Hilbert envelope (tidal = direct periodic signal)")

    # Run each station
    station_results = {}
    for sid in args.stations:
        if sid not in STATIONS:
            print(f"\nWARNING: Unknown station {sid}, skipping")
            continue
        station_results[sid] = run_station_analysis(sid, STATIONS[sid], args.year)

    if not station_results:
        print("\nERROR: No stations processed successfully")
        return

    # Cross-station summary
    print(f"\n{'=' * 60}")
    print("  CROSS-STATION SUMMARY")
    print(f"{'=' * 60}")

    cross_summary = compute_cross_station_summary(station_results)

    print(f"\n  Stations processed:     {cross_summary['n_stations']}")
    print(f"  Constituents tested:    {cross_summary['total_constituents_tested']}")
    print(f"  CONFIRMED:              {cross_summary['total_confirmed']}")
    print(f"  PROBABLE:               {cross_summary['total_probable']}")
    print(f"  CANDIDATE:              {cross_summary['total_candidate']}")
    print(f"  NOISE:                  {cross_summary['total_noise']}")
    print(f"  Overall mean CV:        {cross_summary['overall_mean_cv']:.6f}%")
    print(f"  Overall mean error:     {cross_summary['overall_mean_freq_error_pct']:.4f}%")
    print(f"  Overall detection rate: {cross_summary['overall_detection_rate']:.3f}")

    # Per-constituent summary
    print(f"\n  {'Constituent':<6} {'CV Mean':<12} {'CV Range':<24} {'Error Mean':<12}")
    print(f"  {'-' * 54}")
    for cname in TIDAL_CONSTITUENTS:
        key = cname.lower()
        cv_mean = cross_summary[f"{key}_cv_mean"]
        cv_range = cross_summary[f"{key}_cv_range"]
        err_mean = cross_summary[f"{key}_error_mean"]
        print(f"  {cname:<6} {cv_mean:<12.6f} [{cv_range[0]:.6f}, {cv_range[1]:.6f}]  {err_mean:<12.4f}")

    # Success criteria check
    print(f"\n  --- Success Criteria ---")
    m2_confirmed = sum(1 for sid, s in station_results.items()
                       for c in s["constituents"]
                       if c["constituent_name"] == "M2" and c["confidence_tier"] == "CONFIRMED")
    k1_confirmed = sum(1 for sid, s in station_results.items()
                       for c in s["constituents"]
                       if c["constituent_name"] == "K1" and c["confidence_tier"] == "CONFIRMED")
    all_confirmed_count = cross_summary["total_confirmed"]
    n_total = cross_summary["total_constituents_tested"]
    overall_cv = cross_summary["overall_mean_cv"]
    overall_err = cross_summary["overall_mean_freq_error_pct"]

    min_pass = m2_confirmed >= 4 and k1_confirmed >= 4
    target_pass = all_confirmed_count >= (n_total - n_total // 6) and overall_cv < 1.0
    stretch_pass = overall_err < 0.05

    print(f"  MINIMUM  (M2+K1 CONFIRMED at >=4 stations): {'PASS' if min_pass else 'FAIL'} "
          f"(M2={m2_confirmed}, K1={k1_confirmed})")
    print(f"  TARGET   (>=4 stations all CONFIRMED, CV<1%): {'PASS' if target_pass else 'FAIL'} "
          f"(confirmed={all_confirmed_count}/{n_total}, CV={overall_cv:.4f}%)")
    print(f"  STRETCH  (mean error < 0.05%): {'PASS' if stretch_pass else 'FAIL'} "
          f"(error={overall_err:.4f}%)")

    # Save results and plot
    save_multi_station_results(station_results, cross_summary)
    plot_multi_station_summary(station_results, cross_summary)

    # Final per-station table
    print(f"\n{'=' * 60}")
    print("  PER-STATION RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  {'Station':<18} {'M2':<10} {'S2':<10} {'K1':<10} {'O1':<10} {'Mean CV':<10} {'Mean Err'}")
    print(f"  {'-' * 78}")
    for sid, sdata in station_results.items():
        name = sdata["name"]
        tiers = {c["constituent_name"]: c["confidence_tier"] for c in sdata["constituents"]}
        cv = sdata["summary"]["mean_cv"]
        err = sdata["summary"]["mean_freq_error_pct"]
        print(f"  {name:<18} {tiers['M2']:<10} {tiers['S2']:<10} {tiers['K1']:<10} "
              f"{tiers['O1']:<10} {cv:<10.4f} {err:.4f}%")


if __name__ == "__main__":
    main()

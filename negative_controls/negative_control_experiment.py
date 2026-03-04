"""
Ab Astris Negative Control Analysis: Combined Orchestrator

Runs all three negative control analyses and compiles a cross-domain summary:
1. Cryptocurrency (BTC/ETH) — stochastic
2. Sunspot number (11-year + 27-day) — chaotic physics
3. Heart Rate Variability (HRV) — biological regulation

Produces the cross-domain CV comparison table for the paper.

Usage:
    python negative_controls/negative_control_experiment.py
    python negative_controls/negative_control_experiment.py --skip-hrv
    python negative_controls/negative_control_experiment.py --skip-crypto --skip-sunspot
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import experiment runners
sys.path.insert(0, str(Path(__file__).parent))
from crypto_experiment import run_crypto_experiment
from sunspot_experiment import run_sunspot_experiment
from hrv_experiment import run_hrv_experiment


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent
PLOT_DIR = OUTPUT_DIR / "plots"
RESULTS_JSON = OUTPUT_DIR / "negative_controls_results.json"

# Cross-domain reference values (positive controls from Ab Astris)
POSITIVE_CONTROLS = [
    {"domain": "Bearings", "signal": "Defect frequencies",
     "forcing": "Fixed geometry", "cv_mean": 0.008, "category": "Physics-constrained"},
    {"domain": "Tides (S2)", "signal": "Solar semidiurnal",
     "forcing": "Orbital mechanics", "cv_mean": 0.000, "category": "Physics-constrained"},
    {"domain": "Tides (M2)", "signal": "Lunar semidiurnal",
     "forcing": "Orbital mechanics", "cv_mean": 0.100, "category": "Physics-constrained"},
    {"domain": "Tides (K1, O1)", "signal": "Diurnal",
     "forcing": "Orbital + modulation", "cv_mean": 0.300, "category": "Physics-constrained"},
    {"domain": "Structural", "signal": "Building resonance",
     "forcing": "Mass/stiffness", "cv_mean": 1.019, "category": "Physics-constrained"},
    {"domain": "Volcanic tremor", "signal": "Conduit resonance",
     "forcing": "Evolving source", "cv_mean": 3.96, "category": "Quasi-periodic"},
]

COLORS = {
    "background": "#0d1117",
    "panel": "#161b22",
    "accent": "#00b4d8",
    "accent2": "#ff6b6b",
    "accent3": "#4ecdc4",
    "text": "#c9d1d9",
    "grid": "#21262d",
    "physics": "#4ecdc4",
    "quasi": "#00b4d8",
    "biological": "#f0ad4e",
    "chaotic": "#9b59b6",
    "stochastic": "#ff6b6b",
}

CATEGORY_COLORS = {
    "Physics-constrained": COLORS["physics"],
    "Quasi-periodic": COLORS["quasi"],
    "Biological": COLORS["biological"],
    "Chaotic": COLORS["chaotic"],
    "Stochastic": COLORS["stochastic"],
}


# ============================================================================
# CROSS-DOMAIN COMPILATION
# ============================================================================

def compile_cross_domain_summary(
    crypto_results=None,
    sunspot_results=None,
    hrv_results=None,
) -> List[Dict]:
    """Compile the full cross-domain CV comparison table.

    Returns list of dicts with domain, signal, forcing, cv_mean, category — matching
    the table specification in negative_control_protocol.md.
    """
    rows = list(POSITIVE_CONTROLS)

    # Add negative control results
    if hrv_results:
        for band in ["HF", "LF", "VLF"]:
            cvs = [hrv_results[s].bands[band].multi_window_cv_clean
                   for s in hrv_results if band in hrv_results[s].bands]
            if cvs:
                from hrv_experiment import HRV_BANDS
                band_info = HRV_BANDS[band]
                rows.append({
                    "domain": f"HRV ({band})",
                    "signal": band_info["name"],
                    "forcing": "Autonomic regulation",
                    "cv_mean": float(np.mean(cvs)),
                    "category": "Biological",
                })

    if sunspot_results:
        if "monthly_11yr" in sunspot_results:
            r = sunspot_results["monthly_11yr"]
            rows.append({
                "domain": "Sunspots (11-yr)",
                "signal": "Solar cycle",
                "forcing": "MHD dynamo",
                "cv_mean": float(r.multi_window_cv_clean),
                "category": "Chaotic",
            })
        if "daily_27day" in sunspot_results:
            r = sunspot_results["daily_27day"]
            rows.append({
                "domain": "Sunspots (27-day)",
                "signal": "Solar rotation",
                "forcing": "Differential rotation",
                "cv_mean": float(r.multi_window_cv_clean),
                "category": "Chaotic",
            })

    if crypto_results:
        for asset in crypto_results:
            r = crypto_results[asset]
            rows.append({
                "domain": f"Crypto ({asset})",
                "signal": "Price oscillation",
                "forcing": "Market sentiment",
                "cv_mean": float(r.multi_window_cv_clean),
                "category": "Stochastic",
            })

    # Sort by CV
    rows.sort(key=lambda x: x["cv_mean"])

    return rows


# ============================================================================
# PLOTTING
# ============================================================================

def plot_combined_summary(cross_domain: List[Dict]):
    """Create cross-domain CV ladder plot (the main paper figure)."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.patch.set_facecolor(COLORS["background"])

    for ax in axes:
        ax.set_facecolor(COLORS["panel"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])

    # Panel 1: CV ladder (horizontal bar, log scale)
    ax = axes[0]
    domains = [r["domain"] for r in cross_domain]
    cvs = [r["cv_mean"] for r in cross_domain]
    categories = [r["category"] for r in cross_domain]
    bar_colors = [CATEGORY_COLORS.get(c, COLORS["text"]) for c in categories]

    y_pos = np.arange(len(domains))
    bars = ax.barh(y_pos, cvs, color=bar_colors, edgecolor=COLORS["grid"], alpha=0.8)

    # Add CV labels
    for i, (bar, cv) in enumerate(zip(bars, cvs)):
        label_x = max(cv, 0.001)
        ax.text(label_x * 1.5, i, f"{cv:.2f}%", va="center",
                color=COLORS["text"], fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(domains, color=COLORS["text"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Coefficient of Variation (%) — log scale", color=COLORS["text"])
    ax.set_title("Cross-Domain CV Ladder", color=COLORS["text"], fontweight="bold", fontsize=14)
    ax.axvline(15, color=COLORS["accent2"], linestyle=":", linewidth=2,
               label="15% threshold")

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS[cat], edgecolor=COLORS["grid"], label=cat)
        for cat in ["Physics-constrained", "Quasi-periodic", "Biological",
                     "Chaotic", "Stochastic"]
        if cat in set(categories)
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # Panel 2: Category summary (boxplot-like)
    ax = axes[1]
    category_order = ["Physics-constrained", "Quasi-periodic", "Biological",
                      "Chaotic", "Stochastic"]
    present_cats = [c for c in category_order if c in set(categories)]

    cat_data = {}
    for row in cross_domain:
        cat = row["category"]
        if cat not in cat_data:
            cat_data[cat] = []
        cat_data[cat].append(row["cv_mean"])

    x_pos = np.arange(len(present_cats))
    for i, cat in enumerate(present_cats):
        vals = cat_data.get(cat, [])
        if vals:
            mean_v = np.mean(vals)
            for v in vals:
                ax.scatter(i, v, color=CATEGORY_COLORS[cat], s=80, alpha=0.7,
                           edgecolor=COLORS["grid"], zorder=3)
            ax.scatter(i, mean_v, color=CATEGORY_COLORS[cat], s=200, marker="D",
                       edgecolor="white", linewidth=1.5, zorder=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(present_cats, color=COLORS["text"], fontsize=9, rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("CV (%) — log scale", color=COLORS["text"])
    ax.set_title("CV by Forcing Category", color=COLORS["text"], fontweight="bold", fontsize=14)
    ax.axhline(15, color=COLORS["accent2"], linestyle=":", linewidth=2, label="15% threshold")
    ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    plt.tight_layout()
    plot_path = PLOT_DIR / "negative_controls_summary.png"
    fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Summary plot saved: {plot_path}")


# ============================================================================
# RESULTS SERIALIZATION
# ============================================================================

def save_combined_results(
    cross_domain: List[Dict],
    crypto_results=None,
    sunspot_results=None,
    hrv_results=None,
):
    """Save combined results to JSON."""
    output = {
        "experiment": "Ab Astris Negative Control Analysis (Combined)",
        "methodology": "Ab Astris (Lomb-Scargle + Multi-window + Bootstrap)",
        "timestamp": datetime.now().isoformat(),
        "negative_controls_run": {
            "crypto": crypto_results is not None,
            "sunspot": sunspot_results is not None,
            "hrv": hrv_results is not None,
        },
        "cross_domain_table": cross_domain,
        "summary": {
            "n_positive_controls": len(POSITIVE_CONTROLS),
            "n_negative_controls": len(cross_domain) - len(POSITIVE_CONTROLS),
            "physics_constrained_max_cv": max(
                r["cv_mean"] for r in cross_domain
                if r["category"] == "Physics-constrained"
            ),
            "negative_control_min_cv": min(
                (r["cv_mean"] for r in cross_domain
                 if r["category"] in ("Biological", "Chaotic", "Stochastic")),
                default=None,
            ),
        },
    }

    # Add per-experiment summaries
    if crypto_results:
        output["crypto_summary"] = {
            asset: {
                "cv_clean": float(r.multi_window_cv_clean),
                "detection_rate": float(r.detection_rate),
                "tier": r.confidence_tier,
                "period_days": float(r.ls_period_days),
            }
            for asset, r in crypto_results.items()
        }

    if sunspot_results:
        output["sunspot_summary"] = {}
        for key, r in sunspot_results.items():
            output["sunspot_summary"][key] = {
                "cv_clean": float(r.multi_window_cv_clean),
                "detection_rate": float(r.detection_rate),
                "tier": r.confidence_tier,
                "period_years": float(r.ls_period_years),
            }

    if hrv_results:
        band_summary = {}
        for band_name in ["HF", "LF", "VLF"]:
            cvs = [hrv_results[s].bands[band_name].multi_window_cv_clean
                   for s in hrv_results if band_name in hrv_results[s].bands]
            if cvs:
                band_summary[band_name] = {
                    "mean_cv": float(np.mean(cvs)),
                    "std_cv": float(np.std(cvs)),
                    "n_subjects": len(cvs),
                }
        output["hrv_summary"] = band_summary

    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Combined results saved: {RESULTS_JSON}")


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_summary_table(cross_domain: List[Dict]):
    """Print formatted cross-domain CV comparison table."""
    print(f"\n{'='*90}")
    print("  CROSS-DOMAIN CV COMPARISON TABLE (for paper)")
    print(f"{'='*90}")
    print(f"  {'Domain':<25} {'Signal':<25} {'Forcing':<25} {'CV (%)':<10} {'Category'}")
    print(f"  {'-'*25} {'-'*25} {'-'*25} {'-'*10} {'-'*20}")
    for row in cross_domain:
        print(f"  {row['domain']:<25} {row['signal']:<25} "
              f"{row['forcing']:<25} {row['cv_mean']:<10.3f} {row['category']}")
    print(f"{'='*90}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ab Astris Negative Control Analysis (Combined)"
    )
    parser.add_argument("--skip-crypto", action="store_true",
                        help="Skip cryptocurrency analysis")
    parser.add_argument("--skip-hrv", action="store_true",
                        help="Skip heart rate variability analysis")
    parser.add_argument("--skip-sunspot", action="store_true",
                        help="Skip sunspot analysis")
    parser.add_argument("--year", type=int, default=2023,
                        help="Year for crypto data (default: 2023)")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"#  AB ASTRIS NEGATIVE CONTROL ANALYSIS")
    print(f"#  Crypto: {'SKIP' if args.skip_crypto else 'RUN'}")
    print(f"#  Sunspot: {'SKIP' if args.skip_sunspot else 'RUN'}")
    print(f"#  HRV: {'SKIP' if args.skip_hrv else 'RUN'}")
    print(f"{'#'*70}")

    crypto_results = None
    sunspot_results = None
    hrv_results = None

    if not args.skip_crypto:
        crypto_results = run_crypto_experiment(year=args.year)

    if not args.skip_sunspot:
        sunspot_results = run_sunspot_experiment()

    if not args.skip_hrv:
        hrv_results = run_hrv_experiment()

    # Compile cross-domain summary
    cross_domain = compile_cross_domain_summary(
        crypto_results=crypto_results,
        sunspot_results=sunspot_results,
        hrv_results=hrv_results,
    )

    # Print, save, plot
    print_summary_table(cross_domain)
    save_combined_results(cross_domain, crypto_results, sunspot_results, hrv_results)
    plot_combined_summary(cross_domain)

    print(f"\n  All negative control analyses complete.")
    print(f"  Results: {RESULTS_JSON}")
    print(f"  Plot: {PLOT_DIR / 'negative_controls_summary.png'}")


if __name__ == "__main__":
    main()

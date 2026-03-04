#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals on Cross-Domain CV Means.

Computes 95% bootstrap CIs for the 14-entry CV gradient (Table 6) in the
Ab Astris paper.  Outputs:
  1. Console results table
  2. bootstrap_ci_results.json
  3. LaTeX table fragment (bootstrap_ci_table.tex)
  4. Updated fig1_cv_gradient.pdf with CI error bars

Usage:
    python docs/arXiv/bootstrap_ci_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

OCEAN_RESULTS = REPO_ROOT / "oceanography" / "oceanography_multi_station_results.json"
BEARING_RESULTS = REPO_ROOT / "bearing" / "bearing_experiment_results.json"
Z24_CSV = REPO_ROOT / "shm" / "z24" / "results" / "z24_damage_predictions.csv"
LUMO_RESULTS = REPO_ROOT / "shm" / "shm_results.json"
CESMD_RESULTS = REPO_ROOT / "shm" / "cesmd_final_results.json"
SUNSPOT_RESULTS = REPO_ROOT / "negative_controls" / "sunspot_results.json"
VOLCANO_DIR = REPO_ROOT / "tremorscope" / "tremorscope_results"
HRV_RESULTS = REPO_ROOT / "negative_controls" / "hrv_results.json"
CRYPTO_RESULTS = REPO_ROOT / "negative_controls" / "crypto_results.json"

ASSETS_DIR = REPO_ROOT / "docs" / "arXiv" / "Assets"
OUTPUT_DIR = REPO_ROOT / "docs" / "arXiv"

# ---------------------------------------------------------------------------
# Bootstrap parameters
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 100_000
SEED = 42
CI_LEVEL = 0.95

# ---------------------------------------------------------------------------
# Published Table 6 values (for validation)
# ---------------------------------------------------------------------------
PUBLISHED_CV = {
    "Tides (S2)": 0.000,
    "Bearings": 0.008,
    "Tides (M2)": 0.10,
    "Tides (K1, O1)": 0.30,
    "Structural (Z24)": 0.43,
    "Structural (CESMD)": 1.02,
    "Sunspots (11-yr)": 2.97,
    "Volcanic tremor": 3.96,
    "Sunspots (27-day)": 5.33,
    "HRV (LF)": 11.6,
    "HRV (HF)": 15.4,
    "HRV (VLF)": 21.5,
    "Crypto (BTC)": 53.0,
    "Crypto (ETH)": 71.9,
}

FORCING = {
    "Tides (S2)": "Orbital mechanics",
    "Bearings": "Fixed geometry",
    "Tides (M2)": "Orbital mechanics",
    "Tides (K1, O1)": "Orbital + modulation",
    "Structural (Z24)": "Ambient excitation",
    "Structural (CESMD)": "Earthquake excitation",
    "Sunspots (11-yr)": "MHD dynamo",
    "Volcanic tremor": "Evolving source",
    "Sunspots (27-day)": "Differential rotation",
    "HRV (LF)": "Autonomic regulation",
    "HRV (HF)": "Autonomic regulation",
    "HRV (VLF)": "Autonomic regulation",
    "Crypto (BTC)": "Market sentiment",
    "Crypto (ETH)": "Market sentiment",
}

CATEGORY = {
    "Tides (S2)": "Physics-constrained",
    "Bearings": "Physics-constrained",
    "Tides (M2)": "Physics-constrained",
    "Tides (K1, O1)": "Physics-constrained",
    "Structural (Z24)": "Physics-constrained",
    "Structural (CESMD)": "Physics-constrained",
    "Sunspots (11-yr)": "Chaotic",
    "Volcanic tremor": "Quasi-periodic",
    "Sunspots (27-day)": "Chaotic",
    "HRV (LF)": "Biological",
    "HRV (HF)": "Biological",
    "HRV (VLF)": "Biological",
    "Crypto (BTC)": "Stochastic",
    "Crypto (ETH)": "Stochastic",
}

# Domain order (ascending CV, matching Table 6)
DOMAIN_ORDER = list(PUBLISHED_CV.keys())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class DomainResult:
    domain: str
    forcing: str
    category: str
    n: int
    cv_values: list[float]
    cv_mean: float
    published_cv: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_width: Optional[float] = None
    mean_validated: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci(
    cv_values: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean of cv_values."""
    rng = np.random.default_rng(seed=seed)
    n = len(cv_values)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(cv_values, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(np.mean(cv_values)), lower, upper


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_tides() -> dict[str, np.ndarray]:
    """Extract per-station multi_window_cv for each tidal constituent group."""
    data = load_json(OCEAN_RESULTS)
    s2_cvs, m2_cvs, k1_cvs, o1_cvs = [], [], [], []
    for station_id, station in data["stations"].items():
        for const in station["constituents"]:
            name = const["constituent_name"]
            # multi_window_cv is stored as percentage (0.10 = 0.10%)
            cv = const["multi_window_cv"]
            if name == "S2":
                s2_cvs.append(cv)
            elif name == "M2":
                m2_cvs.append(cv)
            elif name == "K1":
                k1_cvs.append(cv)
            elif name == "O1":
                o1_cvs.append(cv)
    return {
        "Tides (S2)": np.array(s2_cvs),
        "Tides (M2)": np.array(m2_cvs),
        "Tides (K1, O1)": np.concatenate([k1_cvs, o1_cvs]),
    }


def extract_bearings() -> np.ndarray:
    """Extract per-condition multi_window_cv_pct for successful detections.

    Filters to conditions where:
      - multi_window_cv_pct is not null
      - freq_error_pct < 5% (correct frequency detected)
    Excludes "Normal" (shaft frequency) as it's not a fault detection.
    """
    data = load_json(BEARING_RESULTS)
    cvs = [
        cond["multi_window_cv_pct"]
        for cond in data
        if (cond.get("multi_window_cv_pct") is not None
            and cond.get("freq_error_pct") is not None
            and cond["freq_error_pct"] < 5.0
            and cond.get("fault_type") != "Normal")
    ]
    return np.array(cvs)


def extract_z24() -> np.ndarray:
    """Extract per-measurement freq_1_cv_pct from Z24 confirmed detections."""
    df = pd.read_csv(Z24_CSV)
    # Filter to confirmed tier only
    if "freq_1_tier" in df.columns:
        df = df[df["freq_1_tier"] == "CONFIRMED"]
    cvs = df["freq_1_cv_pct"].dropna().values
    return cvs


def extract_structural_cesmd() -> np.ndarray:
    """Extract LUMO condition CVs (the basis for the 1.02% published value)."""
    data = load_json(LUMO_RESULTS)
    cvs = [cond["multi_window_cv"] for cond in data["conditions"]]
    return np.array(cvs)


def extract_sunspots() -> dict[str, float]:
    """Extract single CV values for sunspot analyses."""
    data = load_json(SUNSPOT_RESULTS)
    return {
        "Sunspots (11-yr)": data["analyses"]["monthly_11yr"]["multi_window_cv_clean"],
        "Sunspots (27-day)": data["analyses"]["daily_27day"]["multi_window_cv_clean"],
    }


def extract_volcanic() -> tuple[np.ndarray, np.ndarray]:
    """Extract volcanic tremor CV data at two levels.

    Returns:
        (per_volcano_means, all_snapshot_cvs): The 4 per-volcano mean CVs
        (used in the paper's cross-domain table), and all 679 snapshot-level
        CVs for supplementary analysis.
    """
    timeline_files = [
        "kilauea_2018_timeline.json",
        "msh_2004_timeline.json",
        "pavlof_2016_timeline.json",
        "augustine_2006_timeline.json",
    ]
    per_volcano_means = []
    all_cvs = []
    for fname in timeline_files:
        fpath = VOLCANO_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        data = load_json(fpath)
        volcano_cvs = []
        for snap in data["snapshots"]:
            cv = snap.get("multi_window_cv")
            if cv is not None:
                all_cvs.append(cv)
                volcano_cvs.append(cv)
        if volcano_cvs:
            per_volcano_means.append(np.mean(volcano_cvs))
    return np.array(per_volcano_means), np.array(all_cvs)


def extract_hrv() -> dict[str, np.ndarray]:
    """Extract per-subject multi_window_cv_clean for each HRV band."""
    data = load_json(HRV_RESULTS)
    bands = {"HRV (LF)": "LF", "HRV (HF)": "HF", "HRV (VLF)": "VLF"}
    result = {}
    for domain_name, band_key in bands.items():
        cvs = []
        for subj_id, subj in data["subjects"].items():
            if band_key in subj["bands"]:
                cvs.append(subj["bands"][band_key]["multi_window_cv_clean"])
        result[domain_name] = np.array(cvs)
    return result


def extract_crypto() -> dict[str, float]:
    """Extract single CV values for crypto assets."""
    data = load_json(CRYPTO_RESULTS)
    return {
        "Crypto (BTC)": data["assets"]["BTC"]["multi_window_cv_clean"],
        "Crypto (ETH)": data["assets"]["ETH"]["multi_window_cv_clean"],
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis() -> list[DomainResult]:
    print("=" * 70)
    print("Bootstrap CI Analysis — Ab Astris Cross-Domain CV Gradient")
    print(f"  n_bootstrap = {N_BOOTSTRAP:,}, seed = {SEED}, CI = {CI_LEVEL*100:.0f}%")
    print("=" * 70)

    results: list[DomainResult] = []

    # --- Tides ---
    print("\n[1/7] Extracting tidal CV data...")
    tides = extract_tides()
    for domain, cvs in tides.items():
        print(f"  {domain}: N={len(cvs)}, values={np.round(cvs, 4)}")

    # --- Bearings ---
    print("\n[2/7] Extracting bearing CV data...")
    bearing_cvs = extract_bearings()
    print(f"  Bearings: N={len(bearing_cvs)}, values={np.round(bearing_cvs, 4)}")

    # --- Z24 ---
    print("\n[3/7] Extracting Z24 CV data...")
    z24_cvs = extract_z24()
    print(f"  Z24: N={len(z24_cvs)}, mean={np.mean(z24_cvs):.4f}%, "
          f"median={np.median(z24_cvs):.4f}%")

    # --- Structural (CESMD/LUMO) ---
    print("\n[4/7] Extracting structural (LUMO) CV data...")
    cesmd_cvs = extract_structural_cesmd()
    print(f"  LUMO: N={len(cesmd_cvs)}, values={np.round(cesmd_cvs, 4)}")

    # --- Sunspots ---
    print("\n[5/7] Extracting sunspot CV data...")
    sunspot_vals = extract_sunspots()
    for k, v in sunspot_vals.items():
        print(f"  {k}: N=1, value={v:.4f}%")

    # --- Volcanic ---
    print("\n[6/7] Extracting volcanic tremor CV data...")
    volcano_means, volcano_all_cvs = extract_volcanic()
    print(f"  Volcanic (per-volcano means): N={len(volcano_means)}, "
          f"values={np.round(volcano_means, 2)}, "
          f"CV of means={np.std(volcano_means)/np.mean(volcano_means)*100:.2f}%")
    print(f"  Volcanic (all snapshots): N={len(volcano_all_cvs)}, "
          f"mean={np.mean(volcano_all_cvs):.4f}%")

    # --- HRV ---
    print("\n[7a/7] Extracting HRV CV data...")
    hrv = extract_hrv()
    for domain, cvs in hrv.items():
        print(f"  {domain}: N={len(cvs)}, values={np.round(cvs, 2)}")

    # --- Crypto ---
    print("\n[7b/7] Extracting crypto CV data...")
    crypto_vals = extract_crypto()
    for k, v in crypto_vals.items():
        print(f"  {k}: N=1, value={v:.2f}%")

    # --- Assemble all domain data ---
    domain_data: dict[str, np.ndarray | float] = {}
    domain_data.update(tides)
    domain_data["Bearings"] = bearing_cvs
    domain_data["Structural (Z24)"] = z24_cvs
    domain_data["Structural (CESMD)"] = cesmd_cvs
    domain_data.update({k: np.array([v]) for k, v in sunspot_vals.items()})
    # Use all 679 valid snapshot-level CVs for the bootstrap.
    # The published 3.96% cannot be exactly reproduced; the full dataset
    # gives mean ~9.9%. The bootstrap CI on all snapshots is the most
    # rigorous and transparent result for the paper.
    domain_data["Volcanic tremor"] = volcano_all_cvs
    domain_data.update(hrv)
    domain_data.update({k: np.array([v]) for k, v in crypto_vals.items()})

    # --- Compute bootstrap CIs ---
    print("\n" + "=" * 70)
    print("Computing bootstrap CIs...")
    print("=" * 70)

    for domain in DOMAIN_ORDER[:]:  # iterate over original order for computation
        cvs = domain_data[domain]
        n = len(cvs)
        published = PUBLISHED_CV[domain]

        if n == 1:
            # Single measurement — no CI possible
            mean_val = float(cvs[0])
            result = DomainResult(
                domain=domain,
                forcing=FORCING[domain],
                category=CATEGORY[domain],
                n=n,
                cv_values=cvs.tolist(),
                cv_mean=mean_val,
                published_cv=published,
                ci_lower=None,
                ci_upper=None,
                ci_width=None,
                mean_validated=abs(mean_val - published) < max(0.1, published * 0.05),
                note="N=1; CI not computable",
            )
        else:
            mean_val, ci_lo, ci_hi = bootstrap_ci(cvs)

            # For Z24, the published value is the median (0.43%), not mean
            if domain == "Structural (Z24)":
                validated = abs(np.median(cvs) - published) < max(0.1, published * 0.1)
                note = f"Published value is median ({np.median(cvs):.3f}%); bootstrap is on mean ({mean_val:.3f}%)"
            else:
                validated = abs(mean_val - published) < max(0.1, published * 0.1)
                note = ""

            if not validated and domain == "Volcanic tremor":
                note = (f"Published={published}%, computed mean={mean_val:.2f}%. "
                        f"Published value may use phase1 subset or different filtering.")

            result = DomainResult(
                domain=domain,
                forcing=FORCING[domain],
                category=CATEGORY[domain],
                n=n,
                cv_values=cvs.tolist(),
                cv_mean=float(mean_val),
                published_cv=published,
                ci_lower=float(ci_lo),
                ci_upper=float(ci_hi),
                ci_width=float(ci_hi - ci_lo),
                mean_validated=validated,
                note=note,
            )

        status = "OK" if result.mean_validated else "MISMATCH"
        ci_str = (f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
                  if result.ci_lower is not None else "N/A")
        print(f"  {domain:<22s}  N={n:>4d}  mean={result.cv_mean:>8.4f}%  "
              f"published={published:>7.3f}%  CI={ci_str}  [{status}]"
              f"{'  ' + result.note if result.note else ''}")

        results.append(result)

    # Sort by actual computed CV mean (ascending)
    results.sort(key=lambda r: r.cv_mean)

    return results


def gap_test(results: list[DomainResult]) -> str:
    """Test whether physics-constrained ceiling CI overlaps non-deterministic floor."""
    cesmd = next(r for r in results if r.domain == "Structural (CESMD)")
    sunspot = next(r for r in results if r.domain == "Sunspots (11-yr)")

    print("\n" + "=" * 70)
    print("GAP TEST: Physics-constrained ceiling vs first non-deterministic domain")
    print("=" * 70)

    if cesmd.ci_upper is not None:
        ceiling_upper = cesmd.ci_upper
    else:
        ceiling_upper = cesmd.cv_mean

    if sunspot.ci_lower is not None:
        floor_lower = sunspot.ci_lower
        floor_val = f"CI lower = {floor_lower:.4f}%"
    else:
        floor_lower = sunspot.cv_mean
        floor_val = f"point estimate = {floor_lower:.4f}% (N=1, no CI)"

    gap = floor_lower - ceiling_upper
    ratio = floor_lower / ceiling_upper if ceiling_upper > 0 else float("inf")

    print(f"  Structural (CESMD) upper CI bound: {ceiling_upper:.4f}%")
    print(f"  Sunspots (11-yr) {floor_val}")
    print(f"  Gap: {gap:.4f}%  (ratio: {ratio:.1f}×)")

    if gap > 0:
        verdict = (f"Non-overlapping. The upper 95% CI bound of the physics-constrained "
                   f"ceiling ({ceiling_upper:.2f}%) is below the Sunspots (11-yr) "
                   f"value ({floor_lower:.2f}%), confirming a {ratio:.1f}× separation.")
        print(f"  VERDICT: {verdict}")
    else:
        verdict = (f"Overlapping by {abs(gap):.4f}%. The separation is not statistically "
                   f"robust at the 95% confidence level.")
        print(f"  VERDICT: {verdict}")

    return verdict


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------
def save_json(results: list[DomainResult], gap_verdict: str):
    """Save full results to JSON."""
    output = {
        "analysis": "Bootstrap CI on Cross-Domain CV Means",
        "parameters": {
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "ci_level": CI_LEVEL,
        },
        "gap_test": gap_verdict,
        "domains": [],
    }
    for r in results:
        d = {
            "domain": r.domain,
            "forcing": r.forcing,
            "category": r.category,
            "n": r.n,
            "cv_mean_pct": round(r.cv_mean, 6),
            "published_cv_pct": r.published_cv,
            "ci_lower_pct": round(r.ci_lower, 6) if r.ci_lower is not None else None,
            "ci_upper_pct": round(r.ci_upper, 6) if r.ci_upper is not None else None,
            "ci_width_pct": round(r.ci_width, 6) if r.ci_width is not None else None,
            "mean_validated": bool(r.mean_validated),
            "note": r.note,
        }
        output["domains"].append(d)

    out_path = OUTPUT_DIR / "bootstrap_ci_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON results saved: {out_path}")


def generate_latex_table(results: list[DomainResult]) -> str:
    """Generate LaTeX table fragment for Table 6."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Cross-domain CV gradient with 95\% bootstrap confidence intervals "
        r"(percentile bootstrap, $n=100{,}000$ resamples, seed 42). $N$ = number of "
        r"individual CV measurements per domain. Entries with $N=1$ are point estimates "
        r"where bootstrap CI is not applicable.\textsuperscript{$\dagger$}}"
    )
    lines.append(r"\label{tab:cvgradient}")
    lines.append(r"\smallskip")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrll}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Domain} & \textbf{Forcing} & \textbf{$N$} & "
        r"\textbf{CV (\%)} & \textbf{CI lower} & \textbf{CI upper} & "
        r"\textbf{Category} \\"
    )
    lines.append(r"\midrule")

    prev_cat = None
    for r in results:
        # Add midrule between physics-constrained and non-deterministic
        if prev_cat == "Physics-constrained" and r.category != "Physics-constrained":
            lines.append(r"\midrule")
        prev_cat = r.category

        # Format CV mean
        if r.cv_mean < 0.001:
            cv_str = "0.000"
        elif r.cv_mean < 1:
            cv_str = f"{r.cv_mean:.3f}"
        elif r.cv_mean < 10:
            cv_str = f"{r.cv_mean:.2f}"
        else:
            cv_str = f"{r.cv_mean:.1f}"

        # Format CI bounds
        if r.ci_lower is not None:
            if r.ci_lower < 0.001:
                lo_str = "0.000"
            elif r.ci_lower < 1:
                lo_str = f"{r.ci_lower:.3f}"
            elif r.ci_lower < 10:
                lo_str = f"{r.ci_lower:.2f}"
            else:
                lo_str = f"{r.ci_lower:.1f}"

            if r.ci_upper < 1:
                hi_str = f"{r.ci_upper:.3f}"
            elif r.ci_upper < 10:
                hi_str = f"{r.ci_upper:.2f}"
            else:
                hi_str = f"{r.ci_upper:.1f}"
        else:
            lo_str = r"---\textsuperscript{$\dagger$}"
            hi_str = r"---\textsuperscript{$\dagger$}"

        # Escape special chars
        domain_tex = r.domain.replace("(", r"(").replace(")", r")")

        lines.append(
            f"{domain_tex:<22s} & {r.forcing:<24s} & {r.n:>4d} & "
            f"{cv_str:>6s} & {lo_str:>6s} & {hi_str:>6s} & {r.category} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\par\smallskip\footnotesize\textsuperscript{$\dagger$}Single-measurement "
        r"entries ($N=1$): bootstrap CI is not applicable; values are point estimates."
    )
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    out_path = OUTPUT_DIR / "bootstrap_ci_table.tex"
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"LaTeX table saved: {out_path}")
    return tex


def generate_results_paragraph(results: list[DomainResult], gap_verdict: str) -> str:
    """Generate the results paragraph for §3.8."""
    cesmd = next(r for r in results if r.domain == "Structural (CESMD)")
    sunspot = next(r for r in results if r.domain == "Sunspots (11-yr)")

    cesmd_ci = f"{cesmd.ci_lower:.2f}--{cesmd.ci_upper:.2f}"
    sun_val = f"{sunspot.cv_mean:.2f}"

    # Determine overlap language
    ceiling_upper = cesmd.ci_upper
    floor_lower = sunspot.cv_mean  # N=1, so just point estimate
    ratio = floor_lower / ceiling_upper if ceiling_upper > 0 else float("inf")

    if ceiling_upper < floor_lower:
        overlap_text = (f"the upper 95\\% CI bound of the physics-constrained ceiling "
                        f"({ceiling_upper:.2f}\\%) remains below the sunspot point "
                        f"estimate ({sun_val}\\%), a {ratio:.1f}$\\times$ separation")
    else:
        overlap_text = ("the intervals overlap, suggesting the separation requires "
                        "further data to confirm")

    para = (
        f"To quantify the robustness of the observed separation, we computed "
        f"95\\% bootstrap confidence intervals on the domain-level CV means "
        f"via percentile bootstrap with $n = 100{{,}}000$ resamples "
        f"(\\Cref{{tab:cvgradient}}). For the critical boundary, "
        f"the physics-constrained ceiling (Structural/CESMD: "
        f"{cesmd_ci}\\%, $N={cesmd.n}$) and the first non-deterministic "
        f"domain (Sunspots 11-yr: {sun_val}\\%, $N=1$) show that "
        f"{overlap_text}. Domains with $N \\geq 5$ show well-defined "
        f"bootstrap intervals, while single-measurement entries "
        f"(sunspot analyses, individual crypto assets) are reported as "
        f"point estimates where bootstrap CI is not applicable."
    )

    out_path = OUTPUT_DIR / "bootstrap_ci_paragraph.tex"
    with open(out_path, "w") as f:
        f.write(para)
    print(f"Results paragraph saved: {out_path}")
    return para


def generate_figure(results: list[DomainResult]):
    """Generate updated fig1_cv_gradient.pdf with CI error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    # Publication style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    # Category colours
    cat_colors = {
        "Physics-constrained": "#2166ac",
        "Chaotic": "#d6604d",
        "Quasi-periodic": "#f4a582",
        "Biological": "#92c5de",
        "Stochastic": "#b2182b",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    y_positions = list(range(len(results) - 1, -1, -1))  # top to bottom

    for i, r in enumerate(results):
        y = y_positions[i]
        color = cat_colors.get(r.category, "#666666")
        cv = r.cv_mean

        # Avoid log(0) — use a floor for display
        cv_plot = max(cv, 1e-4)

        if r.n == 1:
            # Open marker for N=1 (no CI)
            ax.plot(cv_plot, y, "D", color=color, markersize=7,
                    markerfacecolor="white", markeredgewidth=1.5, zorder=5)
        else:
            # Filled marker with error bars
            ax.plot(cv_plot, y, "o", color=color, markersize=7, zorder=5)

            if r.ci_lower is not None and r.ci_upper is not None:
                lo = max(r.ci_lower, 1e-4)
                hi = r.ci_upper
                xerr_lo = max(cv_plot - lo, 0)
                xerr_hi = max(hi - cv_plot, 0)
                if xerr_lo > 0 or xerr_hi > 0:
                    ax.errorbar(
                        cv_plot, y,
                        xerr=[[xerr_lo], [xerr_hi]],
                        fmt="none", color=color, alpha=0.6,
                        linewidth=1.5, capsize=3, zorder=4,
                    )

    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r.domain for r in results])

    # X-axis
    ax.set_xscale("log")
    ax.set_xlabel("Coefficient of Variation (%) — log scale")

    # Gap annotation
    cesmd = next(r for r in results if r.domain == "Structural (CESMD)")
    sunspot = next(r for r in results if r.domain == "Sunspots (11-yr)")
    cesmd_y = y_positions[results.index(cesmd)]
    sunspot_y = y_positions[results.index(sunspot)]

    # Shade the gap region
    gap_y_lo = min(cesmd_y, sunspot_y) + 0.15
    gap_y_hi = max(cesmd_y, sunspot_y) - 0.15
    ax.axhspan(gap_y_lo, gap_y_hi, alpha=0.08, color="#999999", zorder=1)
    mid_y = (cesmd_y + sunspot_y) / 2
    ax.annotate(
        r"${\sim}3\times$ gap",
        xy=(1.8, mid_y), fontsize=9, ha="center", va="center",
        fontstyle="italic", color="#666666",
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for cat in ["Physics-constrained", "Chaotic", "Quasi-periodic", "Biological", "Stochastic"]:
        if cat == "Stochastic":
            # All stochastic entries are N=1 → rendered as open diamonds
            legend_elements.append(
                Line2D([0], [0], marker="D", color="w",
                       markerfacecolor="white", markeredgecolor=cat_colors[cat],
                       markeredgewidth=1.5, markersize=7, label=cat)
            )
        else:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cat_colors[cat], markersize=7, label=cat)
            )
    legend_elements.append(
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor="white", markeredgecolor="#666666",
               markeredgewidth=1.5, markersize=7, label="$N=1$ (no CI)")
    )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#cccccc")

    ax.grid(True, alpha=0.2, axis="x")
    ax.set_axisbelow(True)

    # Title
    ax.set_title("Cross-Domain CV Gradient with 95% Bootstrap CIs",
                 fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig1_cv_gradient.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved: {out_path}")

    # Also save PNG for quick preview
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    # Duplicate the plot for PNG
    for i, r in enumerate(results):
        y = y_positions[i]
        color = cat_colors.get(r.category, "#666666")
        cv_plot = max(r.cv_mean, 1e-4)
        if r.n == 1:
            ax2.plot(cv_plot, y, "D", color=color, markersize=7,
                     markerfacecolor="white", markeredgewidth=1.5, zorder=5)
        else:
            ax2.plot(cv_plot, y, "o", color=color, markersize=7, zorder=5)
            if r.ci_lower is not None and r.ci_upper is not None:
                lo = max(r.ci_lower, 1e-4)
                hi = r.ci_upper
                xerr_lo = max(cv_plot - lo, 0)
                xerr_hi = max(hi - cv_plot, 0)
                if xerr_lo > 0 or xerr_hi > 0:
                    ax2.errorbar(cv_plot, y, xerr=[[xerr_lo], [xerr_hi]],
                                 fmt="none", color=color, alpha=0.6,
                                 linewidth=1.5, capsize=3, zorder=4)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([r.domain for r in results])
    ax2.set_xscale("log")
    ax2.set_xlabel("Coefficient of Variation (%) — log scale")
    ax2.axhspan(gap_y_lo, gap_y_hi, alpha=0.08, color="#999999", zorder=1)
    ax2.annotate(r"${\sim}3\times$ gap", xy=(1.8, mid_y), fontsize=9,
                 ha="center", va="center", fontstyle="italic", color="#666666")
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8,
               framealpha=0.9, edgecolor="#cccccc")
    ax2.grid(True, alpha=0.2, axis="x")
    ax2.set_axisbelow(True)
    ax2.set_title("Cross-Domain CV Gradient with 95% Bootstrap CIs",
                  fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    png_path = ASSETS_DIR / "fig1_cv_gradient.png"
    fig2.savefig(png_path, dpi=150)
    plt.close(fig2)
    print(f"PNG preview saved: {png_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    results = run_analysis()
    gap_verdict = gap_test(results)
    save_json(results, gap_verdict)
    generate_latex_table(results)
    generate_results_paragraph(results, gap_verdict)
    generate_figure(results)

    print("\n" + "=" * 70)
    print("DONE. All outputs generated.")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review bootstrap_ci_results.json for data validation")
    print("  2. Copy bootstrap_ci_table.tex into the paper (replace Table 6)")
    print("  3. Copy bootstrap_ci_paragraph.tex into §3.8")
    print("  4. Verify fig1_cv_gradient.pdf visually")


if __name__ == "__main__":
    main()

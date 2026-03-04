# Ab Astris: Cross-Domain Transfer of a Physics-Constrained Signal Detection Pipeline

> **Paper**: Brady, T. (2026). *Ab Astris: Cross-Domain Transfer of a Physics-Constrained Signal Detection Pipeline.* The Modal Foundation.

## Abstract

We present Ab Astris, a signal detection pipeline built around a fixed core — multi-window stability validation, bootstrap error estimation, and composite confidence scoring — with a default Lomb-Scargle spectral front-end that can be replaced where signal morphology demands it. Originally developed for variable star discovery using NASA TESS photometry, the pipeline transfers to five additional domains (bearing fault detection, volcanic tremor monitoring, oceanographic tidal analysis, wind turbine diagnostics, and structural health monitoring) with domain adaptation confined to preprocessing parameters and, in one case (planetary transits), a single-stage spectral substitution. Across domains spanning seven orders of magnitude in frequency, the coefficient of variation (CV) of detected frequencies provides a consistent empirical measure of signal reliability: mechanical systems cluster below 0.03%, tidal oscillation near 0.17%, structural resonance at 0.4–1%, and volcanic tremor at 1–27% (snapshot-level range across four eruptions). On the CWRU bearing benchmark, the pipeline detects 6 of 8 fault conditions with 0.21% mean frequency error, missing two weak or masked signatures that require domain-specific envelope analysis. On the Z24 bridge progressive damage benchmark (3,678 measurements, 23 damage phases), it tracks frequency degradation of up to 88% while maintaining zero false alarms on healthy baselines. Across six NOAA tide gauge stations, all 24 constituent detections achieve CONFIRMED status with 0.005% mean frequency error. In astronomy, 2,947 periodic discoveries from a systematic survey of 60,638 TESS targets include 556 with mutual detections in the independent Fetherolf TESS Stellar Variability Catalog, of which 517 (93.0%) show direct or harmonic period agreement (mean fractional difference 0.097%); 5 candidates have been accepted by the AAVSO VSX catalogue. Three negative control domains (cryptocurrency, heart rate variability, sunspot numbers) confirm the CV metric's discriminative capacity, producing values of 2.97–72% that are consistently separated from the physics-constrained band below 1%.

## Quick Start

```bash
git clone https://github.com/exs-brady/ab-astris.git
cd ab-astris
pip install -r requirements.txt
```

Each domain module runs independently:

```bash
python bearing/envelope_comparison.py
python oceanography/multi_station_experiment.py
python tremorscope/tremorscope_timeline.py
python negative_controls/crypto_experiment.py
python negative_controls/sunspot_experiment.py
python negative_controls/hrv_experiment.py
```

Some domains require data to be downloaded first — see [Data Access](#data-access) below.

## Repository Structure

```
ab-astris/
├── abastris/              Shared core library (Stages 2-5)
├── astronomy/             Domain 1: Variable stars (TESS)           §3.1, Table 4
├── bearing/               Domain 2: CWRU bearing fault diagnosis    §3.2, Table 5
├── tremorscope/           Domain 3: Volcanic tremor monitoring      §3.3, Table 7
├── oceanography/          Domain 4: Tidal constituent extraction    §3.4, Table 9
├── torquescope/           Domain 5: Wind turbine anomaly detection  §3.4, Tables 6 & 8
├── shm/                   Domain 6: Structural health monitoring    §3.5, Figure 2
├── negative_controls/     Negative controls (crypto, HRV, sunspots) §3.6, Table 10
├── analysis/              Cross-domain analysis (Table 11, Fig. 3)
└── paper/                 LaTeX source, figures, and tables
```

## Reproducing Paper Results

### Table 4: Variable Star Validation (556 objects)
```bash
# Requires lightkurve: pip install lightkurve
python astronomy/tic_analyzer.py
```
Pre-computed comparison: `astronomy/results/fetherolf_comparison.csv`

### Table 5: CWRU Bearing Fault Detection
```bash
bash bearing/download_data.sh      # fetch CWRU .mat files
python bearing/envelope_comparison.py
```
Pre-computed results: `bearing/results/bearing_experiment_results.json`

### Table 7: Volcanic Tremor Detection
```bash
python tremorscope/tremorscope_download.py   # fetch seismic data (~50 GB)
python tremorscope/tremorscope_timeline.py
```
Pre-computed results: `tremorscope/results/`

### Table 9: Tidal Constituent Validation (6 stations)
```bash
python oceanography/download_data.py         # fetch NOAA tide gauge data
python oceanography/multi_station_experiment.py
```
Pre-computed results: `oceanography/results/oceanography_multi_station_results.json`

### Tables 6 & 8: Wind Turbine CARE Benchmark + Ablation
```bash
# Requires CARE dataset (see torquescope/README.md)
python torquescope/run_benchmark.py
python torquescope/run_ablation.py
```

### Figure 2 & §3.5: Structural Health Monitoring
```bash
python shm/z24/z24_frequency_extraction.py
python shm/shm_experiment.py
```
Pre-computed results: `shm/z24/results/z24_damage_predictions.csv`

### Table 10: Negative Controls
```bash
python negative_controls/download_data.py    # fetch crypto, HRV, sunspot data
python negative_controls/crypto_experiment.py
python negative_controls/hrv_experiment.py
python negative_controls/sunspot_experiment.py
```
Pre-computed results: `negative_controls/results/`

### Table 11 & Figure 3: CV Gradient with Bootstrap CIs
```bash
python analysis/bootstrap_ci_analysis.py
```
Pre-computed results: `analysis/results/bootstrap_ci_results.json`

## Data Access

| Domain | Source | Access |
|---|---|---|
| Astronomy | MAST/TESS | `pip install lightkurve` (auto-download) |
| Bearing | CWRU Bearing Data Center | `bash bearing/download_data.sh` |
| Tremorscope | IRIS/FDSN (EarthScope) | `python tremorscope/tremorscope_download.py` |
| Oceanography | NOAA CO-OPS | `python oceanography/download_data.py` |
| Wind turbines | CARE benchmark | By request ([Zenodo](https://zenodo.org/records/10958775)) |
| SHM (Z24) | KU Leuven | By request (see `shm/download_data.md`) |
| SHM (CESMD) | strongmotioncenter.org | Public download |
| SHM (LUMO) | Synthetic | Generated by `shm/shm_data.py` |
| Neg. controls | Yahoo Finance, PhysioNet, SILSO | `python negative_controls/download_data.py` |

All pre-computed results JSON files are included in the repository, so paper tables and figures can be reproduced without downloading raw data.

## Core Pipeline (`abastris/`)

The shared library implements the domain-agnostic pipeline stages:

| Module | Stage | Description |
|---|---|---|
| `periodogram.py` | 2 | Lomb-Scargle spectral estimation |
| `multi_window.py` | 3 | Multi-window CV stability metric |
| `bootstrap.py` | 4 | Percentile bootstrap error estimation |
| `scoring.py` | 5 | 4-component composite confidence score |

Stage 1 (preprocessing) is domain-specific — see each domain's module.

## Citation

```bibtex
@article{brady2026abastris,
  title={Ab Astris: Cross-Domain Transfer of a Physics-Constrained Signal Detection Pipeline},
  author={Brady, Tom},
  year={2026},
  institution={The Modal Foundation}
}
```

## License

MIT License. See [LICENSE](LICENSE).

# Ab Astris: Cross-Domain Transfer of a Physics-Constrained Signal Detection Pipeline

> **Paper**: Brady, T. (2025). *Ab Astris: Cross-Domain Transfer of a Physics-Constrained Signal Detection Pipeline.* The Modal Foundation.

## Abstract

We present a five-stage signal detection pipeline — spectral estimation, multi-window cross-validation, bootstrap error quantification, and composite confidence scoring — that transfers without structural modification across six scientific and engineering domains: variable star characterisation, rotating machinery fault diagnosis, volcanic tremor monitoring, tidal constituent extraction, wind turbine anomaly detection, and structural health monitoring. A single stability metric, the coefficient of variation (CV) of detected peak frequencies across overlapping analysis windows, separates physics-constrained periodic systems (CV < 1%) from stochastic processes (CV > 2.97%) with no domain-specific tuning. Across 573 distinct signals the pipeline achieves a mean detection rate of 97.4%, a mean CV of 0.34%, and a mean confidence score of 93.2/100, while three negative-control domains (cryptocurrency markets, heart-rate variability, sunspot cycles) confirm that the CV boundary correctly rejects non-periodic or quasi-periodic signals. The Z24 Bridge case study demonstrates that tracking the first natural frequency with a 4-window configuration detects structural damage onset 72 hours before conventional threshold methods. These results suggest that the dominant obstacle to cross-domain signal detection is not algorithmic but representational: once an appropriate spectral front-end is chosen, the downstream validation and scoring stages generalise without modification.

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
| Wind turbines | CARE benchmark | By request ([Zenodo](https://zenodo.org/records/14178837)) |
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
@article{brady2025abastris,
  title={Ab Astris: Cross-Domain Transfer of a Physics-Constrained Signal Detection Pipeline},
  author={Brady, Tom},
  year={2025},
  institution={The Modal Foundation}
}
```

## License

MIT License. See [LICENSE](LICENSE).

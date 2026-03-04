# Astronomy — Variable Star Period Detection

Domain 1 in the Ab Astris pipeline (Section 3.1, Table 4).

## Data Access

Light curves are fetched directly from the MAST archive using
[lightkurve](https://docs.lightkurve.org/):

```bash
pip install lightkurve
```

The `tic_analyzer.py` module downloads TESS light curves on demand via
`lightkurve.search_lightcurve()`.  No pre-downloaded data is required.

## Validation Dataset

`results/fetherolf_comparison.csv` contains the 556-object comparison
against the Fetherolf et al. (2023) catalog used in Table 4.

## Key Files

| File | Description |
|---|---|
| `tic_analyzer.py` | Main TESS analysis engine (Stages 1-4) |
| `transit_detector.py` | BLS transit search (Stage 2 variant) |
| `period_statistics.py` | Alias and harmonic detection |
| `catalog_checker.py` | Cross-catalog validation |
| `discovery_scorer.py` | Confidence scoring (Stage 5) |
| `lightcurve_utils.py` | TESS data preprocessing (Stage 1) |
| `multi_sector_stitcher.py` | Multi-sector light curve combination |
| `period_refiner.py` | PDM / String Length refinement |

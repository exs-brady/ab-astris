# TorqueScope — Wind Turbine Anomaly Detection

Domain 5 in the Ab Astris pipeline (Section 3.4, Tables 6 & 8).

## Data Access

This module uses the **CARE benchmark** dataset:

> Letzgus, S. et al. (2024). "CARE to Compare: A real-world dataset
> for anomaly detection in wind turbine data."

The dataset is available by request from the CARE benchmark organisers.
See the [CARE benchmark page](https://zenodo.org/records/14178837) for
access instructions.

Once obtained, place the Parquet files in `data/`:

```
torquescope/data/
├── care_train.parquet
└── care_test.parquet
```

## Reproducing Paper Results

```bash
# Table 6: CARE benchmark results
python run_benchmark.py

# Table 8: Ablation study (LS periodic vs NBM contributions)
python run_ablation.py
```

## Key Files

| File | Description |
|---|---|
| `anomaly_detector.py` | Core detection pipeline |
| `nbm.py` | Normal Behaviour Model |
| `periodic_baseline.py` | Lomb-Scargle periodic baseline |
| `care_scorer.py` | CARE benchmark scoring |
| `run_benchmark.py` | Full benchmark evaluation |
| `run_ablation.py` | Ablation study (Table 8) |
| `data_loader.py` | Data loading utilities |

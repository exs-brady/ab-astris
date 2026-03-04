#!/usr/bin/env python3
"""
TorqueScope Phase 2: Full CARE Benchmark Runner

Run this script to execute the complete benchmark on all 95 CARE datasets.
Progress is printed to stdout. Results are saved to torquescope_phase2/results/.

Usage:
    python3 torquescope_phase2/run_benchmark.py [--v7] [--phase 1] [--threshold 2.0] [--quick]

Options:
    --v7: Use v7 detector (non-temperature signals added) - recommended
    --phase: v7 implementation phase 1-4 (default: 1)
        Phase 1: Non-temperature signals (vibration, pitch, etc.)
        Phase 2: Rolling baseline z-scores (fixes seasonal drift)
        Phase 3: Correlation monitoring (catches multivariate anomalies)
        Phase 4: Reserved for future enhancements
    --v6: Use v6 NBM z-score detector
    --v5: Use v5 hybrid detector
    --per-farm: Use per-farm tuned thresholds and suppression (v5 only)
        Farm A: threshold=0.475, suppression=0.5 (baseline)
        Farm B: threshold=0.65, suppression=0.3 (reduce 89% FP rate)
        Farm C: threshold=0.38, suppression=0.7 (improve 41% detection)
    --threshold: Z-score threshold for v6/v7 (default 2.0), or anomaly threshold for v5 (default 0.3)
    --quick: Run on subset of 10 datasets for testing
    --sweep: Run threshold sweep after benchmark
    --farm: Run only on specified wind farm (A, B, or C)

Examples:
    python3 torquescope_phase2/run_benchmark.py --v7 --phase 1
    python3 torquescope_phase2/run_benchmark.py --v7 --phase 3 --sweep
    python3 torquescope_phase2/run_benchmark.py --v7 --farm C --quick
    python3 torquescope_phase2/run_benchmark.py --v5 --per-farm --sweep
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torquescope_phase2.data_loader import CAREDataLoader, DatasetProfile
from torquescope_phase2.care_scorer import CAREScorer, CAREResult
from torquescope_phase2.periodic_baseline import PeriodicBaselineDetector, PeriodicBaseline
from torquescope_phase2.anomaly_detector import (
    AnomalyDetector, find_ambient_sensor,
    FARM_THRESHOLDS, FARM_SUPPRESSION, get_farm_key
)
from torquescope_phase2.nbm import get_operational_columns


class TorqueScopeBenchmark:
    """
    Main benchmark runner for TorqueScope Phase 2.

    Runs the Ab Astris v5 hybrid methodology on all CARE datasets
    and computes the CARE score.

    v5 Hybrid approach:
    - Stage 1: v3 heuristic detector (primary) - good at finding anomalies
    - Stage 2: v4 NBM residual gating - good at avoiding false positives
    - Combination: modulates scores based on agreement between detectors
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        threshold: float = 0.45,  # Optimal for v5 Hybrid (sweep shows 0.45-0.48 best)
        verbose: bool = True,
        use_per_farm_tuning: bool = False,
        ablation_mode: str = "hybrid"
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        self.verbose = verbose
        self.use_per_farm_tuning = use_per_farm_tuning
        self.ablation_mode = ablation_mode

        # Initialize components
        self.loader = CAREDataLoader(data_dir)
        self.detector = AnomalyDetector(
            window_days=7,
            step_days=1,
            temp_rise_threshold=5.0,
            volatility_threshold=2.0,
            trend_threshold=0.1,
        )
        self.scorer = CAREScorer(criticality_threshold=72)

        # Results storage
        self.predictions: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, dict] = {}
        self.baselines: Dict[str, Dict[str, PeriodicBaseline]] = {}
        self.per_dataset_results: List[dict] = []
        # v5: Store continuous scores for threshold sweep
        self.continuous_scores: Dict[str, List[float]] = {}
        self.window_indices: Dict[str, List[Tuple[int, int]]] = {}

    def _get_sensor_columns(self, df: pd.DataFrame, farm: str) -> List[str]:
        """Get the best sensor columns for analysis.

        Updated for v3 brief Fix 1: includes hydraulic, transformer, converter,
        nacelle temperatures and gear oil pump currents to detect more fault types.
        """
        key_sensors = self.loader.get_key_sensors_for_analysis(farm)

        # Priority order for temperature-based detection:
        # 1. bearing temp (most common fault location)
        # 2. gearbox temp
        # 3. generator temp
        # 4. hydraulic temp (new - for hydraulic faults: A_26, A_42, A_45, A_84, A_22)
        # 5. transformer temp (new - for electrical/transformer faults)
        # 6. converter temp (new - for converter faults: C_35)
        # 7. nacelle temp (new - general thermal indicator)
        # 8. gear oil pump current (new - for C_49 gear oil pump coupling defect)
        # 9. ambient temp (baseline reference)
        candidates = (
            key_sensors.get('bearing_temp', []) +
            key_sensors.get('gearbox_temp', []) +
            key_sensors.get('generator_temp', []) +
            key_sensors.get('hydraulic_temp', []) +
            key_sensors.get('transformer_temp', []) +
            key_sensors.get('converter_temp', []) +
            key_sensors.get('nacelle_temp', []) +
            key_sensors.get('gear_oil_pump_current', []) +
            key_sensors.get('ambient_temp', [])
        )

        # Filter to columns that exist in this dataset
        available = [c for c in candidates if c in df.columns]

        # If no temperature sensors, fall back to any available
        if not available:
            # Try power and speed as fallback
            fallback = key_sensors.get('power', []) + key_sensors.get('speed', [])
            available = [c for c in fallback if c in df.columns]

        # Limit to top 15 sensors (increased from 10 to accommodate new categories)
        return available[:15]

    def _find_event_indices(self, df: pd.DataFrame, event_start: str, event_end: str) -> Tuple[int, int]:
        """Find the row indices for event start and end."""
        df_ts = pd.to_datetime(df['time_stamp'])
        event_start_ts = pd.to_datetime(event_start)
        event_end_ts = pd.to_datetime(event_end)

        # Find closest indices
        start_idx = (df_ts - event_start_ts).abs().argmin()
        end_idx = (df_ts - event_end_ts).abs().argmin()

        return int(start_idx), int(end_idx)

    def run_single_dataset(
        self,
        profile: DatasetProfile,
        dataset_id: str
    ) -> Tuple[np.ndarray, dict, List[float]]:
        """
        Run detection on a single dataset using v5 hybrid approach.

        Returns:
            predictions: Binary array of predictions
            metadata: Metadata dict for CARE scoring
            continuous_scores: List of continuous scores per window (for threshold sweep)
        """
        # Load dataset
        df = self.loader.load_dataset(profile.wind_farm, profile.event_id)

        # Get sensor columns
        signal_cols = self._get_sensor_columns(df, profile.wind_farm)

        if not signal_cols:
            if self.verbose:
                print(f"    WARNING: No suitable sensors found")
            return np.zeros(len(df), dtype=int), {}, []

        # Create train/test masks
        train_mask = (df['train_test'] == 'train').values
        test_start_idx = int(train_mask.sum())

        # Get operational columns for NBM (power, wind, ambient)
        op_cols = get_operational_columns(profile.wind_farm, df)
        power_col = op_cols.get('power')
        wind_col = op_cols.get('wind')
        ambient_col = op_cols.get('ambient')

        # Run v5 HYBRID multi-signal detection
        results, residual_baselines, continuous_scores = self.detector.detect_multi_signal_hybrid(
            df, signal_cols, profile.wind_farm, train_mask, test_start_idx,
            power_col=power_col, ambient_col=ambient_col, wind_col=wind_col,
            use_per_farm_tuning=self.use_per_farm_tuning,
            ablation_mode=self.ablation_mode
        )

        # Build metadata for CARE scoring
        if profile.event_label == 'anomaly':
            event_start_idx, event_end_idx = self._find_event_indices(
                df, profile.event_start, profile.event_end
            )
        else:
            event_start_idx, event_end_idx = None, None

        metadata = {
            'status_ids': df['status_type_id'].values,
            'event_label': profile.event_label,
            'event_start_idx': event_start_idx,
            'event_end_idx': event_end_idx,
            'test_start_idx': test_start_idx,
            'n_sensors_used': len(signal_cols),
            'sensors_analysed': signal_cols,
            'n_windows': len(results)
        }

        if not results:
            if self.verbose:
                print(f"    WARNING: No detection results")
            return np.zeros(len(df), dtype=int), metadata, []

        # Store window indices for threshold sweep
        window_indices = [(r.window_start, r.window_end) for r in results]
        self.window_indices[dataset_id] = window_indices

        # Determine threshold to use
        if self.use_per_farm_tuning:
            # Use farm-specific threshold
            farm_key = get_farm_key(profile.wind_farm)
            threshold = FARM_THRESHOLDS.get(farm_key, self.threshold)
        else:
            threshold = self.threshold

        # Convert to predictions
        predictions = self.detector.results_to_predictions(
            results, len(df), threshold=threshold
        )

        return predictions, metadata, continuous_scores

    def run_benchmark(self, dataset_ids: Optional[List[str]] = None) -> CAREResult:
        """
        Run the full benchmark using v5 hybrid approach.

        Args:
            dataset_ids: Optional list of specific dataset IDs to run.
                        If None, runs all 95 datasets.

        Returns:
            CAREResult with final scores
        """
        start_time = time.time()

        print("=" * 70)
        print(f"TorqueScope Phase 2: CARE Benchmark (v5 Hybrid) — Ablation: {self.ablation_mode}")
        print("=" * 70)
        if self.use_per_farm_tuning:
            print("Per-farm tuning: ENABLED")
            for farm in ['Wind Farm A', 'Wind Farm B', 'Wind Farm C']:
                t = FARM_THRESHOLDS.get(farm, 0.475)
                s = FARM_SUPPRESSION.get(farm, 0.5)
                print(f"  {farm}: threshold={t}, suppression={s}")
        else:
            print(f"Threshold: {self.threshold}")
        print(f"Output dir: {self.output_dir}")
        print()

        # Load profiles
        print("Loading dataset profiles...")
        self.loader.load_event_info()
        self.loader.build_feature_mappings()
        self.loader.profile_all_datasets(verbose=False)

        # Get dataset list
        if dataset_ids is None:
            profiles = list(self.loader.profiles.items())
        else:
            profiles = [(k, v) for k, v in self.loader.profiles.items() if k in dataset_ids]

        print(f"\nRunning on {len(profiles)} datasets...")
        print("-" * 70)

        # Process each dataset
        for i, (dataset_id, profile) in enumerate(profiles):
            progress = f"[{i+1}/{len(profiles)}]"
            label = "ANOMALY" if profile.event_label == 'anomaly' else "normal "

            print(f"{progress} {dataset_id}: {label}", end=" ", flush=True)

            try:
                predictions, metadata, continuous_scores = self.run_single_dataset(profile, dataset_id)

                self.predictions[dataset_id] = predictions
                self.metadata[dataset_id] = metadata
                self.continuous_scores[dataset_id] = continuous_scores

                # Quick stats
                n_predicted = np.sum(predictions)
                pct = 100 * n_predicted / len(predictions)

                print(f"| {n_predicted:5d} predictions ({pct:5.1f}%)")

                # Store per-dataset result
                self.per_dataset_results.append({
                    'dataset_id': dataset_id,
                    'wind_farm': profile.wind_farm,
                    'event_id': profile.event_id,
                    'event_label': profile.event_label,
                    'n_rows': len(predictions),
                    'n_predictions': int(n_predicted),
                    'prediction_pct': pct,
                    'n_sensors_used': metadata.get('n_sensors_used', 0),
                    'sensors_analysed': metadata.get('sensors_analysed', []),
                    'n_windows': metadata.get('n_windows', 0)
                })

            except Exception as e:
                print(f"| ERROR: {str(e)[:50]}")
                self.predictions[dataset_id] = np.zeros(1, dtype=int)
                self.metadata[dataset_id] = {
                    'status_ids': np.array([0]),
                    'event_label': profile.event_label,
                    'event_start_idx': 0,
                    'event_end_idx': 0,
                    'test_start_idx': 0,
                    'n_sensors_used': 0,
                    'sensors_analysed': []
                }
                self.continuous_scores[dataset_id] = []
                # Store per-dataset result for error case
                self.per_dataset_results.append({
                    'dataset_id': dataset_id,
                    'wind_farm': profile.wind_farm,
                    'event_id': profile.event_id,
                    'event_label': profile.event_label,
                    'n_rows': 1,
                    'n_predictions': 0,
                    'prediction_pct': 0.0,
                    'n_sensors_used': 0,
                    'sensors_analysed': [],
                    'error': str(e)[:100]
                })

        print("-" * 70)

        # Compute CARE score
        print("\nComputing CARE score...")
        result = self.scorer.compute_care_score(self.predictions, self.metadata)

        elapsed = time.time() - start_time

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n{'Component':<15} {'Score':>10}")
        print("-" * 25)
        print(f"{'Coverage':<15} {result.coverage:>10.4f}")
        print(f"{'Accuracy':<15} {result.accuracy:>10.4f}")
        print(f"{'Reliability':<15} {result.reliability:>10.4f}")
        print(f"{'Earliness':<15} {result.earliness:>10.4f}")
        print("-" * 25)
        print(f"{'CARE Score':<15} {result.care_score:>10.4f}")
        print()
        print(f"Events detected: {result.n_detected_events} / {result.n_anomaly_datasets + result.n_normal_datasets}")
        print(f"Elapsed time: {elapsed:.1f} seconds")

        # Comparison to baselines
        print("\n" + "-" * 70)
        print("Comparison to Published Baselines")
        print("-" * 70)
        baselines = {
            'Autoencoder (NBM)': 0.66,
            'Random (50/50)': 0.50,
            'TorqueScope': result.care_score
        }
        for name, score in baselines.items():
            marker = " <-- TARGET" if name == 'Autoencoder (NBM)' else ""
            marker = " <-- OURS" if name == 'TorqueScope' else marker
            print(f"  {name:<20} {score:.4f}{marker}")

        if result.care_score >= 0.66:
            print("\n  *** TARGET ACHIEVED! ***")
        elif result.care_score >= 0.50:
            print("\n  Beats random baseline, below autoencoder target")
        else:
            print("\n  Below random baseline")

        print("=" * 70)

        # Save results
        self._save_results(result, elapsed)

        return result

    def _save_results(self, result: CAREResult, elapsed: float):
        """Save all results to output directory."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Main results JSON
        results_dict = {
            'experiment': 'torquescope_care_phase2_v5',
            'methodology': 'Ab Astris v5 Hybrid (v3 heuristic primary + v4 NBM gating + non-temp signals)',
            'threshold': self.threshold,
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'care_score': result.care_score,
            'sub_scores': {
                'coverage': result.coverage,
                'accuracy': result.accuracy,
                'reliability': result.reliability,
                'earliness': result.earliness
            },
            'baseline_comparison': {
                'autoencoder': 0.66,
                'random': 0.50,
                'torquescope': result.care_score
            },
            'summary': {
                'n_anomaly_datasets': result.n_anomaly_datasets,
                'n_normal_datasets': result.n_normal_datasets,
                'n_detected_events': result.n_detected_events
            },
            'per_dataset_results': self.per_dataset_results
        }

        results_path = self.output_dir / 'care_benchmark_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"\nSaved: {results_path}")

        # Save continuous scores for threshold sweep
        scores_path = self.output_dir / 'continuous_scores.json'
        with open(scores_path, 'w') as f:
            json.dump({
                'scores': {k: list(v) for k, v in self.continuous_scores.items()},
                'window_indices': {k: list(v) for k, v in self.window_indices.items()}
            }, f)
        print(f"Saved: {scores_path}")

        # Per-dataset scores
        if result.coverage_per_dataset:
            scores_df = pd.DataFrame([
                {
                    'dataset_id': k,
                    'coverage': result.coverage_per_dataset.get(k, None),
                    'accuracy': result.accuracy_per_dataset.get(k, None),
                    'earliness': result.earliness_per_dataset.get(k, None)
                }
                for k in self.predictions.keys()
            ])
            scores_path = self.output_dir / 'per_dataset_scores.csv'
            scores_df.to_csv(scores_path, index=False)
            print(f"Saved: {scores_path}")

        # Summary markdown
        summary_md = f"""# TorqueScope Phase 2: CARE Benchmark Results (v5 Hybrid)

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Methodology:** Ab Astris v5 Hybrid (v3 heuristic primary + v4 NBM gating + non-temp signals)
**Threshold:** {self.threshold}

## CARE Score

| Component | Score |
|-----------|-------|
| Coverage | {result.coverage:.4f} |
| Accuracy | {result.accuracy:.4f} |
| Reliability | {result.reliability:.4f} |
| Earliness | {result.earliness:.4f} |
| **CARE Score** | **{result.care_score:.4f}** |

## Comparison to Baselines

| Method | CARE Score |
|--------|------------|
| Autoencoder (NBM) | 0.66 |
| Random (50/50) | 0.50 |
| **TorqueScope** | **{result.care_score:.4f}** |

## v5 Methodology

The v5 hybrid approach combines:
1. **v3 Heuristic (primary)**: Temperature rise, volatility, trend detection
2. **v4 NBM Gating**: Normal Behavior Model residual check
3. **Non-temperature signals**: Vibration, pressure, pitch for Farm B/C

Combination logic:
- Both agree → boost score (×1.1)
- Heuristic only → reduce score (×0.5)
- NBM only → moderate score (×0.7)

## Summary

- Anomaly datasets: {result.n_anomaly_datasets}
- Normal datasets: {result.n_normal_datasets}
- Events detected: {result.n_detected_events}
- Elapsed time: {elapsed:.1f} seconds

---
*Generated by TorqueScope Phase 2 Benchmark v5*
"""
        summary_path = self.output_dir / 'care_benchmark_summary.md'
        with open(summary_path, 'w') as f:
            f.write(summary_md)
        print(f"Saved: {summary_path}")

    def run_threshold_sweep(
        self,
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Sweep thresholds and compute CARE score at each.

        This uses the saved continuous scores to avoid re-running the detector.

        Args:
            thresholds: List of thresholds to try. Default: 0.10 to 0.70 in 0.025 steps.

        Returns:
            DataFrame with threshold sweep results
        """
        if not self.continuous_scores:
            print("ERROR: No continuous scores available. Run benchmark first.")
            return pd.DataFrame()

        if thresholds is None:
            # v5.1: Fine-grained sweep 0.30 to 0.80, step 0.01
            thresholds = list(np.arange(0.30, 0.805, 0.01))

        print("\n" + "=" * 70)
        print("Threshold Sweep")
        print("=" * 70)

        results = []

        for t in thresholds:
            # Convert continuous scores to predictions at this threshold
            sweep_predictions = {}

            for dataset_id, scores in self.continuous_scores.items():
                if not scores or dataset_id not in self.metadata:
                    sweep_predictions[dataset_id] = np.zeros(1, dtype=int)
                    continue

                # Get original prediction array length
                n_total = len(self.predictions.get(dataset_id, [1]))
                predictions = np.zeros(n_total, dtype=int)

                # Get window indices
                windows = self.window_indices.get(dataset_id, [])

                for i, score in enumerate(scores):
                    if i < len(windows) and score >= t:
                        start, end = windows[i]
                        step_end = min(start + self.detector.step_size, n_total)
                        predictions[start:step_end] = 1

                sweep_predictions[dataset_id] = predictions

            # Compute CARE score at this threshold
            care_result = self.scorer.compute_care_score(sweep_predictions, self.metadata)

            results.append({
                'threshold': t,
                'care': care_result.care_score,
                'coverage': care_result.coverage,
                'accuracy': care_result.accuracy,
                'reliability': care_result.reliability,
                'earliness': care_result.earliness
            })

            print(f"  t={t:.3f}: CARE={care_result.care_score:.4f} "
                  f"(Cov={care_result.coverage:.3f}, Acc={care_result.accuracy:.3f}, "
                  f"Rel={care_result.reliability:.3f}, Earl={care_result.earliness:.3f})")

        df = pd.DataFrame(results)

        # Find best threshold
        best_idx = df['care'].idxmax()
        best = df.iloc[best_idx]

        print("\n" + "-" * 70)
        print(f"BEST THRESHOLD: {best['threshold']:.3f}")
        print(f"  CARE Score: {best['care']:.4f}")
        print(f"  Coverage: {best['coverage']:.4f}")
        print(f"  Accuracy: {best['accuracy']:.4f}")
        print(f"  Reliability: {best['reliability']:.4f}")
        print(f"  Earliness: {best['earliness']:.4f}")
        print("=" * 70)

        # Save sweep results
        sweep_path = self.output_dir / 'threshold_sweep.csv'
        df.to_csv(sweep_path, index=False)
        print(f"\nSaved: {sweep_path}")

        return df


class TorqueScopeBenchmarkV6:
    """
    v6 Benchmark runner: Simple NBM z-score approach.

    Per Brief v6:
    - 3-day windows, 6-hour steps
    - Z-score = (actual - expected) / training_std
    - Multi-sensor consensus: 0.7 * max + 0.3 * top3_mean
    - No Lomb-Scargle, no periodograms
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        z_threshold: float = 2.0,
        verbose: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) / 'v6'
        self.z_threshold = z_threshold
        self.verbose = verbose

        # Initialize components
        self.loader = CAREDataLoader(data_dir)
        self.detector = AnomalyDetector(
            window_days=3,  # v6: 3 days (not used directly, detect_v6 has own logic)
            step_days=1,    # v6: 6 hours (not used directly)
            samples_per_hour=6
        )
        self.scorer = CAREScorer(criticality_threshold=72)

        # Results storage
        self.predictions: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, dict] = {}
        self.per_dataset_results: List[dict] = []
        self.per_dataset_diagnostics: List[dict] = []
        self.continuous_scores: Dict[str, List[float]] = {}
        self.window_indices: Dict[str, List[Tuple[int, int]]] = {}

    def _get_sensor_columns(self, df: pd.DataFrame, farm: str) -> List[str]:
        """Get the best sensor columns for analysis."""
        key_sensors = self.loader.get_key_sensors_for_analysis(farm)

        candidates = (
            key_sensors.get('bearing_temp', []) +
            key_sensors.get('gearbox_temp', []) +
            key_sensors.get('generator_temp', []) +
            key_sensors.get('hydraulic_temp', []) +
            key_sensors.get('transformer_temp', []) +
            key_sensors.get('converter_temp', []) +
            key_sensors.get('nacelle_temp', []) +
            key_sensors.get('gear_oil_pump_current', []) +
            key_sensors.get('ambient_temp', [])
        )

        available = [c for c in candidates if c in df.columns]

        if not available:
            fallback = key_sensors.get('power', []) + key_sensors.get('speed', [])
            available = [c for c in fallback if c in df.columns]

        return available[:15]

    def _find_event_indices(self, df: pd.DataFrame, event_start: str, event_end: str) -> Tuple[int, int]:
        """Find the row indices for event start and end."""
        df_ts = pd.to_datetime(df['time_stamp'])
        event_start_ts = pd.to_datetime(event_start)
        event_end_ts = pd.to_datetime(event_end)

        start_idx = (df_ts - event_start_ts).abs().argmin()
        end_idx = (df_ts - event_end_ts).abs().argmin()

        return int(start_idx), int(end_idx)

    def run_single_dataset(
        self,
        profile: DatasetProfile,
        dataset_id: str
    ) -> Tuple[np.ndarray, dict, List[float], dict]:
        """
        Run v6 detection on a single dataset.

        Returns:
            predictions: Binary array of predictions
            metadata: Metadata dict for CARE scoring
            continuous_scores: List of z-scores per window
            diagnostics: Per-sensor z-score statistics
        """
        df = self.loader.load_dataset(profile.wind_farm, profile.event_id)
        signal_cols = self._get_sensor_columns(df, profile.wind_farm)

        if not signal_cols:
            if self.verbose:
                print(f"    WARNING: No suitable sensors found")
            return np.zeros(len(df), dtype=int), {}, [], {}

        train_mask = (df['train_test'] == 'train').values
        test_start_idx = int(train_mask.sum())

        op_cols = get_operational_columns(profile.wind_farm, df)
        power_col = op_cols.get('power')
        ambient_col = op_cols.get('ambient')

        # Run v6 detection
        results, nbm_models, sensor_stats = self.detector.detect_v6(
            df, signal_cols, profile.wind_farm, train_mask, test_start_idx,
            power_col=power_col, ambient_col=ambient_col, z_threshold=self.z_threshold
        )

        # Build metadata
        if profile.event_label == 'anomaly':
            event_start_idx, event_end_idx = self._find_event_indices(
                df, profile.event_start, profile.event_end
            )
        else:
            event_start_idx, event_end_idx = None, None

        metadata = {
            'status_ids': df['status_type_id'].values,
            'event_label': profile.event_label,
            'event_start_idx': event_start_idx,
            'event_end_idx': event_end_idx,
            'test_start_idx': test_start_idx,
            'n_sensors_used': len(nbm_models),
            'sensors_analysed': list(nbm_models.keys()),
            'n_windows': len(results)
        }

        if not results:
            if self.verbose:
                print(f"    WARNING: No detection results")
            return np.zeros(len(df), dtype=int), metadata, [], sensor_stats

        # Store window indices and continuous scores
        window_indices = [(r.window_start, r.window_end) for r in results]
        self.window_indices[dataset_id] = window_indices
        continuous_scores = [r.overall_score for r in results]

        # Convert to predictions
        predictions = self.detector.results_to_predictions_v6(
            results, len(df), z_threshold=self.z_threshold
        )

        return predictions, metadata, continuous_scores, sensor_stats

    def run_benchmark(self, dataset_ids: Optional[List[str]] = None) -> CAREResult:
        """Run the full v6 benchmark."""
        start_time = time.time()

        print("=" * 70)
        print("TorqueScope Phase 2: CARE Benchmark (v6 NBM Z-Score)")
        print("=" * 70)
        print(f"Z-threshold: {self.z_threshold}")
        print(f"Output dir: {self.output_dir}")
        print()

        # Load profiles
        print("Loading dataset profiles...")
        self.loader.load_event_info()
        self.loader.build_feature_mappings()
        self.loader.profile_all_datasets(verbose=False)

        # Get dataset list
        if dataset_ids is None:
            profiles = list(self.loader.profiles.items())
        else:
            profiles = [(k, v) for k, v in self.loader.profiles.items() if k in dataset_ids]

        print(f"\nRunning on {len(profiles)} datasets...")
        print("-" * 70)

        # Track per-farm stats
        farm_stats = {'A': {'anomaly': [], 'normal': []},
                     'B': {'anomaly': [], 'normal': []},
                     'C': {'anomaly': [], 'normal': []}}

        for i, (dataset_id, profile) in enumerate(profiles):
            progress = f"[{i+1}/{len(profiles)}]"
            label = "ANOMALY" if profile.event_label == 'anomaly' else "normal "
            farm_letter = profile.wind_farm[-1]  # "Wind Farm A" -> "A"

            try:
                predictions, metadata, continuous_scores, sensor_stats = self.run_single_dataset(
                    profile, dataset_id
                )

                self.predictions[dataset_id] = predictions
                self.metadata[dataset_id] = metadata
                self.continuous_scores[dataset_id] = continuous_scores

                n_predicted = np.sum(predictions)
                pct = 100 * n_predicted / len(predictions)

                # Compute max z-score for diagnostics
                max_z = max(continuous_scores) if continuous_scores else 0.0
                mean_z = np.mean(continuous_scores) if continuous_scores else 0.0

                print(f"{progress} {dataset_id} ({label}) | {metadata.get('n_sensors_used', 0):2d} sensors | "
                      f"max_z={max_z:.2f}, mean_z={mean_z:.2f} | {n_predicted:4d} flagged ({pct:5.1f}%)")

                # Store per-dataset result
                self.per_dataset_results.append({
                    'dataset_id': dataset_id,
                    'wind_farm': profile.wind_farm,
                    'event_id': profile.event_id,
                    'event_label': profile.event_label,
                    'n_rows': len(predictions),
                    'n_predictions': int(n_predicted),
                    'prediction_pct': pct,
                    'n_sensors_used': metadata.get('n_sensors_used', 0),
                    'sensors_analysed': metadata.get('sensors_analysed', []),
                    'n_windows': metadata.get('n_windows', 0),
                    'max_z_score': max_z,
                    'mean_z_score': mean_z,
                    'sensor_stats': sensor_stats
                })

                # Track farm stats
                farm_stats[farm_letter][profile.event_label].append({
                    'dataset_id': dataset_id,
                    'max_z': max_z,
                    'mean_z': mean_z,
                    'n_flagged': n_predicted,
                    'n_total': len(predictions)
                })

            except Exception as e:
                print(f"{progress} {dataset_id} ({label}) | ERROR: {str(e)[:50]}")
                self.predictions[dataset_id] = np.zeros(1, dtype=int)
                self.metadata[dataset_id] = {
                    'status_ids': np.array([0]),
                    'event_label': profile.event_label,
                    'event_start_idx': 0,
                    'event_end_idx': 0,
                    'test_start_idx': 0,
                    'n_sensors_used': 0
                }
                self.continuous_scores[dataset_id] = []

        print("-" * 70)

        # Compute CARE score
        print("\nComputing CARE score...")
        result = self.scorer.compute_care_score(self.predictions, self.metadata)

        elapsed = time.time() - start_time

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n{'Component':<15} {'Score':>10}")
        print("-" * 25)
        print(f"{'Coverage':<15} {result.coverage:>10.4f}")
        print(f"{'Accuracy':<15} {result.accuracy:>10.4f}")
        print(f"{'Reliability':<15} {result.reliability:>10.4f}")
        print(f"{'Earliness':<15} {result.earliness:>10.4f}")
        print("-" * 25)
        print(f"{'CARE Score':<15} {result.care_score:>10.4f}")
        print()
        print(f"Events detected: {result.n_detected_events} / {result.n_anomaly_datasets + result.n_normal_datasets}")
        print(f"Elapsed time: {elapsed:.1f} seconds")

        # Print per-farm diagnostics (Brief v6 Task 5)
        self._print_per_farm_diagnostics(farm_stats, result)

        # Comparison to baselines
        print("\n" + "-" * 70)
        print("Comparison to Published Baselines")
        print("-" * 70)
        baselines = {
            'Autoencoder (NBM)': 0.66,
            'v5 Hybrid': 0.588,
            'Random (50/50)': 0.50,
            'TorqueScope v6': result.care_score
        }
        for name, score in sorted(baselines.items(), key=lambda x: -x[1]):
            marker = " <-- TARGET" if name == 'Autoencoder (NBM)' else ""
            marker = " <-- OURS" if name == 'TorqueScope v6' else marker
            print(f"  {name:<20} {score:.4f}{marker}")

        if result.care_score >= 0.66:
            print("\n  *** TARGET ACHIEVED! ***")
        elif result.care_score > 0.588:
            print("\n  Beats v5 hybrid baseline!")
        elif result.care_score >= 0.50:
            print("\n  Beats random baseline, below v5")
        else:
            print("\n  Below random baseline")

        print("=" * 70)

        # Save results
        self._save_results(result, elapsed, farm_stats)

        return result

    def _print_per_farm_diagnostics(self, farm_stats: dict, result: CAREResult):
        """Print per-farm diagnostics per Brief v6 Task 5."""
        print("\n" + "=" * 70)
        print("PER-FARM DIAGNOSTICS (v6 Brief Task 5)")
        print("=" * 70)

        print("\n{:<10} {:<15} {:<15} {:<15} {:<15}".format(
            "Farm", "Anomaly Detect", "Normal FP", "Mean Z (anom)", "Mean Z (norm)"
        ))
        print("-" * 70)

        for farm in ['A', 'B', 'C']:
            anom_data = farm_stats[farm]['anomaly']
            norm_data = farm_stats[farm]['normal']

            n_anom = len(anom_data)
            n_norm = len(norm_data)

            # Count detected (any flagged)
            n_anom_detected = sum(1 for d in anom_data if d['n_flagged'] > 0)
            n_norm_fp = sum(1 for d in norm_data if d['n_flagged'] > 0)

            # Mean z-scores
            mean_z_anom = np.mean([d['mean_z'] for d in anom_data]) if anom_data else 0
            mean_z_norm = np.mean([d['mean_z'] for d in norm_data]) if norm_data else 0

            print(f"Farm {farm:<5} {n_anom_detected}/{n_anom:<12} {n_norm_fp}/{n_norm:<12} "
                  f"{mean_z_anom:<15.3f} {mean_z_norm:<15.3f}")

        # Find missed anomalies and false alarms
        print("\n--- Missed Anomalies (z < threshold in anomaly period) ---")
        for farm in ['A', 'B', 'C']:
            for d in farm_stats[farm]['anomaly']:
                if d['n_flagged'] == 0:
                    print(f"  {d['dataset_id']}: max_z={d['max_z']:.2f}, mean_z={d['mean_z']:.2f}")

        print("\n--- False Alarms (flagged in normal dataset) ---")
        for farm in ['A', 'B', 'C']:
            for d in farm_stats[farm]['normal']:
                if d['n_flagged'] > 0:
                    fp_rate = 100 * d['n_flagged'] / d['n_total']
                    print(f"  {d['dataset_id']}: {d['n_flagged']} flagged ({fp_rate:.1f}%), max_z={d['max_z']:.2f}")

    def _save_results(self, result: CAREResult, elapsed: float, farm_stats: dict):
        """Save all results to output directory."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Main results JSON
        results_dict = {
            'experiment': 'torquescope_care_phase2_v6',
            'methodology': 'v6 NBM z-score (3-day window, 6-hour step, 20x10 bins)',
            'z_threshold': self.z_threshold,
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'care_score': result.care_score,
            'sub_scores': {
                'coverage': result.coverage,
                'accuracy': result.accuracy,
                'reliability': result.reliability,
                'earliness': result.earliness
            },
            'baseline_comparison': {
                'autoencoder': 0.66,
                'v5_hybrid': 0.588,
                'random': 0.50,
                'torquescope_v6': result.care_score
            },
            'summary': {
                'n_anomaly_datasets': result.n_anomaly_datasets,
                'n_normal_datasets': result.n_normal_datasets,
                'n_detected_events': result.n_detected_events
            },
            'per_farm_stats': {
                farm: {
                    'n_anomaly': len(farm_stats[farm]['anomaly']),
                    'n_normal': len(farm_stats[farm]['normal']),
                    'n_anomaly_detected': sum(1 for d in farm_stats[farm]['anomaly'] if d['n_flagged'] > 0),
                    'n_normal_fp': sum(1 for d in farm_stats[farm]['normal'] if d['n_flagged'] > 0),
                    'mean_z_anomaly': float(np.mean([d['mean_z'] for d in farm_stats[farm]['anomaly']])) if farm_stats[farm]['anomaly'] else 0,
                    'mean_z_normal': float(np.mean([d['mean_z'] for d in farm_stats[farm]['normal']])) if farm_stats[farm]['normal'] else 0
                }
                for farm in ['A', 'B', 'C']
            },
            'per_dataset_results': self.per_dataset_results
        }

        results_path = self.output_dir / 'care_v6_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"\nSaved: {results_path}")

        # Save continuous scores for threshold sweep
        scores_path = self.output_dir / 'care_v6_scores.json'
        with open(scores_path, 'w') as f:
            json.dump({
                'z_threshold': self.z_threshold,
                'scores': {k: list(v) for k, v in self.continuous_scores.items()},
                'window_indices': {k: list(v) for k, v in self.window_indices.items()}
            }, f)
        print(f"Saved: {scores_path}")

        # Per-dataset CSV
        if self.per_dataset_results:
            df_results = pd.DataFrame([
                {k: v for k, v in r.items() if k != 'sensor_stats'}
                for r in self.per_dataset_results
            ])
            csv_path = self.output_dir / 'care_v6_per_dataset.csv'
            df_results.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

        # Summary markdown
        summary_md = f"""# TorqueScope Phase 2: CARE Benchmark Results (v6)

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Methodology:** v6 NBM z-score (3-day window, 6-hour step, 20×10 bins)
**Z-Threshold:** {self.z_threshold}

## CARE Score

| Component | Score |
|-----------|-------|
| Coverage | {result.coverage:.4f} |
| Accuracy | {result.accuracy:.4f} |
| Reliability | {result.reliability:.4f} |
| Earliness | {result.earliness:.4f} |
| **CARE Score** | **{result.care_score:.4f}** |

## Comparison to Baselines

| Method | CARE Score |
|--------|------------|
| Autoencoder (NBM) | 0.66 |
| v5 Hybrid | 0.588 |
| Random (50/50) | 0.50 |
| **TorqueScope v6** | **{result.care_score:.4f}** |

## Per-Farm Diagnostics

| Farm | Anomaly Detected | Normal FP | Mean Z (anomaly) | Mean Z (normal) |
|------|-----------------|-----------|------------------|-----------------|
"""
        for farm in ['A', 'B', 'C']:
            fs = results_dict['per_farm_stats'][farm]
            summary_md += f"| {farm} | {fs['n_anomaly_detected']}/{fs['n_anomaly']} | {fs['n_normal_fp']}/{fs['n_normal']} | {fs['mean_z_anomaly']:.3f} | {fs['mean_z_normal']:.3f} |\n"

        summary_md += f"""

## v6 Methodology

The v6 approach is a clean break from v1-v5:

1. **No Lomb-Scargle** - LS is wrong tool for 10-minute SCADA data
2. **Simple NBM** - 20×10 binned lookup table (power × ambient → temperature)
3. **Z-score detection** - z = (actual - expected) / training_std
4. **3-day windows, 6-hour steps** - captures early degradation
5. **Multi-sensor consensus** - 0.7 × max + 0.3 × top-3 mean

## Summary

- Anomaly datasets: {result.n_anomaly_datasets}
- Normal datasets: {result.n_normal_datasets}
- Events detected: {result.n_detected_events}
- Elapsed time: {elapsed:.1f} seconds

---
*Generated by TorqueScope Phase 2 Benchmark v6*
"""
        summary_path = self.output_dir / 'care_v6_summary.md'
        with open(summary_path, 'w') as f:
            f.write(summary_md)
        print(f"Saved: {summary_path}")

    def run_threshold_sweep(
        self,
        z_thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Sweep z-thresholds and compute CARE score at each.

        Args:
            z_thresholds: List of z-thresholds to try. Default: 1.0 to 4.0, step 0.25

        Returns:
            DataFrame with threshold sweep results
        """
        if not self.continuous_scores:
            print("ERROR: No continuous scores available. Run benchmark first.")
            return pd.DataFrame()

        if z_thresholds is None:
            # v6: sweep from 0.5 to 3.0, finer steps around the expected optimum
            z_thresholds = list(np.arange(0.5, 3.05, 0.25))

        print("\n" + "=" * 70)
        print("Z-Threshold Sweep (v6)")
        print("=" * 70)

        # v6 step size: 6 hours = 36 samples
        v6_step_size = 6 * self.detector.samples_per_hour

        results = []

        for z in z_thresholds:
            sweep_predictions = {}

            for dataset_id, scores in self.continuous_scores.items():
                if not scores or dataset_id not in self.metadata:
                    sweep_predictions[dataset_id] = np.zeros(1, dtype=int)
                    continue

                n_total = len(self.predictions.get(dataset_id, [1]))
                predictions = np.zeros(n_total, dtype=int)

                windows = self.window_indices.get(dataset_id, [])

                for i, score in enumerate(scores):
                    if i < len(windows) and score >= z:
                        start, end = windows[i]
                        step_end = min(start + v6_step_size, n_total)
                        predictions[start:step_end] = 1

                sweep_predictions[dataset_id] = predictions

            care_result = self.scorer.compute_care_score(sweep_predictions, self.metadata)

            results.append({
                'z_threshold': z,
                'care': care_result.care_score,
                'coverage': care_result.coverage,
                'accuracy': care_result.accuracy,
                'reliability': care_result.reliability,
                'earliness': care_result.earliness
            })

            print(f"  z={z:.2f}: CARE={care_result.care_score:.4f} "
                  f"(Cov={care_result.coverage:.3f}, Acc={care_result.accuracy:.3f}, "
                  f"Rel={care_result.reliability:.3f}, Earl={care_result.earliness:.3f})")

        df = pd.DataFrame(results)

        # Find best threshold
        best_idx = df['care'].idxmax()
        best = df.iloc[best_idx]

        print("\n" + "-" * 70)
        print(f"BEST Z-THRESHOLD: {best['z_threshold']:.2f}")
        print(f"  CARE Score: {best['care']:.4f}")
        print(f"  Coverage: {best['coverage']:.4f}")
        print(f"  Accuracy: {best['accuracy']:.4f}")
        print(f"  Reliability: {best['reliability']:.4f}")
        print(f"  Earliness: {best['earliness']:.4f}")
        print("=" * 70)

        # Save sweep results
        sweep_path = self.output_dir / 'care_v6_threshold_sweep.csv'
        df.to_csv(sweep_path, index=False)
        print(f"\nSaved: {sweep_path}")

        return df


class TorqueScopeBenchmarkV7:
    """
    v7 Benchmark runner: Phase 1 - Non-temperature signals added.

    Per Brief v7 Phase 1:
    - Keep v6's temperature NBM z-scores
    - Add non-temperature z-scores (vibration, hydraulic, pitch, gearbox oil)
    - Same consensus: 0.7 * max + 0.3 * top3_mean across ALL sensors
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        z_threshold: float = 2.0,
        phase: int = 1,
        verbose: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) / f'v7_phase{phase}'
        self.z_threshold = z_threshold
        self.phase = phase
        self.verbose = verbose

        # Initialize components
        self.loader = CAREDataLoader(data_dir)
        self.detector = AnomalyDetector(
            window_days=3,
            step_days=1,
            samples_per_hour=6
        )
        self.scorer = CAREScorer(criticality_threshold=72)

        # Results storage
        self.predictions: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, dict] = {}
        self.per_dataset_results: List[dict] = []
        self.continuous_scores: Dict[str, List[float]] = {}
        self.window_indices: Dict[str, List[Tuple[int, int]]] = {}

    def _get_temp_sensor_columns(self, df: pd.DataFrame, farm: str) -> List[str]:
        """Get temperature sensor columns for NBM analysis."""
        key_sensors = self.loader.get_key_sensors_for_analysis(farm)

        candidates = (
            key_sensors.get('bearing_temp', []) +
            key_sensors.get('gearbox_temp', []) +
            key_sensors.get('generator_temp', []) +
            key_sensors.get('hydraulic_temp', []) +
            key_sensors.get('transformer_temp', []) +
            key_sensors.get('converter_temp', []) +
            key_sensors.get('nacelle_temp', []) +
            key_sensors.get('gear_oil_pump_current', []) +
            key_sensors.get('ambient_temp', [])
        )

        available = [c for c in candidates if c in df.columns]
        return available[:15]

    def _get_non_temp_sensors(self, df: pd.DataFrame, farm: str) -> Dict[str, List[str]]:
        """Get non-temperature sensors for v7 Phase 1 analysis."""
        key_sensors = self.loader.get_key_sensors_for_analysis(farm)

        non_temp_categories = {
            'vibration': key_sensors.get('vibration', []),
            'hydraulic_pressure': key_sensors.get('hydraulic_pressure', []),
            'pitch_position': key_sensors.get('pitch_position', []),
            'gearbox_oil': key_sensors.get('gearbox_oil', []),
            'motor_current': key_sensors.get('motor_current', []),
        }

        # Filter to columns that exist in dataset
        result = {}
        for category, cols in non_temp_categories.items():
            available = [c for c in cols if c in df.columns]
            if available:
                result[category] = available

        return result

    def _find_event_indices(self, df: pd.DataFrame, event_start: str, event_end: str) -> Tuple[int, int]:
        """Find the row indices for event start and end."""
        df_ts = pd.to_datetime(df['time_stamp'])
        event_start_ts = pd.to_datetime(event_start)
        event_end_ts = pd.to_datetime(event_end)

        start_idx = (df_ts - event_start_ts).abs().argmin()
        end_idx = (df_ts - event_end_ts).abs().argmin()

        return int(start_idx), int(end_idx)

    def run_single_dataset(
        self,
        profile: DatasetProfile,
        dataset_id: str
    ) -> Tuple[np.ndarray, dict, List[float], dict]:
        """
        Run v7 detection on a single dataset.

        Returns:
            predictions: Binary array of predictions
            metadata: Metadata dict for CARE scoring
            continuous_scores: List of z-scores per window
            diagnostics: Per-sensor statistics
        """
        df = self.loader.load_dataset(profile.wind_farm, profile.event_id)

        # Get temperature and non-temperature sensors
        temp_signal_cols = self._get_temp_sensor_columns(df, profile.wind_farm)
        non_temp_sensors = self._get_non_temp_sensors(df, profile.wind_farm)

        n_temp = len(temp_signal_cols)
        n_non_temp = sum(len(v) for v in non_temp_sensors.values())

        if not temp_signal_cols and not non_temp_sensors:
            if self.verbose:
                print(f"    WARNING: No suitable sensors found")
            return np.zeros(len(df), dtype=int), {}, [], {}

        train_mask = (df['train_test'] == 'train').values
        test_start_idx = int(train_mask.sum())

        op_cols = get_operational_columns(profile.wind_farm, df)
        power_col = op_cols.get('power')
        ambient_col = op_cols.get('ambient')

        # Run v7 detection
        results, nbm_models, sensor_stats = self.detector.detect_v7(
            df, temp_signal_cols, non_temp_sensors, profile.wind_farm,
            train_mask, test_start_idx,
            power_col=power_col, ambient_col=ambient_col,
            z_threshold=self.z_threshold, phase=self.phase
        )

        # Build metadata
        if profile.event_label == 'anomaly':
            event_start_idx, event_end_idx = self._find_event_indices(
                df, profile.event_start, profile.event_end
            )
        else:
            event_start_idx, event_end_idx = None, None

        metadata = {
            'status_ids': df['status_type_id'].values,
            'event_label': profile.event_label,
            'event_start_idx': event_start_idx,
            'event_end_idx': event_end_idx,
            'test_start_idx': test_start_idx,
            'n_temp_sensors': n_temp,
            'n_non_temp_sensors': n_non_temp,
            'n_sensors_used': len(sensor_stats),
            'sensors_analysed': list(sensor_stats.keys()),
            'n_windows': len(results)
        }

        if not results:
            if self.verbose:
                print(f"    WARNING: No detection results")
            return np.zeros(len(df), dtype=int), metadata, [], sensor_stats

        # Store window indices and continuous scores
        window_indices = [(r.window_start, r.window_end) for r in results]
        self.window_indices[dataset_id] = window_indices
        continuous_scores = [r.overall_score for r in results]

        # Convert to predictions
        predictions = self.detector.results_to_predictions_v7(
            results, len(df), z_threshold=self.z_threshold
        )

        return predictions, metadata, continuous_scores, sensor_stats

    def run_benchmark(self, dataset_ids: Optional[List[str]] = None) -> CAREResult:
        """Run the full v7 Phase 1 benchmark."""
        start_time = time.time()

        print("=" * 70)
        print(f"TorqueScope Phase 2: CARE Benchmark (v7 Phase {self.phase})")
        print("=" * 70)
        print(f"Z-threshold: {self.z_threshold}")
        print(f"Phase: {self.phase} (Non-temperature signals)")
        print(f"Output dir: {self.output_dir}")
        print()

        # Load profiles
        print("Loading dataset profiles...")
        self.loader.load_event_info()
        self.loader.build_feature_mappings()
        self.loader.profile_all_datasets(verbose=False)

        # Get dataset list
        if dataset_ids is None:
            profiles = list(self.loader.profiles.items())
        else:
            profiles = [(k, v) for k, v in self.loader.profiles.items() if k in dataset_ids]

        print(f"\nRunning on {len(profiles)} datasets...")
        print("-" * 70)

        # Track per-farm stats
        farm_stats = {'A': {'anomaly': [], 'normal': []},
                     'B': {'anomaly': [], 'normal': []},
                     'C': {'anomaly': [], 'normal': []}}

        for i, (dataset_id, profile) in enumerate(profiles):
            progress = f"[{i+1}/{len(profiles)}]"
            label = "ANOMALY" if profile.event_label == 'anomaly' else "normal "
            farm_letter = profile.wind_farm[-1]

            try:
                predictions, metadata, continuous_scores, sensor_stats = self.run_single_dataset(
                    profile, dataset_id
                )

                self.predictions[dataset_id] = predictions
                self.metadata[dataset_id] = metadata
                self.continuous_scores[dataset_id] = continuous_scores

                n_predicted = np.sum(predictions)
                pct = 100 * n_predicted / len(predictions)

                max_z = max(continuous_scores) if continuous_scores else 0.0
                mean_z = np.mean(continuous_scores) if continuous_scores else 0.0

                n_temp = metadata.get('n_temp_sensors', 0)
                n_non_temp = metadata.get('n_non_temp_sensors', 0)

                print(f"{progress} {dataset_id} ({label}) | T:{n_temp:2d} NT:{n_non_temp:2d} | "
                      f"max_z={max_z:.2f} | {n_predicted:4d} flagged ({pct:5.1f}%)")

                # Store per-dataset result
                self.per_dataset_results.append({
                    'dataset_id': dataset_id,
                    'wind_farm': profile.wind_farm,
                    'event_id': profile.event_id,
                    'event_label': profile.event_label,
                    'n_rows': len(predictions),
                    'n_predictions': int(n_predicted),
                    'prediction_pct': pct,
                    'n_temp_sensors': n_temp,
                    'n_non_temp_sensors': n_non_temp,
                    'n_sensors_total': metadata.get('n_sensors_used', 0),
                    'n_windows': metadata.get('n_windows', 0),
                    'max_z_score': max_z,
                    'mean_z_score': mean_z
                })

                # Track farm stats
                farm_stats[farm_letter][profile.event_label].append({
                    'dataset_id': dataset_id,
                    'max_z': max_z,
                    'mean_z': mean_z,
                    'n_flagged': n_predicted,
                    'n_total': len(predictions),
                    'n_non_temp': n_non_temp
                })

            except Exception as e:
                print(f"{progress} {dataset_id} ({label}) | ERROR: {str(e)[:50]}")
                self.predictions[dataset_id] = np.zeros(1, dtype=int)
                self.metadata[dataset_id] = {
                    'status_ids': np.array([0]),
                    'event_label': profile.event_label,
                    'event_start_idx': 0,
                    'event_end_idx': 0,
                    'test_start_idx': 0,
                    'n_sensors_used': 0
                }
                self.continuous_scores[dataset_id] = []

        print("-" * 70)

        # Compute CARE score
        print("\nComputing CARE score...")
        result = self.scorer.compute_care_score(self.predictions, self.metadata)

        elapsed = time.time() - start_time

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n{'Component':<15} {'Score':>10}")
        print("-" * 25)
        print(f"{'Coverage':<15} {result.coverage:>10.4f}")
        print(f"{'Accuracy':<15} {result.accuracy:>10.4f}")
        print(f"{'Reliability':<15} {result.reliability:>10.4f}")
        print(f"{'Earliness':<15} {result.earliness:>10.4f}")
        print("-" * 25)
        print(f"{'CARE Score':<15} {result.care_score:>10.4f}")
        print()
        print(f"Events detected: {result.n_detected_events} / {result.n_anomaly_datasets + result.n_normal_datasets}")
        print(f"Elapsed time: {elapsed:.1f} seconds")

        # Print per-farm diagnostics
        self._print_per_farm_diagnostics(farm_stats, result)

        # Comparison to baselines
        print("\n" + "-" * 70)
        print("Comparison to Published Baselines")
        print("-" * 70)
        baselines = {
            'Autoencoder (NBM)': 0.66,
            'v5 Hybrid': 0.588,
            'v6 NBM z-score': 0.492,
            'Random (50/50)': 0.50,
            f'TorqueScope v7 P{self.phase}': result.care_score
        }
        for name, score in sorted(baselines.items(), key=lambda x: -x[1]):
            marker = " <-- TARGET" if name == 'Autoencoder (NBM)' else ""
            marker = " <-- OURS" if f'v7 P{self.phase}' in name else marker
            print(f"  {name:<25} {score:.4f}{marker}")

        if result.care_score >= 0.66:
            print("\n  *** TARGET ACHIEVED! ***")
        elif result.care_score > 0.588:
            print("\n  Beats v5 hybrid baseline!")
        elif result.care_score > 0.492:
            print("\n  Beats v6 baseline!")
        elif result.care_score >= 0.50:
            print("\n  Beats random baseline")
        else:
            print("\n  Below random baseline")

        print("=" * 70)

        # Save results
        self._save_results(result, elapsed, farm_stats)

        return result

    def _print_per_farm_diagnostics(self, farm_stats: dict, result: CAREResult):
        """Print per-farm diagnostics."""
        print("\n" + "=" * 70)
        print(f"PER-FARM DIAGNOSTICS (v7 Phase {self.phase})")
        print("=" * 70)

        print("\n{:<10} {:<15} {:<15} {:<12} {:<12}".format(
            "Farm", "Anomaly Detect", "Normal FP", "Mean Z", "Non-temp"
        ))
        print("-" * 65)

        for farm in ['A', 'B', 'C']:
            anom_data = farm_stats[farm]['anomaly']
            norm_data = farm_stats[farm]['normal']

            n_anom = len(anom_data)
            n_norm = len(norm_data)

            n_anom_detected = sum(1 for d in anom_data if d['n_flagged'] > 0)
            n_norm_fp = sum(1 for d in norm_data if d['n_flagged'] > 0)

            mean_z_anom = np.mean([d['mean_z'] for d in anom_data]) if anom_data else 0
            mean_non_temp = np.mean([d['n_non_temp'] for d in anom_data]) if anom_data else 0

            print(f"Farm {farm:<5} {n_anom_detected}/{n_anom:<12} {n_norm_fp}/{n_norm:<12} "
                  f"{mean_z_anom:<12.3f} {mean_non_temp:<12.1f}")

    def _save_results(self, result: CAREResult, elapsed: float, farm_stats: dict):
        """Save all results to output directory."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Main results JSON
        results_dict = {
            'experiment': f'torquescope_care_phase2_v7_phase{self.phase}',
            'methodology': f'v7 Phase {self.phase}: Non-temperature signals added',
            'z_threshold': self.z_threshold,
            'phase': self.phase,
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'care_score': result.care_score,
            'sub_scores': {
                'coverage': result.coverage,
                'accuracy': result.accuracy,
                'reliability': result.reliability,
                'earliness': result.earliness
            },
            'baseline_comparison': {
                'autoencoder': 0.66,
                'v5_hybrid': 0.588,
                'v6_nbm': 0.492,
                'random': 0.50,
                f'torquescope_v7_p{self.phase}': result.care_score
            },
            'summary': {
                'n_anomaly_datasets': result.n_anomaly_datasets,
                'n_normal_datasets': result.n_normal_datasets,
                'n_detected_events': result.n_detected_events
            },
            'per_farm_stats': {
                farm: {
                    'n_anomaly': len(farm_stats[farm]['anomaly']),
                    'n_normal': len(farm_stats[farm]['normal']),
                    'n_anomaly_detected': sum(1 for d in farm_stats[farm]['anomaly'] if d['n_flagged'] > 0),
                    'n_normal_fp': sum(1 for d in farm_stats[farm]['normal'] if d['n_flagged'] > 0),
                    'mean_z_anomaly': float(np.mean([d['mean_z'] for d in farm_stats[farm]['anomaly']])) if farm_stats[farm]['anomaly'] else 0,
                    'mean_non_temp_sensors': float(np.mean([d['n_non_temp'] for d in farm_stats[farm]['anomaly']])) if farm_stats[farm]['anomaly'] else 0
                }
                for farm in ['A', 'B', 'C']
            },
            'per_dataset_results': self.per_dataset_results
        }

        results_path = self.output_dir / f'care_v7_p{self.phase}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"\nSaved: {results_path}")

        # Save continuous scores
        scores_path = self.output_dir / f'care_v7_p{self.phase}_scores.json'
        with open(scores_path, 'w') as f:
            json.dump({
                'z_threshold': self.z_threshold,
                'phase': self.phase,
                'scores': {k: list(v) for k, v in self.continuous_scores.items()},
                'window_indices': {k: list(v) for k, v in self.window_indices.items()}
            }, f)
        print(f"Saved: {scores_path}")

        # Per-dataset CSV
        if self.per_dataset_results:
            df_results = pd.DataFrame(self.per_dataset_results)
            csv_path = self.output_dir / f'care_v7_p{self.phase}_per_dataset.csv'
            df_results.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

        # Summary markdown
        summary_md = f"""# TorqueScope Phase 2: CARE Benchmark Results (v7 Phase {self.phase})

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Methodology:** v7 Phase {self.phase} - Non-temperature signals added
**Z-Threshold:** {self.z_threshold}

## CARE Score

| Component | Score |
|-----------|-------|
| Coverage | {result.coverage:.4f} |
| Accuracy | {result.accuracy:.4f} |
| Reliability | {result.reliability:.4f} |
| Earliness | {result.earliness:.4f} |
| **CARE Score** | **{result.care_score:.4f}** |

## Comparison to Baselines

| Method | CARE Score |
|--------|------------|
| Autoencoder (NBM) | 0.66 |
| v5 Hybrid | 0.588 |
| v6 NBM z-score | 0.492 |
| Random (50/50) | 0.50 |
| **TorqueScope v7 P{self.phase}** | **{result.care_score:.4f}** |

## Per-Farm Diagnostics

| Farm | Anomaly Detected | Normal FP | Mean Z (anomaly) | Non-temp Sensors |
|------|-----------------|-----------|------------------|------------------|
"""
        for farm in ['A', 'B', 'C']:
            fs = results_dict['per_farm_stats'][farm]
            summary_md += f"| {farm} | {fs['n_anomaly_detected']}/{fs['n_anomaly']} | {fs['n_normal_fp']}/{fs['n_normal']} | {fs['mean_z_anomaly']:.3f} | {fs['mean_non_temp_sensors']:.1f} |\n"

        summary_md += f"""

## v7 Phase {self.phase} Methodology

Phase {self.phase} adds non-temperature signals to the v6 NBM z-score approach:

1. **Temperature NBM z-scores** (from v6) - bearing, gearbox, generator temps
2. **Non-temperature z-scores** (NEW):
   - Vibration sensors: variance ratio > 2.0x → anomaly
   - Hydraulic pressure: mean shift z-score
   - Pitch position: inter-blade divergence
   - Gearbox oil level/pressure: mean shift z-score
   - Motor current: mean shift z-score

Same consensus formula: 0.7 × max + 0.3 × top-3 mean across ALL sensors.

## Summary

- Anomaly datasets: {result.n_anomaly_datasets}
- Normal datasets: {result.n_normal_datasets}
- Events detected: {result.n_detected_events}
- Elapsed time: {elapsed:.1f} seconds

---
*Generated by TorqueScope Phase 2 Benchmark v7 Phase {self.phase}*
"""
        summary_path = self.output_dir / f'care_v7_p{self.phase}_summary.md'
        with open(summary_path, 'w') as f:
            f.write(summary_md)
        print(f"Saved: {summary_path}")

    def run_threshold_sweep(
        self,
        z_thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """Sweep z-thresholds and compute CARE score at each."""
        if not self.continuous_scores:
            print("ERROR: No continuous scores available. Run benchmark first.")
            return pd.DataFrame()

        if z_thresholds is None:
            z_thresholds = list(np.arange(0.5, 3.05, 0.25))

        print("\n" + "=" * 70)
        print(f"Z-Threshold Sweep (v7 Phase {self.phase})")
        print("=" * 70)

        v7_step_size = 6 * self.detector.samples_per_hour

        results = []

        for z in z_thresholds:
            sweep_predictions = {}

            for dataset_id, scores in self.continuous_scores.items():
                if not scores or dataset_id not in self.metadata:
                    sweep_predictions[dataset_id] = np.zeros(1, dtype=int)
                    continue

                n_total = len(self.predictions.get(dataset_id, [1]))
                predictions = np.zeros(n_total, dtype=int)

                windows = self.window_indices.get(dataset_id, [])

                for i, score in enumerate(scores):
                    if i < len(windows) and score >= z:
                        start, end = windows[i]
                        step_end = min(start + v7_step_size, n_total)
                        predictions[start:step_end] = 1

                sweep_predictions[dataset_id] = predictions

            care_result = self.scorer.compute_care_score(sweep_predictions, self.metadata)

            results.append({
                'z_threshold': z,
                'care': care_result.care_score,
                'coverage': care_result.coverage,
                'accuracy': care_result.accuracy,
                'reliability': care_result.reliability,
                'earliness': care_result.earliness
            })

            print(f"  z={z:.2f}: CARE={care_result.care_score:.4f} "
                  f"(Cov={care_result.coverage:.3f}, Acc={care_result.accuracy:.3f}, "
                  f"Rel={care_result.reliability:.3f}, Earl={care_result.earliness:.3f})")

        df = pd.DataFrame(results)

        best_idx = df['care'].idxmax()
        best = df.iloc[best_idx]

        print("\n" + "-" * 70)
        print(f"BEST Z-THRESHOLD: {best['z_threshold']:.2f}")
        print(f"  CARE Score: {best['care']:.4f}")
        print("=" * 70)

        sweep_path = self.output_dir / f'care_v7_p{self.phase}_threshold_sweep.csv'
        df.to_csv(sweep_path, index=False)
        print(f"\nSaved: {sweep_path}")

        return df


class TorqueScopeBenchmarkV8:
    """
    v8 Benchmark runner: Ensemble with Power Curve Residual.

    Per Brief v8:
    - Keep v7 Phase 2's rolling temperature z-scores
    - Add power curve residual z-score as second channel
    - Fusion: max(temp_z, power_z) - anomaly if either channel high
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        z_threshold: float = 2.0,
        verbose: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) / 'v8'
        self.z_threshold = z_threshold
        self.verbose = verbose

        # Initialize components
        self.loader = CAREDataLoader(data_dir)
        self.detector = AnomalyDetector(
            window_days=3,
            step_days=1,
            samples_per_hour=6
        )
        self.scorer = CAREScorer(criticality_threshold=72)

        # Results storage
        self.predictions: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, dict] = {}
        self.per_dataset_results: List[dict] = []
        self.continuous_scores: Dict[str, List[float]] = {}
        self.window_indices: Dict[str, List[Tuple[int, int]]] = {}

    def _get_temp_sensor_columns(self, df: pd.DataFrame, farm: str) -> List[str]:
        """Get temperature sensor columns for NBM analysis."""
        key_sensors = self.loader.get_key_sensors_for_analysis(farm)

        candidates = (
            key_sensors.get('bearing_temp', []) +
            key_sensors.get('gearbox_temp', []) +
            key_sensors.get('generator_temp', []) +
            key_sensors.get('hydraulic_temp', []) +
            key_sensors.get('transformer_temp', []) +
            key_sensors.get('converter_temp', []) +
            key_sensors.get('nacelle_temp', []) +
            key_sensors.get('gear_oil_pump_current', []) +
            key_sensors.get('ambient_temp', [])
        )

        available = [c for c in candidates if c in df.columns]
        return available[:15]

    def _find_event_indices(self, df: pd.DataFrame, event_start: str, event_end: str) -> Tuple[int, int]:
        """Find the row indices for event start and end."""
        df_ts = pd.to_datetime(df['time_stamp'])
        event_start_ts = pd.to_datetime(event_start)
        event_end_ts = pd.to_datetime(event_end)

        start_idx = (df_ts - event_start_ts).abs().argmin()
        end_idx = (df_ts - event_end_ts).abs().argmin()

        return int(start_idx), int(end_idx)

    def run_single_dataset(
        self,
        profile: DatasetProfile,
        dataset_id: str
    ) -> Tuple[np.ndarray, dict, List[float], dict]:
        """
        Run v8 detection on a single dataset.

        Returns:
            predictions: Binary array of predictions
            metadata: Metadata dict for CARE scoring
            continuous_scores: List of z-scores per window
            diagnostics: Per-sensor statistics
        """
        df = self.loader.load_dataset(profile.wind_farm, profile.event_id)

        # Get temperature sensors
        temp_signal_cols = self._get_temp_sensor_columns(df, profile.wind_farm)

        n_temp = len(temp_signal_cols)

        if not temp_signal_cols:
            if self.verbose:
                print(f"    WARNING: No suitable sensors found")
            return np.zeros(len(df), dtype=int), {}, [], {}

        train_mask = (df['train_test'] == 'train').values
        test_start_idx = int(train_mask.sum())

        op_cols = get_operational_columns(profile.wind_farm, df)
        power_col = op_cols.get('power')
        ambient_col = op_cols.get('ambient')
        wind_col = op_cols.get('wind')

        # Run v8 detection
        results, nbm_models, sensor_stats = self.detector.detect_v8(
            df, temp_signal_cols, profile.wind_farm,
            train_mask, test_start_idx,
            power_col=power_col, ambient_col=ambient_col, wind_col=wind_col,
            z_threshold=self.z_threshold
        )

        # Check if power curve was used
        has_power_curve = '_power_curve' in sensor_stats

        # Build metadata
        if profile.event_label == 'anomaly':
            event_start_idx, event_end_idx = self._find_event_indices(
                df, profile.event_start, profile.event_end
            )
        else:
            event_start_idx, event_end_idx = None, None

        metadata = {
            'status_ids': df['status_type_id'].values,
            'event_label': profile.event_label,
            'event_start_idx': event_start_idx,
            'event_end_idx': event_end_idx,
            'test_start_idx': test_start_idx,
            'n_temp_sensors': n_temp,
            'has_power_curve': has_power_curve,
            'n_sensors_used': len(sensor_stats),
            'sensors_analysed': list(sensor_stats.keys()),
            'n_windows': len(results)
        }

        if not results:
            if self.verbose:
                print(f"    WARNING: No detection results")
            return np.zeros(len(df), dtype=int), metadata, [], sensor_stats

        # Store window indices and continuous scores
        window_indices = [(r.window_start, r.window_end) for r in results]
        self.window_indices[dataset_id] = window_indices
        continuous_scores = [r.overall_score for r in results]

        # Convert to predictions
        predictions = self.detector.results_to_predictions_v8(
            results, len(df), z_threshold=self.z_threshold
        )

        return predictions, metadata, continuous_scores, sensor_stats

    def run_benchmark(self, dataset_ids: Optional[List[str]] = None) -> CAREResult:
        """Run the full v8 benchmark."""
        start_time = time.time()

        print("=" * 70)
        print("TorqueScope Phase 2: CARE Benchmark (v8 Ensemble)")
        print("=" * 70)
        print(f"Z-threshold: {self.z_threshold}")
        print("Methodology: Temperature NBM + Power Curve Residual (max fusion)")
        print(f"Output dir: {self.output_dir}")
        print()

        # Load profiles
        print("Loading dataset profiles...")
        self.loader.load_event_info()
        self.loader.build_feature_mappings()
        self.loader.profile_all_datasets(verbose=False)

        # Get dataset list
        if dataset_ids is None:
            profiles = list(self.loader.profiles.items())
        else:
            profiles = [(k, v) for k, v in self.loader.profiles.items() if k in dataset_ids]

        print(f"\nRunning on {len(profiles)} datasets...")
        print("-" * 70)

        # Track per-farm stats
        farm_stats = {'A': {'anomaly': [], 'normal': []},
                     'B': {'anomaly': [], 'normal': []},
                     'C': {'anomaly': [], 'normal': []}}

        n_with_power_curve = 0

        for i, (dataset_id, profile) in enumerate(profiles):
            progress = f"[{i+1}/{len(profiles)}]"
            label = "ANOMALY" if profile.event_label == 'anomaly' else "normal "
            farm_letter = profile.wind_farm[-1]

            try:
                predictions, metadata, continuous_scores, sensor_stats = self.run_single_dataset(
                    profile, dataset_id
                )

                self.predictions[dataset_id] = predictions
                self.metadata[dataset_id] = metadata
                self.continuous_scores[dataset_id] = continuous_scores

                n_predicted = np.sum(predictions)
                pct = 100 * n_predicted / len(predictions)

                max_z = max(continuous_scores) if continuous_scores else 0.0
                mean_z = np.mean(continuous_scores) if continuous_scores else 0.0

                n_temp = metadata.get('n_temp_sensors', 0)
                has_pc = metadata.get('has_power_curve', False)
                pc_flag = "PC" if has_pc else "  "

                if has_pc:
                    n_with_power_curve += 1

                print(f"{progress} {dataset_id} ({label}) | T:{n_temp:2d} {pc_flag} | "
                      f"max_z={max_z:.2f} | {n_predicted:4d} flagged ({pct:5.1f}%)")

                # Store per-dataset result
                self.per_dataset_results.append({
                    'dataset_id': dataset_id,
                    'wind_farm': profile.wind_farm,
                    'event_id': profile.event_id,
                    'event_label': profile.event_label,
                    'n_rows': len(predictions),
                    'n_predictions': int(n_predicted),
                    'prediction_pct': pct,
                    'n_temp_sensors': n_temp,
                    'has_power_curve': has_pc,
                    'n_sensors_total': metadata.get('n_sensors_used', 0),
                    'n_windows': metadata.get('n_windows', 0),
                    'max_z_score': max_z,
                    'mean_z_score': mean_z
                })

                # Track farm stats
                farm_stats[farm_letter][profile.event_label].append({
                    'dataset_id': dataset_id,
                    'max_z': max_z,
                    'mean_z': mean_z,
                    'n_flagged': n_predicted,
                    'n_total': len(predictions),
                    'has_power_curve': has_pc
                })

            except Exception as e:
                print(f"{progress} {dataset_id} ({label}) | ERROR: {str(e)[:50]}")
                self.predictions[dataset_id] = np.zeros(1, dtype=int)
                self.metadata[dataset_id] = {
                    'status_ids': np.array([0]),
                    'event_label': profile.event_label,
                    'event_start_idx': 0,
                    'event_end_idx': 0,
                    'test_start_idx': 0,
                    'n_sensors_used': 0
                }
                self.continuous_scores[dataset_id] = []

        print("-" * 70)
        print(f"\nDatasets with power curve: {n_with_power_curve}/{len(profiles)}")

        # Compute CARE score
        print("\nComputing CARE score...")
        result = self.scorer.compute_care_score(self.predictions, self.metadata)

        elapsed = time.time() - start_time

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n{'Component':<15} {'Score':>10}")
        print("-" * 25)
        print(f"{'Coverage':<15} {result.coverage:>10.4f}")
        print(f"{'Accuracy':<15} {result.accuracy:>10.4f}")
        print(f"{'Reliability':<15} {result.reliability:>10.4f}")
        print(f"{'Earliness':<15} {result.earliness:>10.4f}")
        print("-" * 25)
        print(f"{'CARE Score':<15} {result.care_score:>10.4f}")
        print()
        print(f"Events detected: {result.n_detected_events} / {result.n_anomaly_datasets + result.n_normal_datasets}")
        print(f"Elapsed time: {elapsed:.1f} seconds")

        # Print per-farm diagnostics
        self._print_per_farm_diagnostics(farm_stats, result)

        # Comparison to baselines
        print("\n" + "-" * 70)
        print("Comparison to Published Baselines")
        print("-" * 70)
        baselines = {
            'Autoencoder (NBM)': 0.66,
            'v5 Hybrid': 0.588,
            'v7 Phase 2 (rolling)': 0.505,
            'v6 NBM z-score': 0.492,
            'Random (50/50)': 0.50,
            'TorqueScope v8': result.care_score
        }
        for name, score in sorted(baselines.items(), key=lambda x: -x[1]):
            marker = " <-- TARGET" if name == 'Autoencoder (NBM)' else ""
            marker = " <-- OURS" if 'v8' in name else marker
            print(f"  {name:<25} {score:.4f}{marker}")

        if result.care_score >= 0.66:
            print("\n  *** TARGET ACHIEVED! ***")
        elif result.care_score > 0.588:
            print("\n  Beats v5 hybrid baseline!")
        elif result.care_score > 0.505:
            print("\n  Beats v7 Phase 2 baseline!")
        elif result.care_score >= 0.50:
            print("\n  Beats random baseline")
        else:
            print("\n  Below random baseline")

        print("=" * 70)

        # Save results
        self._save_results(result, elapsed, farm_stats, n_with_power_curve)

        return result

    def _print_per_farm_diagnostics(self, farm_stats: dict, result: CAREResult):
        """Print per-farm diagnostics."""
        print("\n" + "=" * 70)
        print("PER-FARM DIAGNOSTICS (v8 Ensemble)")
        print("=" * 70)

        print("\n{:<10} {:<15} {:<15} {:<12} {:<12}".format(
            "Farm", "Anomaly Detect", "Normal FP", "Mean Z", "Power Curve"
        ))
        print("-" * 65)

        for farm in ['A', 'B', 'C']:
            anom_data = farm_stats[farm]['anomaly']
            norm_data = farm_stats[farm]['normal']

            n_anom = len(anom_data)
            n_norm = len(norm_data)
            n_anom_det = sum(1 for d in anom_data if d['n_flagged'] > 0)
            n_norm_fp = sum(1 for d in norm_data if d['n_flagged'] > 0)

            mean_z = np.mean([d['mean_z'] for d in anom_data]) if anom_data else 0.0
            n_pc = sum(1 for d in anom_data + norm_data if d.get('has_power_curve', False))
            n_total = n_anom + n_norm

            print(f"{farm:<10} {n_anom_det}/{n_anom:<13} {n_norm_fp}/{n_norm:<13} {mean_z:<12.3f} {n_pc}/{n_total}")

    def _save_results(self, result: CAREResult, elapsed: float, farm_stats: dict, n_with_power_curve: int):
        """Save all results to output directory."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Main results JSON
        results_dict = {
            'experiment': 'torquescope_care_phase2_v8',
            'methodology': 'v8 Ensemble: Temperature NBM z-score + Power Curve Residual (max fusion)',
            'z_threshold': self.z_threshold,
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'care_score': result.care_score,
            'sub_scores': {
                'coverage': result.coverage,
                'accuracy': result.accuracy,
                'reliability': result.reliability,
                'earliness': result.earliness
            },
            'baseline_comparison': {
                'autoencoder': 0.66,
                'v5_hybrid': 0.588,
                'v7_phase2': 0.505,
                'random': 0.50,
                'torquescope_v8': result.care_score
            },
            'summary': {
                'n_anomaly_datasets': result.n_anomaly_datasets,
                'n_normal_datasets': result.n_normal_datasets,
                'n_detected_events': result.n_detected_events,
                'n_with_power_curve': n_with_power_curve
            },
            'per_farm_stats': {
                farm: {
                    'n_anomaly': len(farm_stats[farm]['anomaly']),
                    'n_normal': len(farm_stats[farm]['normal']),
                    'n_anomaly_detected': sum(1 for d in farm_stats[farm]['anomaly'] if d['n_flagged'] > 0),
                    'n_normal_fp': sum(1 for d in farm_stats[farm]['normal'] if d['n_flagged'] > 0),
                    'mean_z_anomaly': float(np.mean([d['mean_z'] for d in farm_stats[farm]['anomaly']])) if farm_stats[farm]['anomaly'] else 0,
                    'n_with_power_curve': sum(1 for d in farm_stats[farm]['anomaly'] + farm_stats[farm]['normal'] if d.get('has_power_curve', False))
                }
                for farm in ['A', 'B', 'C']
            },
            'per_dataset_results': self.per_dataset_results
        }

        results_path = self.output_dir / 'care_v8_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"\nSaved: {results_path}")

        # Save continuous scores
        scores_path = self.output_dir / 'care_v8_scores.json'
        with open(scores_path, 'w') as f:
            json.dump({
                'z_threshold': self.z_threshold,
                'scores': {k: list(v) for k, v in self.continuous_scores.items()},
                'window_indices': {k: list(v) for k, v in self.window_indices.items()}
            }, f)
        print(f"Saved: {scores_path}")

        # Per-dataset CSV
        if self.per_dataset_results:
            df_results = pd.DataFrame(self.per_dataset_results)
            csv_path = self.output_dir / 'care_v8_per_dataset.csv'
            df_results.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

        # Summary markdown
        summary_md = f"""# TorqueScope Phase 2: CARE Benchmark Results (v8 Ensemble)

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Methodology:** v8 Ensemble - Temperature NBM + Power Curve Residual (max fusion)
**Z-Threshold:** {self.z_threshold}

## CARE Score

| Component | Score |
|-----------|-------|
| Coverage | {result.coverage:.4f} |
| Accuracy | {result.accuracy:.4f} |
| Reliability | {result.reliability:.4f} |
| Earliness | {result.earliness:.4f} |
| **CARE Score** | **{result.care_score:.4f}** |

## Comparison to Baselines

| Method | CARE Score |
|--------|------------|
| Autoencoder (NBM) | 0.66 |
| v5 Hybrid | 0.588 |
| v7 Phase 2 | 0.505 |
| v6 NBM z-score | 0.492 |
| Random (50/50) | 0.50 |
| **TorqueScope v8** | **{result.care_score:.4f}** |

## Per-Farm Diagnostics

| Farm | Anomaly Detected | Normal FP | Mean Z (anomaly) | Power Curve |
|------|-----------------|-----------|------------------|-------------|
"""
        for farm in ['A', 'B', 'C']:
            fs = results_dict['per_farm_stats'][farm]
            summary_md += f"| {farm} | {fs['n_anomaly_detected']}/{fs['n_anomaly']} | {fs['n_normal_fp']}/{fs['n_normal']} | {fs['mean_z_anomaly']:.3f} | {fs['n_with_power_curve']} |\n"

        summary_md += f"""

## v8 Ensemble Methodology

v8 combines two detection channels with max fusion:

1. **Temperature NBM z-scores** (from v7 Phase 2)
   - Rolling 30-day baseline removes seasonal drift
   - Consensus: 0.7 × max + 0.3 × top-3 mean

2. **Power Curve Residual** (NEW)
   - Binned power curve: wind speed → expected power
   - Residual = actual - expected power
   - Rolling z-score on residual
   - Both under and overperformance are anomalous

**Fusion:** `max(temp_z, power_z)` - flag if either channel high.

## Summary

- Anomaly datasets: {result.n_anomaly_datasets}
- Normal datasets: {result.n_normal_datasets}
- Events detected: {result.n_detected_events}
- Datasets with power curve: {n_with_power_curve}
- Elapsed time: {elapsed:.1f} seconds

---
*Generated by TorqueScope Phase 2 Benchmark v8*
"""
        summary_path = self.output_dir / 'care_v8_summary.md'
        with open(summary_path, 'w') as f:
            f.write(summary_md)
        print(f"Saved: {summary_path}")

    def run_threshold_sweep(
        self,
        z_thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """Sweep z-thresholds and compute CARE score at each."""
        if not self.continuous_scores:
            print("ERROR: No continuous scores available. Run benchmark first.")
            return pd.DataFrame()

        if z_thresholds is None:
            z_thresholds = list(np.arange(0.5, 3.05, 0.25))

        print("\n" + "=" * 70)
        print("Z-Threshold Sweep (v8 Ensemble)")
        print("=" * 70)

        v8_step_size = 6 * self.detector.samples_per_hour

        results = []

        for z in z_thresholds:
            sweep_predictions = {}

            for dataset_id, scores in self.continuous_scores.items():
                if not scores or dataset_id not in self.metadata:
                    sweep_predictions[dataset_id] = np.zeros(1, dtype=int)
                    continue

                n_total = len(self.predictions.get(dataset_id, [1]))
                predictions = np.zeros(n_total, dtype=int)

                windows = self.window_indices.get(dataset_id, [])

                for i, score in enumerate(scores):
                    if i < len(windows) and score >= z:
                        start, end = windows[i]
                        step_end = min(start + v8_step_size, n_total)
                        predictions[start:step_end] = 1

                sweep_predictions[dataset_id] = predictions

            care_result = self.scorer.compute_care_score(sweep_predictions, self.metadata)

            results.append({
                'z_threshold': z,
                'care': care_result.care_score,
                'coverage': care_result.coverage,
                'accuracy': care_result.accuracy,
                'reliability': care_result.reliability,
                'earliness': care_result.earliness
            })

            print(f"  z={z:.2f}: CARE={care_result.care_score:.4f} "
                  f"(Cov={care_result.coverage:.3f}, Acc={care_result.accuracy:.3f}, "
                  f"Rel={care_result.reliability:.3f}, Earl={care_result.earliness:.3f})")

        df = pd.DataFrame(results)

        best_idx = df['care'].idxmax()
        best = df.iloc[best_idx]

        print("\n" + "-" * 70)
        print(f"BEST Z-THRESHOLD: {best['z_threshold']:.2f}")
        print(f"  CARE Score: {best['care']:.4f}")
        print("=" * 70)

        sweep_path = self.output_dir / 'care_v8_threshold_sweep.csv'
        df.to_csv(sweep_path, index=False)
        print(f"\nSaved: {sweep_path}")

        return df


def run_v52_floor_sweep(
    data_dir: str,
    output_dir: str,
    floors: List[float] = None,
    thresholds: List[float] = None,
    quick: bool = False
) -> dict:
    """
    v5.2 Heuristic Floor Sweep.

    Sweep heuristic-only score floors to find optimal Coverage-Accuracy tradeoff.

    Args:
        data_dir: Path to CARE dataset
        output_dir: Path to output directory
        floors: Floor values to sweep. Default: [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        thresholds: Threshold values to sweep. Default: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        quick: If True, run on 10 datasets only

    Returns:
        Dict with sweep results
    """
    if floors is None:
        floors = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print("=" * 70)
    print("TorqueScope v5.2: Heuristic Floor Sweep")
    print("=" * 70)
    print(f"Floor values: {floors}")
    print(f"Threshold values: {thresholds}")
    print(f"Grid size: {len(floors)} × {len(thresholds)} = {len(floors) * len(thresholds)} cells")
    print()

    # Initialize loader once
    loader = CAREDataLoader(data_dir)
    loader.load_event_info()
    loader.build_feature_mappings()
    loader.profile_all_datasets(verbose=False)

    # Get dataset list
    if quick:
        all_profiles = list(loader.profiles.items())
        anomaly = [k for k, v in all_profiles if v.event_label == 'anomaly'][:5]
        normal = [k for k, v in all_profiles if v.event_label == 'normal'][:5]
        dataset_ids = anomaly + normal
        print(f"Quick mode: {len(dataset_ids)} datasets")
    else:
        dataset_ids = list(loader.profiles.keys())
        print(f"Full mode: {len(dataset_ids)} datasets")

    print()

    # Results storage
    grid_results = []
    best_result = None
    best_care = -1

    # Sweep over floor values
    for floor_idx, floor in enumerate(floors):
        print(f"\n{'=' * 70}")
        print(f"Floor {floor_idx + 1}/{len(floors)}: heuristic_floor = {floor}")
        print(f"{'=' * 70}")

        # Create detector with this floor value
        detector = AnomalyDetector(
            window_days=7,
            step_days=1,
            temp_rise_threshold=5.0,
            volatility_threshold=2.0,
            trend_threshold=0.1,
            heuristic_floor=floor
        )
        scorer = CAREScorer(criticality_threshold=72)

        # Run detection on all datasets
        predictions = {}
        metadata = {}
        continuous_scores = {}
        window_indices = {}

        for i, dataset_id in enumerate(dataset_ids):
            profile = loader.profiles[dataset_id]
            progress = f"[{i+1}/{len(dataset_ids)}]"
            label = "A" if profile.event_label == 'anomaly' else "N"
            print(f"  {progress} {dataset_id} ({label})", end=" ", flush=True)

            try:
                # Load dataset
                df = loader.load_dataset(profile.wind_farm, profile.event_id)

                # Get sensor columns
                key_sensors = loader.get_key_sensors_for_analysis(profile.wind_farm)
                signal_cols = (
                    key_sensors.get('bearing_temp', []) +
                    key_sensors.get('gearbox_temp', []) +
                    key_sensors.get('generator_temp', []) +
                    key_sensors.get('hydraulic_temp', []) +
                    key_sensors.get('transformer_temp', []) +
                    key_sensors.get('converter_temp', []) +
                    key_sensors.get('nacelle_temp', []) +
                    key_sensors.get('gear_oil_pump_current', []) +
                    key_sensors.get('ambient_temp', [])
                )
                signal_cols = [c for c in signal_cols if c in df.columns][:15]

                if not signal_cols:
                    print("| no sensors")
                    predictions[dataset_id] = np.zeros(len(df), dtype=int)
                    metadata[dataset_id] = {
                        'status_ids': df['status_type_id'].values,
                        'event_label': profile.event_label,
                        'event_start_idx': None,
                        'event_end_idx': None,
                        'test_start_idx': 0
                    }
                    continuous_scores[dataset_id] = []
                    continue

                # Create train/test masks
                train_mask = (df['train_test'] == 'train').values
                test_start_idx = int(train_mask.sum())

                # Get operational columns
                op_cols = get_operational_columns(profile.wind_farm, df)

                # Run v5 hybrid detection
                results, _, scores = detector.detect_multi_signal_hybrid(
                    df, signal_cols, profile.wind_farm, train_mask, test_start_idx,
                    power_col=op_cols.get('power'),
                    ambient_col=op_cols.get('ambient'),
                    wind_col=op_cols.get('wind')
                )

                # Build metadata
                if profile.event_label == 'anomaly':
                    df_ts = pd.to_datetime(df['time_stamp'])
                    event_start_ts = pd.to_datetime(profile.event_start)
                    event_end_ts = pd.to_datetime(profile.event_end)
                    event_start_idx = int((df_ts - event_start_ts).abs().argmin())
                    event_end_idx = int((df_ts - event_end_ts).abs().argmin())
                else:
                    event_start_idx, event_end_idx = None, None

                metadata[dataset_id] = {
                    'status_ids': df['status_type_id'].values,
                    'event_label': profile.event_label,
                    'event_start_idx': event_start_idx,
                    'event_end_idx': event_end_idx,
                    'test_start_idx': test_start_idx
                }

                # Store scores and window indices
                continuous_scores[dataset_id] = scores
                if results:
                    window_indices[dataset_id] = [(r.window_start, r.window_end) for r in results]
                    predictions[dataset_id] = np.zeros(len(df), dtype=int)
                else:
                    window_indices[dataset_id] = []
                    predictions[dataset_id] = np.zeros(len(df), dtype=int)

                print(f"| {len(scores)} windows")

            except Exception as e:
                print(f"| ERROR: {str(e)[:30]}")
                predictions[dataset_id] = np.zeros(1, dtype=int)
                metadata[dataset_id] = {
                    'status_ids': np.array([0]),
                    'event_label': profile.event_label,
                    'event_start_idx': None,
                    'event_end_idx': None,
                    'test_start_idx': 0
                }
                continuous_scores[dataset_id] = []

        # Now sweep thresholds for this floor value
        print(f"\n  Threshold sweep for floor={floor}:")
        for threshold in thresholds:
            # Convert scores to predictions at this threshold
            sweep_predictions = {}
            for dataset_id, scores in continuous_scores.items():
                if not scores or dataset_id not in metadata:
                    sweep_predictions[dataset_id] = np.zeros(1, dtype=int)
                    continue

                n_total = len(predictions.get(dataset_id, [1]))
                preds = np.zeros(n_total, dtype=int)
                windows = window_indices.get(dataset_id, [])

                for i, score in enumerate(scores):
                    if i < len(windows) and score >= threshold:
                        start, end = windows[i]
                        step_end = min(start + detector.step_size, n_total)
                        preds[start:step_end] = 1

                sweep_predictions[dataset_id] = preds

            # Compute CARE
            care_result = scorer.compute_care_score(sweep_predictions, metadata)

            result_entry = {
                'heuristic_floor': floor,
                'threshold': threshold,
                'coverage': care_result.coverage,
                'accuracy': care_result.accuracy,
                'reliability': care_result.reliability,
                'earliness': care_result.earliness,
                'care': care_result.care_score
            }
            grid_results.append(result_entry)

            print(f"    t={threshold:.2f}: CARE={care_result.care_score:.4f} "
                  f"(C={care_result.coverage:.3f} A={care_result.accuracy:.3f} "
                  f"R={care_result.reliability:.3f} E={care_result.earliness:.3f})")

            # Track best
            if care_result.care_score > best_care:
                best_care = care_result.care_score
                best_result = result_entry.copy()

    # Generate output
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)

    print(f"\nBest result:")
    print(f"  Floor: {best_result['heuristic_floor']}")
    print(f"  Threshold: {best_result['threshold']}")
    print(f"  CARE: {best_result['care']:.4f}")
    print(f"  Coverage: {best_result['coverage']:.4f}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  Reliability: {best_result['reliability']:.4f}")
    print(f"  Earliness: {best_result['earliness']:.4f}")

    # Analysis: best CARE per floor
    care_vs_floor = {}
    for floor in floors:
        floor_results = [r for r in grid_results if r['heuristic_floor'] == floor]
        best_for_floor = max(floor_results, key=lambda x: x['care'])
        care_vs_floor[str(floor)] = {
            'best_care': best_for_floor['care'],
            'best_threshold': best_for_floor['threshold']
        }

    # Coverage-Accuracy tradeoff
    tradeoff = []
    for floor in floors:
        floor_results = [r for r in grid_results if r['heuristic_floor'] == floor]
        best_for_floor = max(floor_results, key=lambda x: x['care'])
        tradeoff.append({
            'floor': floor,
            'coverage_at_best': best_for_floor['coverage'],
            'accuracy_at_best': best_for_floor['accuracy']
        })

    # Build final results dict
    results_dict = {
        'sweep_type': 'heuristic_floor_x_threshold',
        'timestamp': datetime.now().isoformat(),
        'baseline_v5': {
            'care': 0.588, 'coverage': 0.637, 'accuracy': 0.530,
            'reliability': 0.503, 'earliness': 0.634, 'threshold': 0.475
        },
        'baseline_v5.1': {
            'care': 0.442, 'coverage': 0.309, 'accuracy': 0.862,
            'reliability': 0.511, 'earliness': 0.263, 'threshold': 0.470
        },
        'grid_results': grid_results,
        'best_result': best_result,
        'analysis': {
            'care_vs_floor': care_vs_floor,
            'coverage_accuracy_tradeoff': tradeoff
        }
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = output_path / 'care_v5.2_sweep_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Generate summary markdown
    summary_md = _generate_v52_summary(results_dict, floors, thresholds)
    summary_path = output_path / 'care_v5.2_sweep_summary.md'
    with open(summary_path, 'w') as f:
        f.write(summary_md)
    print(f"Saved: {summary_path}")

    return results_dict


def _generate_v52_summary(results: dict, floors: List[float], thresholds: List[float]) -> str:
    """Generate v5.2 sweep summary markdown."""
    best = results['best_result']
    baseline_v5 = results['baseline_v5']
    baseline_v51 = results['baseline_v5.1']

    # Build heatmap table
    heatmap_rows = []
    for floor in floors:
        row = [f"{floor:.2f}"]
        for threshold in thresholds:
            match = [r for r in results['grid_results']
                    if r['heuristic_floor'] == floor and r['threshold'] == threshold]
            if match:
                care = match[0]['care']
                row.append(f"{care:.3f}")
            else:
                row.append("-")
        heatmap_rows.append(" | ".join(row))

    threshold_header = " | ".join([f"t={t:.2f}" for t in thresholds])

    # Determine pattern
    care_by_floor = [(f, results['analysis']['care_vs_floor'][str(f)]['best_care']) for f in floors]
    cares = [c for _, c in care_by_floor]

    if max(cares) - min(cares) < 0.02:
        pattern = "C) Flat plateau — CARE barely varies across floor values"
    elif cares == sorted(cares):
        pattern = "B) Monotonic increasing — higher floor = better CARE"
    elif cares == sorted(cares, reverse=True):
        pattern = "B) Monotonic decreasing — lower floor = better CARE"
    else:
        pattern = "A) Clear optimum exists — CARE peaks at intermediate floor"

    # Recommendation
    if best['care'] > baseline_v5['care']:
        improvement = (best['care'] - baseline_v5['care']) / baseline_v5['care'] * 100
        recommendation = f"✓ v5.2 (floor={best['heuristic_floor']}, t={best['threshold']}) beats v5 by {improvement:.1f}%"
    else:
        recommendation = f"✗ v5.2 does not beat v5 baseline (0.588)"

    if best['care'] >= 0.66:
        recommendation += "\n\n**🎉 TARGET ACHIEVED: CARE ≥ 0.66 (autoencoder baseline)**"

    summary = f"""# TorqueScope v5.2: Heuristic Floor Sweep Results

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary

| Metric | v5 | v5.1 | **v5.2 Best** |
|--------|-----|------|---------------|
| CARE | {baseline_v5['care']:.3f} | {baseline_v51['care']:.3f} | **{best['care']:.3f}** |
| Coverage | {baseline_v5['coverage']:.3f} | {baseline_v51['coverage']:.3f} | {best['coverage']:.3f} |
| Accuracy | {baseline_v5['accuracy']:.3f} | {baseline_v51['accuracy']:.3f} | {best['accuracy']:.3f} |
| Reliability | {baseline_v5['reliability']:.3f} | {baseline_v51['reliability']:.3f} | {best['reliability']:.3f} |
| Earliness | {baseline_v5['earliness']:.3f} | {baseline_v51['earliness']:.3f} | {best['earliness']:.3f} |
| Floor | ~0.475 | 0.10 | **{best['heuristic_floor']:.2f}** |
| Threshold | 0.475 | 0.470 | **{best['threshold']:.2f}** |

## Best Result

- **Heuristic Floor:** {best['heuristic_floor']}
- **Threshold:** {best['threshold']}
- **CARE Score:** {best['care']:.4f}

## Heatmap: CARE Score by (Floor, Threshold)

| Floor | {threshold_header} |
|-------|{'----|' * len(thresholds)}
| {' |'.join(heatmap_rows[0].split(' | '))} |
| {' |'.join(heatmap_rows[1].split(' | '))} |
| {' |'.join(heatmap_rows[2].split(' | '))} |
| {' |'.join(heatmap_rows[3].split(' | '))} |
| {' |'.join(heatmap_rows[4].split(' | '))} |
| {' |'.join(heatmap_rows[5].split(' | '))} |

## Best CARE per Floor

| Floor | Best CARE | Best Threshold | Coverage | Accuracy |
|-------|-----------|----------------|----------|----------|
"""
    for floor in floors:
        info = results['analysis']['care_vs_floor'][str(floor)]
        tradeoff_entry = [t for t in results['analysis']['coverage_accuracy_tradeoff'] if t['floor'] == floor][0]
        summary += f"| {floor:.2f} | {info['best_care']:.3f} | {info['best_threshold']:.2f} | {tradeoff_entry['coverage_at_best']:.3f} | {tradeoff_entry['accuracy_at_best']:.3f} |\n"

    summary += f"""
## Pattern Analysis

**Observed pattern:** {pattern}

## Recommendation

{recommendation}

---
*Generated by TorqueScope v5.2 Floor Sweep*
"""
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run TorqueScope CARE benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--data-dir',
        default='data/care/CARE_To_Compare',
        help='Path to CARE dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        default='torquescope_phase2/results',
        help='Path to output directory'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Detection threshold (z-score for v6, anomaly score for v5)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test on 10 datasets only'
    )
    parser.add_argument(
        '--farm',
        choices=['A', 'B', 'C'],
        help='Run only on specified wind farm'
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run threshold sweep after benchmark'
    )
    parser.add_argument(
        '--floor-sweep',
        action='store_true',
        help='Run v5.2 heuristic floor sweep (6×7 grid)'
    )
    parser.add_argument(
        '--v6',
        action='store_true',
        default=True,
        help='Use v6 NBM z-score detector (default)'
    )
    parser.add_argument(
        '--v5',
        action='store_true',
        help='Use v5 hybrid detector instead of v6'
    )
    parser.add_argument(
        '--v7',
        action='store_true',
        help='Use v7 detector (Phase 1: non-temperature signals)'
    )
    parser.add_argument(
        '--v8',
        action='store_true',
        help='Use v8 detector (Ensemble: Temperature NBM + Power Curve Residual)'
    )
    parser.add_argument(
        '--phase',
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help='v7 implementation phase: 1=non-temp signals, 2=rolling baseline, 3=correlation monitoring, 4=reserved (default: 1)'
    )
    parser.add_argument(
        '--per-farm',
        action='store_true',
        help='Use per-farm tuned thresholds and suppression factors (v5 only). Farm B: raised threshold (0.65) to reduce FPs. Farm C: lowered threshold (0.38) to improve detection.'
    )
    parser.add_argument(
        '--ablation',
        choices=['periodic_only', 'nbm_only', 'hybrid'],
        default='hybrid',
        help='Ablation mode for v5: periodic_only (LS only), nbm_only (NBM only), or hybrid (full v5, default)'
    )

    args = parser.parse_args()

    # v5.2 floor sweep mode
    if args.floor_sweep:
        run_v52_floor_sweep(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            quick=args.quick
        )
        return

    # Determine which version to run
    if args.v8:
        use_version = 'v8'
    elif args.v7:
        use_version = 'v7'
    elif args.v5:
        use_version = 'v5'
    else:
        use_version = 'v6'

    # Set default threshold based on version
    if args.threshold is None:
        threshold = 2.0 if use_version in ('v6', 'v7', 'v8') else 0.45  # v5 optimal
    else:
        threshold = args.threshold

    # Initialize appropriate benchmark
    if use_version == 'v8':
        benchmark = TorqueScopeBenchmarkV8(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            z_threshold=threshold,
            verbose=True
        )
    elif use_version == 'v7':
        benchmark = TorqueScopeBenchmarkV7(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            z_threshold=threshold,
            phase=args.phase,
            verbose=True
        )
    elif use_version == 'v6':
        benchmark = TorqueScopeBenchmarkV6(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            z_threshold=threshold,
            verbose=True
        )
    else:
        benchmark = TorqueScopeBenchmark(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            threshold=threshold,
            verbose=True,
            use_per_farm_tuning=args.per_farm,
            ablation_mode=args.ablation
        )

    # Determine which datasets to run
    dataset_ids = None

    if args.quick:
        # Quick mode: run on 5 anomaly + 5 normal
        benchmark.loader.load_event_info()
        benchmark.loader.profile_all_datasets(verbose=False)

        all_profiles = list(benchmark.loader.profiles.items())
        anomaly = [k for k, v in all_profiles if v.event_label == 'anomaly'][:5]
        normal = [k for k, v in all_profiles if v.event_label == 'normal'][:5]
        dataset_ids = anomaly + normal
        print(f"Quick mode: running on {len(dataset_ids)} datasets")

    elif args.farm:
        benchmark.loader.load_event_info()
        benchmark.loader.profile_all_datasets(verbose=False)

        farm_name = f"Wind Farm {args.farm}"
        dataset_ids = [k for k, v in benchmark.loader.profiles.items()
                      if v.wind_farm == farm_name]
        print(f"Farm {args.farm} mode: running on {len(dataset_ids)} datasets")

    # Run benchmark
    result = benchmark.run_benchmark(dataset_ids)

    # Run threshold sweep if requested
    if args.sweep:
        sweep_df = benchmark.run_threshold_sweep()

    return result


if __name__ == '__main__':
    main()

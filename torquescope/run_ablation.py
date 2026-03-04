#!/usr/bin/env python3
"""
TorqueScope Ablation Study Runner

Runs three ablation modes to isolate LS periodic vs NBM contributions
in the v5 hybrid pipeline:

1. periodic_only: LS periodic baseline scores only (no NBM, no heuristics)
2. nbm_only: NBM temperature deviation scores only (no periodic, no heuristics)
3. hybrid: Full v5 (reconfirmation)

Each mode runs the full CARE benchmark with threshold sweep.

Usage:
    cd torquescope/
    python3 run_ablation.py --data-dir ../data/care/CARE_To_Compare
    python3 run_ablation.py --data-dir ../data/care/CARE_To_Compare --quick
    python3 run_ablation.py --data-dir ../data/care/CARE_To_Compare --mode periodic_only
"""

import sys
import json
import argparse
import time
import importlib
from pathlib import Path
from datetime import datetime

# Add parent directory to path (same pattern as run_benchmark.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

# The torquescope directory was renamed from torquescope_phase2, but internal
# imports still reference the old name. Register the alias so they resolve.
import torquescope as _ts_pkg
sys.modules.setdefault('torquescope_phase2', _ts_pkg)

from torquescope.run_benchmark import TorqueScopeBenchmark
from torquescope.care_scorer import CAREResult


ABLATION_MODES = ["periodic_only", "nbm_only", "hybrid"]


def run_ablation_study(
    data_dir: str,
    output_base: str = None,
    quick: bool = False,
    modes: list = None,
    farm: str = None,
):
    """
    Run complete ablation study across specified modes.

    Args:
        data_dir: Path to CARE dataset directory
        output_base: Base output directory for results
        quick: If True, run on 10 datasets only
        modes: List of modes to run (default: all three)
        farm: Optional farm filter ('A', 'B', or 'C')
    """
    if modes is None:
        modes = ABLATION_MODES

    if output_base is None:
        output_base = str(Path(__file__).parent / "archive" / "results" / "ablation")

    output_dir = Path(output_base)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "ablation_study": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "purpose": "Isolate LS periodic vs NBM contribution for arXiv paper",
            "quick_mode": quick,
            "farm_filter": farm,
            "results": {}
        }
    }

    total_start = time.time()

    for mode in modes:
        print("\n" + "=" * 80)
        print(f"  ABLATION MODE: {mode.upper()}")
        print("=" * 80)

        # Create mode-specific output directory
        mode_output = output_dir / mode
        mode_output.mkdir(exist_ok=True)

        # Initialize benchmark with ablation mode
        benchmark = TorqueScopeBenchmark(
            data_dir=data_dir,
            output_dir=str(mode_output),
            threshold=0.45,
            verbose=True,
            use_per_farm_tuning=False,
            ablation_mode=mode,
        )

        # Determine dataset filter
        dataset_ids = None
        if quick or farm:
            benchmark.loader.load_event_info()
            benchmark.loader.profile_all_datasets(verbose=False)
            all_profiles = list(benchmark.loader.profiles.items())

            if farm:
                all_profiles = [
                    (k, v) for k, v in all_profiles
                    if farm.upper() in v.wind_farm.upper()
                ]

            if quick:
                anomaly = [k for k, v in all_profiles if v.event_label == 'anomaly'][:5]
                normal = [k for k, v in all_profiles if v.event_label == 'normal'][:5]
                dataset_ids = anomaly + normal
            else:
                dataset_ids = [k for k, _ in all_profiles]

        # Run benchmark
        mode_start = time.time()
        care_result = benchmark.run_benchmark(dataset_ids=dataset_ids)
        mode_elapsed = time.time() - mode_start

        # Run threshold sweep
        print(f"\n{'=' * 80}")
        print(f"  THRESHOLD SWEEP: {mode.upper()}")
        print("=" * 80)

        sweep_df = benchmark.run_threshold_sweep()

        if sweep_df.empty:
            print(f"WARNING: No sweep results for {mode}")
            results["ablation_study"]["results"][mode] = {
                "care_score": care_result.care_score,
                "coverage": care_result.coverage,
                "accuracy": care_result.accuracy,
                "reliability": care_result.reliability,
                "earliness": care_result.earliness,
                "n_detected_events": care_result.n_detected_events,
                "threshold": 0.45,
                "elapsed_seconds": round(mode_elapsed, 1),
            }
            continue

        # Find best threshold from sweep
        best_idx = sweep_df['care'].idxmax()
        best = sweep_df.iloc[best_idx]

        # Store summary
        results["ablation_study"]["results"][mode] = {
            "care_score": round(float(best['care']), 4),
            "coverage": round(float(best['coverage']), 4),
            "accuracy": round(float(best['accuracy']), 4),
            "reliability": round(float(best['reliability']), 4),
            "earliness": round(float(best['earliness']), 4),
            "n_detected_events": care_result.n_detected_events,
            "threshold": round(float(best['threshold']), 3),
            "elapsed_seconds": round(mode_elapsed, 1),
        }

        # Save per-mode sweep
        sweep_path = mode_output / f"{mode}_threshold_sweep.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(f"Saved: {sweep_path}")

    # Save consolidated results
    total_elapsed = time.time() - total_start
    results["ablation_study"]["total_elapsed_seconds"] = round(total_elapsed, 1)

    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 80)
    print("  ABLATION STUDY COMPLETE")
    print("=" * 80)
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")
    print(f"Summary saved: {summary_path}\n")

    hdr = f"{'Mode':<18} {'CARE':<10} {'Coverage':<10} {'Accuracy':<10} {'Reliab.':<10} {'Earliness':<10} {'Thresh':<8}"
    print(hdr)
    print("-" * len(hdr))

    for mode in modes:
        r = results["ablation_study"]["results"].get(mode, {})
        print(
            f"{mode:<18} "
            f"{r.get('care_score', 0):<10.4f} "
            f"{r.get('coverage', 0):<10.4f} "
            f"{r.get('accuracy', 0):<10.4f} "
            f"{r.get('reliability', 0):<10.4f} "
            f"{r.get('earliness', 0):<10.4f} "
            f"{r.get('threshold', 0):<8.3f}"
        )

    print("-" * len(hdr))

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run TorqueScope ablation study (v5 hybrid component isolation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data-dir',
        default=str(Path(__file__).parent / 'data' / 'CARE_To_Compare'),
        help='Path to CARE dataset directory',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (default: torquescope/archive/results/ablation/)',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test on 10 datasets only',
    )
    parser.add_argument(
        '--mode',
        choices=ABLATION_MODES,
        help='Run only specified mode (default: all three)',
    )
    parser.add_argument(
        '--farm',
        choices=['A', 'B', 'C'],
        help='Run only on specified wind farm',
    )

    args = parser.parse_args()

    modes = [args.mode] if args.mode else ABLATION_MODES

    run_ablation_study(
        data_dir=args.data_dir,
        output_base=args.output_dir,
        quick=args.quick,
        modes=modes,
        farm=args.farm,
    )


if __name__ == "__main__":
    main()

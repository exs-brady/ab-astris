"""
Z24 Phase 5: Extract Frequencies from Damage Phases 02-24

Extends Phase 2 frequency extraction to all damage states.
Reuses validated Ab Astris pipeline from z24_frequency_extraction.py.

Usage:
    # Extract all damage phases (4-6 hours)
    python shm/z24/run_phase5_frequency_extraction.py

    # Test with limited measurements
    python shm/z24/run_phase5_frequency_extraction.py --max 10

Expected output:
    - shm/z24/results/z24_frequencies_damage.csv (phases 02-24)
    - shm/z24/results/z24_frequencies_all.csv (combined healthy + damage)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Import Phase 2 frequency extraction pipeline
sys.path.append(str(Path(__file__).parent.parent.parent))
from shm.z24.z24_frequency_extraction import extract_z24_frequencies
from shm.z24.z24_data_loader import inventory_z24_dataset


def main(max_measurements: int = None, phases_only: list = None):
    """Extract frequencies from Z24 damage phases 02-24.

    Args:
        max_measurements: Limit per phase (for testing)
        phases_only: Specific phases to process (default: all 02-24)
    """
    print("=" * 80)
    print("Z24 Phase 5: Damage State Frequency Extraction")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Paths
    data_dir = Path(__file__).parent.parent / 'data' / 'Z24'
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # Define damage phases to process
    if phases_only:
        phases_to_process = phases_only
    else:
        # All damage phases: 02 through 24
        # Note: Phase 01 (healthy) already processed in Phase 2
        phases_to_process = [f'{i:02d}' for i in range(2, 25)]

    print(f"Damage phases to process: {phases_to_process}")
    print(f"Data directory: {data_dir}")
    print()

    # Step 1: Verify dataset inventory
    print("Step 1: Loading dataset inventory...")
    inventory = inventory_z24_dataset(str(data_dir))
    total_measurements = len(inventory)
    print(f"  Total measurements in dataset: {total_measurements:,}")

    # Filter to damage phases
    damage_inventory = inventory[inventory['phase'].isin(phases_to_process)]
    n_damage = len(damage_inventory)
    print(f"  Damage phase measurements (02-24): {n_damage:,}")

    if max_measurements:
        print(f"  [TEST MODE] Limiting to {max_measurements} measurements per phase")
        damage_inventory = damage_inventory.groupby('phase').head(max_measurements)
        n_damage = len(damage_inventory)
        print(f"  Limited measurements: {n_damage:,}")

    print()

    # Step 2: Extract frequencies using Ab Astris pipeline
    print("Step 2: Running Ab Astris frequency extraction...")
    print("  This will take approximately 40-50 seconds per measurement")
    if not max_measurements:
        estimated_hours = (n_damage * 45) / 3600
        print(f"  Estimated time: {estimated_hours:.1f} hours for {n_damage:,} measurements")
    print()

    output_csv_damage = results_dir / 'z24_frequencies_damage.csv'

    try:
        df_damage = extract_z24_frequencies(
            data_dir=str(data_dir),
            output_csv=str(output_csv_damage),
            phase_filter=phases_to_process,
            max_measurements=max_measurements
        )

        print()
        print(f"✓ Damage phase extraction complete!")
        print(f"  Processed: {len(df_damage):,} measurements")
        print(f"  Output: {output_csv_damage}")
        print()

    except Exception as e:
        print(f"✗ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Combine with healthy baseline from Phase 2
    print("Step 3: Combining with healthy baseline...")
    healthy_csv = results_dir / 'z24_frequencies_healthy.csv'

    if healthy_csv.exists():
        df_healthy = pd.read_csv(healthy_csv)
        print(f"  Loaded healthy baseline: {len(df_healthy):,} measurements (Phase 01)")

        # Combine
        df_all = pd.concat([df_healthy, df_damage], ignore_index=True)
        df_all = df_all.sort_values(['phase', 'config', 'seq']).reset_index(drop=True)

        # Save combined
        output_csv_all = results_dir / 'z24_frequencies_all.csv'
        df_all.to_csv(output_csv_all, index=False)

        print(f"  Combined dataset: {len(df_all):,} measurements")
        print(f"  Output: {output_csv_all}")
        print()
    else:
        print(f"  Warning: Healthy baseline not found at {healthy_csv}")
        print(f"  Skipping combined output (damage phases only)")
        df_all = df_damage
        output_csv_all = output_csv_damage
        print()

    # Step 4: Summary statistics
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)

    # Phases covered
    phases_processed = sorted(df_all['phase'].unique(), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else 0)
    print(f"Phases processed: {phases_processed}")
    print(f"Total measurements: {len(df_all):,}")
    print()

    # Frequency statistics (mode 1)
    print("Fundamental Frequency (f₁) Statistics:")
    print(f"  Mean: {df_all['freq_1_hz'].mean():.4f} Hz")
    print(f"  Std:  {df_all['freq_1_hz'].std():.4f} Hz")
    print(f"  Min:  {df_all['freq_1_hz'].min():.4f} Hz")
    print(f"  Max:  {df_all['freq_1_hz'].max():.4f} Hz")
    print()

    # CV statistics
    print("Coefficient of Variation (CV) Statistics:")
    print(f"  Median CV: {df_all['freq_1_cv_pct'].median():.2f}%")
    print(f"  Mean CV:   {df_all['freq_1_cv_pct'].mean():.2f}%")
    print(f"  CV <1%:    {(df_all['freq_1_cv_pct'] < 1.0).sum():,} measurements ({(df_all['freq_1_cv_pct'] < 1.0).mean()*100:.1f}%)")
    print(f"  CV <5%:    {(df_all['freq_1_cv_pct'] < 5.0).sum():,} measurements ({(df_all['freq_1_cv_pct'] < 5.0).mean()*100:.1f}%)")
    print()

    # Tier distribution
    print("Confidence Tier Distribution:")
    tier_counts = df_all['freq_1_tier'].value_counts()
    for tier in ['CONFIRMED', 'PROBABLE', 'CANDIDATE']:
        count = tier_counts.get(tier, 0)
        pct = (count / len(df_all)) * 100
        print(f"  {tier:12s}: {count:5,} ({pct:5.1f}%)")
    print()

    # Detection rate
    print("Detection Performance:")
    print(f"  Mean detection rate: {df_all['detection_rate'].mean():.1%}")
    print(f"  Measurements with >80% detection: {(df_all['detection_rate'] > 0.8).sum():,}")
    print()

    # Frequency progression (check for damage-induced drop)
    if '01' in phases_processed and len(phases_processed) > 1:
        print("Frequency Progression (Damage Detection):")
        phase_stats = df_all.groupby('phase')['freq_1_hz'].agg(['mean', 'std', 'count'])
        print(f"  Phase 01 (healthy): {phase_stats.loc['01', 'mean']:.4f} ± {phase_stats.loc['01', 'std']:.4f} Hz (n={int(phase_stats.loc['01', 'count'])})")

        if '02' in phases_processed:
            f1_drop_hz = phase_stats.loc['01', 'mean'] - phase_stats.loc['02', 'mean']
            f1_drop_pct = (f1_drop_hz / phase_stats.loc['01', 'mean']) * 100
            print(f"  Phase 02 (early damage): {phase_stats.loc['02', 'mean']:.4f} ± {phase_stats.loc['02', 'std']:.4f} Hz (n={int(phase_stats.loc['02', 'count'])})")
            print(f"  Frequency drop (01→02): {f1_drop_hz:.4f} Hz ({f1_drop_pct:.2f}%)")

        if len(phases_processed) >= 3:
            last_phase = phases_processed[-1]
            f1_drop_hz = phase_stats.loc['01', 'mean'] - phase_stats.loc[last_phase, 'mean']
            f1_drop_pct = (f1_drop_hz / phase_stats.loc['01', 'mean']) * 100
            print(f"  Phase {last_phase} (late damage): {phase_stats.loc[last_phase, 'mean']:.4f} ± {phase_stats.loc[last_phase, 'std']:.4f} Hz (n={int(phase_stats.loc[last_phase, 'count'])})")
            print(f"  Total frequency drop (01→{last_phase}): {f1_drop_hz:.4f} Hz ({f1_drop_pct:.2f}%)")
        print()

    # Validation checks
    print("=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    median_cv = df_all['freq_1_cv_pct'].median()
    check_cv = median_cv < 5.0
    print(f"✓ Median CV <5%: {median_cv:.2f}% {'PASS' if check_cv else 'FAIL'}")

    detection_rate = df_all['detection_rate'].mean()
    check_detection = detection_rate > 0.7
    print(f"✓ Detection rate >70%: {detection_rate:.1%} {'PASS' if check_detection else 'FAIL'}")

    confirmed_rate = (df_all['freq_1_tier'] == 'CONFIRMED').mean()
    check_confirmed = confirmed_rate > 0.5
    print(f"✓ CONFIRMED tier >50%: {confirmed_rate:.1%} {'PASS' if check_confirmed else 'FAIL'}")

    # Monotonic decrease check (if multiple phases)
    if len(phases_processed) > 1:
        phase_means = df_all.groupby('phase')['freq_1_hz'].mean()
        is_monotonic = (phase_means.diff()[1:] <= 0.01).all()  # Allow 0.01 Hz tolerance
        print(f"✓ Frequency decreases with damage: {'PASS' if is_monotonic else 'FAIL'}")

    print()

    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Next step: Run z24_real_damage_localization.py to apply PINN")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract frequencies from Z24 damage phases 02-24'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Limit measurements per phase (for testing)'
    )
    parser.add_argument(
        '--phases',
        type=str,
        default=None,
        help='Specific phases to process (comma-separated, e.g., "02,03,04")'
    )

    args = parser.parse_args()

    phases_only = None
    if args.phases:
        phases_only = [p.strip() for p in args.phases.split(',')]

    exit_code = main(max_measurements=args.max, phases_only=phases_only)
    sys.exit(exit_code)

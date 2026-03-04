"""Z24 Bridge Dataset Loader

Parses Z24 acceleration time-history files (.aaa format) from zip archives.
The Z24 dataset contains measurements from a 30m concrete box girder bridge
subjected to progressive damage (1997-1998).

File format:
- .aaa files: ASCII format with channel name, n_samples, dt, acceleration values
- .env files: Environmental sensor data (temperature, wind, etc.)
- Zip archives contain multiple channels from one measurement session

Dataset structure:
- Z24ems1/, Z24ems2/, Z24ems3/: Three measurement campaigns
- File naming: [phase][config][seq].zip (e.g., 01C14.zip)
  - phase: 01-24 (progressive damage stages)
  - config: A-G (measurement configuration)
  - seq: 00-99 (test sequence number)
"""

import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict


def parse_aaa_file(file_obj) -> Tuple[str, np.ndarray, float]:
    """Parse a single Z24 .aaa acceleration file.

    Args:
        file_obj: File-like object (from zip.open() or regular file)

    Returns:
        Tuple of (channel_name, acceleration_data, sampling_rate_hz)
        - channel_name: str, sensor channel identifier
        - acceleration_data: np.ndarray, shape (n_samples,)
        - sampling_rate_hz: float, sampling frequency in Hz

    Raises:
        ValueError: If file format is invalid
    """
    lines = file_obj.readlines()

    if len(lines) < 4:
        raise ValueError(f"Invalid .aaa file: only {len(lines)} lines")

    # Line 1: Channel name (e.g., "01B1303.aaa")
    channel_name = lines[0].decode('utf-8').strip()

    # Line 2: Number of samples
    try:
        n_samples = int(lines[1].decode('utf-8').strip())
    except ValueError:
        raise ValueError(f"Invalid n_samples on line 2: {lines[1]}")

    # Line 3: Time step (dt in seconds)
    try:
        dt = float(lines[2].decode('utf-8').strip())
        sampling_rate = 1.0 / dt if dt > 0 else 0.0
    except ValueError:
        raise ValueError(f"Invalid dt on line 3: {lines[2]}")

    # Lines 4+: Acceleration values (one per line, scientific notation)
    acceleration = []
    for i, line in enumerate(lines[3:], start=4):
        try:
            value = float(line.decode('utf-8').strip())
            acceleration.append(value)
        except ValueError:
            # Some files may have trailing whitespace or comments - stop at first error
            break

    acceleration = np.array(acceleration)

    # Validate expected number of samples
    if len(acceleration) != n_samples:
        print(f"Warning: Expected {n_samples} samples, got {len(acceleration)} in {channel_name}")

    return channel_name, acceleration, sampling_rate


def load_z24_measurement(zip_path: str) -> Dict:
    """Load a complete Z24 measurement from a zip archive.

    Args:
        zip_path: Path to .zip file (e.g., "Z24ems1/01C14.zip")

    Returns:
        Dictionary with keys:
        - 'phase': str, damage phase (e.g., '01')
        - 'config': str, measurement configuration (e.g., 'C')
        - 'seq': str, sequence number (e.g., '14')
        - 'channels': List[str], channel names
        - 'acceleration': np.ndarray, shape (n_samples, n_channels)
        - 'sampling_rate': float, Hz
        - 'duration': float, seconds
        - 'n_samples': int
        - 'timestamp': str, acquisition timestamp (from zip metadata)

    Raises:
        FileNotFoundError: If zip file doesn't exist
        ValueError: If no .aaa files found or inconsistent sampling rates
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    # Parse filename to extract phase, config, seq
    # Format: [phase][config][seq].zip (e.g., 01C14.zip)
    match = re.match(r'(\d{2})([A-Z])(\d{2})\.zip', zip_path.name)
    if not match:
        raise ValueError(f"Invalid zip filename format: {zip_path.name}")

    phase, config, seq = match.groups()

    # Extract all .aaa files from zip
    channels = []
    accelerations = []
    sampling_rates = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get zip file timestamp (first file)
        info_list = zf.infolist()
        timestamp = f"{info_list[0].date_time}" if info_list else "unknown"

        # Find all .aaa files (acceleration data, not .env environmental files)
        aaa_files = [f for f in zf.namelist() if f.endswith('.aaa')]

        if len(aaa_files) == 0:
            raise ValueError(f"No .aaa files found in {zip_path.name}")

        # Parse each channel
        for filename in sorted(aaa_files):
            with zf.open(filename, 'r') as f:
                try:
                    channel_name, accel, fs = parse_aaa_file(f)
                    channels.append(channel_name)
                    accelerations.append(accel)
                    sampling_rates.append(fs)
                except ValueError as e:
                    print(f"Warning: Skipping {filename}: {e}")
                    continue

    if len(accelerations) == 0:
        raise ValueError(f"No valid .aaa files parsed from {zip_path.name}")

    # Verify consistent sampling rates across channels
    unique_fs = np.unique(sampling_rates)
    if len(unique_fs) > 1:
        print(f"Warning: Multiple sampling rates detected: {unique_fs} Hz")
        print(f"Using most common: {np.median(sampling_rates)} Hz")

    sampling_rate = float(np.median(sampling_rates))

    # Find minimum length across channels (some may have slightly different lengths)
    min_length = min(len(a) for a in accelerations)

    # Truncate all channels to minimum length and stack
    acceleration_matrix = np.column_stack([a[:min_length] for a in accelerations])

    duration = min_length / sampling_rate if sampling_rate > 0 else 0.0

    return {
        'phase': phase,
        'config': config,
        'seq': seq,
        'channels': channels,
        'acceleration': acceleration_matrix,
        'sampling_rate': sampling_rate,
        'duration': duration,
        'n_samples': min_length,
        'timestamp': timestamp,
    }


def inventory_z24_dataset(data_dir: str) -> pd.DataFrame:
    """Scan Z24 dataset directory and create inventory of all measurements.

    Args:
        data_dir: Path to Z24 root directory (containing Z24ems1/, Z24ems2/, Z24ems3/)

    Returns:
        pandas DataFrame with columns:
        - file_path: Full path to zip file
        - campaign: ems1, ems2, or ems3
        - phase: Damage phase (01-24)
        - config: Measurement configuration (A-G)
        - seq: Sequence number (00-99)
        - file_size_mb: File size in megabytes
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    inventory = []

    # Scan all three campaigns
    for campaign_dir in ['Z24ems1', 'Z24ems2', 'Z24ems3']:
        campaign_path = data_dir / campaign_dir

        if not campaign_path.exists():
            print(f"Warning: Campaign directory not found: {campaign_path}")
            continue

        # Find all .zip files
        zip_files = sorted(campaign_path.glob('*.zip'))

        for zip_path in zip_files:
            # Parse filename
            match = re.match(r'(\d{2})([A-Z])(\d{2})\.zip', zip_path.name)
            if not match:
                print(f"Warning: Skipping invalid filename: {zip_path.name}")
                continue

            phase, config, seq = match.groups()

            inventory.append({
                'file_path': str(zip_path),
                'campaign': campaign_dir,
                'phase': phase,
                'config': config,
                'seq': seq,
                'file_size_mb': zip_path.stat().st_size / (1024 * 1024),
            })

    df = pd.DataFrame(inventory)

    if len(df) == 0:
        raise ValueError(f"No .zip files found in {data_dir}")

    return df


def identify_damage_states(inventory_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group measurements by damage state based on phase number.

    The Z24 bridge was subjected to progressive damage. Phase numbers
    correspond to increasing damage severity:
    - Phase 01: Healthy baseline
    - Phase 02-09: Early damage (minor pier settlement)
    - Phase 10-19: Moderate damage
    - Phase 20+: Severe damage (before demolition)

    Args:
        inventory_df: DataFrame from inventory_z24_dataset()

    Returns:
        Dictionary mapping damage state labels to lists of file paths:
        {
            'healthy': [...],
            'damage_early': [...],
            'damage_moderate': [...],
            'damage_severe': [...]
        }
    """
    states = defaultdict(list)

    for _, row in inventory_df.iterrows():
        phase_num = int(row['phase'])
        file_path = row['file_path']

        if phase_num == 1:
            states['healthy'].append(file_path)
        elif 2 <= phase_num <= 9:
            states['damage_early'].append(file_path)
        elif 10 <= phase_num <= 19:
            states['damage_moderate'].append(file_path)
        else:  # phase_num >= 20
            states['damage_severe'].append(file_path)

    return dict(states)


def load_healthy_baseline(data_dir: str, campaign: str = 'Z24ems1') -> Dict:
    """Load a representative healthy baseline measurement.

    Args:
        data_dir: Path to Z24 root directory
        campaign: Which campaign to use ('Z24ems1', 'Z24ems2', or 'Z24ems3')

    Returns:
        Measurement dictionary from load_z24_measurement()
    """
    inventory = inventory_z24_dataset(data_dir)

    # Filter to specified campaign and phase 01 (healthy)
    healthy_files = inventory[
        (inventory['campaign'] == campaign) &
        (inventory['phase'] == '01')
    ]

    if len(healthy_files) == 0:
        raise ValueError(f"No healthy baseline (phase 01) found in {campaign}")

    # Use first healthy measurement
    baseline_path = healthy_files.iloc[0]['file_path']

    print(f"Loading healthy baseline: {Path(baseline_path).name}")
    return load_z24_measurement(baseline_path)


if __name__ == '__main__':
    # Example usage / testing
    import sys

    data_dir = Path(__file__).parent.parent / 'data' / 'Z24'

    if not data_dir.exists():
        print(f"Error: Z24 data directory not found at {data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Z24 Dataset Inventory")
    print("=" * 60)

    # Create inventory
    inventory = inventory_z24_dataset(data_dir)

    print(f"\nTotal measurements: {len(inventory)}")
    print(f"Campaigns: {inventory['campaign'].unique()}")
    print(f"Phases: {sorted(inventory['phase'].unique())}")
    print(f"Configs: {sorted(inventory['config'].unique())}")
    print(f"Total size: {inventory['file_size_mb'].sum():.1f} MB")

    # Identify damage states
    damage_states = identify_damage_states(inventory)
    print(f"\nDamage States:")
    for state, files in damage_states.items():
        print(f"  {state:20s}: {len(files):4d} measurements")

    # Load one example
    print("\n" + "=" * 60)
    print("Loading Sample Measurement")
    print("=" * 60)

    baseline = load_healthy_baseline(data_dir)

    print(f"\nPhase: {baseline['phase']}, Config: {baseline['config']}, Seq: {baseline['seq']}")
    print(f"Sampling rate: {baseline['sampling_rate']:.1f} Hz")
    print(f"Duration: {baseline['duration']:.1f} seconds")
    print(f"Channels: {len(baseline['channels'])}")
    print(f"Acceleration shape: {baseline['acceleration'].shape}")
    print(f"First 5 channels: {baseline['channels'][:5]}")

    # Basic statistics
    print(f"\nAcceleration statistics (all channels):")
    print(f"  Mean: {np.mean(baseline['acceleration']):.6e}")
    print(f"  Std:  {np.std(baseline['acceleration']):.6e}")
    print(f"  Min:  {np.min(baseline['acceleration']):.6e}")
    print(f"  Max:  {np.max(baseline['acceleration']):.6e}")

    print("\n" + "=" * 60)
    print("Data loader validation complete!")
    print("=" * 60)

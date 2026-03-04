"""
TorqueScope Phase 2: CARE Dataset Loader and Profiler

Handles loading, profiling, and feature mapping for the CARE benchmark dataset.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class DatasetProfile:
    """Profile of a single CARE dataset."""
    event_id: int
    wind_farm: str
    asset_id: int
    event_label: str  # 'anomaly' or 'normal'
    event_description: str
    n_features: int
    n_rows: int
    n_train_rows: int
    n_test_rows: int
    date_start: str
    date_end: str
    train_date_end: str
    test_date_start: str
    event_start: str
    event_end: str
    event_duration_hours: float
    status_counts: Dict[int, int] = field(default_factory=dict)
    train_normal_fraction: float = 0.0  # Fraction of status 0 or 2 in training
    test_normal_fraction: float = 0.0   # Fraction of status 0 or 2 in test
    file_path: str = ""


@dataclass
class FeatureMapping:
    """Mapping of sensor names to semantic categories."""
    sensor_name: str
    wind_farm: str
    description: str
    unit: str
    category: str  # temperature, speed, power, environmental, electrical, other
    is_temperature: bool = False
    is_bearing_temp: bool = False
    is_gearbox_temp: bool = False
    is_generator_temp: bool = False
    is_speed: bool = False
    is_power: bool = False
    is_wind: bool = False
    # New categories for Fix 1 (v3 brief)
    is_hydraulic_temp: bool = False
    is_transformer_temp: bool = False
    is_converter_temp: bool = False
    is_nacelle_temp: bool = False
    is_gear_oil_pump_current: bool = False
    is_ambient_temp: bool = False
    # Non-temperature categories for Phase 1 v7 (per v6.2 brief)
    is_vibration: bool = False
    is_hydraulic_pressure: bool = False
    is_pitch_position: bool = False
    is_gearbox_oil: bool = False  # Oil level/pressure, distinct from gearbox_temp
    is_motor_current: bool = False


class CAREDataLoader:
    """Loader for CARE benchmark dataset."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.wind_farms = ['Wind Farm A', 'Wind Farm B', 'Wind Farm C']
        self.profiles: Dict[str, DatasetProfile] = {}
        self.feature_mappings: Dict[str, List[FeatureMapping]] = {}
        self.event_info: Dict[str, pd.DataFrame] = {}

    def load_event_info(self) -> Dict[str, pd.DataFrame]:
        """Load event_info.csv for each wind farm."""
        for farm in self.wind_farms:
            path = self.data_dir / farm / 'event_info.csv'
            if path.exists():
                df = pd.read_csv(path, sep=';')
                self.event_info[farm] = df
        return self.event_info

    def load_feature_descriptions(self) -> Dict[str, pd.DataFrame]:
        """Load feature_description.csv for each wind farm."""
        descriptions = {}
        for farm in self.wind_farms:
            path = self.data_dir / farm / 'feature_description.csv'
            if path.exists():
                df = pd.read_csv(path, sep=';')
                descriptions[farm] = df
        return descriptions

    def categorize_sensor(self, description: str, sensor_name: str) -> dict:
        """
        Categorize a sensor based on its description.
        Returns: dict with all category flags
        """
        desc_lower = description.lower()
        name_lower = sensor_name.lower()

        # Temperature sensors
        is_temp = 'temperature' in desc_lower or 'temp' in desc_lower
        is_bearing = is_temp and ('bearing' in desc_lower or 'rotor bearing' in desc_lower)
        is_gearbox = is_temp and ('gearbox' in desc_lower or 'gear oil' in desc_lower) and 'pump' not in desc_lower
        is_generator = is_temp and ('generator' in desc_lower or 'stator' in desc_lower)

        # New temperature subcategories (Fix 1)
        is_hydraulic_temp = is_temp and 'hydraulic' in desc_lower
        is_transformer_temp = is_temp and 'transformer' in desc_lower
        is_converter_temp = is_temp and ('converter' in desc_lower or 'igbt' in desc_lower)
        is_nacelle_temp = is_temp and ('nacelle' in desc_lower or 'hub' in desc_lower)
        is_ambient_temp = is_temp and ('ambient' in desc_lower or 'outside' in desc_lower or 'external' in desc_lower)

        # Gear oil pump current (for C_49 fault detection)
        is_gear_oil_pump_current = 'gear oil pump' in desc_lower and 'current' in desc_lower

        # Speed sensors
        is_speed = any(x in desc_lower for x in ['rpm', 'speed', 'rotational', 'rotor speed'])

        # Power sensors
        is_power = 'power' in name_lower or 'power' in desc_lower

        # Wind sensors
        is_wind = 'wind' in name_lower or 'wind speed' in desc_lower

        # ===== Non-temperature categories for Phase 1 v7 =====
        # Per v6.2 brief: be selective about which non-temp sensors to use

        # Vibration sensors - ONLY nacelle/tower vibration (Farm C: 90-93)
        # Exclude "generator acceleration" which isn't structural vibration
        is_vibration = (
            ('nacelle' in desc_lower and 'vibration' in desc_lower) or
            ('tower' in desc_lower and 'vibration' in desc_lower) or
            ('drivetrain' in desc_lower and 'vibration' in desc_lower)
        )

        # Hydraulic pressure sensors (Farm C: 48-55)
        # Focus on rotor brake, aggregate system, azimuth
        is_hydraulic_pressure = (
            not is_temp and
            'hydraulic' in desc_lower and
            'pressure' in desc_lower
        )

        # Pitch/blade position sensors (Farm C: 103-105)
        # "Position rotor blade axis" or "min/max pitch angle"
        is_pitch_position = (
            ('position' in desc_lower and 'blade' in desc_lower) or
            ('position' in desc_lower and 'rotor' in desc_lower and 'axis' in desc_lower) or
            ('pitch' in desc_lower and 'angle' in desc_lower)
        )

        # Gearbox oil level/pressure (not temperature)
        # Farm C: sensors 94, 117-118
        is_gearbox_oil = (
            not is_temp and
            ('gear' in desc_lower or 'gearbox' in desc_lower) and
            ('oil' in desc_lower) and
            ('level' in desc_lower or 'pressure' in desc_lower or 'filter' in desc_lower)
        )

        # Motor current sensors - ONLY pitch blade motors (Farm B: 28-30)
        # Exclude yaw motors, cooler fans, hydraulic pumps, etc.
        is_motor_current = (
            not is_gear_oil_pump_current and
            (
                ('pitch' in desc_lower and ('motor' in desc_lower or 'current' in desc_lower)) or
                ('blade' in desc_lower and 'motor' in desc_lower and 'current' in desc_lower)
            )
        )

        # Determine category
        if is_temp:
            category = 'temperature'
        elif is_speed:
            category = 'speed'
        elif is_power:
            category = 'power'
        elif is_wind:
            category = 'environmental'
        elif is_vibration:
            category = 'vibration'
        elif is_hydraulic_pressure:
            category = 'hydraulic'
        elif is_pitch_position:
            category = 'pitch'
        elif is_gearbox_oil:
            category = 'gearbox_oil'
        elif is_motor_current or is_gear_oil_pump_current:
            category = 'electrical'
        elif any(x in desc_lower for x in ['current', 'voltage', 'frequency']):
            category = 'electrical'
        elif any(x in desc_lower for x in ['pressure', 'hydraulic']):
            category = 'mechanical'
        elif any(x in desc_lower for x in ['angle', 'direction', 'position']):
            category = 'positional'
        else:
            category = 'other'

        return {
            'category': category,
            'is_temperature': is_temp,
            'is_bearing_temp': is_bearing,
            'is_gearbox_temp': is_gearbox,
            'is_generator_temp': is_generator,
            'is_speed': is_speed,
            'is_power': is_power,
            'is_wind': is_wind,
            'is_hydraulic_temp': is_hydraulic_temp,
            'is_transformer_temp': is_transformer_temp,
            'is_converter_temp': is_converter_temp,
            'is_nacelle_temp': is_nacelle_temp,
            'is_gear_oil_pump_current': is_gear_oil_pump_current,
            'is_ambient_temp': is_ambient_temp,
            # Non-temperature categories (Phase 1 v7)
            'is_vibration': is_vibration,
            'is_hydraulic_pressure': is_hydraulic_pressure,
            'is_pitch_position': is_pitch_position,
            'is_gearbox_oil': is_gearbox_oil,
            'is_motor_current': is_motor_current,
        }

    def build_feature_mappings(self) -> Dict[str, List[FeatureMapping]]:
        """Build feature mappings for each wind farm."""
        descriptions = self.load_feature_descriptions()

        for farm, df in descriptions.items():
            mappings = []
            for _, row in df.iterrows():
                sensor_name = row['sensor_name']
                description = row.get('description', '')
                unit = row.get('unit', '')

                cats = self.categorize_sensor(description, sensor_name)

                mapping = FeatureMapping(
                    sensor_name=sensor_name,
                    wind_farm=farm,
                    description=description,
                    unit=unit,
                    category=cats['category'],
                    is_temperature=cats['is_temperature'],
                    is_bearing_temp=cats['is_bearing_temp'],
                    is_gearbox_temp=cats['is_gearbox_temp'],
                    is_generator_temp=cats['is_generator_temp'],
                    is_speed=cats['is_speed'],
                    is_power=cats['is_power'],
                    is_wind=cats['is_wind'],
                    is_hydraulic_temp=cats['is_hydraulic_temp'],
                    is_transformer_temp=cats['is_transformer_temp'],
                    is_converter_temp=cats['is_converter_temp'],
                    is_nacelle_temp=cats['is_nacelle_temp'],
                    is_gear_oil_pump_current=cats['is_gear_oil_pump_current'],
                    is_ambient_temp=cats['is_ambient_temp'],
                    # Non-temperature categories (Phase 1 v7)
                    is_vibration=cats['is_vibration'],
                    is_hydraulic_pressure=cats['is_hydraulic_pressure'],
                    is_pitch_position=cats['is_pitch_position'],
                    is_gearbox_oil=cats['is_gearbox_oil'],
                    is_motor_current=cats['is_motor_current'],
                )
                mappings.append(mapping)

            self.feature_mappings[farm] = mappings

        return self.feature_mappings

    def load_dataset(self, farm: str, event_id: int) -> pd.DataFrame:
        """Load a single dataset CSV."""
        path = self.data_dir / farm / 'datasets' / f'{event_id}.csv'
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path, sep=';')
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        return df

    def profile_dataset(self, farm: str, event_id: int) -> DatasetProfile:
        """Create a profile for a single dataset."""
        # Get event info
        event_info = self.event_info.get(farm)
        if event_info is None:
            self.load_event_info()
            event_info = self.event_info[farm]

        event_row = event_info[event_info['event_id'] == event_id].iloc[0]

        # Load the dataset
        df = self.load_dataset(farm, event_id)

        # Split train/test
        train_df = df[df['train_test'] == 'train']
        test_df = df[df['train_test'] == 'test']

        # Status counts
        status_counts = df['status_type_id'].value_counts().to_dict()

        # Normal status fraction (status 0 or 2)
        train_normal = train_df['status_type_id'].isin([0, 2]).mean() if len(train_df) > 0 else 0
        test_normal = test_df['status_type_id'].isin([0, 2]).mean() if len(test_df) > 0 else 0

        # Event duration
        event_start = pd.to_datetime(event_row['event_start'])
        event_end = pd.to_datetime(event_row['event_end'])
        event_duration = (event_end - event_start).total_seconds() / 3600  # hours

        # Feature count (exclude descriptive columns)
        desc_cols = ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']
        n_features = len([c for c in df.columns if c not in desc_cols])

        profile = DatasetProfile(
            event_id=event_id,
            wind_farm=farm,
            asset_id=int(df['asset_id'].iloc[0]),
            event_label=event_row['event_label'],
            event_description=str(event_row.get('event_description', '')),
            n_features=n_features,
            n_rows=len(df),
            n_train_rows=len(train_df),
            n_test_rows=len(test_df),
            date_start=str(df['time_stamp'].min()),
            date_end=str(df['time_stamp'].max()),
            train_date_end=str(train_df['time_stamp'].max()) if len(train_df) > 0 else '',
            test_date_start=str(test_df['time_stamp'].min()) if len(test_df) > 0 else '',
            event_start=str(event_row['event_start']),
            event_end=str(event_row['event_end']),
            event_duration_hours=event_duration,
            status_counts=status_counts,
            train_normal_fraction=train_normal,
            test_normal_fraction=test_normal,
            file_path=str(self.data_dir / farm / 'datasets' / f'{event_id}.csv')
        )

        return profile

    def profile_all_datasets(self, verbose: bool = True) -> Dict[str, DatasetProfile]:
        """Profile all datasets in the CARE collection."""
        self.load_event_info()

        for farm in self.wind_farms:
            event_info = self.event_info.get(farm)
            if event_info is None:
                continue

            for event_id in event_info['event_id']:
                key = f"{farm}_{event_id}"
                try:
                    profile = self.profile_dataset(farm, event_id)
                    self.profiles[key] = profile
                    if verbose:
                        label = "ANOMALY" if profile.event_label == 'anomaly' else "normal "
                        print(f"  {key}: {label} | {profile.n_rows:,} rows | {profile.event_duration_hours:.1f}h event")
                except Exception as e:
                    print(f"  ERROR profiling {key}: {e}")

        return self.profiles

    def get_anomaly_datasets(self) -> List[DatasetProfile]:
        """Get all anomaly dataset profiles."""
        return [p for p in self.profiles.values() if p.event_label == 'anomaly']

    def get_normal_datasets(self) -> List[DatasetProfile]:
        """Get all normal dataset profiles."""
        return [p for p in self.profiles.values() if p.event_label == 'normal']

    def get_temperature_sensors(self, farm: str) -> List[FeatureMapping]:
        """Get all temperature sensors for a wind farm."""
        if not self.feature_mappings:
            self.build_feature_mappings()
        return [m for m in self.feature_mappings.get(farm, []) if m.is_temperature]

    def get_bearing_temp_sensors(self, farm: str) -> List[FeatureMapping]:
        """Get bearing temperature sensors for a wind farm."""
        if not self.feature_mappings:
            self.build_feature_mappings()
        return [m for m in self.feature_mappings.get(farm, []) if m.is_bearing_temp]

    def get_key_sensors_for_analysis(self, farm: str) -> Dict[str, List[str]]:
        """
        Get the key sensors recommended for analysis per farm.
        Returns dict mapping category to list of sensor column names.

        Updated for v7 Phase 1: includes non-temperature signals
        (vibration, hydraulic pressure, pitch position, gearbox oil, motor current).
        """
        if not self.feature_mappings:
            self.build_feature_mappings()

        mappings = self.feature_mappings.get(farm, [])

        result = {
            # Temperature categories
            'bearing_temp': [],
            'gearbox_temp': [],
            'generator_temp': [],
            'ambient_temp': [],
            'hydraulic_temp': [],
            'transformer_temp': [],
            'converter_temp': [],
            'nacelle_temp': [],
            # Operational
            'speed': [],
            'power': [],
            'wind': [],
            # Electrical
            'gear_oil_pump_current': [],
            'motor_current': [],
            # Non-temperature categories (Phase 1 v7)
            'vibration': [],
            'hydraulic_pressure': [],
            'pitch_position': [],
            'gearbox_oil': [],
        }

        for m in mappings:
            # Get the average column name
            col_name = f"{m.sensor_name}_avg"

            # Temperature categories (mutually exclusive)
            if m.is_bearing_temp:
                result['bearing_temp'].append(col_name)
            elif m.is_gearbox_temp:
                result['gearbox_temp'].append(col_name)
            elif m.is_generator_temp:
                result['generator_temp'].append(col_name)
            elif m.is_ambient_temp:
                result['ambient_temp'].append(col_name)
            elif m.is_hydraulic_temp:
                result['hydraulic_temp'].append(col_name)
            elif m.is_transformer_temp:
                result['transformer_temp'].append(col_name)
            elif m.is_converter_temp:
                result['converter_temp'].append(col_name)
            elif m.is_nacelle_temp:
                result['nacelle_temp'].append(col_name)
            # Non-temperature categories (Phase 1 v7)
            elif m.is_vibration:
                result['vibration'].append(col_name)
            elif m.is_hydraulic_pressure:
                result['hydraulic_pressure'].append(col_name)
            elif m.is_pitch_position:
                result['pitch_position'].append(col_name)
            elif m.is_gearbox_oil:
                result['gearbox_oil'].append(col_name)
            elif m.is_motor_current:
                result['motor_current'].append(col_name)
            elif m.is_gear_oil_pump_current:
                result['gear_oil_pump_current'].append(col_name)
            # Operational
            elif m.is_speed:
                result['speed'].append(col_name)
            elif m.is_power:
                result['power'].append(col_name)
            elif m.is_wind:
                result['wind'].append(col_name)

        return result

    def get_non_temp_sensors_for_analysis(self, farm: str) -> Dict[str, List[str]]:
        """
        Get non-temperature sensors for v7 Phase 1 analysis.
        Returns dict mapping category to list of sensor column names.

        Per v6.2 brief:
        - Farm C: vibration (90-93), hydraulic (48-55), pitch (100-102), gearbox oil (94, 117-118)
        - Farm B: vibration (54-56), motor current (28-30)
        """
        all_sensors = self.get_key_sensors_for_analysis(farm)

        non_temp_categories = [
            'vibration',
            'hydraulic_pressure',
            'pitch_position',
            'gearbox_oil',
            'motor_current',
            'gear_oil_pump_current',
        ]

        return {cat: all_sensors.get(cat, []) for cat in non_temp_categories}

    def generate_summary_report(self) -> Dict:
        """Generate a summary report of the dataset collection."""
        if not self.profiles:
            self.profile_all_datasets(verbose=False)

        anomaly_profiles = self.get_anomaly_datasets()
        normal_profiles = self.get_normal_datasets()

        # Per-farm breakdown
        farm_stats = {}
        for farm in self.wind_farms:
            farm_profiles = [p for p in self.profiles.values() if p.wind_farm == farm]
            farm_anomalies = [p for p in farm_profiles if p.event_label == 'anomaly']
            farm_normals = [p for p in farm_profiles if p.event_label == 'normal']

            farm_stats[farm] = {
                'total': len(farm_profiles),
                'anomaly': len(farm_anomalies),
                'normal': len(farm_normals),
                'n_features': farm_profiles[0].n_features if farm_profiles else 0,
                'anomaly_types': list(set(p.event_description for p in farm_anomalies if p.event_description))
            }

        # Anomaly duration stats
        anomaly_durations = [p.event_duration_hours for p in anomaly_profiles]

        summary = {
            'total_datasets': len(self.profiles),
            'anomaly_datasets': len(anomaly_profiles),
            'normal_datasets': len(normal_profiles),
            'per_farm': farm_stats,
            'anomaly_duration_stats': {
                'min_hours': min(anomaly_durations) if anomaly_durations else 0,
                'max_hours': max(anomaly_durations) if anomaly_durations else 0,
                'mean_hours': np.mean(anomaly_durations) if anomaly_durations else 0,
                'median_hours': np.median(anomaly_durations) if anomaly_durations else 0
            }
        }

        return summary

    def save_profiles_to_json(self, output_path: str):
        """Save all profiles to a JSON file."""
        profiles_dict = {k: asdict(v) for k, v in self.profiles.items()}
        with open(output_path, 'w') as f:
            json.dump(profiles_dict, f, indent=2, default=str)

    def save_feature_mappings_to_csv(self, output_path: str):
        """Save feature mappings to CSV."""
        if not self.feature_mappings:
            self.build_feature_mappings()

        rows = []
        for farm, mappings in self.feature_mappings.items():
            for m in mappings:
                rows.append(asdict(m))

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)


def run_data_exploration(data_dir: str, output_dir: str):
    """Run full data exploration and save results."""
    print("=" * 60)
    print("TorqueScope Phase 2: CARE Dataset Exploration")
    print("=" * 60)

    loader = CAREDataLoader(data_dir)

    # Profile all datasets
    print("\n1. Profiling all datasets...")
    loader.profile_all_datasets(verbose=True)

    # Build feature mappings
    print("\n2. Building feature mappings...")
    loader.build_feature_mappings()

    # Generate summary
    print("\n3. Generating summary report...")
    summary = loader.generate_summary_report()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total datasets: {summary['total_datasets']}")
    print(f"  Anomaly: {summary['anomaly_datasets']}")
    print(f"  Normal: {summary['normal_datasets']}")

    print("\nPer-farm breakdown:")
    for farm, stats in summary['per_farm'].items():
        print(f"  {farm}: {stats['total']} datasets ({stats['anomaly']} anomaly, {stats['normal']} normal)")
        print(f"    Features: {stats['n_features']}")
        if stats['anomaly_types']:
            print(f"    Fault types: {', '.join(stats['anomaly_types'][:5])}...")

    print(f"\nAnomaly event duration:")
    print(f"  Min: {summary['anomaly_duration_stats']['min_hours']:.1f} hours")
    print(f"  Max: {summary['anomaly_duration_stats']['max_hours']:.1f} hours")
    print(f"  Mean: {summary['anomaly_duration_stats']['mean_hours']:.1f} hours")
    print(f"  Median: {summary['anomaly_duration_stats']['median_hours']:.1f} hours")

    # Key sensors per farm
    print("\n4. Key sensors for LS analysis:")
    for farm in loader.wind_farms:
        key_sensors = loader.get_key_sensors_for_analysis(farm)
        print(f"\n  {farm}:")
        for category, sensors in key_sensors.items():
            if sensors:
                print(f"    {category}: {len(sensors)} sensors")

    # Save outputs
    print("\n5. Saving outputs...")
    os.makedirs(output_dir, exist_ok=True)

    loader.save_profiles_to_json(os.path.join(output_dir, 'dataset_profiles.json'))
    print(f"  Saved: {output_dir}/dataset_profiles.json")

    loader.save_feature_mappings_to_csv(os.path.join(output_dir, 'feature_mappings.csv'))
    print(f"  Saved: {output_dir}/feature_mappings.csv")

    with open(os.path.join(output_dir, 'exploration_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {output_dir}/exploration_summary.json")

    print("\n" + "=" * 60)
    print("Data exploration complete!")
    print("=" * 60)

    return loader, summary


if __name__ == '__main__':
    import sys

    # Default paths
    data_dir = 'data/care/CARE_To_Compare'
    output_dir = 'torquescope_phase2/results'

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    run_data_exploration(data_dir, output_dir)

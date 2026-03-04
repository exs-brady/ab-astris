"""
SHM Data Module for Ab Astris Structural Health Monitoring Validation

Generates physics-based synthetic structural vibration data simulating
a lattice tower structure (similar to LUMO benchmark) with multiple damage states.

Physics basis:
- Natural frequency f = (1/2π) * sqrt(k/m) where k=stiffness, m=mass
- Damage reduces stiffness → frequency drops
- Multiple resonant modes typical for multi-story structures
- Environmental noise from wind, temperature effects

Reference: LUMO (Leibniz University Test Structure for Monitoring)
- 9m lattice tower, 1651.61 Hz sampling, accelerometers at 10 levels
- Damage mechanisms: bracing removal (stiffness+mass reduction) or bolt loosening (stiffness only)
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StructuralCondition:
    """Represents a structural state (healthy or damaged)."""
    name: str
    description: str
    state_label: int  # 2=healthy, 3=damaged(removed), 4=damaged(loosened)
    frequency_factor: float  # Multiplier for base frequency (1.0=healthy, <1.0=damaged)
    damping_factor: float  # Multiplier for damping ratio
    noise_factor: float  # Multiplier for noise level


# Define structural conditions based on LUMO damage states
STRUCTURAL_CONDITIONS = {
    'healthy': StructuralCondition(
        name='Healthy Baseline',
        description='Undamaged structure, all bracings intact',
        state_label=2,
        frequency_factor=1.0,
        damping_factor=1.0,
        noise_factor=1.0
    ),
    'dam3_partial': StructuralCondition(
        name='DAM3 Partial',
        description='Single bracing loosened at level 3 (bolt loosening)',
        state_label=4,
        frequency_factor=0.97,  # ~3% frequency drop
        damping_factor=1.1,
        noise_factor=1.05
    ),
    'dam3_full': StructuralCondition(
        name='DAM3 Full',
        description='All bracings removed at level 3 (stiffness+mass reduction)',
        state_label=3,
        frequency_factor=0.92,  # ~8% frequency drop
        damping_factor=1.2,
        noise_factor=1.1
    ),
    'dam6_partial': StructuralCondition(
        name='DAM6 Partial',
        description='Single bracing loosened at level 6 (near base)',
        state_label=4,
        frequency_factor=0.95,  # ~5% frequency drop (base damage more significant)
        damping_factor=1.15,
        noise_factor=1.08
    ),
    'dam6_full': StructuralCondition(
        name='DAM6 Full',
        description='All bracings removed at level 6 (severe base damage)',
        state_label=3,
        frequency_factor=0.85,  # ~15% frequency drop
        damping_factor=1.4,
        noise_factor=1.2
    ),
    'multi_damage': StructuralCondition(
        name='Multi-Level Damage',
        description='Damage at both DAM3 and DAM6 (multiple mechanism removal)',
        state_label=3,
        frequency_factor=0.78,  # ~22% frequency drop
        damping_factor=1.6,
        noise_factor=1.3
    )
}


class StructuralDataGenerator:
    """
    Generates synthetic structural vibration data with physics-based parameters.

    Simulates a lattice tower structure with:
    - Multiple resonant modes (fundamental + harmonics)
    - Realistic damping characteristics
    - Environmental noise (wind, temperature effects)
    - Sensor noise (accelerometer characteristics)
    """

    def __init__(
        self,
        sampling_rate: float = 1651.61,  # LUMO sampling rate
        base_frequency: float = 2.8,  # Fundamental mode ~2.8 Hz (typical for 9m tower)
        base_damping: float = 0.02,  # 2% damping ratio (typical for steel structures)
        duration: float = 600.0,  # 10 minutes of data
        seed: Optional[int] = None
    ):
        self.fs = sampling_rate
        self.f0 = base_frequency
        self.zeta = base_damping
        self.duration = duration
        self.rng = np.random.default_rng(seed)

        # Modal frequencies (first 5 modes)
        # Typical mode shape ratios for cantilever-like structures
        self.mode_ratios = np.array([1.0, 2.8, 5.4, 8.9, 13.2])

        # Mode participation factors (how much each mode contributes)
        self.mode_weights = np.array([1.0, 0.3, 0.15, 0.08, 0.04])

    def generate_condition(
        self,
        condition: StructuralCondition,
        n_windows: int = 8
    ) -> Dict:
        """
        Generate vibration data for a specific structural condition.

        Returns dict with:
        - time: time vector
        - acceleration: acceleration signal (m/s²)
        - envelope: Hilbert envelope of bandpass-filtered signal
        - condition: condition metadata
        - windows: list of window data for multi-window analysis
        """
        n_samples = int(self.fs * self.duration)
        t = np.arange(n_samples) / self.fs

        # Compute modified parameters based on damage
        f_fundamental = self.f0 * condition.frequency_factor
        zeta = self.zeta * condition.damping_factor

        # Generate multi-modal response
        accel = np.zeros(n_samples)

        for i, (ratio, weight) in enumerate(zip(self.mode_ratios, self.mode_weights)):
            f_mode = f_fundamental * ratio

            # Skip modes above Nyquist
            if f_mode >= self.fs / 2:
                continue

            # Generate damped sinusoidal response with amplitude modulation
            # (simulates wind-induced vibration)
            amplitude = weight * self._generate_amplitude_modulation(n_samples, f_mode)
            phase = self.rng.uniform(0, 2 * np.pi)

            # Add slight frequency wandering (temperature effects)
            freq_wander = 1 + 0.002 * np.sin(2 * np.pi * t / 300)  # 5-minute cycle

            accel += amplitude * np.sin(2 * np.pi * f_mode * freq_wander * t + phase)

        # Add environmental noise (broadband wind noise)
        wind_noise = self._generate_wind_noise(n_samples, condition.noise_factor)
        accel += wind_noise

        # Add sensor noise (accelerometer noise floor)
        sensor_noise = self.rng.normal(0, 0.001 * condition.noise_factor, n_samples)
        accel += sensor_noise

        # Scale to realistic acceleration levels (m/s²)
        # Ambient vibration typically 0.001-0.01 m/s² RMS
        accel = accel * 0.005

        # Compute envelope for analysis
        envelope = self._compute_envelope(accel, f_fundamental)

        # Split into windows for multi-window analysis
        windows = self._create_windows(t, accel, envelope, n_windows)

        return {
            'time': t,
            'acceleration': accel,
            'envelope': envelope,
            'condition': condition,
            'fundamental_freq': f_fundamental,
            'sampling_rate': self.fs,
            'windows': windows
        }

    def _generate_amplitude_modulation(
        self,
        n_samples: int,
        freq: float
    ) -> np.ndarray:
        """Generate realistic amplitude modulation (wind gusts, traffic)."""
        # Slow modulation (wind gusts, 0.01-0.1 Hz)
        t = np.arange(n_samples) / self.fs
        slow_mod = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 0.02 * t + self.rng.uniform(0, 2*np.pi)))

        # Medium modulation (traffic, pedestrians, 0.1-1 Hz)
        medium_mod = 0.8 + 0.2 * self.rng.normal(0, 1, n_samples)
        medium_mod = np.convolve(medium_mod, np.ones(int(self.fs/2))/int(self.fs/2), mode='same')
        medium_mod = np.clip(medium_mod, 0.3, 1.5)

        return slow_mod * medium_mod

    def _generate_wind_noise(
        self,
        n_samples: int,
        noise_factor: float
    ) -> np.ndarray:
        """Generate colored noise simulating wind-induced vibrations."""
        # Pink noise (1/f spectrum) - characteristic of wind
        white = self.rng.normal(0, 1, n_samples)

        # Create pink noise via filtering
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        pink = signal.lfilter(b, a, white)

        # Scale by noise factor
        return pink * 0.1 * noise_factor

    def _compute_envelope(
        self,
        accel: np.ndarray,
        f_fundamental: float
    ) -> np.ndarray:
        """Compute Hilbert envelope of bandpass-filtered acceleration."""
        # Bandpass filter around fundamental frequency
        f_low = max(0.5, f_fundamental * 0.5)
        f_high = min(f_fundamental * 2.0, self.fs / 2 - 1)

        # Design bandpass filter
        nyq = self.fs / 2
        low = f_low / nyq
        high = f_high / nyq

        # Ensure valid filter parameters
        if low >= high or high >= 1.0:
            # Fallback: just use the signal as-is
            return np.abs(signal.hilbert(accel))

        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, accel)

        # Hilbert envelope
        analytic = signal.hilbert(filtered)
        envelope = np.abs(analytic)

        return envelope

    def _create_windows(
        self,
        t: np.ndarray,
        accel: np.ndarray,
        envelope: np.ndarray,
        n_windows: int
    ) -> List[Dict]:
        """Split data into overlapping windows for multi-window analysis."""
        n_samples = len(t)
        window_size = n_samples // (n_windows // 2 + 1)  # 50% overlap
        step = window_size // 2

        windows = []
        for i in range(n_windows):
            start = i * step
            end = start + window_size

            if end > n_samples:
                break

            windows.append({
                'time': t[start:end],
                'acceleration': accel[start:end],
                'envelope': envelope[start:end],
                'start_idx': start,
                'end_idx': end
            })

        return windows


def generate_all_conditions(
    seed: int = 42,
    duration: float = 600.0
) -> Dict[str, Dict]:
    """
    Generate data for all structural conditions.

    Returns dict mapping condition names to their data.
    """
    generator = StructuralDataGenerator(seed=seed, duration=duration)

    results = {}
    for name, condition in STRUCTURAL_CONDITIONS.items():
        print(f"Generating data for: {condition.name}")
        results[name] = generator.generate_condition(condition)

    return results


if __name__ == '__main__':
    # Test data generation
    print("Testing structural data generator...")
    data = generate_all_conditions(seed=42, duration=60.0)

    for name, d in data.items():
        cond = d['condition']
        print(f"\n{cond.name}:")
        print(f"  Fundamental freq: {d['fundamental_freq']:.3f} Hz")
        print(f"  Accel RMS: {np.std(d['acceleration'])*1000:.3f} mm/s²")
        print(f"  Envelope mean: {np.mean(d['envelope'])*1000:.3f} mm/s²")
        print(f"  Windows: {len(d['windows'])}")

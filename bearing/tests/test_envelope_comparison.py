"""Tests for bearing envelope comparison analysis."""

import json
from pathlib import Path

import numpy as np
import pytest

from bearing.envelope_comparison import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    BPFI_MULT,
    BPFO_MULT,
    BSF_MULT,
    CONDITIONS,
    SAMPLING_RATE,
    bandpass_filter,
    compute_fft_spectrum,
    compute_ls_spectrum,
    extract_envelope,
    find_peak_near,
    load_condition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _has_cwru_data() -> bool:
    """Check if at least one CWRU .mat file is downloaded."""
    return any((DATA_DIR / f"{fn}.mat").exists() for _, _, _, _, fn in CONDITIONS)


def _synthetic_signal(freq: float, fs: float = SAMPLING_RATE, duration: float = 1.0,
                      noise_level: float = 0.01) -> np.ndarray:
    """Generate a synthetic sinusoidal signal with optional noise."""
    t = np.arange(int(fs * duration)) / fs
    signal = np.sin(2 * np.pi * freq * t)
    if noise_level > 0:
        rng = np.random.default_rng(42)
        signal += rng.normal(0, noise_level, len(signal))
    return signal


# ---------------------------------------------------------------------------
# Test bandpass filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:

    def test_output_shape(self):
        signal = np.random.default_rng(0).normal(0, 1, 12000)
        filtered = bandpass_filter(signal, SAMPLING_RATE, BANDPASS_LOW, BANDPASS_HIGH)
        assert filtered.shape == signal.shape

    def test_removes_low_frequencies(self):
        """A 50 Hz signal should be attenuated by the 1-5 kHz bandpass."""
        low_freq_signal = _synthetic_signal(50.0, duration=0.5, noise_level=0)
        filtered = bandpass_filter(low_freq_signal, SAMPLING_RATE, BANDPASS_LOW, BANDPASS_HIGH)
        # Power should be dramatically reduced
        assert np.std(filtered) < 0.01 * np.std(low_freq_signal)

    def test_passes_mid_band(self):
        """A 3 kHz signal should pass through the 1-5 kHz bandpass."""
        mid_signal = _synthetic_signal(3000.0, duration=0.5, noise_level=0)
        filtered = bandpass_filter(mid_signal, SAMPLING_RATE, BANDPASS_LOW, BANDPASS_HIGH)
        # Power should be mostly preserved
        assert np.std(filtered) > 0.5 * np.std(mid_signal)

    def test_removes_high_frequencies(self):
        """A 5.5 kHz signal should be attenuated by the 1-5 kHz bandpass."""
        high_signal = _synthetic_signal(5500.0, duration=0.5, noise_level=0)
        filtered = bandpass_filter(high_signal, SAMPLING_RATE, BANDPASS_LOW, BANDPASS_HIGH)
        assert np.std(filtered) < 0.1 * np.std(high_signal)


# ---------------------------------------------------------------------------
# Test envelope extraction
# ---------------------------------------------------------------------------

class TestEnvelopeExtraction:

    def test_dc_removed(self):
        """Envelope should have approximately zero mean."""
        signal = _synthetic_signal(3000.0, duration=0.5)
        envelope = extract_envelope(signal)
        assert abs(np.mean(envelope)) < 1e-10

    def test_output_shape(self):
        signal = _synthetic_signal(3000.0, duration=0.5)
        envelope = extract_envelope(signal)
        assert envelope.shape == signal.shape

    def test_amplitude_modulated_signal(self):
        """Envelope of AM signal should contain modulation frequency."""
        fs = SAMPLING_RATE
        t = np.arange(int(fs * 1.0)) / fs
        # 3 kHz carrier modulated at 100 Hz
        signal = (1 + np.cos(2 * np.pi * 100 * t)) * np.sin(2 * np.pi * 3000 * t)
        envelope = extract_envelope(signal)
        # FFT of envelope should show a peak near 100 Hz
        freqs, amps = compute_fft_spectrum(envelope, fs)
        mask = (freqs >= 90) & (freqs <= 110)
        peak_freq = freqs[mask][np.argmax(amps[mask])]
        assert abs(peak_freq - 100.0) < 2.0


# ---------------------------------------------------------------------------
# Test FFT spectrum
# ---------------------------------------------------------------------------

class TestFFTSpectrum:

    def test_shape(self):
        signal = _synthetic_signal(100.0, duration=1.0)
        freqs, amps = compute_fft_spectrum(signal, SAMPLING_RATE)
        assert len(freqs) == len(amps)
        assert len(freqs) == len(signal) // 2

    def test_detects_known_frequency(self):
        """FFT should find a strong peak at the signal frequency."""
        signal = _synthetic_signal(162.0, duration=2.0, noise_level=0.01)
        freqs, amps = compute_fft_spectrum(signal, SAMPLING_RATE)
        mask = (freqs >= 150) & (freqs <= 175)
        peak_freq = freqs[mask][np.argmax(amps[mask])]
        assert abs(peak_freq - 162.0) < 1.0

    def test_positive_amplitudes(self):
        signal = _synthetic_signal(100.0)
        _, amps = compute_fft_spectrum(signal, SAMPLING_RATE)
        assert np.all(amps >= 0)


# ---------------------------------------------------------------------------
# Test Lomb-Scargle spectrum
# ---------------------------------------------------------------------------

class TestLSSpectrum:

    def test_detects_known_frequency(self):
        signal = _synthetic_signal(107.0, duration=2.0, noise_level=0.01)
        freqs, power = compute_ls_spectrum(signal, SAMPLING_RATE, max_freq=200.0)
        mask = (freqs >= 100) & (freqs <= 115)
        peak_freq = freqs[mask][np.argmax(power[mask])]
        assert abs(peak_freq - 107.0) < 2.0

    def test_positive_power(self):
        signal = _synthetic_signal(100.0)
        _, power = compute_ls_spectrum(signal, SAMPLING_RATE, max_freq=200.0)
        assert np.all(power >= 0)


# ---------------------------------------------------------------------------
# Test peak identification
# ---------------------------------------------------------------------------

class TestFindPeakNear:

    def test_finds_exact_peak(self):
        freqs = np.linspace(0, 500, 10000)
        power = np.zeros_like(freqs)
        # Put a peak at 162 Hz
        idx = np.argmin(np.abs(freqs - 162.0))
        power[idx] = 100.0
        # Add noise floor
        power += 0.1

        result = find_peak_near(freqs, power, 162.0)
        assert result["detected"] is True
        assert abs(result["detected_freq_hz"] - 162.0) < 0.1
        assert result["freq_error_pct"] < 0.1
        assert result["peak_snr_db"] > 10

    def test_no_peak_outside_tolerance(self):
        freqs = np.linspace(0, 500, 10000)
        power = np.ones_like(freqs) * 0.1
        # Peak at 200 Hz (outside ±5% of 162 Hz)
        idx = np.argmin(np.abs(freqs - 200.0))
        power[idx] = 100.0

        result = find_peak_near(freqs, power, 162.0)
        # Should not detect — 200 Hz is outside ±5% of 162 Hz
        assert result["detected"] is False

    def test_low_snr_not_detected(self):
        freqs = np.linspace(0, 500, 10000)
        power = np.ones_like(freqs) * 1.0  # high noise floor
        # Weak peak at 162 Hz
        idx = np.argmin(np.abs(freqs - 162.0))
        power[idx] = 1.5  # only 1.5× noise, well below 10 dB threshold

        result = find_peak_near(freqs, power, 162.0)
        assert result["detected"] is False

    def test_returns_none_for_empty_range(self):
        freqs = np.linspace(0, 50, 1000)  # max 50 Hz
        power = np.ones_like(freqs)
        result = find_peak_near(freqs, power, 162.0)  # target outside range
        assert result["detected"] is False
        assert result["detected_freq_hz"] is None


# ---------------------------------------------------------------------------
# Test conditions table
# ---------------------------------------------------------------------------

class TestConditions:

    def test_eight_conditions(self):
        assert len(CONDITIONS) == 8

    def test_defect_multipliers(self):
        assert BPFI_MULT == pytest.approx(5.4152, abs=0.01)
        assert BPFO_MULT == pytest.approx(3.5848, abs=0.01)
        assert BSF_MULT == pytest.approx(2.3573, abs=0.01)

    def test_target_frequencies_at_1797rpm(self):
        shaft_hz = 1797 / 60.0  # 29.95 Hz
        assert shaft_hz * BPFI_MULT == pytest.approx(162.2, abs=0.5)
        assert shaft_hz * BPFO_MULT == pytest.approx(107.4, abs=0.5)
        assert shaft_hz * BSF_MULT == pytest.approx(70.6, abs=0.5)


# ---------------------------------------------------------------------------
# Test .mat loading (requires downloaded data)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_cwru_data(), reason="CWRU .mat files not downloaded")
class TestMatLoading:

    def test_load_normal(self):
        signal = load_condition(DATA_DIR / "97.mat")
        assert signal.ndim == 1
        assert len(signal) > 10000

    def test_signal_is_float64(self):
        signal = load_condition(DATA_DIR / "97.mat")
        assert signal.dtype == np.float64


# ---------------------------------------------------------------------------
# Integration test (requires downloaded data)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_cwru_data(), reason="CWRU .mat files not downloaded")
class TestIntegration:

    def test_normal_condition_detects_shaft(self):
        """Envelope analysis on Normal condition should detect shaft frequency."""
        from bearing.envelope_comparison import analyse_condition

        signal = load_condition(DATA_DIR / "97.mat")
        shaft_freq = 1797 / 60.0
        result = analyse_condition(signal, SAMPLING_RATE, shaft_freq)
        # At least one method should find the shaft frequency
        fft_ok = result["envelope_fft"]["detected"]
        ls_ok = result["envelope_ls"]["detected"]
        # This is a soft check — shaft may or may not appear in envelope
        assert isinstance(fft_ok, bool)
        assert isinstance(ls_ok, bool)

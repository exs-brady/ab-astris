"""
Stage 2: Spectral estimation via Lomb-Scargle periodogram.

The Lomb-Scargle periodogram handles unevenly sampled data natively and
provides a formal false alarm probability (FAP) for each spectral peak.
This is the default spectral front-end for all Ab Astris domains; the only
substitution is BLS for planetary transit detection (see astronomy/).

Reference:
    Lomb (1976), Scargle (1982), VanderPlas (2018)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.timeseries import LombScargle


@dataclass
class LSResult:
    """Results from a Lomb-Scargle periodogram analysis."""
    frequency: float          # Peak frequency (Hz)
    power: float              # Peak power (dimensionless)
    fap: float                # False alarm probability
    frequencies: np.ndarray = field(repr=False)  # Full frequency grid
    powers: np.ndarray = field(repr=False)        # Full power spectrum


def run_lomb_scargle(
    time: np.ndarray,
    signal: np.ndarray,
    freq_min: float,
    freq_max: float,
    n_points: int = 10_000,
    normalize: bool = True,
) -> LSResult:
    """Compute a Lomb-Scargle periodogram over a specified frequency range.

    Parameters
    ----------
    time : array
        Observation times (any consistent unit).
    signal : array
        Signal values.
    freq_min, freq_max : float
        Frequency range bounds (same unit as 1/time).
    n_points : int
        Number of logarithmically spaced frequency grid points.
    normalize : bool
        If True, z-score normalize the signal before analysis.

    Returns
    -------
    LSResult
        Peak frequency, power, FAP, and full spectrum arrays.
    """
    if normalize:
        mu = np.mean(signal)
        sigma = np.std(signal)
        data = (signal - mu) / sigma if sigma > 1e-15 else signal - mu
    else:
        data = signal

    frequencies = np.linspace(freq_min, freq_max, n_points)
    ls = LombScargle(time, data)
    powers = ls.power(frequencies)

    best_idx = int(np.argmax(powers))
    best_freq = float(frequencies[best_idx])
    best_power = float(powers[best_idx])

    try:
        fap = float(ls.false_alarm_probability(best_power))
    except Exception:
        fap = 1.0

    return LSResult(
        frequency=best_freq,
        power=best_power,
        fap=fap,
        frequencies=frequencies,
        powers=powers,
    )

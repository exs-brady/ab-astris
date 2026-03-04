"""
Stage 4: Bootstrap error estimation for detected frequencies.

Percentile bootstrap resampling provides non-parametric confidence
intervals on the peak frequency detected by Lomb-Scargle.  The signal
is resampled with replacement, temporal order is restored, and the
periodogram is recomputed on each resample.

Reference: Section 2.1 (Stage 4) of the paper;
           Efron & Tibshirani (1993).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .periodogram import run_lomb_scargle


@dataclass
class BootstrapResult:
    """Results from bootstrap frequency estimation."""
    freq_mean: float       # Mean of bootstrap frequency distribution
    freq_std: float        # Std dev (1-sigma uncertainty)
    freq_ci_low: float     # 2.5th percentile (95% CI lower)
    freq_ci_high: float    # 97.5th percentile (95% CI upper)
    n_successful: int      # Number of successful bootstrap iterations


def run_bootstrap(
    time: np.ndarray,
    signal: np.ndarray,
    freq_min: float,
    freq_max: float,
    n_bootstrap: int = 1_000,
    n_points: int = 10_000,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap resampling for frequency uncertainty estimation.

    Parameters
    ----------
    time : array
        Observation times.
    signal : array
        Signal values.
    freq_min, freq_max : float
        Frequency range for Lomb-Scargle analysis.
    n_bootstrap : int
        Number of bootstrap iterations (default 1,000).
    n_points : int
        Frequency grid points per iteration.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
        Mean, std, and 95% CI of the bootstrap frequency distribution.
    """
    n_samples = len(time)
    bootstrap_freqs = []
    rng = np.random.default_rng(seed)

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        indices = np.sort(indices)  # Maintain temporal order

        t_boot = time[indices]
        s_boot = signal[indices]

        try:
            result = run_lomb_scargle(
                t_boot, s_boot,
                freq_min=freq_min, freq_max=freq_max,
                n_points=n_points,
            )
            bootstrap_freqs.append(result.frequency)
        except Exception:
            continue

    bootstrap_freqs = np.array(bootstrap_freqs)
    n_successful = len(bootstrap_freqs)

    if n_successful < 2:
        return BootstrapResult(
            freq_mean=float("nan"),
            freq_std=float("nan"),
            freq_ci_low=float("nan"),
            freq_ci_high=float("nan"),
            n_successful=n_successful,
        )

    return BootstrapResult(
        freq_mean=float(np.mean(bootstrap_freqs)),
        freq_std=float(np.std(bootstrap_freqs)),
        freq_ci_low=float(np.percentile(bootstrap_freqs, 2.5)),
        freq_ci_high=float(np.percentile(bootstrap_freqs, 97.5)),
        n_successful=n_successful,
    )

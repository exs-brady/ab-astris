"""
Ab Astris — Cross-Domain Signal Detection Pipeline.

A five-stage pipeline for periodic signal detection:
  1. Preprocessing (domain-specific bandpass + Hilbert envelope)
  2. Spectral estimation (Lomb-Scargle periodogram)
  3. Multi-window validation (CV stability metric)
  4. Bootstrap error estimation
  5. Composite confidence scoring

See: Brady (2025), "Ab Astris: Cross-Domain Transfer of a
Physics-Constrained Signal Detection Pipeline"
"""

from .periodogram import run_lomb_scargle, LSResult
from .multi_window import run_multi_window_validation, create_windows, MultiWindowResult
from .bootstrap import run_bootstrap, BootstrapResult
from .scoring import compute_confidence_score

__all__ = [
    "run_lomb_scargle",
    "LSResult",
    "run_multi_window_validation",
    "create_windows",
    "MultiWindowResult",
    "run_bootstrap",
    "BootstrapResult",
    "compute_confidence_score",
]

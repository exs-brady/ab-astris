"""
Stage 5: Composite confidence scoring.

Four-component scoring system (0-100 points) that combines frequency
accuracy, statistical significance, multi-window stability, and
detection rate into a single confidence metric.

    Component               Weight
    ─────────────────────── ──────
    Frequency accuracy       30 pts
    FAP significance         25 pts
    CV stability             25 pts
    Detection rate           20 pts

Tier thresholds: CONFIRMED ≥ 90, PROBABLE ≥ 75, CANDIDATE ≥ 60.

Reference: Section 2.1 (Stage 5) and Table 1 of the paper.
"""

from __future__ import annotations

from typing import Tuple

from .periodogram import LSResult
from .multi_window import MultiWindowResult


def compute_confidence_score(
    ls_result: LSResult,
    mw_result: MultiWindowResult,
    target_freq: float,
) -> Tuple[float, str]:
    """Compute composite confidence score (0-100).

    Parameters
    ----------
    ls_result : LSResult
        Lomb-Scargle periodogram result.
    mw_result : MultiWindowResult
        Multi-window validation result.
    target_freq : float
        Expected / target frequency for accuracy scoring.

    Returns
    -------
    score : float
        Composite score (0-100).
    tier : str
        One of "CONFIRMED", "PROBABLE", "CANDIDATE", "NOISE".
    """
    score = 0.0

    # 1. Frequency accuracy (30 pts)
    freq_error = abs(ls_result.frequency - target_freq) / target_freq
    if freq_error < 0.001:
        score += 30
    elif freq_error < 0.005:
        score += 25
    elif freq_error < 0.01:
        score += 20
    elif freq_error < 0.05:
        score += 15
    elif freq_error < 0.10:
        score += 5

    # 2. FAP significance (25 pts)
    fap = ls_result.fap
    if fap < 1e-10:
        score += 25
    elif fap < 1e-5:
        score += 20
    elif fap < 1e-3:
        score += 15
    elif fap < 0.01:
        score += 10
    elif fap < 0.05:
        score += 5

    # 3. CV stability (25 pts)
    cv = mw_result.cv_clean
    if cv < 0.01:
        score += 25
    elif cv < 0.1:
        score += 20
    elif cv < 1.0:
        score += 15
    elif cv < 5.0:
        score += 10
    elif cv < 10.0:
        score += 5

    # 4. Detection rate (20 pts)
    score += 20.0 * mw_result.detection_rate

    # Tier assignment
    if score >= 90:
        tier = "CONFIRMED"
    elif score >= 75:
        tier = "PROBABLE"
    elif score >= 60:
        tier = "CANDIDATE"
    else:
        tier = "NOISE"

    return round(score, 1), tier

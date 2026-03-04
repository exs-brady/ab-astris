#!/usr/bin/env python3
"""
Unit tests for TorqueScope ablation study.

Tests the ablation_mode parameter logic using mocked data structures.
These tests verify the ablation branching logic without requiring the
full CARE dataset or all torquescope_phase2 dependencies.

Usage:
    cd /path/to/pinn
    python3 -m pytest torquescope/tests/test_ablation.py -v
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

# ============================================================================
# Mock dataclasses matching anomaly_detector.py definitions
# (avoids importing anomaly_detector.py which has torquescope_phase2 deps)
# ============================================================================


@dataclass
class SignalAnomaly:
    signal_name: str
    anomaly_score: float
    trend_score: float
    volatility_score: float
    periodic_score: float
    periodic_degraded: bool = False


@dataclass
class WindowResult:
    window_start: int
    window_end: int
    overall_score: float
    n_signals_anomalous: int
    n_periodic_degraded: int
    signal_results: List[SignalAnomaly]
    periodic_baseline_used: bool = False


# ============================================================================
# Ablation scoring functions extracted from anomaly_detector.py
# These mirror the ablation branches in detect_multi_signal_hybrid() Stage 4
# ============================================================================


def score_periodic_only(heur_result: WindowResult) -> float:
    """Compute periodic-only ablation score (mirrors anomaly_detector.py logic)."""
    periodic_scores = []
    n_degraded = 0
    for sr in heur_result.signal_results:
        periodic_scores.append(sr.periodic_score)
        if sr.periodic_degraded:
            n_degraded += 1

    if n_degraded >= 3:
        return 0.95
    elif n_degraded >= 2:
        return 0.85
    elif periodic_scores:
        return max(periodic_scores)
    else:
        return 0.0


def score_nbm_only(nbm_score: float) -> float:
    """Compute nbm-only ablation score (mirrors anomaly_detector.py logic)."""
    return nbm_score


def score_hybrid(
    heuristic_score: float,
    nbm_score: float,
    suppression_factor: float = 0.5,
) -> float:
    """Compute hybrid score (mirrors anomaly_detector.py logic)."""
    if heuristic_score >= 0.3 and nbm_score >= 0.2:
        return min(1.0, 0.7 + 0.3 * max(heuristic_score, nbm_score))
    elif heuristic_score >= 0.3 and nbm_score < 0.2:
        return suppression_factor * heuristic_score
    elif heuristic_score < 0.3 and nbm_score >= 0.3:
        return 0.4 + 0.2 * nbm_score
    else:
        return max(heuristic_score, nbm_score) * 0.1


# ============================================================================
# Test Fixtures
# ============================================================================


def make_window_result(
    periodic_scores=None,
    periodic_degraded=None,
    overall_score=0.5,
    n_signals=3,
):
    """Create a WindowResult with specified periodic scores."""
    if periodic_scores is None:
        periodic_scores = [0.1, 0.2, 0.15]
    if periodic_degraded is None:
        periodic_degraded = [False] * len(periodic_scores)

    signal_results = [
        SignalAnomaly(
            signal_name=f"signal_{i}",
            anomaly_score=overall_score,
            trend_score=0.3,
            volatility_score=0.2,
            periodic_score=ps,
            periodic_degraded=pd,
        )
        for i, (ps, pd) in enumerate(zip(periodic_scores, periodic_degraded))
    ]

    return WindowResult(
        window_start=0,
        window_end=1000,
        overall_score=overall_score,
        n_signals_anomalous=sum(1 for s in signal_results if s.anomaly_score > 0.3),
        n_periodic_degraded=sum(1 for pd in periodic_degraded if pd),
        signal_results=signal_results,
        periodic_baseline_used=True,
    )


# ============================================================================
# Tests
# ============================================================================


class TestPeriodicOnlyScoring:
    """Test periodic_only ablation scoring logic."""

    def test_max_periodic_score(self):
        """Should return max periodic score across signals."""
        wr = make_window_result(periodic_scores=[0.1, 0.6, 0.3])
        assert score_periodic_only(wr) == 0.6

    def test_degraded_two_signals(self):
        """Two degraded signals should score 0.85."""
        wr = make_window_result(
            periodic_scores=[0.5, 0.4, 0.1],
            periodic_degraded=[True, True, False],
        )
        assert score_periodic_only(wr) == 0.85

    def test_degraded_three_signals(self):
        """Three degraded signals should score 0.95."""
        wr = make_window_result(
            periodic_scores=[0.5, 0.4, 0.3],
            periodic_degraded=[True, True, True],
        )
        assert score_periodic_only(wr) == 0.95

    def test_no_signals(self):
        """Empty signal results should score 0.0."""
        wr = WindowResult(
            window_start=0, window_end=1000, overall_score=0.5,
            n_signals_anomalous=0, n_periodic_degraded=0,
            signal_results=[], periodic_baseline_used=False,
        )
        assert score_periodic_only(wr) == 0.0

    def test_ignores_heuristic_score(self):
        """periodic_only should NOT use heur_result.overall_score."""
        wr = make_window_result(
            periodic_scores=[0.1, 0.05],
            overall_score=0.9,  # High heuristic, low periodic
        )
        # Should return max periodic (0.1), not overall (0.9)
        assert score_periodic_only(wr) == 0.1

    def test_score_in_valid_range(self):
        """All periodic scores should be in [0, 1]."""
        for ps in [[0.0], [0.5], [1.0], [0.3, 0.7, 0.9]]:
            wr = make_window_result(periodic_scores=ps)
            score = score_periodic_only(wr)
            assert 0.0 <= score <= 1.0


class TestNbmOnlyScoring:
    """Test nbm_only ablation scoring logic."""

    def test_passthrough(self):
        """NBM score should be used directly."""
        assert score_nbm_only(0.0) == 0.0
        assert score_nbm_only(0.5) == 0.5
        assert score_nbm_only(1.0) == 1.0

    def test_ignores_heuristic(self):
        """NBM-only mode should not be affected by heuristic scores."""
        # The score_nbm_only function only takes nbm_score,
        # confirming heuristic is not used.
        assert score_nbm_only(0.3) == 0.3


class TestHybridScoring:
    """Test hybrid scoring logic matches v5."""

    def test_both_agree(self):
        """Both agree: score should be in 0.7-1.0 range."""
        score = score_hybrid(0.5, 0.4)
        assert 0.7 <= score <= 1.0

    def test_heuristic_only(self):
        """Heuristic only: score should be suppressed (0.5x)."""
        score = score_hybrid(0.6, 0.1)
        assert abs(score - 0.3) < 0.01  # 0.5 * 0.6 = 0.3

    def test_nbm_only(self):
        """NBM only: score should be moderate (0.4-0.6)."""
        score = score_hybrid(0.1, 0.5)
        assert 0.4 <= score <= 0.6

    def test_neither(self):
        """Neither fires: score should be low (0.0-0.1)."""
        score = score_hybrid(0.1, 0.1)
        assert score <= 0.1

    def test_per_farm_suppression(self):
        """Per-farm suppression factor should scale heuristic-only score."""
        # Farm B: aggressive suppression (0.3)
        score_b = score_hybrid(0.6, 0.1, suppression_factor=0.3)
        assert abs(score_b - 0.18) < 0.01  # 0.3 * 0.6

        # Farm C: permissive (0.7)
        score_c = score_hybrid(0.6, 0.1, suppression_factor=0.7)
        assert abs(score_c - 0.42) < 0.01  # 0.7 * 0.6


class TestAblationModeDifferences:
    """Test that different modes produce different scores for the same input."""

    def test_modes_diverge_on_typical_input(self):
        """Given typical inputs, the three modes should produce different scores."""
        wr = make_window_result(
            periodic_scores=[0.2, 0.15, 0.1],
            periodic_degraded=[False, False, False],
            overall_score=0.6,
        )
        heuristic_score = wr.overall_score
        nbm_score = 0.4

        periodic_score = score_periodic_only(wr)  # max(0.2, 0.15, 0.1) = 0.2
        nbm_only_score = score_nbm_only(nbm_score)  # 0.4
        hybrid_score_val = score_hybrid(heuristic_score, nbm_score)  # both agree => 0.7+

        # All three should be different
        scores = [periodic_score, nbm_only_score, hybrid_score_val]
        assert len(set(scores)) == 3, f"Expected 3 different scores, got {scores}"

    def test_periodic_only_lower_than_hybrid_when_periodic_weak(self):
        """When periodic signals are weak, periodic_only should score lower than hybrid."""
        wr = make_window_result(
            periodic_scores=[0.05, 0.03],
            overall_score=0.7,
        )
        periodic_score = score_periodic_only(wr)
        hybrid_score_val = score_hybrid(0.7, 0.5)  # Both agree

        assert periodic_score < hybrid_score_val


class TestScoreRanges:
    """Validate that all ablation modes produce scores in [0, 1]."""

    def test_periodic_only_bounds(self):
        for _ in range(100):
            ps = np.random.uniform(0, 1, size=np.random.randint(1, 6)).tolist()
            pd = [np.random.random() > 0.5 for _ in ps]
            wr = make_window_result(periodic_scores=ps, periodic_degraded=pd)
            score = score_periodic_only(wr)
            assert 0.0 <= score <= 1.0

    def test_nbm_only_bounds(self):
        for nbm in np.linspace(0, 1, 50):
            score = score_nbm_only(nbm)
            assert 0.0 <= score <= 1.0

    def test_hybrid_bounds(self):
        for h in np.linspace(0, 1, 20):
            for n in np.linspace(0, 1, 20):
                score = score_hybrid(h, n)
                assert 0.0 <= score <= 1.0, f"hybrid({h}, {n}) = {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
TorqueScope Phase 2: CARE Score Evaluation Framework

Implements the exact CARE-score calculation as specified in Gück et al. (2024).
Four components: Coverage, Accuracy, Reliability, Earliness (weights 1,1,1,2).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class CAREResult:
    """Result of CARE-score calculation."""
    coverage: float
    accuracy: float
    reliability: float
    earliness: float
    care_score: float

    # Sub-metrics for debugging
    n_anomaly_datasets: int
    n_normal_datasets: int
    n_detected_events: int  # Event-level detections (criticality >= 72)
    coverage_per_dataset: Dict[str, float] = None
    accuracy_per_dataset: Dict[str, float] = None
    earliness_per_dataset: Dict[str, float] = None


def compute_criticality(status_ids: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """
    Compute criticality counter for converting per-timestamp predictions to event-level alarms.

    Args:
        status_ids: Array where 1 = normal status (original 0 or 2), 0 = abnormal
        predictions: Array where 1 = anomaly detected, 0 = normal

    Returns:
        Criticality array (same length as input)
    """
    N = len(predictions)
    crit = np.zeros(N + 1, dtype=np.int32)

    for i in range(N):
        if status_ids[i] == 1:  # Normal status timestamp
            if predictions[i] == 1:  # Anomaly detected
                crit[i + 1] = crit[i] + 1
            else:
                crit[i + 1] = max(crit[i] - 1, 0)
        else:  # Abnormal status timestamp - carry forward
            crit[i + 1] = crit[i]

    return crit[1:]


def f_beta_score(tp: int, fp: int, fn: int, beta: float = 0.5) -> float:
    """
    Compute F-beta score.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        beta: Beta value (0.5 for CARE = penalizes FP more than FN)

    Returns:
        F-beta score
    """
    if tp == 0:
        return 0.0

    numerator = (1 + beta**2) * tp
    denominator = (1 + beta**2) * tp + beta**2 * fn + fp

    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_coverage_single(
    predictions: np.ndarray,
    status_ids: np.ndarray,
    event_start_idx: int,
    event_end_idx: int,
    test_start_idx: int
) -> float:
    """
    Compute Coverage (F_0.5) for a single anomaly dataset.

    Only considers rows in the prediction period with normal status (0 or 2).
    Ground truth = 1 for all rows in [event_start, event_end], 0 otherwise.

    Args:
        predictions: Binary predictions for entire dataset
        status_ids: Status IDs for entire dataset (0,1,2,3,4,5)
        event_start_idx: Index of event start in the dataset
        event_end_idx: Index of event end in the dataset
        test_start_idx: Index where prediction period starts

    Returns:
        F_0.5 score for this dataset
    """
    # Filter to prediction period only
    pred_mask = np.arange(len(predictions)) >= test_start_idx

    # Filter to normal status only (0 or 2)
    normal_mask = np.isin(status_ids, [0, 2])

    # Combined mask
    valid_mask = pred_mask & normal_mask

    if not np.any(valid_mask):
        return 0.0

    # Ground truth: 1 if in event period, 0 otherwise
    indices = np.arange(len(predictions))
    ground_truth = ((indices >= event_start_idx) & (indices <= event_end_idx)).astype(int)

    # Apply mask
    y_true = ground_truth[valid_mask]
    y_pred = predictions[valid_mask]

    # Calculate TP, FP, FN
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    return f_beta_score(tp, fp, fn, beta=0.5)


def compute_accuracy_single(
    predictions: np.ndarray,
    status_ids: np.ndarray,
    test_start_idx: int
) -> float:
    """
    Compute Accuracy (TN / (FP + TN)) for a single normal dataset.

    Args:
        predictions: Binary predictions for entire dataset
        status_ids: Status IDs for entire dataset
        test_start_idx: Index where prediction period starts

    Returns:
        Accuracy score (true negative rate)
    """
    # Filter to prediction period only
    pred_mask = np.arange(len(predictions)) >= test_start_idx

    # Filter to normal status only (0 or 2)
    normal_mask = np.isin(status_ids, [0, 2])

    # Combined mask
    valid_mask = pred_mask & normal_mask

    if not np.any(valid_mask):
        return 1.0  # No valid predictions = perfect accuracy

    # For normal datasets, all ground truth = 0
    y_pred = predictions[valid_mask]

    # TN = predictions of 0, FP = predictions of 1
    tn = np.sum(y_pred == 0)
    fp = np.sum(y_pred == 1)

    if (tn + fp) == 0:
        return 1.0

    return tn / (tn + fp)


def compute_earliness_single(
    predictions: np.ndarray,
    status_ids: np.ndarray,
    event_start_idx: int,
    event_end_idx: int
) -> float:
    """
    Compute Earliness (weighted score) for a single anomaly dataset.

    Weight function:
    - First half of event period: weight = 1.0
    - Second half: weight decreases linearly from 1.0 to 0.0

    Args:
        predictions: Binary predictions for entire dataset
        status_ids: Status IDs for entire dataset
        event_start_idx: Index of event start
        event_end_idx: Index of event end

    Returns:
        Weighted earliness score
    """
    # Get event period
    event_length = event_end_idx - event_start_idx + 1
    if event_length <= 0:
        return 0.0

    midpoint = event_length / 2

    total_weight = 0.0
    weighted_score = 0.0

    for i in range(event_start_idx, event_end_idx + 1):
        if i >= len(predictions):
            break

        # Only count normal status timestamps
        if status_ids[i] not in [0, 2]:
            continue

        position_in_event = i - event_start_idx

        # Calculate weight
        if position_in_event <= midpoint:
            weight = 1.0
        else:
            weight = max(0.0, 1.0 - (position_in_event - midpoint) / midpoint)

        total_weight += weight
        if predictions[i] == 1:
            weighted_score += weight

    if total_weight == 0:
        return 0.0

    return weighted_score / total_weight


def compute_reliability(
    dataset_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, bool]],
    threshold: int = 72
) -> float:
    """
    Compute Reliability (event-level F_0.5) across all datasets.

    Uses criticality counter to convert per-timestamp predictions to event-level.

    Args:
        dataset_predictions: Dict mapping dataset_id to (predictions, normal_status_mask, is_anomaly)
            - predictions: Binary predictions for entire dataset
            - normal_status_mask: 1 where status is normal (0 or 2), 0 otherwise
            - is_anomaly: True if this is an anomaly dataset
        threshold: Criticality threshold for alarm (default 72 = 12 hours)

    Returns:
        Event-level F_0.5 score
    """
    tp = 0
    fp = 0
    fn = 0

    for dataset_id, (predictions, status_mask, is_anomaly) in dataset_predictions.items():
        # Compute criticality
        crit = compute_criticality(status_mask, predictions)

        # Event detected if max criticality >= threshold
        event_detected = np.max(crit) >= threshold

        if is_anomaly:
            if event_detected:
                tp += 1
            else:
                fn += 1
        else:
            if event_detected:
                fp += 1
            # TN not counted in F-score

    return f_beta_score(tp, fp, fn, beta=0.5)


class CAREScorer:
    """
    CARE Score Calculator.

    Computes all four components and the final weighted score.
    Weights: Coverage=1, Accuracy=1, Reliability=1, Earliness=2
    """

    def __init__(self, criticality_threshold: int = 72):
        """
        Args:
            criticality_threshold: Threshold for event-level alarm (default 72 = 12 hours)
        """
        self.criticality_threshold = criticality_threshold

    def compute_care_score(
        self,
        predictions: Dict[str, np.ndarray],
        metadata: Dict[str, dict]
    ) -> CAREResult:
        """
        Compute full CARE score.

        Args:
            predictions: Dict mapping dataset_id to binary prediction array
            metadata: Dict mapping dataset_id to metadata dict with keys:
                - status_ids: np.ndarray of status IDs
                - event_label: 'anomaly' or 'normal'
                - event_start_idx: int (for anomaly datasets)
                - event_end_idx: int (for anomaly datasets)
                - test_start_idx: int

        Returns:
            CAREResult with all scores
        """
        anomaly_datasets = [k for k, v in metadata.items() if v['event_label'] == 'anomaly']
        normal_datasets = [k for k, v in metadata.items() if v['event_label'] == 'normal']

        # Compute Coverage (anomaly datasets only)
        coverage_scores = {}
        for dataset_id in anomaly_datasets:
            pred = predictions[dataset_id]
            meta = metadata[dataset_id]

            score = compute_coverage_single(
                pred,
                meta['status_ids'],
                meta['event_start_idx'],
                meta['event_end_idx'],
                meta['test_start_idx']
            )
            coverage_scores[dataset_id] = score

        coverage = np.mean(list(coverage_scores.values())) if coverage_scores else 0.0

        # Compute Accuracy (normal datasets only)
        accuracy_scores = {}
        for dataset_id in normal_datasets:
            pred = predictions[dataset_id]
            meta = metadata[dataset_id]

            score = compute_accuracy_single(
                pred,
                meta['status_ids'],
                meta['test_start_idx']
            )
            accuracy_scores[dataset_id] = score

        accuracy = np.mean(list(accuracy_scores.values())) if accuracy_scores else 0.0

        # Compute Earliness (anomaly datasets only)
        earliness_scores = {}
        for dataset_id in anomaly_datasets:
            pred = predictions[dataset_id]
            meta = metadata[dataset_id]

            score = compute_earliness_single(
                pred,
                meta['status_ids'],
                meta['event_start_idx'],
                meta['event_end_idx']
            )
            earliness_scores[dataset_id] = score

        earliness = np.mean(list(earliness_scores.values())) if earliness_scores else 0.0

        # Compute Reliability (all datasets)
        reliability_data = {}
        for dataset_id in predictions:
            pred = predictions[dataset_id]
            meta = metadata[dataset_id]

            # Create normal status mask (1 = normal, 0 = abnormal)
            status_mask = np.isin(meta['status_ids'], [0, 2]).astype(int)
            is_anomaly = meta['event_label'] == 'anomaly'

            reliability_data[dataset_id] = (pred, status_mask, is_anomaly)

        reliability = compute_reliability(reliability_data, self.criticality_threshold)

        # Count detected events
        n_detected = 0
        for dataset_id, (pred, status_mask, _) in reliability_data.items():
            crit = compute_criticality(status_mask, pred)
            if np.max(crit) >= self.criticality_threshold:
                n_detected += 1

        # Compute final CARE score with constraints
        # Constraint 1: CARE = 0 if no anomalies detected across all datasets
        if n_detected == 0 and len(anomaly_datasets) > 0:
            care_score = 0.0
        # Constraint 2: CARE = Accuracy if Accuracy < 0.5
        elif accuracy < 0.5:
            care_score = accuracy
        else:
            # Normal calculation: (Coverage + Accuracy + Reliability + 2*Earliness) / 5
            care_score = (coverage + accuracy + reliability + 2 * earliness) / 5

        return CAREResult(
            coverage=coverage,
            accuracy=accuracy,
            reliability=reliability,
            earliness=earliness,
            care_score=care_score,
            n_anomaly_datasets=len(anomaly_datasets),
            n_normal_datasets=len(normal_datasets),
            n_detected_events=n_detected,
            coverage_per_dataset=coverage_scores,
            accuracy_per_dataset=accuracy_scores,
            earliness_per_dataset=earliness_scores
        )


def validate_scorer_with_baselines():
    """
    Validate CARE scorer with trivial baselines.

    Expected results:
    - All anomaly: Very low accuracy (~0), high coverage
    - All normal: Coverage = 0, high accuracy
    - Random 50/50: ~0.50 CARE
    """
    print("=" * 60)
    print("CARE Scorer Validation with Trivial Baselines")
    print("=" * 60)

    # Create synthetic data for 10 anomaly + 10 normal datasets
    np.random.seed(42)

    n_anomaly = 10
    n_normal = 10
    n_timestamps = 1000
    train_split = 700  # 70% train, 30% test
    event_start = 800  # Event starts at row 800
    event_end = 950    # Event ends at row 950

    # Build metadata
    metadata = {}
    for i in range(n_anomaly):
        dataset_id = f"anomaly_{i}"
        # Status: mostly normal (0), some abnormal
        status_ids = np.zeros(n_timestamps, dtype=int)
        status_ids[np.random.choice(n_timestamps, size=50, replace=False)] = 4  # Some faults

        metadata[dataset_id] = {
            'status_ids': status_ids,
            'event_label': 'anomaly',
            'event_start_idx': event_start,
            'event_end_idx': event_end,
            'test_start_idx': train_split
        }

    for i in range(n_normal):
        dataset_id = f"normal_{i}"
        status_ids = np.zeros(n_timestamps, dtype=int)
        status_ids[np.random.choice(n_timestamps, size=30, replace=False)] = 2  # Some idling

        metadata[dataset_id] = {
            'status_ids': status_ids,
            'event_label': 'normal',
            'event_start_idx': None,
            'event_end_idx': None,
            'test_start_idx': train_split
        }

    scorer = CAREScorer(criticality_threshold=72)

    # Baseline 1: All anomaly (predict 1 everywhere)
    print("\n1. All Anomaly Baseline (predict 1 everywhere)")
    predictions_all_anomaly = {k: np.ones(n_timestamps, dtype=int) for k in metadata}
    result = scorer.compute_care_score(predictions_all_anomaly, metadata)
    print(f"   Coverage:    {result.coverage:.3f}")
    print(f"   Accuracy:    {result.accuracy:.3f}")
    print(f"   Reliability: {result.reliability:.3f}")
    print(f"   Earliness:   {result.earliness:.3f}")
    print(f"   CARE Score:  {result.care_score:.3f}")
    print(f"   Events detected: {result.n_detected_events}")

    # Baseline 2: All normal (predict 0 everywhere)
    print("\n2. All Normal Baseline (predict 0 everywhere)")
    predictions_all_normal = {k: np.zeros(n_timestamps, dtype=int) for k in metadata}
    result = scorer.compute_care_score(predictions_all_normal, metadata)
    print(f"   Coverage:    {result.coverage:.3f}")
    print(f"   Accuracy:    {result.accuracy:.3f}")
    print(f"   Reliability: {result.reliability:.3f}")
    print(f"   Earliness:   {result.earliness:.3f}")
    print(f"   CARE Score:  {result.care_score:.3f}")
    print(f"   Events detected: {result.n_detected_events}")

    # Baseline 3: Random 50/50
    print("\n3. Random 50/50 Baseline")
    predictions_random = {k: np.random.randint(0, 2, n_timestamps) for k in metadata}
    result = scorer.compute_care_score(predictions_random, metadata)
    print(f"   Coverage:    {result.coverage:.3f}")
    print(f"   Accuracy:    {result.accuracy:.3f}")
    print(f"   Reliability: {result.reliability:.3f}")
    print(f"   Earliness:   {result.earliness:.3f}")
    print(f"   CARE Score:  {result.care_score:.3f}")
    print(f"   Events detected: {result.n_detected_events}")

    # Baseline 4: Perfect detector (anomaly in event period only)
    print("\n4. Perfect Detector (anomaly only in event period)")
    predictions_perfect = {}
    for k, meta in metadata.items():
        pred = np.zeros(n_timestamps, dtype=int)
        if meta['event_label'] == 'anomaly':
            pred[meta['event_start_idx']:meta['event_end_idx']+1] = 1
        predictions_perfect[k] = pred

    result = scorer.compute_care_score(predictions_perfect, metadata)
    print(f"   Coverage:    {result.coverage:.3f}")
    print(f"   Accuracy:    {result.accuracy:.3f}")
    print(f"   Reliability: {result.reliability:.3f}")
    print(f"   Earliness:   {result.earliness:.3f}")
    print(f"   CARE Score:  {result.care_score:.3f}")
    print(f"   Events detected: {result.n_detected_events}")

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)


if __name__ == '__main__':
    validate_scorer_with_baselines()

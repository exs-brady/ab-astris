"""
Discovery Scoring Algorithm v2.0
Two-Tier System: Periodic Discoveries vs Variable Detections

Based on: docs/abastris_scoring_rubric_v2.md
Configuration: config/discovery_thresholds.py

Scoring Philosophy:
- Quality over quantity
- Publication-readiness focus
- Ultra-short period emphasis (AbAstris's specialty)
- Transparent scoring with full breakdown
"""

from typing import Dict, Any, Optional
import logging
import sys
import os

# Add project root to path for config import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import discovery_thresholds as config

logger = logging.getLogger(__name__)


class DiscoveryScorerV2:
    """
    Scores variable star discoveries using Rubric v2.0

    Score Range: 0-100 points
    - Base score: 0-80 points (variability + period_sig + quality)
    - Bonuses: +20 ultra-short, +5 high PM, +5 consistency

    Discovery Types:
    - Periodic Discovery: Has significant period (FAP < 0.01)
    - Variable Detection: Variable but no period

    Tier Labels:
    - 90-100: Exceptional
    - 75-89: High-Quality
    - 60-74: Solid
    - 50-59: Moderate
    - 40-49: Marginal
    - 25-39: Questionable
    - 0-24: Low-Quality
    """

    def __init__(self):
        """Initialize scorer with configuration"""
        self.config = config

    def score_discovery(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive discovery score

        Args:
            result: Analysis result dictionary with:
                - variability: float (fractional amplitude)
                - period: Optional[float] (days)
                - fap: float (false alarm probability, 1.0 if no period)
                - n_sectors: int (number of TESS sectors)
                - n_points: int (number of data points)
                - pm_total: Optional[float] (proper motion in mas/yr)
                - discovery_tier: str (periodic_discovery, variable_detection, etc.)

        Returns:
            Dictionary with:
                - score: int (0-100)
                - breakdown: dict (component scores)
                - tier_label: str
                - tier_display: dict (emoji, color, etc.)
                - user_message: str
        """
        # Extract values
        variability = result.get('variability', 0.0)
        period = result.get('period')
        fap = result.get('fap', 1.0)
        n_sectors = result.get('n_sectors', 1)
        n_points = result.get('n_points', 0)
        pm_total = result.get('pm_total', 0.0)
        discovery_tier = result.get('discovery_tier', 'non_variable')

        is_periodic = (discovery_tier == 'periodic_discovery')

        # Initialize breakdown
        breakdown = {
            'base_score': 0,
            'variability_score': 0,
            'period_significance_score': 0,
            'data_quality_score': 0,
            'ultra_short_bonus': 0,
            'high_pm_bonus': 0,
            'consistency_bonus': 0,
            'final_score': 0,
            'components': []
        }

        # ====================================================================
        # COMPONENT 1: Variability Score (0-40 points)
        # ====================================================================
        variability_percent = variability * 100  # Convert to percentage
        variability_score = min(
            config.VARIABILITY_CAP,
            variability_percent * config.VARIABILITY_MULTIPLIER
        )
        breakdown['variability_score'] = int(variability_score)
        breakdown['components'].append({
            'name': 'Variability',
            'value': variability_percent,
            'score': int(variability_score),
            'max': config.VARIABILITY_CAP,
            'description': f'{variability_percent:.2f}% amplitude variation'
        })

        # ====================================================================
        # COMPONENT 2: Period Significance Score (0-25 points)
        # Only for periodic discoveries
        # ====================================================================
        period_sig_score = 0
        if is_periodic and period is not None:
            # Determine FAP bracket
            if fap < config.PERIOD_SIG_BRACKETS['exceptional']['fap']:
                period_sig_score = config.PERIOD_SIG_BRACKETS['exceptional']['points']
                fap_label = 'Exceptional (< 0.01%)'
            elif fap < config.PERIOD_SIG_BRACKETS['high']['fap']:
                period_sig_score = config.PERIOD_SIG_BRACKETS['high']['points']
                fap_label = 'High (< 0.1%)'
            elif fap < config.PERIOD_SIG_BRACKETS['good']['fap']:
                period_sig_score = config.PERIOD_SIG_BRACKETS['good']['points']
                fap_label = 'Good (< 1%)'
            else:
                period_sig_score = 0
                fap_label = 'Below threshold'

            breakdown['period_significance_score'] = period_sig_score
            breakdown['components'].append({
                'name': 'Period Significance',
                'value': fap,
                'score': period_sig_score,
                'max': 25,
                'description': f'FAP: {fap:.2e} ({fap_label})'
            })

        # ====================================================================
        # COMPONENT 3: Data Quality Score (0-15 points)
        # ====================================================================
        # Sector bonus (0-10 points)
        # Handle None values
        n_sectors = n_sectors if n_sectors is not None else 1
        n_points = n_points if n_points is not None else 0

        sector_score = min(
            config.SECTOR_POINTS_CAP,
            n_sectors * config.SECTOR_POINTS_PER_SECTOR
        )

        # Data point bonus (0-5 points)
        data_point_score = 0
        for bracket_name, bracket_config in config.DATA_POINTS_BRACKETS.items():
            if n_points >= bracket_config['threshold']:
                data_point_score = bracket_config['points']
                break

        data_quality_score = sector_score + data_point_score
        breakdown['data_quality_score'] = data_quality_score
        breakdown['components'].append({
            'name': 'Data Quality',
            'value': {'sectors': n_sectors, 'points': n_points},
            'score': data_quality_score,
            'max': 15,
            'description': f'{n_sectors} sectors, {n_points} data points'
        })

        # ====================================================================
        # BASE SCORE (sum of components)
        # ====================================================================
        base_score = (
            breakdown['variability_score'] +
            breakdown['period_significance_score'] +
            breakdown['data_quality_score']
        )
        breakdown['base_score'] = base_score

        # ====================================================================
        # BONUS 1: Ultra-Short Period (+20 points)
        # Only for periodic discoveries with P < 0.1 days
        # ====================================================================
        if is_periodic and period is not None and period < config.ULTRA_SHORT_CUTOFF:
            breakdown['ultra_short_bonus'] = config.ULTRA_SHORT_BONUS
            period_hours = period * 24
            breakdown['components'].append({
                'name': 'Ultra-Short Period Bonus',
                'value': period,
                'score': config.ULTRA_SHORT_BONUS,
                'max': config.ULTRA_SHORT_BONUS,
                'description': f'P = {period:.4f}d ({period_hours:.1f}h) - AbAstris specialty!'
            })

        # ====================================================================
        # BONUS 2: High Proper Motion (+5 points)
        # ====================================================================
        if pm_total is not None and pm_total > config.HIGH_PM_THRESHOLD:
            breakdown['high_pm_bonus'] = config.HIGH_PM_BONUS
            breakdown['components'].append({
                'name': 'High Proper Motion Bonus',
                'value': pm_total,
                'score': config.HIGH_PM_BONUS,
                'max': config.HIGH_PM_BONUS,
                'description': f'PM = {pm_total:.1f} mas/yr (6.7x enrichment in ultra-short)'
            })

        # ====================================================================
        # BONUS 3: Multi-Sector Consistency (future feature, +5 points)
        # ====================================================================
        # TODO: Implement period consistency check across sectors
        # For now, this is always 0
        breakdown['consistency_bonus'] = 0

        # ====================================================================
        # FINAL SCORE (cap at 100)
        # ====================================================================
        final_score = min(100, (
            breakdown['base_score'] +
            breakdown['ultra_short_bonus'] +
            breakdown['high_pm_bonus'] +
            breakdown['consistency_bonus']
        ))

        # ====================================================================
        # APPLY SINGLE-SECTOR CONFIDENCE LIMITS
        # ====================================================================
        # Single-sector detections are preliminary - cap to prevent false excitement
        # Data shows 10 discoveries with scores 72-97 dropped to 17-60 after enrichment
        if n_sectors == 1:
            original_score = final_score
            final_score = min(final_score, config.SINGLE_SECTOR_MAX_SCORE)

            if original_score > config.SINGLE_SECTOR_MAX_SCORE:
                breakdown['single_sector_cap'] = {
                    'applied': True,
                    'original_score': int(original_score),
                    'capped_score': int(final_score),
                    'reason': 'Single-sector preliminary detection'
                }
                breakdown['components'].append({
                    'name': 'Single-Sector Cap',
                    'value': n_sectors,
                    'score': int(final_score - original_score),  # Negative
                    'max': 0,
                    'description': f'Capped from {int(original_score)} (single-sector uncertainty)'
                })

        breakdown['final_score'] = int(final_score)

        # ====================================================================
        # TIER CLASSIFICATION
        # ====================================================================
        tier_label = config.get_tier_label(final_score)
        tier_display = config.get_tier_display(final_score)
        user_message = config.get_user_message(final_score, is_periodic)

        # ====================================================================
        # RETURN COMPREHENSIVE RESULT
        # ====================================================================
        return {
            'score': final_score,
            'breakdown': breakdown,
            'tier_label': tier_label,
            'tier_display': tier_display,
            'user_message': user_message,
            'is_periodic_discovery': is_periodic
        }

    def explain_score(self, result: Dict[str, Any]) -> str:
        """
        Generate human-readable score explanation

        Args:
            result: Result from score_discovery()

        Returns:
            Formatted string explaining the score
        """
        score = result['score']
        breakdown = result['breakdown']
        tier_display = result['tier_display']

        lines = [
            f"{'='*70}",
            f"DISCOVERY SCORE: {score}/100 {tier_display['emoji']}",
            f"Tier: {tier_display['label']}",
            f"{'='*70}",
            "",
            "Score Breakdown:",
            f"  Base Score: {breakdown['base_score']}/80",
        ]

        # Component details
        for component in breakdown['components']:
            lines.append(
                f"    • {component['name']}: {component['score']}/{component['max']} pts"
            )
            lines.append(
                f"      {component['description']}"
            )

        # Bonuses
        if breakdown['ultra_short_bonus'] > 0:
            lines.append(f"  + Ultra-Short Bonus: {breakdown['ultra_short_bonus']} pts")
        if breakdown['high_pm_bonus'] > 0:
            lines.append(f"  + High PM Bonus: {breakdown['high_pm_bonus']} pts")
        if breakdown['consistency_bonus'] > 0:
            lines.append(f"  + Consistency Bonus: {breakdown['consistency_bonus']} pts")

        lines.extend([
            "",
            f"Final Score: {score}/100",
            "",
            f"User Message:",
            f"  {result['user_message']}",
            f"{'='*70}"
        ])

        return "\n".join(lines)

    def compare_scores(self, results: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple discoveries and rank them

        Args:
            results: List of analysis results

        Returns:
            Dictionary with rankings and statistics
        """
        scored_results = []
        for result in results:
            score_data = self.score_discovery(result)
            scored_results.append({
                'tic': result.get('tic'),
                'score': score_data['score'],
                'tier': score_data['tier_label'],
                'is_periodic': score_data['is_periodic_discovery'],
                'period': result.get('period'),
                'variability': result.get('variability')
            })

        # Sort by score descending (handle None scores by treating them as 0)
        scored_results.sort(key=lambda x: x['score'] if x['score'] is not None else 0, reverse=True)

        # Calculate statistics (filter out None scores)
        scores = [r['score'] for r in scored_results if r['score'] is not None]
        periodic_count = sum(1 for r in scored_results if r['is_periodic'])
        variable_count = len(scored_results) - periodic_count

        return {
            'ranked': scored_results,
            'statistics': {
                'total': len(scored_results),
                'periodic_discoveries': periodic_count,
                'variable_detections': variable_count,
                'average_score': sum(scores) / len(scores) if scores else 0,
                'median_score': sorted(scores)[len(scores)//2] if scores else 0,
                'max_score': max(scores) if scores else 0,
                'min_score': min(scores) if scores else 0,
                'exceptional_count': sum(1 for s in scores if s >= 90),
                'high_quality_count': sum(1 for s in scores if 75 <= s < 90),
                'solid_count': sum(1 for s in scores if 60 <= s < 75),
            }
        }


# ============================================================================
# LEGACY SUPPORT (for backward compatibility during migration)
# ============================================================================

class DiscoveryScorer(DiscoveryScorerV2):
    """
    Legacy class name for backward compatibility

    This is just an alias to DiscoveryScorerV2.
    Use DiscoveryScorerV2 in new code.
    """
    pass


# ============================================================================
# MODULE-LEVEL FUNCTIONS (convenience)
# ============================================================================

def score_discovery(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to score a discovery

    Args:
        result: Analysis result dictionary

    Returns:
        Score result dictionary
    """
    scorer = DiscoveryScorerV2()
    return scorer.score_discovery(result)


def classify_discovery_tier(
    period: Optional[float],
    period_fap: float,
    variability: float,
    is_known_variable: bool
) -> str:
    """
    Convenience function to classify discovery tier

    Args:
        period: Period in days (None if no period)
        period_fap: False alarm probability
        variability: Variability metric
        is_known_variable: True if in catalogs

    Returns:
        Discovery tier string
    """
    return config.validate_discovery_classification(
        period=period,
        period_fap=period_fap,
        variability=variability,
        is_known_variable=is_known_variable
    )

"""
Simulation Scoring Engine.

Evaluates agent decisions at the end of a simulation run across five
dimensions, then normalizes the final score to [0, 1].

Dimensions:
  1. Solvency       — Did the company survive without going bankrupt?
  2. Debt Clearance — What fraction of total invoices were paid?
  3. Fiscal Discipline — How well did the agent avoid late fees and interest?
  4. Credit Prudence — How conservatively was the credit line used?
  5. Cash Management — Did the agent finish with more cash than it started?

Each dimension produces a sub-score in [0, 1]. The final score is
a weighted average of all five.
"""

from typing import Dict, Any
from models import SimulationResult, DayLog


# ═══════════════════════════════════════════════════════
# Individual Scoring Dimensions
# ═══════════════════════════════════════════════════════

def _solvency_score(result: SimulationResult) -> float:
    """
    Binary survival check + cash-health gradient.
    - If final cash < -10,000 (bankrupt): 0.0
    - Otherwise: scaled between 0.3 (barely alive) and 1.0 (healthy cash)
    """
    if result.final_cash < -10_000:
        return 0.0
    max_healthy_cash = 50_000 
    cash_ratio = max(0.0, result.final_cash) / max_healthy_cash
    return 0.3 + 0.7 * min(1.0, cash_ratio)


def _debt_clearance_score(result: SimulationResult) -> float:
    """
    What fraction of invoices that appeared were fully paid?
    Perfect score = all paid, 0 = none paid.
    """
    total = result.total_invoices
    if total == 0:
        return 1.0
    return result.invoices_paid / total


def _fiscal_discipline_score(result: SimulationResult) -> float:
    """
    Penalizes accumulated late fees and interest relative to total
    invoice volume. A perfect agent pays zero fees.

    Score = 1.0 - (total_penalties / max_penalty_budget)
    where max_penalty_budget is calibrated per difficulty.
    """
    total_penalties = result.total_late_fees + result.total_interest

    if total_penalties <= 0:
        return 1.0

    budget_map = {"easy": 2_000, "medium": 5_000, "hard": 10_000}
    budget = budget_map.get(result.difficulty, 5_000)

    penalty_ratio = total_penalties / budget
    return max(0.0, 1.0 - penalty_ratio)


def _credit_prudence_score(result: SimulationResult) -> float:
    """
    Penalizes ending with high credit utilization.
    - 0% utilization = 1.0
    - 100% utilization = 0.0
    - Quadratic penalty to more severely punish high usage.
    """
    if result.final_credit_used <= 0:
        return 1.0

    credit_limit = 10_000 
    if result.days:
        first_day = result.days[0]
        credit_limit = max(credit_limit, result.final_credit_used * 1.1)

    utilization = min(1.0, result.final_credit_used / (credit_limit + 1.0))
    return max(0.0, 1.0 - utilization ** 2)


def _cash_management_score(result: SimulationResult) -> float:
    """
    Did the agent grow or preserve cash relative to its starting position?
    - Ending with more cash than starting = 1.0
    - Ending with less but still positive = proportional
    - Ending negative = 0.0
    """
    if not result.days:
        return 0.5

    starting_cash = result.days[0].opening_cash

    if starting_cash <= 0:
        return 1.0 if result.final_cash > 0 else 0.0

    if result.final_cash <= 0:
        return 0.0

    ratio = result.final_cash / starting_cash
    # Ratio > 1.0 means growth, cap at 1.0
    return min(1.0, ratio)



# Weights for each dimension (sum to 1.0)
SCORE_WEIGHTS = {
    "solvency":          0.25,
    "debt_clearance":    0.30,
    "fiscal_discipline": 0.20,
    "credit_prudence":   0.10,
    "cash_management":   0.15,
}


def compute_simulation_score(result: SimulationResult) -> Dict[str, Any]:
    """
    Compute the final normalized score for a simulation run.

    Returns a dict with:
      - "score": float in [0, 1]
      - "breakdown": per-dimension sub-scores
      - "grade": letter grade (A/B/C/D/F)
    """
    breakdown = {
        "solvency":          round(_solvency_score(result), 4),
        "debt_clearance":    round(_debt_clearance_score(result), 4),
        "fiscal_discipline": round(_fiscal_discipline_score(result), 4),
        "credit_prudence":   round(_credit_prudence_score(result), 4),
        "cash_management":   round(_cash_management_score(result), 4),
    }

    # Weighted average
    final_score = sum(
        breakdown[dim] * SCORE_WEIGHTS[dim] for dim in SCORE_WEIGHTS
    )
    final_score = round(max(0.0, min(1.0, final_score)), 4)

    # Letter grade
    if final_score >= 0.90:
        grade = "A"
    elif final_score >= 0.75:
        grade = "B"
    elif final_score >= 0.55:
        grade = "C"
    elif final_score >= 0.35:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": final_score,
        "breakdown": breakdown,
        "grade": grade,
    }

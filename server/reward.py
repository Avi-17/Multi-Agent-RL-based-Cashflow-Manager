"""
Reward function for CFO agent training using OpenEnv Rubrics.

Components:
  + Liquidity preservation
  + Bill settlement (paying invoices)
  - Late fee penalties
  - Interest penalties
  - Credit usage penalties
"""

from typing import Any, Dict
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum

class BankruptcyRubric(Rubric):
    """Heavy penalty for bankruptcy, small bonus for survival."""
    def forward(self, action: Any, obs: Dict[str, Any]) -> float:
        day_log = obs["day_log"]
        if day_log.closing_cash < -10000:
            return -500.0
        return 10.0  # Survival bonus

class CreditUtilizationRubric(Rubric):
    """Penalizes high credit utilization to prevent gaming."""
    def forward(self, action: Any, obs: Dict[str, Any]) -> float:
        state = obs["state"]
        day_log = obs["day_log"]
        # Credit utilization
        utilization = day_log.closing_credit_used / (state.credit_limit + 1.0)
        return -(utilization ** 2) * 500.0

class LiquidityRubric(Rubric):
    """Rewards paying bills on time. Small cash-holding bonus when no fees."""
    def forward(self, action: Any, obs: Dict[str, Any]) -> float:
        day_log = obs["day_log"]
        fees = day_log.late_fees_incurred
        paid = day_log.invoices_paid_today
        
        # Reward paying invoices; small bonus for zero late fees
        if fees == 0:
            return 20.0 + (paid * 15.0)
        return -(fees * 0.5)

class OperationsRubric(Rubric):
    """Evaluates the day-to-day operations: fees, interest, and invoice clearance."""
    def forward(self, action: Any, obs: Dict[str, Any]) -> float:
        day_log = obs["day_log"]
        
        fees = day_log.late_fees_incurred
        interest = day_log.interest_incurred
        paid = day_log.invoices_paid_today
        overdue = day_log.overdue_invoice_count

        # Penalize overdue invoices heavily, but NOT the total backlog
        # (having unpaid invoices that aren't due yet is fine)
        overdue_penalty = overdue * 15.0
        
        return (
            - 2.0 * fees
            - 1.5 * interest
            - overdue_penalty
            + 120.0 * paid
        )

class CashflowRubric(Rubric):
    """Master rubric orchestrating the scoring system."""
    def __init__(self):
        super().__init__()
        self.composition = WeightedSum(
            [
                BankruptcyRubric(), 
                CreditUtilizationRubric(),
                LiquidityRubric(),
                OperationsRubric()
            ],
            #WeightedSum requires weights to sum to 1.0.
            weights=[0.25, 0.25, 0.25, 0.25]
        )
        
    def forward(self, action: Any, obs: Dict[str, Any]) -> float:
        # we multiply by 4 to preserve the original mathematical magnitude.
        return round(self.composition(action, obs) * 4.0, 2)
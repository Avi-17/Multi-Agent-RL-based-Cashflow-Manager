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
        # Credit limit is tracked in the state
        utilization = day_log.closing_credit_used / (state.credit_limit + 1.0)
        # Scaled up from 50 to 500 to severely punish maxing out credit just to get invoice bonuses
        return -(utilization ** 2) * 500.0

class LiquidityRubric(Rubric):
    """Rewards holding cash, but penalizes it if there are late fees."""
    def forward(self, action: Any, obs: Dict[str, Any]) -> float:
        day_log = obs["day_log"]
        cash = day_log.closing_cash
        fees = day_log.late_fees_incurred
        
        return 0.001 * cash if fees == 0 else -0.001 * cash

class OperationsRubric(Rubric):
    """Evaluates the day-to-day operations: fees, interest, and backlog."""
    def forward(self, action: Any, obs: Dict[str, Any]) -> float:
        day_log = obs["day_log"]
        
        fees = day_log.late_fees_incurred
        interest = day_log.interest_incurred
        paid = day_log.invoices_paid_today
        overdue = day_log.overdue_invoice_count
        active = day_log.active_invoice_count

        overdue_penalty = overdue * 10.0
        backlog_penalty = active * 2.0
        
        return (
            - 2.0 * fees
            - 1.5 * interest
            - overdue_penalty
            - backlog_penalty
            + 100.0 * paid
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
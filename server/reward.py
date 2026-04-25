"""
Reward function for CFO agent training.

Components:
  + Liquidity preservation
  + Bill settlement (paying invoices)
  - Late fee penalties
  - Interest penalties
  - Credit usage penalties
"""

def compute_day_reward(
    cash: float,
    invoices_paid: int,
    late_fees: float,
    interest: float,
    credit_used: float,
    credit_limit: float,
    overdue_count: int,
    total_active: int,
) -> float:
    """
    Compute reward for a single day of simulation.
    Scaled down for smaller numbers.
    """
    # 1. CRITICAL: Bankruptcy Penalty
    if cash < -10000:
        return -500.0

    # 2. Credit Utilization Penalty
    utilization = credit_used / (credit_limit + 1.0)
    util_penalty = (utilization ** 2) * 50.0

    # 3. Liquidity vs Debt Balance
    liquidity_reward = 0.001 * cash if late_fees == 0 else -0.001 * cash

    # 4. Progress Penalties
    overdue_penalty = overdue_count * 10.0
    backlog_penalty = total_active * 2.0

    reward = (
        liquidity_reward
        - 2.0 * late_fees
        - 1.5 * interest
        - util_penalty
        - overdue_penalty
        - backlog_penalty
        + 100.0 * invoices_paid
    )

    # Success Bonus (Survival)
    reward += 10.0

    return round(reward, 2)
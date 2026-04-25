"""
Reward function for the Simulation Logic module.

Calibrated for small numbers (₹1K – ₹50K range).

Rewards:
  + Paying invoices on time
  + Maintaining healthy cash
  + Paying before due date (early payment bonus)

Penalties:
  - Late fees incurred
  - Interest accrued
  - Credit usage
  - Overdue invoices still open
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
    Compute reward for a single day.

    Args:
        cash: current cash balance
        invoices_paid: number of invoices fully paid today
        late_fees: total late fees incurred today
        interest: total interest accrued today
        credit_used: current credit utilization
        credit_limit: maximum credit allowed
        overdue_count: number of overdue invoices at end of day
        total_active: total active invoices at end of day

    Returns:
        float: reward for the day (can be negative)
    """
    reward = 0.0

    # ── Positive: Paying invoices ──
    reward += 50.0 * invoices_paid

    # ── Positive: Having cash (small bonus) ──
    if cash > 0:
        reward += min(cash * 0.01, 100.0)  # cap at 100

    # ── Negative: Late fees ──
    reward -= 5.0 * late_fees  # each ₹1 of late fee costs 5 points

    # ── Negative: Interest ──
    reward -= 3.0 * interest

    # ── Negative: Credit usage ──
    if credit_limit > 0:
        utilization = credit_used / credit_limit
        reward -= (utilization ** 2) * 200.0  # escalates quickly

    # ── Negative: Overdue invoices ──
    reward -= 30.0 * overdue_count

    # ── Survival bonus ──
    reward += 10.0  # small reward for each day alive

    return round(reward, 2)

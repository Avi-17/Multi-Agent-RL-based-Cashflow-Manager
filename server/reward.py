"""
Reward function for CFO agent training.

Components:
  + Liquidity preservation (keeping cash)
  + Bill settlement (paying invoices)
  + Negotiation success (vendor accepts terms)
  + Shock absorption (surviving hidden events)
  - Late fee penalties
  - Interest penalties
  - Credit usage penalties
"""


def compute_reward(cash, late_fee, interest, credit_used, credit_limit, paid, 
                   is_bankrupt, negotiation_success=False):
    """
    High-Stakes Reward for CFO training.
    Focuses on survival, debt reduction, and avoiding predatory fees.
    """
    # 1. CRITICAL: Bankruptcy Penalty
    if is_bankrupt or cash < -100000:
        return -5000.0  # Massive penalty for total failure

    # 2. Credit Utilization Scaling (Exponential)
    # Using 10% is fine, using 90% is a crisis
    utilization = credit_used / (credit_limit + 1.0)
    util_penalty = (utilization ** 2) * 500.0  # Escalates quickly as limit is hit

    # 3. Liquidity vs Debt Balance
    # We reward cash, but penalize it if we have late fees (idleness)
    liquidity_reward = 0.001 * cash if late_fee == 0 else -0.001 * cash

    reward = (
        liquidity_reward                # Reward for healthy cash
        - 5.0 * late_fee                # Increased penalty for late fees
        - 2.5 * interest                # Penalty for interest
        - util_penalty                  # Scaleable credit penalty
        + 30.0 * paid                   # Incentive for settlement
        + 25.0 * (1 if negotiation_success else 0) # Successful negotiation
    )

    # 4. Success Bonus
    # Small daily reward for staying alive
    reward += 10.0

    return round(reward, 4)
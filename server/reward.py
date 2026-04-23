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


def compute_reward(cash, late_fee, interest, credit_used, paid,
                   vendor_trust_change, shock_absorbed, negotiation_success=False):
    """
    Comprehensive reward for CFO training.
    
    Args:
        cash: Current cash balance
        late_fee: Total late fees incurred this step
        interest: Total interest incurred this step
        credit_used: Total credit drawn so far
        paid: Number of invoices fully settled this step
        vendor_trust_change: Net trust change (+ is good)
        shock_absorbed: Whether a cash shock was survived
        negotiation_success: Whether negotiation succeeded
    
    Returns:
        float: reward signal
    """
    # Cap credit penalty to avoid runaway negatives (credit_used is cumulative)
    credit_penalty = min(credit_used * 0.01, 50.0)

    reward = (
        0.002 * cash                    # Reward for liquidity
        - 3.0 * late_fee                # Heavy penalty for late fees
        - 2.0 * interest                # Penalty for interest
        - credit_penalty                # Capped penalty for credit reliance
        + 20.0 * paid                   # High incentive for settlement
        + 50.0 * vendor_trust_change    # Reward for trust improvement
        + 30.0 * (1 if shock_absorbed else 0)      # Surviving shocks
        + 15.0 * (1 if negotiation_success else 0) # Successful negotiation
    )
    return round(reward, 4)
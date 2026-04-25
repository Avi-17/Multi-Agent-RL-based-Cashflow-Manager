"""
Data Generator for the Simulation Logic module.

Generates small, human-readable numbers:
  - Invoices:    ₹1,000 – ₹8,000
  - Receivables: ₹2,000 – ₹10,000
  - Cash:        ₹15,000 – ₹50,000

Also generates "incoming invoices" — future bills that the agent
knows are coming (count only) but can't see the amounts until they activate.

All difficulty levels are SOLVABLE:
  total_liabilities <= (cash + credit + expected_inflows) * 0.8
"""

import random
from typing import Dict, Any, List
from simulation_logic.models import SimInvoice, SimReceivable, IncomingInvoice


# ─────────────────────────────────────────────
# Vendor and Customer names (readable)
# ─────────────────────────────────────────────

VENDOR_NAMES = ["FastLogistics", "GlobalParts", "EnergyCo", "CloudHost", "ProConsulting"]
CUSTOMER_NAMES = ["AlphaCorp", "BetaInc", "GammaTech", "DeltaRetail", "OmegaServices"]


def generate_scenario(difficulty: str = "medium", sim_window: int = 7, seed: int = 42) -> Dict[str, Any]:
    """
    Generate a complete scenario for the simulation.

    Returns a dict with:
      - company: {cash, credit_limit}
      - invoices: list of SimInvoice dicts
      - receivables: list of SimReceivable dicts
      - incoming_invoices: list of IncomingInvoice dicts (hidden from agent)
    """
    random.seed(seed)

    # ── Company profile by difficulty ──
    if difficulty == "easy":
        cash = random.randint(40000, 50000)
        credit_limit = 20000
        num_invoices = random.randint(3, 4)
        num_receivables = random.randint(3, 4)
        num_incoming = random.randint(1, 2)
    elif difficulty == "hard":
        cash = random.randint(12000, 18000)
        credit_limit = 5000
        num_invoices = random.randint(6, 8)
        num_receivables = random.randint(1, 2)
        num_incoming = random.randint(3, 4)
    else:  # medium
        cash = random.randint(25000, 35000)
        credit_limit = 10000
        num_invoices = random.randint(4, 6)
        num_receivables = random.randint(2, 3)
        num_incoming = random.randint(2, 3)

    # ── Generate current invoices ──
    invoices = _generate_invoices(num_invoices, difficulty, sim_window)

    # ── Generate receivables ──
    receivables = _generate_receivables(num_receivables, difficulty, sim_window)

    # ── Generate incoming (hidden) invoices ──
    incoming = _generate_incoming_invoices(num_incoming, difficulty, sim_window)

    # ── Solvability check ──
    # We want: total_liability <= 0.8 * (cash + credit_limit + expected_inflows)
    # Solving for cash: cash >= (total_liability / 0.8) - credit_limit - expected_inflows
    total_liability = sum(inv.amount for inv in invoices) + sum(inc.hidden_amount for inc in incoming)
    expected_inflows = sum(r.amount * r.probability for r in receivables)
    min_cash_needed = (total_liability / 0.8) - credit_limit - expected_inflows

    if cash < min_cash_needed:
        cash = int(min_cash_needed) + 2000  # add ₹2,000 buffer

    return {
        "company": {
            "starting_cash": cash,
            "credit_limit": credit_limit,
        },
        "invoices": [inv.model_dump() for inv in invoices],
        "receivables": [rec.model_dump() for rec in receivables],
        "incoming_invoices": [inc.model_dump() for inc in incoming],
    }


def _generate_invoices(n: int, difficulty: str, sim_window: int) -> List[SimInvoice]:
    """Generate current invoices with small, readable amounts."""
    invoices = []
    for i in range(n):
        vendor = VENDOR_NAMES[i % len(VENDOR_NAMES)]

        if difficulty == "easy":
            amount = random.randint(1, 4) * 1000           # ₹1,000 – ₹4,000
            due_in = random.randint(2, sim_window)          # plenty of time
            late_fee = random.randint(50, 150)              # ₹50 – ₹150/day
            interest = round(random.uniform(0.01, 0.02), 3) # 1–2%
        elif difficulty == "hard":
            amount = random.randint(3, 8) * 1000            # ₹3,000 – ₹8,000
            due_in = random.randint(1, max(2, sim_window // 2))  # tight deadlines
            late_fee = random.randint(200, 500)              # ₹200 – ₹500/day
            interest = round(random.uniform(0.03, 0.05), 3)  # 3–5%
        else:  # medium
            amount = random.randint(2, 6) * 1000            # ₹2,000 – ₹6,000
            due_in = random.randint(1, sim_window - 1)      # moderate deadlines
            late_fee = random.randint(100, 300)              # ₹100 – ₹300/day
            interest = round(random.uniform(0.02, 0.03), 3)  # 2–3%

        invoices.append(SimInvoice(
            id=f"INV-{i+1:03d}",
            vendor=vendor,
            amount=float(amount),
            original_amount=float(amount),
            due_in=due_in,
            late_fee=float(late_fee),
            interest_rate=interest,
            status="unpaid",
        ))

    return invoices


def _generate_receivables(n: int, difficulty: str, sim_window: int) -> List[SimReceivable]:
    """Generate expected customer payments."""
    receivables = []
    for i in range(n):
        customer = CUSTOMER_NAMES[i % len(CUSTOMER_NAMES)]

        if difficulty == "easy":
            amount = random.randint(3, 10) * 1000           # ₹3,000 – ₹10,000
            arrives = random.randint(1, sim_window // 2)     # arrive early
            prob = round(random.uniform(0.85, 0.95), 2)
        elif difficulty == "hard":
            amount = random.randint(1, 5) * 1000            # ₹1,000 – ₹5,000
            arrives = random.randint(sim_window // 2, sim_window)  # arrive late
            prob = round(random.uniform(0.50, 0.70), 2)
        else:  # medium
            amount = random.randint(2, 8) * 1000            # ₹2,000 – ₹8,000
            arrives = random.randint(2, sim_window - 1)
            prob = round(random.uniform(0.70, 0.85), 2)

        receivables.append(SimReceivable(
            id=f"RCV-{i+1:03d}",
            customer=customer,
            amount=float(amount),
            arrives_on_day=arrives,
            probability=prob,
        ))

    return receivables


def _generate_incoming_invoices(n: int, difficulty: str, sim_window: int) -> List[IncomingInvoice]:
    """Generate future invoices that the agent is warned about but can't see details of."""
    incoming = []
    for i in range(n):
        vendor = VENDOR_NAMES[(i + 2) % len(VENDOR_NAMES)]  # offset so different vendors
        appears_day = random.randint(2, sim_window - 1)

        if difficulty == "easy":
            amount = random.randint(1, 3) * 1000
            due_after = random.randint(3, 5)     # generous deadline after appearing
            late_fee = random.randint(50, 100)
            interest = round(random.uniform(0.01, 0.02), 3)
        elif difficulty == "hard":
            amount = random.randint(3, 7) * 1000
            due_after = random.randint(1, 2)     # tight deadline after appearing
            late_fee = random.randint(200, 400)
            interest = round(random.uniform(0.03, 0.05), 3)
        else:  # medium
            amount = random.randint(2, 5) * 1000
            due_after = random.randint(2, 4)
            late_fee = random.randint(100, 250)
            interest = round(random.uniform(0.02, 0.03), 3)

        incoming.append(IncomingInvoice(
            id=f"FUT-{i+1:03d}",
            vendor=vendor,
            appears_on_day=appears_day,
            hidden_amount=float(amount),
            hidden_due_in=due_after,
            hidden_late_fee=float(late_fee),
            hidden_interest_rate=interest,
        ))

    return incoming


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json
    for diff in ["easy", "medium", "hard"]:
        scenario = generate_scenario(difficulty=diff, sim_window=7, seed=42)
        total_inv = sum(i["amount"] for i in scenario["invoices"])
        total_inc = sum(i["hidden_amount"] for i in scenario["incoming_invoices"])
        total_rec = sum(r["amount"] * r["probability"] for r in scenario["receivables"])
        cash = scenario["company"]["starting_cash"]
        credit = scenario["company"]["credit_limit"]
        print(f"\n{'='*50}")
        print(f"DIFFICULTY: {diff.upper()}")
        print(f"  Cash: ₹{cash:,} | Credit: ₹{credit:,}")
        print(f"  Invoices: {len(scenario['invoices'])} totalling ₹{total_inv:,.0f}")
        print(f"  Incoming: {len(scenario['incoming_invoices'])} totalling ₹{total_inc:,.0f}")
        print(f"  Receivables: {len(scenario['receivables'])} totalling ₹{total_rec:,.0f} (weighted)")
        print(f"  Total resources: ₹{cash + credit + total_rec:,.0f}")
        print(f"  Total liability: ₹{total_inv + total_inc:,.0f}")
        solvable = (total_inv + total_inc) <= (cash + credit + total_rec) * 0.8
        print(f"  Solvable: {'✅ YES' if solvable else '❌ NO (auto-adjusted)'}")

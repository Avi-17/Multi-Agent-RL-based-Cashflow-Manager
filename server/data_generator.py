import random
from typing import Dict, Any, List
from uuid import uuid4

try:
    from models import Invoice, Receivable, IncomingInvoice, VendorProfile
except ImportError:
    from .models import Invoice, Receivable, IncomingInvoice, VendorProfile


VENDOR_NAMES = ["FastLogistics", "GlobalParts", "EnergyCo", "CloudHost", "ProConsulting"]
CUSTOMER_NAMES = ["AlphaCorp", "BetaInc", "GammaTech", "DeltaRetail", "OmegaServices"]


def generate_vendors(difficulty: str = "medium") -> List[Dict[str, Any]]:
    vendors = []
    for i, name in enumerate(VENDOR_NAMES):
        base_trust = 0.7 if difficulty == "easy" else 0.5 if difficulty == "medium" else 0.3
        v = VendorProfile(
            id=f"v-{i}",
            name=name,
            trust_score=random.uniform(base_trust, base_trust + 0.2),
            negotiation_flexibility=random.uniform(0.2, 0.6)
        )
        vendors.append(v.model_dump())
    return vendors


def generate_scenario(difficulty: str = "medium", sim_window: int = 3, seed: int = 42) -> Dict[str, Any]:
    """
    Generate a complete scenario for the simulation.
    """
    random.seed(seed)

    vendors = generate_vendors(difficulty)

    if difficulty == "easy":
        cash = random.randint(40000, 50000)
        credit_limit = 20000
        num_invoices = random.randint(3, 4)
        num_receivables = random.randint(3, 4)
        num_incoming = random.randint(1, 2)
    elif difficulty == "hard":
        cash = random.randint(2000, 5000)
        credit_limit = 2000
        num_invoices = random.randint(6, 8)
        num_receivables = random.randint(2, 3) # Need receivables so there's future hope
        num_incoming = random.randint(3, 4)
    else:  # medium
        cash = random.randint(25000, 35000)
        credit_limit = 10000
        num_invoices = random.randint(4, 6)
        num_receivables = random.randint(2, 3)
        num_incoming = random.randint(2, 3)

    invoices = _generate_invoices(num_invoices, difficulty, sim_window, vendors)
    receivables = _generate_receivables(num_receivables, difficulty, sim_window)
    incoming = _generate_incoming_invoices(num_incoming, difficulty, sim_window, vendors)

    # Solvability check
    # We guarantee the scenario is mathematically solvable over the window, 
    # but upfront cash should be limited to force negotiation/deferral.
    total_liability = sum(inv.amount for inv in invoices) + sum(inc.hidden_amount for inc in incoming)
    expected_inflows = sum(r.amount * r.probability for r in receivables)
    
    if difficulty == "hard":
        # Exactly enough resources to pay debt, forcing perfect timing and negotiation
        min_cash_needed = (total_liability * 1.05) - credit_limit - expected_inflows
        buffer = 500
    elif difficulty == "medium":
        min_cash_needed = (total_liability * 1.2) - credit_limit - expected_inflows
        buffer = 2000
    else:
        min_cash_needed = (total_liability / 0.8) - credit_limit - expected_inflows
        buffer = 5000

    if cash < min_cash_needed:
        cash = int(min_cash_needed) + buffer
    elif difficulty == "hard" and cash > min_cash_needed + 2000:
        # Cap the cash in hard mode so they don't randomly get a wealthy seed
        cash = int(min_cash_needed) + 1000
        if cash < 0: cash = 1000

    return {
        "company": {
            "name": f"Enterprise-{random.randint(100, 999)}",
            "starting_cash": cash,
            "credit_limit": credit_limit,
            "revenue_target": cash * 2
        },
        "vendors": vendors,
        "initial_invoices": [inv.model_dump() for inv in invoices],
        "initial_receivables": [rec.model_dump() for rec in receivables],
        "incoming_invoices": [inc.model_dump() for inc in incoming],
    }


def _generate_invoices(n: int, difficulty: str, sim_window: int, vendors: List[Dict]) -> List[Invoice]:
    invoices = []
    for i in range(n):
        vendor = random.choice(vendors)

        if difficulty == "easy":
            amount = random.randint(1, 4) * 1000
            due_in = random.randint(2, sim_window)
            late_fee = random.randint(50, 150)
            interest = round(random.uniform(0.01, 0.02), 3)
        elif difficulty == "hard":
            amount = random.randint(5, 12) * 1000
            due_in = random.randint(1, 2)  # At least 1 day to react
            late_fee = random.randint(300, 800)
            interest = round(random.uniform(0.04, 0.10), 3)
        else:  # medium
            amount = random.randint(2, 6) * 1000
            due_in = random.randint(1, sim_window - 1)
            late_fee = random.randint(100, 300)
            interest = round(random.uniform(0.02, 0.03), 3)

        invoices.append(Invoice(
            id=f"INV-{i+1:03d}",
            vendor_id=vendor["id"],
            amount=float(amount),
            due_in=due_in,
            late_fee=float(late_fee),
            min_payment=float(amount) * 0.3,
            interest=interest,
            status="unpaid",
        ))
    return invoices


def _generate_receivables(n: int, difficulty: str, sim_window: int) -> List[Receivable]:
    receivables = []
    for i in range(n):
        customer = CUSTOMER_NAMES[i % len(CUSTOMER_NAMES)]

        if difficulty == "easy":
            amount = random.randint(3, 10) * 1000
            arrives = random.randint(1, sim_window // 2)
            prob = round(random.uniform(0.85, 0.95), 2)
        elif difficulty == "hard":
            amount = random.randint(5, 10) * 1000
            arrives = random.randint(sim_window // 2, sim_window) # Arrives late, forcing short-term survival
            prob = round(random.uniform(0.60, 0.80), 2)
        else:  # medium
            amount = random.randint(2, 8) * 1000
            arrives = random.randint(2, sim_window - 1)
            prob = round(random.uniform(0.70, 0.85), 2)

        receivables.append(Receivable(
            id=f"RCV-{i+1:03d}",
            customer_id=customer,
            amount=float(amount),
            expected_in=arrives,
            probability=prob,
        ))
    return receivables


def _generate_incoming_invoices(n: int, difficulty: str, sim_window: int, vendors: List[Dict]) -> List[IncomingInvoice]:
    incoming = []
    for i in range(n):
        vendor = random.choice(vendors)
        appears_day = random.randint(2, sim_window - 1) if sim_window > 2 else 2

        if difficulty == "easy":
            amount = random.randint(1, 3) * 1000
            due_after = random.randint(3, 5)
            late_fee = random.randint(50, 100)
            interest = round(random.uniform(0.01, 0.02), 3)
        elif difficulty == "hard":
            amount = random.randint(3, 7) * 1000
            due_after = random.randint(1, 2)
            late_fee = random.randint(200, 400)
            interest = round(random.uniform(0.03, 0.05), 3)
        else:
            amount = random.randint(2, 5) * 1000
            due_after = random.randint(2, 4)
            late_fee = random.randint(100, 250)
            interest = round(random.uniform(0.02, 0.03), 3)

        incoming.append(IncomingInvoice(
            id=f"FUT-{i+1:03d}",
            vendor_id=vendor["id"],
            appears_on_day=appears_day,
            hidden_amount=float(amount),
            hidden_due_in=due_after,
            hidden_late_fee=float(late_fee),
            hidden_interest=interest,
        ))
    return incoming


if __name__ == "__main__":
    import json
    scenario = generate_scenario()
    print(json.dumps(scenario, indent=2))

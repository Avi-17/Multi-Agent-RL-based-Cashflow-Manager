import json
import random
from typing import List, Dict, Any
from uuid import uuid4

try:
    from models import Invoice, Receivable, VendorProfile
except ImportError:
    try:
        from cashflowmanager.models import Invoice, Receivable, VendorProfile
    except ImportError:
        from ..models import Invoice, Receivable, VendorProfile

def generate_company_profile(difficulty="medium"):
    starting_cash = 1000000
    if difficulty == "easy": starting_cash = 1500000
    elif difficulty == "hard": starting_cash = 500000
    
    return {
        "name": f"Enterprise-{random.randint(100, 999)}",
        "starting_cash": starting_cash,
        "credit_limit": 500000,
        "revenue_target": 2000000
    }

def generate_vendors(n=5, difficulty="medium"):
    vendors = []
    names = ["FastLogistics", "GlobalParts", "EnergyCo", "SecureCloud", "ProConsulting"]
    for i in range(min(n, len(names))):
        base_trust = 0.7 if difficulty == "easy" else 0.5 if difficulty == "medium" else 0.3
        vendors.append({
            "id": f"v-{i}",
            "name": names[i],
            "trust_score": random.uniform(base_trust, base_trust + 0.2),
            "negotiation_flexibility": random.uniform(0.2, 0.6)
        })
    return vendors

def generate_invoices(vendors, difficulty="medium"):
    # Easy: 5 invoices, Medium: 10, Hard: 15
    n = 5 if difficulty == "easy" else 10 if difficulty == "medium" else 15
    invoices = []
    # Hard mode has higher interest, late fees, and AMOUNTS
    int_mult = 0.5 if difficulty == "easy" else 1.0 if difficulty == "medium" else 2.0
    amt_mult = 0.7 if difficulty == "easy" else 1.0 if difficulty == "medium" else 1.5
    
    for _ in range(n):
        vendor = random.choice(vendors)
        invoices.append({
            "id": str(uuid4())[:8],
            "vendor_id": vendor["id"],
            "amount": random.uniform(50000, 300000) * amt_mult,
            "due_in": random.randint(1, 10),
            "late_fee": random.uniform(5000, 20000) * int_mult,
            "min_payment": random.uniform(10000, 50000),
            "interest": random.uniform(0.01, 0.05) * int_mult,
            "status": "unpaid"
        })
    return invoices

def generate_receivables(difficulty="medium"):
    # Easy: 7 receivables, Medium: 5, Hard: 3
    n = 7 if difficulty == "easy" else 5 if difficulty == "medium" else 3
    receivables = []
    # Hard mode has lower payment probability
    prob_base = 0.85 if difficulty == "easy" else 0.75 if difficulty == "medium" else 0.55
    for i in range(n):
        receivables.append({
            "id": f"r-{i}",
            "customer_id": f"c-{i}",
            "amount": random.uniform(100000, 500000),
            "expected_in": random.randint(2, 12),
            "probability": random.uniform(prob_base, prob_base + 0.1)
        })
    return receivables

def generate_scenario(difficulty="medium"):
    vendors = generate_vendors(difficulty=difficulty)
    return {
        "company": generate_company_profile(difficulty=difficulty),
        "vendors": vendors,
        "initial_invoices": generate_invoices(vendors, difficulty=difficulty),
        "initial_receivables": generate_receivables(difficulty=difficulty),
        "hidden_events": [
            {"day": random.randint(3, 7), "type": "cash_shock", "amount": -random.uniform(100000, 300000), "desc": "Equipment Failure"},
            {"day": random.randint(5, 10), "type": "payment_delay", "target_id": "r-0", "delay": 3}
        ]
    }

if __name__ == "__main__":
    scenario = generate_scenario()
    print(json.dumps(scenario, indent=2))

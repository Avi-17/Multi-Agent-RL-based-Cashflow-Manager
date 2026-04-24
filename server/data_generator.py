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

def generate_company_profile():
    sectors = ["Tech", "Manufacturing", "Retail", "Services"]
    return {
        "name": f"Enterprise-{random.randint(100, 999)}",
        "sector": random.choice(sectors),
        "starting_cash": random.uniform(500000, 1500000),
        "credit_limit": 500000,
        "revenue_target": 2000000
    }

def generate_vendors(n: int = 5) -> List[Dict[str, Any]]:
    vendors = []
    names = ["FastLogistics", "GlobalParts", "EnergyCo", "SecureCloud", "ProConsulting"]
    for i in range(min(n, len(names))):
        v = VendorProfile(
            id=f"v-{i}",
            name=names[i],
            trust_score=random.uniform(0.6, 0.9),
            negotiation_flexibility=random.uniform(0.3, 0.7),
            late_fee_waiver_prob=random.uniform(0.1, 0.4)
        )
        vendors.append(v.model_dump())
    return vendors

def generate_invoices(vendors: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    invoices = []
    for _ in range(n):
        vendor = random.choice(vendors)
        inv = Invoice(
            id=str(uuid4())[:8],
            vendor_id=vendor["id"],
            amount=random.uniform(50000, 300000),
            due_in=random.randint(1, 10),
            late_fee=random.uniform(5000, 20000),
            min_payment=random.uniform(10000, 50000),
            interest=random.uniform(0.01, 0.05),
            status="unpaid"
        )
        invoices.append(inv.model_dump())
    return invoices

def generate_receivables(n: int = 5) -> List[Dict[str, Any]]:
    receivables = []
    for i in range(n):
        rec = Receivable(
            id=f"r-{i}",
            customer_id=f"c-{i}",
            amount=random.uniform(100000, 500000),
            expected_in=random.randint(2, 12),
            probability=random.uniform(0.7, 0.95)
        )
        receivables.append(rec.model_dump())
    return receivables

def generate_scenario():
    vendors = generate_vendors()
    return {
        "company": generate_company_profile(),
        "vendors": vendors,
        "initial_invoices": generate_invoices(vendors),
        "initial_receivables": generate_receivables(),
        "hidden_events": [
            {"day": random.randint(3, 7), "type": "cash_shock", "amount": -random.uniform(100000, 300000), "desc": "Equipment Failure"},
            {"day": random.randint(5, 10), "type": "payment_delay", "target_id": "r-0", "delay": 3}
        ]
    }

if __name__ == "__main__":
    scenario = generate_scenario()
    print(json.dumps(scenario, indent=2))

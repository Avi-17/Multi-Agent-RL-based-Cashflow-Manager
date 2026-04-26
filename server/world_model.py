"""
World Model — Hidden dynamics engine for the Cashflow Multi-Agent Environment.

Manages probabilistic events that are NOT directly visible to agents:
- Future inflow uncertainty (customer payment delays)
- Cash shocks (equipment failure, tax audit, fraud)
- Vendor behavior shifts (trust decay/growth)
- Market conditions (interest rate changes)

The world model is updated AFTER every environment step (Step 8 in workflow).
"""

import random
from typing import List, Dict, Any, Optional
from uuid import uuid4


class WorldEvent:
    """A single hidden event that may trigger on a specific day."""
    def __init__(self, day: int, event_type: str, severity: float,
                 description: str, target_id: Optional[str] = None,
                 amount: float = 0.0, probability: float = 1.0):
        self.day = day
        self.event_type = event_type  # cash_shock, payment_delay, vendor_shift, revenue_miss, fraud
        self.severity = severity      # 0.0 to 1.0
        self.description = description
        self.target_id = target_id    # affected invoice/receivable/vendor ID
        self.amount = amount
        self.probability = probability
        self.triggered = False


class WorldModel:
    """
    Hidden state tracker that evolves probabilistically each day.
    
    The environment calls:
      - initialize() on reset
      - update(day) after each step to check for triggered events
    
    Agents get PARTIAL views — they don't see the full event list.
    """

    def __init__(self):
        self.events: List[WorldEvent] = []
        self.triggered_log: List[Dict[str, Any]] = []
        self.market_stress: float = 0.0    # 0.0 = calm, 1.0 = crisis
        self.vendor_mood: Dict[str, float] = {}  # vendor_id -> mood modifier
        self.day_effects: Dict[str, Any] = {}    # per-day cache of effects

    def initialize(self, scenario: Dict[str, Any], sim_window: int = 3):
        """Generate hidden events from scenario data.

        Amounts are scaled to the scenario's starting cash so shocks hurt the
        score without instantly bankrupting short-window simulations.
        """
        self.events = []
        self.triggered_log = []
        self.market_stress = random.uniform(0.0, 0.3)
        self.vendor_mood = {}
        self.day_effects = {}

        starting_cash = float(scenario.get("company", {}).get("starting_cash", 10000))

        vendors = scenario.get("vendors", [])
        for v in vendors:
            self.vendor_mood[v["id"]] = 0.0

        # --- Cash shocks (scaled to starting cash, count scaled to window) ---
        # Survivable but score-impacting: each shock ~8-25% of starting cash.
        num_shocks = max(1, sim_window // 2)
        shock_types = [
            ("Equipment failure",    0.15, 0.25),
            ("Tax audit penalty",    0.12, 0.20),
            ("Emergency repair",     0.08, 0.15),
            ("Regulatory fine",      0.10, 0.18),
            ("Supplier price hike",  0.05, 0.12),
        ]
        for _ in range(num_shocks):
            description, lo_pct, hi_pct = random.choice(shock_types)
            self.events.append(WorldEvent(
                day=random.randint(1, sim_window),
                event_type="cash_shock",
                severity=random.uniform(0.6, 1.0),
                description=description,
                amount=-starting_cash * random.uniform(lo_pct, hi_pct),
                probability=random.uniform(0.5, 0.8),
            ))

        # --- Payment delays (lower hit rate; receivables already roll probability) ---
        receivables = scenario.get("initial_receivables", [])
        for rec in receivables:
            if random.random() < 0.5:
                self.events.append(WorldEvent(
                    day=random.randint(1, sim_window),
                    event_type="payment_delay",
                    severity=random.uniform(0.5, 0.9),
                    description=f"Customer {rec['customer_id']} payment delayed",
                    target_id=rec["id"],
                    amount=random.randint(1, 3),  # delay in days
                    probability=random.uniform(0.6, 0.85),
                ))

        # --- Revenue miss (rare, raises market stress) ---
        if random.random() < 0.4:
            self.events.append(WorldEvent(
                day=random.randint(1, sim_window),
                event_type="revenue_miss",
                severity=random.uniform(0.6, 0.9),
                description="Quarterly revenue target missed — board review triggered",
                probability=random.uniform(0.6, 0.85),
            ))

        # vendor_shift events intentionally omitted: no consumer for vendor
        # trust deltas in current state. Re-enable when vendor_profiles is
        # plumbed into State.

        # --- Fraud (rare, scaled to cash) ---
        if random.random() < 0.2:
            self.events.append(WorldEvent(
                day=random.randint(1, sim_window),
                event_type="fraud",
                severity=0.9,
                description="Suspicious transaction detected — investigation required",
                amount=-starting_cash * random.uniform(0.10, 0.20),
                probability=0.5,
            ))

    def update(self, day: int) -> Dict[str, Any]:
        """
        Check and trigger events for the given day.
        Returns a dict of effects to apply to the environment.
        """
        effects = {
            "cash_delta": 0.0,
            "payment_delays": [],       # list of (receivable_id, extra_days)
            "vendor_trust_deltas": {},   # vendor_id -> trust_delta
            "shock_occurred": False,
            "shock_description": None,
            "fraud_alert": False,
            "revenue_miss": False,
            "events_triggered": [],
        }

        for event in self.events:
            if event.day == day and not event.triggered:
                # Roll the dice
                if random.random() < event.probability:
                    event.triggered = True

                    if event.event_type == "cash_shock":
                        effects["cash_delta"] += event.amount
                        effects["shock_occurred"] = True
                        effects["shock_description"] = event.description

                    elif event.event_type == "payment_delay":
                        effects["payment_delays"].append(
                            (event.target_id, int(event.amount))
                        )

                    elif event.event_type == "vendor_shift":
                        vid = event.target_id
                        effects["vendor_trust_deltas"][vid] = event.amount
                        self.vendor_mood[vid] = self.vendor_mood.get(vid, 0) + event.amount

                    elif event.event_type == "revenue_miss":
                        effects["revenue_miss"] = True
                        self.market_stress = min(1.0, self.market_stress + 0.2)

                    elif event.event_type == "fraud":
                        effects["fraud_alert"] = True
                        effects["cash_delta"] += event.amount

                    effects["events_triggered"].append({
                        "type": event.event_type,
                        "description": event.description,
                        "severity": event.severity,
                    })

                    self.triggered_log.append({
                        "day": day,
                        "type": event.event_type,
                        "description": event.description,
                        "amount": event.amount,
                    })

        # Market stress naturally decays
        self.market_stress = max(0.0, self.market_stress - 0.02)

        self.day_effects[day] = effects
        return effects

    def get_risk_hints(self, day: int) -> Dict[str, Any]:
        """
        Partial information for the Risk Agent.
        Reveals SOME upcoming threats but not exact amounts/days.
        """
        hints = {
            "market_stress": round(self.market_stress, 2),
            "upcoming_risk_level": "low",
            "vendor_sentiment": {},
        }

        # Give a vague warning about upcoming shocks (within 2 days)
        upcoming_threats = 0
        for event in self.events:
            if not event.triggered and abs(event.day - day) <= 2:
                if event.event_type in ("cash_shock", "fraud", "payment_delay", "revenue_miss"):
                    upcoming_threats += 1

        if upcoming_threats >= 2:
            hints["upcoming_risk_level"] = "critical"
        elif upcoming_threats == 1:
            hints["upcoming_risk_level"] = "elevated"

        # Vendor sentiment (partial view)
        for vid, mood in self.vendor_mood.items():
            if mood < -0.1:
                hints["vendor_sentiment"][vid] = "negative"
            elif mood > 0.05:
                hints["vendor_sentiment"][vid] = "positive"
            else:
                hints["vendor_sentiment"][vid] = "neutral"

        return hints

    def get_triggered_events(self) -> List[Dict]:
        """Full log of all triggered events (for grading/logging)."""
        return self.triggered_log

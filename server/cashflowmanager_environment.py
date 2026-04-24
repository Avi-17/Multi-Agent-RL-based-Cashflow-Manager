"""
Cashflow Multi-Agent RL Environment.

Follows the strict 11-step workflow:
  1. RESET → generate cash, invoices, revenue
  2. WORLD MODEL → initialize hidden dynamics
  3. AGENTS OBSERVE → partial views
  4. MULTI-AGENT INTERACTION → advisors produce memos
  5. CFO AGENT → resolves conflict, chooses action
  6. IF NEGOTIATE → vendor agent responds
  7. ENVIRONMENT STEP → update cash, invoices, credit
  8. WORLD MODEL UPDATE → trigger probabilistic events
  9. REWARD COMPUTATION → penalties, interest, liquidity, negotiation
  10. STORE TRANSITION → (state, action, reward, reasoning)
  11. LOOP → until episode ends

OpenEnv Interface: reset(), step(), state
"""

import random
from uuid import uuid4
from typing import List, Dict, Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import (
        CashflowmanagerAction, CashflowmanagerObservation,
        Invoice, Receivable, VendorProfile, NegotiationResult, Transition
    )
    from server.world_model import WorldModel
    from server.data_generator import generate_scenario
    from server.agents import (
        expenditure_agent, revenue_agent, risk_agent, risk_agent_icl, vendor_agent, format_memo
    )
    from server.reward import compute_reward
except ImportError:
    try:
        from cashflowmanager.models import (
            CashflowmanagerAction, CashflowmanagerObservation,
            Invoice, Receivable, VendorProfile, NegotiationResult, Transition
        )
        from cashflowmanager.server.world_model import WorldModel
        from cashflowmanager.server.data_generator import generate_scenario
        from cashflowmanager.server.agents import (
            expenditure_agent, revenue_agent, risk_agent, risk_agent_icl, vendor_agent, format_memo
        )
        from cashflowmanager.server.reward import compute_reward
    except ImportError:
        from ..models import (
            CashflowmanagerAction, CashflowmanagerObservation,
            Invoice, Receivable, VendorProfile, NegotiationResult, Transition
        )
        from .world_model import WorldModel
        from .data_generator import generate_scenario
        from .agents import (
            expenditure_agent, revenue_agent, risk_agent, risk_agent_icl, vendor_agent, format_memo
        )
        from .reward import compute_reward


MAX_DAYS = 10


class CashflowmanagerEnvironment(Environment):
    """
    Multi-Agent Cashflow RL Environment with Hidden World Dynamics.

    Themes:
      - #1 Multi-Agent Interactions: CFO + Expenditure + Revenue + Risk + Vendor
      - #3.1 World Modeling: Hidden events, partial observability

    OpenEnv API:
      - reset() → CashflowmanagerObservation
      - step(action) → CashflowmanagerObservation
      - state → State
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty="medium", seed=None):
        self.difficulty = difficulty
        self.seed = seed
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.world_model = WorldModel()
        self.transitions: List[Transition] = []
        self.done = False

        # These get populated on reset()
        self.cash = 0.0
        self.credit_used = 0.0
        self.credit_limit = 5000.0
        self.invoices: List[Invoice] = []
        self.receivables: List[Receivable] = []
        self.vendors: Dict[str, VendorProfile] = {}
        self.day = 1
        self.last_negotiation: Optional[NegotiationResult] = None
        self.last_world_events: List[str] = []

    # ═══════════════════════════════════════════════════
    # STEP 1: RESET — Generate cash, invoices, revenue
    # ═══════════════════════════════════════════════════
    def reset(self, difficulty="medium", seed=None) -> CashflowmanagerObservation:
        if difficulty:
            self.difficulty = difficulty
        print(f"--- RESETTING ENVIRONMENT | DIFFICULTY: {self.difficulty} | SEED: {seed} ---")
        if seed is not None:
            random.seed(seed)

        # Generate synthetic scenario based on difficulty
        scenario = generate_scenario(difficulty=self.difficulty)

        self.day = 1
        self.cash = scenario["company"]["starting_cash"]
        self.credit_used = 0.0
        self.credit_limit = scenario["company"]["credit_limit"]
        self.done = False

        self.vendors = {v["id"]: VendorProfile(**v) for v in scenario["vendors"]}
        self.invoices = [Invoice(**i) for i in scenario["initial_invoices"]]
        self.receivables = [Receivable(**r) for r in scenario["initial_receivables"]]

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.transitions = []
        self.last_negotiation = None
        self.last_world_events = []

        # STEP 2: WORLD MODEL — initialize hidden dynamics
        self.world_model.initialize(scenario, max_days=MAX_DAYS)

        # STEPS 3-4: Agents observe and produce initial memos
        advisor_memos, advisor_messages = self._run_advisors()

        return self._build_obs(
            reward=0.0,
            advisor_memos=advisor_memos,
            advisor_messages=advisor_messages,
        )

    # ═══════════════════════════════════════════════════
    # STEPS 5-10: STEP — Process action, update world, compute reward
    # ═══════════════════════════════════════════════════
    def step(self, action: CashflowmanagerAction) -> CashflowmanagerObservation:
        self._state.step_count += 1

        # Capture pre-state for transition logging
        pre_state = f"Day:{self.day} Cash:{self.cash:.0f} Credit:{self.credit_used:.0f} Invoices:{len(self.invoices)}"

        paid = 0
        late_fee_total = 0.0
        interest_total = 0.0
        trust_change = 0.0
        negotiation_success = False
        shock_absorbed = False
        self.last_negotiation = None
        self.last_world_events = []

        # ── STEP 5: CFO action is received ──
        # ── STEP 6: If NEGOTIATE → vendor responds ──
        # ── STEP 7: Update cash, invoices, credit ──

        if action.type == "pay" and action.invoice_id:
            inv = self._find_invoice(action.invoice_id)
            if inv and inv.status != "paid":
                pay_amount = action.amount if action.amount and action.amount > 0 else inv.amount
                pay_amount = min(pay_amount, inv.amount)
                available = self.cash + (self.credit_limit - self.credit_used)
                pay_amount = min(pay_amount, available)

                if pay_amount >= inv.amount:
                    inv.status = "paid"
                    inv.amount = 0
                    paid += 1
                    trust_change += 0.05
                else:
                    inv.amount -= pay_amount
                    inv.status = "partial"

                self._update_cash(-pay_amount)

        elif action.type == "partial" and action.invoice_id:
            inv = self._find_invoice(action.invoice_id)
            if inv and inv.status != "paid":
                pay_amount = action.amount if action.amount and action.amount > 0 else inv.min_payment
                pay_amount = min(pay_amount, inv.amount)
                available = self.cash + (self.credit_limit - self.credit_used)
                pay_amount = min(pay_amount, available)

                inv.amount -= pay_amount
                if inv.amount <= 0:
                    inv.amount = 0
                    inv.status = "paid"
                    paid += 1
                else:
                    inv.status = "partial"
                self._update_cash(-pay_amount)

        elif action.type == "negotiate" and action.invoice_id:
            inv = self._find_invoice(action.invoice_id)
            if inv and inv.status != "paid":
                vendor = self.vendors.get(inv.vendor_id)
                if vendor:
                    # STEP 6: Vendor Agent responds with mood awareness
                    mood = self.world_model.vendor_mood.get(inv.vendor_id, 0.0)
                    neg_result = vendor_agent(
                        {"id": vendor.id, "name": vendor.name}, 
                        inv, 
                        vendor_mood=mood,
                        trust_score=vendor.trust_score
                    )
                    
                    self.last_negotiation = {
                        "accepted": neg_result["accepted"],
                        "message": neg_result["vendor_message"]
                    }

                    if neg_result["accepted"]:
                        inv.due_in += neg_result["extension_days"]
                        negotiation_success = True
                        trust_change += 0.02
                    else:
                        trust_change -= 0.05  # Trust drops on rejection
                    
                    # Execute Payment (Good faith payment or forced payment)
                    pay_amount = action.amount if action.amount and action.amount > 0 else 0.0
                    available = self.cash + (self.credit_limit - self.credit_used)
                    pay_amount = min(pay_amount, inv.amount, available)
                    
                    if pay_amount > 0:
                        inv.amount -= pay_amount
                        self._update_cash(-pay_amount)
                        if inv.amount <= 0:
                            inv.amount = 0
                            inv.status = "paid"
                            paid += 1
                        else:
                            inv.status = "partial"

        elif action.type == "credit":
            draw_amount = action.amount if action.amount and action.amount > 0 else 100000.0
            remaining_credit = self.credit_limit - self.credit_used
            draw_amount = min(draw_amount, remaining_credit)
            self.cash += draw_amount
            self.credit_used += draw_amount

        elif action.type == "defer":
            pass  # Explicit skip — no action taken

        # Update vendor trust scores
        for vid, delta in [(v, trust_change) for v in self.vendors]:
            vendor = self.vendors[vid]
            vendor.trust_score = max(0.0, min(1.0, vendor.trust_score + trust_change / len(self.vendors)))

        # ── STEP 8: WORLD MODEL UPDATE ──
        day_ended = self._check_day_end()
        if day_ended:
            # World model triggers hidden events for the day that just ended
            effects = self.world_model.update(self.day)

            # Apply cash shocks
            if effects["cash_delta"] != 0:
                self._update_cash(effects["cash_delta"])
                if effects["shock_occurred"] and self.cash >= 0:
                    shock_absorbed = True
            
            # Log all triggered world events
            for ev in effects["events_triggered"]:
                if ev["type"] in ["cash_shock", "fraud"]:
                    self.last_world_events.append(f"💥 {ev['description']}")
                elif ev["type"] == "revenue_miss":
                    self.last_world_events.append(f"📉 {ev['description']}")
                elif ev["type"] == "vendor_shift":
                    self.last_world_events.append(f"🤝 {ev['description']}")

            # Age all unpaid invoices
            for inv in self.invoices:
                if inv.status != "paid":
                    inv.due_in -= 1
                    if inv.due_in < 0 and inv.amount > 0:
                        late_fee_total += inv.late_fee
                        inv.amount += inv.late_fee
                    if inv.amount > 0:
                        interest = inv.amount * inv.interest
                        interest_total += interest
                        inv.amount += interest

            self.day += 1

            # Apply payment delays (Revenue Delay -> Expected Revenue = 0)
            for rec_id, extra_days in effects["payment_delays"]:
                rec = next((r for r in self.receivables if r.id == rec_id), None)
                if rec:
                    rec.amount = 0
                    rec.probability = 0
                    self.last_world_events.append(f"⚠️ Revenue Delay: Payment from {rec.customer_id} failed (Expected Revenue = 0)")
                    self.receivables.remove(rec)

            # Apply vendor trust changes
            for vid, delta in effects["vendor_trust_deltas"].items():
                if vid in self.vendors:
                    self.vendors[vid].trust_score = max(0.0, min(1.0, self.vendors[vid].trust_score + delta))

            # Process receivables
            for rec in list(self.receivables):
                rec.expected_in -= 1
                if rec.expected_in <= 0:
                    if random.random() < rec.probability:
                        self.cash += rec.amount
                        self.last_world_events.append(f"💰 Received ${rec.amount:.0f} from {rec.customer_id}")
                    else:
                        self.last_world_events.append(f"❌ Payment from {rec.customer_id} defaulted")
                    self.receivables.remove(rec)

            # Check for fraud
            if effects["fraud_alert"]:
                self.last_world_events.append("🚨 Fraud alert — investigation triggered")

            if effects["revenue_miss"]:
                self.last_world_events.append("📉 Revenue target missed — board review")

        # Episode termination
        if self.day > MAX_DAYS:
            self.done = True

        # Check for bankruptcy before reward
        if self.cash < -500000: # Threshold for total collapse
            self.done = True

        # ── STEP 9: REWARD COMPUTATION ──
        reward = compute_reward(
            cash=self.cash,
            late_fee=late_fee_total,
            interest=interest_total,
            credit_used=self.credit_used,
            credit_limit=self.credit_limit,
            paid=paid,
            is_bankrupt=self.done,
            negotiation_success=negotiation_success
        )

        # ── STEPS 3-4: Advisors observe new state and produce memos ──
        advisor_memos, advisor_messages = self._run_advisors()

        # ── STEP 10: STORE TRANSITION ──
        post_state = f"Day:{self.day} Cash:{self.cash:.0f} Credit:{self.credit_used:.0f} Invoices:{len([i for i in self.invoices if i.status != 'paid'])}"
        self.transitions.append(Transition(
            day=self.day,
            state_summary=pre_state,
            advisor_memos=advisor_memos,
            action={"type": action.type, "invoice_id": action.invoice_id, "amount": action.amount},
            reward=reward,
            reasoning=action.memo or "",
            next_state_summary=post_state,
        ))

        # ── STEP 11: Build observation and LOOP ──
        return self._build_obs(
            reward=reward,
            advisor_memos=advisor_memos,
            advisor_messages=advisor_messages,
            late_fee=late_fee_total,
            interest=interest_total,
        )

    # ═══════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════

    def _run_advisors(self):
        """Steps 3-4: Each agent observes partial state and produces a memo."""
        world_hints = self.world_model.get_risk_hints(self.day)

        # Run Revenue Agent first to get projection (now aware of market stress)
        rev_memo = revenue_agent(
            self.receivables, 
            self.invoices, 
            self.cash, 
            self.day, 
            market_stress=world_hints.get("market_stress", 0.0)
        )
        
        # Pass net 3-day position to Expenditure Agent for smarter prioritisation
        exp_memo = expenditure_agent(
            [inv for inv in self.invoices if inv.status != "paid"],
            self.cash,
            revenue_projection=rev_memo.get("net_3day_position", self.cash)
        )
        
        risk_memo = risk_agent(
            self.cash, 
            exp_memo.get("total_outstanding", 0.0),
            self.credit_used, 
            self.credit_limit, 
            world_hints
        )

        advisor_memos = {
            "Expenditure": exp_memo,
            "Revenue": rev_memo,
            "Risk": risk_memo,
        }
        advisor_messages = {
            "Expenditure": format_memo("Expenditure", exp_memo),
            "Revenue": format_memo("Revenue", rev_memo),
            "Risk": format_memo("Risk", risk_memo),
        }
        return advisor_memos, advisor_messages

    def _find_invoice(self, invoice_id: str) -> Optional[Invoice]:
        return next((i for i in self.invoices if i.id == invoice_id), None)

    def _update_cash(self, amount: float):
        self.cash += amount
        if self.cash < 0:
            self.credit_used += abs(self.cash)
            self.cash = 0.0

    def _check_day_end(self) -> bool:
        # Day ends every 3 steps (morning, afternoon, evening review cycles)
        return self._state.step_count % 3 == 0

    def _build_obs(self, reward=0.0, advisor_memos=None, advisor_messages=None,
                   late_fee=0.0, interest=0.0) -> CashflowmanagerObservation:
        active_invoices = [inv for inv in self.invoices if inv.status != "paid"]
        vendor_info = {
            vid: {"name": v.name, "trust_score": round(v.trust_score, 2)}
            for vid, v in self.vendors.items()
        }

        return CashflowmanagerObservation(
            day=self.day,
            cash=round(self.cash, 2),
            credit_used=round(self.credit_used, 2),
            credit_limit=self.credit_limit,
            invoices=active_invoices,
            receivables=self.receivables,
            vendor_profiles=vendor_info,
            advisor_memos=advisor_memos or {},
            advisor_messages=advisor_messages or {},
            negotiation_result=self.last_negotiation,
            world_events=self.last_world_events,
            reward=round(reward, 4),
            done=self.done,
            metadata={
                "step": self._state.step_count,
                "late_fee": round(late_fee, 2),
                "interest": round(interest, 4),
                "day_ended": self._state.step_count > 0 and self._state.step_count % 3 == 0,
                "world_events_count": len(self.world_model.get_triggered_events()),
            },
        )

    def get_transitions(self) -> List[Dict]:
        """Return all stored transitions for training data export."""
        return [t.model_dump() for t in self.transitions]

    @property
    def state(self) -> State:
        return self._state
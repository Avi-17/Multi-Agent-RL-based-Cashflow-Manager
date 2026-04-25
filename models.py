"""
Data models for the Cashflow Multi-Agent RL Environment.

Models:
  - Invoice: A bill from a vendor
  - Receivable: Expected payment from a customer
  - VendorProfile: Vendor negotiation traits
  - NegotiationResult: Outcome of a negotiate action
  - CashflowmanagerAction: CFO's decision (pay/defer/partial/negotiate/credit)
  - CashflowmanagerObservation: What the CFO sees after each step
  - Transition: Full (state, action, reward, reasoning) tuple for training
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from openenv.core.env_server.types import Action, Observation


class Invoice(BaseModel):
    id: str
    vendor_id: str
    amount: float
    due_in: int
    late_fee: float
    min_payment: float
    interest: float
    status: str = "unpaid"  # unpaid, partial, negotiating, deferred, paid


class Receivable(BaseModel):
    id: str
    customer_id: str
    amount: float
    expected_in: int
    probability: float


class IncomingInvoice(BaseModel):
    """An invoice that hasn't arrived yet. The CFO only knows they are coming, not the amount."""
    id: str
    vendor_id: str
    appears_on_day: int
    hidden_amount: float
    hidden_due_in: int
    hidden_late_fee: float
    hidden_interest: float


class VendorProfile(BaseModel):
    id: str
    name: str
    trust_score: float         # 0.0 to 1.0
    negotiation_flexibility: float  # 0.0 to 1.0


class NegotiationResult(BaseModel):
    """Outcome from the Vendor Agent when CFO chooses 'negotiate'."""
    accepted: bool = False
    success_probability: float = 0.0
    vendor_message: str = ""
    is_predatory: bool = False
    extension_days: int = 0


class CashflowmanagerAction(Action):
    type: str = Field(..., description="pay, defer, partial, negotiate, credit")
    invoice_id: Optional[str] = None
    amount: Optional[float] = 0.0
    memo: Optional[str] = None


class CashflowmanagerObservation(Observation):
    day: int
    cash: float
    credit_used: float
    credit_limit: float = 5000.0
    active_invoices: List[Invoice] = Field(default_factory=list)
    paid_invoices: List[Invoice] = Field(default_factory=list)
    partially_paid_invoices: List[Invoice] = Field(default_factory=list)
    overdue_invoices: List[Invoice] = Field(default_factory=list)
    upcoming_invoice_count: int = 0
    receivables: List[Receivable]
    vendor_profiles: Dict[str, Any] = Field(default_factory=dict)
    advisor_memos: Dict[str, Any] = Field(default_factory=dict)
    advisor_messages: Dict[str, str] = Field(default_factory=dict)
    negotiation_result: Optional[NegotiationResult] = None
    world_events: List[str] = Field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Transition(BaseModel):
    """Full training transition for RL/SFT data collection."""
    day: int
    state_summary: str
    advisor_memos: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    reasoning: str = ""
    next_state_summary: str = ""


# ─────────────────────────────────────────────
# Simulation Engine Data Models
# ─────────────────────────────────────────────

class DayLog(BaseModel):
    """Everything that happened on a single day."""
    day: int

    # State at the start of the day (before any actions)
    opening_cash: float
    opening_credit_used: float
    active_invoice_count: int
    overdue_invoice_count: int

    # What the advisors said
    advisor_memos: Dict[str, str] = Field(default_factory=dict)

    # Actions the CFO took
    actions: List[CashflowmanagerAction] = Field(default_factory=list)

    # Events that happened (incoming invoices activated, receivables collected, etc.)
    events: List[str] = Field(default_factory=list)

    # State at the end of the day (after all actions and events)
    closing_cash: float = 0.0
    closing_credit_used: float = 0.0
    invoices_paid_today: int = 0
    late_fees_incurred: float = 0.0
    interest_incurred: float = 0.0
    revenue_collected: float = 0.0
    reward: float = 0.0


class SimulationResult(BaseModel):
    """Output of a complete simulation run."""
    difficulty: str
    sim_window: int
    seed: int

    days: List[DayLog] = Field(default_factory=list)

    # Final summary
    final_cash: float = 0.0
    final_credit_used: float = 0.0
    total_invoices: int = 0
    invoices_paid: int = 0
    invoices_overdue: int = 0
    total_late_fees: float = 0.0
    total_interest: float = 0.0
    total_revenue_collected: float = 0.0
    total_reward: float = 0.0
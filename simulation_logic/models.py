"""
Models for the Simulation Logic module.

Defines the state structure, day logs, and simulation results.
Reuses Invoice and Receivable from the root models.py where possible.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────
# Invoice (local copy — keeps this module self-contained)
# ─────────────────────────────────────────────

class SimInvoice(BaseModel):
    """A bill the company must pay to a vendor."""
    id: str
    vendor: str                     # human-readable vendor name
    amount: float                   # remaining amount owed (₹)
    original_amount: float          # what it was when first created
    due_in: int                     # days until due (negative = overdue)
    late_fee: float                 # flat fee charged per day when overdue
    interest_rate: float            # daily interest rate (e.g. 0.02 = 2%)
    status: str = "unpaid"          # unpaid | partial | paid | overdue


class SimReceivable(BaseModel):
    """Money the company expects to receive from a customer."""
    id: str
    customer: str                   # human-readable customer name
    amount: float
    arrives_on_day: int             # the day this money arrives
    probability: float              # chance the payment actually comes through


class IncomingInvoice(BaseModel):
    """A future invoice the agent knows is coming but can't see details of."""
    id: str
    vendor: str
    appears_on_day: int             # when it becomes an active invoice
    # ── Agent does NOT see these until the invoice activates ──
    hidden_amount: float
    hidden_due_in: int              # days to pay after it appears
    hidden_late_fee: float
    hidden_interest_rate: float


# ─────────────────────────────────────────────
# Simulation State
# ─────────────────────────────────────────────

class SimState(BaseModel):
    """Complete snapshot of the simulation at a given point in time."""
    day: int
    cash: float
    credit_used: float
    credit_limit: float

    # ── Invoice buckets ──
    # Only PAID invoices are removed from active_invoices.
    # Partial and overdue invoices stay in active_invoices.
    active_invoices: List[SimInvoice] = Field(default_factory=list)
    paid_invoices: List[SimInvoice] = Field(default_factory=list)
    partially_paid_invoices: List[SimInvoice] = Field(default_factory=list)
    overdue_invoices: List[SimInvoice] = Field(default_factory=list)

    receivables: List[SimReceivable] = Field(default_factory=list)

    # Agent sees the COUNT of upcoming invoices, not their amounts
    upcoming_invoice_count: int = 0


# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────

class SimAction(BaseModel):
    """One action taken by the CFO agent."""
    type: str                       # pay | partial | defer | negotiate | credit
    invoice_id: Optional[str] = None
    amount: Optional[float] = None
    reasoning: str = ""


# ─────────────────────────────────────────────
# Day Log
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
    actions: List[SimAction] = Field(default_factory=list)

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


# ─────────────────────────────────────────────
# Simulation Result
# ─────────────────────────────────────────────

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

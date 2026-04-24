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
    invoices: List[Invoice]
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
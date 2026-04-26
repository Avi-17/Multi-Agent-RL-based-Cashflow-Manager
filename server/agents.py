"""
Multi-Agent Advisory System for Cashflow Environment.

Each agent receives a PARTIAL view of the state and produces a structured memo
using In-Context Learning (Llama 3.1-8B).

Agent roles:
  - Expenditure Agent: Analyzes liabilities and prioritizes payments
  - Revenue Agent: Tracks expected inflows and cash projections
  - Risk Agent: Monitors hidden threats and recommends buffers
  - CFO Agent: Reads memos and executes actions
"""

import json
import os
from typing import List, Dict, Any

from models import CashflowmanagerAction
from server.client import get_model_response
from server.state_serializer import serialize_state, serialize_state_minimal, serialize_history

DEFAULT_CFO_MODEL = os.environ.get("MODEL_NAME") or "llama-3.1-8b-instant"
EXPENDITURE_MODEL = os.environ.get("EXPENDITURE_MODEL_NAME") or os.environ.get("ADVISOR_MODEL_NAME") or "llama-3.1-8b-instant"
REVENUE_MODEL = os.environ.get("REVENUE_MODEL_NAME") or os.environ.get("ADVISOR_MODEL_NAME") or "llama-3.1-8b-instant"
RISK_MODEL = os.environ.get("RISK_MODEL_NAME") or os.environ.get("ADVISOR_MODEL_NAME") or "llama-3.1-8b-instant"
CFO_MODEL = os.environ.get("CFO_MODEL_NAME") or DEFAULT_CFO_MODEL

# Per-agent API key indices. With 3 keys in GROQ_API_KEYS, advisors get distinct
# keys (one each) so their parallel calls don't share a TPM bucket. CFO reuses
# the Risk key (index 2) since Risk has the smallest prompt -> most TPM headroom
# left, and CFO runs after advisors finish so there's no concurrent contention.
EXPENDITURE_KEY_INDEX = int(os.environ.get("EXPENDITURE_KEY_INDEX", "0"))
REVENUE_KEY_INDEX = int(os.environ.get("REVENUE_KEY_INDEX", "1"))
RISK_KEY_INDEX = int(os.environ.get("RISK_KEY_INDEX", "2"))
CFO_KEY_INDEX = int(os.environ.get("CFO_KEY_INDEX", "2"))


# ═══════════════════════════════════════════════════════
# SYSTEM PROMPTS (with Few-Shot & CoT)
# ═══════════════════════════════════════════════════════

EXPENDITURE_SYSTEM_PROMPT = """You are the Expenditure Agent for a company's finance team.
Your role: Analyze all unpaid invoices and advise the CFO on payment priority.

You must output a JSON object with:
- "priority_list": ordered list of invoice IDs from most to least urgent
- "critical_invoices": list of invoice IDs that MUST be paid today
- "recommended_action": one of "pay_critical", "defer_all", "partial_payments"
- "reasoning": 1-2 sentence explanation summary
- "total_liability": total amount owed across all invoices

Rules:
- Overdue invoices (due_in <= 0) are CRITICAL — late fees compound daily
- Invoices due in 1-2 days are URGENT
- Factor in available cash when recommending actions
- All numerical fields must be FINAL CALCULATED NUMBERS. Never include mathematical expressions (like 100-50).

Example output:
{
  "priority_list": ["INV-001", "INV-002"],
  "critical_invoices": ["INV-001"],
  "recommended_action": "pay_critical",
  "reasoning": "Pay the overdue invoice to stop penalties, defer the rest.",
  "total_liability": 15000.0
}"""

REVENUE_SYSTEM_PROMPT = """You are the Revenue Agent for a company's finance team.
Your role: Track expected cash inflows and advise the CFO on liquidity outlook.

You must output a JSON object with:
- "total_expected_inflow": sum of all expected receivables (a single number)
- "reliable_inflows": list of receivable IDs with probability > 0.85
- "at_risk_inflows": list of receivable IDs with probability < 0.8
- "cash_projection_3day": estimated cash position in 3 days (MUST be a single pre-calculated number)
- "recommendation": one of "cash_sufficient", "cash_tight", "cash_critical"
- "reasoning": 1-2 sentence explanation summary

Rules:
- Weight inflows by their probability
- Consider timing — money due in 1 day is more relevant than in 10 days
- Flag any inflows that seem unreliable

CRITICAL: Every numerical value MUST be a single final number — never write arithmetic expressions.
  WRONG: "cash_projection_3day": 45660.0 - 2079.0 - 104.0
  RIGHT: "cash_projection_3day": 43477.0

Example output:
{
  "total_expected_inflow": 14000.0,
  "reliable_inflows": ["RCV-001"],
  "at_risk_inflows": ["RCV-002"],
  "cash_projection_3day": 12000.0,
  "recommendation": "cash_tight",
  "reasoning": "We have one reliable inflow, but overall liquidity remains tight."
}"""

RISK_SYSTEM_PROMPT = """You are the Risk Agent for a company's finance team.
Your role: Monitor financial health and warn about potential threats.

You must output a JSON object with:
- "risk_level": one of "low", "moderate", "elevated", "critical"
- "credit_utilization": percentage of credit limit used
- "recommended_buffer": minimum cash to keep as safety reserve
- "threats": list of identified threat descriptions
- "recommendation": one of "maintain_buffer", "draw_credit", "reduce_spending"
- "reasoning": 1-2 sentence explanation summary

Rules:
- Credit utilization above 60% is concerning
- Suggest higher buffers if debt ratio is high
- All numerical fields must be FINAL CALCULATED NUMBERS. Never include mathematical expressions (like 100-50).

Example output:
{
  "risk_level": "critical",
  "credit_utilization": 80.0,
  "recommended_buffer": 5000.0,
  "threats": ["High credit usage", "Multiple overdue invoices"],
  "recommendation": "reduce_spending",
  "reasoning": "Stop all non-critical spending immediately due to high debt load."
}"""

CFO_SYSTEM_PROMPT = """You are the CFO of a company managing daily cash flow decisions.
You receive memos from three advisors (Expenditure, Revenue, Risk) and a list of unpaid invoices.

Output ONLY a single valid JSON object — no prose before or after it:
{
  "thought_process": "<your reasoning here>",
  "actions": [
    {"invoice_id": "INV-001", "type": "pay", "amount": 5000.0, "reasoning": "Overdue, pay now."},
    {"invoice_id": "INV-002", "type": "defer", "amount": 0.0, "reasoning": "Not due yet."}
  ],
  "overall_strategy": "<1 sentence>",
  "confidence": 0.9
}

Rules:
- Include one action per invoice listed.
- Valid types: pay, defer, partial, negotiate, credit.
- amount MUST be a plain number (e.g. 5000.0). No math expressions.
- You CANNOT spend more cash than you currently have.
- Use credit only if bankruptcy is imminent — it reduces your score.
- Pay overdue invoices first to stop compounding late fees.
- Defer non-urgent invoices to preserve cash.
- "confidence" MUST be your true self-assessed value in [0,1], not a default/template value.
- If there are overdue invoices and enough cash, confidence should be lower unless your plan pays them."""


def _normalize_action_type(raw_type: Any) -> str:
    t = str(raw_type or "defer").strip().lower()
    aliases = {
        "payment": "pay",
        "full_pay": "pay",
        "part_pay": "partial",
        "draw_credit": "credit",
        "wait": "defer",
    }
    t = aliases.get(t, t)
    return t if t in {"pay", "defer", "partial", "negotiate", "credit"} else "defer"


def _calibrate_confidence(state, actions: List[CashflowmanagerAction], model_confidence: float) -> float:
    """
    Prevent static/template confidence values by blending model self-rating
    with a plan-quality score derived from the actual action set.
    """
    unpaid = [inv for inv in state.active_invoices if inv.status != "paid"]
    if not unpaid:
        return max(0.55, min(model_confidence, 0.95))

    overdue_ids = {inv.id for inv in unpaid if inv.due_in < 0}
    due_soon_ids = {inv.id for inv in unpaid if inv.due_in <= 1}

    acted_ids = {a.invoice_id for a in actions if a.invoice_id}
    paid_or_partial_ids = {
        a.invoice_id for a in actions
        if a.invoice_id and a.type in {"pay", "partial"} and (a.amount or 0.0) > 0
    }

    coverage = len(acted_ids) / max(len(unpaid), 1)
    overdue_coverage = (len(overdue_ids & paid_or_partial_ids) / max(len(overdue_ids), 1)) if overdue_ids else 1.0
    due_soon_coverage = (len(due_soon_ids & paid_or_partial_ids) / max(len(due_soon_ids), 1)) if due_soon_ids else 1.0
    has_credit = any(a.type == "credit" and (a.amount or 0.0) > 0 for a in actions)
    defer_ratio = (
        len([a for a in actions if a.type == "defer"]) / max(len(actions), 1)
        if actions else 1.0
    )

    heuristic = (
        0.35 * coverage
        + 0.35 * overdue_coverage
        + 0.20 * due_soon_coverage
        + 0.10 * (1.0 - min(defer_ratio, 1.0))
        - (0.08 if has_credit else 0.0)
    )
    heuristic = max(0.0, min(heuristic, 1.0))

    # Blend model confidence with action-quality confidence; this avoids flat 0.90 every day.
    blended = 0.45 * max(0.0, min(model_confidence, 1.0)) + 0.55 * heuristic
    return max(0.05, min(blended, 0.98))


# ═══════════════════════════════════════════════════════
# ICL AGENT IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════

def expenditure_agent(state, past_logs: list) -> Dict[str, Any]:
    # Plan B: advisors skip history (CFO still gets it); state is minimal since
    # the invoice list below already covers what this agent needs.
    state_text = serialize_state_minimal(state)

    inv_lines = []
    for inv in state.active_invoices:
        if inv.status == "paid":
            continue
        due_label = "OVERDUE" if inv.due_in < 0 else ("TODAY" if inv.due_in == 0 else f"in {inv.due_in}d")
        inv_lines.append(f"  {inv.id}: ₹{inv.amount:,.0f} due {due_label}, late_fee=₹{inv.late_fee:,.0f}, interest={inv.interest*100:.1f}%")
    invoices_text = "\n".join(inv_lines) if inv_lines else "  No unpaid invoices."

    prompt = f"""{state_text}

UNPAID INVOICES:
{invoices_text}

Analyze these invoices and provide your memo."""

    data = get_model_response(
        prompt,
        system_prompt=EXPENDITURE_SYSTEM_PROMPT,
        response_format="json",
        max_tokens=384,
        model_name=EXPENDITURE_MODEL,
        key_index=EXPENDITURE_KEY_INDEX,
    )

    if data and isinstance(data, dict) and ("recommended_action" in data or "recommendation" in data):
        return data

    # Safe Fallback
    return {
        "thought_process": "Fallback triggered.",
        "priority_list": [],
        "critical_invoices": [],
        "recommended_action": "defer_all",
        "reasoning": "Fallback to defer due to parsing failure.",
        "total_liability": 0.0
    }


def revenue_agent(state, past_logs: list) -> Dict[str, Any]:
    state_text = serialize_state_minimal(state)

    rec_lines = []
    for r in state.receivables:
        rec_lines.append(f"  {r.id}: ₹{r.amount:,.0f} from {r.customer_id}, expected Day {r.expected_in}, prob={r.probability*100:.0f}%")
    rec_text = "\n".join(rec_lines) if rec_lines else "  No pending receivables."

    prompt = f"""{state_text}

RECEIVABLES:
{rec_text}

Project cash inflows and provide your memo."""

    data = get_model_response(
        prompt,
        system_prompt=REVENUE_SYSTEM_PROMPT,
        response_format="json",
        max_tokens=384,
        model_name=REVENUE_MODEL,
        key_index=REVENUE_KEY_INDEX,
    )

    if data and isinstance(data, dict) and "recommendation" in data:
        return data

    # Safe Fallback
    return {
        "thought_process": "Fallback triggered.",
        "total_expected_inflow": 0.0,
        "reliable_inflows": [],
        "at_risk_inflows": [],
        "cash_projection_3day": state.cash,
        "recommendation": "cash_sufficient",
        "reasoning": "Fallback."
    }


def risk_agent(state, past_logs: list) -> Dict[str, Any]:
    state_text = serialize_state_minimal(state)

    total_debt = sum(inv.amount for inv in state.active_invoices)
    credit_util = state.credit_used / (state.credit_limit + 1.0)

    prompt = f"""{state_text}

RISK METRICS:
  Debt-to-Cash Ratio: {total_debt / (state.cash + 1.0):.1f}
  Credit Utilization: {credit_util*100:.0f}%

Assess financial risk and provide your memo."""

    data = get_model_response(
        prompt,
        system_prompt=RISK_SYSTEM_PROMPT,
        response_format="json",
        max_tokens=384,
        model_name=RISK_MODEL,
        key_index=RISK_KEY_INDEX,
    )

    if data and isinstance(data, dict) and "risk_level" in data:
        return data

    # Safe Fallback
    return {
        "thought_process": "Fallback triggered.",
        "risk_level": "moderate",
        "credit_utilization": 0.0,
        "recommended_buffer": 0.0,
        "threats": [],
        "recommendation": "maintain_buffer",
        "reasoning": "Fallback."
    }


def cfo_decide_with_metadata(state, advisor_memos: Dict[str, str], past_logs: list) -> Dict[str, Any]:
    cfo_past_logs = past_logs[-2:] if past_logs else []
    history_text = serialize_history(cfo_past_logs)
    state_text = serialize_state(state)
    memo_text = "\n".join([f"[{name}]: {memo}" for name, memo in advisor_memos.items()])

    inv_lines = []
    for inv in state.active_invoices:
        if inv.status == "paid":
            continue
        due_label = "OVERDUE" if inv.due_in < 0 else ("TODAY" if inv.due_in == 0 else f"in {inv.due_in}d")
        inv_lines.append(f"  {inv.id}: ₹{inv.amount:,.0f} due {due_label}")
    inv_text = "\n".join(inv_lines) if inv_lines else "  No active invoices."

    prompt = f"""{history_text}

{state_text}

ADVISOR MEMOS:
{memo_text}

INVOICES REQUIRING DECISION:
{inv_text}

For EACH invoice, decide an action.
You CANNOT spend more than ₹{state.cash:,.0f} (current cash).
Credit available: ₹{state.credit_limit - state.credit_used:,.0f}."""

    data = get_model_response(
        prompt,
        system_prompt=CFO_SYSTEM_PROMPT,
        response_format="json",
        max_tokens=512,
        model_name=CFO_MODEL,
        key_index=CFO_KEY_INDEX,
    )

    if data and isinstance(data, dict) and "actions" in data:
        actions = []
        for a in data["actions"]:
            try:
                actions.append(CashflowmanagerAction(
                    type=_normalize_action_type(a.get("type", "defer")),
                    invoice_id=a.get("invoice_id"),
                    amount=float(a.get("amount", 0.0)),
                    memo=a.get("reasoning", ""),
                ))
            except Exception:
                continue
        if actions:
            try:
                model_confidence = float(data.get("confidence", 0.0))
            except Exception:
                model_confidence = 0.0
            confidence = _calibrate_confidence(state, actions, model_confidence)
            return {
                "actions": actions,
                "confidence": max(0.0, min(confidence, 1.0)),
                "fallback": False,
            }

    # Safe Fallback
    print(f"[ICL CFO] LLM response invalid (got: {str(data)[:200]}), falling back to safe defer.")
    return {
        "actions": [
            CashflowmanagerAction(type="defer", invoice_id=inv.id, memo="Fallback defer")
            for inv in state.active_invoices if inv.status != "paid"
        ],
        "confidence": 0.0,
        "fallback": True,
    }


def cfo_decide(state, advisor_memos: Dict[str, str], past_logs: list) -> list:
    return cfo_decide_with_metadata(state, advisor_memos, past_logs)["actions"]


# ═══════════════════════════════════════════════════════
# CONVENIENCE FORMATTER
# ═══════════════════════════════════════════════════════

def format_memo(agent_name: str, memo: Dict[str, Any]) -> str:
    """Convert structured memo to a human-readable string for the observation."""
    if agent_name == "Expenditure":
        return (
            f"[{memo.get('recommended_action', 'UNKNOWN').upper()}] "
            f"Total Debt: ₹{memo.get('total_liability', 0):.0f}. "
            f"{memo.get('reasoning', '')}"
        )
    elif agent_name == "Revenue":
        return (
            f"[{memo.get('recommendation', 'UNKNOWN').upper()}] "
            f"Inflow: ₹{memo.get('total_expected_inflow', 0):.0f}. "
            f"Net Position: ₹{memo.get('cash_projection_3day', 0):.0f}. "
            f"{memo.get('reasoning', '')}"
        )
    elif agent_name == "Risk":
        return (
            f"[{memo.get('risk_level', 'UNKNOWN').upper()}] "
            f"Credit: {memo.get('credit_utilization', 'N/A')}%. "
            f"Survival Target: ₹{memo.get('recommended_buffer', 0):.0f}. "
            f"Threats: {len(memo.get('threats', []))}. "
            f"{memo.get('reasoning', '')}"
        )
    return "No memo available."

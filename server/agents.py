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
from typing import List, Dict, Any

from models import CashflowmanagerAction
from server.client import get_model_response
from server.state_serializer import serialize_state, serialize_history


# ═══════════════════════════════════════════════════════
# SYSTEM PROMPTS (with Few-Shot & CoT)
# ═══════════════════════════════════════════════════════

EXPENDITURE_SYSTEM_PROMPT = """You are the Expenditure Agent for a company's finance team.
Your role: Analyze all unpaid invoices and advise the CFO on payment priority.

You must output a JSON object with:
- "thought_process": Your chain of thought analyzing the invoices (DO THIS FIRST)
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
  "thought_process": "INV-001 is overdue, so it will incur penalties today. INV-002 is due in 5 days. We have enough cash to pay INV-001 but not both.",
  "priority_list": ["INV-001", "INV-002"],
  "critical_invoices": ["INV-001"],
  "recommended_action": "pay_critical",
  "reasoning": "Pay the overdue invoice to stop penalties, defer the rest.",
  "total_liability": 15000.0
}"""

REVENUE_SYSTEM_PROMPT = """You are the Revenue Agent for a company's finance team.
Your role: Track expected cash inflows and advise the CFO on liquidity outlook.

You must output a JSON object with:
- "thought_process": Your chain of thought analyzing the receivables (DO THIS FIRST)
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

CRITICAL: Every numerical value MUST be a single final number. Do your math in thought_process, then write ONLY the result.
  WRONG: "cash_projection_3day": 45660.0 - 2079.0 - 104.0
  RIGHT: "cash_projection_3day": 43477.0

Example output:
{
  "thought_process": "RCV-001 has 90% probability, very reliable. RCV-002 has 50% probability, highly at risk. Cash is 20000, expected outflows ~8000, so 3-day projection is 20000-8000=12000.",
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
- "thought_process": Your chain of thought analyzing debt and credit usage (DO THIS FIRST)
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
  "thought_process": "Credit utilization is at 80%, which is very high. We have 3 overdue invoices compounding penalties. We need to halt non-essential spending.",
  "risk_level": "critical",
  "credit_utilization": 80.0,
  "recommended_buffer": 5000.0,
  "threats": ["High credit usage", "Multiple overdue invoices"],
  "recommendation": "reduce_spending",
  "reasoning": "Stop all non-critical spending immediately due to high debt load."
}"""

CFO_SYSTEM_PROMPT = """You are the CFO of a company managing daily cash flow decisions.
You receive memos from three advisors: Expenditure, Revenue, and Risk.

Based on their advice, history feedback, and the current state, decide the action for EACH invoice. Also consider the fact that they are giving advice, they might not be right always.

You must output a JSON object with:
- "thought_process": Your chain of thought analyzing memos and invoices (DO THIS FIRST)
- "actions": list of {"invoice_id": "...", "type": "pay|defer|partial|negotiate|credit", "amount": float, "reasoning": "..."}
- "overall_strategy": 1-2 sentence explanation of your approach
- "confidence": float 0.0-1.0

Rules:
- Use negotiate when vendor trust is high and you need relief
- Use credit only when necessary — it reduces your score
- Consider advisor warnings seriously
- All numerical fields must be FINAL CALCULATED NUMBERS. Never include mathematical expressions (like 100-50). draw credit if bankruptcy is imminent.
- Pay overdue invoices to stop compounding late fees.
- Defer non-urgent invoices to preserve cash.

Example output:
{
  "thought_process": "INV-001 is overdue and racking up late fees. I have enough cash to pay it. INV-002 is due in 4 days, so I can defer it to keep my cash buffer.",
  "actions": [
    {"invoice_id": "INV-001", "type": "pay", "amount": 5000, "reasoning": "Pay overdue invoice to stop penalties."},
    {"invoice_id": "INV-002", "type": "defer", "amount": 0, "reasoning": "Not due yet, preserving cash."}
  ],
  "overall_strategy": "Clear urgent debt while deferring future liabilities.",
  "confidence": 0.95
}"""


# ═══════════════════════════════════════════════════════
# ICL AGENT IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════

def expenditure_agent(state, past_logs: list) -> Dict[str, Any]:
    state_text = serialize_state(state)
    history_text = serialize_history(past_logs)

    inv_lines = []
    for inv in state.active_invoices:
        if inv.status == "paid":
            continue
        due_label = "OVERDUE" if inv.due_in < 0 else ("TODAY" if inv.due_in == 0 else f"in {inv.due_in}d")
        inv_lines.append(f"  {inv.id}: ₹{inv.amount:,.0f} due {due_label}, late_fee=₹{inv.late_fee:,.0f}, interest={inv.interest*100:.1f}%")
    invoices_text = "\n".join(inv_lines) if inv_lines else "  No unpaid invoices."

    prompt = f"""{history_text}

{state_text}

UNPAID INVOICES:
{invoices_text}

Available Cash: ₹{state.cash:,.0f}

Analyze these invoices and provide your memo."""

    data = get_model_response(prompt, system_prompt=EXPENDITURE_SYSTEM_PROMPT, response_format="json")

    if data and isinstance(data, dict) and "recommendation" in data:
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
    state_text = serialize_state(state)
    history_text = serialize_history(past_logs)

    rec_lines = []
    for r in state.receivables:
        rec_lines.append(f"  {r.id}: ₹{r.amount:,.0f} from {r.customer_id}, expected Day {r.expected_in}, prob={r.probability*100:.0f}%")
    rec_text = "\n".join(rec_lines) if rec_lines else "  No pending receivables."

    prompt = f"""{history_text}

{state_text}

RECEIVABLES:
{rec_text}

Project cash inflows and provide your memo."""

    data = get_model_response(prompt, system_prompt=REVENUE_SYSTEM_PROMPT, response_format="json")

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
    state_text = serialize_state(state)
    history_text = serialize_history(past_logs)

    total_debt = sum(inv.amount for inv in state.active_invoices)
    credit_util = state.credit_used / (state.credit_limit + 1.0)

    prompt = f"""{history_text}

{state_text}

RISK METRICS:
  Total Debt: ₹{total_debt:,.0f}
  Debt-to-Cash Ratio: {total_debt / (state.cash + 1.0):.1f}
  Credit Utilization: {credit_util*100:.0f}%
  Overdue Invoices: {len(state.overdue_invoices)}

Assess financial risk and provide your memo."""

    data = get_model_response(prompt, system_prompt=RISK_SYSTEM_PROMPT, response_format="json")

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


def cfo_decide(state, advisor_memos: Dict[str, str], past_logs: list) -> list:
    state_text = serialize_state(state)
    history_text = serialize_history(past_logs)

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
        max_tokens=1024,
    )

    if data and isinstance(data, dict) and "actions" in data:
        actions = []
        for a in data["actions"]:
            try:
                actions.append(CashflowmanagerAction(
                    type=a.get("type", "defer"),
                    invoice_id=a.get("invoice_id"),
                    amount=float(a.get("amount", 0.0)),
                    memo=a.get("reasoning", ""),
                ))
            except Exception:
                continue
        if actions:
            return actions

    # Safe Fallback
    print("[ICL CFO] LLM response invalid, falling back to safe defer.")
    return [
        CashflowmanagerAction(type="defer", invoice_id=inv.id, memo="Fallback defer")
        for inv in state.active_invoices if inv.status != "paid"
    ]


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

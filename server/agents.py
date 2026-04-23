"""
Multi-Agent Advisory System for Cashflow Environment.

Each agent receives a PARTIAL view of the state and produces a structured memo.
These memos are:
  1. Fed to the CFO agent as context for decision-making
  2. Logged as SFT training data for fine-tuning smaller models

Agent roles:
  - Expenditure Agent: Analyzes liabilities and prioritizes payments
  - Revenue Agent: Tracks expected inflows and cash projections
  - Risk Agent: Monitors hidden threats and recommends buffers
  - Vendor Agent: Responds to negotiation requests
"""

import random
import json
from typing import List, Dict, Any, Optional


# ═══════════════════════════════════════════════════════
# SYSTEM PROMPTS (used for SFT data generation & inference)
# ═══════════════════════════════════════════════════════

EXPENDITURE_SYSTEM_PROMPT = """You are the Expenditure Agent for a company's finance team.
Your role: Analyze all unpaid invoices and advise the CFO on payment priority.

You must output a JSON object with:
- "priority_list": ordered list of invoice IDs from most to least urgent
- "critical_invoices": list of invoice IDs that MUST be paid today
- "recommended_action": one of "pay_critical", "defer_all", "partial_payments"
- "reasoning": 1-2 sentence explanation
- "total_liability": total amount owed across all invoices

Rules:
- Overdue invoices (due_in <= 0) are CRITICAL — late fees compound daily
- Invoices due in 1-2 days are URGENT
- Consider the interest rate when prioritizing
- Factor in available cash when recommending actions"""

REVENUE_SYSTEM_PROMPT = """You are the Revenue Agent for a company's finance team.
Your role: Track expected cash inflows and advise the CFO on liquidity outlook.

You must output a JSON object with:
- "total_expected_inflow": sum of all expected receivables
- "reliable_inflows": list of receivable IDs with probability > 0.85
- "at_risk_inflows": list of receivable IDs with probability < 0.8
- "cash_projection_3day": estimated cash position in 3 days
- "recommendation": one of "cash_sufficient", "cash_tight", "cash_critical"
- "reasoning": 1-2 sentence explanation

Rules:
- Weight inflows by their probability
- Consider timing — money due in 1 day is more relevant than in 10 days
- Flag any inflows that seem unreliable"""

RISK_SYSTEM_PROMPT = """You are the Risk Agent for a company's finance team.
Your role: Monitor financial health and warn about potential threats.

You must output a JSON object with:
- "risk_level": one of "low", "moderate", "elevated", "critical"
- "credit_utilization": percentage of credit limit used
- "recommended_buffer": minimum cash to keep as safety reserve
- "threats": list of identified threat descriptions
- "recommendation": one of "maintain_buffer", "draw_credit", "reduce_spending"
- "reasoning": 1-2 sentence explanation

Rules:
- Credit utilization above 60% is concerning
- If market stress is elevated, recommend larger buffers
- Consider vendor sentiment when assessing risk"""

VENDOR_SYSTEM_PROMPT = """You are a Vendor negotiating with a customer company.
The company wants to renegotiate payment terms on an invoice.

You must output a JSON object with:
- "decision": one of "accept", "reject", "counter"
- "late_fee_waiver": true/false (only if accepting)
- "extension_days": number of extra days granted (0 if rejecting)
- "counter_terms": string describing counter-offer (only if countering)
- "reasoning": 1-2 sentence explanation

Rules:
- Your trust_score and negotiation_flexibility affect your willingness
- If trust is high (>0.7), you're more likely to accept
- If the company has a history of late payments, you're less flexible
- Counter-offers typically offer partial relief"""

CFO_SYSTEM_PROMPT = """You are the CFO of a company managing daily cash flow decisions.
You receive memos from three advisors:
- Expenditure Agent: payment priorities
- Revenue Agent: cash inflow projections
- Risk Agent: threat assessment

Based on their advice and the current financial state, decide the action for each invoice.

You must output a JSON object with:
- "actions": list of {"invoice_id": "...", "type": "pay|defer|partial|negotiate|credit", "amount": float, "reasoning": "..."}
- "overall_strategy": 1-2 sentence explanation of your approach
- "confidence": float 0.0-1.0

Rules:
- You CANNOT spend more cash than available (cash + remaining credit)
- Balance between paying urgent invoices and maintaining reserves
- Use negotiate when vendor trust is high and you need relief
- Use credit only when necessary — it reduces your score
- Consider advisor warnings seriously"""


# ═══════════════════════════════════════════════════════
# RULE-BASED AGENT IMPLEMENTATIONS
# (Deterministic logic that produces structured outputs)
# ═══════════════════════════════════════════════════════

def expenditure_agent(invoices: list, cash: float) -> Dict[str, Any]:
    """
    Expenditure Agent: Analyzes invoices and produces a priority memo.
    """
    if not invoices:
        return {
            "priority_list": [],
            "critical_invoices": [],
            "recommended_action": "defer_all",
            "reasoning": "No active invoices.",
            "total_liability": 0.0,
        }

    scored = []
    for inv in invoices:
        if inv.status == "paid":
            continue
        urgency = 0
        if inv.due_in <= 0:
            urgency = 100 + abs(inv.due_in) * 10  # Overdue — highest priority
        elif inv.due_in <= 2:
            urgency = 70 + (3 - inv.due_in) * 10  # Urgent
        elif inv.due_in <= 4:
            urgency = 40
        else:
            urgency = 10
        # Factor in interest rate and amount
        urgency += inv.interest * 100
        urgency += inv.amount / 500
        scored.append((inv, urgency))

    scored.sort(key=lambda x: -x[1])
    priority_list = [s[0].id for s in scored]
    critical = [s[0].id for s in scored if s[0].due_in <= 0]
    total_liability = sum(inv.amount for inv, _ in scored)

    if critical:
        action = "pay_critical"
        reasoning = f"{len(critical)} invoices are OVERDUE. Immediate payment required to stop late fee compounding."
    elif total_liability > cash * 0.8:
        action = "partial_payments"
        reasoning = f"Total liability ({total_liability:.0f}) exceeds 80% of cash ({cash:.0f}). Recommend partial payments on most urgent."
    else:
        action = "pay_critical"
        reasoning = f"Cash is sufficient. Pay invoices in priority order."

    return {
        "priority_list": priority_list,
        "critical_invoices": critical,
        "recommended_action": action,
        "reasoning": reasoning,
        "total_liability": round(total_liability, 2),
    }


def revenue_agent(receivables: list, cash: float, day: int) -> Dict[str, Any]:
    """
    Revenue Agent: Analyzes expected inflows and projects cash position.
    """
    if not receivables:
        return {
            "total_expected_inflow": 0.0,
            "reliable_inflows": [],
            "at_risk_inflows": [],
            "cash_projection_3day": cash,
            "recommendation": "cash_critical",
            "reasoning": "No expected receivables. Cash position depends entirely on current balance.",
        }

    total = sum(r.amount * r.probability for r in receivables)
    reliable = [r.id for r in receivables if r.probability >= 0.85]
    at_risk = [r.id for r in receivables if r.probability < 0.8]

    # 3-day projection: sum inflows due within 3 days weighted by probability
    inflow_3d = sum(
        r.amount * r.probability
        for r in receivables
        if r.expected_in <= 3
    )
    projection = cash + inflow_3d

    if projection > 500000:
        rec = "cash_sufficient"
        reasoning = f"3-day projection: {projection:.0f}. Inflows look healthy."
    elif projection > 200000:
        rec = "cash_tight"
        reasoning = f"3-day projection: {projection:.0f}. Cash is tight, conserve spending."
    else:
        rec = "cash_critical"
        reasoning = f"3-day projection: {projection:.0f}. Critical — consider drawing credit."

    return {
        "total_expected_inflow": round(total, 2),
        "reliable_inflows": reliable,
        "at_risk_inflows": at_risk,
        "cash_projection_3day": round(projection, 2),
        "recommendation": rec,
        "reasoning": reasoning,
    }


def risk_agent(cash: float, credit_used: float, credit_limit: float,
               world_hints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Risk Agent: Assesses financial health and hidden threats.
    Gets PARTIAL information from the world model via world_hints.
    """
    credit_util = (credit_used / credit_limit * 100) if credit_limit > 0 else 0
    market_stress = world_hints.get("market_stress", 0.0)
    risk_level_hint = world_hints.get("upcoming_risk_level", "low")
    vendor_sentiment = world_hints.get("vendor_sentiment", {})

    threats = []
    if credit_util > 60:
        threats.append(f"Credit utilization at {credit_util:.0f}% — approaching limit")
    if market_stress > 0.2:
        threats.append(f"Market stress elevated ({market_stress:.2f}) — expect volatility")
    if risk_level_hint == "critical":
        threats.append("Intelligence suggests CRITICAL upcoming financial events")
    elif risk_level_hint == "elevated":
        threats.append("Elevated risk of upcoming cash shocks detected")
    for vid, sentiment in vendor_sentiment.items():
        if sentiment == "negative":
            threats.append(f"Vendor {vid} sentiment is negative — negotiation may fail")

    if len(threats) >= 3 or risk_level_hint == "critical":
        risk_level = "critical"
        rec = "reduce_spending"
        buffer = cash * 0.5
    elif len(threats) >= 2 or risk_level_hint == "elevated":
        risk_level = "elevated"
        rec = "draw_credit"
        buffer = cash * 0.4
    elif len(threats) >= 1:
        risk_level = "moderate"
        rec = "maintain_buffer"
        buffer = cash * 0.3
    else:
        risk_level = "low"
        rec = "maintain_buffer"
        buffer = cash * 0.2

    reasoning = f"Risk level: {risk_level}. " + (threats[0] if threats else "No immediate threats detected.")

    return {
        "risk_level": risk_level,
        "credit_utilization": round(credit_util, 1),
        "recommended_buffer": round(buffer, 2),
        "threats": threats,
        "recommendation": rec,
        "reasoning": reasoning,
    }


def vendor_agent(vendor_profile, invoice, company_trust_history: float = 0.5) -> Dict[str, Any]:
    """
    Vendor Agent: Responds to negotiation requests.
    Decision is probabilistic based on vendor profile and company history.
    """
    flexibility = vendor_profile.trust_score * vendor_profile.negotiation_flexibility
    # Company trust history modifies the outcome
    accept_prob = flexibility * (0.5 + 0.5 * company_trust_history)

    roll = random.random()

    if roll < accept_prob:
        return {
            "decision": "accept",
            "late_fee_waiver": True,
            "extension_days": random.randint(2, 5),
            "counter_terms": "",
            "reasoning": f"Good relationship with this customer. Waiving late fee and extending by {random.randint(2,5)} days.",
        }
    elif roll < accept_prob + 0.3:
        ext = random.randint(1, 3)
        return {
            "decision": "counter",
            "late_fee_waiver": False,
            "extension_days": ext,
            "counter_terms": f"Extend deadline by {ext} days but late fee remains.",
            "reasoning": "Willing to extend but cannot waive penalties.",
        }
    else:
        return {
            "decision": "reject",
            "late_fee_waiver": False,
            "extension_days": 0,
            "counter_terms": "",
            "reasoning": "Cannot accommodate renegotiation at this time. Full payment expected.",
        }


# ═══════════════════════════════════════════════════════
# ICL-BASED RISK AGENT
# ═══════════════════════════════════════════════════════

RISK_ICL_PROMPT = """You are the Risk Agent. Analyze the financial state and provide a risk memo in JSON format.
Use the following examples to guide your reasoning:

Example 1:
Input: Cash: ₹100000, Credit Used: ₹450000/₹500000, Market Stress: 0.8, Upcoming Risk: critical
Output: {{"risk_level": "critical", "credit_utilization": 90.0, "recommended_buffer": 50000.0, "threats": ["Near credit limit", "High market stress", "Critical upcoming events"], "recommendation": "reduce_spending", "reasoning": "Extreme liquidity crisis. Minimize all non-essential outflows."}}

Example 2:
Input: Cash: ₹800000, Credit Used: ₹0/₹500000, Market Stress: 0.1, Upcoming Risk: low
Output: {{"risk_level": "low", "credit_utilization": 0.0, "recommended_buffer": 200000.0, "threats": [], "recommendation": "maintain_buffer", "reasoning": "Financial position is strong with zero credit reliance."}}

Now analyze this state:
Input: Cash: ₹{cash:.0f}, Credit Used: ₹{credit_used:.0f}/₹{credit_limit:.0f}, Market Stress: {market_stress:.2f}, Upcoming Risk: {risk_level_hint}
Output:"""

def risk_agent_icl(cash: float, credit_used: float, credit_limit: float, world_hints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Risk Agent: Uses In-Context Learning (Few-Shot) to assess risk.
    Uses the unified model interface to support both API and Local HF models.
    """
    from server.client import get_model_response
    
    market_stress = world_hints.get("market_stress", 0.0)
    risk_level_hint = world_hints.get("upcoming_risk_level", "low")

    prompt = RISK_ICL_PROMPT.format(
        cash=cash,
        credit_used=credit_used,
        credit_limit=credit_limit,
        market_stress=market_stress,
        risk_level_hint=risk_level_hint
    )

    data = get_model_response(prompt, system_prompt=RISK_SYSTEM_PROMPT, response_format="json")
    
    if data and isinstance(data, dict):
        return data
    
    # Fallback to rule-based risk agent if LLM fails
    return risk_agent(cash, credit_used, credit_limit, world_hints)


# ═══════════════════════════════════════════════════════
# CONVENIENCE: Format all memos as strings for observation
# ═══════════════════════════════════════════════════════

def format_memo(agent_name: str, memo: Dict[str, Any]) -> str:
    """Convert structured memo to a human-readable string for the observation."""
    if agent_name == "Expenditure":
        critical = memo.get("critical_invoices", [])
        return (
            f"[{memo.get('recommended_action', 'UNKNOWN').upper()}] "
            f"Total liability: ₹{memo.get('total_liability', 0):.0f}. "
            f"Critical invoices: {critical if critical else 'None'}. "
            f"{memo.get('reasoning', '')}"
        )
    elif agent_name == "Revenue":
        return (
            f"[{memo.get('recommendation', 'UNKNOWN').upper()}] "
            f"Expected inflow: ₹{memo.get('total_expected_inflow', 0):.0f}. "
            f"3-day projection: ₹{memo.get('cash_projection_3day', 0):.0f}. "
            f"{memo.get('reasoning', '')}"
        )
    elif agent_name == "Risk":
        return (
            f"[{memo.get('risk_level', 'UNKNOWN').upper()}] "
            f"Credit util: {memo.get('credit_utilization', 0):.0f}%. "
            f"Buffer: ₹{memo.get('recommended_buffer', 0):.0f}. "
            f"Threats: {len(memo.get('threats', []))}. "
            f"{memo.get('reasoning', '')}"
        )
    return json.dumps(memo)

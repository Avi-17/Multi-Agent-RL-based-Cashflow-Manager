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

def expenditure_agent(invoices: list, cash: float, revenue_projection: float = 0.0) -> Dict[str, Any]:
    """
    Expenditure Agent: Prioritizes debt based on financial impact (Interest + Fees).
    """
    if not invoices:
        return {
            "priority_invoices": [],
            "total_debt": 0.0,
            "recommendation": "conserve_cash",
            "reasoning": "No outstanding invoices. Maintain liquidity."
        }

    scored_invoices = []
    total_debt = 0.0
    
    for inv in invoices:
        if inv.status == "paid": continue
        
        total_debt += inv.amount
        # Penalty = Late Fee (if due in <=1 day) + daily interest
        immediate_penalty = inv.late_fee if inv.due_in <= 1 else 0.0
        daily_interest = (inv.amount * inv.interest) / 30
        
        pain_score = immediate_penalty + daily_interest
        
        scored_invoices.append({
            "id": inv.id,
            "amount": inv.amount,
            "due_in": inv.due_in,
            "pain_score": round(pain_score, 2),
            "interest_rate": f"{inv.interest*100:.1f}%"
        })

    # Sort by pain score (highest first)
    scored_invoices.sort(key=lambda x: x["pain_score"], reverse=True)
    top_priority = scored_invoices[0] if scored_invoices else None
    
    if not top_priority:
        rec = "conserve_cash"
        reasoning = "All debt handled."
    elif top_priority["pain_score"] > 5000:
        rec = "pay_immediately"
        reasoning = f"Invoice {top_priority['id']} has a high penalty cost (₹{top_priority['pain_score']:.0f}). Pay immediately."
    elif float(top_priority["interest_rate"].replace('%','')) > 15.0:
        # Strategic Negotiation: Even if we have cash, negotiate with predatory vendors
        rec = "negotiate_or_defer"
        reasoning = f"Invoice {top_priority['id']} is from a predatory vendor ({top_priority['interest_rate']}). Strategically negotiate to delay this expensive debt."
    elif (cash + revenue_projection) < (total_debt * 1.2):
        rec = "negotiate_or_defer"
        reasoning = f"Total debt (₹{total_debt:.0f}) is too high relative to liquidity. Negotiate for time."
    else:
        rec = "pay_partial"
        reasoning = "Cash position is healthy. Pay high-interest items."

    return {
        "priority_queue": scored_invoices[:3],
        "total_outstanding": round(total_debt, 2),
        "suggested_payment": top_priority["amount"] if top_priority else 0.0,
        "recommendation": rec,
        "reasoning": reasoning
    }


def revenue_agent(receivables: list, invoices: list, cash: float, day: int, market_stress: float = 0.0) -> Dict[str, Any]:
    """
    Revenue Agent: Projects liquidity while adjusting for economic stress.
    """
    if not receivables and not invoices:
        return {
            "total_expected_inflow": 0.0,
            "net_3day_position": cash,
            "recommendation": "cash_sufficient",
            "reasoning": "Stable state, no activity."
        }

    # Apply "Market Haircut" to probabilities
    # High market stress makes customer payments less reliable
    stress_haircut = 1.0 - (market_stress * 0.5) 
    
    total_inflow = sum(r.amount * r.probability * stress_haircut for r in receivables)
    
    # Identify high-value targets for "Nudging"
    sorted_rec = sorted(receivables, key=lambda x: x.amount, reverse=True)
    high_value = [r.id for r in sorted_rec[:3]]
    
    # 3-day projection with stress adjustment
    inflow_3d = sum(r.amount * (r.probability * stress_haircut) for r in receivables if r.expected_in <= 3)
    outflow_3d = sum(i.amount for i in invoices if i.due_in <= 3 and i.status != "paid")
    
    net_position = (cash + inflow_3d) - outflow_3d
    
    # Strategic Buffer Recommendation
    credit_buffer_needed = abs(net_position) if net_position < 0 else 0.0
    
    if net_position < 0:
        rec = "cash_critical"
        reasoning = f"Economic stress ({market_stress:.1f}) is impacting collections. Deficit of ₹{abs(net_position):.0f} expected. Draw credit immediately."
    elif net_position < (outflow_3d * 0.5):
        rec = "cash_tight"
        reasoning = "Liquidity is thin. High-value collections (IDs: " + ", ".join(high_value) + ") are critical."
    else:
        rec = "cash_sufficient"
        reasoning = f"Net position of ₹{net_position:.0f} is safe despite market stress."

    return {
        "expected_inflow_total": round(total_inflow, 2),
        "high_value_receivables": high_value,
        "net_3day_position": round(net_position, 2),
        "credit_buffer_needed": round(credit_buffer_needed, 2),
        "recommendation": rec,
        "reasoning": reasoning,
    }


def risk_agent(cash: float, total_debt: float, credit_used: float, credit_limit: float, world_hints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Risk Agent: Evaluates both external (market) and internal (debt) threats.
    """
    market_stress = world_hints.get("market_stress", 0.0)
    risk_level_hint = world_hints.get("upcoming_risk_level", "low")
    vendor_sentiment = world_hints.get("vendor_sentiment", {})

    threats = []
    
    # Internal Threat: Liquidity vs Debt
    debt_ratio = total_debt / (cash + 1.0)
    if debt_ratio > 3.0:
        threats.append("Critical Debt Overload")
    elif debt_ratio > 1.5:
        threats.append("High Debt-to-Cash Ratio")

    # Internal Threat: Credit Exhaustion
    credit_utilization = credit_used / (credit_limit + 1.0)
    if credit_utilization > 0.85:
        threats.append("Credit Line Exhausted")
    elif credit_utilization > 0.6:
        threats.append("Low Credit Availability")

    # External Threats
    if market_stress > 0.4:
        threats.append(f"Market Volatility (Level: {market_stress:.1f})")
    if risk_level_hint == "critical":
        threats.append("Upcoming Hidden Financial Threat Detected")
    elif risk_level_hint == "elevated":
        threats.append("Unspecified Hidden Threats Detected")

    # Final Risk Assessment
    if any("Critical" in t or "Exhausted" in t for t in threats) or len(threats) >= 3:
        risk_level = "critical"
        rec = "emergency_halt"
        survival_target = total_debt * 0.8
    elif len(threats) >= 2 or market_stress > 0.5:
        risk_level = "elevated"
        rec = "conserve_and_buffer"
        survival_target = total_debt * 0.5
    else:
        risk_level = "low"
        rec = "normal_operations"
        survival_target = total_debt * 0.2

    reasoning = f"Overall risk is {risk_level}. Detected {len(threats)} threats. Survival requires ₹{survival_target:.0f} buffer."

    return {
        "risk_level": risk_level,
        "active_threats": threats,
        "credit_utilization": f"{credit_utilization*100:.0f}%",
        "survival_cash_target": round(survival_target, 2),
        "recommendation": rec,
        "reasoning": reasoning
    }


def vendor_agent(vendor_profile: Dict[str, Any], invoice: Any, vendor_mood: float = 0.0, trust_score: float = 0.5) -> Dict[str, Any]:
    """
    Vendor Agent: Decides whether to accept or reject a negotiation request.
    Logic: (Trust + Mood) vs (Amount + Risk)
    """
    # Success Probability Base
    success_prob = 0.4 + (trust_score * 0.4) + (vendor_mood * 0.3)
    
    # "Large Bill" Friction: Every ₹100k reduces success chance by 10%
    amount_penalty = (invoice.amount / 100000) * 0.1
    success_prob -= amount_penalty
    
    # "Bad Vendor" check: If the vendor is a high-interest predator, they are less likely to negotiate
    is_predatory = invoice.interest > 0.15
    if is_predatory:
        success_prob -= 0.2
        
    success_prob = max(0.05, min(0.95, success_prob))
    
    roll = random.random()
    accepted = roll < success_prob
    
    if accepted:
        msg = f"Vendor accepted. Terms: 5-day extension."
    else:
        if vendor_mood < -0.2:
            msg = "Vendor is frustrated with payment delays. Request flatly REJECTED."
        elif amount_penalty > 0.3:
            msg = "The amount is too large for an automated extension. Request REJECTED."
        else:
            msg = "Negotiation failed. Standard terms apply."

    return {
        "accepted": accepted,
        "success_probability": round(success_prob, 2),
        "vendor_message": msg,
        "is_predatory": is_predatory,
        "extension_days": 5 if accepted else 0
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
        return (
            f"[{memo.get('recommendation', 'UNKNOWN').upper()}] "
            f"Total Debt: ₹{memo.get('total_outstanding', 0):.0f}. "
            f"Top Priority: {memo.get('priority_queue', [{}])[0].get('id', 'None')}. "
            f"{memo.get('reasoning', '')}"
        )
    elif agent_name == "Revenue":
        return (
            f"[{memo.get('recommendation', 'UNKNOWN').upper()}] "
            f"Inflow: ₹{memo.get('expected_inflow_total', 0):.0f}. "
            f"Net Position: ₹{memo.get('net_3day_position', 0):.0f}. "
            f"{memo.get('reasoning', '')}"
        )
    elif agent_name == "Risk":
        return (
            f"[{memo.get('risk_level', 'UNKNOWN').upper()}] "
            f"Credit: {memo.get('credit_utilization', 'N/A')}. "
            f"Survival Target: ₹{memo.get('survival_cash_target', 0):.0f}. "
            f"Threats: {len(memo.get('active_threats', []))}. "
            f"{memo.get('reasoning', '')}"
        )
    return "No memo available."
    return json.dumps(memo)

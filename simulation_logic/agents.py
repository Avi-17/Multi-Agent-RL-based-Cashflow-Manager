"""
Agents for the Simulation Logic module.

Uses the REAL agents from server/agents.py with adapter functions
to bridge the data model differences between SimInvoice/SimReceivable
and the server's Invoice/Receivable models.

Also uses the expert rule-based CFO from server/client.py.
"""

import random
from typing import List, Dict, Any, Optional
from simulation_logic.models import SimState, SimAction, SimInvoice

# ─────────────────────────────────────────────
# Import the real agents from the server
# ─────────────────────────────────────────────

from server.agents import (
    expenditure_agent as _server_expenditure_agent,
    revenue_agent as _server_revenue_agent,
    risk_agent as _server_risk_agent,
    format_memo,
)
from models import Invoice, Receivable


# ─────────────────────────────────────────────
# Adapters: Convert SimInvoice/SimReceivable → Invoice/Receivable
# ─────────────────────────────────────────────

def _sim_invoice_to_invoice(sim_inv: SimInvoice) -> Invoice:
    """Convert a SimInvoice to a server Invoice for agent compatibility."""
    return Invoice(
        id=sim_inv.id,
        vendor_id=sim_inv.vendor,           # SimInvoice uses 'vendor', Invoice uses 'vendor_id'
        amount=sim_inv.amount,
        due_in=sim_inv.due_in,
        late_fee=sim_inv.late_fee,
        min_payment=sim_inv.amount * 0.3,   # SimInvoice doesn't have min_payment, estimate 30%
        interest=sim_inv.interest_rate,      # SimInvoice uses 'interest_rate', Invoice uses 'interest'
        status=sim_inv.status,
    )


def _sim_receivable_to_receivable(sim_rec) -> Receivable:
    """Convert a SimReceivable to a server Receivable for agent compatibility."""
    return Receivable(
        id=sim_rec.id,
        customer_id=sim_rec.customer,        # SimReceivable uses 'customer', Receivable uses 'customer_id'
        amount=sim_rec.amount,
        expected_in=sim_rec.arrives_on_day,   # SimReceivable uses 'arrives_on_day', Receivable uses 'expected_in'
        probability=sim_rec.probability,
    )


# ─────────────────────────────────────────────
# EXPENDITURE ADVISOR (uses server's expenditure_agent)
# ─────────────────────────────────────────────

def expenditure_advisor(state: SimState) -> str:
    """
    Calls the server's expenditure_agent and formats the result as a string memo.
    """
    # Convert SimInvoices → Invoices
    invoices = [_sim_invoice_to_invoice(inv) for inv in state.active_invoices]

    # Calculate revenue projection from receivables
    revenue_projection = sum(r.amount * r.probability for r in state.receivables)

    # Call the real agent
    memo = _server_expenditure_agent(invoices, state.cash, revenue_projection)

    # Format as readable string using server's format_memo
    result = format_memo("Expenditure", memo)

    # Append upcoming invoice warning (simulation-specific feature)
    if state.upcoming_invoice_count > 0:
        result += f"\n⚠️ {state.upcoming_invoice_count} more invoices arriving soon — save some cash!"

    return result


# ─────────────────────────────────────────────
# REVENUE ADVISOR (uses server's revenue_agent)
# ─────────────────────────────────────────────

def revenue_advisor(state: SimState) -> str:
    """
    Calls the server's revenue_agent and formats the result as a string memo.
    """
    # Convert to server types
    receivables = [_sim_receivable_to_receivable(r) for r in state.receivables]
    invoices = [_sim_invoice_to_invoice(inv) for inv in state.active_invoices]

    # Call the real agent
    memo = _server_revenue_agent(receivables, invoices, state.cash, state.day)

    # Format as readable string
    return format_memo("Revenue", memo)


# ─────────────────────────────────────────────
# RISK ADVISOR (uses server's risk_agent)
# ─────────────────────────────────────────────

def risk_advisor(state: SimState) -> str:
    """
    Calls the server's risk_agent and formats the result as a string memo.
    """
    total_debt = sum(inv.amount for inv in state.active_invoices)

    # Build world_hints dict (simulation doesn't have a WorldModel, so we approximate)
    world_hints = {
        "market_stress": 0.0,
        "upcoming_risk_level": "low",
        "vendor_sentiment": {},
    }

    # If there are overdue invoices, signal elevated risk
    if len(state.overdue_invoices) >= 2:
        world_hints["upcoming_risk_level"] = "critical"
    elif len(state.overdue_invoices) >= 1:
        world_hints["upcoming_risk_level"] = "elevated"

    # Call the real agent
    memo = _server_risk_agent(state.cash, total_debt, state.credit_used, state.credit_limit, world_hints)

    # Format as readable string
    result = format_memo("Risk", memo)

    # Append upcoming invoice pressure (simulation-specific)
    if state.upcoming_invoice_count > 0:
        result += f"\n📋 {state.upcoming_invoice_count} invoices are coming soon. Maintain a cash buffer."

    return result


# ─────────────────────────────────────────────
# CFO DECISION AGENT (adapted from server/client.py _cfo_rule_decide)
# ─────────────────────────────────────────────

def cfo_decide(state: SimState) -> List[SimAction]:
    """
    Rule-based CFO that decides what to do each day.

    Adapted from server/client.py's _cfo_rule_decide, but handles
    multiple invoices per day (the server handles one per step).

    Strategy (same as the server's expert CFO):
      1. Pay critical invoices (overdue / due today) — sorted by (late_fee, interest)
      2. Draw credit if needed for critical debt
      3. Negotiate if can't afford critical debt
      4. Pay smallest invoices to simplify the ledger
      5. Defer everything else
    """
    actions = []
    available_cash = state.cash
    credit_available = state.credit_limit - state.credit_used

    # Keep a buffer for upcoming invoices
    buffer = 1000 * state.upcoming_invoice_count if state.upcoming_invoice_count > 0 else 0

    # ── Step 1: Identify critical invoices (overdue or due today/tomorrow) ──
    critical = sorted(
        [inv for inv in state.active_invoices if inv.due_in <= 1 or inv.status == "overdue"],
        key=lambda x: (x.late_fee, x.interest_rate),
        reverse=True,
    )

    # ── Step 2: Handle critical invoices ──
    for inv in critical:
        if inv.status == "paid":
            continue

        spendable = available_cash - buffer

        if spendable >= inv.amount:
            # Pay in full
            actions.append(SimAction(
                type="pay",
                invoice_id=inv.id,
                amount=inv.amount,
                reasoning=f"Paying critical invoice {inv.id} (₹{inv.amount:,.0f}) to avoid penalties",
            ))
            available_cash -= inv.amount

        elif credit_available >= inv.amount and spendable + credit_available >= inv.amount:
            # Draw credit first, then pay
            draw_amount = inv.amount - spendable
            actions.append(SimAction(
                type="credit",
                amount=draw_amount,
                reasoning=f"Drawing ₹{draw_amount:,.0f} credit to pay urgent debt {inv.id}",
            ))
            available_cash += draw_amount
            credit_available -= draw_amount

            actions.append(SimAction(
                type="pay",
                invoice_id=inv.id,
                amount=inv.amount,
                reasoning=f"Paying critical {inv.id} with credit funds",
            ))
            available_cash -= inv.amount

        elif spendable >= inv.amount * 0.5:
            # Partial payment to reduce damage
            pay_amount = round(spendable * 0.8, 2)
            actions.append(SimAction(
                type="partial",
                invoice_id=inv.id,
                amount=pay_amount,
                reasoning=f"Partial on critical {inv.id}: ₹{pay_amount:,.0f} to reduce damage",
            ))
            available_cash -= pay_amount

        else:
            # Can't afford it — negotiate
            actions.append(SimAction(
                type="defer",
                invoice_id=inv.id,
                reasoning=f"Cannot afford critical {inv.id} (₹{inv.amount:,.0f}). No cash or credit.",
            ))

    # ── Step 3: Pay smallest non-critical invoices to simplify ledger ──
    remaining = [
        inv for inv in state.active_invoices
        if inv.status != "paid" and inv.id not in {a.invoice_id for a in actions}
    ]
    remaining.sort(key=lambda x: x.amount)

    for inv in remaining:
        spendable = available_cash - buffer
        if spendable >= inv.amount and inv.due_in <= 3:
            actions.append(SimAction(
                type="pay",
                invoice_id=inv.id,
                amount=inv.amount,
                reasoning=f"Paying small invoice {inv.id} (₹{inv.amount:,.0f}) to simplify ledger",
            ))
            available_cash -= inv.amount
        else:
            actions.append(SimAction(
                type="defer",
                invoice_id=inv.id,
                reasoning=f"Deferring {inv.id} — due in {inv.due_in} days, preserving cash",
            ))

    return actions

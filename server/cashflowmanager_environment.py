"""
Core Simulation Engine.
"""

import random
from typing import List, Dict, Any

from models import (
    CashflowmanagerObservation as State, 
    Invoice, Receivable, IncomingInvoice,
    CashflowmanagerAction, DayLog, SimulationResult,
)
from server.data_generator import generate_scenario
from server.agents import expenditure_agent, revenue_agent, risk_agent, format_memo
from server.reward import compute_day_reward


# ─────────────────────────────────────────────
# Advisors & CFO Wrappers
# ─────────────────────────────────────────────

def expenditure_advisor(state: State) -> str:
    revenue_projection = sum(r.amount * r.probability for r in state.receivables)
    memo = expenditure_agent(state.active_invoices, state.cash, revenue_projection)
    result = format_memo("Expenditure", memo)
    if state.upcoming_invoice_count > 0:
        result += f"\n⚠️ {state.upcoming_invoice_count} more invoices arriving soon — save some cash!"
    return result

def revenue_advisor(state: State) -> str:
    memo = revenue_agent(state.receivables, state.active_invoices, state.cash, state.day)
    return format_memo("Revenue", memo)

def risk_advisor(state: State) -> str:
    total_debt = sum(inv.amount for inv in state.active_invoices)
    world_hints = {
        "market_stress": 0.0,
        "upcoming_risk_level": "low",
        "vendor_sentiment": {},
    }
    if len(state.overdue_invoices) >= 2:
        world_hints["upcoming_risk_level"] = "critical"
    elif len(state.overdue_invoices) >= 1:
        world_hints["upcoming_risk_level"] = "elevated"

    memo = risk_agent(state.cash, total_debt, state.credit_used, state.credit_limit, world_hints)
    result = format_memo("Risk", memo)
    if state.upcoming_invoice_count > 0:
        result += f"\n📋 {state.upcoming_invoice_count} invoices are coming soon. Maintain a cash buffer."
    return result

def cfo_decide(state: State) -> List[CashflowmanagerAction]:
    """
    Rule-based CFO that decides what to do each day.
    """
    actions = []
    available_cash = state.cash
    credit_available = state.credit_limit - state.credit_used
    buffer = 1000 * state.upcoming_invoice_count if state.upcoming_invoice_count > 0 else 0

    critical = sorted(
        [inv for inv in state.active_invoices if inv.due_in <= 1 or inv.status == "overdue"],
        key=lambda x: (x.late_fee, x.interest),
        reverse=True,
    )

    for inv in critical:
        if inv.status == "paid":
            continue

        spendable = available_cash - buffer

        if spendable >= inv.amount:
            actions.append(CashflowmanagerAction(
                type="pay",
                invoice_id=inv.id,
                amount=inv.amount,
                memo=f"Paying critical invoice {inv.id} (₹{inv.amount:,.0f}) to avoid penalties",
            ))
            available_cash -= inv.amount

        elif credit_available >= inv.amount and spendable + credit_available >= inv.amount:
            draw_amount = inv.amount - spendable
            actions.append(CashflowmanagerAction(
                type="credit",
                amount=draw_amount,
                memo=f"Drawing ₹{draw_amount:,.0f} credit to pay urgent debt {inv.id}",
            ))
            available_cash += draw_amount
            credit_available -= draw_amount

            actions.append(CashflowmanagerAction(
                type="pay",
                invoice_id=inv.id,
                amount=inv.amount,
                memo=f"Paying critical {inv.id} with credit funds",
            ))
            available_cash -= inv.amount

        elif spendable >= inv.amount * 0.5:
            pay_amount = round(spendable * 0.8, 2)
            actions.append(CashflowmanagerAction(
                type="partial",
                invoice_id=inv.id,
                amount=pay_amount,
                memo=f"Partial on critical {inv.id}: ₹{pay_amount:,.0f} to reduce damage",
            ))
            available_cash -= pay_amount
        else:
            actions.append(CashflowmanagerAction(
                type="defer",
                invoice_id=inv.id,
                memo=f"Cannot afford critical {inv.id} (₹{inv.amount:,.0f}). No cash or credit.",
            ))

    remaining = [
        inv for inv in state.active_invoices
        if inv.status != "paid" and inv.id not in {a.invoice_id for a in actions}
    ]
    remaining.sort(key=lambda x: x.amount)

    for inv in remaining:
        spendable = available_cash - buffer
        if spendable >= inv.amount and inv.due_in <= 3:
            actions.append(CashflowmanagerAction(
                type="pay",
                invoice_id=inv.id,
                amount=inv.amount,
                memo=f"Paying small invoice {inv.id} (₹{inv.amount:,.0f}) to simplify ledger",
            ))
            available_cash -= inv.amount
        else:
            actions.append(CashflowmanagerAction(
                type="defer",
                invoice_id=inv.id,
                memo=f"Deferring {inv.id} — due in {inv.due_in} days, preserving cash",
            ))

    return actions


# ─────────────────────────────────────────────
# Core Engine
# ─────────────────────────────────────────────

def init_simulation(
    difficulty: str = "medium",
    sim_window: int = 7,
    seed: int = 42,
) -> tuple:
    scenario = generate_scenario(difficulty=difficulty, sim_window=sim_window, seed=seed)

    state = State(
        day=0,
        cash=float(scenario["company"]["starting_cash"]),
        credit_used=0.0,
        credit_limit=float(scenario["company"]["credit_limit"]),
        active_invoices=[Invoice(**inv) for inv in scenario["initial_invoices"]],
        paid_invoices=[],
        partially_paid_invoices=[],
        overdue_invoices=[],
        receivables=[Receivable(**rec) for rec in scenario["initial_receivables"]],
        upcoming_invoice_count=len(scenario["incoming_invoices"]),
        metadata={}
    )

    incoming_invoices = [IncomingInvoice(**inc) for inc in scenario["incoming_invoices"]]

    return state, incoming_invoices


def step_one_day(state: State, incoming_invoices: List[IncomingInvoice]) -> DayLog:
    day = state.day + 1
    state.day = day

    day_log = DayLog(
        day=day,
        opening_cash=state.cash,
        opening_credit_used=state.credit_used,
        active_invoice_count=len(state.active_invoices),
        overdue_invoice_count=len(state.overdue_invoices),
    )

    newly_activated = _activate_incoming(state, incoming_invoices, day)
    for inv in newly_activated:
        day_log.events.append(
            f"📨 New invoice arrived: {inv.id} from {inv.vendor_id} — ₹{inv.amount:,.0f} due in {inv.due_in} days"
        )

    _age_invoices(state, day_log)

    day_log.active_invoice_count = len(state.active_invoices)
    day_log.overdue_invoice_count = len(state.overdue_invoices)

    revenue_today = _collect_receivables(state, day, day_log)
    day_log.revenue_collected = revenue_today

    day_log.advisor_memos = {
        "Expenditure": expenditure_advisor(state),
        "Revenue": revenue_advisor(state),
        "Risk": risk_advisor(state),
    }

    actions = cfo_decide(state)
    day_log.actions = actions

    paid_today = _apply_actions(state, actions, day_log)
    day_log.invoices_paid_today = paid_today

    fees, interest = _apply_daily_charges(state, day_log)
    day_log.late_fees_incurred = fees
    day_log.interest_incurred = interest

    day_log.reward = compute_day_reward(
        cash=state.cash,
        invoices_paid=paid_today,
        late_fees=fees,
        interest=interest,
        credit_used=state.credit_used,
        credit_limit=state.credit_limit,
        overdue_count=len(state.overdue_invoices),
        total_active=len(state.active_invoices),
    )

    day_log.closing_cash = state.cash
    day_log.closing_credit_used = state.credit_used

    return day_log


def run_simulation(
    difficulty: str = "medium",
    sim_window: int = 7,
    seed: int = 42,
) -> SimulationResult:
    state, incoming_invoices = init_simulation(difficulty, sim_window, seed)

    result = SimulationResult(
        difficulty=difficulty,
        sim_window=sim_window,
        seed=seed,
    )

    total_reward = 0.0

    for _ in range(sim_window):
        day_log = step_one_day(state, incoming_invoices)
        result.days.append(day_log)
        total_reward += day_log.reward

    result.final_cash = state.cash
    result.final_credit_used = state.credit_used
    result.total_invoices = len(state.paid_invoices) + len(state.active_invoices)
    result.invoices_paid = len(state.paid_invoices)
    result.invoices_overdue = len(state.overdue_invoices)
    result.total_late_fees = sum(d.late_fees_incurred for d in result.days)
    result.total_interest = sum(d.interest_incurred for d in result.days)
    result.total_revenue_collected = sum(d.revenue_collected for d in result.days)
    result.total_reward = round(total_reward, 2)

    return result


def _activate_incoming(
    state: State,
    incoming_invoices: List[IncomingInvoice],
    day: int,
) -> List[Invoice]:
    activated = []
    remaining = []

    for inc in incoming_invoices:
        if inc.appears_on_day == day:
            new_inv = Invoice(
                id=inc.id,
                vendor_id=inc.vendor_id,
                amount=inc.hidden_amount,
                due_in=inc.hidden_due_in,
                late_fee=inc.hidden_late_fee,
                interest=inc.hidden_interest,
                min_payment=inc.hidden_amount * 0.3,
                status="unpaid",
            )
            state.active_invoices.append(new_inv)
            activated.append(new_inv)
        else:
            remaining.append(inc)

    incoming_invoices.clear()
    incoming_invoices.extend(remaining)
    state.upcoming_invoice_count = len(remaining)

    return activated


def _age_invoices(state: State, day_log: DayLog):
    for inv in state.active_invoices:
        if inv.status == "paid":
            continue
        inv.due_in -= 1

        if inv.due_in < 0 and inv.status != "overdue":
            inv.status = "overdue"
            if inv not in state.overdue_invoices:
                state.overdue_invoices.append(inv)
            day_log.events.append(
                f"⏰ Invoice {inv.id} from {inv.vendor_id} is now OVERDUE! (₹{inv.amount:,.0f})"
            )


def _collect_receivables(state: State, day: int, day_log: DayLog) -> float:
    collected = 0.0
    remaining = []

    for rec in state.receivables:
        if rec.expected_in == day:
            if random.random() < rec.probability:
                state.cash += rec.amount
                collected += rec.amount
                day_log.events.append(
                    f"💰 Received ₹{rec.amount:,.0f} from {rec.customer_id}"
                )
            else:
                day_log.events.append(
                    f"❌ Payment from {rec.customer_id} (₹{rec.amount:,.0f}) FAILED to arrive"
                )
        else:
            remaining.append(rec)

    state.receivables = remaining
    return collected


def _apply_actions(state: State, actions: List[CashflowmanagerAction], day_log: DayLog) -> int:
    paid_count = 0

    for action in actions:
        if action.type == "pay" and action.invoice_id:
            inv = _find_invoice(state, action.invoice_id)
            if inv and inv.status != "paid" and action.amount:
                pay_amount = min(action.amount, inv.amount, state.cash)
                if pay_amount <= 0:
                    continue

                state.cash -= pay_amount
                inv.amount -= pay_amount

                if inv.amount <= 0:
                    inv.amount = 0
                    inv.status = "paid"
                    state.paid_invoices.append(inv)
                    state.active_invoices = [i for i in state.active_invoices if i.id != inv.id]
                    state.overdue_invoices = [i for i in state.overdue_invoices if i.id != inv.id]
                    state.partially_paid_invoices = [i for i in state.partially_paid_invoices if i.id != inv.id]
                    paid_count += 1
                    day_log.events.append(f"✅ Paid {inv.id} in full (₹{pay_amount:,.0f})")

        elif action.type == "partial" and action.invoice_id:
            inv = _find_invoice(state, action.invoice_id)
            if inv and inv.status != "paid" and action.amount:
                pay_amount = min(action.amount, inv.amount, state.cash)
                if pay_amount <= 0:
                    continue

                state.cash -= pay_amount
                inv.amount -= pay_amount

                if inv.amount <= 0:
                    inv.amount = 0
                    inv.status = "paid"
                    state.paid_invoices.append(inv)
                    state.active_invoices = [i for i in state.active_invoices if i.id != inv.id]
                    state.overdue_invoices = [i for i in state.overdue_invoices if i.id != inv.id]
                    state.partially_paid_invoices = [i for i in state.partially_paid_invoices if i.id != inv.id]
                    paid_count += 1
                    day_log.events.append(f"✅ Partial payment completed {inv.id} (₹{pay_amount:,.0f})")
                else:
                    inv.status = "partial"
                    if inv not in state.partially_paid_invoices:
                        state.partially_paid_invoices.append(inv)
                    day_log.events.append(
                        f"💳 Partial payment on {inv.id}: ₹{pay_amount:,.0f} — ₹{inv.amount:,.0f} remaining"
                    )

        elif action.type == "credit":
            credit_available = state.credit_limit - state.credit_used
            draw = min(action.amount or 0, credit_available)
            if draw > 0:
                state.cash += draw
                state.credit_used += draw
                day_log.events.append(f"🏦 Drew ₹{draw:,.0f} from credit line")

        elif action.type == "defer":
            pass

    return paid_count


def _apply_daily_charges(state: State, day_log: DayLog) -> tuple:
    total_fees = 0.0
    total_interest = 0.0

    for inv in state.active_invoices:
        if inv.status == "paid":
            continue

        interest = round(inv.amount * inv.interest, 2)
        inv.amount += interest
        total_interest += interest

        if inv.due_in < 0:
            inv.amount += inv.late_fee
            total_fees += inv.late_fee

    return total_fees, total_interest


def _find_invoice(state: State, invoice_id: str):
    for inv in state.active_invoices:
        if inv.id == invoice_id:
            return inv
    return None

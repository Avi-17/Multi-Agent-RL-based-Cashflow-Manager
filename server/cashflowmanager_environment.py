"""
Core Simulation Engine.
"""

import random
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from models import (
    CashflowmanagerObservation as State,
    Invoice, Receivable, IncomingInvoice,
    CashflowmanagerAction, DayLog, SimulationResult,
)
from server.data_generator import generate_scenario
from server.agents import (
    expenditure_agent, revenue_agent, risk_agent, format_memo,
    cfo_decide, cfo_decide_with_metadata
)
from server.reward import CashflowRubric
from server.scoring import compute_simulation_score


# Initialize the Rubric system
reward_rubric = CashflowRubric()
CFO_CONFIDENCE_THRESHOLD = float(os.environ.get("CFO_CONFIDENCE_THRESHOLD", "0.85"))

# ─────────────────────────────────────────────
# Advisors & CFO Wrappers
# ─────────────────────────────────────────────

def expenditure_advisor(state, past_logs=None):
    memo = expenditure_agent(state, past_logs or [])
    result = format_memo("Expenditure", memo)
    if state.upcoming_invoice_count > 0:
        result += f"\n⚠️ {state.upcoming_invoice_count} more invoices arriving soon — save some cash!"
    return result

def revenue_advisor(state, past_logs=None):
    memo = revenue_agent(state, past_logs or [])
    return format_memo("Revenue", memo)

def risk_advisor(state, past_logs=None):
    memo = risk_agent(state, past_logs or [])
    result = format_memo("Risk", memo)
    if state.upcoming_invoice_count > 0:
        result += f"\n📋 {state.upcoming_invoice_count} invoices are coming soon. Maintain a cash buffer."
    return result


# ─────────────────────────────────────────────
# Heuristic Fast Path (no API call)
# ─────────────────────────────────────────────

def _try_fast_path(state: State):
    """
    Attempt to generate rule-based actions without calling the LLM.
    Returns a list of CashflowmanagerAction if the situation is simple enough,
    or None if the LLM should handle it.
    
    Conditions for fast path:
      - No overdue invoices
      - Total unpaid amount <= cash available
      - 3 or fewer unpaid invoices
    """
    unpaid = [inv for inv in state.active_invoices if inv.status != "paid"]
    
    if not unpaid:
        return []  # Nothing to do
    
    overdue = [inv for inv in unpaid if inv.due_in < 0]
    if overdue:
        return None  # Complex — need LLM

    total_owed = sum(inv.amount for inv in unpaid)
    if total_owed > state.cash:
        return None  # Not enough cash to pay everything — need LLM to prioritize

    if len(unpaid) > 3:
        return None  # Too many invoices — need LLM

    # Simple case: pay invoices due soon, defer the rest
    actions = []
    remaining_cash = state.cash
    for inv in sorted(unpaid, key=lambda x: x.due_in):
        if inv.due_in <= 2 and remaining_cash >= inv.amount:
            actions.append(CashflowmanagerAction(
                type="pay", invoice_id=inv.id, amount=inv.amount,
                memo=f"Fast-path: pay (due in {inv.due_in}d)"
            ))
            remaining_cash -= inv.amount
        else:
            actions.append(CashflowmanagerAction(
                type="defer", invoice_id=inv.id, amount=0.0,
                memo=f"Fast-path: defer (due in {inv.due_in}d)"
            ))
    return actions


# ─────────────────────────────────────────────
# Core Engine
# ─────────────────────────────────────────────

def init_simulation(
    difficulty: str = "medium",
    sim_window: int = 3,
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


def step_one_day(state: State, incoming_invoices: List[IncomingInvoice], past_logs: List[DayLog] = None) -> DayLog:
    day = state.day + 1
    state.day = day
    if past_logs is None:
        past_logs = []

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

    # ── CFO Gating Flow ──
    # Step 1: Evaluate state complexity heuristically (NO API call)
    day_log.advisor_memos = {}
    fast_actions = _try_fast_path(state)

    if fast_actions is not None:
        # Fast path — trivially simple day, skip ALL API calls
        day_log.advisor_memos["Routing"] = "[FAST_PATH] Simple state — rule-based actions, no LLM needed."
        day_log.actions = fast_actions
    else:
        # Complex day — run the three advisors in parallel on separate API keys
        # (see EXPENDITURE/REVENUE/RISK_KEY_INDEX in agents.py). Each advisor
        # hits its own TPM bucket, so concurrency doesn't cause rate-limit
        # contention. CFO runs sequentially after, on the Risk key.
        day_log.advisor_memos["Routing"] = "[FULL_PATH] Complex state — consulting advisors (parallel) + CFO."

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                "Expenditure": executor.submit(expenditure_advisor, state, past_logs),
                "Revenue": executor.submit(revenue_advisor, state, past_logs),
                "Risk": executor.submit(risk_advisor, state, past_logs),
            }
            for name, future in futures.items():
                try:
                    day_log.advisor_memos[name] = future.result()
                except Exception as e:
                    day_log.advisor_memos[name] = f"[ERROR] {name} advisor failed: {e}"

        day_log.actions = cfo_decide(state, day_log.advisor_memos, past_logs)

    paid_today = _apply_actions(state, day_log.actions or [], day_log)
    day_log.invoices_paid_today = paid_today

    fees, interest = _apply_daily_charges(state, day_log)
    day_log.late_fees_incurred += fees
    day_log.interest_incurred = interest

    obs = {
        "state": state,
        "day_log": day_log
    }
    
    day_log.reward = reward_rubric(
        action=day_log.actions,
        observation=obs
    )

    day_log.closing_cash = state.cash
    day_log.closing_credit_used = state.credit_used

    return day_log


def run_simulation(
    difficulty: str = "medium",
    sim_window: int = 3,
    seed: int = 42,
) -> SimulationResult:
    state, incoming_invoices = init_simulation(difficulty, sim_window, seed)

    result = SimulationResult(
        difficulty=difficulty,
        sim_window=sim_window,
        seed=seed,
    )

    total_reward = 0.0
    past_logs = []

    for _ in range(sim_window):
        day_log = step_one_day(state, incoming_invoices, past_logs)
        result.days.append(day_log)
        past_logs.append(day_log)
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

    # Compute normalized evaluation score
    eval_result = compute_simulation_score(result)
    result.score = eval_result["score"]
    result.score_breakdown = eval_result["breakdown"]
    result.grade = eval_result["grade"]

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
            inv.amount += inv.late_fee
            day_log.late_fees_incurred += inv.late_fee
            if inv not in state.overdue_invoices:
                state.overdue_invoices.append(inv)
            day_log.events.append(
                f"⏰ Invoice {inv.id} from {inv.vendor_id} is now OVERDUE! (₹{inv.amount:,.0f} incl. fee)"
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
                rec.expected_in += 1
                remaining.append(rec)
        else:
            remaining.append(rec)

    state.receivables = remaining
    return collected


def _apply_actions(state: State, actions: List[CashflowmanagerAction], day_log: DayLog) -> int:
    paid_count = 0

    for action in actions:
        if action.type == "pay" and action.invoice_id:
            inv = _find_invoice(state, action.invoice_id)
            if inv and inv.status != "paid":
                amount_to_pay = action.amount if action.amount else inv.amount
                pay_amount = min(amount_to_pay, inv.amount, state.cash)
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
            if inv and inv.status != "paid":
                amount_to_pay = action.amount if action.amount else inv.amount
                pay_amount = min(amount_to_pay, inv.amount, state.cash)
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

    return total_fees, total_interest


def _find_invoice(state: State, invoice_id: str):
    for inv in state.active_invoices:
        if inv.id == invoice_id:
            return inv
    return None

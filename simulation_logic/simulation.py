"""
Core Simulation Engine for the Simulation Logic module.

Runs the entire simulation window in one shot.
For each day:
  1. Activate any incoming invoices scheduled for today
  2. Age invoices (decrement due_in, mark newly overdue)
  3. Collect receivables that are due today
  4. Run advisors → produce memos
  5. CFO decides actions for all active invoices
  6. Apply actions (pay, partial, defer, credit)
  7. Apply daily interest and late fees on remaining invoices
  8. Compute reward
  9. Log everything into a DayLog

Returns a SimulationResult with all day logs and a final summary.
"""

import random
from typing import List, Dict, Any
from simulation_logic.models import (
    SimState, SimInvoice, SimReceivable, IncomingInvoice,
    SimAction, DayLog, SimulationResult,
)
from simulation_logic.data_generator import generate_scenario
from simulation_logic.agents import expenditure_advisor, revenue_advisor, risk_advisor, cfo_decide
from simulation_logic.reward import compute_day_reward


def init_simulation(
    difficulty: str = "medium",
    sim_window: int = 7,
    seed: int = 42,
) -> tuple:
    """
    Initialize a simulation without running any days.

    Returns:
        (state, incoming_invoices, scenario_meta) — ready for step_one_day()
    """
    scenario = generate_scenario(difficulty=difficulty, sim_window=sim_window, seed=seed)

    state = SimState(
        day=0,
        cash=float(scenario["company"]["starting_cash"]),
        credit_used=0.0,
        credit_limit=float(scenario["company"]["credit_limit"]),
        active_invoices=[SimInvoice(**inv) for inv in scenario["invoices"]],
        paid_invoices=[],
        partially_paid_invoices=[],
        overdue_invoices=[],
        receivables=[SimReceivable(**rec) for rec in scenario["receivables"]],
        upcoming_invoice_count=len(scenario["incoming_invoices"]),
    )

    incoming_invoices = [IncomingInvoice(**inc) for inc in scenario["incoming_invoices"]]

    return state, incoming_invoices


def step_one_day(state: SimState, incoming_invoices: List[IncomingInvoice]) -> DayLog:
    """
    Advance the simulation by exactly one day.

    Mutates state in-place and returns a DayLog for that day.
    """
    day = state.day + 1
    state.day = day

    day_log = DayLog(
        day=day,
        opening_cash=state.cash,
        opening_credit_used=state.credit_used,
        active_invoice_count=len(state.active_invoices),
        overdue_invoice_count=len(state.overdue_invoices),
    )

    # Phase 1: Activate incoming invoices
    newly_activated = _activate_incoming(state, incoming_invoices, day)
    for inv in newly_activated:
        day_log.events.append(
            f"📨 New invoice arrived: {inv.id} from {inv.vendor} — ₹{inv.amount:,.0f} due in {inv.due_in} days"
        )

    # Phase 2: Age invoices
    _age_invoices(state, day_log)

    # Update log counts to reflect newly arrived and newly overdue invoices
    day_log.active_invoice_count = len(state.active_invoices)
    day_log.overdue_invoice_count = len(state.overdue_invoices)

    # Phase 3: Collect receivables
    revenue_today = _collect_receivables(state, day, day_log)
    day_log.revenue_collected = revenue_today

    # Phase 4: Run advisors
    day_log.advisor_memos = {
        "Expenditure": expenditure_advisor(state),
        "Revenue": revenue_advisor(state),
        "Risk": risk_advisor(state),
    }

    # Phase 5: CFO decides actions
    actions = cfo_decide(state)
    day_log.actions = actions

    # Phase 6: Apply actions
    paid_today = _apply_actions(state, actions, day_log)
    day_log.invoices_paid_today = paid_today

    # Phase 7: Apply daily interest and late fees
    fees, interest = _apply_daily_charges(state, day_log)
    day_log.late_fees_incurred = fees
    day_log.interest_incurred = interest

    # Phase 8: Compute reward
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

    # Phase 9: Log closing state
    day_log.closing_cash = state.cash
    day_log.closing_credit_used = state.credit_used

    return day_log


def run_simulation(
    difficulty: str = "medium",
    sim_window: int = 7,
    seed: int = 42,
) -> SimulationResult:
    """
    Run a complete cashflow simulation (all days at once).

    Uses init_simulation() and step_one_day() internally.
    """
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

    # Final summary
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



# ─────────────────────────────────────────────
# Internal helper functions
# ─────────────────────────────────────────────

def _activate_incoming(
    state: SimState,
    incoming_invoices: List[IncomingInvoice],
    day: int,
) -> List[SimInvoice]:
    """Activate any incoming invoices scheduled for today."""
    activated = []
    remaining = []

    for inc in incoming_invoices:
        if inc.appears_on_day == day:
            new_inv = SimInvoice(
                id=inc.id,
                vendor=inc.vendor,
                amount=inc.hidden_amount,
                original_amount=inc.hidden_amount,
                due_in=inc.hidden_due_in,
                late_fee=inc.hidden_late_fee,
                interest_rate=inc.hidden_interest_rate,
                status="unpaid",
            )
            state.active_invoices.append(new_inv)
            activated.append(new_inv)
        else:
            remaining.append(inc)

    # Update the count the agent sees
    incoming_invoices.clear()
    incoming_invoices.extend(remaining)
    state.upcoming_invoice_count = len(remaining)

    return activated


def _age_invoices(state: SimState, day_log: DayLog):
    """Decrement due_in for all active invoices. Mark newly overdue ones."""
    for inv in state.active_invoices:
        if inv.status == "paid":
            continue
        inv.due_in -= 1

        # Check if newly overdue
        if inv.due_in < 0 and inv.status != "overdue":
            inv.status = "overdue"
            if inv not in state.overdue_invoices:
                state.overdue_invoices.append(inv)
            day_log.events.append(
                f"⏰ Invoice {inv.id} from {inv.vendor} is now OVERDUE! (₹{inv.amount:,.0f})"
            )


def _collect_receivables(state: SimState, day: int, day_log: DayLog) -> float:
    """Collect any receivables that arrive today."""
    collected = 0.0
    remaining = []

    for rec in state.receivables:
        if rec.arrives_on_day == day:
            if random.random() < rec.probability:
                state.cash += rec.amount
                collected += rec.amount
                day_log.events.append(
                    f"💰 Received ₹{rec.amount:,.0f} from {rec.customer}"
                )
            else:
                day_log.events.append(
                    f"❌ Payment from {rec.customer} (₹{rec.amount:,.0f}) FAILED to arrive"
                )
        else:
            remaining.append(rec)

    state.receivables = remaining
    return collected


def _apply_actions(state: SimState, actions: List[SimAction], day_log: DayLog) -> int:
    """Apply CFO actions to the state. Returns number of invoices fully paid."""
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
            pass  # Explicitly doing nothing

    return paid_count


def _apply_daily_charges(state: SimState, day_log: DayLog) -> tuple:
    """Apply late fees and interest to all active (non-paid) invoices."""
    total_fees = 0.0
    total_interest = 0.0

    for inv in state.active_invoices:
        if inv.status == "paid":
            continue

        # Interest accrues on all active invoices
        interest = round(inv.amount * inv.interest_rate, 2)
        inv.amount += interest
        total_interest += interest

        # Late fee only for overdue invoices
        if inv.due_in < 0:
            inv.amount += inv.late_fee
            total_fees += inv.late_fee

    return total_fees, total_interest


def _find_invoice(state: SimState, invoice_id: str):
    """Find an invoice in the active list by ID."""
    for inv in state.active_invoices:
        if inv.id == invoice_id:
            return inv
    return None


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    result = run_simulation(difficulty="medium", sim_window=7, seed=42)

    for day_log in result.days:
        print(f"\n{'='*60}")
        print(f"DAY {day_log.day}")
        print(f"  Opening:  Cash ₹{day_log.opening_cash:,.0f} | Credit ₹{day_log.opening_credit_used:,.0f}")
        print(f"  Active: {day_log.active_invoice_count} | Overdue: {day_log.overdue_invoice_count}")

        for event in day_log.events:
            print(f"  {event}")

        for action in day_log.actions:
            print(f"  → [{action.type.upper()}] {action.invoice_id or ''}: {action.reasoning}")

        print(f"  Closing: Cash ₹{day_log.closing_cash:,.0f} | Credit ₹{day_log.closing_credit_used:,.0f}")
        print(f"  Reward: {day_log.reward:.1f} | Late fees: ₹{day_log.late_fees_incurred:,.0f} | Interest: ₹{day_log.interest_incurred:,.0f}")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"  Cash: ₹{result.final_cash:,.0f} | Credit: ₹{result.final_credit_used:,.0f}")
    print(f"  Paid: {result.invoices_paid}/{result.total_invoices} | Overdue: {result.invoices_overdue}")
    print(f"  Total Reward: {result.total_reward:.1f}")

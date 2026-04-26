"""
State serialization utilities for ICL prompts.

Converts simulation state and historical day logs into
human-readable text blocks that can be injected into LLM prompts.
"""

from typing import List
from models import CashflowmanagerObservation as State, DayLog


# Maximum number of past days to include in full detail.
# Older days get collapsed into a summary line.
MAX_HISTORY_DAYS = 5


def serialize_state(state: State) -> str:
    """Convert the current simulation state into a text block for LLM prompts."""
    lines = [
        f"=== FINANCIAL STATE (Day {state.day}) ===",
        f"Cash: ₹{state.cash:,.0f} | Credit Used: ₹{state.credit_used:,.0f} / ₹{state.credit_limit:,.0f}",
    ]

    # Active invoices table
    if state.active_invoices:
        lines.append(f"\nActive Invoices ({len(state.active_invoices)}):")
        for inv in state.active_invoices:
            due_label = "OVERDUE" if inv.due_in < 0 else ("TODAY" if inv.due_in == 0 else f"{inv.due_in}d")
            status_flag = f" ⚠ {inv.status.upper()}" if inv.status in ("overdue", "partial") else ""
            lines.append(
                f"  {inv.id} | {inv.vendor_id} | ₹{inv.amount:,.0f} | Due: {due_label} "
                f"| Late Fee: ₹{inv.late_fee:,.0f} | Interest: {inv.interest*100:.1f}%{status_flag}"
            )
    else:
        lines.append("\nActive Invoices: None")

    # Receivables
    if state.receivables:
        lines.append(f"\nReceivables ({len(state.receivables)}):")
        for rec in state.receivables:
            lines.append(
                f"  {rec.id} | {rec.customer_id} | ₹{rec.amount:,.0f} "
                f"| Expected: Day {rec.expected_in} | Prob: {rec.probability*100:.0f}%"
            )
    else:
        lines.append("\nReceivables: None")

    # Upcoming invoices (hidden details)
    if state.upcoming_invoice_count > 0:
        lines.append(f"\n⚠ {state.upcoming_invoice_count} more invoices arriving soon (amounts unknown)")

    # Summary stats
    total_debt = sum(inv.amount for inv in state.active_invoices)
    total_expected = sum(r.amount * r.probability for r in state.receivables)
    credit_available = state.credit_limit - state.credit_used
    lines.append(f"\nSummary: Total Debt ₹{total_debt:,.0f} | Expected Inflows ₹{total_expected:,.0f} | Credit Available ₹{credit_available:,.0f}")

    return "\n".join(lines)


def serialize_history(past_logs: List[DayLog]) -> str:
    """Convert past day logs into a learning context block for ICL prompts.
    
    The model sees what it did on previous days and how the rubric scored it,
    enabling in-context adaptation.
    """
    if not past_logs:
        return "=== HISTORY ===\nNo previous days. This is Day 1."

    lines = ["=== HISTORY ==="]

    # If history is longer than MAX_HISTORY_DAYS, summarize older days
    if len(past_logs) > MAX_HISTORY_DAYS:
        old_logs = past_logs[:-MAX_HISTORY_DAYS]
        recent_logs = past_logs[-MAX_HISTORY_DAYS:]

        total_old_reward = sum(d.reward for d in old_logs)
        total_old_fees = sum(d.late_fees_incurred for d in old_logs)
        total_old_paid = sum(d.invoices_paid_today for d in old_logs)
        lines.append(
            f"Days 1-{old_logs[-1].day} Summary: "
            f"Reward={total_old_reward:.1f} | Invoices Paid={total_old_paid} | "
            f"Late Fees=₹{total_old_fees:,.0f}"
        )
        lines.append("")
    else:
        recent_logs = past_logs

    # Detailed entries for recent days
    for day_log in recent_logs:
        # Summarize actions
        action_strs = []
        for a in day_log.actions:
            if a.type in ("pay", "partial"):
                action_strs.append(f"{a.type} {a.invoice_id} ₹{a.amount:,.0f}")
            elif a.type == "credit":
                action_strs.append(f"credit ₹{a.amount:,.0f}")
            elif a.type in ("defer", "negotiate"):
                action_strs.append(f"{a.type} {a.invoice_id or ''}")
        actions_text = ", ".join(action_strs) if action_strs else "no actions"

        lines.append(
            f"Day {day_log.day}: Actions=[{actions_text}] | "
            f"Reward: {day_log.reward:.1f} | "
            f"Late Fees: ₹{day_log.late_fees_incurred:,.0f} | "
            f"Interest: ₹{day_log.interest_incurred:,.0f} | "
            f"Revenue: ₹{day_log.revenue_collected:,.0f}"
        )

        # Generate feedback line based on reward
        feedback = _generate_feedback(day_log)
        if feedback:
            lines.append(f"  → {feedback}")
        lines.append("")

    return "\n".join(lines)


def _generate_feedback(day_log: DayLog) -> str:
    """Generate a natural-language feedback line from the day's results."""
    parts = []

    if day_log.reward < -100:
        parts.append("⚠ CRITICAL: Massive penalty. Rethink strategy entirely.")
    elif day_log.reward < 0:
        parts.append("⚠ Negative reward.")

    if day_log.late_fees_incurred > 0:
        parts.append(f"Late fees hurt (₹{day_log.late_fees_incurred:,.0f}). Prioritize overdue invoices.")

    if day_log.closing_credit_used > 0 and day_log.reward < 0:
        parts.append("Credit usage amplified the penalty. Use credit sparingly.")

    if day_log.invoices_paid_today > 0 and day_log.reward > 0:
        parts.append(f"Paying {day_log.invoices_paid_today} invoice(s) was rewarded.")

    if day_log.revenue_collected > 0:
        parts.append(f"Collected ₹{day_log.revenue_collected:,.0f} — good timing.")

    if not parts:
        if day_log.reward > 50:
            return "✅ Strong performance. Maintain this strategy."
        else:
            return "Neutral day. Look for optimization opportunities."

    return " ".join(parts)

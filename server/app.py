"""
FastAPI + Gradio server for the Cashflow Simulation Engine.
"""

import sys
import os
print(f"DEBUG: Starting server from CWD: {os.getcwd()}")
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"DEBUG: Root dir added to path: {root_dir}")
sys.path.insert(0, root_dir)

from fastapi import FastAPI
import gradio as gr
import pandas as pd
import time
import random

from server.cashflowmanager_environment import run_simulation, init_simulation, step_one_day
from models import SimulationResult, DayLog

# Initialize FastAPI app for mounting Gradio
app = FastAPI()

# ═══════════════════════════════════════════════
# Day-by-Day state (persisted between button clicks)
# ═══════════════════════════════════════════════

_day_state = None          # State object
_day_incoming = None       # list of IncomingInvoice
_day_logs = []             # accumulated DayLog entries
_day_sim_window = 7        # max days
_day_seed = 0              # seed used
_day_difficulty = "medium"


# ═══════════════════════════════════════════════
# MODE 1: Full Simulation (run all days at once)
# ═══════════════════════════════════════════════

def run_full_simulation(difficulty: str, sim_window: int, seed: int):
    """Run all days in one shot and return formatted results."""
    if not seed or seed <= 0:
        seed = int(time.time() * 1000) % 100000
    else:
        seed = int(seed)

    result = run_simulation(
        difficulty=difficulty,
        sim_window=int(sim_window),
        seed=seed,
    )
    summary, log, chart = _format_result(result)
    return summary, log, chart


# ═══════════════════════════════════════════════
# MODE 2: Day-by-Day Simulation
# ═══════════════════════════════════════════════

def start_day_by_day(difficulty: str, sim_window: int, seed: int):
    """Initialize a new day-by-day simulation. Does NOT run any days yet."""
    global _day_state, _day_incoming, _day_logs, _day_sim_window, _day_seed, _day_difficulty

    if not seed or seed <= 0:
        seed = int(time.time() * 1000) % 100000
    else:
        seed = int(seed)

    _day_state, _day_incoming = init_simulation(
        difficulty=difficulty,
        sim_window=int(sim_window),
        seed=seed,
    )
    _day_logs = []
    _day_sim_window = int(sim_window)
    _day_seed = seed
    _day_difficulty = difficulty

    status = (
        f"## 🎬 Simulation Initialized\n"
        f"**Difficulty:** {difficulty.upper()} | **Window:** {sim_window} days | **Seed:** {seed}\n\n"
        f"**Starting Cash:** ₹{_day_state.cash:,.0f} | **Credit Limit:** ₹{_day_state.credit_limit:,.0f}\n\n"
        f"**Active Invoices:** {len(_day_state.active_invoices)} | "
        f"**Upcoming Invoices:** {_day_state.upcoming_invoice_count} | "
        f"**Receivables:** {len(_day_state.receivables)}\n\n"
        f"*Click **▶ Next Day** to step through the simulation.*"
    )
    return status, "", ""


def advance_one_day():
    """Step the simulation forward by one day and append the log."""
    global _day_state, _day_incoming, _day_logs

    if _day_state is None:
        return "⚠️ *No simulation initialized. Click **🎬 Start New** first.*", "", ""

    current_day = _day_state.day + 1
    if current_day > _day_sim_window:
        status = _build_status_panel() + "\n\n## 🏁 Simulation Complete!"
        return status, _build_metrics_panel(), _format_day_logs(_day_logs)

    # Step one day
    day_log = step_one_day(_day_state, _day_incoming)
    _day_logs.append(day_log)

    # Check if this was the last day
    if current_day >= _day_sim_window:
        status = _build_status_panel() + "\n\n## 🏁 Simulation Complete!"
    else:
        status = _build_status_panel()

    return status, _build_metrics_panel(), _format_day_logs(_day_logs)


def _build_status_panel() -> str:
    """Build the left panel: status summary table."""
    if not _day_logs:
        return ""

    total_reward = sum(d.reward for d in _day_logs)
    total_fees = sum(d.late_fees_incurred for d in _day_logs)
    total_interest = sum(d.interest_incurred for d in _day_logs)
    total_revenue = sum(d.revenue_collected for d in _day_logs)
    paid_count = sum(d.invoices_paid_today for d in _day_logs)

    last = _day_logs[-1]
    return (
        f"## 📊 Status after Day {last.day} / {_day_sim_window}\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| **Difficulty** | {_day_difficulty.upper()} |\n"
        f"| **Seed** | {_day_seed} |\n"
        f"| **Current Cash** | ₹{last.closing_cash:,.0f} |\n"
        f"| **Credit Used** | ₹{last.closing_credit_used:,.0f} |\n"
        f"| **Invoices Paid (so far)** | {paid_count} |\n"
        f"| **Active Invoices** | {len(_day_state.active_invoices)} |\n"
        f"| **Overdue** | {len(_day_state.overdue_invoices)} |\n"
        f"| **Total Late Fees** | ₹{total_fees:,.0f} |\n"
        f"| **Total Interest** | ₹{total_interest:,.0f} |\n"
        f"| **Revenue Collected** | ₹{total_revenue:,.0f} |\n"
        f"| **Total Reward** | {total_reward:.1f} |\n"
    )


def _build_metrics_panel() -> str:
    """Build the right panel: per-day running metrics table."""
    if not _day_logs:
        return ""

    md = "## 📈 Running Metrics\n"
    md += "| Day | Cash | Reward | Late Fees | Interest |\n"
    md += "|-----|------|--------|-----------|----------|\n"
    for d in _day_logs:
        md += f"| {d.day} | ₹{d.closing_cash:,.0f} | {d.reward:.1f} | ₹{d.late_fees_incurred:,.0f} | ₹{d.interest_incurred:,.0f} |\n"

    return md


def _format_day_logs(logs: list) -> str:
    """Format accumulated day logs into markdown."""
    lines = []
    for day_log in logs:
        lines.append(f"### 📅 Day {day_log.day}")
        lines.append(f"**Opening:** Cash ₹{day_log.opening_cash:,.0f} | Credit Used ₹{day_log.opening_credit_used:,.0f}")
        lines.append(f"**Active Invoices:** {day_log.active_invoice_count} | **Overdue:** {day_log.overdue_invoice_count}")
        lines.append("")

        if day_log.events:
            lines.append("**Events:**")
            for event in day_log.events:
                lines.append(f"- {event}")
            lines.append("")

        lines.append("**Advisor Memos:**")
        for agent, memo in day_log.advisor_memos.items():
            lines.append(f"**{agent}:**")
            for line in memo.split("\n"):
                lines.append(f"> {line}")
            lines.append("")

        if day_log.actions:
            lines.append("**CFO Actions:**")
            for action in day_log.actions:
                icon = {"pay": "✅", "partial": "💳", "credit": "🏦", "defer": "⏳", "negotiate": "🤝"}.get(action.type, "❓")
                target = f" → {action.invoice_id}" if action.invoice_id else ""
                amount_str = f" (₹{action.amount:,.0f})" if action.amount else ""
                # "reasoning" is mapped to "memo" in CashflowmanagerAction
                lines.append(f"- {icon} **{action.type.upper()}**{target}{amount_str}: {action.memo}")
            lines.append("")

        lines.append(f"**Closing:** Cash ₹{day_log.closing_cash:,.0f} | Credit ₹{day_log.closing_credit_used:,.0f}")
        lines.append(f"📈 Reward: **{day_log.reward:.1f}** | Late fees: ₹{day_log.late_fees_incurred:,.0f} | Interest: ₹{day_log.interest_incurred:,.0f}")
        if day_log.revenue_collected > 0:
            lines.append(f"💰 Revenue collected: ₹{day_log.revenue_collected:,.0f}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _build_day_chart():
    if not _day_logs:
        return _empty_chart()
    return pd.DataFrame({
        "Day": [d.day for d in _day_logs],
        "Cash": [d.closing_cash for d in _day_logs],
        "Reward": [d.reward for d in _day_logs],
        "Late Fees": [d.late_fees_incurred for d in _day_logs],
    })


def _empty_chart():
    return pd.DataFrame({"Day": [], "Cash": [], "Reward": [], "Late Fees": []})


# ═══════════════════════════════════════════════
# Formatting for Full Simulation mode
# ═══════════════════════════════════════════════

def _format_result(result: SimulationResult):
    summary = f"""## 📊 Simulation Summary
| Metric | Value |
|--------|-------|
| **Difficulty** | {result.difficulty.upper()} |
| **Window** | {result.sim_window} days |
| **Seed** | {result.seed} |
| **Final Cash** | ₹{result.final_cash:,.0f} |
| **Credit Used** | ₹{result.final_credit_used:,.0f} |
| **Invoices Paid** | {result.invoices_paid} / {result.total_invoices} |
| **Invoices Overdue** | {result.invoices_overdue} |
| **Total Late Fees** | ₹{result.total_late_fees:,.0f} |
| **Total Interest** | ₹{result.total_interest:,.0f} |
| **Revenue Collected** | ₹{result.total_revenue_collected:,.0f} |
| **Total Reward** | {result.total_reward:.1f} |
"""

    log = _format_day_logs(result.days)

    chart_data = pd.DataFrame({
        "Day": [d.day for d in result.days],
        "Cash": [d.closing_cash for d in result.days],
        "Reward": [d.reward for d in result.days],
        "Late Fees": [d.late_fees_incurred for d in result.days],
    })

    return summary, log, chart_data


# ═══════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="Cashflow Simulation") as demo:
        gr.Markdown("# 🏢 Cashflow Simulation Engine")

        with gr.Row():
            difficulty = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="medium",
                label="Difficulty",
                scale=1,
            )
            sim_window = gr.Slider(
                minimum=7, maximum=30, value=7, step=1,
                label="Simulation Window (days)",
                scale=2,
            )
            seed = gr.Number(value=0, label="Seed (0 = random)", precision=0, scale=1)

        with gr.Tabs():

            # ── Tab 1: Full Simulation ──
            with gr.TabItem("🚀 Full Simulation"):
                full_run_btn = gr.Button("🚀 Run Full Simulation", variant="primary")

                with gr.Row():
                    with gr.Column(scale=2):
                        full_summary = gr.Markdown("*Click 'Run Full Simulation' to start.*")
                    with gr.Column(scale=3):
                        full_chart = gr.Dataframe(
                            headers=["Day", "Cash", "Reward", "Late Fees"],
                            label="Day-by-Day Metrics",
                            interactive=False,
                        )

                gr.Markdown("---")
                gr.Markdown("## 📜 Day-by-Day Log")
                full_log = gr.Markdown("*Logs will appear here after running.*")

                full_run_btn.click(
                    fn=run_full_simulation,
                    inputs=[difficulty, sim_window, seed],
                    outputs=[full_summary, full_log, full_chart],
                )

            # ── Tab 2: Day-by-Day ──
            with gr.TabItem("📅 Day-by-Day"):
                with gr.Row():
                    start_btn = gr.Button("🎬 Start New", variant="secondary", scale=1)
                    next_btn = gr.Button("▶ Next Day", variant="primary", scale=2)

                with gr.Row():
                    with gr.Column(scale=2):
                        day_status = gr.Markdown("*Click '🎬 Start New' to initialize, then '▶ Next Day' to step.*")
                    with gr.Column(scale=3):
                        day_metrics = gr.Markdown("")

                gr.Markdown("---")
                gr.Markdown("## 📜 Accumulated Log")
                day_log_md = gr.Markdown("*Logs will append here as you step through days.*")

                start_btn.click(
                    fn=start_day_by_day,
                    inputs=[difficulty, sim_window, seed],
                    outputs=[day_status, day_metrics, day_log_md],
                )
                next_btn.click(
                    fn=advance_one_day,
                    outputs=[day_status, day_metrics, day_log_md],
                )

    return demo

gradio_app = build_ui()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)

if __name__ == "__main__":
    main()
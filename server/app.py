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
from server.cashflowmanager_environment import CashflowmanagerAction
from models import SimulationResult, DayLog

# Initialize FastAPI app for mounting Gradio
app = FastAPI()

# ═══════════════════════════════════════════════
# Day-by-Day state (persisted between button clicks)
# ═══════════════════════════════════════════════

_day_state = None          # State object
_day_incoming = None       # list of IncomingInvoice
_day_world_model = None    # WorldModel instance for the current day-by-day run
_day_logs = []             # accumulated DayLog entries
_day_sim_window = 3        # max days
_day_seed = 0              # seed used
_day_difficulty = "medium"


# ═══════════════════════════════════════════════
# MODE 1: Full Simulation (run all days at once)
# ═══════════════════════════════════════════════

def run_full_simulation(difficulty: str, sim_window: int, seed: int):
    """Run all days in one shot and return formatted results."""
    if not seed or seed <= 0:
        seed = random.randint(10000, 99999)
    else:
        seed = int(seed)

    result = run_simulation(
        difficulty=difficulty,
        sim_window=int(sim_window),
        seed=seed,
    )
    summary, log, chart, right_panel = _format_result(result)
    # Outputs in fixed order: summary, log, chart, right_panel, seed
    return summary, log, chart, right_panel, 0

def preview_full_simulation(difficulty: str, sim_window: int, seed: int):
    """Preview the initial state without running."""
    if not seed or seed <= 0:
        seed = random.randint(10000, 99999)
    else:
        seed = int(seed)

    state, _, _ = init_simulation(difficulty=difficulty, sim_window=int(sim_window), seed=seed)
    
    status = (
        f"## 🔍 Scenario Preview\n"
        f"**Difficulty:** {difficulty.upper()} | **Window:** {sim_window} days | **Seed:** {seed}\n\n"
        f"**Starting Cash:** ₹{state.cash:,.0f} | **Credit Limit:** ₹{state.credit_limit:,.0f}\n\n"
        f"**Active Invoices:** {len(state.active_invoices)} | "
        f"**Upcoming Invoices:** {state.upcoming_invoice_count} | "
        f"**Receivables:** {len(state.receivables)}\n\n"
        f"*Click **🚀 Run Full Simulation** to execute this scenario.*"
    )
    # outputs: full_summary, full_chart, shared_log_md, full_right_panel, seed
    return status, _empty_chart(), "", "", 0


# ═══════════════════════════════════════════════
# MODE 2: Day-by-Day Simulation
# ═══════════════════════════════════════════════

def start_day_by_day(difficulty: str, sim_window: int, seed: int):
    """Initialize a new day-by-day simulation. Does NOT run any days yet."""
    global _day_state, _day_incoming, _day_world_model, _day_logs, _day_sim_window, _day_seed, _day_difficulty

    if not seed or seed <= 0:
        seed = random.randint(10000, 99999)
    else:
        seed = int(seed)

    _day_state, _day_incoming, _day_world_model = init_simulation(
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
    return status, "", "", 0  # Reset seed to 0


def advance_one_day():
    """Step the simulation forward by one day and append the log."""
    global _day_state, _day_incoming, _day_world_model, _day_logs

    if _day_state is None:
        return "⚠️ *No simulation initialized. Click **🎬 Start New** first.*", "", ""

    current_day = _day_state.day + 1
    if current_day > _day_sim_window:
        status = _build_status_panel() + "\n\n## 🏁 Simulation Complete!"
        return status, _build_metrics_panel(), _format_day_logs(_day_logs)

    # Step one day
    day_log = step_one_day(_day_state, _day_incoming, _day_logs, _day_world_model)
    _day_logs.append(day_log)

    # Check if this was the last day
    if current_day >= _day_sim_window:
        # Build the SimulationResult so we can compute the final score
        from server.scoring import compute_simulation_score
        result = SimulationResult(
            difficulty=_day_difficulty,
            sim_window=_day_sim_window,
            seed=0,
        )
        result.days = _day_logs
        result.final_cash = _day_state.cash
        result.final_credit_used = _day_state.credit_used
        result.invoices_paid = len(_day_state.paid_invoices)
        result.invoices_overdue = len(_day_state.overdue_invoices)
        result.total_invoices = result.invoices_paid + len(_day_state.active_invoices)
        result.total_late_fees = sum(d.late_fees_incurred for d in result.days)
        result.total_interest = sum(d.interest_incurred for d in result.days)
        result.total_revenue_collected = sum(d.revenue_collected for d in result.days)
        result.total_reward = round(sum(d.reward for d in result.days), 2)

        eval_result = compute_simulation_score(result)
        result.score = eval_result["score"]
        result.score_breakdown = eval_result["breakdown"]
        result.grade = eval_result["grade"]

        status = _build_status_panel() + "\n\n## 🏁 Simulation Complete!"
        metrics = _build_metrics_panel(result=result)
    else:
        status = _build_status_panel()
        metrics = _build_metrics_panel()

    return status, metrics, _format_day_logs(_day_logs)


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


def _build_metrics_panel(result: SimulationResult = None) -> str:
    """Build the right panel: per-day running metrics table.

    When a finished SimulationResult is passed, also renders the score panel
    below the metrics — fills the empty space on the right when sim completes.
    """
    if not _day_logs:
        return ""

    md = "## 📈 Running Metrics\n"
    md += "| Day | Cash | Reward | Late Fees | Interest |\n"
    md += "|-----|------|--------|-----------|----------|\n"
    for d in _day_logs:
        md += f"| {d.day} | ₹{d.closing_cash:,.0f} | {d.reward:.1f} | ₹{d.late_fees_incurred:,.0f} | ₹{d.interest_incurred:,.0f} |\n"

    if result is not None and result.score_breakdown:
        md += "\n\n" + _build_score_panel(result)

    return md


def _build_score_panel(result: SimulationResult) -> str:
    """Prominent score banner + per-dimension breakdown, suitable for either
    the day-by-day metrics column or the full-simulation summary column."""
    rows = ""
    for dim, val in (result.score_breakdown or {}).items():
        label = dim.replace("_", " ").title()
        bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
        rows += f"| {label} | `{bar}` | {val:.2f} |\n"

    return (
        f'<div class="score-banner">\n'
        f'<div class="score-label">🏆 Agent Score</div>\n'
        f'<div class="score-num">{result.score:.2f} <span class="score-denom">/ 1.00</span></div>\n'
        f'<div class="score-grade grade-{result.grade}">Grade {result.grade}</div>\n'
        f'</div>\n\n'
        f'| Dimension |  | Score |\n'
        f'|-----------|---|------|\n'
        f'{rows}'
    )


def _format_day_logs(logs: list) -> str:
    """Format accumulated day logs into markdown — each day wrapped as a card."""
    cards = []
    for day_log in logs:
        body = []
        body.append(f"### 📅 Day {day_log.day}")
        body.append(f"**Opening:** Cash ₹{day_log.opening_cash:,.0f} | Credit Used ₹{day_log.opening_credit_used:,.0f}")
        body.append(f"**Active Invoices:** {day_log.active_invoice_count} | **Overdue:** {day_log.overdue_invoice_count}")

        if day_log.events:
            body.append("\n**Events:**")
            for event in day_log.events:
                body.append(f"- {event}")

        if day_log.advisor_memos:
            body.append("\n**Advisor Memos:**")
            for agent, memo in day_log.advisor_memos.items():
                body.append(f"**{agent}:**")
                for line in memo.split("\n"):
                    body.append(f"> {line}")

        if day_log.actions:
            body.append("\n**CFO Actions:**")
            for action in day_log.actions:
                icon = {"pay": "✅", "partial": "💳", "credit": "🏦", "defer": "⏳", "negotiate": "🤝"}.get(action.type, "❓")
                target = f" → {action.invoice_id}" if action.invoice_id else ""
                amount_str = f" (₹{action.amount:,.0f})" if action.amount else ""
                body.append(f"- {icon} **{action.type.upper()}**{target}{amount_str}: {action.memo}")

        body.append(f"\n**Closing:** Cash ₹{day_log.closing_cash:,.0f} | Credit ₹{day_log.closing_credit_used:,.0f}")
        body.append(f"📈 Reward: **{day_log.reward:.1f}** | Late fees: ₹{day_log.late_fees_incurred:,.0f} | Interest: ₹{day_log.interest_incurred:,.0f}")
        if day_log.revenue_collected > 0:
            body.append(f"💰 Revenue collected: ₹{day_log.revenue_collected:,.0f}")

        # Wrap the day's markdown in a styled card. The blank lines inside the
        # div are required so Gradio's markdown renderer parses the inner block.
        card_md = "\n\n".join(body)
        cards.append(f'<div class="day-card">\n\n{card_md}\n\n</div>')

    return "\n\n".join(cards)


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
    """Returns (left_summary, day_logs, chart_data, right_score_panel).

    Two-column Full Sim layout (mirrors Day-by-Day):
      LEFT column gets the Simulation Summary table.
      RIGHT column gets the Day-by-Day Metrics chart + Score banner + breakdown.
    """
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

    right_panel = _build_score_panel(result)

    return summary, log, chart_data, right_panel


# ═══════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════

css = """
/* ── Force blue everywhere via CSS variable overrides (Gradio 6.x) ── */
:root, .gradio-container, .dark, gradio-app {
    --button-primary-background-fill: #2563eb !important;
    --button-primary-background-fill-hover: #1d4ed8 !important;
    --button-primary-background-fill-focus: #1d4ed8 !important;
    --button-primary-text-color: #ffffff !important;
    --button-primary-border-color: #2563eb !important;
    --button-primary-border-color-hover: #1d4ed8 !important;
    --color-accent: #2563eb !important;
    --color-accent-soft: rgba(37, 99, 235, 0.18) !important;
    --primary-100: #dbeafe !important;
    --primary-200: #bfdbfe !important;
    --primary-300: #93c5fd !important;
    --primary-400: #60a5fa !important;
    --primary-500: #3b82f6 !important;
    --primary-600: #2563eb !important;
    --primary-700: #1d4ed8 !important;
    --primary-800: #1e40af !important;
    --primary-900: #1e3a8a !important;
    --link-text-color: #60a5fa !important;
}

/* Direct button overrides as belt-and-suspenders */
button.primary, .gr-button-primary, button[data-testid*="primary"] {
    background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%) !important;
    background-color: #2563eb !important;
    border-color: #1d4ed8 !important;
    color: #ffffff !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: #1d4ed8 !important;
    background-color: #1d4ed8 !important;
}

/* ── Viewport lock: page itself never scrolls; sections scroll internally ── */
html, body, .gradio-container, gradio-app {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
}
.gradio-container {
    padding: 12px !important;
    box-sizing: border-box !important;
}
footer { display: none !important; }

/* The main row containing sidebar + content fills remaining height */
.main-row { height: calc(100vh - 80px) !important; }
.main-row > * { height: 100% !important; max-height: 100% !important; }

/* Sidebar stays put; doesn't scroll */
.sidebar-col {
    height: 100% !important;
    max-height: 100% !important;
    overflow-y: auto !important;
    padding: 8px !important;
}

/* Each tab pane: bounded height, internal vertical scroll only */
.tabitem, [role="tabpanel"] {
    height: calc(100vh - 160px) !important;
    max-height: calc(100vh - 160px) !important;
    overflow-y: auto !important;
    padding: 10px 12px !important;
}

/* Two-column layouts inside a tab fill the available height */
.tab-grid { height: 100% !important; }
.tab-grid > * {
    height: 100% !important;
    overflow-y: auto !important;
    padding: 8px 12px !important;
}

/* ── Logs tab: scrollable container with day cards ── */
/* Light defaults; `.dark` overrides further down. */
.scrollable-logs {
    height: calc(100vh - 200px) !important;
    max-height: calc(100vh - 200px) !important;
    overflow-y: auto !important;
    padding: 14px !important;
    border-radius: 12px;
    background-color: #f8fafc;
    border: 1px solid #e5e7eb;
}
.dark .scrollable-logs {
    background-color: #0b1220;
    border-color: #1f2a3d;
}

/* ── Day card: each day's log block gets its own panel ── */
.day-card {
    background: #f1f5f9;
    color: #0f172a;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 0 0 14px 0;
    border-left: 4px solid #2563eb;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
.day-card h3 {
    margin-top: 0 !important;
    margin-bottom: 10px !important;
    color: #1d4ed8 !important;
    border-bottom: none !important;
}
.day-card hr { display: none !important; }
.day-card blockquote {
    border-left: 2px solid #cbd5e1;
    margin: 4px 0;
    padding: 2px 10px;
    color: #475569;
}
.dark .day-card {
    background: #1f2937;
    color: #f3f4f6;
    border-left-color: #3b82f6;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
}
.dark .day-card h3 { color: #60a5fa !important; }
.dark .day-card blockquote {
    border-left-color: #475569;
    color: #cbd5e1;
}

/* ── Score banner: compact horizontal layout ── */
.score-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    color: #ffffff;
    padding: 14px 26px;
    border-radius: 14px;
    margin: 0 0 14px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25);
}
.score-banner .score-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.85;
    white-space: nowrap;
}
.score-banner .score-num {
    font-size: 36px;
    font-weight: 800;
    line-height: 1;
    white-space: nowrap;
}
.score-banner .score-denom {
    font-size: 16px;
    font-weight: 500;
    opacity: 0.7;
}
.score-banner .score-grade {
    padding: 6px 16px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.15);
    font-weight: 700;
    letter-spacing: 0.04em;
    white-space: nowrap;
}
.score-banner .grade-A { background: rgba(34, 197, 94, 0.25); }
.score-banner .grade-B { background: rgba(59, 130, 246, 0.25); }
.score-banner .grade-C { background: rgba(234, 179, 8, 0.25); }
.score-banner .grade-D, .score-banner .grade-F { background: rgba(239, 68, 68, 0.25); }

/* ── Buttons: lock primary to blue regardless of variant ── */
button.primary, .gr-button.primary, button[variant="primary"] {
    background: #2563eb !important;
    border-color: #2563eb !important;
    color: #ffffff !important;
}
button.primary:hover, .gr-button.primary:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
button.secondary, .gr-button.secondary {
    background: #1f2937 !important;
    border-color: #334155 !important;
    color: #e2e8f0 !important;
}
button.secondary:hover, .gr-button.secondary:hover {
    background: #334155 !important;
}

/* Keep the H1 from eating vertical space */
h1 { margin: 4px 0 8px 0 !important; font-size: 22px !important; }

/* ── Theme toggle button ── */
.theme-toggle-btn {
    min-width: 0 !important;
    width: 44px !important;
    height: 44px !important;
    padding: 0 !important;
    font-size: 20px !important;
    border-radius: 999px !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    color: #0f172a !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    margin: 6px 0 !important;
}
.theme-toggle-btn:hover {
    background: #f1f5f9 !important;
    border-color: #94a3b8 !important;
}
.dark .theme-toggle-btn {
    background: #1f2937 !important;
    color: #f3f4f6 !important;
    border-color: #374151 !important;
}
.dark .theme-toggle-btn:hover {
    background: #334155 !important;
    border-color: #475569 !important;
}
.header-row { align-items: center !important; }

/* ── Make Markdown / prose fill column width ── */
/* Gradio's .prose styling caps width around 65ch — override so tables and
   the score banner extend the full width of their column. */
.prose, .gr-markdown, .markdown-body, .gradio-container .prose {
    max-width: 100% !important;
}
.prose table, .gr-markdown table, .markdown-body table {
    width: 100% !important;
    max-width: 100% !important;
}
.prose img, .gr-markdown img { max-width: 100%; }

/* Tighter table row padding so summaries don't waste vertical space */
.prose td, .prose th, .gr-markdown td, .gr-markdown th {
    padding: 6px 10px !important;
}
"""

def build_ui():
    # Gradio 6.0 deprecated `theme` and `css` on `Blocks()` (moved to `launch()`),
    # but this app is mounted via `mount_gradio_app` — there is no `launch()`.
    # Injecting a <style> tag directly into the layout always works.
    with gr.Blocks(title="Cashflow Simulation") as demo:
        gr.HTML(f"<style>{css}</style>")
        with gr.Row(elem_classes="header-row"):
            with gr.Column(scale=10):
                gr.Markdown("# 🏢 Cashflow Simulation Engine")
            with gr.Column(scale=1, min_width=80):
                theme_toggle = gr.Button("🌓", size="sm", elem_classes="theme-toggle-btn")
        # JS-only toggle: flip Gradio's `.dark` class on the root element.
        # Gradio's own CSS keys off this; our CSS does too (see light/dark
        # rules below). Buttons stay blue regardless via CSS variable overrides.
        theme_toggle.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="""() => {
                const root = document.documentElement;
                const body = document.body;
                const isDark = root.classList.contains('dark') || body.classList.contains('dark');
                if (isDark) {
                    root.classList.remove('dark');
                    body.classList.remove('dark');
                } else {
                    root.classList.add('dark');
                    body.classList.add('dark');
                }
            }""",
        )

        with gr.Row(elem_classes="main-row"):
            # ── Sidebar (Inputs & Controls) ──
            with gr.Column(scale=1, variant="panel", elem_classes="sidebar-col"):
                gr.Markdown("### ⚙️ Settings")
                difficulty = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="medium",
                    label="Difficulty",
                )
                sim_window = gr.Number(value=3, visible=False)
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                
                if not os.environ.get("GROQ_API_KEY") and not os.environ.get("USE_LOCAL_HF") == "true":
                    gr.Markdown("⚠️ *Warning: Neither GROQ_API_KEY nor USE_LOCAL_HF are set. LLM agents may fail.*")

                gr.Markdown("---")
                gr.Markdown("### 🚀 Actions")
                
                gr.Markdown("**Full Simulation**")
                preview_btn = gr.Button("🔍 Preview Scenario", variant="secondary")
                full_run_btn = gr.Button("🚀 Run Full Simulation", variant="primary")
                
                gr.Markdown("<br>**Day-by-Day**")
                start_btn = gr.Button("🎬 Start New", variant="secondary", interactive=False)
                next_btn = gr.Button("▶ Next Day", variant="primary", interactive=False)

            # ── Main Content Area ──
            with gr.Column(scale=4):
                with gr.Tabs():

                    # ── Tab 1: Full Simulation ──
                    # Two-column layout (mirrors Day-by-Day): summary table on
                    # the left; chart + score banner + breakdown on the right
                    # so neither column is left empty.
                    with gr.TabItem("🚀 Full Simulation") as tab_full:
                        with gr.Row(elem_classes="tab-grid"):
                            with gr.Column(scale=1):
                                full_summary = gr.Markdown("*Click 'Run Full Simulation' to start.*")
                            with gr.Column(scale=1):
                                full_chart = gr.Dataframe(
                                    headers=["Day", "Cash", "Reward", "Late Fees"],
                                    label="Day-by-Day Metrics",
                                    interactive=False,
                                )
                                full_right_panel = gr.Markdown("")

                    # ── Tab 2: Day-by-Day ──
                    with gr.TabItem("📅 Day-by-Day") as tab_day:
                        with gr.Row(elem_classes="tab-grid"):
                            with gr.Column(scale=1):
                                day_status = gr.Markdown("*Click '🎬 Start New' to initialize, then '▶ Next Day' to step.*")
                            with gr.Column(scale=1):
                                day_metrics = gr.Markdown("")

                    # ── Tab 3: Logs ──
                    with gr.TabItem("📜 Logs") as tab_logs:
                        with gr.Column(elem_classes="scrollable-logs"):
                            shared_log_md = gr.Markdown("*Logs will appear here after running a simulation.*")

                # Wire up the events
                preview_btn.click(
                    fn=preview_full_simulation,
                    inputs=[difficulty, sim_window, seed],
                    outputs=[full_summary, full_chart, shared_log_md, full_right_panel, seed],
                )

                full_run_btn.click(
                    fn=run_full_simulation,
                    inputs=[difficulty, sim_window, seed],
                    outputs=[full_summary, shared_log_md, full_chart, full_right_panel, seed],
                )

                start_btn.click(
                    fn=start_day_by_day,
                    inputs=[difficulty, sim_window, seed],
                    outputs=[day_status, day_metrics, shared_log_md, seed],
                )
                next_btn.click(
                    fn=advance_one_day,
                    outputs=[day_status, day_metrics, shared_log_md],
                )

                # Tab switching logic for buttons
                tab_full.select(
                    fn=lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)),
                    inputs=None,
                    outputs=[preview_btn, full_run_btn, start_btn, next_btn]
                )
                tab_day.select(
                    fn=lambda: (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)),
                    inputs=None,
                    outputs=[preview_btn, full_run_btn, start_btn, next_btn]
                )

    return demo

gradio_app = build_ui()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()





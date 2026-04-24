"""
FastAPI + Gradio server for the Cashflow Multi-Agent RL Environment.
"""

from fastapi import FastAPI
import gradio as gr
import pandas as pd
import time
import json

from openenv.core.env_server.http_server import create_app

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from models import CashflowmanagerAction, CashflowmanagerObservation
    from server.cashflowmanager_environment import CashflowmanagerEnvironment
except ImportError:
    try:
        from cashflowmanager.models import CashflowmanagerAction, CashflowmanagerObservation
        from cashflowmanager.server.cashflowmanager_environment import CashflowmanagerEnvironment
    except ImportError:
        from ..models import CashflowmanagerAction, CashflowmanagerObservation
        from .cashflowmanager_environment import CashflowmanagerEnvironment


app: FastAPI = create_app(
    CashflowmanagerEnvironment,
    CashflowmanagerAction,
    CashflowmanagerObservation,
    env_name="cashflowmanager",
    max_concurrent_envs=1,
)

try:
    from server.client import groq_policy, clear_action_cache
except ImportError:
    try:
        from cashflowmanager.server.client import groq_policy, clear_action_cache
    except ImportError:
        from .client import groq_policy, clear_action_cache


# Global state for the interactive UI
_env_instance = None
_last_obs = None
_history = []

def get_env(seed=42):
    global _env_instance, _last_obs, _history
    if _env_instance is None:
        _env_instance = CashflowmanagerEnvironment()
        clear_action_cache()
        _last_obs = _env_instance.reset(seed=seed)
        _history = []
    return _env_instance, _last_obs

def format_invoices(invoices):
    if not invoices:
        return "No active invoices."
    rows = []
    for inv in invoices:
        urgency = "🔴 OVERDUE" if inv.due_in <= 0 else "🟡 URGENT" if inv.due_in <= 2 else "🟢 OK"
        rows.append(f"{urgency} | {inv.id} | ₹{inv.amount:.0f} | Due: {inv.due_in}d | Vendor: {inv.vendor_id}")
    return "\n".join(rows)

def format_receivables(receivables):
    if not receivables:
        return "No expected inflows."
    rows = []
    for rec in receivables:
        rows.append(f"₹{rec.amount:.0f} from {rec.customer_id} in {rec.expected_in}d (prob: {rec.probability*100:.0f}%)")
    return "\n".join(rows)

def process_step(action_type, invoice_id=None, amount=0.0, memo=None):
    global _env_instance, _last_obs, _history
    env, obs = get_env()
    
    if obs.done:
        return update_ui()

    # Create action
    action = CashflowmanagerAction(type=action_type, invoice_id=invoice_id, amount=amount, memo=memo)
    
    # Step environment
    new_obs = env.step(action)
    
    # Log to history
    entry = {
        "Step": env.state.step_count,
        "Day": new_obs.day,
        "Action": f"{action.type}({action.invoice_id or 'N/A'})",
        "Amount": f"₹{action.amount:.0f}" if action.amount else "N/A",
        "Cash": f"₹{new_obs.cash:.0f}",
        "Reward": round(new_obs.reward, 2),
        "Reasoning": action.memo or "Manual Action",
        "Events": " | ".join(new_obs.world_events) if new_obs.world_events else "None"
    }
    _history.insert(0, entry)
    _last_obs = new_obs
    
    return update_ui()

def ai_step():
    global _last_obs
    if _last_obs is None or _last_obs.done:
        return update_ui()
    
    # Let the policy decide
    action = groq_policy(_last_obs, [])
    return process_step(action.type, action.invoice_id, action.amount, memo=action.memo)

def reset_sim(seed):
    global _env_instance, _last_obs, _history
    _env_instance = None
    get_env(seed=int(seed))
    return update_ui()

def update_ui():
    global _last_obs, _history
    obs = _last_obs
    
    history_df = pd.DataFrame(_history)
    
    status = f"### Day {obs.day} | Step {obs.metadata.get('step', 0)}\n"
    status += f"**Cash:** ₹{obs.cash:.0f} | **Credit Used:** ₹{obs.credit_used:.0f}/{obs.credit_limit:.0f}\n"
    if obs.done:
        status += "## 🏁 EPISODE FINISHED\n"
        
    memos = "#### 🤖 Advisor Memos\n"
    for agent, msg in obs.advisor_messages.items():
        memos += f"- **{agent}:** {msg}\n"
        
    world = "#### 🌍 World Events\n"
    if obs.world_events:
        for e in obs.world_events:
            world += f"- {e}\n"
    else:
        world += "- No events this step."

    invoice_list = [inv.id for inv in obs.invoices]
    
    return (
        status, 
        memos, 
        world, 
        format_invoices(obs.invoices), 
        format_receivables(obs.receivables),
        history_df,
        gr.Dropdown(choices=invoice_list, value=invoice_list[0] if invoice_list else None)
    )

def build_ui():
    with gr.Blocks(title="Cashflow Multi-Agent RL Simulator") as demo:
        gr.Markdown("# 🏢 Cashflow Management Dashboard")
        
        with gr.Row():
            # --- Column 1: Financial State & Control ---
            with gr.Column(scale=3):
                status_md = gr.Markdown("### 💰 Financial Status")
                
                with gr.Row():
                    seed_input = gr.Number(value=42, label="Sim Seed", precision=0)
                    reset_btn = gr.Button("🔄 Reset", variant="secondary")
                    ai_btn = gr.Button("🤖 AI Next Step", variant="primary")
                
                with gr.Tabs():
                    with gr.TabItem("📋 Active Invoices"):
                        invoice_display = gr.Code(label="Debts", language="markdown")
                    with gr.TabItem("📈 Receivables"):
                        receivable_display = gr.Code(label="Expected Inflows", language="markdown")
                
                with gr.Group():
                    gr.Markdown("#### 🕹️ Manual Action")
                    with gr.Row():
                        target_inv = gr.Dropdown(label="Invoice ID", choices=[])
                        pay_amount = gr.Number(label="Amount (₹)", value=0)
                    
                    with gr.Row():
                        pay_btn = gr.Button("Pay Full", variant="stop")
                        partial_btn = gr.Button("Partial")
                        neg_btn = gr.Button("Negotiate", variant="primary")
                        credit_btn = gr.Button("Draw Credit")
                        defer_btn = gr.Button("Defer Step", variant="secondary")

            # --- Column 2: Agent Intel & World Logs ---
            with gr.Column(scale=2):
                memo_md = gr.Markdown("#### 🤖 Advisor Intelligence")
                world_md = gr.Markdown("#### 🌍 World Events & Shocks")
        
        gr.Markdown("---")
        gr.Markdown("### 📜 Transaction & Decision Log")
        history_table = gr.Dataframe(
            headers=["Step", "Day", "Action", "Amount", "Cash", "Reward", "Reasoning", "Events"],
            datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
            interactive=False
        )

        # Event handlers
        demo.load(reset_sim, inputs=[seed_input], outputs=[status_md, memo_md, world_md, invoice_display, receivable_display, history_table, target_inv])
        reset_btn.click(reset_sim, inputs=[seed_input], outputs=[status_md, memo_md, world_md, invoice_display, receivable_display, history_table, target_inv])
        ai_btn.click(ai_step, outputs=[status_md, memo_md, world_md, invoice_display, receivable_display, history_table, target_inv])
        
        pay_btn.click(lambda id, amt: process_step("pay", id, amt), inputs=[target_inv, pay_amount], outputs=[status_md, memo_md, world_md, invoice_display, receivable_display, history_table, target_inv])
        partial_btn.click(lambda id, amt: process_step("partial", id, amt), inputs=[target_inv, pay_amount], outputs=[status_md, memo_md, world_md, invoice_display, receivable_display, history_table, target_inv])
        neg_btn.click(lambda id: process_step("negotiate", id), inputs=[target_inv], outputs=[status_md, memo_md, world_md, invoice_display, receivable_display, history_table, target_inv])
        credit_btn.click(lambda amt: process_step("credit", amount=amt), inputs=[pay_amount], outputs=[status_md, memo_md, world_md, invoice_display, receivable_display, history_table, target_inv])
        defer_btn.click(lambda: process_step("defer"), outputs=[status_md, memo_md, world_md, invoice_display, receivable_display, history_table, target_inv])

    return demo

gradio_app = build_ui()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
"""
SFT Data Generator for Cashflow Multi-Agent Environment.

Runs multiple episodes using rule-based agents, collects:
  1. Expenditure Agent training pairs (invoice state → priority memo)
  2. Revenue Agent training pairs (receivable state → projection memo)
  3. Risk Agent training pairs (financial state → risk assessment)
  4. CFO Agent training pairs (memos + state → action + reasoning)

Output: JSONL files ready for Unsloth SFT training.

Usage:
    python scripts/generate_sft_data.py --episodes 200 --output data/
"""

import os
import sys
import json
import random
import argparse

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.cashflowmanager_environment import CashflowmanagerEnvironment
from server.client import _cfo_rule_decide, clear_action_cache
from server.agents import (
    expenditure_agent, revenue_agent, risk_agent,
    EXPENDITURE_SYSTEM_PROMPT, REVENUE_SYSTEM_PROMPT,
    RISK_SYSTEM_PROMPT, CFO_SYSTEM_PROMPT,
)
from models import CashflowmanagerAction


def build_expenditure_sample(invoices, cash, memo):
    """Build one SFT training sample for the Expenditure Agent."""
    inv_str = "\n".join([
        f"- ID: {inv.id} | Amount: ₹{inv.amount:.0f} | Due: {inv.due_in}d | "
        f"Late Fee: ₹{inv.late_fee:.0f} | Interest: {inv.interest*100:.1f}% | Status: {inv.status}"
        for inv in invoices if inv.status != "paid"
    ])
    user_msg = f"Cash available: ₹{cash:.0f}\n\nInvoices:\n{inv_str}"
    assistant_msg = json.dumps(memo, indent=2)
    return {
        "messages": [
            {"role": "system", "content": EXPENDITURE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def build_revenue_sample(receivables, cash, day, memo):
    """Build one SFT training sample for the Revenue Agent."""
    rec_str = "\n".join([
        f"- ID: {r.id} | Amount: ₹{r.amount:.0f} | Due: {r.expected_in}d | Prob: {r.probability*100:.0f}%"
        for r in receivables
    ])
    user_msg = f"Day: {day} | Cash: ₹{cash:.0f}\n\nReceivables:\n{rec_str}"
    assistant_msg = json.dumps(memo, indent=2)
    return {
        "messages": [
            {"role": "system", "content": REVENUE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def build_risk_sample(cash, credit_used, credit_limit, world_hints, memo):
    """Build one SFT training sample for the Risk Agent."""
    user_msg = (
        f"Cash: ₹{cash:.0f} | Credit Used: ₹{credit_used:.0f}/{credit_limit:.0f}\n"
        f"Market stress: {world_hints.get('market_stress', 0)}\n"
        f"Risk level hint: {world_hints.get('upcoming_risk_level', 'unknown')}\n"
        f"Vendor sentiment: {json.dumps(world_hints.get('vendor_sentiment', {}))}"
    )
    assistant_msg = json.dumps(memo, indent=2)
    return {
        "messages": [
            {"role": "system", "content": RISK_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def build_cfo_sample(obs, action, reward):
    """Build one SFT training sample for the CFO Agent."""
    advisor_str = "\n".join([f"[{k}]: {v}" for k, v in obs.advisor_messages.items()])
    inv_str = "\n".join([
        f"- {inv.id}: ₹{inv.amount:.0f} due in {inv.due_in}d"
        for inv in obs.invoices if inv.status != "paid"
    ][:8])

    user_msg = (
        f"Day: {obs.day} | Cash: ₹{obs.cash:.0f} | Credit: ₹{obs.credit_used:.0f}/{obs.credit_limit:.0f}\n\n"
        f"ADVISORS:\n{advisor_str}\n\n"
        f"INVOICES:\n{inv_str}"
    )
    assistant_msg = json.dumps({
        "type": action.type,
        "invoice_id": action.invoice_id,
        "amount": action.amount,
        "reasoning": action.memo or f"Reward was {reward:.2f}",
    }, indent=2)
    return {
        "messages": [
            {"role": "system", "content": CFO_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def generate_data(num_episodes=200, output_dir="data"):
    """Run episodes and collect SFT training data."""
    os.makedirs(output_dir, exist_ok=True)

    exp_data, rev_data, risk_data, cfo_data = [], [], [], []

    for ep in range(num_episodes):
        env = CashflowmanagerEnvironment()
        clear_action_cache()
        obs = env.reset(seed=ep * 7 + 42)

        steps = 0
        while not obs.done and steps < 50:
            steps += 1
            active = [i for i in obs.invoices if i.status != "paid"]

            # Collect agent memos as training samples
            exp_memo = obs.advisor_memos.get("Expenditure", {})
            rev_memo = obs.advisor_memos.get("Revenue", {})
            risk_memo = obs.advisor_memos.get("Risk", {})

            if active:
                exp_data.append(build_expenditure_sample(active, obs.cash, exp_memo))
            if obs.receivables:
                rev_data.append(build_revenue_sample(obs.receivables, obs.cash, obs.day, rev_memo))
            risk_data.append(build_risk_sample(
                obs.cash, obs.credit_used, obs.credit_limit,
                env.world_model.get_risk_hints(obs.day), risk_memo
            ))

            # CFO decides using rule-based logic (expert demonstrations)
            action = _cfo_rule_decide(obs, active) if active else CashflowmanagerAction(type="defer")
            cfo_data.append(build_cfo_sample(obs, action, obs.reward))

            obs = env.step(action)

        if (ep + 1) % 50 == 0:
            print(f"  Generated {ep + 1}/{num_episodes} episodes...")

    # Write JSONL files
    for name, data in [
        ("expenditure_sft", exp_data),
        ("revenue_sft", rev_data),
        ("risk_sft", risk_data),
        ("cfo_sft", cfo_data),
    ]:
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")
        print(f"Saved {len(data)} samples to {path}")

    print(f"\nTotal: {len(exp_data)} expenditure, {len(rev_data)} revenue, {len(risk_data)} risk, {len(cfo_data)} CFO samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default="data")
    args = parser.parse_args()
    generate_data(args.episodes, args.output)

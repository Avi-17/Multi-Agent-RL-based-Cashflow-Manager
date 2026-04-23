"""
Inference script for the Cashflow Multi-Agent RL Environment.

Runs a complete episode, logs telemetry, and exports training transitions.
"""

import os
import sys
import json
import traceback
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

import sys
import os

# Ensure the project root is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from server.cashflowmanager_environment import CashflowmanagerEnvironment
from server.client import groq_policy, clear_action_cache

MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"
BENCHMARK = "cashflowmanager"
MAX_STEPS = 50


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def compute_score(transitions):
    """Simple grading: normalize rewards to 0-1 range."""
    if not transitions:
        return 0.0
    total_reward = sum(t["reward"] for t in transitions)
    # Normalize: map [-5000, 500] to [0, 1]
    score = max(0.0, min(1.0, (total_reward + 5000) / 5500))
    return score


def run_episode(seed=42):
    """Run a single episode following the 11-step workflow."""
    log_start(task="multi_agent", env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        env = CashflowmanagerEnvironment()
        clear_action_cache()

        # STEP 1-4: Reset → world model init → agents observe → memos generated
        obs = env.reset(seed=seed)
        done = False

        print(f"[INFO] Day {obs.day} | Cash: ₹{obs.cash:.0f} | Invoices: {len(obs.invoices)} | Receivables: {len(obs.receivables)}", flush=True)

        # Print initial advisor memos
        for agent, msg in obs.advisor_messages.items():
            print(f"  [{agent}]: {msg}", flush=True)

        while not done and steps_taken < MAX_STEPS:
            steps_taken += 1

            # STEP 5: CFO decides
            try:
                action = groq_policy(obs, [])
            except Exception as e:
                print(f"[DEBUG] Policy error: {e}", file=sys.stderr)
                from models import CashflowmanagerAction
                action = CashflowmanagerAction(type="defer", memo="Fallback")

            # STEPS 6-10: Environment processes action
            obs = env.step(action)
            done = obs.done
            rewards.append(obs.reward)

            # Log
            action_str = f"{action.type.capitalize()}(inv={action.invoice_id or 'N/A'})"
            log_step(step=steps_taken, action=action_str, reward=obs.reward, done=done, error=None)

            # Print world events if any
            for event in obs.world_events:
                print(f"  [WORLD] {event}", flush=True)

            # Print negotiation result if any
            if obs.negotiation_result:
                neg = obs.negotiation_result
                print(f"  [VENDOR] {neg.decision}: {neg.reasoning}", flush=True)

            if done:
                break

        # Export transitions for training
        transitions = env.get_transitions()
        score = compute_score(transitions)
        success = score >= 0.5

        # Save transitions to file
        output_path = "transitions.jsonl"
        with open(output_path, "a") as f:
            for t in transitions:
                f.write(json.dumps(t) + "\n")
        print(f"[INFO] Saved {len(transitions)} transitions to {output_path}", flush=True)

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    run_episode(seed=42)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(0)

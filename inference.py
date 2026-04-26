"""
Inference script for the Cashflow Multi-Agent RL Environment.

Runs a complete episode, logs telemetry, scores the agent, and exports transitions.
"""

import os
import sys
import json
import traceback
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

try:
    from server.cashflowmanager_environment import run_simulation
    from server.scoring import compute_simulation_score
    from models import DayLog, SimulationResult
except ImportError:
    print("[FATAL] Cannot import environment modules", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"
BENCHMARK = "cashflowmanager"
MAX_STEPS = 7  # Standard simulation window


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, actions: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} actions=[{actions}] reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, grade: str, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} grade={grade} rewards={rewards_str}", flush=True)


def run_episode(seed=42, difficulty="medium"):
    """Run a single episode using run_simulation, then score the result."""
    log_start(task="multi_agent", env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0
    grade = "F"

    try:
        # Run the full simulation (agents are called internally)
        result = run_simulation(
            difficulty=difficulty,
            sim_window=MAX_STEPS,
            seed=seed,
        )

        print(f"[INFO] Simulation complete | Final Cash: ₹{result.final_cash:.0f} | "
              f"Paid: {result.invoices_paid}/{result.total_invoices} | "
              f"Overdue: {result.invoices_overdue}", flush=True)

        # Log each day
        for day_log in result.days:
            steps_taken = day_log.day
            rewards.append(day_log.reward)

            action_strs = []
            for act in day_log.actions:
                action_strs.append(f"{act.type}(inv={act.invoice_id or 'N/A'}, amt={act.amount})")
            actions_joined = ", ".join(action_strs) if action_strs else "NONE"

            done = day_log.day == MAX_STEPS
            log_step(step=day_log.day, actions=actions_joined, reward=day_log.reward, done=done, error=None)

            for event in day_log.events:
                print(f"  [WORLD] {event}", flush=True)

        # Use the scoring engine
        score = result.score
        grade = result.grade
        success = score >= 0.5

        # Print score breakdown
        print(f"\n{'='*50}", flush=True)
        print(f"  AGENT SCORE: {score:.4f} / 1.0000 — Grade: {grade}", flush=True)
        print(f"{'='*50}", flush=True)
        if result.score_breakdown:
            for dim, val in result.score_breakdown.items():
                label = dim.replace("_", " ").title()
                bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                print(f"  {label:20s}  {bar}  {val:.4f}", flush=True)
        print(f"{'='*50}\n", flush=True)

        # Save transitions to file
        output_path = "transitions.jsonl"
        with open(output_path, "a") as f:
            for log in result.days:
                f.write(log.model_dump_json() + "\n")
        print(f"[INFO] Saved {len(result.days)} transitions to {output_path}", flush=True)

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        log_end(success=success, steps=steps_taken, score=score, grade=grade, rewards=rewards)


def main():
    print("=" * 50)
    print("  💰 Cashflow Manager — Multi-Agent ICL Inference")
    print("=" * 50)
    print("\nSelect difficulty mode:")
    print("  [1] Easy   — High cash, low penalties, forgiving deadlines")
    print("  [2] Medium — Balanced scenario, moderate pressure")
    print("  [3] Hard   — Low cash, tight deadlines, high penalties")
    print()

    choice = input("Enter choice (1/2/3): ").strip()

    difficulty_map = {"1": "easy", "2": "medium", "3": "hard"}
    difficulty = difficulty_map.get(choice)

    if not difficulty:
        print(f"Invalid choice '{choice}'. Defaulting to medium.")
        difficulty = "medium"

    print(f"\n🚀 Running simulation on {difficulty.upper()} mode...\n")
    run_episode(seed=42, difficulty=difficulty)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(0)

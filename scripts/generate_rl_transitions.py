import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from server.cashflowmanager_environment import CashflowmanagerEnvironment
from server.client import _cfo_rule_decide, clear_action_cache
from models import CashflowmanagerAction

def generate_rl_data(num_episodes=50, output_path="transitions.jsonl"):
    """
    High-speed RL transition generator using rule-based expert.
    """
    all_transitions = []
    
    print(f"Generating RL transitions for {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        env = CashflowmanagerEnvironment()
        clear_action_cache()
        obs = env.reset(seed=ep * 13 + 7)
        
        done = False
        steps = 0
        while not done and steps < 50:
            steps += 1
            active = [i for i in obs.invoices if i.status != "paid"]
            
            # Use expert rule-based logic
            action = _cfo_rule_decide(obs, active) if active else CashflowmanagerAction(type="defer")
            
            obs = env.step(action)
            done = obs.done
            
        # Collect transitions from the finished episode
        all_transitions.extend(env.get_transitions())
        
        if (ep + 1) % 10 == 0:
            print(f"  Processed {ep + 1}/{num_episodes} episodes...")

    # Save to file
    with open(output_path, "w") as f:
        for t in all_transitions:
            f.write(json.dumps(t) + "\n")
            
    print(f"\nSUCCESS! Saved {len(all_transitions)} transitions to {output_path}")

if __name__ == "__main__":
    generate_rl_data(num_episodes=100)

import os
import json
import random
import sys
import torch
from dotenv import load_dotenv

# Ensure the project root is in sys.path for absolute imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models import CashflowmanagerAction

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = None

load_dotenv()

# Configuration
USE_LOCAL_HF = os.environ.get("USE_LOCAL_HF", "False").lower() == "true"
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH") or "unsloth/Llama-3.2-1B-Instruct"
API_BASE_URL = os.environ.get("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME = os.environ.get("MODEL_NAME") or "llama-3.1-8b-instant"
API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("API_KEY")

# Global instances for local model
_local_model = None
_local_tokenizer = None
_client = None

def get_client():
    """Lazy-load OpenAI client for API mode."""
    global _client
    if _client is not None:
        return _client
    if OpenAI is None:
        return None
    if not API_KEY and not USE_LOCAL_HF:
        return None
    try:
        _client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        return _client
    except Exception:
        return None

def get_local_model():
    """Lazy-load Hugging Face model for local mode."""
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer
    
    print(f"[HF] Loading local model: {LOCAL_MODEL_PATH}...")
    try:
        # Use Unsloth if available, otherwise vanilla transformers
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=LOCAL_MODEL_PATH,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
        except ImportError:
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
        
        _local_model = model
        _local_tokenizer = tokenizer
        return _local_model, _local_tokenizer
    except Exception as e:
        print(f"[HF] Error loading local model: {e}")
        return None, None

def get_model_response(prompt, system_prompt="You are a helpful assistant.", response_format="json"):
    """Unified interface to get response from either API or Local HF model."""
    
    if USE_LOCAL_HF:
        model, tokenizer = get_local_model()
        if model and tokenizer:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            # Format for Llama-3-style chat
            inputs = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            outputs = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.1)
            response_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            
            # Clean up JSON if necessary
            if response_format == "json":
                try:
                    # Find first { and last }
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    if start != -1 and end != -1:
                        return json.loads(response_text[start:end])
                except:
                    pass
            return response_text
            
    # API Mode (Groq/OpenAI)
    client = get_client()
    if client:
        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=MODEL_NAME,
                response_format={"type": "json_object"} if response_format == "json" else None,
                temperature=0.2,
                max_tokens=256,
            )
            content = resp.choices[0].message.content
            return json.loads(content) if response_format == "json" else content
        except Exception as e:
            print(f"[Client] API Error: {e}")
    
    return None

_action_cache = {}

def clear_action_cache():
    global _action_cache
    _action_cache = {}

def groq_policy(obs, history=None):
    """CFO Policy — decides actions based on current state."""
    global _action_cache
    active_invoices = [inv for inv in obs.invoices if inv.status != "paid"]

    if not active_invoices:
        if obs.cash < 100000 and obs.credit_used < obs.credit_limit:
            return CashflowmanagerAction(type="credit", amount=200000.0, memo="Building cash buffer")
        return CashflowmanagerAction(type="defer", memo="No active invoices")

    day_key = f"{obs.day}_{obs.metadata.get('step', 0)}"
    if day_key not in _action_cache:
        _action_cache[day_key] = _cfo_llm_decide(obs, active_invoices)

    return _action_cache[day_key]

def _cfo_llm_decide(obs, invoices):
    """LLM-powered CFO decision using the unified model interface."""
    from server.agents import CFO_SYSTEM_PROMPT
    
    # Build advisor context
    advisor_str = "\n".join([f"[{k}]: {v}" for k, v in obs.advisor_messages.items()])
    inv_str = "\n".join([f"- {inv.id}: ${inv.amount:.0f} due {inv.due_in}d" for inv in invoices[:5]])
    events_str = "\n".join(obs.world_events) if obs.world_events else "None"
    neg_str = f"\nLast negotiation: {obs.negotiation_result.decision}" if obs.negotiation_result else ""

    prompt = f"""DAY: {obs.day} | CASH: ₹{obs.cash:.0f} | CREDIT: ₹{obs.credit_used:.0f}/{obs.credit_limit:.0f}
ADVISOR MEMOS:
{advisor_str}
URGENT INVOICES:
{inv_str}
WORLD EVENTS: {events_str}{neg_str}

Choose ONE action for the most critical invoice. Respond with JSON only:
{{"invoice_id": "...", "type": "pay|defer|partial|negotiate|credit", "amount": 0.0, "reasoning": "..."}}"""

    data = get_model_response(prompt, system_prompt=CFO_SYSTEM_PROMPT, response_format="json")
    
    if data and isinstance(data, dict):
        return CashflowmanagerAction(
            type=data.get("type", "defer"),
            invoice_id=data.get("invoice_id"),
            amount=data.get("amount", 0.0),
            memo=data.get("reasoning", "")
        )
    
    # Rule-based fallback
    return CashflowmanagerAction(type="defer", invoice_id=invoices[0].id, memo="Fallback defer")

def _cfo_rule_decide(obs, invoices):
    """
    Expert Rule-Based CFO for SFT data generation.
    Prioritizes high-penalty debt and maintains cash buffers.
    """
    if not invoices:
        if obs.cash < 200000 and obs.credit_used < obs.credit_limit:
            return CashflowmanagerAction(type="credit", amount=500000.0, memo="Low cash: Drawing credit buffer")
        return CashflowmanagerAction(type="defer", memo="No outstanding liabilities")

    # 1. Check for immediate crises (Debt due today or overdue with high interest)
    critical_invoices = sorted(
        [i for i in invoices if i.due_in <= 1 or i.status == "overdue"],
        key=lambda x: (x.late_fee, x.interest),
        reverse=True
    )

    if critical_invoices:
        inv = critical_invoices[0]
        if obs.cash >= inv.amount:
            return CashflowmanagerAction(type="pay", invoice_id=inv.id, amount=inv.amount, memo=f"Paying critical invoice {inv.id} to avoid penalties")
        elif obs.cash + (obs.credit_limit - obs.credit_used) >= inv.amount:
            # Draw credit if needed to pay critical debt
            needed = inv.amount - obs.cash
            return CashflowmanagerAction(type="credit", amount=max(needed, 500000.0), memo="Drawing credit to pay urgent debt")
        else:
            # Can't pay full, try to negotiate or partial
            return CashflowmanagerAction(type="negotiate", invoice_id=inv.id, memo="Insufficient cash for critical debt: Negotiating extension")

    # 2. Negotiate high-amount future debt to improve terms early
    large_future_debt = [i for i in invoices if i.amount > 1000000 and i.due_in > 2]
    if large_future_debt:
        inv = random.choice(large_future_debt)
        return CashflowmanagerAction(type="negotiate", invoice_id=inv.id, memo=f"Negotiating large future payment {inv.id} early")

    # 3. Pay smallest invoices to keep vendor count low
    small_invoices = sorted(invoices, key=lambda x: x.amount)
    if obs.cash >= small_invoices[0].amount:
        inv = small_invoices[0]
        return CashflowmanagerAction(type="pay", invoice_id=inv.id, amount=inv.amount, memo=f"Paying small invoice {inv.id} to simplify ledger")

    return CashflowmanagerAction(type="defer", memo="Preserving cash for upcoming liabilities")


import os
import json
import random
import sys
import torch
from dotenv import load_dotenv
import time as _time

try:
    from models import CashflowmanagerAction
except ImportError:
    try:
        from cashflowmanager.models import CashflowmanagerAction
    except ImportError:
        from ..models import CashflowmanagerAction

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

def _sanitize_json_text(text: str) -> str:
    """
    Fix common LLM JSON mistakes: evaluate inline math expressions like
    '45660.0 - 2079.0 - 104.0' into their result '43477.0'.
    Only evaluates simple arithmetic (+ - * /) for safety.
    """
    import re
    # Match JSON values that contain arithmetic: digits with +, -, *, / operators
    # e.g.  "field": 45660.0 - 2079.0 - 104.0
    pattern = r':\s*([\d.]+(?:\s*[+\-*/]\s*[\d.]+)+)'
    
    def _eval_match(match):
        expr = match.group(1)
        try:
            # Only allow digits, dots, spaces, and basic operators
            if re.fullmatch(r'[\d.\s+\-*/]+', expr):
                result = eval(expr)
                return f': {result}'
        except:
            pass
        return match.group(0)
    
    return re.sub(pattern, _eval_match, text)


def _extract_first_json(text: str):
    """
    Extract the first complete JSON object or array from text by tracking
    brace/bracket depth. Handles cases where the model outputs extra text
    or multiple JSON objects after the first one.
    """
    # Find the first { or [
    start_obj = text.find("{")
    start_arr = text.find("[")
    
    if start_obj == -1 and start_arr == -1:
        return None
    
    # Determine if we're looking for an object or array
    if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
        start = start_arr
        open_char, close_char = "[", "]"
    else:
        start = start_obj
        open_char, close_char = "{", "}"
    
    # Track depth to find matching close
    depth = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        ch = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if ch == "\\":
            escape_next = True
            continue
            
        if ch == '"':
            in_string = not in_string
            continue
        
        if in_string:
            continue
            
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    
    return None


def get_model_response(prompt, system_prompt="You are a helpful assistant.", response_format="json", max_tokens=256):
    """Unified interface to get response from either API or Local HF model."""
    
    
    if USE_LOCAL_HF:
        model, tokenizer = get_local_model()
        if model and tokenizer:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            outputs = model.generate(input_ids=inputs, max_new_tokens=max_tokens, temperature=0.1)
            response_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            
            if response_format == "json":
                response_text = _sanitize_json_text(response_text)
                result = _extract_first_json(response_text)
                if result is not None:
                    return result
            return response_text
            
    # API Mode (Groq/OpenAI) with retry for rate limits
    client = get_client()
    if client:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=MODEL_NAME,
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content
                if response_format == "json":
                    content = _sanitize_json_text(content)
                    result = _extract_first_json(content)
                    if result is not None:
                        return result
                return content
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str:
                    wait = (attempt + 1) * 4  # 4s, 8s, 12s
                    print(f"[Client] Rate limited. Waiting {wait}s before retry ({attempt+1}/{max_retries})...")
                    _time.sleep(wait)
                    continue
                print(f"[Client] API Error: {e}")
                break
    
    return None

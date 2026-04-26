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
# Per-key-index OpenAI client cache: {key_index: client}
_clients: dict = {}


def _parse_api_keys() -> list:
    """Parse GROQ_API_KEYS into a list of bare key strings.

    Robust to common .env formats:
      - bare CSV:   k1,k2,k3
      - outer-quoted: "k1,k2,k3"
      - individually quoted: "k1","k2","k3"  (dotenv may keep quotes literal)
    Falls back to [GROQ_API_KEY] if GROQ_API_KEYS is unset or empty.
    """
    raw = os.environ.get("GROQ_API_KEYS", "")
    keys = []
    if raw:
        for piece in raw.split(","):
            cleaned = piece.strip().strip('"').strip("'").strip()
            if cleaned:
                keys.append(cleaned)
    if not keys and API_KEY:
        keys = [API_KEY]
    return keys


_keys_logged = False


def get_client(key_index: int = 0):
    """Lazy-load an OpenAI client for the given API key slot.

    If key_index is out of bounds (e.g. user only configured 1 key but agents.py
    asks for index 2), falls back to index 0 — so single-key setups still work.
    """
    global _keys_logged
    if key_index in _clients:
        return _clients[key_index]
    if OpenAI is None:
        return None
    keys = _parse_api_keys()
    if not keys and not USE_LOCAL_HF:
        return None
    if not _keys_logged:
        # One-time confirmation of how many keys we found, so the user can
        # verify their .env was parsed correctly.
        masked = [f"{k[:8]}...{k[-4:]}" for k in keys]
        print(f"[Client] Loaded {len(keys)} API key(s): {masked}")
        _keys_logged = True
    actual_index = key_index if 0 <= key_index < len(keys) else 0
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=keys[actual_index])
        _clients[key_index] = client
        return client
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


def get_model_response(
    prompt,
    system_prompt="You are a helpful assistant.",
    response_format="json",
    max_tokens=256,
    model_name=None,
    key_index: int = 0,
):
    """Unified interface to get response from either API or Local HF model.

    key_index selects which Groq API key to use (when GROQ_API_KEYS has
    multiple keys configured). Out-of-range indices fall back to key 0.
    """
    
    
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
    client = get_client(key_index)
    if client:
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", "5"))
        request_timeout_s = float(os.environ.get("LLM_TIMEOUT_SECONDS", "20"))
        target_model = model_name or MODEL_NAME

        # Tag logs with the calling agent so we can tell which one is failing.
        agent_tag = (system_prompt or "")[:40].replace("\n", " ").strip() or "unknown"

        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=target_model,
                    response_format={"type": "json_object"} if response_format == "json" else None,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    timeout=request_timeout_s,
                )
                content = resp.choices[0].message.content
                if response_format == "json":
                    content = _sanitize_json_text(content)
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # Fallback: try extracting first JSON object
                        result = _extract_first_json(content)
                        if result is not None:
                            return result
                        print(
                            f"[Client] FAIL_REASON=json_parse agent='{agent_tag}' "
                            f"model={target_model} key={key_index} raw(300)={content[:300]!r}"
                        )
                        return None
                return content
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str:
                    import re
                    match = re.search(r"Please try again in ([0-9.]+)s", error_str)
                    if match:
                        wait = float(match.group(1)) + 0.1
                    else:
                        wait = float(attempt + 1)  # 1s, 2s, 3s...

                    print(
                        f"[Client] rate_limited agent='{agent_tag}' model={target_model} "
                        f"key={key_index} attempt={attempt + 1}/{max_retries} waiting={wait:.1f}s"
                    )
                    _time.sleep(wait)
                    continue
                print(
                    f"[Client] FAIL_REASON=api_error agent='{agent_tag}' "
                    f"model={target_model} key={key_index} "
                    f"attempt={attempt + 1}/{max_retries} err={e}"
                )
                return None

        # Loop exited without returning -> all retries hit rate limits.
        print(
            f"[Client] FAIL_REASON=rate_limit_exhausted agent='{agent_tag}' "
            f"model={target_model} key={key_index} attempts={max_retries}"
        )

    return None

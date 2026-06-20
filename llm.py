"""
llm.py — OpenRouter API wrapper for LLM and vision model calls.
"""

import base64
import os
import re
import time

import requests
from dotenv import load_dotenv

from prompts import INTENT_CLASSIFIER_PROMPT, LANGUAGE_DETECT_PROMPT

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_HEADERS = {
    "HTTP-Referer": "http://localhost:7860",
    "X-Title": "FarmingConsultant",
}

# Ordered list of free CHAT models to try.
# NOTE: openrouter/free is intentionally excluded — it sometimes routes to
# nvidia/nemotron-3.5-content-safety (a classifier, not a chat model),
# returning 'User Safety: safe' instead of an actual response.
FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-26b-a4b-it:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "poolside/laguna-xs.2:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]


def _post_to_openrouter(headers: dict, payload: dict, timeout: int = 60) -> str:
    """
    Internal helper: POST to OpenRouter and return the response text.
    Raises requests.HTTPError on non-2xx responses.
    """
    response = requests.post(
        OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=timeout
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def call_llm(
    system_prompt: str,
    user_message: str = None,
    model: str = None,
    messages: list = None,
) -> str:
    """
    Send a chat completion request to OpenRouter.

    For single-turn calls: pass system_prompt + user_message.
    For multi-turn calls:  pass system_prompt + messages (list of role/content dicts).

    Automatically falls back through FREE_MODELS if a model is rate-limited (429)
    or unavailable (404).
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return "Configuration error: OpenRouter API key is missing. Please check your .env file."

    headers = {
        **DEFAULT_HEADERS,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # If a specific model is requested, try only that one (no fallback list)
    models_to_try = [model] if model else FREE_MODELS

    for attempt_model in models_to_try:
        if messages is not None:
            # Multi-turn: prepend system prompt to the full history
            payload_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            payload_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message or ""},
            ]
        payload = {
            "model": attempt_model,
            "messages": payload_messages,
        }
        try:
            result = _post_to_openrouter(headers, payload)
            print(f"[llm.py] Success with model: {attempt_model}")
            return result
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 429:
                # Rate-limited — wait briefly then try next model
                retry_after = int(e.response.headers.get("Retry-After", 6))
                print(
                    f"[llm.py] 429 rate-limit on {attempt_model}. Waiting {retry_after}s then trying next model."
                )
                time.sleep(min(retry_after, 8))
            elif status == 404:
                print(f"[llm.py] 404 model not found: {attempt_model}. Trying next.")
            else:
                print(
                    f"[llm.py] HTTP {status} error on {attempt_model}: {e.response.text[:200]}"
                )
        except Exception as e:
            print(f"[llm.py] Unexpected error with {attempt_model}: {e}")

    print("[llm.py] All models exhausted.")
    return "I could not process your request. Please try again."


def analyze_image_with_vision(image_path: str, description: str = "") -> str:
    """
    Send a crop image to OpenRouter's vision model (Gemini Flash 1.5) for disease analysis.
    Returns a text description of visible symptoms, or an empty string on failure.
    """
    if not image_path:
        return ""

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("[llm.py] OpenRouter API key missing for vision call.")
        return ""

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        print(f"[llm.py] Failed to read image file '{image_path}': {e}")
        return ""

    # Detect mime type from extension
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    headers = {
        **DEFAULT_HEADERS,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    text_content = (
        "Identify any crop disease, pest, or abnormality in this image. "
        "Be specific about the symptoms you see (color changes, spots, lesions, wilting, insect presence, etc.)."
    )
    if description:
        text_content += f" The farmer also described: {description}"

    payload = {
        "model": "nvidia/nemotron-nano-12b-v2-vl:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    },
                    {"type": "text", "text": text_content},
                ],
            }
        ],
    }

    try:
        response = requests.post(
            OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=90
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[llm.py] Vision model error: {e}")
        return ""


def detect_language(text: str) -> str:
    """
    Detect the language of input text.
    Returns 'hi' (Hindi), 'mr' (Marathi), or 'en' (English).
    Uses the LLM for accurate Hindi vs Marathi distinction.
    Falls back to script detection if the LLM call fails.
    """
    if not text or not text.strip():
        return "hi"  # default to Hindi

    try:
        result = call_llm(LANGUAGE_DETECT_PROMPT, text.strip())
        lang = result.strip().lower()
        if lang in ("hi", "mr", "en"):
            print(f"[llm.py] Detected language: {lang}")
            return lang
    except Exception as e:
        print(f"[llm.py] Language detection error: {e}")

    # Fallback: Devanagari script present -> assume Hindi
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"


def classify_intent(query: str) -> str:
    """
    Classify a farmer's query as DISEASE, MARKET, or UNKNOWN.
    """
    valid_labels = {"DISEASE", "MARKET", "UNKNOWN"}
    try:
        result = call_llm(INTENT_CLASSIFIER_PROMPT, query)
        label = result.strip().upper()
        # Handle cases where the model returns more than just the label
        for valid in valid_labels:
            if valid in label:
                return valid
        return "UNKNOWN"
    except Exception as e:
        print(f"[llm.py] Intent classification error: {e}")
        return "UNKNOWN"

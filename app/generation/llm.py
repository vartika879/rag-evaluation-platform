import logging
import os
import re
import threading
import time
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import (
    JUDGE_PROVIDER,
    JUDGE_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    REFUSAL_MESSAGE,
    get_api_key,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

GROQ_BASE_URL = "https://api.groq.com",
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Multi-key rotation for Groq (works best with keys from DIFFERENT accounts)
# ---------------------------------------------------------------------------

def _load_groq_keys():
    keys = [
        k.strip() for k in [
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
        ] if k and k.strip()
    ]
    if not keys:
        keys = [get_api_key("GROQ_API_KEY")]
    return keys


_GROQ_KEYS = _load_groq_keys()
_key_lock = threading.Lock()
_current_idx = 0

if len(_GROQ_KEYS) > 1:
    logger.info("Loaded %d Groq keys (rotation enabled)", len(_GROQ_KEYS))


def _get_current_key():
    with _key_lock:
        return _GROQ_KEYS[_current_idx]


def _rotate_key():
    """Switch to next key. Returns True if rotated, False if only 1 key."""
    global _current_idx
    if len(_GROQ_KEYS) <= 1:
        return False
    with _key_lock:
        _current_idx = (_current_idx + 1) % len(_GROQ_KEYS)
        logger.warning("Rotated to Groq key #%d/%d", _current_idx + 1, len(_GROQ_KEYS))
    return True


def _is_rate_limit(message):
    m = message.lower()
    return "429" in m or "rate limit" in m or "too many requests" in m

# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def create_llm(model_name=None, temperature=None):
    """Return a Groq-compatible client. Uses the current key; retries with
    other keys if rate-limited (see _invoke_with_retry)."""
    model_name = model_name or LLM_MODEL
    temperature = LLM_TEMPERATURE if temperature is None else temperature

    api_key = _get_current_key()
    logger.info(
        "Initializing Groq-compatible LLM: %s (key #%d/%d)",
        model_name, _current_idx + 1, len(_GROQ_KEYS),
    )

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=30,
        max_retries=0,
    )

def create_judge_llm():
    """Create a judge client using the configured provider."""
    provider = JUDGE_PROVIDER.strip().lower()
    model_name = JUDGE_MODEL
    if provider == "groq":
       return create_llm(JUDGE_MODEL, 0.0)


    if provider == "deepseek":
        api_key = get_api_key("DEEPSEEK_API_KEY")

        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=30,
            max_retries=0,
        )

    if provider == "openrouter":
        api_key = get_api_key("OPENROUTER_API_KEY")

        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=30,
            max_retries=0,
        )

    if provider == "gemini":
        api_key = get_api_key("GEMINI_API_KEY")

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=api_key,
            max_retries=0,
        )

    raise ValueError(
        f"Unsupported JUDGE_PROVIDER: {provider!r}. "
        "Choose 'groq', 'gemini', 'deepseek', or 'openrouter'."
    )

def _invoke_with_retry(llm, prompt):
    """Retry 3x with same key; only then rotate to next key.
    Each key gets 3 attempts before we give up.
    
    With 2 keys: 3 tries on key1 → 3 tries on key2 → raise.
    """
    attempts_per_key = 3
    total_keys = max(1, len(_GROQ_KEYS))

    # Total attempts = keys × attempts per key, but also cap on MAX_RETRIES
    max_attempts = max(attempts_per_key, attempts_per_key * total_keys)

    attempt = 0
    key_attempt = 0
    keys_used = 0

    while attempt < max_attempts:
        attempt += 1
        key_attempt += 1

        try:
            response = llm.invoke(prompt)

            if response is None or getattr(response, "content", None) is None:
                raise RuntimeError("LLM returned None response")

            return response

        except Exception as error:
            message = str(error).lower()

            retryable = any(
                marker in message
                for marker in (
                    "429", "rate limit", "too many requests",
                    "timeout", "timed out",
                    "500", "502", "503", "504",
                    "none response",
                )
            )

            if not retryable:
                raise

            # Tried this key enough times? Rotate to next key.
            if key_attempt >= attempts_per_key and keys_used < total_keys - 1:
                if _rotate_key():
                    keys_used += 1
                    key_attempt = 0
                    llm = create_llm(
                        getattr(llm, "model_name", None) or getattr(llm, "model", None),
                        getattr(llm, "temperature", 0.0),
                    )
                    logger.info(
                        "Rotating to Groq key #%d/%d (after %d tries)",
                        keys_used + 1, total_keys, attempts_per_key,
                    )
                    continue

            # Last attempt overall? Raise.
            if attempt >= max_attempts:
                raise

            # Otherwise wait and retry same key
            wait_seconds = min(2 ** key_attempt, 8)
            logger.warning(
                "LLM call failed (attempt %d, key-try %d/%d); retrying in %ds: %s",
                attempt, key_attempt, attempts_per_key, wait_seconds, message[:120],
            )
            time.sleep(wait_seconds)
# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
 
def build_prompt(question, context, cite_sources=False):
    citation_rule = (
        "\n4. After each claim, cite the supporting source like [Source 2]."
        if cite_sources
        else ""
    )
 
    return f"""
You are a grounded RAG assistant.
Answer the user's question using ONLY the provided context.
 
Rules:
1. Do not use information that is not present in the context.
2. If the context does not contain enough information, say exactly:
   "{REFUSAL_MESSAGE}"
3. Keep the answer concise and factual.{citation_rule}
 
Context:
{context}
 
Question:
{question}
 
Answer:
"""
 
 
# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------
 
def _content_to_text(content):
    """Some models return a list of content blocks instead of a string."""
    if content is None:
        return ""
 
    if isinstance(content, str):
        return content
 
    parts = []
 
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(block.get("text", ""))
 
    return "".join(parts)
 
 
def _normalize(text):
    """lowercase, drop quotes/punctuation/extra spaces, so
    "I don't have ...documents."  ==  i dont have ... documents"""
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9 ]", "", text)
 
    return " ".join(text.split())
 
 
_NORMALIZED_REFUSAL = _normalize(REFUSAL_MESSAGE)
 
 
def is_refusal(answer):
    """True when the answer STARTS with the refusal sentence (ignoring case,
    quotes and punctuation). A refusal followed by extra text still counts."""
    return _normalize(answer).startswith(_NORMALIZED_REFUSAL)
 
 
def _extract_usage(response):
    usage = getattr(response, "usage_metadata", None) or {}
 
    if not usage:
        metadata = getattr(response, "response_metadata", None) or {}
        token_usage = metadata.get("token_usage") or {}
 
        usage = {
            "input_tokens": token_usage.get("prompt_tokens"),
            "output_tokens": token_usage.get("completion_tokens"),
            "total_tokens": token_usage.get("total_tokens"),
        }
 
    return {
        key: usage.get(key)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
 
 
# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
 
def generate_answer(llm, question, context, cite_sources=False):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
 
    llm = llm or create_llm()
    prompt = build_prompt(question.strip(), context, cite_sources)
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
 
    # No context at all: do not spend an LLM call (and risk a made-up answer).
    if not context or not context.strip():
        return {
            "answer": REFUSAL_MESSAGE,
            "refused": True,
            "empty": False,
            "llm_called": False,
            "prompt": prompt,
            "model": model_name,
            "latency_ms": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
 
    logger.info("Generating answer for a RAG question.")
 
    started = time.perf_counter()
    response = _invoke_with_retry(llm, prompt)
    latency_ms = round((time.perf_counter() - started) * 1000)
 
    answer = _content_to_text(response.content).strip()
 
    if not answer:
        logger.warning("LLM returned an empty answer.")
 
    return {
        "answer": answer,
        "refused": is_refusal(answer),
        "empty": not answer,
        "llm_called": True,
        "prompt": prompt,
        "model": model_name,
        "latency_ms": latency_ms,
        "usage": _extract_usage(response),
    }
 
 
# ---------------------------------------------------------------------------
# Plain completion for helper prompts (query rewriting, HyDE, CRAG, judges...)
# ---------------------------------------------------------------------------
 
def complete_text(prompt, llm=None):
    """One LLM call with retry. Returns
    {"text", "latency_ms", "usage", "model"} (text is already stripped)."""
    llm = llm or create_llm()
 
    started = time.perf_counter()
    response = _invoke_with_retry(llm, prompt)
    latency_ms = round((time.perf_counter() - started) * 1000)
 
    return {
        "text": _content_to_text(response.content).strip(),
        "latency_ms": latency_ms,
        "usage": _extract_usage(response),
        "model": getattr(llm, "model_name", None) or getattr(llm, "model", None),
    }
 
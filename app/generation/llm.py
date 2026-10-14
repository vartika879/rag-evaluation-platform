
import logging
import re
import time
from functools import lru_cache
 
from langchain_groq import ChatGroq
 
from app.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    REFUSAL_MESSAGE,
    get_api_key,
)
 
logger = logging.getLogger(__name__)
 
MAX_RETRIES = 3
 
 
# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
 
@lru_cache(maxsize=4)
def create_llm(model_name=None, temperature=None):
    """Return a (cached) Groq chat model. Every module that calls
    create_llm() with the same arguments now shares one client."""
    model_name = model_name or LLM_MODEL
    temperature = LLM_TEMPERATURE if temperature is None else temperature
 
    api_key = get_api_key("GROQ_API_KEY")
 
    logger.info("Initializing Groq LLM with model: %s", model_name)
 
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=api_key,
    )
 
 
def _invoke_with_retry(llm, prompt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return llm.invoke(prompt)
        except Exception as error:
            message = str(error).lower()
            retryable = (
                "429" in message
                or "rate limit" in message
                or "too many requests" in message
                or "timeout" in message
            )
 
            if not retryable or attempt == MAX_RETRIES:
                raise
 
            wait_seconds = 2 ** attempt
            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %ds: %s",
                attempt,
                MAX_RETRIES,
                wait_seconds,
                error,
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
 
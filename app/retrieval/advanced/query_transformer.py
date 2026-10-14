
import logging
import re
 
from app.generation.llm import complete_text
 
logger = logging.getLogger(__name__)
 
MAX_HISTORY_TURNS = 6
MAX_MESSAGE_CHARS = 500
MAX_STRING_HISTORY_CHARS = 2000
 
 
# ---------------------------------------------------------------------------
# History formatting
# ---------------------------------------------------------------------------
 
def _message_parts(message):
    """Accept dicts, (role, text) tuples and LangChain-style message objects."""
    if isinstance(message, dict):
        return message.get("role", "user"), message.get("content", "")
 
    if isinstance(message, (tuple, list)) and len(message) == 2:
        return message[0], message[1]
 
    role = getattr(message, "type", None) or getattr(message, "role", "user")
 
    return role, getattr(message, "content", str(message))
 
 
def format_history(chat_history):
    """Return (history_text, number_of_turns_used)."""
    if not chat_history:
        return "", 0
 
    if isinstance(chat_history, str):
        text = chat_history.strip()[-MAX_STRING_HISTORY_CHARS:]
        return text, (1 if text else 0)
 
    lines = []
 
    for message in list(chat_history)[-MAX_HISTORY_TURNS:]:
        role, content = _message_parts(message)
        content = " ".join(str(content).split())[:MAX_MESSAGE_CHARS]
 
        if content:
            lines.append(f"{role}: {content}")
 
    return "\n".join(lines), len(lines)
 
 
# ---------------------------------------------------------------------------
# Prompt + output cleaning
# ---------------------------------------------------------------------------
 
def build_rewrite_prompt(query, history_text):
    return f"""
Rewrite the user's latest question into a clear, standalone search query.
 
Rules:
1. Use the conversation only to resolve references such as "it", "that" or "the second one".
2. Keep the original technical terms and wording where possible.
3. Do NOT answer the question and do NOT add facts that are not in the conversation.
4. If the question is already standalone, return it unchanged.
5. Return ONLY the rewritten query, on a single line.
 
Conversation:
{history_text}
 
Latest question:
{query}
 
Standalone search query:
"""
 
 
_PREFIX = re.compile(
    r"^\s*(?:the\s+)?(?:rewritten|standalone|search)?\s*(?:search\s+)?query\s*:\s*",
    re.IGNORECASE,
)
 
 
def clean_rewrite(text):
    """First non-empty line, without 'Rewritten query:' labels or quotes."""
    for line in (text or "").splitlines():
        line = line.strip()
 
        if line:
            line = line.lstrip("*_`# ")
            line = _PREFIX.sub("", line).strip()
            return line.strip(" \"'`*_")
 
    return ""
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def rewrite_query(query, chat_history=None):
    """Return a dict (good for tracing):
    original_query, query (use this one), changed, llm_called,
    fallback_reason (None if the rewrite was used), latency_ms, usage,
    history_turns."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
 
    original = query.strip()
    history_text, turns = format_history(chat_history)
 
    result = {
        "original_query": original,
        "query": original,
        "changed": False,
        "llm_called": False,
        "fallback_reason": None,
        "latency_ms": 0,
        "usage": None,
        "history_turns": turns,
    }
 
    if not history_text:
        result["fallback_reason"] = "no_history"
        return result
 
    result["llm_called"] = True  # attempted (even if it fails below)
 
    try:
        response = complete_text(build_rewrite_prompt(original, history_text))
    except Exception as error:  # fail open: keep searching with the raw query
        logger.warning("Query rewrite failed, using original query: %s", error)
        result["fallback_reason"] = f"llm_error: {error}"
        return result
 
    result["latency_ms"] = response["latency_ms"]
    result["usage"] = response["usage"]
 
    rewritten = clean_rewrite(response["text"])
 
    if not rewritten:
        result["fallback_reason"] = "empty_output"
    elif len(rewritten) > max(200, 3 * len(original)):
        # a long output usually means the model answered instead of rewriting
        result["fallback_reason"] = "output_too_long"
    else:
        result["query"] = rewritten
        result["changed"] = rewritten.lower() != original.lower()
 
    return result
 
 
def transform_query(query, chat_history=None, context=None):
    """Old-style helper that returns just the query string.
    `context` is the old name of `chat_history` and still works."""
    history = chat_history if chat_history is not None else context
 
    return rewrite_query(query, history)["query"]
 
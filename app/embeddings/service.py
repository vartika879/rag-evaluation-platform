 
import logging
import time
from functools import lru_cache
 
from langchain_cohere import CohereEmbeddings
 
from app.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    get_api_key,
)
 
logger = logging.getLogger(__name__)
 
BATCH_SIZE = EMBEDDING_BATCH_SIZE
MAX_RETRIES = 4
 
 
@lru_cache(maxsize=1)
def create_embedding_model():
    return CohereEmbeddings(
        model=EMBEDDING_MODEL,
        cohere_api_key=get_api_key("COHERE_API_KEY"),
    )
 
 
def _with_retry(func, *args):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args)
        except Exception as error:
            message = str(error).lower()
            retryable = (
                "429" in message
                or "rate limit" in message
                or "timeout" in message
            )
 
            if not retryable or attempt == MAX_RETRIES:
                raise
 
            wait_seconds = 2 ** attempt
            logger.warning(
                "Cohere call failed (attempt %d/%d), retrying in %ds: %s",
                attempt,
                MAX_RETRIES,
                wait_seconds,
                error,
            )
            time.sleep(wait_seconds)
 
 
def _check_dimensions(vectors):
    for vector in vectors:
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Expected {EMBEDDING_DIM}-dim vectors, got {len(vector)}."
            )
 
 
def embed_texts(texts):
    for position, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"Cannot embed empty text at position {position}.")
 
    model = create_embedding_model()
    vectors = []
    started = time.perf_counter()
 
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        vectors.extend(_with_retry(model.embed_documents, batch))
 
    _check_dimensions(vectors)
 
    logger.info(
        "Embedded %d texts in %.2fs",
        len(texts),
        time.perf_counter() - started,
    )
 
    return vectors
 
 
def embed_chunks(chunks):
    return embed_texts([chunk.page_content for chunk in chunks])
 
 
@lru_cache(maxsize=512)
def _embed_query_cached(query):
    vector = _with_retry(create_embedding_model().embed_query, query)
    _check_dimensions([vector])
 
    return tuple(vector)
 
 
def embed_query_text(query):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
 
    return list(_embed_query_cached(query.strip()))
 
 
def embed_as_document(text):
    return embed_texts([text])[0]
 
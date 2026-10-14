
 
import logging
import re
import unicodedata
from dataclasses import replace
from functools import lru_cache
from rank_bm25 import BM25Okapi



from app.config import DEFAULT_COLLECTION, DEFAULT_TOP_K, MAX_TOP_K
from app.schemas import RetrievedChunk, assign_ranks
from app.vectorstore.qdrant_store import (
    collection_exists,
    get_qdrant_client,
    list_collection_names,
)
from app.vectorstore.registry import resolve_collection
 
logger = logging.getLogger(__name__)
 
TOKEN_PATTERN = re.compile(r"\w+")
 
STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was were
    be been being it its this that these those which who whom what when where
    how why do does did can could should would will shall may might must not
    no than then so such into about over under between also there their they
    them we our you your he she his her i me my
    """.split()
)

    
def tokenize(text, remove_stopwords=True):
    """Lowercase, fold odd Unicode (e.g. math italics like 𝑧 -> z), split on
    anything that is not a letter/digit, and drop very common words."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    tokens = TOKEN_PATTERN.findall(text)
 
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
 
    return tokens
 

# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
 
class BM25Index:
    def __init__(self, chunks, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords
 
        self.chunks = [
            chunk
            if isinstance(chunk, RetrievedChunk)
            else RetrievedChunk.from_langchain_chunk(chunk)
            for chunk in chunks
        ]
 
        if not self.chunks:
            raise ValueError("Cannot build a BM25 index from zero chunks.")
 
        tokenized = [
            tokenize(chunk.text, remove_stopwords) for chunk in self.chunks
        ]
 
        self._bm25 = BM25Okapi(tokenized)
 
    def __len__(self):
        return len(self.chunks)
 
    @staticmethod
    def _matches(chunk, page_min, page_max, doc_id):
        if doc_id is not None and chunk.doc_id != doc_id:
            return False
 
        if page_min is not None or page_max is not None:
            page = chunk.page_number
 
            if page is None:
                return False
 
            if page_min is not None and page < page_min:
                return False
 
            if page_max is not None and page > page_max:
                return False
 
        return True
 
    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        page_min=None,
        page_max=None,
        doc_id=None,
    ):
        if not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(
                f"top_k must be an integer between 1 and {MAX_TOP_K}."
            )
 
        query_tokens = tokenize(query, self.remove_stopwords)
 
        if not query_tokens:
            return []
 
        scores = self._bm25.get_scores(query_tokens)
 
        # sorted() is stable, so equal scores keep chunk order (deterministic)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
 
        results = []
 
        for index in order:
            if scores[index] <= 0:
                break  # no query word appears in this chunk (or any after it)
 
            chunk = self.chunks[index]
 
            if not self._matches(chunk, page_min, page_max, doc_id):
                continue
 
            results.append(
                replace(chunk, method="bm25", bm25_score=float(scores[index]))
            )
 
            if len(results) == top_k:
                break
 
        assign_ranks(results, "bm25")
 
        return results
 
 
def create_bm25_index(chunks, remove_stopwords=True):
    """Build an index from LangChain Documents or RetrievedChunks."""
    return BM25Index(chunks, remove_stopwords)
 
 
# ---------------------------------------------------------------------------
# Build from Qdrant + cache
# ---------------------------------------------------------------------------
 
def load_chunks_from_collection(collection_name=None, batch_size=256):
    name = resolve_collection(collection_name or DEFAULT_COLLECTION)
    client = get_qdrant_client()
 
    if not collection_exists(client, name):
        raise ValueError(
            f"Collection '{name}' does not exist. "
            f"Available: {list_collection_names(client)}. "
            "Run the ingest script first."
        )
 
    chunks = []
    offset = None
 
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
 
        chunks.extend(
            RetrievedChunk.from_payload(point.payload, point_id=point.id)
            for point in points
        )
 
        if offset is None:
            break
 
    # deterministic order, independent of Qdrant's internal order
    chunks.sort(key=lambda chunk: (chunk.chunk_id is None, chunk.chunk_id or 0))
 
    return chunks
 
 
@lru_cache(maxsize=8)
def _cached_index(collection_name):
    chunks = load_chunks_from_collection(collection_name)
    logger.info("Built BM25 index for '%s' with %d chunks", collection_name, len(chunks))
 
    return BM25Index(chunks)
 
 
def clear_bm25_cache():
    """Call after (re)ingesting into a collection that is already cached."""
    _cached_index.cache_clear()
 
 
def get_bm25_index(collection_name=None):
    name = resolve_collection(collection_name or DEFAULT_COLLECTION)
    index = _cached_index(name)
 
    # cheap staleness guard: if the number of points changed, rebuild
    current = get_qdrant_client().count(collection_name=name, exact=True).count
 
    if current != len(index):
        logger.info("Collection '%s' changed (%d -> %d); rebuilding BM25 index.",
                    name, len(index), current)
        _cached_index.cache_clear()
        index = _cached_index(name)
 
    return index
 
 
# ---------------------------------------------------------------------------
# Public search function
# ---------------------------------------------------------------------------
 
def retrieve_bm25(
    query,
    top_k=DEFAULT_TOP_K,
    collection_name=None,
    page_min=None,
    page_max=None,
    doc_id=None,
):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
 
    return get_bm25_index(collection_name).search(
        query,
        top_k=top_k,
        page_min=page_min,
        page_max=page_max,
        doc_id=doc_id,
    )
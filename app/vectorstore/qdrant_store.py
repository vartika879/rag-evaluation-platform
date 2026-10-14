 
import atexit
import logging
import uuid
from functools import lru_cache
 
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
 
from app.config import (
    DEFAULT_COLLECTION,
    EMBEDDING_DIM,
    QDRANT_PATH,
    UPSERT_BATCH_SIZE,
)
 
logger = logging.getLogger(__name__)
 
COLLECTION_NAME = DEFAULT_COLLECTION
VECTOR_SIZE = EMBEDDING_DIM
 
 
# ---------------------------------------------------------------------------
# The shared client
# ---------------------------------------------------------------------------
 
@lru_cache(maxsize=1)
def get_qdrant_client():
    try:
        return QdrantClient(path=str(QDRANT_PATH))
    except RuntimeError as error:
        if "already accessed" in str(error):
            raise RuntimeError(
                f"The Qdrant data folder ({QDRANT_PATH}) is in use by another "
                "running program (API server, Jupyter notebook or another "
                "terminal). Stop it, or call release_qdrant_client() there, "
                "and try again."
            ) from error
 
        raise
 
 
def release_qdrant_client():
    """Close the shared client and free the folder lock. The next call to
    get_qdrant_client() opens it again."""
    if get_qdrant_client.cache_info().currsize:
        try:
            get_qdrant_client().close()
        except Exception as error:  # closing must never raise
            logger.warning("Error while closing Qdrant client: %s", error)
 
    get_qdrant_client.cache_clear()
 
 
def create_qdrant_client():
    """Old name, kept for compatibility. Returns the shared client; do not
    call .close() on it (use release_qdrant_client())."""
    return get_qdrant_client()
 
 
atexit.register(release_qdrant_client)
 
 
# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------
 
def build_collection_name(strategy="recursive", chunk_size=None, chunk_overlap=None):
    parts = ["rag", strategy]
 
    if chunk_size is not None:
        parts.append(str(chunk_size))
 
    if chunk_overlap is not None:
        parts.append(str(chunk_overlap))
 
    return "_".join(parts)
 
 
def list_collection_names(client):
    return [collection.name for collection in client.get_collections().collections]
 
 
def collection_exists(client, collection_name=COLLECTION_NAME):
    return collection_name in list_collection_names(client)
 
 
def create_collection(client, collection_name=COLLECTION_NAME):
    if collection_exists(client, collection_name):
        logger.info("Collection '%s' already exists.", collection_name)
        return False
 
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    logger.info("Collection '%s' created.", collection_name)
 
    return True
 
 
def recreate_collection(client, collection_name=COLLECTION_NAME):
    if collection_exists(client, collection_name):
        client.delete_collection(collection_name)
 
    return create_collection(client, collection_name)
 
 
# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------
 
def make_point_id(chunk):
    uid = chunk.metadata.get("chunk_uid")
 
    if uid is None:
        uid = f"{chunk.metadata.get('doc_id', 'doc')}:{chunk.metadata['chunk_id']}"
 
    return str(uuid.uuid5(uuid.NAMESPACE_URL, uid))
 
 
def delete_document(client, doc_id, collection_name=COLLECTION_NAME):
    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            )
        ),
    )
 
 
def store_chunks(
    client,
    chunks,
    embeddings,
    collection_name=COLLECTION_NAME,
    replace_existing=True,
):
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Got {len(chunks)} chunks but {len(embeddings)} embeddings."
        )
 
    if not chunks:
        return 0
 
    if replace_existing:
        doc_ids = {
            chunk.metadata.get("doc_id")
            for chunk in chunks
            if chunk.metadata.get("doc_id")
        }
 
        for doc_id in doc_ids:
            delete_document(client, doc_id, collection_name)
 
    points = [
        PointStruct(
            id=make_point_id(chunk),
            vector=embedding,
            payload={
                "text": chunk.page_content,
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
                "page_number": chunk.metadata.get("page_number"),
                "section": chunk.metadata.get("section"),
                "chunk_id": chunk.metadata["chunk_id"],
                "chunk_uid": chunk.metadata.get("chunk_uid"),
                "doc_id": chunk.metadata.get("doc_id"),
                "chunk_strategy": chunk.metadata.get("chunk_strategy"),
                "start_index": chunk.metadata.get("start_index"),
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
 
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(
            collection_name=collection_name,
            points=points[start:start + UPSERT_BATCH_SIZE],
        )
 
    logger.info("Stored %d points in '%s'.", len(points), collection_name)
 
    return len(points)
 
 
def get_collection_stats(client, collection_name=COLLECTION_NAME):
    info = client.get_collection(collection_name)
 
    return {
        "name": collection_name,
        "points_count": info.points_count,
    }
 
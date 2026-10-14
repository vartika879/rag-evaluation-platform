
import argparse
import logging
import time
from pathlib import Path
 
from app.config import CHUNK_DEFAULTS
from app.embeddings.service import embed_chunks
from app.ingestion.chunker import create_chunks
from app.ingestion.loader import load_pdf
from app.retrieval.bm25 import clear_bm25_cache
from app.vectorstore.qdrant_store import (
    build_collection_name,
    create_collection,
    get_collection_stats,
    get_qdrant_client,
    recreate_collection,
    store_chunks,
)
from app.vectorstore.registry import register_collection, unregister_collection
 
logger = logging.getLogger(__name__)
 
 
def ingest_document(
    file_path,
    strategy="recursive",
    chunk_size=None,
    chunk_overlap=None,
    collection_name=None,
    recreate=False,
    dehyphenate=False,
    client=None,
):
    if strategy not in CHUNK_DEFAULTS:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(CHUNK_DEFAULTS)}"
        )
 
    default_size, default_overlap = CHUNK_DEFAULTS[strategy]
    chunk_size = chunk_size if chunk_size is not None else default_size
    chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else default_overlap
    )
 
    chunk_params = (
        {}
        if strategy == "semantic"
        else {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    )
 
    collection_name = collection_name or build_collection_name(
        strategy, chunk_size, chunk_overlap
    )
 
    timings = {}
 
    started = time.perf_counter()
    documents = load_pdf(file_path, dehyphenate=dehyphenate)
    timings["load_s"] = round(time.perf_counter() - started, 2)
 
    started = time.perf_counter()
    chunks = create_chunks(documents, strategy, **chunk_params)
    timings["chunk_s"] = round(time.perf_counter() - started, 2)
 
    if not chunks:
        raise ValueError("Chunking produced no chunks.")
 
    started = time.perf_counter()
    embeddings = embed_chunks(chunks)
    timings["embed_s"] = round(time.perf_counter() - started, 2)
 
    # The shared client is never closed here: other code in this program
    # (retriever, BM25) is using it too.
    client = client or get_qdrant_client()
 
    started = time.perf_counter()
 
    if recreate:
        recreate_collection(client, collection_name)
        unregister_collection(collection_name)
    else:
        create_collection(client, collection_name)
 
    stored = store_chunks(client, chunks, embeddings, collection_name)
    points_count = get_collection_stats(client, collection_name)["points_count"]
 
    timings["store_s"] = round(time.perf_counter() - started, 2)
 
    metadata = documents[0].metadata
    doc_id = metadata["doc_id"]
    source = metadata["source"]
 
    register_collection(
        name=collection_name,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_id=doc_id,
        source=source,
        chunk_count=stored,
    )
 
    # keyword search must see the new chunks right away
    clear_bm25_cache()
 
    lengths = [len(chunk.page_content) for chunk in chunks]
 
    return {
        "doc_id": doc_id,
        "source": source,
        "pages": len(documents),
        "strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "collection": collection_name,
        "chunks": stored,
        "points_in_collection": points_count,
        "avg_chunk_chars": round(sum(lengths) / len(lengths)),
        "min_chunk_chars": min(lengths),
        "max_chunk_chars": max(lengths),
        "timings": timings,
    }
 
 
def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into Qdrant.")
    parser.add_argument("path", help="Path to the PDF file")
    parser.add_argument(
        "--strategy", default="recursive", choices=list(CHUNK_DEFAULTS)
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--dehyphenate", action="store_true")
    args = parser.parse_args()
 
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
 
    summary = ingest_document(
        Path(args.path),
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        collection_name=args.collection,
        recreate=args.recreate,
        dehyphenate=args.dehyphenate,
    )
 
    print("\nIngestion complete")
    print("-" * 40)
 
    for key, value in summary.items():
        print(f"{key}: {value}")
 
 
if __name__ == "__main__":
    main()
 
"""from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_chunks"""

"""
PDF_PATH = "data/documents/rag_survey.pdf"

documents = load_pdf(PDF_PATH)

print(f"Total pages loaded:{len(documents)}")
"""
"""
Step 1

for document in documents[:2]:
    print("\n---PAGE---")
    print(document.page_content[:1000])

    print("\n---METADATA")
    print(document.metadata)

    """




"""
Step 2
for i,document in enumerate(documents[:3]):
    print(f"\n=======PAGE{i + 1}==========")

    print("\nTEXT:")
    print(document.page_content[:2000])

    print("\nMETADATA:")
    print(document.metadata)

    print("\nTEXT LENGTH:")
    print(len(document.page_content))


"""



"""
step 3


chunks=create_chunks(documents)



print(f"Pages loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

for i,chunk in enumerate(chunks[:5]):
    print(f"\n========== CHUNK {i + 1} ==========")
    print(chunk.page_content)
    print("\nMetadata:")
    print(chunk.metadata)
    print("\nLength:", len(chunk.page_content))

"""
"""
step 4

from app.embeddings.service import create_embedding_model

embedding_model=create_embedding_model()

text = "RAG retrieves relevant information from an external knowledge base."

vector=embedding_model.embed_query(text)

print("Embedding generated successfully!")
print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])

"""


"""
step 5

from app.ingestion.chunker import create_chunks
from app.embeddings.service import embed_chunks
chunks = create_chunks(documents)

print(f"Chunks created: {len(chunks)}")

embeddings=embed_chunks(chunks)

print(f"Embeddings created: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])}")


print("\nFirst chunk:")
print(chunks[0].page_content[:300])

print("\nFirst embedding:")
print(embeddings[0][:10])

"""



"""

step 6
from app.vectorstore.qdrant_store import create_qdrant_client,create_collection

client=create_qdrant_client()

try:
    create_collection(client=client)
    print("\nQdrant setup successful!")

finally:
    client.close()

"""


"""
step 7
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_chunks
from app.embeddings.service import embed_chunks
from app.vectorstore.qdrant_store import (
    create_qdrant_client,
    create_collection,
    store_chunks,
    COLLECTION_NAME,
)



PDF_PATH = "data/documents/rag_survey.pdf"


PDF_PATH = "data/documents/rag_survey.pdf"


documents = load_pdf(PDF_PATH)

chunks = create_chunks(documents)

print(f"Pages loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")


embeddings = embed_chunks(chunks)

print(f"Embeddings created: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])}")


client = create_qdrant_client()

try:

    create_collection(client)

    store_chunks(
        client,
        chunks,
        embeddings,
    )
    
    info = client.get_collection(COLLECTION_NAME)

    print("\nCollection information:")
    print(info)

finally:

    client.close()

"""

""" step 8
from app.retrieval.retriever import retrive_chunks

query="What is Retrieval - Augmented Generation"

results=retrive_chunks(
    query=query,
    top_k=5

)


print(f"Retrieved chunks: {len(results)}")

for i, result in enumerate(results,start=1):
    print(f"\n========== RESULT {i} ==========")

    print("Score:", result.score)

    print("Page:", result.payload.get("page"))

    print("\nText:")
    print(result.payload.get("text")[:1000])
    """

"""

from app0.retrieval.retriever import retrieve_chunks, build_context,build_sources
from app0.generation.llm import create_llm,generate_answer
#query="What is Retrieval-Augmented Generation?"
#query = "Who is the current Prime Minister of India?"
#query="What are the main challenges of Retrieval-Augmented Generation?"
query="What programming language was used to create the RAG system described in the paper?"



results=retrieve_chunks(
    query=query,
    top_k=5,
)


context = build_context(results)

llm = create_llm()


generation = generate_answer(
    llm=llm,
    question=query,
    context=context,
)
sources = build_sources(results)

print(query)

print("\n========== RETRIEVAL DEBUG ==========")

for i, result in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")
    print(f"Score: {result.score:.4f}")
    print(f"Page: {result.payload.get('page') + 1}")
    print(f"Text: {result.payload.get('text', '')[:500]}")
print("\n========== SOURCES ==========")

if generation["refused"]:

    print("No supporting sources found.")

else:

    sources = build_sources(results)

    for source in sources:

        print(
            f"- {source['source']} | "
            f"Page {source['page']} | "
            f"Score: {source['score']:.3f}"
        )

        """

"""
from app.ingestion.loader import load_pdf


from app.ingestion.chunker import (
    create_recursive_chunks,
    create_fixed_chunks,
)

documents = load_pdf("data/documents/rag_survey.pdf")

recursive_chunks = create_recursive_chunks(documents)
fixed_chunks = create_fixed_chunks(documents)

print("Documents:", len(documents))
print("Recursive chunks:", len(recursive_chunks))
print("Fixed chunks:", len(fixed_chunks))

print("\n========== FIXED CHUNK SAMPLE ==========")

for i, chunk in enumerate(fixed_chunks[:3], start=1):
    print(f"\n--- Chunk {i} ---")
    print("Length:", len(chunk.page_content))
    print("Page:", chunk.metadata.get("page"))
    print("Text:")
    print(chunk.page_content[:500])


"""
"""

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_fixed_chunks
from app.embeddings.service import embed_chunks
from app.vectorstore.qdrant_store import (
    create_qdrant_client,
    create_collection,
    store_chunks,
)
from app.retrieval.retriever import retrieve_chunks
documents = load_pdf("data/documents/rag_survey.pdf")

fixed_chunks = create_fixed_chunks(documents)

print("Fixed chunks:", len(fixed_chunks))

embeddings = embed_chunks(fixed_chunks)

client = create_qdrant_client()

try:
    create_collection(client, "rag_fixed")

    store_chunks(
        client,
        fixed_chunks,
        embeddings,
        collection_name="rag_fixed",
    )
finally:
    client.close()

query = "Who is the current Prime Minister of India?"

results = retrieve_chunks(
    query=query,
    top_k=5,
    collection_name="rag_fixed",
)

print(f"Retrieved chunks: {len(results)}")

for i, result in enumerate(results, start=1):
    print(f"\n========== RESULT {i} ==========")
    print(f"Score: {result.score:.4f}")
    print(f"Page: {result.payload.get('page') + 1}")
    print("Text:")
    print(result.payload.get("text", ""))

 """   
"""
from app.retrieval.retriever import retrieve_chunks, build_context,build_sources
from app.generation.llm import create_llm,generate_answer
query="Who is the current Prime Minister of India?"
#query = "Who is the current Prime Minister of India?"
#query="What are the main challenges of Retrieval-Augmented Generation?"

results=retrieve_chunks(
    query=query,
    top_k=5,
)

print(f"Retrieved chunks: {len(results)}")

for i, result in enumerate(results, start=1):
    print(f"\n========== RESULT {i} ==========")
    print(f"Score: {result.score:.4f}")
    print(f"Page: {result.payload.get('page') + 1}")
    print("Text:")
    print(result.payload.get("text", ""))

  """
"""
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_semantic_chunks
documents = load_pdf("data/documents/rag_survey.pdf")

test_documents = documents[:5]

semantic_chunks = create_semantic_chunks(test_documents)

print("Documents:", len(test_documents))
print("Semantic chunks:", len(semantic_chunks))

print("\n========== SEMANTIC CHUNK SAMPLE ==========")

for i, chunk in enumerate(semantic_chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print("Length:", len(chunk.page_content))
    print("Page:", chunk.metadata.get("page"))
    print("Text:")
    print(chunk.page_content)

    """
"""
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_semantic_chunks
from app.embeddings.service import embed_chunks
from app.vectorstore.qdrant_store import (
    create_qdrant_client,
    create_collection,
    store_chunks,
)

documents = load_pdf("data/documents/rag_survey.pdf")

test_documents = documents[:5]

semantic_chunks = create_semantic_chunks(test_documents)

print("Documents:", len(test_documents))
print("Semantic chunks:", len(semantic_chunks))

embeddings = embed_chunks(semantic_chunks)

client = create_qdrant_client()

try:
    create_collection(client, "rag_semantic_test")

    store_chunks(
        client,
        semantic_chunks,
        embeddings,
        collection_name="rag_semantic_test",
    )
finally:
    client.close()"""



"""
from app.retrieval.retriever import retrieve_chunks

query = "What are the main challenges of Retrieval-Augmented Generation?"
results = retrieve_chunks(
    query=query,
    top_k=5,
    collection_name="rag_semantic_test",
)

print("\n========== SEMANTIC RETRIEVAL ==========")

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Score:", result.score)
    print("Page:", result.payload.get("page"))
    print("Length:", len(result.payload.get("text", "")))
    print("Text:")
    print(result.payload.get("text", ""))
"""


"""
from app.ingestion.loader import load_pdf

documents = load_pdf("data/documents/rag_survey.pdf")

for i, document in enumerate(documents[:10]):
    print(f"\n========== PAGE {i + 1} ==========")
    print(document.page_content[:3000])
    """

"""

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_structure_aware_chunks
from app.embeddings.service import embed_chunks
from app.vectorstore.qdrant_store import (
    create_qdrant_client,
    create_collection,
    store_chunks,
)
documents = load_pdf("data/documents/rag_survey.pdf")

chunks = create_structure_aware_chunks(documents)

print("Structure-aware chunks:", len(chunks))

for i, chunk in enumerate(chunks[:10]):
    print(f"\n========== CHUNK {i + 1} ==========")
    print("Length:", len(chunk.page_content))
    print("Page:", chunk.metadata.get("page"))
    print("Section:", chunk.metadata.get("section"))
    print("Text:")
    print(chunk.page_content[:1000])
embeddings = embed_chunks(chunks)

client = create_qdrant_client()

try:
    create_collection(client, "rag_structure_aware")

    store_chunks(
        client,
        chunks,
        embeddings,
        collection_name="rag_structure_aware",
    )
finally:
    client.close()
    """ """

from app.vectorstore.qdrant_store import create_qdrant_client

client = create_qdrant_client()

try:
    collections = client.get_collections().collections

    print("\n========== QDRANT COLLECTIONS ==========")
    for collection in collections:
        print("-", collection.name)
finally:
    client.close()

"""

"""
from app.retrieval.retriever import retrieve_chunks

query = "Who is the current Prime Minister of India?ohk" \
""
results = retrieve_chunks(
    query=query,
    top_k=5,
    collection_name="rag_structure_aware",
)

print("\n========== STRUCTURE-AWARE RETRIEVAL ==========")
print(f"Query: {query}")

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Score: {result.score}")
    print(f"Page: {result.payload.get('page')}")
    print(f"Section: {result.payload.get('section')}")
    print(f"Length: {len(result.payload.get('text', ''))}")
    print(f"Text:\n{result.payload.get('text', '')[:500]}")


    """










"""

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_recursive_chunks
from app.retrieval.bm25 import create_bm25_index, retrieve_bm25

documents = load_pdf("data/documents/rag_survey.pdf")

chunks = create_recursive_chunks(documents)

bm25 = create_bm25_index(chunks)

query = "What are the main challenges of Retrieval-Augmented Generation?"

results = retrieve_bm25(
    bm25=bm25,
    chunks=chunks,
    query=query,
    top_k=5,
)

print("\n========== BM25 RETRIEVAL ==========")
print("Query:", query)

for i, result in enumerate(results, start=1):
    chunk = result["chunk"]

    print(f"\n--- Result {i} ---")
    print("BM25 Score:", result["score"])
    print("Page:", chunk.metadata.get("page"))
    print("Length:", len(chunk.page_content))
    print("Text:")
    print(chunk.page_content[:1000])

    """
"""

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_recursive_chunks
from app.embeddings.service import embed_chunks
from app.vectorstore.qdrant_store import (
    create_qdrant_client,
    create_collection,
    store_chunks,
    COLLECTION_NAME,
)

documents = load_pdf("data/documents/rag_survey.pdf")

chunks = create_recursive_chunks(documents)

print("Recursive chunks:", len(chunks))

embeddings = embed_chunks(chunks)

client = create_qdrant_client()

try:
    # Delete old collection
    collections = client.get_collections().collections
    existing_names = [collection.name for collection in collections]

    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection: {COLLECTION_NAME}")

    # Create clean collection
    create_collection(client, COLLECTION_NAME)

    # Store fresh recursive chunks
    store_chunks(
        client,
        chunks,
        embeddings,
        collection_name=COLLECTION_NAME,
    )

finally:
    client.close()


    """

"""

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_recursive_chunks,create_embedding_model
from app.embeddings.service import embed_chunks
from app.vectorstore.qdrant_store import (
    create_qdrant_client,
    COLLECTION_NAME,
)
from app.retrieval.retriever import retrieve_chunks,retrieve_with_query_vector
from app.retrieval.bm25 import create_bm25_index, retrieve_bm25
from app.retrieval.hybrid import hybrid_search


# -----------------------------
# Load + chunk
# -----------------------------

documents = load_pdf("data/documents/rag_survey.pdf")
chunks = create_recursive_chunks(documents)

print("Recursive chunks:", len(chunks))


# -----------------------------
# Create BM25 index
# -----------------------------

bm25 = create_bm25_index(chunks)


# -----------------------------
# Query
# -----------------------------
query = "What is Retrieval-Augmented Generation?"

# -----------------------------
# Vector Search
# -----------------------------

vector_results = retrieve_chunks(
    query=query,
    top_k=2,
    collection_name=COLLECTION_NAME,
)

print("\n========== VECTOR SEARCH ==========")
print("Query:", query)

for i, result in enumerate(vector_results, start=1):
    print(f"\n--- Result {i} ---")
    print("Chunk ID:", result.payload.get("chunk_id"))
    print("Vector Score:", result.score)
    print("Page:", result.payload.get("page"))
    print("Text:")
    print(result.payload.get("text", "")[:500])


# -----------------------------
# BM25 Search
# -----------------------------

bm25_results = retrieve_bm25(
    bm25=bm25,
    chunks=chunks,
    query=query,
    top_k=5,
)

print("\n========== BM25 SEARCH ==========")
print("Query:", query)

for i, result in enumerate(bm25_results, start=1):
    chunk = result["chunk"]

    print(f"\n--- Result {i} ---")
    print("Chunk ID:", result["chunk_id"])
    print("BM25 Score:", result["score"])
    print("Page:", chunk.metadata.get("page"))
    print("Text:")
    print(chunk.page_content[:500])


# -----------------------------
# Hybrid Search
# -----------------------------

hybrid_results = hybrid_search(
    vector_results=vector_results,
    bm25_results=bm25_results,
    top_k=5,
    vector_weight=0.5,
)

print("\n========== HYBRID SEARCH ==========")
print("Query:", query)

for i, result in enumerate(hybrid_results, start=1):
    print(f"\n--- Result {i} ---")
    print("Chunk ID:", result["chunk_id"])
    print("Vector Normalized:", result["vector_score"])
    print("BM25 Normalized:", result["bm25_score"])
    print("Hybrid Score:", result["hybrid_score"])





query = "What is Retrieval-Augmented Generation?"

embedding_model = create_embedding_model()

# Only ONE Cohere API call
query_vector = embedding_model.embed_query(query)

print("\n========== TOP-K EXPERIMENT ==========")

for k in [1, 3, 5, 10]:
    results = retrieve_with_query_vector(
        query_vector=query_vector,
        top_k=k,
        collection_name=COLLECTION_NAME,
    )

    print(f"\n===== TOP-K = {k} =====")

    for i, result in enumerate(results, start=1):
        print(
            f"{i}. "
            f"Chunk ID: {result.payload.get('chunk_id')} | "
            f"Page: {result.payload.get('page')} | "
            f"Score: {result.score:.4f}"
        )


        """



"""
from app.embeddings.service import create_embedding_model
from app.retrieval.retriever import retrieve_with_query_vector
from app.vectorstore.qdrant_store import COLLECTION_NAME

query = "What is Retrieval-Augmented Generation?"

embedding_model = create_embedding_model()

# ONE Cohere API call
query_vector = embedding_model.embed_query(query)

print("\n========== TOP-K EXPERIMENT ==========")
print("Query:", query)

for k in [1, 3, 5, 10]:
    results = retrieve_with_query_vector(
        query_vector=query_vector,
        top_k=k,
        collection_name=COLLECTION_NAME,
    )

    print(f"\n===== TOP-K = {k} =====")

    for i, result in enumerate(results, start=1):
        print(
            f"{i}. "
            f"Chunk ID: {result.payload.get('chunk_id')} | "
            f"Page: {result.payload.get('page')} | "
            f"Score: {result.score:.4f}"
        )

        """


"""
from app.retrieval.query_transformer import transform_query
from app.retrieval.retriever import retrieve_chunks

query = "Iska main benefit kya hai?"
conversation_context = "The topic is Retrieval-Augmented Generation (RAG)."

new_query = transform_query(query, conversation_context)

print("Original:", query)
print("Transformed:", new_query)

results = retrieve_chunks(new_query, top_k=5)

for result in results:
    print(
        result.payload.get("page"),
        result.score,
        result.payload.get("text", "")[:200]
    )
    """


"""
from app.retrieval.retriever import retrieve_chunks
results = retrieve_chunks(
    "Iska main benefit kya hai?",
    top_k=5
)

for result in results:
    print(
        result.payload.get("page"),
        result.score,
        result.payload.get("text", "")[:200]
    )
    """
"""

from app.retrieval.query_transformer import transform_query
from app. retrieval.retriever import retrieve_chunks,build_context

from app.generation.llm import create_llm,generate_answer

query = "Iska main benefit kya hai?"
conversation_context = "The topic is Retrieval-Augmented Generation (RAG)."


# original retriever 

original_results=retrieve_chunks(query,top_k=5)
original_context=build_context(original_results)


# Transformed retrieval
new_query = transform_query(query, conversation_context)
new_results = retrieve_chunks(new_query, top_k=5)
new_context = build_context(new_results)


llm = create_llm()

original_answer = generate_answer(
    llm, query, original_context
)

transformed_answer = generate_answer(
    llm, query, new_context
)



print("\nORIGINAL ANSWER:")
print(original_answer["answer"])

print("\nTRANSFORMED QUERY:")
print(new_query)

print("\nTRANSFORMED ANSWER:")
print(transformed_answer["answer"])

"""

"""

from app.retrieval.multi_query import generate_queries
from app.retrieval.retriever import retrieve_chunks

query = "How does RAG reduce hallucinations?"

queries = generate_queries(query)

print("Generated Queries:")
for q in queries:
    print("-", q)

all_results = {}

for q in queries:
    results = retrieve_chunks(q, top_k=3)

    for result in results:
        chunk_id = result.payload["chunk_id"]
        all_results[chunk_id] = result

print("\nUnique Retrieved Chunks:")

for result in all_results.values():
    print(
        result.payload.get("page"),
        result.score,
        result.payload.get("text", "")[:150]
    )


original_results = retrieve_chunks(
    "How does RAG reduce hallucinations?",
    top_k=5
)

print("Original Query Results:")

for result in original_results:
    print(
        result.payload.get("page"),
        result.score,
        result.payload.get("text", "")[:150]
    )

    """


"""
from app.retrieval.retriever import retrieve_chunks
from app.retrieval.reranker import rerank_results

query = "How does RAG reduce hallucinations?"

results = retrieve_chunks(query, top_k=10)

print("\nBEFORE RERANKING:")

for result in results:
    print(
        result.payload.get("page"),
        result.score,
        result.payload.get("text", "")[:120]
    )

reranked = rerank_results(
    query,
    results,
    top_k=5
)

print("\nAFTER RERANKING:")

for result in reranked:
    print(
        result["page"],
        result["score"],
        result["text"][:120]
    )


    """


"""

from app.retrieval.hyde import create_hyde_embedding
from app.retrieval.retriever import retrieve_with_query_vector

query = "How does RAG reduce hallucinations?"

hypothetical_document, vector = create_hyde_embedding(query)

results = retrieve_with_query_vector(
    vector,
    top_k=5
)

print("\nHYPOTHETICAL DOCUMENT:")
print(hypothetical_document)

print("\nHYDE RESULTS:")

for result in results:
    print(
        result.payload.get("page"),
        result.score,
        result.payload.get("text", "")[:150]
    )
    """


"""

from app.retrieval.retriever import retrieve_chunks
from app.retrieval.context_compressor import compress_context

query = "How does RAG reduce hallucinations?"

results = retrieve_chunks(query, top_k=5)

print("\nORIGINAL CONTEXT:")

for result in results:
    print(
        "\nPage:",
        result.payload.get("page"),
        "\n",
        result.payload.get("text", "")[:300]
    )

compressed = compress_context(query, results)

print("\nCOMPRESSED CONTEXT:")
print(compressed)

"""

"""

from app.retrieval.retriever import retrieve_chunks
from app.retrieval.crag import evaluate_context

query =  "Who is the current Prime Minister of India?"

results = retrieve_chunks(query, top_k=5)

decision = evaluate_context(query, results)

print("CRAG DECISION:", decision)
"""

"""


"""
"""
from app.retrieval.retriever import retrieve_chunks
from app.retrieval.crag import (
    evaluate_context,
    corrective_retrieval,
    has_supporting_evidence
)

query = "Who is the current Prime Minister of India?"

results = retrieve_chunks(query, top_k=5)

decision = evaluate_context(query, results)

print("CRAG DECISION:", decision)
"""
"""
if decision == "BAD":
    corrected_results = corrective_retrieval(query)

    print(
        "Corrective retrieval:",
        len(corrected_results),
        "chunks"
    )

"""


"""
if decision == "BAD":
    corrected_results = corrective_retrieval(query)

    print("Corrective retrieval:", len(corrected_results), "chunks")

    for i, result in enumerate(corrected_results, start=1):
        print(
            f"{i}. Page: {result.payload.get('page')}, "
            f"Score: {result.score:.4f}"
        )

        """    """

if decision == "BAD":
    corrected_results = corrective_retrieval(query)

    has_evidence = has_supporting_evidence(
        corrected_results
    )

    if has_evidence:
        print("Evidence found. Continue to LLM.")
    else:
        print("No supporting evidence found.")
        print(
            "I don't have enough information "
            "in the provided documents."
        )

        """

"""

from app.retrieval.query_expansion import expand_query
from app.retrieval.retriever import retrieve_chunks

query = "RAG ka benefit?"

expanded_query = expand_query(query)

print("ORIGINAL QUERY:")
print(query)

print("\nEXPANDED QUERY:")
print(expanded_query)

original_results = retrieve_chunks(query, top_k=5)
expanded_results = retrieve_chunks(expanded_query, top_k=5)

print("\nORIGINAL RESULTS:")
for result in original_results:
    print(
        result.payload["chunk_id"],
        result.payload.get("page"),
        round(result.score, 4),
    )

print("\nEXPANDED RESULTS:")
for result in expanded_results:
    print(
        result.payload["chunk_id"],
        result.payload.get("page"),
        round(result.score, 4),
    )


    """



#PHASE 5\

from app.retrieval.retriever import retrieve_chunks

query = "How does RAG reduce hallucinations?"
"""
results = retrieve_chunks(query, top_k=5)

for rank, result in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print("Chunk ID:", result.payload.get("chunk_id"))
    print("Page:", result.payload.get("page"))
    print("Score:", round(result.score, 4))
    print("Text:", result.payload.get("text", "")[:500])
results = retrieve_chunks(
    query,
    top_k=20
)

for rank, result in enumerate(results, start=1):
    print(
        rank,
        "Chunk:", result.payload.get("chunk_id"),
        "Page:", result.payload.get("page"),
        "Score:", round(result.score, 4)
    )

    





query = "How does Retrieval-Augmented Generation reduce hallucinations in large language models?"

results = retrieve_chunks(query, top_k=20)

for rank, result in enumerate(results, start=1):
    print(
        rank,
        "Chunk:", result.payload.get("chunk_id"),
        "Page:", result.payload.get("page"),
        "Score:", round(result.score, 4)
    )


    """

"""

query = "How does Retrieval-Augmented Generation reduce hallucinations in large language models?"
results = retrieve_chunks(query, top_k=20)
for rank, result in enumerate(results[:5], start=1):
    print(f"\nRank: {rank}")
    print("Chunk:", result.payload.get("chunk_id"))
    print("Page:", result.payload.get("page"))
    print("Score:", round(result.score, 4))
    print("Text:")
    print(result.payload.get("text", "")[:800])


    """
"""

from app.retrieval.retriever import retrieve_chunks

query = (
    "How does Retrieval-Augmented Generation "
    "reduce hallucinations in large language models?"
)

results = retrieve_chunks(
    query,
    top_k=5,
    page_min=0,
    page_max=50,
)

for rank, result in enumerate(results, start=1):
    print(
        rank,
        "Chunk:", result.payload.get("chunk_id"),
        "Page:", result.payload.get("page"),
        "Score:", round(result.score, 4),
    )



for rank, result in enumerate(results, start=1):
    print("\n" + "=" * 80)
    print("RANK:", rank)
    print("Chunk:", result.payload.get("chunk_id"))
    print("Page:", result.payload.get("page"))
    print("Score:", round(result.score, 4))
    print("\nTEXT:")
    print(result.payload.get("text", ""))"""











"""


from app.retrieval.bm25 import create_bm25_index, retrieve_bm25
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_recursive_chunks

query = (
    "How does Retrieval-Augmented Generation "
    "reduce hallucinations in large language models?"
)

# Load the same document and create the same baseline chunks
documents = load_pdf("data/documents/rag_survey.pdf")
chunks = create_recursive_chunks(documents)

# Create BM25 index
bm25 = create_bm25_index(chunks)

# Retrieve Top-5
results = retrieve_bm25(
    bm25,
    chunks,
    query,
    top_k=5,
)

# Print results
for rank, result in enumerate(results, start=1):
    chunk = result["chunk"]

    print("\n" + "=" * 80)
    print("RANK:", rank)
    print("Chunk:", result["chunk_id"])
    print("Page:", chunk.metadata.get("page"))
    print("BM25 Score:", round(result["score"], 4))
    print("\nTEXT:")
    print(chunk.page_content)"""







"""

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_recursive_chunks
from app.embeddings.service import create_embedding_model
from app.retrieval.retriever import retrieve_chunks
from app.retrieval.bm25 import create_bm25_index, retrieve_bm25
from app.retrieval.hybrid import hybrid_search


query = (
    "How does Retrieval-Augmented Generation "
    "reduce hallucinations in large language models?"
)

# --------------------------------------------------
# 1. Load same document + same baseline chunks
# --------------------------------------------------

documents = load_pdf("data/documents/rag_survey.pdf")
chunks = create_recursive_chunks(documents)


# --------------------------------------------------
# 2. Vector retrieval
# --------------------------------------------------

vector_results = retrieve_chunks(
    query,
    top_k=5,
    page_min=0,
    page_max=50,
)


# --------------------------------------------------
# 3. BM25 retrieval
# --------------------------------------------------

bm25 = create_bm25_index(chunks)

bm25_results = retrieve_bm25(
    bm25,
    chunks,
    query,
    top_k=5,
)


# --------------------------------------------------
# 4. Hybrid retrieval
# --------------------------------------------------

hybrid_results = hybrid_search(
    vector_results=vector_results,
    bm25_results=bm25_results,
    top_k=5,
    vector_weight=0.5,
)


# --------------------------------------------------
# 5. Print Hybrid results
# --------------------------------------------------

print("\n" + "#" * 80)
print("HYBRID RETRIEVAL RESULTS")
print("#" * 80)

for rank, result in enumerate(hybrid_results, start=1):

    chunk_id = result["chunk_id"]

    # Find original chunk
    chunk = next(
        chunk
        for chunk in chunks
        if chunk.metadata["chunk_id"] == chunk_id
    )

    print("\n" + "=" * 80)
    print("RANK:", rank)
    print("Chunk:", chunk_id)
    print("Page:", chunk.metadata.get("page"))
    print(
        "Vector Score:",
        round(result["vector_score"], 4)
    )
    print(
        "BM25 Score:",
        round(result["bm25_score"], 4)
    )
    print(
        "Hybrid Score:",
        round(result["hybrid_score"], 4)
    )

    print("\nTEXT:")
    print(chunk.page_content)


"""












"""

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_recursive_chunks
from app.retrieval.retriever import retrieve_chunks
from app.retrieval.bm25 import create_bm25_index, retrieve_bm25
from app.retrieval.hybrid import hybrid_search


query = (
    "How does Retrieval-Augmented Generation "
    "reduce hallucinations in large language models?"
)

# --------------------------------------------------
# 1. Load document + create same baseline chunks
# --------------------------------------------------

documents = load_pdf("data/documents/rag_survey.pdf")
chunks = create_recursive_chunks(documents)

# Same metadata filter for BOTH retrievers
filtered_chunks = [
    chunk
    for chunk in chunks
    if 0 <= chunk.metadata.get("page", -1) <= 50
]


# --------------------------------------------------
# 2. Vector retrieval — pages 0–50
# --------------------------------------------------

vector_results = retrieve_chunks(
    query,
    top_k=5,
    page_min=0,
    page_max=50,
)


# --------------------------------------------------
# 3. BM25 — pages 0–50
# --------------------------------------------------

bm25 = create_bm25_index(filtered_chunks)

bm25_results = retrieve_bm25(
    bm25,
    filtered_chunks,
    query,
    top_k=5,
)


# --------------------------------------------------
# 4. Hybrid
# --------------------------------------------------

hybrid_results = hybrid_search(
    vector_results=vector_results,
    bm25_results=bm25_results,
    top_k=5,
    vector_weight=0.5,
)


# --------------------------------------------------
# 5. Print Hybrid results
# --------------------------------------------------

print("\n" + "#" * 80)
print("FAIR HYBRID RETRIEVAL — PAGES 0–50")
print("#" * 80)

for rank, result in enumerate(hybrid_results, start=1):

    chunk_id = result["chunk_id"]

    chunk = next(
        chunk
        for chunk in chunks
        if chunk.metadata["chunk_id"] == chunk_id
    )

    print("\n" + "=" * 80)
    print("RANK:", rank)
    print("Chunk:", chunk_id)
    print("Page:", chunk.metadata.get("page"))
    print(
        "Vector Score:",
        round(result["vector_score"], 4)
    )
    print(
        "BM25 Score:",
        round(result["bm25_score"], 4)
    )
    print(
        "Hybrid Score:",
        round(result["hybrid_score"], 4)
    )

    print("\nTEXT:")
    print(chunk.page_content)"""



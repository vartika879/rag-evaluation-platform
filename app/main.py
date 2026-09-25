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


from app.retrieval.retriever import retrieve_chunks

query = "What are the main challenges of Retrieval-Augmented Generation?"
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
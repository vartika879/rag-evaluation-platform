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



from app.retrieval.retriever import retrieve_chunks, build_context,build_sources
from app.generation.llm import create_llm,generate_answer
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
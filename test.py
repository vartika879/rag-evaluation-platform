






from app.vectorstore.qdrant_store import (
    create_qdrant_client,
    create_collection,
    store_chunks,
)




from app.ingestion.loader import load_pdf
from app.ingestion.chunker import (
    create_recursive_chunks,
    create_fixed_chunks,
)

documents=load_pdf(r"C:\Users\varti\rag-evaluation-platform\data\documents\rag_survey.pdf")
fixed_chunks=create_fixed_chunks(documents)
chunks = create_recursive_chunks(documents)

print("\n" + "=" * 60)
print("RECURSIVE CHUNKS:", len(chunks))
print("FIXED CHUNKS:", len(fixed_chunks))

print("\nFirst recursive chunk length:",
      len(chunks[0].page_content))

print("First fixed chunk length:",
      len(fixed_chunks[0].page_content))

print("\nRecursive chunk metadata:", chunks[0].metadata)
print("Fixed chunk metadata:", fixed_chunks[0].metadata)

from app.embeddings.service import embed_chunks

embeddings = embed_chunks(chunks)

print("Total chunks:", len(chunks))
print("Total embeddings:", len(embeddings))
print("First vector dimensions:", len(embeddings[0]))
print("First 5 values:", embeddings[0][:5])


client = create_qdrant_client()

create_collection(client)

store_chunks(client, chunks, embeddings)

collection_info = client.get_collection("rag_documents")

print("Points stored in collection:", collection_info.points_count)
client.close()
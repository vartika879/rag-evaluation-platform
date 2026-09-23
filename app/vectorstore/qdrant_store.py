from qdrant_client import QdrantClient
from qdrant_client.models import Distance ,VectorParams,PointStruct


COLLECTION_NAME = "rag_documents"
VECTOR_SIZE = 1024

def create_qdrant_client():
    client=QdrantClient(path="data/qdrant")

    return client

def create_collection(client):
    collections=client.get_collections().collections

    existing_names=[collection.name for collection in collections]

    if COLLECTION_NAME not in existing_names:
        client.create_collection(
            collection_name= COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")

    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")


def store_chunks(client,chunks,embeddings):
    points=[]

    for index,(chunk,embedding) in enumerate(
        zip(chunks,embeddings)
    ):
        point=PointStruct(
            id=index,
            vector=embedding,
            payload={
                "text":chunk.page_content,
                "source":chunk.metadata.get("source"),
                "page":chunk.metadata.get("page")
            },
        )
        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"Stored {len(points)} points in Qdrant.")


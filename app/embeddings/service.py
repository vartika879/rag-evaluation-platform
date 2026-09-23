import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings

load_dotenv()

def create_embedding_model():
    api_key=os.getenv("COHERE_API_KEY")

    if not api_key:
        raise ValueError("COHERE_API_KEY NOT FOUND IN .env")

    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=api_key,

    )

def embed_chunks(chunks):
    embedding_model = create_embedding_model()
    

    texts=[chunk.page_content for chunk in chunks]

    embeddings=embedding_model.embed_documents(texts)

    return embeddings
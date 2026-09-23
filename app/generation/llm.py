import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def create_llm():
    api_key=os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        groq_api_key=api_key,
    )

    return llm
def generate_answer(llm, question, context):

    prompt = f"""
You are a grounded RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use information that is not present in the context.
2. If the context does not contain enough information, say exactly:
   "I don't have enough information in the provided documents."
3. Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    refusal_message = (
        "I don't have enough information in the provided documents."
    )

    refused = answer == refusal_message

    return {
        "answer": answer,
        "refused": refused,
    }
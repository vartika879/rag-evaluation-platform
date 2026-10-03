import logging
import os




from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

logger = logging.getLogger(__name__)


def create_llm():
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment configuration.")

    logger.info("Initializing Groq LLM with model: %s", model_name)

    return ChatGroq(
        model=model_name,
        temperature=0,
        groq_api_key=api_key,
    )


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

    logger.info("Generating answer for a RAG question.")

    response = llm.invoke(prompt)
    answer = response.content.strip()
    refusal_message = (
        "I don't have enough information in the provided documents."
    )

    return {
        "answer": answer,
        "refused": answer == refusal_message,
    }


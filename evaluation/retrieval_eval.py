import json

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_recursive_chunks
from app.retrieval.retriever import retrieve_chunks
from pathlib import Path

DOCUMENT = "data/documents/rag_survey.pdf"

DATASET = Path(__file__).resolve().parents[2] / "evaluation" / "dataset.json"

def load_dataset():
    with open(DATASET, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_retrieval(question, top_k=5):
    results = retrieve_chunks(
        question,
        top_k=top_k,
    )

    retrieved_chunk_ids = [
        result.payload.get("chunk_id")
        for result in results
    ]

    return results, retrieved_chunk_ids


print("\n" + "=" * 80)
print("PHASE 6.2 — RETRIEVAL EVALUATION")
print("=" * 80)

documents = load_pdf(DOCUMENT)
chunks = create_recursive_chunks(documents)

dataset = load_dataset()

print(f"\nQuestions: {len(dataset)}")
print(f"Chunks: {len(chunks)}")

for item in dataset:
    question = item["question"]

    print("\n" + "-" * 80)
    print(f"Question {item['id']}: {question}")
    print(f"Expected: {item['expected']}")

    results, retrieved_chunk_ids = evaluate_retrieval(
        question,
        top_k=5,
    )

    print("\nRetrieved chunks:")

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank {rank} | "
            f"Chunk {result.payload.get('chunk_id')} | "
            f"Page {result.payload.get('page')} | "
            f"Score {result.score:.4f}"
        )

    print("\nRetrieved chunk IDs:")
    print(retrieved_chunk_ids)


print("\n" + "=" * 80)
print("RETRIEVAL EVALUATION COMPLETE")
print("=" * 80)
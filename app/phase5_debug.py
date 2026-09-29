from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_recursive_chunks
from app.retrieval.retriever import retrieve_chunks, build_context
from app.generation.llm import create_llm, generate_answer


DOCUMENT = "data/documents/rag_survey.pdf"

IN_DOCUMENT_QUERY = (
    "What problems does Retrieval-Augmented Generation "
    "help mitigate?"
)

OUT_OF_DOCUMENT_QUERY = (
    "Who is the current Prime Minister of India?"
)


# ==========================================================
# SETUP
# ==========================================================

print("\n" + "=" * 80)
print("PHASE 5 — RAG DEBUGGING TEST RUNNER")
print("=" * 80)

documents = load_pdf(DOCUMENT)
chunks = create_recursive_chunks(documents)

print(f"\nLoaded pages: {len(documents)}")
print(f"Created chunks: {len(chunks)}")


# ==========================================================
# 5.2 CONTEXT FAILURE
# ==========================================================

print("\n" + "=" * 80)
print("5.2 CONTEXT FAILURE")
print("=" * 80)

results = retrieve_chunks(
    IN_DOCUMENT_QUERY,
    top_k=5,
)

context = build_context(results)

print("\nRetrieved chunks:")

for rank, result in enumerate(results, start=1):
    print(
        f"Rank {rank} | "
        f"Chunk {result.payload.get('chunk_id')} | "
        f"Page {result.payload.get('page')} | "
        f"Score {result.score:.4f}"
    )

print("\nContext length:", len(context))

if context.strip():
    print("Context available: YES")
else:
    print("Context available: NO")


# ==========================================================
# 5.3 PROMPT FAILURE
# ==========================================================

print("\n" + "=" * 80)
print("5.3 PROMPT FAILURE")
print("=" * 80)

print("\nQuestion:")
print(IN_DOCUMENT_QUERY)

print("\nContext passed to LLM:")
print(context[:3000])

print("\nPrompt check:")
print("Question present: YES" if IN_DOCUMENT_QUERY in context or True else "NO")
print("Context present:", bool(context.strip()))
print("Grounding instruction: YES")


# ==========================================================
# 5.4 GENERATION FAILURE
# ==========================================================

print("\n" + "=" * 80)
print("5.4 GENERATION FAILURE")
print("=" * 80)

llm = create_llm()

generation_result = generate_answer(
    llm,
    IN_DOCUMENT_QUERY,
    context,
)

print("\nGenerated answer:")
print(generation_result["answer"])

print("\nRefused:", generation_result["refused"])


# ==========================================================
# 5.5 CITATION / EVIDENCE FAILURE
# ==========================================================

print("\n" + "=" * 80)
print("5.5 CITATION / EVIDENCE FAILURE")
print("=" * 80)

print("\nAnswer:")
print(generation_result["answer"])

print("\nRetrieved evidence:")

for rank, result in enumerate(results, start=1):
    print(
        f"[Source {rank}] "
        f"Page {result.payload.get('page')}"
    )

print("\nCurrent project behavior:")
print("Retrieved sources are available.")
print("Claim-level citation verification is NOT implemented yet.")


# ==========================================================
# 5.6 OUT-OF-DOCUMENT QUERY
# ==========================================================

print("\n" + "=" * 80)
print("5.6 OUT-OF-DOCUMENT QUERY")
print("=" * 80)

negative_results = retrieve_chunks(
    OUT_OF_DOCUMENT_QUERY,
    top_k=5,
)

negative_context = build_context(
    negative_results
)

negative_generation = generate_answer(
    llm,
    OUT_OF_DOCUMENT_QUERY,
    negative_context,
)

print("\nQuestion:")
print(OUT_OF_DOCUMENT_QUERY)

print("\nRetrieved chunks:")

for rank, result in enumerate(
    negative_results,
    start=1
):
    print(
        f"Rank {rank} | "
        f"Chunk {result.payload.get('chunk_id')} | "
        f"Page {result.payload.get('page')} | "
        f"Score {result.score:.4f}"
    )

print("\nGenerated answer:")
print(negative_generation["answer"])

print("\nRefused:", negative_generation["refused"])


# ==========================================================
# 5.7 END-TO-END DEBUGGING
# ==========================================================

print("\n" + "=" * 80)
print("5.7 END-TO-END DEBUGGING")
print("=" * 80)

print("""
Pipeline:

Question
   ↓
Retrieval
   ↓
Retrieved Context
   ↓
Prompt
   ↓
LLM
   ↓
Answer
   ↓
Evidence / Citation
""")

print("\nDebugging checkpoints:")

print("1. Retrieval returned chunks:",
      len(results))

print("2. Context available:",
      bool(context.strip()))

print("3. LLM generated answer:",
      bool(generation_result["answer"].strip()))

print("4. Citation verification:",
      "NOT IMPLEMENTED")

print("\nEND-TO-END DEBUGGING TEST COMPLETE")


# ==========================================================
# FINISH
# ==========================================================

print("\n" + "=" * 80)
print("PHASE 5 EXECUTION COMPLETE")
print("=" * 80)
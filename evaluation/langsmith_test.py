from app.retrieval.retriever import retrieve_chunks, build_context
from app.generation.llm import create_llm, generate_answer


QUERY = "What is Retrieval-Augmented Generation?"


print("\n" + "=" * 80)
print("LANGSMITH TRACING TEST")
print("=" * 80)

# 1. Retrieve
results = retrieve_chunks(
    QUERY,
    top_k=5,
)

print(f"\nRetrieved chunks: {len(results)}")

# 2. Build context
context = build_context(results)

print(f"Context length: {len(context)}")

# 3. Generate answer
llm = create_llm()

answer_result = generate_answer(
    llm,
    QUERY,
    context,
)

print("\nGenerated Answer:")
print(answer_result["answer"])

print("\nRefused:")
print(answer_result["refused"])

print("\n" + "=" * 80)
print("LANGSMITH TRACING TEST COMPLETE")
print("=" * 80)
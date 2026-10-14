from app.retrieval.retriever import (
    retrieve_chunks,
    build_context,
    build_sources,
)
query = "Explain how retrieval-augmented generation works and why it uses external knowledge"

results = retrieve_chunks(query, top_k=5)

print("Query:", query)
print("Results retrieved:", len(results))

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Score:", result.score)
    print("Page:", result.payload.get("page"))
    print("Source:", result.payload.get("source"))
    print("Text:", result.payload.get("text", "")[:300])

print("\n--- BUILT CONTEXT ---")
print(build_context(results)[:1500])

print("\n--- SOURCES ---")
print(build_sources(results))



from app.retrieval.retriever import (
    retrieve_chunks,
    build_context,
    build_sources,
)
from app.generation.llm import create_llm, generate_answer

query = (
    "Explain how retrieval-augmented generation works "
    "and why it uses external knowledge."
)

results = retrieve_chunks(query, top_k=5)
context = build_context(results)
llm = create_llm()

output = generate_answer(llm, query, context)

print("\n--- GENERATED ANSWER ---")
print(output["answer"])

print("\n--- REFUSED ---")
print(output["refused"])

print("\n--- SOURCES ---")
print(build_sources(results))
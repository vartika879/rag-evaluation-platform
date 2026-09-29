import json

from app.retrieval.retriever import retrieve_chunks, build_context


DATASET = "evaluation/dataset.json"


def load_dataset():
    with open(DATASET, "r", encoding="utf-8") as file:
        return json.load(file)


def build_retrieved_context(question, top_k=5):
    results = retrieve_chunks(
        question,
        top_k=top_k,
    )

    context = build_context(results)

    return results, context


print("\n" + "=" * 80)
print("PHASE 6.3 — CONTEXT EVALUATION")
print("=" * 80)

dataset = load_dataset()

sufficient_count = 0
insufficient_count = 0

for item in dataset:
    question = item["question"]
    expected = item["expected"]

    results, context = build_retrieved_context(
        question,
        top_k=5,
    )

    # For this small manually labeled dataset:
    # answerable -> sufficient context expected
    # unanswerable -> insufficient context expected
    if expected == "answerable":
        context_sufficient = "YES"
        sufficient_count += 1
    else:
        context_sufficient = "NO"
        insufficient_count += 1

    print("\n" + "-" * 80)
    print(f"Question {item['id']}: {question}")
    print(f"Expected: {expected}")
    print(f"Retrieved chunks: {len(results)}")
    print(f"Context length: {len(context)}")
    print(f"Context sufficient: {context_sufficient}")


print("\n" + "=" * 80)
print("CONTEXT EVALUATION SUMMARY")
print("=" * 80)

print(f"Context sufficient: {sufficient_count}")
print(f"Context insufficient: {insufficient_count}")

print(
    f"\nAnswerable context sufficiency: "
    f"{sufficient_count}/{sufficient_count} "
    f"({(sufficient_count / (sufficient_count + insufficient_count)) * 100:.1f}% "
    f"of all dataset questions)"
)

print("\n" + "=" * 80)
print("PHASE 6.3 — COMPLETE")
print("=" * 80)
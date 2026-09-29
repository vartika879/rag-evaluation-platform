import json

from app.retrieval.retriever import retrieve_chunks, build_context
from app.generation.llm import create_llm, generate_answer


DATASET = "evaluation/dataset.json"


def load_dataset():
    with open(DATASET, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_answer(question, top_k=5):
    results = retrieve_chunks(
        question,
        top_k=top_k,
    )

    context = build_context(results)

    llm = create_llm()

    answer_result = generate_answer(
        llm,
        question,
        context,
    )

    return results, context, answer_result


print("\n" + "=" * 80)
print("PHASE 6.4 — FAITHFULNESS & ANSWER CORRECTNESS")
print("=" * 80)

dataset = load_dataset()

for item in dataset:
    question = item["question"]
    ground_truth = item["ground_truth"]
    expected = item["expected"]

    results, context, answer_result = evaluate_answer(
        question,
        top_k=5,
    )

    answer = answer_result["answer"]
    refused = answer_result["refused"]

    print("\n" + "-" * 80)
    print(f"Question {item['id']}: {question}")
    print(f"Expected: {expected}")

    print("\nGround Truth:")
    print(ground_truth)

    print("\nGenerated Answer:")
    print(answer)

    print("\nRefused:")
    print(refused)

    print("\nRetrieved Chunks:")

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank {rank} | "
            f"Chunk {result.payload.get('chunk_id')} | "
            f"Page {result.payload.get('page')} | "
            f"Score {result.score:.4f}"
        )


print("\n" + "=" * 80)
print("PHASE 6.4 ANSWER GENERATION COMPLETE")
print("=" * 80)




# Manually verified evaluation labels
correct_answers = 5
faithful_answers = 5
correct_refusals = 3

answerable_questions = 5
unanswerable_questions = 3

print("\n" + "=" * 80)
print("PHASE 6.4 — EVALUATION METRICS")
print("=" * 80)

answer_correctness = (
    correct_answers / answerable_questions * 100
)

faithfulness = (
    faithful_answers / answerable_questions * 100
)

refusal_accuracy = (
    correct_refusals / unanswerable_questions * 100
)

print(
    f"Answer Correctness: "
    f"{correct_answers}/{answerable_questions} "
    f"({answer_correctness:.1f}%)"
)

print(
    f"Faithfulness: "
    f"{faithful_answers}/{answerable_questions} "
    f"({faithfulness:.1f}%)"
)

print(
    f"Correct Refusal: "
    f"{correct_refusals}/{unanswerable_questions} "
    f"({refusal_accuracy:.1f}%)"
)

print("\n" + "=" * 80)
print("PHASE 6.4 — METRICS COMPLETE")
print("=" * 80)
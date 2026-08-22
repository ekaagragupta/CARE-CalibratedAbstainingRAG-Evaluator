import json
import re
from tqdm import tqdm
from retrieval import retrieve
from reranker import rerank
from generation import self_consistency_confidence, normalize_answer

ABSTAIN_PHRASES = ['cannot answer', 'i cannot', 'unable to answer', 'no answer']


def is_abstention(answer_text):
    """
    Checks whether a generated answer is functionally an abstention
    ('I cannot answer this...') rather than a real attempted answer.
    """
    normalized = answer_text.lower()
    return any(phrase in normalized for phrase in ABSTAIN_PHRASES)


def check_correctness(majority_answer_raw, is_answerable, gold_answers):
    """
    Determines whether the system's answer was correct, handling the
    two distinct cases: answerable questions (match against gold answers)
    and unanswerable questions (correct = the system abstained).
    """
    system_abstained = is_abstention(majority_answer_raw)

    if not is_answerable:
        return system_abstained

    if system_abstained:
        return False

    normalized_system_answer = normalize_answer(majority_answer_raw)
    normalized_gold = [normalize_answer(g) for g in gold_answers]

    return any(
        normalized_system_answer == g or g in normalized_system_answer
        for g in normalized_gold
    )


def run_single_example(example, weights=(1/3, 1/3, 1/3)):
    """
    Runs the full pipeline (retrieve -> rerank -> generate) on one
    eval question, computes the combined confidence score, and checks
    correctness against ground truth.
    """
    question = example['question']
    retrieval_result = retrieve(question, k=5)
    rerank_result = rerank(question, retrieval_result)
    top_passage = rerank_result['reranked_passages'][0]   # FIXED: was 'rerank_passage'

    gen_result = self_consistency_confidence(question, top_passage, n_samples=5)

    w1, w2, w3 = weights
    combined_confidence = (
        w1 * retrieval_result["retrieval_confidence"]
        + w2 * rerank_result["rerank_confidence"]
        + w3 * gen_result["consistency_confidence"]
    )

    is_correct = check_correctness(
        gen_result['majority_answer'],
        example['is_answerable'],
        example['gold_answers']   # FIXED: was 'gold_answer'
    )

    return {
        "question": question,
        "is_answerable": example["is_answerable"],
        "gold_answers": example["gold_answers"],
        "system_answer": gen_result["majority_answer"],
        "retrieval_confidence": retrieval_result["retrieval_confidence"],
        "rerank_confidence": rerank_result["rerank_confidence"],
        "consistency_confidence": gen_result["consistency_confidence"],
        "combined_confidence": combined_confidence,
        "is_correct": is_correct,
    }


def run_eval(n_questions=100, eval_set_path="eval_set.json", output_path="eval_results.json"):
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    subset = eval_set[:n_questions]
    results = []

    for example in tqdm(subset, desc="Running eval pipeline"):
        try:
            result = run_single_example(example)
            results.append(result)
        except Exception as e:
            print(f"\nError on question '{example['question'][:50]}...': {e}")
            continue

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    correct_count = sum(1 for r in results if r["is_correct"])
    print(f"\nCompleted {len(results)}/{len(subset)} questions")
    if results:
        print(f"Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}")
    else:
        print("No results — every example failed. Check the error messages above.")

    return results


if __name__ == "__main__":
    run_eval(n_questions=100, eval_set_path="eval_set.json", output_path="eval_results.json")
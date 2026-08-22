## Step 6: Wiring the Full Pipeline + Running It Over the Eval Set

### What this step actually builds
Up to now, each signal has lived in its own file, tested in isolation with 4 hand-picked queries. Step 6 does three new things:

1. **Combines all three signals into one function** — for a given question, run retrieval → rerank → generate (5x), and produce one dictionary with everything: the three individual confidence scores, a combined score, the system's final answer, and whether it *should have* abstained.
2. **Checks correctness against ground truth** — compares the system's generated answer to the question's `gold_answers` (if answerable) or checks whether the system correctly recognized an unanswerable question. This is the step that turns "the system said something" into "the system was right or wrong," which is what calibration validation actually needs.
3. **Runs this over the eval set and saves results** — so Step 7 can load a stable results file and compute the reliability diagram / ECE without re-running the expensive pipeline again.

### Design decision #1: how do we combine the three signals into one score?
As discussed back in Stage 0, we're deliberately starting with the simple, interpretable option: a **weighted average** of the three individual confidence scores (bi-encoder, cross-encoder, self-consistency), each already in a roughly [0,1] range from earlier steps.

```
combined_confidence = w1 * retrieval_confidence + w2 * rerank_confidence + w3 * consistency_confidence
```

We'll start with equal weights (⅓ each) — this is a deliberately naive baseline. **The honest, interview-ready framing:** *"I started with equal-weighted averaging as a baseline, specifically so I'd have something to compare a learned calibrator against later — if equal weighting already calibrates well, that's a finding; if it doesn't, that motivates the learned version."* Don't tune the weights by hand based on vibes — that would be overfitting to our own eyeballing, which defeats the purpose of doing this rigorously.

### Design decision #2: what counts as "correct"?
This needs to handle two different cases cleanly:

- **If the question is answerable:** compare our generated (normalized) answer against each of the normalized `gold_answers` — if it matches *any* of them, or contains one as a substring (partial credit for slightly verbose answers), call it correct.
- **If the question is unanswerable:** the system is "correct" if it said something equivalent to "I cannot answer this" — i.e., it correctly recognized the question was unanswerable, rather than hallucinating a confident-sounding wrong answer.

This second case is the one naive RAG systems get catastrophically wrong, and it's exactly the case your whole project is built to handle well — so getting this correctness check right matters as much as the generation itself.

### Design decision #3: trial run first
As discussed, we'll parameterize the eval run with `n_questions`, defaulting to a small number (20) for a cheap smoke test before committing to the full 200-question run — which, at ~6 calls per question, is roughly 1,200 API calls end-to-end.

### Technologies for this step
| Tool | Purpose |
|---|---|
| `json` | Load `eval_set.json`, save results |
| `tqdm` | Progress bar — with ~1,200 calls at stake, you want visibility into progress and rough time-remaining, not a silent terminal for 10+ minutes |
| Everything from Steps 2-4 (`retrieve`, `rerank`, `self_consistency_confidence`) | Reused directly — this step's whole point is composition, not new logic |

Add to `requirements.txt`:
```
tqdm==4.66.4
```

### Code for this step

```python
# step6_run_eval.py

import json
import re
from tqdm import tqdm
from retrieval import retrieve
from reranker import rerank
from generation import self_consistency_confidence, normalize_answer

ABSTAIN_PHRASES = ["cannot answer", "i cannot", "unable to answer", "no answer"]


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
        # Correct behavior for an unanswerable question IS abstaining.
        return system_abstained

    if system_abstained:
        # Answerable question, but system refused to answer — wrong.
        return False

    normalized_system_answer = normalize_answer(majority_answer_raw)
    normalized_gold = [normalize_answer(g) for g in gold_answers]

    # Match if the system's answer equals a gold answer, OR contains one
    # as a substring (handles slightly verbose but correct answers).
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
    question = example["question"]

    retrieval_result = retrieve(question, k=5)
    rerank_result = rerank(question, retrieval_result)
    top_passage = rerank_result["reranked_passages"][0]  # use RERANKED top passage

    gen_result = self_consistency_confidence(question, top_passage, n_samples=5)

    w1, w2, w3 = weights
    combined_confidence = (
        w1 * retrieval_result["retrieval_confidence"]
        + w2 * rerank_result["rerank_confidence"]
        + w3 * gen_result["consistency_confidence"]
    )

    is_correct = check_correctness(
        gen_result["majority_answer"],  # note: this is normalized text already
        example["is_answerable"],
        example["gold_answers"]
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


def run_eval(n_questions=20, eval_set_path="eval_set.json", output_path="eval_results.json"):
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    subset = eval_set[:n_questions]
    results = []

    for example in tqdm(subset, desc="Running eval pipeline"):
        try:
            result = run_single_example(example)
            results.append(result)
        except Exception as e:
            # Don't let one bad example kill a long run — log it and continue.
            print(f"\nError on question '{example['question'][:50]}...': {e}")
            continue

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Quick summary so we know if something's obviously broken before Step 7
    correct_count = sum(1 for r in results if r["is_correct"])
    print(f"\nCompleted {len(results)}/{len(subset)} questions")
    print(f"Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.2%}")

    return results


if __name__ == "__main__":
    # SMOKE TEST FIRST — small n before committing to the full 200-question run
    run_eval(n_questions=20, output_path="eval_results_trial.json")
```

### What you should be able to explain about each part
- **Why we use the reranked top passage (from `rerank_result`), not the raw retrieval top passage** — this keeps the pipeline internally consistent: whichever passage our system judged best *after* reranking is the one it should actually generate an answer from, matching what a real deployed system would do.
- **Why correctness-checking is a separate function (`check_correctness`) from confidence computation** — same separation-of-concerns principle from Step 4: confidence scoring and correctness evaluation are two independent concerns, and keeping them decoupled means you can change how you check correctness later without touching any confidence logic.
- **Why we wrap each example in try/except inside the loop** — with ~1,200 API calls, transient network errors or an occasional malformed response are statistically likely at some point; you don't want a single failure at question 47 to lose all progress on the first 46. This is a legitimate engineering practice worth mentioning if asked about production robustness.
- **Why the trial run saves to a different file (`eval_results_trial.json`)** — so it never gets confused with or accidentally overwrites your real 200-question results later.

---

Run this as-is (20 questions, ~2-3 minutes given the API calls involved) and paste the output — I want to see the accuracy summary and confirm nothing errors out before we commit to the full 200-question run.
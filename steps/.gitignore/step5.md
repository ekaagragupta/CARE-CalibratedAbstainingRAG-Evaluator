## Step 5: Building a Labeled Evaluation Set

### How do you know your confidence scores are actually meaningful?

For each question in our evaluation set, we need to know, independently of our system:
1. **Is the question actually answerable from the corpus?** (SQuAD 2.0 already tells us this — that's *why* we picked it in Step 1.)
2. **If answerable, what's the correct answer?** (SQuAD 2.0 provides this too — human-annotated gold answers.)

With these two ground-truth facts per question, we can finally ask the real question this whole project is about: **"Does my combined confidence score reliably separate the questions my system gets right from the questions it gets wrong (or should refuse)?"** That's a measurable claim, not a vibe.

### The key design decision: sample size and composition
We won't run this over the full ~12,000 SQuAD 2.0 validation questions — that would mean thousands of expensive LLM calls (remember, self-consistency alone is 5 calls per question) and take far too long for a 4-6 week solo project. We need a **stratified sample**: a smaller set, but deliberately balanced between answerable and unanswerable questions, so both categories are well-represented for evaluation. SQuAD 2.0 conveniently marks unanswerable questions with an empty `answers` list — that's our label source, already built into the dataset, not something we invent.

**Trade-off to be upfront about:** a stratified sample of, say, 150-200 questions (split roughly 50/50 answerable/unanswerable) is enough to compute meaningful statistics (reliability diagrams, ECE) without burning excessive API calls or time. This is a real methodological choice — small enough to be practical solo, large enough to draw a real conclusion — and you should be ready to justify that specific number if asked ("why 150-200 and not 20, or not 5000").

### What we're NOT doing yet in this step (important scoping)
This step only **builds and saves the eval set** — it does not yet run our pipeline over it or compute calibration metrics. That's Step 6 and 7. Keeping this step scoped narrowly (just constructing clean, labeled data) versus running everything at once means if something's wrong with the eval set itself, we catch it in isolation, not tangled up with pipeline bugs.

### Technologies for this step
| Tool | Purpose | Why this one |
|---|---|---|
| `datasets` (already used) | Access to SQuAD 2.0 with its answerable/unanswerable labels | Same dependency as Step 1, no new library needed |
| `random` (standard library) | Stratified sampling with a fixed seed | `random.seed()` ensures your sample is **reproducible** — a real requirement for any evaluation you'll report numbers from; without a fixed seed, rerunning would silently give you a different sample each time, making your results non-reproducible, which is a red flag in any evaluation methodology |
| `json` (standard library) | Save the eval set to disk | So Step 6/7 can load a stable, fixed eval set rather than re-sampling every run |

### Code for this step

```python
# step5_build_eval_set.py

import random
import json
from datasets import load_dataset

random.seed(42)  # fixed seed — makes this sample reproducible, a real requirement for evaluation work

def build_eval_set(n_answerable=100, n_unanswerable=100):
    """
    Builds a stratified evaluation set from SQuAD 2.0's validation split.
    Each example includes: question, context passage, whether it's
    answerable, and the gold answer (if any) — all ground truth from
    SQuAD's own human annotations, not something we're inferring.
    """
    dataset = load_dataset("squad_v2", split="validation")

    answerable = []
    unanswerable = []

    for example in dataset:
        is_answerable = len(example["answers"]["text"]) > 0

        entry = {
            "question": example["question"],
            "context": example["context"],
            "is_answerable": is_answerable,
            # gold answers: a list, since SQuAD sometimes has multiple
            # acceptable phrasings for the same answer, annotated by
            # different human raters
            "gold_answers": example["answers"]["text"] if is_answerable else []
        }

        if is_answerable:
            answerable.append(entry)
        else:
            unanswerable.append(entry)

    print(f"Total answerable questions available: {len(answerable)}")
    print(f"Total unanswerable questions available: {len(unanswerable)}")

    # Random sampling WITHOUT replacement — each question appears at most once
    sampled_answerable = random.sample(answerable, n_answerable)
    sampled_unanswerable = random.sample(unanswerable, n_unanswerable)

    eval_set = sampled_answerable + sampled_unanswerable
    random.shuffle(eval_set)  # mix the two categories so order doesn't leak the label

    return eval_set


if __name__ == "__main__":
    eval_set = build_eval_set(n_answerable=100, n_unanswerable=100)

    print(f"\nFinal eval set size: {len(eval_set)}")
    print(f"Answerable: {sum(1 for e in eval_set if e['is_answerable'])}")
    print(f"Unanswerable: {sum(1 for e in eval_set if not e['is_answerable'])}")

    # Save to disk so later steps use this EXACT fixed set, not a re-sample
    with open("eval_set.json", "w") as f:
        json.dump(eval_set, f, indent=2)

    print("\nSaved to eval_set.json")

    # Sanity check: print a couple of examples from each category
    print("\n--- Sample answerable example ---")
    example = next(e for e in eval_set if e["is_answerable"])
    print(f"Q: {example['question']}")
    print(f"Gold answers: {example['gold_answers']}")

    print("\n--- Sample unanswerable example ---")
    example = next(e for e in eval_set if not e["is_answerable"])
    print(f"Q: {example['question']}")
    print(f"Gold answers: {example['gold_answers']}")
```

### What you should be able to explain about each part
- **Why we shuffle after combining** — if we didn't, the eval set would have all 100 answerable questions first, then all 100 unanswerable ones; any later code that processes this list sequentially (e.g. showing progress, or accidentally slicing it) could unintentionally introduce bias. Shuffling removes that risk cheaply.
- **Why `random.sample` and not just slicing the first N** — the dataset is ordered by article/topic as it comes from SQuAD; taking the first 100 would bias your sample toward whichever topics happen to appear first, rather than giving you a representative cross-section.
- **Why we save gold answers as a list, not a single string** — SQuAD deliberately allows multiple valid phrasings of a correct answer (annotated by different crowdworkers), and our later correctness-checking (Step 6/7) needs to check "does our system's answer match *any* of these," not just one fixed string.
- **Why this is saved to a file rather than regenerated each run** — reproducibility again: Step 6 and 7 need to evaluate against the *same* 200 questions every time, not a fresh random sample each run, or comparing results across steps would be meaningless.

---

Run this and paste the output — you should see the total available counts, the final 200-question breakdown, and the two sample examples. Once confirmed, Step 6 is where we actually run our full pipeline (retrieval + rerank + generation, all three confidence signals) over these 200 questions and start collecting the data we need for real calibration.
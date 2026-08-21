import random
import json
from datasets import load_dataset

random.seed(42)

def build_eval_set(n_answerable=100,n_unanswerable=100):
    """
    Builds a stratified evaluation set from SQuAD 2.0's validation split.
    Each example includes: question, context passage, whether it's
    answerable, and the gold answer (if any) — all ground truth from
    SQuAD's own human annotations, not something we're inferring.
    """

    dataset=load_dataset("squad_v2",split="validation")

    answerable=[]
    unanswerable=[]

    for example in dataset:
        is_answerable=len(example['answers']['text'])>0
        entry={
            "question":example['question'],
             "context": example["context"],
            "is_answerable": is_answerable,
            # diff human raters
            "gold_answers": example["answers"]["text"] if is_answerable else []
        }
        if is_answerable:
            answerable.append(entry)
        else:
            unanswerable.append(entry)

    print(f"total answerable questions avaiable:{len(answerable)}")
    print(f"total unawerable questions avaiable:{len(unanswerable)}")

    #random sampling without replacment - each question appears at most once 
    sampled_answerable = random.sample(answerable, n_answerable)
    sampled_unanswerable = random.sample(unanswerable, n_unanswerable)

    eval_set = sampled_answerable + sampled_unanswerable
    random.shuffle(eval_set) 
    return eval_set

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
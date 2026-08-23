## Step 8: Abstention Logic

### What this step builds
A single end-to-end function: given a question, run the full pipeline (retrieve → rerank → generate), compute `combined_confidence` (the raw, non-recalibrated version, per our Step 7.5 conclusion), and make one of three decisions: **answer**, **ask a clarifying question**, or **abstain**. This is the first point where all the machinery from Phases 1-2 actually becomes a usable system rather than a measurement exercise.

### Design decision #1: two thresholds, not one
A single cutoff ("above X = answer, below X = abstain") is the simplest option, but it wastes information you already have. Recall your Step 7 reliability diagram had 5 bins — the low-confidence bins (0.4-0.6) still had meaningfully lower accuracy (63%) than the high bins (0.6-0.8 at 87%). Rather than a hard binary cutoff, we'll use **two thresholds, creating three zones**:

- **High confidence** (above threshold_high): answer directly
- **Medium confidence** (between the two thresholds): this is the ambiguous zone — rather than guessing, ask a clarifying question or flag lower certainty explicitly
- **Low confidence** (below threshold_low): abstain outright

This maps naturally onto a real product decision: you don't want a system that either fully commits or fully refuses with nothing in between — a "medium confidence" tier that hedges is closer to how a careful human expert would behave, and it's a more interesting design to defend in an interview than a single if-statement.

### Design decision #2: how do we actually pick the threshold values?
This is the part that must be **data-driven, not guessed** — otherwise the whole calibration effort from Phase 2 was pointless. We'll use your 76 labeled results to find thresholds that optimize a real trade-off: **precision on answered questions vs. how often you abstain**.

The relevant concept: **selective prediction** — for each candidate threshold, compute:
- **Coverage**: what fraction of questions the system chooses to answer (not abstain)
- **Selective accuracy**: among the questions it chooses to answer, what fraction are correct

As you raise the threshold, coverage drops (you answer fewer questions) but selective accuracy should rise (you're being pickier, so what you do answer is more likely right). This coverage-vs-accuracy trade-off is the standard way selective prediction systems are evaluated — plotting it is itself a legitimate thing to show in your project write-up.

**We won't hand-pick a "nice-looking" threshold** — we'll sweep a range of thresholds and show the trade-off curve, then pick one with an explicit, stated rationale (e.g. "the highest threshold that keeps coverage above 50%, since abstaining on more than half of questions makes the system impractical to use").

### Design decision #3: what does "ask a clarifying question" actually mean here?
Given our SQuAD-based setup, we don't have a live back-and-forth conversation mechanism — so for the medium-confidence tier, the honest, implementable behavior is: **give the answer, but explicitly flag it as uncertain** (e.g., "I found a possible answer, but I'm not fully confident: ..."), rather than fabricating an interactive clarification dialogue we can't actually evaluate. This is a scoping decision worth being upfront about: *"For this project's scope, the medium-confidence tier surfaces a hedged answer with an explicit uncertainty flag, rather than an interactive clarification loop — extending to true interactive clarification would be a natural next step, but requires a different evaluation setup than a static benchmark like SQuAD allows."*

### Technologies for this step
Nothing new — this step is pure composition and analysis logic using `numpy` (already installed) over your existing `eval_results.json`.

### Code for this step

```python
# step8_abstention.py

import json
import numpy as np


def sweep_thresholds(results, thresholds=None):
    """
    For each candidate threshold, computes coverage (fraction of questions
    answered) and selective accuracy (accuracy among only the answered
    questions) — the standard trade-off curve for selective prediction.
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.05)

    confidences = np.array([r["combined_confidence"] for r in results])
    correctness = np.array([1 if r["is_correct"] else 0 for r in results])
    n_total = len(results)

    sweep_results = []
    for t in thresholds:
        answered_mask = confidences >= t
        n_answered = answered_mask.sum()

        coverage = n_answered / n_total
        if n_answered > 0:
            selective_accuracy = correctness[answered_mask].mean()
        else:
            selective_accuracy = None  # no questions answered at this threshold

        sweep_results.append({
            "threshold": float(t),
            "coverage": float(coverage),
            "n_answered": int(n_answered),
            "selective_accuracy": None if selective_accuracy is None else float(selective_accuracy)
        })

    return sweep_results


def print_sweep(sweep_results):
    print(f"{'Threshold':>10} {'Coverage':>10} {'N answered':>12} {'Sel. Accuracy':>15}")
    for row in sweep_results:
        acc_str = f"{row['selective_accuracy']:.3f}" if row['selective_accuracy'] is not None else "N/A"
        print(f"{row['threshold']:>10.2f} {row['coverage']:>10.2%} {row['n_answered']:>12} {acc_str:>15}")


def choose_thresholds(sweep_results, min_coverage=0.5):
    """
    Picks threshold_high and threshold_low with an explicit, stated rule:
    threshold_high = the highest threshold that still keeps coverage
    at or above min_coverage (answering fewer than half of questions
    makes the system impractical, per our stated design rationale).
    threshold_low is set lower, marking the boundary below which we
    abstain outright rather than hedge.
    """
    # Filter to thresholds that meet the minimum coverage requirement
    valid = [r for r in sweep_results if r["coverage"] >= min_coverage]
    if not valid:
        raise ValueError(f"No threshold keeps coverage >= {min_coverage:.0%} — check your data.")

    # Highest such threshold = most selective option that still meets coverage floor
    threshold_high = max(valid, key=lambda r: r["threshold"])["threshold"]

    # threshold_low: a fixed step below threshold_high, marking the abstain boundary
    threshold_low = max(0.0, threshold_high - 0.2)

    return threshold_high, threshold_low


def make_decision(combined_confidence, threshold_high, threshold_low):
    """
    The actual three-way decision function this whole project has been
    building toward: answer confidently, hedge, or abstain.
    """
    if combined_confidence >= threshold_high:
        return "ANSWER"
    elif combined_confidence >= threshold_low:
        return "HEDGE"  # answer, but flagged as uncertain
    else:
        return "ABSTAIN"


if __name__ == "__main__":
    with open("eval_results.json") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results\n")

    sweep_results = sweep_thresholds(results)
    print_sweep(sweep_results)

    threshold_high, threshold_low = choose_thresholds(sweep_results, min_coverage=0.5)
    print(f"\nChosen thresholds: HIGH={threshold_high:.2f}, LOW={threshold_low:.2f}")

    # Apply the decision function to every result and show the breakdown
    for r in results:
        r["decision"] = make_decision(r["combined_confidence"], threshold_high, threshold_low)

    decision_counts = {}
    for r in results:
        decision_counts[r["decision"]] = decision_counts.get(r["decision"], 0) + 1

    print(f"\nDecision breakdown across {len(results)} questions:")
    for decision, count in decision_counts.items():
        print(f"  {decision}: {count} ({count/len(results):.1%})")

    # Accuracy WITHIN each decision category — the real test of whether
    # the decision boundaries are actually doing useful work
    print(f"\nAccuracy by decision category:")
    for decision in decision_counts:
        subset = [r for r in results if r["decision"] == decision]
        acc = sum(1 for r in subset if r["is_correct"]) / len(subset)
        print(f"  {decision}: {acc:.1%} accuracy (n={len(subset)})")

    with open("eval_results_with_decisions.json", "w") as f:
        json.dump(results, f, indent=2)
```

### What you should be able to explain about each part
- **Why we sweep thresholds instead of picking one directly** — this produces the actual evidence (the coverage/accuracy trade-off table) that justifies whatever threshold you land on, rather than an arbitrary guess dressed up as a decision.
- **Why `min_coverage=0.5` specifically** — this is a real, statable design choice: *"I set a floor of answering at least half of all questions, since a system that abstains on the majority of queries isn't useful in practice, regardless of how accurate it is on the remainder it does answer."* You can adjust this number, but you need a reason for whatever you pick.
- **Why we check accuracy *within* each decision category at the end** — this is the actual validation that the abstention logic works: if `ANSWER`-category accuracy is meaningfully higher than `ABSTAIN`-category accuracy (which is what we should expect if abstention would have been "correct" for those), that's concrete proof the confidence-based abstention decision is doing real, useful work — not just splitting data arbitrarily.

---

Run this and paste the full output — the threshold sweep table, the chosen thresholds, and critically the accuracy-by-decision-category breakdown. That last part is the number that tells us whether this whole project's core premise ("the system knows when it doesn't know") actually holds up on your real data.
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


        """
        On my 76-question evaluation set, questions my system answered with high confidence were correct 88.4% of 
        the time, versus 63.3% for hedged answers and 33.3% for the tier it would abstain on. That monotonic relationship 
        between confidence tier and actual accuracy is the empirical validation that combining retrieval, reranking, and generation-consistency 
        signals produces a confidence score that meaningfully predicts correctness — which was the entire premise of the project.
        """
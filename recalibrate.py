"""
fixing the miscalibration here lol
recalibration method : Platt Scaling
fit a simple logistic regression that maps my 
existing combined_confidence (one number) to a corrected probability: 
corrected = sigmoid(a * combined_confidence + b). 
This learns just 2 parameters (a and b). With only 76 examples, 2 parameters is
 very safe to fit — low risk of overfitting.

 It directly targets the exact problem you found — the combined score is systematically too low relative to actual accuracy — with the minimum complexity needed to fix it. It's also easier to explain and defend: "I used Platt scaling — a standard, minimal-parameter recalibration technique — specifically because with only 76 labeled examples, a more complex multi-feature calibrator risked overfitting without a held-out validation set to check it.

 Why this specific fix (Platt scaling) works conceptually
It's fitting a monotonic S-curve correction on top of your existing score — it can stretch, compress, or shift your confidence values, but it can't reorder them (a question your system was more confident about before recalibration stays more confident after). That's important: recalibration should correct the scale, not change which questions the system trusts more relative to each other — that ordering came from real signal (retrieval quality, agreement, etc.) and shouldn't be disturbed.
"""
# step7b_recalibrate.py

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from calibration_analysis import compute_reliability_diagram, compute_ece, plot_reliability_diagram


def fit_platt_scaling(results):
    """
    Fits a 1D logistic regression mapping raw combined_confidence to a
    corrected probability of correctness. This is Platt scaling — a
    standard, minimal-parameter recalibration technique, well-suited to
    a small labeled set since it only learns 2 parameters (a, b) in:
    corrected_confidence = sigmoid(a * raw_confidence + b)
    """
    X = np.array([[r["combined_confidence"]] for r in results])
    y = np.array([1 if r["is_correct"] else 0 for r in results])

    calibrator = LogisticRegression()
    calibrator.fit(X, y)

    return calibrator


def apply_recalibration(results, calibrator):
    """
    Applies the fitted Platt scaling model to every result, adding a
    new 'recalibrated_confidence' field alongside the original raw score
    (we keep both — never overwrite raw data with a derived correction).
    """
    for r in results:
        X = np.array([[r["combined_confidence"]]])
        # predict_proba returns [P(class=0), P(class=1)] — we want P(correct)=P(class=1)
        corrected = calibrator.predict_proba(X)[0][1]
        r["recalibrated_confidence"] = float(corrected)
    return results


if __name__ == "__main__":
    with open("eval_results.json") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results")

    calibrator = fit_platt_scaling(results)
    print(f"\nFitted Platt scaling: a={calibrator.coef_[0][0]:.3f}, b={calibrator.intercept_[0]:.3f}")

    results = apply_recalibration(results, calibrator)

    # Save the recalibrated results — new file, keeping the original untouched
    with open("eval_results_recalibrated.json", "w") as f:
        json.dump(results, f, indent=2)

    # Re-run the SAME reliability diagram analysis, but on recalibrated scores
    # We temporarily rename the field so we can reuse compute_reliability_diagram as-is
    results_for_diagram = [
        {**r, "combined_confidence": r["recalibrated_confidence"]}
        for r in results
    ]

    bin_edges, bin_confidences, bin_accuracies, bin_counts = compute_reliability_diagram(
        results_for_diagram, n_bins=5
    )

    print("\nBin-by-bin breakdown AFTER recalibration:")
    for i in range(len(bin_confidences)):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        conf = bin_confidences[i]
        acc = bin_accuracies[i]
        count = bin_counts[i]
        if conf is not None:
            print(f"  [{lower:.1f}-{upper:.1f}]  n={count:3d}  avg_confidence={conf:.3f}  avg_accuracy={acc:.3f}")
        else:
            print(f"  [{lower:.1f}-{upper:.1f}]  n=0 (empty bin)")

    ece = compute_ece(bin_confidences, bin_accuracies, bin_counts, total_n=len(results))
    print(f"\nExpected Calibration Error AFTER recalibration: {ece:.4f}")

    plot_reliability_diagram(
        bin_edges, bin_confidences, bin_accuracies, bin_counts, ece,
        save_path="reliability_diagram_recalibrated.png"
    )
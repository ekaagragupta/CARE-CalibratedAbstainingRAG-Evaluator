"""
A reliability diagram and ECE are the standard, established 
way researchers answer exactly that question 
for any model that outputs a confidence/probability not just a RAG specific topic 

aab what is well-calibrated actaully ?
among all the times your system said "I'm 80% confident," 
it was actually correct about 80% of the time. Not 95%, not 50% 
stated confidence ==  empirical accuracy at that confidence level.

mechanisms :

    bin predictions by confidence score 
               |   taking all 76 predicts and sorting by combined_confidence and split into buckets 
               |
    For each bin, compute two numbers:    
                |   Average confidence and Average accuracy
                |     
    Plot confidence (x-axis) vs. accuracy (y-axis)
               |
               |
     Deviation from the diagonal is the miscalibration  

     mtlb if    ECE = Σ (n_bin / N) * |accuracy_bin - confidence_bin|
     case 1 . points above the diagonal mean the system is underconfident (it's actually more accurate than it claims)
     case 2 . points below the diagonal mean it's overconfident (claims more certainty than its actual accuracy supports) 
"""

# step7_calibration_analysis.py

import json
import numpy as np
import matplotlib.pyplot as plt

def load_results(path="eval_results.json"):
    with open(path) as f:
        return json.load(f)


def compute_reliability_diagram(results, n_bins=5):
    """
    Bins predictions by confidence score and computes average confidence
    vs. average accuracy per bin — the core data behind a reliability
    diagram. Also returns per-bin counts, needed for ECE weighting.
    """
    confidences = np.array([r["combined_confidence"] for r in results])
    correctness = np.array([1 if r["is_correct"] else 0 for r in results])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        # Include the right edge only on the last bin, so 1.0 isn't dropped
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = mask.sum()
        if count > 0:
            bin_confidences.append(confidences[mask].mean())
            bin_accuracies.append(correctness[mask].mean())
        else:
            bin_confidences.append(None)  # empty bin — nothing to plot
            bin_accuracies.append(None)
        bin_counts.append(count)

    return bin_edges, bin_confidences, bin_accuracies, bin_counts


def compute_ece(bin_confidences, bin_accuracies, bin_counts, total_n):
    """
    Expected Calibration Error: weighted average of |accuracy - confidence|
    across bins, weighted by how many examples fall in each bin.
    """
    ece = 0.0
    for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
        if conf is not None and count > 0:
            ece += (count / total_n) * abs(acc - conf)
    return ece


def plot_reliability_diagram(bin_edges, bin_confidences, bin_accuracies, bin_counts, ece, save_path="reliability_diagram.png"):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    # Our actual bins — only plot non-empty ones
    valid_conf = [c for c in bin_confidences if c is not None]
    valid_acc = [a for a in bin_accuracies if a is not None]
    ax.plot(valid_conf, valid_acc, marker="o", color="steelblue", label="Observed calibration")

    ax.set_xlabel("Confidence (predicted)")
    ax.set_ylabel("Accuracy (observed)")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.3f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved reliability diagram to {save_path}")


if __name__ == "__main__":
    results = load_results("eval_results.json")
    print(f"Loaded {len(results)} results")

    bin_edges, bin_confidences, bin_accuracies, bin_counts = compute_reliability_diagram(results, n_bins=5)

    print("\nBin-by-bin breakdown:")
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
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

    plot_reliability_diagram(bin_edges, bin_confidences, bin_accuracies, bin_counts, ece)


    """
    Underconfidence is actually the safer direction to err in for an abstaining system — 
    it means you'd occasionally abstain on questions you actually would have gotten right (a missed opportunity, but not
      a dangerous failure), rather than confidently answering wrong. Worth stating plainly if asked: "Given the choice, I'd
        rather my system be overly cautious than overly confident — underconfidence costs you some correct answers you decline to
          give; overconfidence risks confidently wrong answers, which is the worse failure mode 
    for a system whose whole premise is trustworthy abstention
    """
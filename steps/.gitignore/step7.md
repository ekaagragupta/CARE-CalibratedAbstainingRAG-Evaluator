## Step 7: Reliability Diagram + Expected Calibration Error (ECE)

### Why this is the step that actually validates your whole project
Everything so far has produced confidence *scores*. This step answers the real question a hiring panel would ask: **"How do you know those scores mean anything?"** A reliability diagram and ECE are the standard, established way researchers answer exactly that question for any model that outputs a confidence/probability — not something specific to RAG, which is worth knowing (it comes from the broader ML calibration literature, e.g. Guo et al.'s "On Calibration of Modern Neural Networks").

### The core idea: what does "well-calibrated" actually mean?
A confidence score is **well-calibrated** if, among all the times your system said "I'm 80% confident," it was actually correct about 80% of the time. Not 95%, not 50% — the *stated* confidence should match the *empirical* accuracy at that confidence level. This is a precise, checkable claim, not a vague notion of "the score seems reasonable."

### How a reliability diagram works, step by step
1. **Bin your predictions by confidence score.** Take all 76 results, sort by `combined_confidence`, and split into buckets (commonly 5 or 10 bins) — e.g., bin 1 = confidence 0.0–0.2, bin 2 = 0.2–0.4, etc.
2. **For each bin, compute two numbers:**
   - **Average confidence** in that bin (the mean of `combined_confidence` for everything that fell in it)
   - **Average accuracy** in that bin (what fraction of `is_correct` was `True` for everything in that bin)
3. **Plot confidence (x-axis) vs. accuracy (y-axis)** for each bin. A perfectly calibrated system produces points that fall exactly on the diagonal line y=x — confidence 0.7 bin has ~70% actual accuracy, confidence 0.3 bin has ~30% actual accuracy, and so on.
4. **Deviation from the diagonal is the miscalibration** — points *above* the diagonal mean the system is **underconfident** (it's actually more accurate than it claims); points *below* the diagonal mean it's **overconfident** (claims more certainty than its actual accuracy supports) — the more dangerous direction for a system meant to know when to abstain.

### The single-number summary: Expected Calibration Error (ECE)
A diagram is great for a picture, but interviewers and reports often want one number. **ECE** is the weighted average of the gap between confidence and accuracy across all bins, weighted by how many examples fall in each bin:

```
ECE = Σ (n_bin / N) * |accuracy_bin - confidence_bin|
```

Lower ECE = better calibrated. A value near 0 means confidence scores are trustworthy; a large ECE (say, > 0.15–0.2) means the scores are meaningfully misleading.

### Design decision: how many bins, given only 76 examples?
This is worth being deliberate about, since it's a real methodological choice you should defend if asked. With standard 10 bins and only 76 examples, some bins would have very few (or zero) points — making their average accuracy noisy or undefined. **We'll use 5 bins instead of the usual 10**, explicitly because of our sample size. This is a legitimate, honest trade-off to state directly: *"With a 76-example eval set, I used 5 bins rather than the standard 10 to keep each bin's accuracy estimate statistically meaningful — 10 bins would leave some bins with only 2-3 examples, making their accuracy estimate unreliable."*

### Technologies for this step
| Tool | Purpose |
|---|---|
| `numpy` | Binning and averaging computations |
| `matplotlib` | Plotting the reliability diagram |

Add to `requirements.txt` if not already present:
```
matplotlib==3.9.0
```

### Code for this step

```python
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
```

### What you should be able to explain about each part
- **Why we handle the last bin's edge specially (`<=` instead of `<`)** — without this, a confidence score of exactly `1.0` would fall outside every bin's range (since all other bins use `< upper`), silently excluding perfect-confidence examples from the whole analysis — a subtle off-by-one that would quietly corrupt the result.
- **Why empty bins are handled explicitly (`None`), not silently skipped** — if a bin has zero examples, its "average accuracy" is mathematically undefined (0/0); plotting it as 0 would misleadingly suggest "0% accuracy" for that confidence range, when really we just have no data there. Being explicit about missing data versus zero data is a real statistical hygiene point.
- **Why ECE weights by bin count** — a bin with 40 examples and a bin with 2 examples shouldn't contribute equally to one summary number; weighting by count means ECE reflects the calibration quality across your *actual* data distribution, not an average-of-averages that treats a 2-example bin as equally important as a 40-example one.

---

Run this and paste the full output (the bin-by-bin printout and the ECE value) — I want to see the actual shape of your calibration before we interpret it together. Given you have real, hard-won data with a documented self-consistency-vs-abstention nuance already in hand, this result should tell a genuinely interesting story either way.




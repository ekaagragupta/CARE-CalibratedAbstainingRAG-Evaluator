## Step 2: The Retrieval Function + Confidence Signal #1 (Score Gap)

### What this step accomplishes
Step 1 built a static index. Step 2 turns that into something that answers: given a new query, get the top-k passages, *and* compute a number that tells us how confident the retrieval was. This is the first of the three signals from Stage 0.

### The core idea: why "top score alone" is a bad confidence signal
Your first instinct might be: "if the top passage's similarity score is high, we're confident." That's wrong, and knowing *why* it's wrong is an interview-worthy insight.

The problem: cosine similarity scores aren't well-calibrated in an absolute sense — a score of 0.75 doesn't mean "75% likely relevant" the same way across every query. Different queries produce different overall score ranges depending on how well-represented their topic is in the corpus, how the query is phrased, etc. So a raw top-1 score is noisy on its own.

**What's more informative is the *relative* structure of the score distribution:**
- If the top result scores much higher than the 2nd, 3rd, 4th results → there's a clear "winner," meaning the corpus likely contains a passage that specifically addresses this query. High confidence.
- If the top few results all score roughly the same → the retriever can't clearly distinguish what's relevant, either because the corpus lacks a good match, or the query is ambiguous, or several passages are plausibly relevant. Low confidence.

This is directly analogous to a concept you may have seen in classification: the "margin" between the top prediction and runner-up. Same intuition, applied to retrieval.

### The specific metric we'll use: normalized score gap
We'll compute:

```
gap = (score_1 - score_2) / score_1
```

where `score_1` is the top passage's similarity and `score_2` is the second-best. Normalizing by `score_1` makes this comparable across queries with different absolute score ranges — a design choice you should be ready to defend: "I normalized because absolute score gaps are misleading when the overall score scale shifts between queries; dividing by score_1 gives me a scale-invariant measure of how dominant the top result is."

**Trade-off to be honest about:** this only looks at the top 2 results. A more sophisticated version could use the full score distribution (e.g., entropy over the top-k, or standard deviation of top-k scores). We're starting with the simplest defensible metric — bringing in entropy later is a legitimate "future work" talking point, not a gap in your understanding.

### Technologies used here
| Tool | Purpose |
|---|---|
| `faiss.Index.search()` | Returns top-k indices + their similarity scores in one call — this is the actual retrieval operation |
| `sentence_transformers.encode()` | Same embedding model from Step 1, now applied to the *query* instead of passages — critical that it's the same model, since query and passage embeddings must live in the same vector space to be comparable |

### Code for this step

```python
# step2_retrieval.py

# step2_retrieval.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from datasets import load_dataset

# Reload the dataset in THIS script's namespace — build_index.py's variables
# don't persist across separate script runs, only what we explicitly saved to disk does.
dataset = load_dataset("squad_v2", split="validation")

# --- Load what Step 1 built ---
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
index = faiss.read_index("corpus_index.faiss")
with open("contexts.pkl", "rb") as f:
    contexts = pickle.load(f)


def compute_retrieval_confidence(scores, gap_weight=0.5, min_score_weight=0.5):
    """
    Combines normalized score gap with absolute top-1 score.
    Both must be reasonably high for genuine confidence — a large gap
    built on a low top score (as with noisy/off-topic queries) should
    NOT count as confident.
    """
    if len(scores) < 2 or scores[0] <= 0:
        return 0.0

    score_gap = (scores[0] - scores[1]) / scores[0]
    top_score = scores[0]  # already in [0,1] range since embeddings are normalized

    # Weighted combination — both terms must contribute; a high gap with a
    # low top_score is dragged down, and vice versa.
    combined_confidence = (gap_weight * score_gap) + (min_score_weight * top_score)
    return float(combined_confidence)


def retrieve(query, k=5):
    """
    Embeds a query, searches the FAISS index, and returns:
    - the top-k passages
    - their similarity scores
    - a combined confidence signal (gap + absolute top score)
    """
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    scores, indices = index.search(query_vec, k)
    scores = scores[0]
    indices = indices[0]

    retrieved_passages = [contexts[i] for i in indices]

    confidence = compute_retrieval_confidence(scores)

    return {
        "query": query,
        "passages": retrieved_passages,
        "scores": scores.tolist(),
        "retrieval_confidence": confidence
    }


if __name__ == "__main__":
    test_queries = [
        "In what country is Normandy located?",   # (A) VERIFIED answerable
        "What river runs through Paris?",         # (B) plausible, coverage unconfirmed
        "What is the recommended dosage of ibuprofen for a dog?",  # (C) off-topic
        "purple elephant quantum sandwich Tuesday",  # (D) gibberish
    ]

    for q in test_queries:
        result = retrieve(q, k=5)   # <-- call retrieve(), not compute_retrieval_confidence() directly
        print(f"\nQuery: {q}")
        print(f"Top-5 scores: {[round(s, 3) for s in result['scores']]}")
        print(f"Combined confidence: {result['retrieval_confidence']:.3f}")
```

**What you should be able to say about each part:**
- Why we re-embed the *query* with the same model/settings as the passages (shared vector space — this trips people up if they use different models for query vs. passage embedding).
- Why we return `scores` alongside `passages` — you need raw scores downstream for the calibration step, not just the text.
- Why the score gap is guarded (`len(scores) >= 2 and scores[0] > 0`) — basic defensive coding against edge cases, worth mentioning briefly if asked about code quality.
- That this is **one of three signals**, not the final confidence score — the reranker signal (Step 3) and generation-agreement signal (Step 4) still need to be added before we get to calibration.

---

Try running this against a few test queries and look at how `retrieval_confidence` behaves — try a query you know is well-covered by the corpus vs. a nonsense/off-topic query, and see if the gap behaves the way we predicted. Does the score-gap reasoning make sense, and are you clear on why we normalized it? Once confirmed, Step 3 is the cross-encoder reranker — our second, independent confidence signal.
"""
given a new query and get the top-k-passages , computing a number which 
will tell me how confident the retrieval was 

core idea of project is 'top score alone " is bad confidence signal if the top passage's similarity score is high, we're confident." That's wrong, and knowing why it's wrong is an interview-worthy insight.

What's more informative is the relative structure of the score distribution:
If the top result scores much higher than the 2nd, 3rd, 4th results → there's a clear "winner," meaning the corpus likely contains a passage that specifically addresses this query. High confidence.
If the top few results all score roughly the same → the retriever can't clearly distinguish what's relevant, either because the corpus lacks a good match, or the query is ambiguous, or several passages are plausibly relevant. Low confidence.
"""

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


        """
        conclusion 
        I started with score gap alone, found empirically that it conflated 'no relevant results' with 
        'many plausibly-relevant results,' so I combined it with the absolute top-1 score. On a small
          manual probe with a verified ground-truth query, this correctly separated answerable from 
          unanswerable queries — but I treated that as a sanity check, not proof, since real calibration 
          validation needs a much larger labeled set, which is a later step in the pipeline.
        """
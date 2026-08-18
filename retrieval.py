"""
given a new query and get the top-k-passages , computing a number which 
will tell me how confident the retrieval was 

core idea of project is 'top score alone " is bad confidence signal if the top passage's similarity score is high, we're confident." That's wrong, and knowing why it's wrong is an interview-worthy insight.

What's more informative is the relative structure of the score distribution:
If the top result scores much higher than the 2nd, 3rd, 4th results → there's a clear "winner," meaning the corpus likely contains a passage that specifically addresses this query. High confidence.
If the top few results all score roughly the same → the retriever can't clearly distinguish what's relevant, either because the corpus lacks a good match, or the query is ambiguous, or several passages are plausibly relevant. Low confidence.
"""
# step2_retrieval.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# --- Load what Step 1 built ---
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
index = faiss.read_index("corpus_index.faiss")
with open("contexts.pkl", "rb") as f:
    contexts = pickle.load(f)


def retrieve(query, k=5):
    """
    Embeds a query, searches the FAISS index, and returns:
    - the top-k passages
    - their similarity scores
    - a normalized 'score gap' confidence signal
    """
    # Same normalize_embeddings=True as Step 1 — MUST match, or cosine similarity breaks.
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    # FAISS search returns (scores, indices) — both shape (1, k) since we passed 1 query.
    scores, indices = index.search(query_vec, k)
    scores = scores[0]      # unwrap to shape (k,)
    indices = indices[0]

    retrieved_passages = [contexts[i] for i in indices]

    # --- Confidence Signal #1: normalized score gap between top-1 and top-2 ---
    if len(scores) >= 2 and scores[0] > 0:
        score_gap = (scores[0] - scores[1]) / scores[0]
    else:
        score_gap = 0.0  # can't compute a gap with fewer than 2 results

    return {
        "query": query,
        "passages": retrieved_passages,
        "scores": scores.tolist(),
        "retrieval_confidence": float(score_gap)
    }


# --- Quick sanity check ---
if __name__ == "__main__":
    result = retrieve("What is the capital of France?", k=5)
    print(f"Retrieval confidence (score gap): {result['retrieval_confidence']:.3f}")
    print(f"Top score: {result['scores'][0]:.3f}, 2nd score: {result['scores'][1]:.3f}")
    print(f"Top passage: {result['passages'][0][:200]}...")

# output of the sanity check:
    """
    Retrieval confidence (score gap): 0.184
Top score: 0.455, 2nd score: 0.371
Top passage: Montpellier was among the most important of the 66 "villes de sûreté" that the Edict of 1598 granted to the Huguenots. The city's political institutions and the university were all handed 
"""
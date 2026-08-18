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

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
index = faiss.read_index("corpus_index.faiss")
with open("contexts.pkl", "rb") as f:
    contexts = pickle.load(f)

def retrieve(query,topK=5):
    """
    Embed a query, search the FAISS index and return :
    - the top-k passages
    - their similarity scores
    - a normalised 'score gap'  of confidence signal
    """
    query_vec=model.encode(
        [query],
        normalize_embeddings=True    #cosine similarity breaks
    ).astype("float32")

    # faiss search return scores and their indices in the original corpus
    scores,indices=index.search(query_vec,topK)
    scores=scores[0]
    indices=indices[0]

    # get the top-k passages
    retrieved_passages=[contexts[i] for i in indices]

    # compute a normalised 'score gap' confidence signal
    if len(scores)>=2 and scores[0]>0:
        score_gap=(scores[0]-scores[1])/scores[0]
    else:   
        score_gap=0.0
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
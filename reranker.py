"""
it takes the query and a candidate passage together, as one combined input, 
and directly outputs a relevance score. Because it sees both texts jointly, it can
 catch relevance nuances a bi-encoder misses (e.g. word-order-sensitive meaning, negation, entity matching)

 If the bi-encoder and cross-encoder agree — both think the same passage is clearly the best — that's two independent
   models converging on the same answer, which is stronger evidence than either alone. If they disagree (e.g. bi-encoder 
   confidently ranks passage A first, but the cross-encoder ranks passage B much higher), that disagreement itself is a
     red flag: it suggests the query is ambiguous or borderline, and the system should be less confident even though the first
       signal alone looked fine. Multiple independent signals catching different failure modes is the actual intellectual core 
       of the whole project.
"""

from sentence_transformers import CrossEncoder
import numpy as np
from retrieval import retrieve,compute_retrieval_confidence

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

def rerank(query,retrieval_result):
    """
    Takes the output of retrieve() (query + top-k passages) and reranks them
    with a cross-encoder. Returns reranked passages + a second, independent
    confidence signal based on the cross-encoder's own score distribution.
    """
    passages=retrieval_result['passages']
    pair=[[query,p] for p in passages]
    rerank_scores=reranker.predict(pair)

    ranked=sorted(zip(passages,rerank_scores),key=lambda x:x[1],reverse=True)
    reranked_passages=[p for p,s in ranked]
    sorted_scores=[float(s) for p,s in ranked]

    rerank_confidence = compute_rerank_confidence(sorted_scores)  
    return {
            "query": query,
            "reranked_passages": reranked_passages,
            "rerank_scores": sorted_scores,
            "rerank_confidence": rerank_confidence
        }


def compute_rerank_confidence(scores, gap_weight=0.5, min_score_weight=0.5):
        """
        Same conceptual formula as Step 2's confidence, but adapted for
        cross-encoder scores, which can be negative (unlike our normalized
        cosine similarities). We apply a sigmoid to squash raw logits into
        (0,1) before combining, so the formula's assumptions still hold.
        """
        if len(scores) < 2:
            return 0.0

        import math
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        squashed = [sigmoid(s) for s in scores]
        top_score = squashed[0]
        score_gap = (squashed[0] - squashed[1]) / squashed[0] if squashed[0] > 0 else 0.0

        return float((gap_weight * score_gap) + (min_score_weight * top_score))


if __name__ == "__main__":
    test_queries = [
            "In what country is Normandy located?",
            "What river runs through Paris?",
            "What is the recommended dosage of ibuprofen for a dog?",
            "purple elephant quantum sandwich Tuesday",
    ]

    for q in test_queries:
            retrieval_result = retrieve(q, k=5)
            rerank_result = rerank(q, retrieval_result)

            print(f"\nQuery: {q}")
            print(f"Bi-encoder confidence:  {retrieval_result['retrieval_confidence']:.3f}")
            print(f"Cross-encoder confidence: {rerank_result['rerank_confidence']:.3f}")
            print(f"Rerank scores: {[round(s, 3) for s in rerank_result['rerank_scores']]}")
"""
# conclusion 
Query: In what country is Normandy located?
Bi-encoder confidence:  0.406
Cross-encoder confidence: 0.925
Rerank scores: [3.083, -2.173, -2.732, -3.922, -4.492]

Query: What river runs through Paris?
Bi-encoder confidence:  0.251
Cross-encoder confidence: 0.381
Rerank scores: [-3.904, -5.276, -6.063, -6.414, -9.944]

Query: What is the recommended dosage of ibuprofen for a dog?
Bi-encoder confidence:  0.132
Cross-encoder confidence: 0.177
Rerank scores: [-10.558, -10.995, -11.004, -11.065, -11.271]

Query: purple elephant quantum sandwich Tuesday
Bi-encoder confidence:  0.188
Cross-encoder confidence: 0.010
Rerank scores: [-11.252, -11.272, -11.313, -11.322, -11.423]

here 
1. The cross-encoder gives much better separation than the bi-encoder did.
2. The two signals mostly agree on ranking, but disagree on magnitude — and that disagreement is informative, not noise
"""

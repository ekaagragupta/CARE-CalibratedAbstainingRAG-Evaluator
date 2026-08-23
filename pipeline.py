"""
The final answer (or hedge/abstain message)
The decision tier (ANSWER / HEDGE / ABSTAIN)
The three individual confidence signals (show/log/debug them)
The combined confidence score
The passage(s) used, for traceability
"""


from retrieval import retrieve
from reranker import rerank
from generation import self_consistency_confidence


DEFAULT_THRESHOLD_HIGH = 0.60
DEFAULT_THRESHOLD_LOW = 0.40

WEIGHTS = (1/3, 1/3, 1/3)  # equal weighting, per Step 6/7's documented baseline choice

# add near the top of step9_pipeline.py
ABSTAIN_PHRASES = ["cannot answer", "i cannot", "unable to answer", "no answer"]

def is_model_abstention(answer_text):
    """
    Checks whether the model's own generated answer was itself an
    abstention ('I cannot answer this...'), as distinct from our
    system-level confidence-based abstention decision. These are two
    different things: the model can refuse even when we'd have chosen
    to trust it, and vice versa.
    """
    return any(phrase in answer_text.lower() for phrase in ABSTAIN_PHRASES)

def make_decision(combined_confidence, threshold_high, threshold_low):
    if combined_confidence >= threshold_high:
        return "ANSWER"
    elif combined_confidence >= threshold_low:
        return "HEDGE"
    else:
        return "ABSTAIN"


def answer_question(query, threshold_high=DEFAULT_THRESHOLD_HIGH, threshold_low=DEFAULT_THRESHOLD_LOW, weights=WEIGHTS):
    """
    The full end-to-end pipeline: retrieve -> rerank -> generate (with
    self-consistency sampling) -> combine confidence signals -> decide
    whether to answer, hedge, or abstain.

    Returns a structured result so the reasoning behind the decision is
    visible, not just a bare answer string.
    """
    retrieval_result = retrieve(query, k=5)
    rerank_result = rerank(query, retrieval_result)
    top_passage = rerank_result["reranked_passages"][0]

    gen_result = self_consistency_confidence(query, top_passage, n_samples=5)

    w1, w2, w3 = weights
    combined_confidence = (
        w1 * retrieval_result["retrieval_confidence"]
        + w2 * rerank_result["rerank_confidence"]
        + w3 * gen_result["consistency_confidence"]
    )

    decision = make_decision(combined_confidence, threshold_high, threshold_low)

    raw_answer = gen_result["majority_answer"]
    model_abstained = is_model_abstention(raw_answer)

    if decision == "ANSWER":
        if model_abstained:
            # High confidence overall, but the model itself couldn't answer —
            # this is itself informative and shouldn't be hidden.
            display_answer = "The system was confident in its retrieval, but the model could not extract an answer from the passage."
        else:
            display_answer = raw_answer
    elif decision == "HEDGE":
        if model_abstained:
            display_answer = "The system has some uncertainty, and the model itself was unable to extract a confident answer from the retrieved passage."
        else:
            display_answer = f"I found a possible answer, but I'm not fully confident: {raw_answer}"
    else:  # ABSTAIN
        display_answer = "I don't have enough confidence in the available information to answer this reliably."

    return {
        "query": query,
        "decision": decision,
        "display_answer": display_answer,
        "raw_answer": raw_answer,
        "combined_confidence": combined_confidence,
        "signals": {
            "retrieval_confidence": retrieval_result["retrieval_confidence"],
            "rerank_confidence": rerank_result["rerank_confidence"],
            "consistency_confidence": gen_result["consistency_confidence"],
        },
        "top_passage": top_passage,
        "raw_answer_samples": gen_result["raw_answers"],  # all 5 samples, for transparency/debugging
    }


if __name__ == "__main__":
    test_queries = [
        "In what country is Normandy located?",
        "What river runs through Paris?",
        "What is the recommended dosage of ibuprofen for a dog?",
        "purple elephant quantum sandwich Tuesday",
    ]

    for q in test_queries:
        result = answer_question(q)
        print(f"\nQuery: {q}")
        print(f"Decision: {result['decision']}")
        print(f"Answer: {result['display_answer']}")
        print(f"Combined confidence: {result['combined_confidence']:.3f}")
        print(f"  Signals -> retrieval: {result['signals']['retrieval_confidence']:.3f}, "
              f"rerank: {result['signals']['rerank_confidence']:.3f}, "
              f"consistency: {result['signals']['consistency_confidence']:.3f}")



"""

Query: In what country is Normandy located?
Decision: ANSWER
Answer: france
Combined confidence: 0.777
  Signals -> retrieval: 0.406, rerank: 0.925, consistency: 1.000

Query: What river runs through Paris?
Decision: HEDGE
Answer: I found a possible answer, but I'm not fully confident: i cannot answer this from given passage
Combined confidence: 0.544
  Signals -> retrieval: 0.251, rerank: 0.381, consistency: 1.000

Query: What is the recommended dosage of ibuprofen for a dog?
Decision: HEDGE
Answer: I found a possible answer, but I'm not fully confident: i cannot answer this from given passage
Combined confidence: 0.436
  Signals -> retrieval: 0.132, rerank: 0.177, consistency: 1.000

Query: purple elephant quantum sandwich Tuesday
Decision: ABSTAIN
Answer: I don't have enough confidence in the available information to answer this reliably.
Combined confidence: 0.400
  Signals -> retrieval: 0.188, rerank: 0.010, consistency: 1.000
apple@ekus-mac calibrated-RAG % """
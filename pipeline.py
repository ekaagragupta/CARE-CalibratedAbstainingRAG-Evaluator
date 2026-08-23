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

    if decision == "ANSWER":
        display_answer = raw_answer
    elif decision == "HEDGE":
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
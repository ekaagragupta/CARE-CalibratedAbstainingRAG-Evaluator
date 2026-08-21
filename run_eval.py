import json
import re
from tqdm import tqdm
from retrieval import retrieve
from reranker import rerank
from generation import self_consistency_confidence,normalize_answer

ABSTAIN_PHRASES=['cannour answer!','i connot','unable to answer','no answer']

def is_abstention(answer_text):
    """
    Checks whether a generated answer is functionally an abstention
    ('I cannot answer this...') rather than a real attempted answer.
    """  
    normalized=answer_text.lower()
    return any(phrase in normalized for phrase in ABSTAIN_PHRASES)

def check_correctness(majority_answer_raw,is_answerable,gold_answers):
    """
    Determines whether the system's answer was correct, handling the
    two distinct cases: answerable questions (match against gold answers)
    and unanswerable questions (correct = the system abstained).
    """
    system_abstained=is_abstention(majority_answer_raw)
    if not is_answerable:
        return system_abstained
    if system_abstained:
        return False
    normalize_system_answer=normalize_answer(majority_answer_raw)
    normalize_gold=[normalize_answer(g) for g in gold_answers]

    # matching if syustem answer is equal to gold ans of ds
    return any(
        normalize_system_answer== g or g in normalize_system_answer
        for g in normalize_gold
    )

def run_single_example(example,weights=(1/3,1/3,1/3)):
    """
    Runs the full pipeline (retrieve -> rerank -> generate) on one
    eval question, computes the combined confidence score, and checks
    correctness against ground truth.
    """
    question=example['question']
    retrieval_result=retrieve(question,k=5)
    rerank_result=rerank(question,retrieval_result)
    top_passage=rerank_result['rerank_passage'][0]

    gen_result=self_consistency_confidence(question,top_passage,n_samples=5)

    w1, w2, w3 = weights
    combined_confidence = (
        w1 * retrieval_result["retrieval_confidence"]
        + w2 * rerank_result["rerank_confidence"]
        + w3 * gen_result["consistency_confidence"]
    )
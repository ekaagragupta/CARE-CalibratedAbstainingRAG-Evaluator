import os
import re
from collections import Counter
from dotenv import load_dotenv
from groq import Groq
from retrieval import retrieve

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])


MODEL_NAME = "openai/gpt-oss-20b"

def generate_answer(query, context_passage, temperature=0.7):
    prompt = f"""Answer the question using ONLY the information in the passage below.
    If the passage does not contain the answer, say "I cannot answer this from the given passage."
    Keep your answer short — a few words or one sentence, no explanation.

    Passage: {context_passage}

    Question: {query}

    Answer:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=300,   # raised from 50 — gives room for reasoning tokens + the actual answer
    )
    return response.choices[0].message.content.strip()

def normalize_answer(text):
    """
    Standard QA-style normalization: lowercase, strip punctuation and
    articles, collapse whitespace. Same normalization style SQuAD's own
    official evaluation script uses.
    """
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def self_consistency_confidence(query, context_passage, n_samples=5):
    """
    Samples n_samples answers from the LLM at temperature > 0, normalizes
    them, and returns the fraction that agree with the majority answer.
    High agreement = high confidence; scattered answers = low confidence.
    """
    raw_answers = [generate_answer(query, context_passage) for _ in range(n_samples)]
    normalized = [normalize_answer(a) for a in raw_answers]

    counts = Counter(normalized)
    majority_answer, majority_count = counts.most_common(1)[0]

    consistency_score = majority_count / n_samples

    return {
        "raw_answers": raw_answers,
        "majority_answer": majority_answer,
        "consistency_confidence": consistency_score
    }


if __name__ == "__main__":
    test_queries = [
        "In what country is Normandy located?",
        "What river runs through Paris?",
        "What is the recommended dosage of ibuprofen for a dog?",
        "purple elephant quantum sandwich Tuesday",
    ]

    for q in test_queries:
        retrieval_result = retrieve(q, k=5)
        top_passage = retrieval_result["passages"][0]

        result = self_consistency_confidence(q, top_passage, n_samples=5)

        print(f"\nQuery: {q}")
        print(f"Top passage used: {top_passage[:150]}...")
        print(f"Raw answers: {result['raw_answers']}")
        print(f"Majority answer: '{result['majority_answer']}'")
        print(f"Self-consistency confidence: {result['consistency_confidence']:.2f}")


        """
        I found that self-consistency alone conflates two different kinds of certainty —
          confidently right and confidently unable to answer. Combining it with retrieval
          -side signals resolves the ambiguity: high consistency plus high retrieval confidence
            means a trustworthy answer, while high consistency plus low retrieval confidence means 
            the system is reliably recognizing when to abstain, which is the actual goal of this project."""
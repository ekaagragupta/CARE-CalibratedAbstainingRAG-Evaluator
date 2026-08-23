from pipeline import answer_question, DEFAULT_THRESHOLD_HIGH, DEFAULT_THRESHOLD_LOW

def print_result(result):
    print(f"\n{'='*60}")
    print(f"Decision: {result['decision']}")
    print(f"{'='*60}")
    print(f"Answer: {result['display_answer']}")
    print(f"\nConfidence breakdown:")
    print(f"  Retrieval (bi-encoder):     {result['signals']['retrieval_confidence']:.3f}")
    print(f"  Rerank (cross-encoder):     {result['signals']['rerank_confidence']:.3f}")
    print(f"  Consistency (generation):   {result['signals']['consistency_confidence']:.3f}")
    print(f"  Combined:                   {result['combined_confidence']:.3f}")
    print(f"\nThresholds: ANSWER >= {DEFAULT_THRESHOLD_HIGH}  |  HEDGE >= {DEFAULT_THRESHOLD_LOW}  |  ABSTAIN below")
    print(f"{'='*60}\n")


def main():
    print("Calibrated, Abstaining RAG — Demo")
    print("Ask a question, or type 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not query:
            continue

        print("\nRetrieving, reranking, and generating (this involves several LLM calls, may take ~10-15s)...")
        result = answer_question(query)
        print_result(result)


if __name__ == "__main__":
    main()
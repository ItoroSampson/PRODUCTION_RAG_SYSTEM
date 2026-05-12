from database import get_vector_store
from sentence_transformers import CrossEncoder

# Loading reranker once globally
reranker_model = CrossEncoder("BAAI/bge-reranker-base")


def retrieve_context(
    query: str, n_results: int = 10
):  # Increased to 10 for better reranking
    collection = get_vector_store()

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # 1. Prepare the results for reranking
    # I'm combinING doc and metadata into a list of dictionaries
    initial_chunks = []
    for i in range(len(results["documents"][0])):
        initial_chunks.append(
            {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            }
        )

    # 2. Run the Reranker
    #  pair the query with each document content
    pairs = [[query, chunk["content"]] for chunk in initial_chunks]
    scores = reranker_model.predict(pairs)

    # 3. Attach scores and sort
    for i, chunk in enumerate(initial_chunks):
        chunk["rerank_score"] = float(scores[i])  # Convert to float for JSON safety

    # Sort by score descending
    reranked_results = sorted(
        initial_chunks, key=lambda x: x["rerank_score"], reverse=True
    )

    # Return top 3 after reranking
    return reranked_results[:3]


if __name__ == "__main__":
    test_query = "How should I handle root user credentials safely?"
    print(f"[*] Searching and Reranking for: '{test_query}'...")

    final_hits = retrieve_context(test_query)

    for i, hit in enumerate(final_hits):
        print(f"\n--- Reranked Result #{i + 1} (Score: {hit['rerank_score']:.4f}) ---")
        print(f"Pillar:  {hit['metadata'].get('pillar') or 'N/A'}")
        print(f"Page:    {hit['metadata'].get('page_number') or 'N/A'}")
        print(f"Content: {hit['content'][:200]}...")

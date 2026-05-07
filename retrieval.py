from database import get_vector_store


def retrieve_context(query: str, n_results: int = 5):
    """

    Searches the 3,160 chunks for the top 'n' most relevant matches.
    """
    collection = get_vector_store()

    # Perform the search
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return results


if __name__ == "__main__":
    # Test Question based on the AWS Framework
    test_query = "How should I handle root user credentials safely?"

    print(f"[*] Searching for: '{test_query}'...")
    hits = retrieve_context(test_query)

    # Updated Display logic in retrieval.py
    for i in range(len(hits["documents"][0])):
        content = hits["documents"][0][i]
        meta = hits["metadatas"][0][i]
        distance = hits["distances"][0][i]

        print(f"\n--- Result #{i + 1} (Score: {distance:.4f}) ---")
        print(f"Pillar:  {meta.get('pillar') or 'N/A'}")
        print(f"BP ID:   {meta.get('best_practice_id') or 'N/A'}")
        print(f"Page:    {meta.get('page_number') or 'N/A'}")
        print(f"URL:     {meta.get('source_url') or 'N/A'}")
        print(f"Content: {content[:250]}...")

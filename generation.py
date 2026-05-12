import os

import requests
from dotenv import load_dotenv
from retrieval import retrieve_context

load_dotenv()


def generate_answer(query: str):
    # 1. Retrieve the reranked context
    # retrieve_context now returns a list of dictionaries: [{"content": ..., "metadata": ...}, ...]
    print("[*] Retrieving and Reranking technical context...")
    reranked_results = retrieve_context(query, n_results=10)

    if not reranked_results:
        return "No relevant context found in the framework.", None

    # Combine only the top 3 reranked chunks into one context block
    context_text = "\n\n".join([hit["content"] for hit in reranked_results])

    # 2. Building the System Prompt
    system_prompt = f"""
    You are an AWS Certified Cloud Architect. 
    Answer the user's question using ONLY the provided context from the AWS Well-Architected Framework.
    If the answer is not in the context, say you don't know. Do not hallucinate or behave like mumu.
    
    Context:
    {context_text}
    """

    # 3. Call Ollama
    print("[*] Generating answer via Ollama...")
    # Using your identified port 11434 (ensure this matches your Docker setup)
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

    payload = {
        "model": "llama3.2",
        "prompt": query,
        "system": system_prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        # We return the answer AND the reranked_results so main.py can show sources
        return response.json()["response"], reranked_results

    except Exception as e:
        return f"Error connecting to Ollama: {e}", None


if __name__ == "__main__":
    user_query = "What is the two-person rule for root user credentials?"
    answer, reranked_data = generate_answer(user_query)

    print("\n=== AWS ARCHITECT RESPONSE ===")
    print(answer)

    if reranked_data:
        print("\nSources (Reranked):")
        for hit in reranked_data:
            meta = hit["metadata"]
            print(
                f"- Page {meta.get('page_number')} | Score: {hit['rerank_score']:.4f}"
            )

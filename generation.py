import os

import requests
from dotenv import load_dotenv
from retrieval import retrieve_context

load_dotenv()


def generate_answer(query: str):
    # 1. Retrieve the context
    print("[*] Retrieving technical context from ChromaDB...")
    search_results = retrieve_context(query, n_results=4)

    # Combine the retrieved chunks into one context block
    context_text = "\n\n".join([doc for doc in search_results["documents"][0]])

    # 2. Building the System Prompt
    system_prompt = f"""
    You are an AWS Certified Cloud Architect. 
    Answer the user's question using ONLY the provided context from the AWS Well-Architected Framework.
    If the answer is not in the context, say you don't know. Do not hallucinate or behave like mumu
    
    Context:
    {context_text}
    """

    # 3. Call Ollama
    print("[*] Generating answer via Ollama...")
    OLLAMA_URL = os.getenv("OLLAMA_URL")
    payload = {
        "model": "llama3.2",
        "prompt": query,
        "system": system_prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"], search_results
    except Exception as e:
        return f"Error connecting to Ollama: {e}", None


if __name__ == "__main__":
    user_query = "What is the two-person rule for root user credentials?"
    answer, metadata = generate_answer(user_query)

    print("\n=== AWS ARCHITECT RESPONSE ===")
    print(answer)
    print("\nSources:")
    # Show page numbers
    pages = set([m["page_number"] for m in metadata["metadatas"][0]])
    print(f"AWS Well-Architected Framework, Pages: {', '.join(map(str, pages))}")

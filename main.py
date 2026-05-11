import hashlib
import os
from typing import List, Optional

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from generation import generate_answer
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = FastAPI(title="AWS Architect RAG API")

# --- CACHE SETUP ---

CACHE_DIR = "api_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def get_query_hash(query: str):
    """Creates a unique filename based on the question."""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


# --- MODELS ---
class QueryRequest(BaseModel):
    question: str


class SourceMetadata(BaseModel):
    page_number: int
    pillar: Optional[str] = None
    source_url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]
    cached: bool = False  # Tells the user if they got a fresh or cached answer


@app.get("/")
def health_check():
    return {"status": "online", "model": ("llama3.2")}


@app.post("/ask", response_model=QueryResponse)
async def ask_architect(request: QueryRequest):
    query_hash = get_query_hash(request.question)
    cache_path = os.path.join(CACHE_DIR, f"{query_hash}.pkl")

    # 1. Check if we have answered this before
    if os.path.exists(cache_path):
        print(f"[*] Cache Hit! Serving instant response for: {request.question}")
        cached_data = joblib.load(cache_path)
        cached_data.cached = True
        return cached_data

    # 2. If not in cache, run the full RAG pipeline
    print(f"[*] Cache Miss. Processing fresh request: {request.question}")
    answer, raw_metadata = generate_answer(request.question)

    if not raw_metadata:
        raise HTTPException(status_code=500, detail="Retrieval failed")

    # Format the sources
    formatted_sources = []
    seen_pages = set()

    for meta in raw_metadata["metadatas"][0]:
        page = meta.get("page_number")
        if page not in seen_pages:
            formatted_sources.append(
                SourceMetadata(
                    page_number=page,
                    pillar=meta.get("pillar"),
                    source_url=meta.get("source_url"),
                )
            )
            seen_pages.add(page)

    # 3. Create the final response object
    response_data = QueryResponse(
        answer=answer, sources=formatted_sources, cached=False
    )

    # 4. Save to cache for next time
    joblib.dump(response_data, cache_path)

    return response_data

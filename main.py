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
    cached: bool = False


@app.post("/ask", response_model=QueryResponse)
async def ask_architect(request: QueryRequest):
    query_hash = get_query_hash(request.question)
    cache_path = os.path.join(CACHE_DIR, f"{query_hash}.pkl")

    # 1. Check Cache
    if os.path.exists(cache_path):
        print("[*] Cache Hit!")
        cached_data = joblib.load(cache_path)
        cached_data.cached = True
        return cached_data

    print("[*] Cache Miss. Processing fresh request...")

    try:
        # 2. Call the updated generation logic (Returns: answer, reranked_results)
        answer, reranked_results = generate_answer(request.question)

        # 3. Format the reranked sources
        formatted_sources = []
        seen_pages = set()

        if reranked_results:
            # reranked_results is now a list of hits like: [{"content":..., "metadata":...}, ...]
            for hit in reranked_results:
                meta = hit.get("metadata", {})
                page = meta.get("page_number")

                if page and page not in seen_pages:
                    formatted_sources.append(
                        SourceMetadata(
                            page_number=page,
                            pillar=meta.get("pillar"),
                            source_url=meta.get("source_url"),
                        )
                    )
                    seen_pages.add(page)
        else:
            print("[!] Warning: No reranked results returned.")

        # 4. Create response
        response_data = QueryResponse(
            answer=answer, sources=formatted_sources, cached=False
        )

        # 5. Save to cache
        joblib.dump(response_data, cache_path)
        return response_data

    except Exception as e:
        print(f"[ERROR] The pipeline crashed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Logic Error: {str(e)}")

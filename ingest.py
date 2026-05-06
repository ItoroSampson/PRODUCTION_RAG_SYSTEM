import asyncio
import re
from typing import List

from database import add_chunks_to_db
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from schema import DocumentChunk


def extract_aws_metadata(text: str):
    """
    Heuristic extraction of AWS Pillar and Best Practice ID from text.
    """
    # Regex to find patterns like REL01-BP02, SEC03-BP01, etc.
    bp_match = re.search(r"([A-Z]{3}\d{2}-BP\d{2})", text)
    bp_id = bp_match.group(1) if bp_match else None

    pillar_map = {
        "OPS": "Operational Excellence",
        "SEC": "Security",
        "REL": "Reliability",
        "PER": "Performance Efficiency",
        "COS": "Cost Optimization",
        "SUS": "Sustainability",
    }

    pillar = None
    if bp_id:
        prefix = bp_id[:3]
        pillar = pillar_map.get(prefix)

    return pillar, bp_id


async def process_document(file_path: str) -> List[DocumentChunk]:
    print(f"[*] Starting ingestion for: {file_path}")

    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    docs = await asyncio.to_thread(loader.load)

    # 2. Split into Chunks
    # Using 800/100 strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    print("[*] Splitting into chunks...")
    raw_chunks = text_splitter.split_documents(docs)

    # 3. Apply the Chunk Contract & Metadata Enrichment
    validated_chunks = []
    print(f"[*] Enriching {len(raw_chunks)} chunks with AWS metadata...")

    for i, chunk in enumerate(raw_chunks):
        pillar, bp_id = extract_aws_metadata(chunk.page_content)

        # Determine source URL if ID exists
        source_url = None
        if bp_id:
            source_url = f"https://docs.aws.amazon.com/wellarchitected/latest/framework/{bp_id}.html"

        # Validating against my Pydantic Schema
        validated_obj = DocumentChunk(
            content=chunk.page_content,
            metadata={
                "source": "AWS_Well_Architected_Framework.pdf",
                "page_number": chunk.metadata.get("page", 0) + 1,
                "pillar": pillar,
                "best_practice_id": bp_id,
                "source_url": source_url,
                "chunk_index": i,
            },
        )
        validated_chunks.append(validated_obj)

    print(f"[✓] Successfully processed and enriched {len(validated_chunks)} chunks.")
    return validated_chunks


if __name__ == "__main__":
    PDF_PATH = "data/AWS_Well_Architected_Framework.pdf"

    try:
        results = asyncio.run(process_document(PDF_PATH))
        # Print a sample to verify enrichment
        sample = next((c for c in results if c.metadata["pillar"]), results[0])
        print("\n--- SAMPLE CHUNK ---")
        print(f"Pillar: {sample.metadata['pillar']}")
        print(f"ID:     {sample.metadata['best_practice_id']}")
        print(f"URL:    {sample.metadata['source_url']}")
        print("------------------------------\n")

        # 3. Push to Vector Store
        print(f"[*] Initializing ChromaDB push for {len(results)} chunks...")
        add_chunks_to_db(results)
    except FileNotFoundError:
        print(f"[!] Error: Could not find the PDF at {PDF_PATH}")

import asyncio

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def process_document(file_path: str):
    print(f"[*] Loading {file_path}...")

    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    docs = await asyncio.to_thread(loader.load)  # Offload heavy I/O to a thread

    # 800 tokens with 100 token overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    # 3. Split into chunks
    print("[*] Splitting into chunks...")
    chunks = text_splitter.split_documents(docs)

    print(f"[✓] Created {len(chunks)} chunks from the AWS Framework.")
    return chunks


if __name__ == "__main__":
    PDF_PATH = "data/AWS_Well_Architected_Framework.pdf"
    asyncio.run(process_document(PDF_PATH))

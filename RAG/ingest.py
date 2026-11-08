from pathlib import Path
import sys

from rag.chunking import load_chunks
from rag.store import get_collection


ROOT = Path(__file__).resolve().parent


def ingest(document_path: Path | None = None) -> int:
    source = document_path or ROOT / "documents"
    source = source.expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Document or folder not found: {source}")

    chunks = load_chunks(source)
    if not chunks:
        raise ValueError("No readable .md, .txt, or .pdf files found.")

    collection = get_collection(ROOT / "chroma_db", reset=True, chunks=chunks)
    print(f"Source: {source}")
    print(f"Ingested {len(chunks)} chunks into Chroma.")
    print(f"Collection count: {collection.count()}")
    return collection.count()


if __name__ == "__main__":
    ingest(Path(sys.argv[1]) if len(sys.argv) > 1 else None)

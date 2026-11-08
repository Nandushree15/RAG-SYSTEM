from pathlib import Path

from rag.chunking import load_chunks
from rag.store import get_collection, retrieve


ROOT = Path(__file__).resolve().parent


def test_document_count() -> None:
    files = list((ROOT / "documents").glob("*.md"))
    assert len(files) >= 10


def test_chunking() -> None:
    chunks = load_chunks(ROOT / "documents")
    assert len(chunks) >= 15
    assert all(50 < len(chunk.text.split()) < 650 for chunk in chunks)


def test_unique_fact_retrieval() -> None:
    collection = get_collection(ROOT / "chroma_db")
    hits = retrieve(
        collection,
        "What checksum phrase opens the Atlas Lantern archive?",
        top_k=3,
    )
    assert any("ember-violet" in hit["text"] for hit in hits)


if __name__ == "__main__":
    test_document_count()
    collection = get_collection(ROOT / "chroma_db", reset=True)
    test_chunking()
    test_unique_fact_retrieval()
    print("All RAG smoke tests passed.")

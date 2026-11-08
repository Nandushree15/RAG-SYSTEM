from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from rag.chunking import load_chunks
from rag.embeddings import make_embedding_function


def get_collection(
    persist_dir: Path,
    reset: bool = False,
    chunks=None,
):
    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            client.delete_collection("project_documents")
        except (ValueError, NotFoundError):
            pass

    collection = client.get_or_create_collection(
        name="project_documents",
        embedding_function=make_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    if reset:
        if chunks is None:
            chunks = load_chunks(
                Path(__file__).resolve().parent.parent / "documents"
            )
        if chunks:
            collection.add(
                ids=[f"{chunk.source}-{chunk.chunk_index}" for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[{"source": chunk.source, "chunk_index": chunk.chunk_index}
                           for chunk in chunks],
            )
    return collection


def retrieve(collection, question: str, top_k: int = 3, source: str | None = None) -> list[dict]:
    where = {"source": source} if source else None
    result = collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append({
            "text": document,
            "source": metadata["source"],
            "distance": distance,
        })
    return hits

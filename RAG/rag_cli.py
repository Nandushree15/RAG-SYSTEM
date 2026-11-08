import argparse
import sys
from pathlib import Path

from rag.chunking import load_chunks
from rag.generation import generate
from rag.store import get_collection, retrieve


ROOT = Path(__file__).resolve().parent


def ingest(document_path: str | None = None) -> int:
    path = Path(document_path).expanduser() if document_path else ROOT / "documents"
    if not path.exists():
        print(f"Document or folder not found: {path}", file=sys.stderr)
        return 1

    chunks = load_chunks(path)
    if not chunks:
        print("No readable .md, .txt, or .pdf documents found.", file=sys.stderr)
        return 1

    collection = get_collection(ROOT / "chroma_db", reset=True, chunks=chunks)
    print(f"Ingested {len(chunks)} chunks into Chroma.")
    print(f"Collection count: {collection.count()}")
    return 0


def retrieve_command(question: str, top_k: int) -> int:
    collection = get_collection(ROOT / "chroma_db")
    hits = retrieve(collection, question, top_k)
    for index, hit in enumerate(hits, start=1):
        print(f"\n[{index}] {hit['source']} (distance={hit['distance']:.4f})")
        print(hit["text"])
    return 0


def ask(question: str, compare: bool) -> int:
    collection = get_collection(ROOT / "chroma_db")
    hits = retrieve(collection, question, 3)
    rag_answer = generate(question, hits)
    print("RAG answer:")
    print(rag_answer)

    if compare:
        print("\nPlain LLM answer (no document context):")
        plain_prompt = f"Answer this question: {question}"
        plain_answer = generate_plain(plain_prompt)
        print(plain_answer)
    return 0


def generate_plain(prompt: str) -> str:
    from rag.generation import _openai_compatible_answer, _ollama_answer

    return (
        _openai_compatible_answer(prompt)
        or _ollama_answer(prompt)
        or "Not enough information: this fact is not from general knowledge."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Chroma RAG demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("document", nargs="?")
    subparsers.choices["ingest"].set_defaults(
        function=lambda args: ingest(args.document)
    )

    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("question")
    retrieve_parser.set_defaults(function=lambda args: retrieve_command(args.question, args.top_k))
    retrieve_parser.add_argument("--top-k", type=int, default=3)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--compare", action="store_true")
    ask_parser.set_defaults(function=lambda args: ask(args.question, args.compare))
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_index: int


def read_markdown(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF support requires pypdf. Run: python -m pip install pypdf"
            ) from exc
        return "\n\n".join(
            page.extract_text() or "" for page in PdfReader(str(path)).pages
        ).strip()
    return path.read_text(encoding="utf-8", errors="replace").strip()


def split_paragraphs(text: str) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def chunk_text(
    text: str,
    source: str,
    target_words: int = 500,
    overlap_words: int = 80,
) -> list[Chunk]:
    paragraphs = split_paragraphs(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if current and current_words + len(words) > target_words:
            chunk = Chunk(
                text="\n\n".join(current),
                source=source,
                chunk_index=len(chunks),
            )
            chunks.append(chunk)
            overlap = " ".join(" ".join(current).split()[-overlap_words:])
            current = [overlap] if overlap else []
            current_words = len(overlap.split())

        current.append(paragraph)
        current_words += len(words)

    if current:
        chunks.append(Chunk(
            text="\n\n".join(current),
            source=source,
            chunk_index=len(chunks),
        ))
    return chunks


def load_chunks(document_path: Path) -> list[Chunk]:
    if document_path.is_file():
        paths = [document_path]
    else:
        paths = [
            path for path in sorted(document_path.iterdir())
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".pdf"}
        ]

    chunks: list[Chunk] = []
    for path in paths:
        for chunk in chunk_text(read_markdown(path), path.name):
            chunks.append(Chunk(chunk.text, chunk.source, len(chunks)))
    return chunks
